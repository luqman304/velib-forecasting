"""Download raw sources and convert them into one long table.

Unified raw schema (one row per station per snapshot, UTC timestamps):

    ts, station_key, name, lat, lon, capacity, bikes_mech, bikes_ebike, operative, source
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import requests

from . import config as C

ARCHIVE_COLS = ["ts", "capacity", "bikes_mech", "bikes_ebike", "name", "geo", "operative"]


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def archive_to_parquet(zip_path: Path = C.ARCHIVE_ZIP, out: Path = C.ARCHIVE_PARQUET,
                       chunksize: int = 2_000_000) -> Path:
    """Stream the 800 MB archive CSV in chunks into a compact parquet (≈10 M rows, low memory).

    The archive has no header and no station identifier, only a display name and 5-decimal
    coordinates ("48.86598,2.27572"). Coordinates are used as the station key; see clean.py.
    Rows that fail to parse are counted and dropped here (cleaning rule R1).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    bad = 0
    total = 0
    reader = pd.read_csv(zip_path, compression="zip", header=None, names=ARCHIVE_COLS, dtype=str,
                         chunksize=chunksize, on_bad_lines="skip", keep_default_na=False)
    for chunk in reader:
        total += len(chunk)
        df = pd.DataFrame({
            "ts": pd.to_datetime(chunk["ts"], format="%Y-%m-%dT%H:%MZ", utc=True, errors="coerce"),
            "station_key": chunk["geo"].str.replace(" ", "", regex=False),
            "name": chunk["name"],
            "capacity": pd.to_numeric(chunk["capacity"], errors="coerce"),
            "bikes_mech": pd.to_numeric(chunk["bikes_mech"], errors="coerce"),
            "bikes_ebike": pd.to_numeric(chunk["bikes_ebike"], errors="coerce"),
            "operative": chunk["operative"].map({"True": True, "False": False}),
        })
        ok = df["ts"].notna() & df["station_key"].str.contains(",", regex=False) & df["operative"].notna()
        bad += int((~ok).sum())
        df = df[ok]
        latlon = df["station_key"].str.split(",", expand=True)
        df["lat"] = pd.to_numeric(latlon[0], errors="coerce").astype("float32")
        df["lon"] = pd.to_numeric(latlon[1], errors="coerce").astype("float32")
        for col in ("capacity", "bikes_mech", "bikes_ebike"):
            df[col] = df[col].astype("Int16")
        df["operative"] = df["operative"].astype(bool)
        df["source"] = "archive"
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out, table.schema, compression="zstd")
        writer.write_table(table)
    if writer is not None:
        writer.close()
    print(f"archive: {total:,} rows read, {bad:,} unparseable rows dropped -> {out}")
    return out


def load_collected(folder: Path = C.COLLECTED) -> pd.DataFrame:
    """Load snapshots written by scripts/collect.py into the unified raw schema.

    Station info is joined as-of the snapshot time, so a capacity change applies from when it was seen.
    """
    status_files = sorted((folder / "status").glob("*.csv.gz"))
    info_files = sorted((folder / "info").glob("*.csv.gz"))
    if not status_files or not info_files:
        return pd.DataFrame()
    st = pd.concat([pd.read_csv(f, dtype={"station_code": str}) for f in status_files], ignore_index=True)
    info = pd.concat([pd.read_csv(f, dtype={"station_code": str}) for f in info_files], ignore_index=True)
    st["ts"] = pd.to_datetime(st["fetched_at"], utc=True)
    info["ts"] = pd.to_datetime(info["fetched_at"], utc=True)
    st = st.sort_values("ts")
    info = info.sort_values("ts").drop(columns="fetched_at")
    df = pd.merge_asof(st, info, on="ts", by="station_code", direction="backward")
    df["operative"] = (df["is_installed"] == 1) & (df["is_renting"] == 1)
    df = df.rename(columns={"station_code": "station_key"})
    df["source"] = "gbfs"
    return df[["ts", "station_key", "name", "lat", "lon", "capacity", "bikes_mech", "bikes_ebike",
               "operative", "source", "last_reported"]]


def fetch_weather(start: str, end: str, out: Path) -> pd.DataFrame:
    """Hourly observed weather for central Paris from the Open-Meteo archive (no API key)."""
    if out.exists():
        return pd.read_parquet(out)
    params = {
        "latitude": C.PARIS_LATLON[0], "longitude": C.PARIS_LATLON[1],
        "start_date": start, "end_date": end, "timezone": "UTC",
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
    }
    r = requests.get(C.WEATHER_URL, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({
        "ts": pd.to_datetime(h["time"], utc=True),
        "temp_c": h["temperature_2m"],
        "precip_mm": h["precipitation"],
        "wind_kmh": h["wind_speed_10m"],
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def fetch_gbfs_info(out: Path) -> pd.DataFrame:
    """Current station list (code, name, coordinates); used to attach today's station codes."""
    if out.exists():
        return pd.read_parquet(out)
    r = requests.get(C.GBFS_INFO_URL, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"]["stations"])[["stationCode", "name", "lat", "lon", "capacity"]]
    df = df.rename(columns={"stationCode": "station_code"})
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def _gz_lines(path: Path) -> int:
    with gzip.open(path, "rt") as f:
        return sum(1 for _ in f)
