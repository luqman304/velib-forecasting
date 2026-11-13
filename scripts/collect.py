#!/usr/bin/env python3
"""Take one snapshot of the Vélib' Métropole GBFS feed and append it to a daily gzip CSV.

Run on a schedule (GitHub Actions cron or a local loop). One row per station per run:

    status/YYYY-MM-DD.csv.gz  fetched_at,station_code,last_reported,bikes_mech,bikes_ebike,docks,
                              is_installed,is_renting,is_returning
    info/YYYY-MM-DD.csv.gz    fetched_at,station_code,capacity,name,lat,lon   (one block per change)

Station information (capacity, name, coordinates) is stored once per day and again whenever it changes,
so a capacity change stays visible in the history instead of being overwritten. A status snapshot of
~1,500 stations is ~20 kB gzipped; a day of 15-minute snapshots is ~2 MB.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole"
STATUS_FIELDS = [
    "fetched_at", "station_code", "last_reported", "bikes_mech", "bikes_ebike", "docks",
    "is_installed", "is_renting", "is_returning",
]
INFO_FIELDS = ["fetched_at", "station_code", "capacity", "name", "lat", "lon"]


def _get(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _bike_types(types: list[dict]) -> tuple[int, int]:
    # The feed nests counts as [{"mechanical": n}, {"ebike": m}]; tolerate missing or reordered keys.
    merged: dict[str, int] = {}
    for t in types or []:
        merged.update(t)
    return int(merged.get("mechanical", 0)), int(merged.get("ebike", 0))


def snapshot() -> tuple[list[dict], list[dict]]:
    status = _get(f"{BASE}/station_status.json")["data"]["stations"]
    info = _get(f"{BASE}/station_information.json")["data"]["stations"]
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status_rows = []
    for s in status:
        mech, ebike = _bike_types(s.get("num_bikes_available_types", []))
        status_rows.append({
            "fetched_at": fetched_at,
            "station_code": str(s.get("stationCode", "")),
            "last_reported": s.get("last_reported", ""),
            "bikes_mech": mech,
            "bikes_ebike": ebike,
            "docks": s.get("num_docks_available", ""),
            "is_installed": s.get("is_installed", ""),
            "is_renting": s.get("is_renting", ""),
            "is_returning": s.get("is_returning", ""),
        })
    info_rows = [{
        "fetched_at": fetched_at,
        "station_code": str(m.get("stationCode", "")),
        "capacity": m.get("capacity", ""),
        "name": m.get("name", ""),
        "lat": round(float(m.get("lat", 0)), 6),
        "lon": round(float(m.get("lon", 0)), 6),
    } for m in info]
    return status_rows, info_rows


def _append(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    if not path.exists():
        writer.writeheader()
    writer.writerows(rows)
    # Appending gzip members is valid gzip; readers (pandas, zcat) concatenate them transparently.
    with gzip.open(path, "at", encoding="utf-8") as f:
        f.write(buf.getvalue())


def _info_key(rows: list[dict]) -> set[tuple]:
    return {(r["station_code"], str(r["capacity"]), r["name"], str(r["lat"]), str(r["lon"])) for r in rows}


def _latest_info(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    last = max(r["fetched_at"] for r in rows)
    return [r for r in rows if r["fetched_at"] == last]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/collected"))
    args = ap.parse_args()
    status_rows, info_rows = snapshot()
    day = status_rows[0]["fetched_at"][:10]
    _append(args.out / "status" / f"{day}.csv.gz", status_rows, STATUS_FIELDS)
    info_path = args.out / "info" / f"{day}.csv.gz"
    changed = _info_key(info_rows) != _info_key(_latest_info(info_path))
    if changed:
        _append(info_path, info_rows, INFO_FIELDS)
    print(f"{status_rows[0]['fetched_at']}: {len(status_rows)} statuses, info {'written' if changed else 'unchanged'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
