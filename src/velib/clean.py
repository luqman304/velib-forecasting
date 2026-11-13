"""Turn irregular raw snapshots into a regular 30-minute panel, one row per station per grid point.

Every rule below is numbered; `clean()` returns a report with how many rows or grid points each rule
touched, which is written to artifacts/cleaning_report.json and shown in the README and the app.

R1  Unparseable rows (timestamp, coordinates or flags) are dropped at ingest.
R2  Duplicate (station, timestamp) rows: keep the last one received.
R3  Station identity = coordinates (the archive has no ID). A name change at the same coordinates is the
    same station; the same name at different coordinates is a different station.
R4  Capacity is kept per snapshot (it changes over time). Capacity 0 means the station is closed or
    being rebuilt, so those snapshots count as offline (R6).
R5  Bikes above capacity are kept: Vélib' stations accept overflow bikes, so it is not a data error.
R6  Snapshots with operative = False (or capacity 0) are masked: the count says nothing about demand.
R7  Frozen stations: an operative station whose counts do not move for 24 h is a stuck sensor, not a
    quiet station (even quiet stations move within a day); those snapshots are masked.
R8  Resampling: each 30-minute grid point takes the latest snapshot at or before it, if it is at most
    30 minutes old (as-of join). Snapshots arrive every ~21 minutes, irregularly.
R9  Gaps of up to 2 grid points (1 hour) are interpolated linearly and flagged `imputed`; longer gaps stay
    missing. Imputed points may be used as model inputs but never as training or evaluation targets.
R10 Station selection: keep stations with at least 80 % of grid points usable, then draw a seeded random
    sample of 60.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def dedupe(raw: pd.DataFrame, report: dict) -> pd.DataFrame:
    before = len(raw)
    raw = raw.sort_values(["station_key", "ts"], kind="stable")
    raw = raw.drop_duplicates(["station_key", "ts"], keep="last")
    report["R2_duplicate_rows_dropped"] = before - len(raw)
    return raw


def mask_offline(raw: pd.DataFrame, report: dict) -> pd.DataFrame:
    raw = raw.copy()
    raw["bikes"] = (raw["bikes_mech"].astype("float32") + raw["bikes_ebike"].astype("float32"))
    cap0 = raw["capacity"].fillna(0) == 0
    off = (~raw["operative"]) | cap0
    report["R4_capacity_zero_rows"] = int(cap0.sum())
    report["R5_bikes_above_capacity_rows_kept"] = int((raw["bikes"] > raw["capacity"]).sum())
    report["R6_offline_rows_masked"] = int(off.sum())
    raw["offline"] = off
    return raw


def mask_frozen(raw: pd.DataFrame, report: dict, hours: int = C.STALE_HOURS) -> pd.DataFrame:
    """Flag runs where (bikes_mech, bikes_ebike) do not change for `hours` while the station is operative."""
    raw = raw.sort_values(["station_key", "ts"]).copy()
    sig = raw["bikes_mech"].astype("int32") * 1000 + raw["bikes_ebike"].astype("int32")
    new_run = (sig != sig.groupby(raw["station_key"]).shift()) | (raw["station_key"] != raw["station_key"].shift())
    run_id = new_run.cumsum()
    by_run = raw.groupby(run_id)["ts"]
    span = by_run.transform("max") - by_run.transform("min")
    frozen = (span >= pd.Timedelta(hours=hours)) & ~raw["offline"]
    report["R7_frozen_rows_masked"] = int(frozen.sum())
    report["R7_frozen_runs"] = int(raw.loc[frozen].groupby(run_id[frozen]).ngroups)
    raw["frozen"] = frozen
    return raw


def to_grid(raw: pd.DataFrame, report: dict) -> pd.DataFrame:
    """R8 + R9: as-of resample every station onto a shared 30-minute UTC grid, then fill short gaps."""
    start = raw["ts"].min().ceil(C.FREQ)
    end = raw["ts"].max().floor(C.FREQ)
    grid = pd.date_range(start, end, freq=C.FREQ, tz="UTC")
    keys = np.sort(raw["station_key"].unique())
    left = pd.DataFrame({
        "ts": grid[np.tile(np.arange(len(grid)), len(keys))],
        "station_key": np.repeat(keys, len(grid)),
    }).sort_values("ts", kind="stable")
    right = raw[["ts", "station_key", "bikes", "capacity", "offline", "frozen", "name"]].sort_values("ts")
    panel = pd.merge_asof(left, right, on="ts", by="station_key", direction="backward",
                          tolerance=pd.Timedelta(C.FREQ))
    panel = panel.sort_values(["station_key", "ts"], kind="stable").reset_index(drop=True)
    n = len(panel)
    no_snap = panel["bikes"].isna()
    masked = panel["offline"].fillna(False).astype(bool) | panel["frozen"].fillna(False).astype(bool)
    panel["observed"] = ~no_snap & ~masked
    panel["y"] = panel["bikes"].where(panel["observed"])
    report["R8_grid_points"] = n
    report["R8_grid_points_without_snapshot"] = int(no_snap.sum())
    report["R8_grid_points_masked_offline_or_frozen"] = int((~no_snap & masked).sum())

    # R9: interpolate only short interior gaps (limit_area='inside' never extrapolates).
    isna = panel["y"].isna()
    run = ((isna != isna.shift()) | (panel["station_key"] != panel["station_key"].shift())).cumsum()
    run_len = isna.groupby(run).transform("size")
    filled = panel.groupby("station_key", sort=False)["y"].transform(lambda s: s.interpolate(limit_area="inside"))
    panel["y_filled"] = filled.where(~isna | (run_len <= C.MAX_INTERP_STEPS)).round()
    panel["imputed"] = panel["y"].isna() & panel["y_filled"].notna()
    report["R9_grid_points_imputed"] = int(panel["imputed"].sum())
    report["R9_grid_points_left_missing"] = int(panel["y_filled"].isna().sum())
    # Capacity: carry the last known value through gaps (it changes rarely).
    g = panel.groupby("station_key", sort=False)
    panel["capacity"] = g["capacity"].ffill().groupby(panel["station_key"]).bfill().astype("float32")
    panel["name"] = g["name"].ffill().groupby(panel["station_key"]).bfill()
    return panel[["station_key", "ts", "name", "capacity", "y", "y_filled", "observed", "imputed"]]


def select_stations(panel: pd.DataFrame, report: dict, n: int = C.N_STATIONS,
                    min_cov: float = C.MIN_COVERAGE, seed: int = C.SEED) -> list[str]:
    cov = panel.groupby("station_key")["y_filled"].apply(lambda s: s.notna().mean())
    # A station that never has a bike (or never has a free dock) is not a forecasting problem.
    spread = panel.groupby("station_key")["y_filled"].std()
    eligible = cov[(cov >= min_cov) & (spread > 1.0)].index
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(np.array(eligible), size=min(n, len(eligible)), replace=False).tolist())
    report["R10_stations_total"] = int(cov.size)
    report["R10_stations_eligible"] = int(len(eligible))
    report["R10_stations_selected"] = len(chosen)
    return chosen


def clean(raw: pd.DataFrame, n_stations: int = C.N_STATIONS) -> tuple[pd.DataFrame, dict]:
    report: dict = {"raw_rows": int(len(raw)), "raw_stations": int(raw["station_key"].nunique())}
    report["R3_station_keys_with_several_names"] = int((raw.groupby("station_key")["name"].nunique() > 1).sum())
    report["R3_names_at_several_locations"] = int((raw.groupby("name")["station_key"].nunique() > 1).sum())
    report["R4_stations_with_capacity_change"] = int(
        (raw[raw["capacity"] > 0].groupby("station_key")["capacity"].nunique() > 1).sum())
    raw = dedupe(raw, report)
    raw = mask_offline(raw, report)
    raw = mask_frozen(raw, report)
    panel = to_grid(raw, report)
    chosen = select_stations(panel, report, n=n_stations)
    panel = panel[panel["station_key"].isin(chosen)].reset_index(drop=True)
    sel = report.setdefault("selected", {})
    sel["grid_points"] = int(len(panel))
    sel["observed_share"] = round(float(panel["observed"].mean()), 4)
    sel["imputed_share"] = round(float(panel["imputed"].mean()), 4)
    sel["missing_share"] = round(float(panel["y_filled"].isna().mean()), 4)
    return panel, report


def clean_collected(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the same rules (R2, R4-R9) to snapshots from scripts/collect.py; no station sampling.

    GBFS rows carry the station's own `last_reported` time as well; a reading that has not been
    refreshed for 24 h is caught by the frozen-station rule like in the archive.
    """
    report: dict = {"raw_rows": int(len(raw)), "raw_stations": int(raw["station_key"].nunique())}
    raw = dedupe(raw, report)
    raw = mask_offline(raw, report)
    raw = mask_frozen(raw, report)
    return to_grid(raw, report), report
