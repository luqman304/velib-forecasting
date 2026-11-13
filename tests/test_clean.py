import numpy as np
import pandas as pd
import pytest

from velib import clean


def _raw(rows):
    df = pd.DataFrame(rows, columns=["ts", "station_key", "bikes_mech", "bikes_ebike", "capacity", "operative"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["name"] = "S " + df["station_key"]
    for c in ("bikes_mech", "bikes_ebike", "capacity"):
        df[c] = df[c].astype("Int16")
    return df


def _prep(raw):
    rep = {}
    raw = clean.dedupe(raw, rep)
    raw = clean.mask_offline(raw, rep)
    raw = clean.mask_frozen(raw, rep)
    return raw, rep


def test_duplicates_keep_last():
    raw = _raw([
        ("2021-01-01T10:00Z", "a", 1, 0, 10, True),
        ("2021-01-01T10:00Z", "a", 5, 0, 10, True),
    ])
    out, rep = _prep(raw)
    assert rep["R2_duplicate_rows_dropped"] == 1
    assert out["bikes"].tolist() == [5]


def test_offline_and_capacity_zero_are_masked():
    raw = _raw([
        ("2021-01-01T10:00Z", "a", 3, 1, 10, False),
        ("2021-01-01T10:10Z", "a", 3, 1, 0, True),
        ("2021-01-01T10:20Z", "a", 3, 2, 10, True),
    ])
    out, rep = _prep(raw)
    assert out["offline"].tolist() == [True, True, False]
    assert rep["R6_offline_rows_masked"] == 2


def test_frozen_run_is_masked_only_past_threshold():
    ts = pd.date_range("2021-01-01", periods=60, freq="30min", tz="UTC")
    rows = [(t.isoformat(), "a", 4, 0, 10, True) for t in ts]  # 29.5 h without change
    rows += [(t.isoformat(), "b", i % 3, 0, 10, True) for i, t in enumerate(ts)]  # moving station
    out, rep = _prep(_raw(rows))
    assert out.loc[out.station_key == "a", "frozen"].all()
    assert not out.loc[out.station_key == "b", "frozen"].any()


def test_grid_asof_tolerance_and_short_gap_interpolation():
    # Snapshots at 10:05 and 11:35 only; 10:30 takes 10:05 (25 min old), 11:00 has nothing within 30 min,
    # 11:30 has nothing (11:35 is in the future), 12:00 takes 11:35.
    raw = _raw([
        ("2021-01-01T10:05Z", "a", 2, 0, 10, True),
        ("2021-01-01T11:35Z", "a", 8, 0, 10, True),
        ("2021-01-01T12:05Z", "a", 8, 0, 10, True),
    ])
    out, rep = _prep(raw)
    panel = clean.to_grid(out, rep)
    p = panel.set_index(panel["ts"].dt.strftime("%H:%M"))
    assert p.loc["10:30", "y"] == 2
    assert np.isnan(p.loc["11:00", "y"]) and np.isnan(p.loc["11:30", "y"])
    # A 2-point gap is interpolated (2 -> 4 -> 6 -> 8) and flagged as imputed.
    assert p.loc["11:00", "y_filled"] == 4 and p.loc["11:30", "y_filled"] == 6
    assert p.loc["11:00", "imputed"] and not p.loc["10:30", "imputed"]


def test_long_gap_stays_missing():
    raw = _raw([
        ("2021-01-01T10:00Z", "a", 2, 0, 10, True),
        ("2021-01-01T13:00Z", "a", 8, 0, 10, True),
    ])
    out, rep = _prep(raw)
    panel = clean.to_grid(out, rep)
    # 10:30 still sees the 10:00 snapshot (exactly 30 min old); 11:00-12:30 is a 4-point gap.
    gap = panel[(panel.ts > "2021-01-01T10:30Z") & (panel.ts < "2021-01-01T13:00Z")]
    assert len(gap) == 4 and gap["y_filled"].isna().all() and not gap["imputed"].any()


def test_offline_points_are_not_targets():
    raw = _raw([
        ("2021-01-01T10:00Z", "a", 2, 0, 10, True),
        ("2021-01-01T10:30Z", "a", 9, 0, 10, False),
        ("2021-01-01T11:00Z", "a", 4, 0, 10, True),
    ])
    out, rep = _prep(raw)
    panel = clean.to_grid(out, rep).set_index("ts")
    assert np.isnan(panel.loc[pd.Timestamp("2021-01-01T10:30Z"), "y"])
    assert not panel.loc[pd.Timestamp("2021-01-01T10:30Z"), "observed"]
