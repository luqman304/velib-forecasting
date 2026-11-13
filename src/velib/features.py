"""Build a direct multi-horizon training table from the cleaned panel.

One row = (station, forecast origin t, horizon h). The target is the observed bike count at t + h.
Every feature is computable at time t: lags of the (gap-filled) series up to t, the same slot on
previous days/weeks relative to the target, and exogenous variables that are known in advance
(calendar, holidays, COVID restrictions announced before they started, and weather, see note below).

The model predicts the change y(t+h) - y(t); the final forecast is y(t) + change, clipped to [0, cap].

Weather note: hourly *observed* weather at the target time stands in for a weather forecast. Over
≤ 24 h a real forecast would be close, but this is still a mild optimistic bias, stated in the README.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config as C

# Public holidays and Paris (zone C) school holidays inside the archive window.
HOLIDAYS = {"2020-12-25", "2021-01-01", "2021-04-05"}
SCHOOL_HOLIDAYS = [("2020-12-19", "2021-01-03"), ("2021-02-13", "2021-02-28"), ("2021-04-06", "2021-04-25")]

# COVID-19 restrictions in Paris, as announced (dates inclusive, local time).
#   lockdown: 2 = strict national lockdown with outing certificates, 1 = "confinement light"
#             (Île-de-France from 20 March 2021, extended nationally on 3 April), 0 = none
#   curfew_start: local hour from which people had to be home (24 = no curfew)
RESTRICTIONS = [
    ("2020-10-30", "2020-12-14", 2, 24),
    ("2020-12-15", "2021-01-15", 0, 20),
    ("2021-01-16", "2021-03-19", 0, 18),
    ("2021-03-20", "2021-12-31", 1, 19),
]

FEATURES = [
    "h", "station_idx", "lat", "lon", "capacity",
    "lag0", "lag1", "lag2", "lag4", "lag8", "d1", "d4", "fill0",
    "seas_day", "seas_week", "seas_day_delta", "seas_week_delta", "slot_mean_7d", "slot_mean_7d_delta",
    "hour", "dow", "weekend", "holiday", "school_holiday",
    "lockdown", "curfew_start", "hours_to_curfew",
    "temp_c", "precip_mm", "wind_kmh", "precip_3h_origin",
]
CATEGORICAL = ["station_idx"]
GROUPS = {
    "covid": ["lockdown", "curfew_start", "hours_to_curfew"],
    "weather": ["temp_c", "precip_mm", "wind_kmh", "precip_3h_origin"],
}


def calendar(ts_utc: pd.DatetimeIndex) -> pd.DataFrame:
    local = ts_utc.tz_convert(C.TZ)
    day = local.strftime("%Y-%m-%d")
    hour = local.hour + local.minute / 60.0
    out = pd.DataFrame({
        "hour": hour.astype("float32"),
        "dow": local.day_of_week.astype("int8"),
    })
    out["weekend"] = (out["dow"] >= 5).astype("int8")
    out["holiday"] = np.isin(day, list(HOLIDAYS)).astype("int8")
    school = np.zeros(len(local), dtype="int8")
    for a, b in SCHOOL_HOLIDAYS:
        school |= ((day >= a) & (day <= b)).astype("int8")
    out["school_holiday"] = school
    lock = np.zeros(len(local), dtype="int8")
    curfew = np.full(len(local), 24, dtype="int8")
    for a, b, lk, cf in RESTRICTIONS:
        m = (day >= a) & (day <= b)
        lock[m] = lk
        curfew[m] = cf
    out["lockdown"] = lock
    out["curfew_start"] = curfew
    # Hours left before curfew (negative = during curfew, which runs until 06:00).
    htc = curfew - out["hour"].to_numpy()
    htc = np.where(out["hour"].to_numpy() < 6, htc - 24, htc)
    out["hours_to_curfew"] = htc.astype("float32")
    return out


def _nanmean(stack: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slots stay NaN, which is intended
        return np.nanmean(stack, axis=0)


def build(panel: pd.DataFrame, weather: pd.DataFrame, horizons: list[int] = C.HORIZONS) -> pd.DataFrame:
    """Return the long feature table for every station, origin and horizon (targets may be NaN)."""
    panel = panel.sort_values(["station_key", "ts"])
    keys = sorted(panel["station_key"].unique())
    grid = pd.DatetimeIndex(panel["ts"].drop_duplicates().sort_values())
    T, S = len(grid), len(keys)
    yf = panel.pivot(index="ts", columns="station_key", values="y_filled").reindex(grid)[keys].to_numpy(
        dtype="float32")
    yo = panel.pivot(index="ts", columns="station_key", values="y").reindex(grid)[keys].to_numpy(
        dtype="float32")
    cap = panel.pivot(index="ts", columns="station_key", values="capacity").reindex(grid)[keys].to_numpy(
        dtype="float32")
    lat = np.array([float(k.split(",")[0]) for k in keys], dtype="float32")
    lon = np.array([float(k.split(",")[1]) for k in keys], dtype="float32")

    def shift(a: np.ndarray, k: int) -> np.ndarray:
        """Value at row i - k (NaN where out of range). k may be negative (future)."""
        out = np.full_like(a, np.nan)
        if k > 0:
            out[k:] = a[:-k]
        elif k < 0:
            out[:k] = a[-k:]
        else:
            out[:] = a
        return out

    w = weather.set_index("ts").reindex(grid, method="ffill", tolerance=pd.Timedelta("1h"))
    precip3 = w["precip_mm"].rolling(6, min_periods=1).sum().to_numpy(dtype="float32")
    cal = calendar(grid)

    frames = []
    lag = {k: shift(yf, k) for k in (0, 1, 2, 4, 8)}
    slot_origin = _nanmean(np.stack([shift(yf, dd * C.STEPS_PER_DAY) for dd in range(1, 8)]))
    for h in horizons:
        target = shift(yo, -h)
        seas_day = shift(yf, C.STEPS_PER_DAY - h)
        seas_week = shift(yf, C.STEPS_PER_WEEK - h)
        slot = _nanmean(np.stack([shift(yf, dd * C.STEPS_PER_DAY - h) for dd in range(1, 8)]))
        tgt_idx = np.clip(np.arange(T) + h, 0, T - 1)
        tgt_ts = grid[tgt_idx]
        valid_t = (np.arange(T) + h) < T
        cal_t = cal.iloc[tgt_idx].reset_index(drop=True)
        w_t = w.iloc[tgt_idx].reset_index(drop=True)
        d = {
            "station_idx": np.tile(np.arange(S, dtype="int16"), T),
            "origin": np.repeat(grid.tz_convert(None).to_numpy(), S),
            "target_ts": np.repeat(tgt_ts.tz_convert(None).to_numpy(), S),
            "h": np.full(T * S, h, dtype="int8"),
            "lat": np.tile(lat, T), "lon": np.tile(lon, T),
            "capacity": cap.ravel(),
            "y_target": np.where(valid_t[:, None], target, np.nan).ravel(),
        }
        for k, a in lag.items():
            d[f"lag{k}"] = a.ravel()
        d["d1"] = d["lag0"] - d["lag1"]
        d["d4"] = d["lag0"] - d["lag4"]
        d["fill0"] = d["lag0"] / np.maximum(d["capacity"], 1)
        d["seas_day"] = seas_day.ravel()
        d["seas_week"] = seas_week.ravel()
        d["seas_day_delta"] = (seas_day - shift(yf, C.STEPS_PER_DAY)).ravel()
        d["seas_week_delta"] = (seas_week - shift(yf, C.STEPS_PER_WEEK)).ravel()
        d["slot_mean_7d"] = slot.ravel()
        d["slot_mean_7d_delta"] = (slot - slot_origin).ravel()
        for col in cal_t.columns:
            d[col] = np.repeat(cal_t[col].to_numpy(), S)
        for col in ("temp_c", "precip_mm", "wind_kmh"):
            d[col] = np.repeat(w_t[col].to_numpy(dtype="float32"), S)
        d["precip_3h_origin"] = np.repeat(precip3, S)
        frames.append(pd.DataFrame(d))
    df = pd.concat(frames, ignore_index=True)
    df["station_key"] = np.array(keys)[df["station_idx"].to_numpy()]
    df["origin"] = pd.to_datetime(df["origin"], utc=True)
    df["target_ts"] = pd.to_datetime(df["target_ts"], utc=True)
    # Origins without a current value cannot be forecast from (no anchor for the delta target).
    df = df[df["lag0"].notna()].reset_index(drop=True)
    return df
