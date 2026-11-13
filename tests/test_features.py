import numpy as np
import pandas as pd

from velib import config as C
from velib import features


def _panel(n_days=9):
    ts = pd.date_range("2021-01-04", periods=n_days * C.STEPS_PER_DAY, freq=C.FREQ, tz="UTC")
    rows = []
    for key in ("48.80000,2.30000", "48.90000,2.40000"):
        y = (np.arange(len(ts)) % 17).astype("float32")
        rows.append(pd.DataFrame({"station_key": key, "ts": ts, "name": "x", "capacity": 20.0, "y": y,
                                  "y_filled": y, "observed": True, "imputed": False}))
    return pd.concat(rows, ignore_index=True)


def _weather(panel):
    ts = pd.date_range(panel.ts.min(), panel.ts.max(), freq="1h", tz="UTC")
    return pd.DataFrame({"ts": ts, "temp_c": 5.0, "precip_mm": 0.0, "wind_kmh": 10.0})


def test_target_is_future_and_lags_are_past():
    panel = _panel()
    df = features.build(panel, _weather(panel), horizons=[1, 4])
    series = panel[panel.station_key == "48.80000,2.30000"].set_index("ts")["y"]
    sample = df[(df.station_key == "48.80000,2.30000")].dropna(subset=["y_target", "seas_week"]).sample(
        50, random_state=0)
    for r in sample.itertuples():
        assert r.target_ts == r.origin + pd.Timedelta(minutes=30 * r.h)
        assert r.y_target == series[r.target_ts]
        assert r.lag0 == series[r.origin]
        assert r.lag1 == series[r.origin - pd.Timedelta(minutes=30)]
        # Seasonal features point at the target slot one day / one week earlier, never later.
        assert r.seas_day == series[r.target_ts - pd.Timedelta(days=1)]
        assert r.seas_week == series[r.target_ts - pd.Timedelta(days=7)]


def test_calendar_curfew_and_dst():
    ts = pd.DatetimeIndex(pd.to_datetime(["2021-02-01T16:30Z", "2021-03-29T16:30Z", "2020-12-01T12:00Z"],
                                         utc=True))
    cal = features.calendar(ts)
    # 17:30 Paris in winter, 18:30 Paris after the 28 March DST switch.
    assert cal.loc[0, "hour"] == 17.5 and cal.loc[1, "hour"] == 18.5
    assert cal.loc[0, "curfew_start"] == 18 and cal.loc[0, "hours_to_curfew"] == 0.5
    assert cal.loc[1, "curfew_start"] == 19 and cal.loc[1, "lockdown"] == 1
    assert cal.loc[2, "lockdown"] == 2
