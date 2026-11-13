"""Streamlit app: station map, history vs forecast, walk-forward metrics, data quality and live feed.

    streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from velib import clean, config as C, ingest  # noqa: E402

st.set_page_config(page_title="Vélib' availability forecasting", page_icon="🚲", layout="wide")

A = C.ARTIFACTS
METHODS = {
    "lightgbm": ("LightGBM", "#1f6feb"),
    "persistence": ("Persistence (last value)", "#8b949e"),
    "seasonal_naive_day": ("Seasonal naive, same time yesterday", "#d29922"),
    "seasonal_naive_week": ("Seasonal naive, same time last week", "#bc4c00"),
}
ABLATIONS = {"lightgbm_no_weather": "LightGBM without weather features",
             "lightgbm_no_covid": "LightGBM without COVID-restriction features"}
LABELS = {k: v[0] for k, v in METHODS.items()} | ABLATIONS
H_LABEL = {1: "30 min", 2: "1 h", 4: "2 h", 8: "4 h", 16: "8 h", 48: "24 h"}
RULES = {
    "R2_duplicate_rows_dropped": "R2 duplicate (station, timestamp) rows dropped",
    "R3_station_keys_with_several_names": "R3 stations renamed at the same coordinates",
    "R3_names_at_several_locations": "R3 names used at several locations (kept apart)",
    "R4_stations_with_capacity_change": "R4 stations whose capacity changed",
    "R4_capacity_zero_rows": "R4 snapshots with capacity 0 (closed / works)",
    "R5_bikes_above_capacity_rows_kept": "R5 snapshots with bikes > capacity (kept, overflow)",
    "R6_offline_rows_masked": "R6 offline snapshots masked",
    "R7_frozen_rows_masked": "R7 frozen-sensor snapshots masked",
    "R7_frozen_runs": "R7 frozen runs (≥ 24 h without any change)",
    "R8_grid_points": "R8 30-minute grid points (all stations)",
    "R8_grid_points_without_snapshot": "R8 grid points with no snapshot in the last 30 min",
    "R8_grid_points_masked_offline_or_frozen": "R8 grid points masked (offline or frozen)",
    "R9_grid_points_imputed": "R9 grid points interpolated (gap ≤ 1 h)",
    "R9_grid_points_left_missing": "R9 grid points left missing (gap > 1 h)",
    "R10_stations_eligible": "R10 stations with ≥ 80 % usable grid points",
    "R10_stations_selected": "R10 stations sampled for modelling",
}


@st.cache_data
def load():
    preds = pd.read_parquet(C.PROCESSED / "predictions.parquet")
    panel = pd.read_parquet(C.PROCESSED / "panel.parquet")
    stations = pd.read_csv(A / "stations.csv", dtype={"station_code_today": str})
    by_station = pd.read_csv(A / "metrics_by_station.csv")
    stations = stations.merge(by_station, on="station_key", how="left")
    stations["skill_vs_week"] = 1 - stations["lightgbm_mae"] / stations["seasonal_naive_week_mae"]
    stations["skill_vs_persistence"] = 1 - stations["lightgbm_mae"] / stations["persistence_mae"]
    return preds, panel, stations


@st.cache_data(ttl=300)
def load_live():
    raw = ingest.load_collected(C.COLLECTED)
    if raw.empty:
        return None, None
    panel, report = clean.clean_collected(raw)
    return raw, panel


def local(ts: pd.Series | pd.DatetimeIndex):
    return ts.dt.tz_convert(C.TZ) if isinstance(ts, pd.Series) else ts.tz_convert(C.TZ)


missing = [p for p in (C.PROCESSED / "predictions.parquet", A / "metrics.json") if not p.exists()]
if missing:
    st.error("Run the pipeline first: `python -m velib.pipeline` (missing: "
             + ", ".join(str(p.relative_to(C.ROOT)) for p in missing) + ")")
    st.stop()

preds, panel, stations = load()
summary = json.loads((A / "metrics.json").read_text())
report = json.loads((A / "cleaning_report.json").read_text())

st.title("🚲 Vélib' bike availability forecasting")
o = summary["overall"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stations", len(stations))
c2.metric("Walk-forward folds", len(summary["folds"]))
c3.metric("LightGBM MAE (bikes)", f"{o['lightgbm_mae']:.2f}",
          f"{o['lightgbm_mae'] - o['seasonal_naive_week_mae']:+.2f} vs seasonal naive (week)",
          delta_color="inverse")
c4.metric("vs persistence", f"{o['persistence_mae']:.2f}",
          f"{o['lightgbm_mae'] - o['persistence_mae']:+.2f} LightGBM − persistence", delta_color="inverse")

# ---------------------------------------------------------------- station selection (map + picker)
stations = stations.sort_values("name").reset_index(drop=True)
labels = [f"{r.name}  ·  {r.capacity:.0f} docks" for r in stations.itertuples()]
qp = st.query_params  # shareable links: ?station=<name>&origin=2021-03-09T08:00&tab=...
if "station" not in st.session_state:
    by_name = {n: i for i, n in enumerate(stations["name"])}
    st.session_state.station = by_name.get(qp.get("station"), int(stations["lightgbm_mae"].fillna(0).idxmax()))

tab_fc, tab_eval, tab_dq, tab_live = st.tabs(["Forecast", "Evaluation", "Data quality", "Live feed"])

with tab_fc:
    left, right = st.columns([2, 3])
    with left:
        color_by = st.radio("Colour stations by", ["LightGBM skill vs seasonal naive (week)",
                                                   "LightGBM skill vs persistence", "Mean bikes available"],
                            horizontal=False)
        m = stations.copy()
        col = {"Mean": "mean_bikes", "LightGBM skill vs seasonal naive (week)": "skill_vs_week",
               "LightGBM skill vs persistence": "skill_vs_persistence"}
        col = "mean_bikes" if color_by.startswith("Mean") else col[color_by]
        # Rank-based colour (red = lowest, green = highest among these stations) so differences stay visible.
        v = m[col].rank(pct=True).fillna(0.5).to_numpy()
        ramp = np.array([[215, 48, 39], [252, 141, 89], [254, 224, 139], [145, 207, 96], [26, 152, 80]])
        pos = v * (len(ramp) - 1)
        lo = np.floor(pos).astype(int).clip(0, len(ramp) - 2)
        frac = (pos - lo)[:, None]
        rgb = (ramp[lo] * (1 - frac) + ramp[lo + 1] * frac).astype(int)
        m["r"], m["g"], m["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        m["sel"] = m.index == st.session_state.station
        m["radius"] = np.where(m["sel"], 260, 130)
        m["skill_txt"] = (m["skill_vs_week"] * 100).round(0).astype("Int64").astype(str) + " %"
        m["mae_txt"] = m["lightgbm_mae"].round(2).astype(str)
        layer = pdk.Layer("ScatterplotLayer", m, id="stations", get_position=["lon", "lat"],
                          get_fill_color=["r", "g", "b", 220], get_radius="radius", radius_min_pixels=4, pickable=True,
                          stroked=True, get_line_color=[40, 40, 40], line_width_min_pixels=1)
        deck = pdk.Deck(layers=[layer], map_style=pdk.map_styles.CARTO_LIGHT,
                        initial_view_state=pdk.ViewState(latitude=m["lat"].mean(), longitude=m["lon"].mean(),
                                                         zoom=10.6),
                        tooltip={"text": "{name}\nLightGBM MAE {mae_txt}\nskill vs week-naive {skill_txt}"})
        event = st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object", height=430,
                                key="map")
        picked = (event.selection or {}).get("indices", {}).get("stations") if event else None
        # The map keeps its selection across reruns; apply it only when it changes so the picker still works.
        if picked and picked[0] != st.session_state.get("map_pick"):
            st.session_state.station = int(picked[0])
        st.session_state.map_pick = picked[0] if picked else None
        st.caption("Click a station on the map or pick it below. Colour = rank among the 60 stations "
                   "(red lowest, green highest). Skill = 1 − MAE(LightGBM) / MAE(baseline).")
        st.selectbox("Station", range(len(stations)), key="station", format_func=lambda i: labels[i])
    s = stations.iloc[st.session_state.station]
    key = s["station_key"]
    sp = preds[preds["station_key"] == key]
    hist = panel[panel["station_key"] == key].set_index("ts")

    with right:
        st.subheader(s["name"])
        st.caption(f"{s['capacity']:.0f} docks · {s['lat']:.5f}, {s['lon']:.5f}"
                   + (f" · today's GBFS station code {s['station_code_today']}"
                      if isinstance(s.get("station_code_today"), str) and s["station_code_today"] else ""))
        k1, k2, k3 = st.columns(3)
        k1.metric("LightGBM MAE", f"{s['lightgbm_mae']:.2f}")
        k2.metric("Seasonal naive (week) MAE", f"{s['seasonal_naive_week_mae']:.2f}")
        k3.metric("Persistence MAE", f"{s['persistence_mae']:.2f}")

        origins = np.sort(sp["origin"].unique())
        o_local = pd.DatetimeIndex(origins).tz_convert(C.TZ)
        default = len(origins) // 2
        if qp.get("origin"):
            want = pd.Timestamp(qp["origin"]).tz_localize(C.TZ)
            default = int(np.abs((o_local - want).total_seconds()).argmin())
        oi = st.select_slider("Forecast origin (Paris time, test period)", options=range(len(origins)),
                              value=default, format_func=lambda i: o_local[i].strftime("%a %d %b %Y %H:%M"))
        origin = origins[oi]
        fc = sp[sp["origin"] == origin].sort_values("h")
        win = hist.loc[origin - pd.Timedelta(days=2): origin + pd.Timedelta(hours=26)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=local(win.index), y=win["y"], mode="lines", name="Observed",
                                 line=dict(color="#24292f", width=2)))
        imp = win[win["imputed"]]
        if len(imp):
            fig.add_trace(go.Scatter(x=local(imp.index), y=imp["y_filled"], mode="markers", name="Interpolated (R9)",
                                     marker=dict(color="#24292f", symbol="circle-open", size=6)))
        y0 = hist.loc[origin, "y_filled"] if origin in hist.index else np.nan
        for mth, (label, colr) in METHODS.items():
            xs = [pd.Timestamp(origin)] + list(fc["target_ts"])
            ys = [y0] + list(fc[mth])
            fig.add_trace(go.Scatter(x=local(pd.DatetimeIndex(xs)), y=ys, mode="lines+markers", name=label,
                                     line=dict(color=colr, width=3 if mth == "lightgbm" else 1.5,
                                               dash=None if mth == "lightgbm" else "dot")))
        fig.add_vline(x=local(pd.DatetimeIndex([origin]))[0], line_dash="dash", line_color="#cf222e")
        fig.add_hline(y=s["capacity"], line_dash="dot", line_color="#8b949e",
                      annotation_text="capacity", annotation_position="top left")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="bikes available",
                          legend=dict(orientation="h", y=-0.2), title="Forecasts from one origin (30 min → 24 h)")
        st.plotly_chart(fig, use_container_width=True)

        h = st.radio("Rolling view: horizon", list(H_LABEL), index=2, format_func=H_LABEL.get, horizontal=True)
        fold_ids = sorted(sp["fold"].unique())
        fsel = st.select_slider("Test week", options=fold_ids,
                                value=int(sp.loc[sp["origin"] == origin, "fold"].iloc[0]),
                                format_func=lambda k: summary["folds"][k]["test_start"][:10])
        r = sp[(sp["h"] == h) & (sp["fold"] == fsel)].sort_values("target_ts")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=local(r["target_ts"]), y=r["y_target"], mode="lines", name="Observed",
                                  line=dict(color="#24292f", width=2)))
        for mth in ("lightgbm", "seasonal_naive_week", "persistence"):
            label, colr = METHODS[mth]
            fig2.add_trace(go.Scatter(x=local(r["target_ts"]), y=r[mth], mode="lines", name=label,
                                      line=dict(color=colr, width=2 if mth == "lightgbm" else 1)))
        fig2.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="bikes available",
                           legend=dict(orientation="h", y=-0.25),
                           title=f"{H_LABEL[h]}-ahead forecasts over the test week")
        st.plotly_chart(fig2, use_container_width=True)

with tab_eval:
    st.markdown("**Walk-forward validation**: 8 weekly folds with an expanding training window; the model "
                "is retrained for every fold on rows whose *target* is before the fold. All methods are "
                f"scored on the same {summary['scored_rows']:,} (station, origin, horizon) points with a real "
                "observed target.")
    rows = []
    for mth in [*METHODS, "lightgbm_no_weather", "lightgbm_no_covid"]:
        if f"{mth}_mae" in o:
            rows.append({"method": LABELS[mth],
                         "MAE": o[f"{mth}_mae"], "RMSE": o[f"{mth}_rmse"]})
    st.dataframe(pd.DataFrame(rows).style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}"}), hide_index=True)
    ci = pd.DataFrame(summary["lightgbm_vs"]).T
    ci.index = [LABELS[i] for i in ci.index]
    ci["blocks"] = ci["blocks"].astype(int)
    st.markdown("**MAE(LightGBM) − MAE(other)**, 95 % block-bootstrap CI over station-days (negative = LightGBM better)")
    st.dataframe(ci[["diff", "ci_low", "ci_high", "blocks"]].style.format(
        {"diff": "{:+.3f}", "ci_low": "{:+.3f}", "ci_high": "{:+.3f}"}))

    bh = pd.read_csv(A / "metrics_by_horizon.csv")
    cc1, cc2 = st.columns(2)
    for metric, holder in (("mae", cc1), ("rmse", cc2)):
        f = go.Figure()
        for mth, (label, colr) in METHODS.items():
            f.add_trace(go.Scatter(x=[H_LABEL[h] for h in bh["h"]], y=bh[f"{mth}_{metric}"], name=label,
                                   mode="lines+markers", line=dict(color=colr, width=3 if mth == "lightgbm" else 1.5)))
        f.update_layout(title=f"{metric.upper()} by horizon", height=340, yaxis_title="bikes",
                        margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
        holder.plotly_chart(f, use_container_width=True)

    bf = pd.read_csv(A / "metrics_by_fold.csv")
    bf.insert(1, "test week", [summary["folds"][k]["test_start"][:10] for k in bf["fold"]])
    bhr = pd.read_csv(A / "metrics_by_hour.csv")
    cc3, cc4 = st.columns(2)
    f = go.Figure()
    for mth, (label, colr) in METHODS.items():
        f.add_trace(go.Bar(x=bf["test week"], y=bf[f"{mth}_mae"], name=label, marker_color=colr))
    f.update_layout(title="MAE by test week", barmode="group", height=340, margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", y=-0.3))
    cc3.plotly_chart(f, use_container_width=True)
    f = go.Figure()
    for mth, (label, colr) in METHODS.items():
        f.add_trace(go.Scatter(x=bhr["hour"], y=bhr[f"{mth}_mae"], name=label, mode="lines",
                               line=dict(color=colr, width=3 if mth == "lightgbm" else 1.5)))
    f.update_layout(title="MAE by hour of day (target, Paris time)", height=340, xaxis_title="hour",
                    margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    cc4.plotly_chart(f, use_container_width=True)

    st.markdown("**Per station** (sortable)")
    tbl = stations[["name", "capacity", "n", "lightgbm_mae", "seasonal_naive_week_mae", "seasonal_naive_day_mae",
                    "persistence_mae", "skill_vs_week", "skill_vs_persistence"]].sort_values("lightgbm_mae")
    st.dataframe(tbl.style.format({c: "{:.2f}" for c in tbl.columns if c.endswith("_mae")}
                                  | {"skill_vs_week": "{:.0%}", "skill_vs_persistence": "{:.0%}",
                                     "capacity": "{:.0f}"}), hide_index=True, height=320)
    imp = pd.read_csv(A / "feature_importance.csv", index_col=0)["gain"].head(15)[::-1]
    f = go.Figure(go.Bar(x=imp / imp.sum(), y=imp.index, orientation="h", marker_color="#1f6feb"))
    f.update_layout(title="LightGBM feature importance (gain share, top 15, mean over folds)", height=420,
                    margin=dict(l=10, r=10, t=40, b=10), xaxis_tickformat=".0%")
    st.plotly_chart(f, use_container_width=True)

with tab_dq:
    st.markdown("Every cleaning rule and how much data it touched (full archive: "
                f"{report['raw_rows']:,} snapshot rows, {report['raw_stations']:,} stations).")
    dq = pd.DataFrame([{"rule": RULES[k], "count": report[k]} for k in RULES if k in report])
    st.dataframe(dq.style.format({"count": "{:,}"}), hide_index=True, use_container_width=True,
                 height=36 * (len(dq) + 1) + 3)
    spd = pd.read_csv(A / "snapshots_per_day.csv")
    f = go.Figure(go.Bar(x=spd["day"], y=spd["snapshots"], marker_color="#8b949e"))
    f.add_hline(y=96, line_dash="dot", annotation_text="96 = one every 15 min")
    f.update_layout(title="Archive snapshots per day (network-wide)", height=300,
                    margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(f, use_container_width=True)
    s = stations.iloc[st.session_state.station]
    hs = panel[panel["station_key"] == s["station_key"]].copy()
    hs["state"] = np.select([hs["observed"], hs["imputed"]], ["observed", "interpolated"], "missing / masked")
    daily = (hs.assign(day=local(hs["ts"]).dt.strftime("%Y-%m-%d"))
             .groupby(["day", "state"]).size().unstack(fill_value=0))
    f = go.Figure()
    for state, colr in (("observed", "#2da44e"), ("interpolated", "#d29922"), ("missing / masked", "#cf222e")):
        if state in daily:
            f.add_trace(go.Bar(x=daily.index, y=daily[state], name=state, marker_color=colr))
    f.update_layout(barmode="stack", title=f"Grid points per day for {s['name']}", height=300,
                    yaxis_title="of 48", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(f, use_container_width=True)

with tab_live:
    raw_live, live = load_live()
    if raw_live is None:
        st.info("No live snapshots yet. Run `python scripts/collect.py` (or the GitHub Actions workflow) "
                "a few times; files land in data/collected/.")
    else:
        n_snap = raw_live["ts"].nunique()
        st.markdown(f"**{n_snap} GBFS snapshots** collected from {local(raw_live['ts']).min():%d %b %H:%M} "
                    f"to {local(raw_live['ts']).max():%d %b %H:%M} (Paris time) for "
                    f"{raw_live['station_key'].nunique():,} stations, cleaned with the same rules (R2, R4–R9).")
        code = s.get("station_code_today")
        if isinstance(code, str) and code and code in set(live["station_key"]):
            lv = live[live["station_key"] == code]
            rl = raw_live[raw_live["station_key"] == code]
            f = go.Figure()
            f.add_trace(go.Scatter(x=local(rl["ts"]), y=rl["bikes_mech"] + rl["bikes_ebike"], mode="markers",
                                   name="raw snapshots", marker=dict(color="#8b949e")))
            f.add_trace(go.Scatter(x=local(lv["ts"]), y=lv["y_filled"], mode="lines+markers",
                                   name="30-min grid (cleaned)", line=dict(color="#1f6feb")))
            f.update_layout(title=f"Live: {s['name']} (station {code})", height=340, yaxis_title="bikes",
                            margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(f, use_container_width=True)
            last = rl.sort_values("ts").iloc[-1]
            st.caption(f"Latest: {int(last['bikes_mech'])} mechanical + {int(last['bikes_ebike'])} e-bikes, "
                       f"capacity {last['capacity']}, last reported by the station "
                       f"{pd.Timestamp(int(last['last_reported']), unit='s', tz='UTC').tz_convert(C.TZ):%d %b %H:%M}.")
        else:
            st.warning("This archive station has no match in today's station list (moved, renamed or closed).")
        st.caption("Forecasting on live data needs at least one week of history for the seasonal features; "
                   "the workflow in .github/workflows/collect.yml builds that history.")
