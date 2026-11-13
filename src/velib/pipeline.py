"""End-to-end pipeline: download -> clean -> features -> walk-forward evaluation -> artifacts.

    python -m velib.pipeline            # everything (≈ 5 min on a laptop after the 236 MB download)
    python -m velib.pipeline --step clean
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import clean, config as C, evaluate, features, ingest

PANEL = C.PROCESSED / "panel.parquet"
FEATS = C.INTERIM / "features.parquet"
PREDS = C.PROCESSED / "predictions.parquet"
WEATHER = C.RAW / "weather_paris.parquet"
GBFS_INFO = C.RAW / "gbfs_station_information.parquet"


def step_ingest() -> None:
    ingest.download(C.ARCHIVE_URL, C.ARCHIVE_ZIP)
    ingest.archive_to_parquet()


def step_clean() -> None:
    raw = pd.read_parquet(C.ARCHIVE_PARQUET)
    per_day = (raw.drop_duplicates("ts")["ts"].dt.tz_convert(C.TZ).dt.strftime("%Y-%m-%d")
               .value_counts().sort_index().rename_axis("day").rename("snapshots"))
    panel, report = clean.clean(raw)
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL, index=False)
    evaluate.save_json(report, C.ARTIFACTS / "cleaning_report.json")
    per_day.to_csv(C.ARTIFACTS / "snapshots_per_day.csv")

    # Station table for the app and README: attach today's GBFS station code by nearest coordinates.
    st = panel.groupby("station_key").agg(name=("name", "last"), capacity=("capacity", "median"),
                                           observed=("observed", "mean"), imputed=("imputed", "mean"),
                                           mean_bikes=("y", "mean")).reset_index()
    st["lat"] = st["station_key"].str.split(",").str[0].astype(float)
    st["lon"] = st["station_key"].str.split(",").str[1].astype(float)
    try:
        info = ingest.fetch_gbfs_info(GBFS_INFO)
        dlat = (st["lat"].to_numpy()[:, None] - info["lat"].to_numpy()[None, :]) * 111_000
        dlon = (st["lon"].to_numpy()[:, None] - info["lon"].to_numpy()[None, :]) * 73_000
        dist = np.hypot(dlat, dlon)
        j = dist.argmin(axis=1)
        st["station_code_today"] = np.where(dist.min(axis=1) < 60, info["station_code"].to_numpy()[j], "")
        st["dist_today_m"] = dist.min(axis=1).round(1)
    except Exception as exc:  # the live feed is optional for the historical pipeline
        print(f"GBFS station info unavailable ({exc}); skipping station-code match")
    st.to_csv(C.ARTIFACTS / "stations.csv", index=False)
    print(json.dumps(report, indent=1))


def step_features() -> None:
    panel = pd.read_parquet(PANEL)
    start = panel["ts"].min().strftime("%Y-%m-%d")
    end = panel["ts"].max().strftime("%Y-%m-%d")
    weather = ingest.fetch_weather(start, end, WEATHER)
    df = features.build(panel, weather)
    df.to_parquet(FEATS, index=False)
    print(f"features: {len(df):,} rows, {df['y_target'].notna().sum():,} with an observed target")


def step_evaluate(ablations: bool = True) -> None:
    df = pd.read_parquet(FEATS)
    variants = {"lightgbm": features.FEATURES}
    if ablations:
        for group, cols in features.GROUPS.items():
            variants[f"lightgbm_no_{group}"] = [f for f in features.FEATURES if f not in cols]
    t0 = time.time()
    preds, meta = evaluate.run(df, variants)
    print(f"walk-forward done in {time.time() - t0:.0f}s")
    preds.to_parquet(PREDS, index=False)
    meta["importance"].rename("gain").to_csv(C.ARTIFACTS / "feature_importance.csv")
    evaluate.save_json({"folds": meta["folds"], "variants": list(variants), "params": evaluate.PARAMS},
                       C.ARTIFACTS / "folds.json")


def step_report() -> None:
    """Metrics, confidence intervals and per-group tables from the saved walk-forward predictions."""
    preds = evaluate.add_shipped(pd.read_parquet(PREDS))
    preds.to_parquet(PREDS, index=False)
    run_meta = json.loads((C.ARTIFACTS / "folds.json").read_text())
    variants = run_meta["variants"]
    ship = evaluate.SHIPPED
    main = [*evaluate.BASELINES, "lightgbm", ship]
    allm = [*evaluate.BASELINES, *variants, ship]
    A = C.ARTIFACTS
    overall = evaluate.metrics(preds, allm)
    by_h = evaluate.metrics(preds, main, ["h"])
    by_fold = evaluate.metrics(preds, main, ["fold"])
    by_fold_h1 = evaluate.metrics(preds[preds["h"] == 1], main, ["fold"])
    by_station = evaluate.metrics(preds, main, ["station_key"])
    by_hour = preds.assign(hour=preds["target_ts"].dt.tz_convert(C.TZ).dt.hour)
    by_hour = evaluate.metrics(by_hour, main, ["hour"])
    by_h.to_csv(A / "metrics_by_horizon.csv", index=False)
    by_fold.to_csv(A / "metrics_by_fold.csv", index=False)
    by_fold_h1.to_csv(A / "metrics_by_fold_30min.csv", index=False)
    by_station.to_csv(A / "metrics_by_station.csv", index=False)
    by_hour.to_csv(A / "metrics_by_hour.csv", index=False)

    others = [*evaluate.BASELINES, *[v for v in variants if v != "lightgbm"]]
    mask = evaluate.scoring_mask(preds, allm)
    summary = {
        "overall": overall.iloc[0].to_dict(),
        "scored_rows": int(mask.sum()),
        "test_rows_with_observed_target": int(preds["y_target"].notna().sum()),
        "folds": run_meta["folds"],
        "shipped": {"persistence_horizons": sorted(evaluate.PERSISTENCE_HORIZONS), "model": "lightgbm"},
        "shipped_vs": {b: evaluate.bootstrap_mae_diff(preds, ship, b, allm) for b in [*evaluate.BASELINES, "lightgbm"]},
        "lightgbm_vs": {b: evaluate.bootstrap_mae_diff(preds, "lightgbm", b, allm) for b in others},
        "lightgbm_vs_by_horizon": {},
        "params": run_meta["params"],
    }
    for h in sorted(preds["h"].unique()):
        ph = preds[preds["h"] == h]
        summary["lightgbm_vs_by_horizon"][int(h)] = {
            b: evaluate.bootstrap_mae_diff(ph, "lightgbm", b, allm, n_boot=1000) for b in evaluate.BASELINES}
    evaluate.save_json(summary, A / "metrics.json")
    print(overall.T)
    print(by_h)
    print(by_fold_h1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["ingest", "clean", "features", "evaluate", "report", "all"], default="all")
    ap.add_argument("--no-ablations", action="store_true")
    args = ap.parse_args()
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    steps = ["ingest", "clean", "features", "evaluate", "report"] if args.step == "all" else [args.step]
    for s in steps:
        t0 = time.time()
        print(f"== {s}")
        if s == "evaluate":
            step_evaluate(ablations=not args.no_ablations)
        else:
            globals()[f"step_{s}"]()
        print(f"== {s} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
