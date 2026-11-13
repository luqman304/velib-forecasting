"""Walk-forward (expanding-origin) evaluation of baselines and LightGBM.

Folds: weekly test windows starting Monday FIRST_TEST_DAY. For fold k the model is retrained on every
row whose *target time* is before the fold start (so no target from the test week leaks into training),
and scored on every origin inside the week. Hyper-parameters are fixed in advance; the number of
boosting rounds is chosen by early stopping on the last 7 days of each training window, never on the
test week.

Scoring set: rows whose target is a real observation (not imputed, not offline/frozen) and where every
baseline is defined, so all methods are compared on exactly the same points.
"""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .features import CATEGORICAL, FEATURES

PARAMS = {
    "objective": "l2",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 4,
    "verbose": -1,
    "seed": C.SEED,
}
MAX_ROUNDS = 2000

BASELINES = {
    "persistence": "lag0",
    "seasonal_naive_day": "seas_day",
    "seasonal_naive_week": "seas_week",
}


def folds(df: pd.DataFrame, first: str = C.FIRST_TEST_DAY, n: int = C.N_FOLDS) -> list[tuple]:
    start = pd.Timestamp(first, tz=C.TZ)
    last = df["origin"].max()
    out = []
    for k in range(n):
        # Calendar weeks in Paris time, so folds still start at Monday 00:00 after the DST switch.
        a = (start + pd.DateOffset(days=7 * k)).tz_convert("UTC")
        b = (start + pd.DateOffset(days=7 * (k + 1))).tz_convert("UTC")
        if a > last:
            break
        out.append((k, a, min(b, last + pd.Timedelta(C.FREQ))))
    return out


def clip_pred(pred: np.ndarray, cap: np.ndarray) -> np.ndarray:
    # Counts can exceed capacity slightly (overflow bikes), so the ceiling has a small margin.
    return np.clip(pred, 0, np.maximum(cap, 1) * 1.1 + 2)


def fit_lgb(train: pd.DataFrame, features: list[str], params: dict = PARAMS) -> lgb.Booster:
    cut = train["target_ts"].max() - pd.Timedelta(days=7)
    tr, va = train[train["target_ts"] <= cut], train[train["target_ts"] > cut]
    cats = [c for c in CATEGORICAL if c in features]
    y_tr = tr["y_target"] - tr["lag0"]
    y_va = va["y_target"] - va["lag0"]
    dtr = lgb.Dataset(tr[features], y_tr, categorical_feature=cats, free_raw_data=True)
    dva = lgb.Dataset(va[features], y_va, categorical_feature=cats, reference=dtr)
    booster = lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    # Refit on the full window with the chosen number of rounds.
    best = booster.best_iteration or MAX_ROUNDS
    dall = lgb.Dataset(train[features], train["y_target"] - train["lag0"], categorical_feature=cats)
    return lgb.train(params, dall, best)


def predict_lgb(model: lgb.Booster, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return clip_pred(df["lag0"].to_numpy() + model.predict(df[features]), df["capacity"].to_numpy())


def run(df: pd.DataFrame, variants: dict[str, list[str]] | None = None,
        log=print) -> tuple[pd.DataFrame, dict]:
    """Return per-row test predictions (all methods) and fold metadata."""
    variants = variants or {"lightgbm": FEATURES}
    labelled = df[df["y_target"].notna()]
    rows, meta = [], []
    importances = []
    for k, a, b in folds(df):
        train = labelled[labelled["target_ts"] < a]
        test = labelled[(labelled["origin"] >= a) & (labelled["origin"] < b)].copy()
        for name, col in BASELINES.items():
            test[name] = test[col]
        for name, feats in variants.items():
            t0 = time.time()
            model = fit_lgb(train, feats)
            test[name] = predict_lgb(model, test, feats)
            log(f"fold {k} {a.tz_convert(C.TZ):%Y-%m-%d}: {name} trained on {len(train):,} rows, "
                f"{model.num_trees()} trees, {time.time() - t0:.0f}s")
            if name == "lightgbm":
                importances.append(pd.Series(model.feature_importance("gain"), index=feats, name=k))
        test["fold"] = k
        meta.append({"fold": k, "test_start": str(a.tz_convert(C.TZ)), "test_end": str(b.tz_convert(C.TZ)),
                     "train_rows": int(len(train)),
                     "test_rows": int(len(test))})
        keep = ["fold", "station_key", "station_idx", "origin", "target_ts", "h", "capacity", "lag0",
                "y_target", *BASELINES, *variants]
        rows.append(test[keep])
    preds = pd.concat(rows, ignore_index=True)
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False) if importances else None
    return preds, {"folds": meta, "importance": imp}


def scoring_mask(preds: pd.DataFrame, methods: list[str]) -> pd.Series:
    return preds[methods].notna().all(axis=1) & preds["y_target"].notna()


def metrics(preds: pd.DataFrame, methods: list[str], by: list[str] | None = None) -> pd.DataFrame:
    p = preds[scoring_mask(preds, methods)]
    out = []
    groups = p.groupby(by) if by else [((), p)]
    for key, g in groups:
        row = dict(zip(by, key if isinstance(key, tuple) else (key,))) if by else {}
        row["n"] = len(g)
        for m in methods:
            err = g[m].to_numpy() - g["y_target"].to_numpy()
            row[f"{m}_mae"] = float(np.abs(err).mean())
            row[f"{m}_rmse"] = float(np.sqrt((err ** 2).mean()))
        out.append(row)
    return pd.DataFrame(out)


def bootstrap_mae_diff(preds: pd.DataFrame, a: str, b: str, methods: list[str], n_boot: int = 2000,
                       seed: int = C.SEED) -> dict:
    """MAE(a) - MAE(b) with a 95 % CI from a block bootstrap over station-days.

    Errors within a station-day are strongly correlated, so resampling single rows would give
    intervals that are far too narrow; whole station-days are resampled instead.
    """
    p = preds[scoring_mask(preds, methods)].copy()
    p["block"] = p["station_key"] + "|" + p["origin"].dt.tz_convert(C.TZ).dt.strftime("%Y-%m-%d")
    p["da"] = (p[a] - p["y_target"]).abs()
    p["db"] = (p[b] - p["y_target"]).abs()
    blk = p.groupby("block").agg(sa=("da", "sum"), sb=("db", "sum"), n=("da", "size"))
    sa, sb, nn = blk["sa"].to_numpy(), blk["sb"].to_numpy(), blk["n"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blk), size=(n_boot, len(blk)))
    diffs = (sa[idx].sum(1) - sb[idx].sum(1)) / nn[idx].sum(1)
    point = (sa.sum() - sb.sum()) / nn.sum()
    return {"diff": float(point), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "blocks": int(len(blk))}


def save_json(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
