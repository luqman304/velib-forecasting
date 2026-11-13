"""Paths and constants shared by the pipeline, the evaluation and the app."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
COLLECTED = DATA / "collected"
ARTIFACTS = ROOT / "artifacts"  # small, committed outputs: metrics, cleaning report, predictions sample

ARCHIVE_URL = "https://github.com/lovasoa/historique-velib-opendata/releases/download/new/stations.zip"
ARCHIVE_ZIP = RAW / "lovasoa_stations.zip"
ARCHIVE_PARQUET = INTERIM / "archive_all.parquet"
GBFS_INFO_URL = (
    "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json"
)
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
PARIS_LATLON = (48.8566, 2.3522)

TZ = "Europe/Paris"
FREQ = "30min"
STEPS_PER_DAY = 48
STEPS_PER_WEEK = 7 * STEPS_PER_DAY

# Forecast horizons in 30-minute steps: 30 min, 1 h, 2 h, 4 h, 8 h, 24 h.
HORIZONS = [1, 2, 4, 8, 16, 48]

N_STATIONS = 60
SEED = 7

# Cleaning thresholds (documented in the README, "Cleaning rules").
ASOF_TOLERANCE = "20min"   # a grid point takes the latest snapshot at most this old
MAX_INTERP_STEPS = 2       # gaps of up to 1 h (2 grid points) are interpolated; longer gaps stay missing
STALE_HOURS = 24           # an operative station whose counts do not move for this long is treated as frozen
MIN_COVERAGE = 0.80        # stations need this share of observed grid points to be eligible

# Walk-forward evaluation: weekly folds, expanding training window.
FIRST_TEST_DAY = "2021-02-15"   # Monday
N_FOLDS = 8
