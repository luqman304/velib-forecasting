Everything in `data/` is generated and git-ignored:

- `raw/` archive zip (236 MB, downloaded), Open-Meteo weather, current GBFS station list
- `interim/` archive as parquet (40 MB), feature table
- `processed/` cleaned 30-minute panel and walk-forward predictions (used by the app)
- `collected/` snapshots written by `scripts/collect.py` when run locally

Run `python -m velib.pipeline` to rebuild it.
