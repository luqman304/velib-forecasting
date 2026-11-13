import gzip
import importlib.util
from pathlib import Path

from velib import ingest

spec = importlib.util.spec_from_file_location("collect", Path(__file__).parents[1] / "scripts" / "collect.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


def test_bike_types_tolerates_order_and_missing_keys():
    assert collect._bike_types([{"ebike": 2}, {"mechanical": 5}]) == (5, 2)
    assert collect._bike_types([{"mechanical": 3}]) == (3, 0)
    assert collect._bike_types([]) == (0, 0)


def test_append_then_load_roundtrip(tmp_path):
    status = [{"fetched_at": "2026-11-13T10:00:00Z", "station_code": "16107", "last_reported": 1,
               "bikes_mech": 4, "bikes_ebike": 1, "docks": 20, "is_installed": 1, "is_renting": 1,
               "is_returning": 1}]
    info = [{"fetched_at": "2026-11-13T10:00:00Z", "station_code": "16107", "capacity": 35,
             "name": "Benjamin Godard", "lat": 48.86, "lon": 2.27}]
    collect._append(tmp_path / "status" / "2026-11-13.csv.gz", status, collect.STATUS_FIELDS)
    collect._append(tmp_path / "status" / "2026-11-13.csv.gz",
                    [{**status[0], "fetched_at": "2026-11-13T10:15:00Z", "is_renting": 0}], collect.STATUS_FIELDS)
    collect._append(tmp_path / "info" / "2026-11-13.csv.gz", info, collect.INFO_FIELDS)
    with gzip.open(tmp_path / "status" / "2026-11-13.csv.gz", "rt") as f:
        assert sum(1 for _ in f) == 3  # one header, two rows across two gzip members
    df = ingest.load_collected(tmp_path)
    assert len(df) == 2
    assert df["capacity"].tolist() == [35, 35]
    assert df["operative"].tolist() == [True, False]
