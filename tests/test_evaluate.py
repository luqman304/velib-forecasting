import pandas as pd

from velib import evaluate


def test_shipped_uses_last_value_at_30_min_and_lightgbm_after():
    preds = pd.DataFrame({
        "h": [1, 2, 48],
        "persistence": [10.0, 10.0, 10.0],
        "lightgbm": [12.0, 13.0, 14.0],
    })
    out = evaluate.add_shipped(preds)
    assert out["shipped"].tolist() == [10.0, 13.0, 14.0]
    assert "shipped" not in preds  # input left untouched
