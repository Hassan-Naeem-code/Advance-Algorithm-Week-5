# ruff: noqa: E402
import sys
import pathlib
# Ensure project root is on sys.path so `src` package is importable during tests
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.features import create_features


def test_create_features_basic():
    # small synthetic dataset to validate feature creation
    dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"date": dates, "sales": range(10, 20)})
    out = create_features(df)
    # Expect at least lag_1 and roll_mean_7 to be present
    assert "lag_1" in out.columns
    assert "roll_mean_7" in out.columns
    # No NA rows remain
    assert out.isna().sum().sum() == 0
