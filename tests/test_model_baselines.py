# ruff: noqa: E402
import sys
import pathlib
# ensure repo root on path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.model_mlp import naive_last, seasonal_naive


def test_naive_last():
    series = pd.Series([1, 2, 3, 4, 5])
    X_test = pd.DataFrame({'a': [0, 0, 0]})
    preds = naive_last(series, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == 3
    assert all(preds == 5)


def test_seasonal_naive():
    # weekly seasonality example
    series = pd.Series(list(range(1, 15)))  # two weeks of daily data
    X_test = pd.DataFrame({'a': [0, 0, 0, 0]})
    preds = seasonal_naive(series, X_test, period=7)
    # last week values are 8..14, repeated
    expected = np.array([8, 9, 10, 11])
    assert np.all(preds == expected)
