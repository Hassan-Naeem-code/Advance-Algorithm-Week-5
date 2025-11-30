"""Model training, baselines, and TimeSeriesSplit tuning utilities.

Provides helpers to build preprocessing pipelines and to train MLP and
tree-based regressors. Also includes several baselines (naive last, seasonal
naive) used for model comparison.
"""
from typing import List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.base import RegressorMixin


def get_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Return a ColumnTransformer for numeric and categorical columns.

    Uses StandardScaler for numeric features and OneHotEncoder for categoricals.
    """
    num_pipe = Pipeline(steps=[("scaler", StandardScaler())])
    # use sparse_output when available; keep compatibility with older sklearn
    cat_pipe = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preproc = ColumnTransformer(transformers=[
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])
    return preproc


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str], estimator: RegressorMixin | None = None) -> Pipeline:
    """Build a scikit-learn Pipeline given feature column lists and an estimator.

    If `estimator` is None, an `MLPRegressor` with reasonable defaults is used.
    """
    preproc = get_preprocessor(numeric_cols, categorical_cols)
    if estimator is None:
        estimator = MLPRegressor(random_state=42, max_iter=1000, early_stopping=True)

    pipe = Pipeline(steps=[("preproc", preproc), ("est", estimator)])
    return pipe


def time_series_grid_search(X: pd.DataFrame, y: pd.Series, numeric_cols: List[str], categorical_cols: List[str],
                            estimator: RegressorMixin | None = None, param_grid: dict | None = None,
                            cv_splits: int = 4, n_jobs: int = -1) -> Tuple[Pipeline, Any]:
    """Run a GridSearchCV with a time-series split and return best pipeline.

    Parameters
    - estimator: optional scikit-learn regressor instance to place in pipeline
    - param_grid: parameter grid matching the estimator step name `est__*`
    - n_jobs: passed to GridSearchCV to control parallelism (-1 uses all cores)
    """
    pipe = build_pipeline(numeric_cols, categorical_cols, estimator=estimator)

    # sensible defaults for MLP if no grid provided
    if param_grid is None:
        param_grid = {
            "est__hidden_layer_sizes": [(32,), (64,), (32, 16)],
            "est__alpha": [1e-4, 1e-3, 1e-2],
            "est__learning_rate_init": [1e-3, 1e-4]
        }

    # Validate that provided param keys are valid for the pipeline to fail fast
    valid_params = set(pipe.get_params().keys())
    invalid = [k for k in param_grid.keys() if k not in valid_params]
    if invalid:
        raise ValueError(
            "Invalid param_grid keys for pipeline: {}. Valid top-level prefixes include 'preproc__' and 'est__'.".format(invalid)
        )

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    gscv = GridSearchCV(pipe, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=n_jobs, verbose=0)
    gscv.fit(X, y)
    return gscv.best_estimator_, gscv


# Baselines
def naive_last(y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    """Predict the last observed value in `y_train` for every test row."""
    last = y_train.iloc[-1]
    return np.full(len(X_test), last)


def seasonal_naive(y_train: pd.Series, X_test: pd.DataFrame, period: int = 7) -> np.ndarray:
    """Seasonal naive forecast: repeat last `period` values to fill the horizon.

    Example: for daily data with weekly seasonality use `period=7`.
    """
    # take the last full period from y_train (chronological order)
    if len(y_train) < period:
        # fallback to last value if not enough history
        return naive_last(y_train, X_test)

    last_period = y_train.iloc[-period:].values
    n = len(X_test)
    preds = np.array([last_period[i % period] for i in range(n)])
    return preds


def naive_lag_k(series: pd.Series, k: int) -> np.ndarray:
    # predict using lag-k value aligned with test indices (assumes series index corresponds to chronological index)
    return series.shift(k).dropna().values
