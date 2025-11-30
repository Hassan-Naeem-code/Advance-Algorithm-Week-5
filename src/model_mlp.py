"""Model training, baseline, and TimeSeriesSplit tuning for an MLPRegressor."""
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import Ridge


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    num_pipe = Pipeline(steps=[("scaler", StandardScaler())])
    cat_pipe = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])

    preproc = ColumnTransformer(transformers=[
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])

    mlp = MLPRegressor(random_state=42, max_iter=500, early_stopping=True)
    pipe = Pipeline(steps=[("preproc", preproc), ("mlp", mlp)])
    return pipe


def time_series_grid_search(X: pd.DataFrame, y: pd.Series, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[Pipeline, dict]:
    pipe = build_pipeline(numeric_cols, categorical_cols)

    param_grid = {
        "mlp__hidden_layer_sizes": [(32,), (64,), (32, 16)],
        "mlp__alpha": [1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [1e-3, 1e-4]
    }

    tscv = TimeSeriesSplit(n_splits=4)
    gscv = GridSearchCV(pipe, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=0)
    gscv.fit(X, y)
    return gscv.best_estimator_, gscv.cv_results_


# Baselines
def naive_last(y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    # predict last observed value of train for all test points
    last = y_train.iloc[-1]
    return np.full(len(X_test), last)


def naive_lag_k(series: pd.Series, k: int) -> np.ndarray:
    # predict using lag-k value aligned with test indices (assumes series index corresponds to chronological index)
    return series.shift(k).dropna().values
