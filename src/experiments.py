"""Run experiments required by the assignment:
- Log1p target transform
- Weekly aggregation
- Expanded features
- Expanded hyperparameter tuning

Saves models to `model_<experiment>.joblib` and plots to `figures/` with suffixes.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import dump
import logging

from src.data_load import load_data
from src.features import create_features
from src.model_mlp import build_pipeline
from src.evaluate import compute_metrics, plot_predictions, plot_residuals

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


OUT = Path(".")
FIGS = OUT / "figures"
FIGS.mkdir(exist_ok=True)
from src.logging_config import configure_logging
configure_logging()
logger = logging.getLogger(__name__)


def temporal_split(df, train_frac=0.7, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end].reset_index(drop=True), df.iloc[train_end:val_end].reset_index(drop=True), df.iloc[val_end:].reset_index(drop=True)


def run_grid_search(X_train, y_train, numeric_cols, categorical_cols, param_grid, cv_splits=4):
    pipe = build_pipeline(numeric_cols, categorical_cols)
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    gscv = GridSearchCV(pipe, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
    gscv.fit(X_train, y_train)
    return gscv.best_estimator_, gscv


def experiment(df, name, transform=None, weekly=False, lags=None, roll_windows=None, param_grid=None):
    logger.info("=== Experiment: %s ===", name)
    if weekly:
        # resample weekly by sum
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').resample('W').sum().reset_index()

    # feature engineering
    df_feat = create_features(df, date_col='date', target_col='sales', lags=lags or [1,7], roll_windows=roll_windows or [7,14,28])

    train, val, test = temporal_split(df_feat)
    feature_cols = [c for c in df_feat.columns if c not in ['date', 'sales']]
    categorical_cols = [c for c in feature_cols if c in ['day_of_week', 'month']]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    X_train = pd.concat([train[feature_cols], val[feature_cols]], ignore_index=True)
    y_train = pd.concat([train['sales'], val['sales']], ignore_index=True)
    X_test = test[feature_cols]
    y_test = test['sales']

    # apply transform to y_train if requested
    if transform:
        # guard for non-positive or NaN targets (e.g., returns), clip to zero before log1p
        arr = y_train.values.copy()
        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr = np.where(arr < 0.0, 0.0, arr)
        y_train_trans = transform(arr)
    else:
        y_train_trans = y_train.values

    # default grid if none provided
    if param_grid is None:
        param_grid = {
            'mlp__hidden_layer_sizes': [(32,), (64,), (32,16)],
            'mlp__alpha': [1e-4, 1e-3],
            'mlp__learning_rate_init': [1e-3, 1e-4]
        }

    best_pipe, gscv = run_grid_search(X_train, y_train_trans, numeric_cols, categorical_cols, param_grid)

    # predict on test; if transform used, inverse accordingly
    y_pred = best_pipe.predict(X_test)
    if transform:
        # assume transform is log1p and inverse is expm1
        y_pred_inv = np.expm1(y_pred)
    else:
        y_pred_inv = y_pred

    metrics = compute_metrics(y_test.values, y_pred_inv)
    logger.info('Test metrics: %s', metrics)

    # save model and plots
    dump(best_pipe, f'model_{name}.joblib')
    plot_predictions(test['date'], y_test.values, y_pred_inv, out_path=str(FIGS / f'pred_vs_actual_{name}.png'))
    plot_residuals(y_test.values, y_pred_inv, out_path=str(FIGS / f'residuals_{name}.png'))

    # return metrics and cv results summary
    return {'name': name, 'metrics': metrics, 'best_params': gscv.best_params_}


def main():
    # load prepared CSV if present
    csv_path = Path('data/raw/sales.csv')
    if not csv_path.exists():
        df = load_data(None)
    else:
        df = pd.read_csv(csv_path, parse_dates=['date'])

    results = []

    # A: Log1p transform on daily
    res_a = experiment(df.copy(), name='daily_log', transform=np.log1p)
    results.append(res_a)

    # B: Weekly aggregation (no transform)
    res_b = experiment(df.copy(), name='weekly', weekly=True)
    results.append(res_b)

    # C+D: Expanded features + expanded hyperparameter grid
    res_cd = experiment(df.copy(), name='daily_expanded', lags=[1,7,14,28], roll_windows=[7,14,28], param_grid={
        'mlp__hidden_layer_sizes': [(16,), (32,), (64,), (32,16), (64,32)],
        'mlp__alpha': [1e-4, 1e-3, 1e-2],
        'mlp__learning_rate_init': [1e-3, 1e-4]
    })
    results.append(res_cd)

    # Save results summary
    with open('results_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\nExperiments complete. Results saved to results_summary.json and figures/*.')


if __name__ == '__main__':
    main()
