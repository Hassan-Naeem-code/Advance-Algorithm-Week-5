"""Consolidated experiments runner (single clean copy).

This implementation centralizes the experiment flow and ensures
imports and definitions appear only once to satisfy linters.
"""

from pathlib import Path
import json
import os
import logging
import numpy as np
import pandas as pd
from joblib import dump

from src.data_load import load_data
from src.features import create_features
from src.model_mlp import time_series_grid_search, build_pipeline
from src.evaluate import compute_metrics, plot_predictions, plot_residuals

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from src.logging_config import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

OUT = Path('.')
FIGS = OUT / 'figures'
FIGS.mkdir(exist_ok=True)


def temporal_split(df, train_frac=0.7, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )


def experiment_prep(df: pd.DataFrame, lags=None, roll_windows=None, weekly=False, transform=None):
    if weekly:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').resample('W').sum().reset_index()

    df_feat = create_features(
        df, date_col='date', target_col='sales', lags=lags or [1, 7], roll_windows=roll_windows or [7, 14, 28]
    )

    train, val, test = temporal_split(df_feat)
    feature_cols = [c for c in df_feat.columns if c not in ['date', 'sales']]
    categorical_cols = [c for c in feature_cols if c in ['day_of_week', 'month']]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    X_train = pd.concat([train[feature_cols], val[feature_cols]], ignore_index=True)
    y_train = pd.concat([train['sales'], val['sales']], ignore_index=True)
    X_test = test[feature_cols]
    y_test = test['sales']

    if transform:
        arr = y_train.values.copy()
        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr = np.where(arr < 0.0, 0.0, arr)
        y_train_trans = transform(arr)
    else:
        y_train_trans = y_train.values

    return {
        'X_train': X_train,
        'y_train': y_train_trans,
        'X_test': X_test,
        'y_test': y_test,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
    }


def main():
    csv_path = Path('data/raw/sales.csv')
    if not csv_path.exists():
        df = load_data(None)
    else:
        df = pd.read_csv(csv_path, parse_dates=['date'])

    experiments = [
        {'name': 'daily_log', 'transform': np.log1p, 'weekly': False, 'lags': [1, 7]},
        {'name': 'weekly', 'transform': None, 'weekly': True, 'lags': [1, 7]},
        {'name': 'daily_expanded', 'transform': None, 'weekly': False, 'lags': [1, 7, 14, 28], 'roll_windows': [7, 14, 28]},
    ]

    estimators = {
        'mlp': MLPRegressor(random_state=42, max_iter=1000, early_stopping=True),
        'rf': RandomForestRegressor(random_state=42, n_jobs=-1),
        'hgb': HistGradientBoostingRegressor(random_state=42),
    }

    grids = {
        'mlp': {
            'est__hidden_layer_sizes': [(32,), (64,), (32, 16)],
            'est__alpha': [1e-4, 1e-3],
            'est__learning_rate_init': [1e-3, 1e-4],
        },
        'rf': {'est__n_estimators': [100, 200], 'est__max_depth': [None, 10, 20]},
        'hgb': {'est__learning_rate': [0.1, 0.05], 'est__max_iter': [100, 200]},
    }

    all_results = []

    for exp in experiments:
        prep = experiment_prep(df.copy(), lags=exp.get('lags'), roll_windows=exp.get('roll_windows'), weekly=exp.get('weekly'), transform=exp.get('transform'))

        X_train = prep['X_train']
        y_train = prep['y_train']
        X_test = prep['X_test']
        y_test = prep['y_test']
        numeric_cols = prep['numeric_cols']
        categorical_cols = prep['categorical_cols']

        for est_name, est in estimators.items():
            logger.info('Running %s with %s', exp['name'], est_name)
            param_grid = grids.get(est_name)

            if param_grid is not None:
                pipe = build_pipeline(numeric_cols, categorical_cols, estimator=est)
                valid = set(pipe.get_params().keys())
                invalid_keys = [k for k in param_grid.keys() if k not in valid]
                if invalid_keys:
                    raise ValueError(f'Invalid param grid keys for pipeline: {invalid_keys}')

            n_jobs = int(os.environ.get('GSCV_N_JOBS', '-1'))

            try:
                result = time_series_grid_search(X_train, y_train, numeric_cols=numeric_cols, categorical_cols=categorical_cols, estimator=est, param_grid=param_grid, cv_splits=4, n_jobs=n_jobs)
            except Exception as e:
                logger.exception('Grid search failed for %s %s: %s', exp['name'], est_name, e)
                continue

            if isinstance(result, tuple) and len(result) == 2:
                best_pipe, cv_obj = result
            else:
                cv_obj = result
                best_pipe = getattr(result, 'best_estimator_', None) or getattr(result, 'best_estimator', None)

            if best_pipe is None:
                logger.error('Could not obtain best pipeline for %s %s', exp['name'], est_name)
                continue

            y_pred = best_pipe.predict(X_test)
            if exp.get('transform'):
                y_pred_inv = np.expm1(y_pred)
            else:
                y_pred_inv = y_pred

            metrics = compute_metrics(y_test.values, y_pred_inv)
            logger.info('Result %s-%s: %s', exp['name'], est_name, metrics)

            model_path = f"model_{exp['name']}_{est_name}.joblib"
            dump(best_pipe, model_path)

            try:
                plot_predictions(
                    X_test['date'] if 'date' in X_test.columns else X_test.index,
                    y_test.values,
                    y_pred_inv,
                    out_path=str(FIGS / f"pred_vs_actual_{exp['name']}_{est_name}.png"),
                )
            except Exception:
                plot_predictions(
                    X_test.index,
                    y_test.values,
                    y_pred_inv,
                    out_path=str(FIGS / f"pred_vs_actual_{exp['name']}_{est_name}.png"),
                )
            plot_residuals(
                y_test.values,
                y_pred_inv,
                out_path=str(FIGS / f"residuals_{exp['name']}_{est_name}.png"),
            )

            best_params = getattr(cv_obj, 'best_params_', None)
            if best_params is None and isinstance(cv_obj, dict):
                best_params = cv_obj.get('best_params_') or cv_obj.get('best_params', {})

            all_results.append({'experiment': exp['name'], 'estimator': est_name, 'metrics': metrics, 'best_params': best_params})

    with open('results_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info('Experiments complete. Results saved to results_summary.json and figures/*.')


if __name__ == '__main__':
    main()
