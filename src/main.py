"""Orchestrator: run data load -> features -> split -> tune -> evaluate.
Run with `python src/main.py` (recommended inside a virtualenv).
"""
from pathlib import Path
import joblib
import pandas as pd
import logging

from src.logging_config import configure_logging
from src.data_load import load_data
from src.features import create_features
from src.model_mlp import time_series_grid_search, naive_last
from src.evaluate import compute_metrics, plot_predictions, plot_residuals


def temporal_train_val_test_split(df: pd.DataFrame, date_col: str = "date", train_frac=0.7, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)
    return train, val, test


def main():
    configure_logging()
    logger = logging.getLogger(__name__)

    out_dir = Path(".")
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data: prefer `data/raw/sales.csv` if present (created after Kaggle download)
    default_csv = Path("data/raw/sales.csv")
    if default_csv.exists():
        df = load_data(csv_path=str(default_csv))
    else:
        df = load_data(csv_path=None)
    logger.info("Loaded %d rows.", len(df))

    # Feature engineering
    df_feat = create_features(df, date_col="date", target_col="sales")
    logger.info("After feature engineering: %d rows and %d columns.", len(df_feat), df_feat.shape[1])

    # Split chronologically
    train_df, val_df, test_df = temporal_train_val_test_split(df_feat)
    logger.info("Splits: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    feature_cols = [c for c in df_feat.columns if c not in ["date", "sales"]]
    # Decide which columns are categorical vs numeric
    categorical_cols = ["day_of_week", "month"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["sales"]
    X_val = val_df[feature_cols]
    y_val = val_df["sales"]
    X_test = test_df[feature_cols]
    y_test = test_df["sales"]

    # Baseline: naive last observed
    y_pred_naive = naive_last(y_train=y_train, X_test=X_test)
    metrics_naive = compute_metrics(y_test.values, y_pred_naive)
    logger.info("Naive baseline metrics: %s", metrics_naive)

    # Combine train + val for final grid search (we'll still use TimeSeriesSplit inside grid search)
    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], ignore_index=True)

    # Hyperparameter tuning
    best_pipe, cv_results = time_series_grid_search(X_train_full, y_train_full, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    logger.info("Best pipeline: %s", best_pipe)

    # Evaluate on test
    y_pred = best_pipe.predict(X_test)
    metrics = compute_metrics(y_test.values, y_pred)
    logger.info("Test metrics: %s", metrics)

    # Save model
    joblib.dump(best_pipe, "model.joblib")

    # Plots
    plot_predictions(test_df["date"], y_test.values, y_pred, out_path=str(figures_dir / "pred_vs_actual.png"))
    plot_residuals(y_test.values, y_pred, out_path=str(figures_dir / "residuals.png"))

    logger.info("Saved plots to %s", figures_dir)


if __name__ == "__main__":
    main()
