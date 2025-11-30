"""Evaluation metrics and plotting utilities."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "MAPE": float(np.nan_to_num(mape, nan=np.nan))}


def plot_predictions(dates: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    plt.figure(figsize=(10, 4))
    plt.plot(dates, y_true, label="Actual", linewidth=1)
    plt.plot(dates, y_pred, label="Predicted", linewidth=1)
    plt.legend()
    plt.title("Predicted vs Actual Sales")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    res = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(res, bins=40)
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
