"""Feature engineering: calendar, lags, and rolling statistics.
All lag/rolling features use `.shift()` to avoid leakage.
"""
from typing import List
import pandas as pd


def create_features(df: pd.DataFrame, date_col: str = "date", target_col: str = "sales",
                    lags: List[int] = [1, 7], roll_windows: List[int] = [7, 14, 28]) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Calendar features
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Lag features
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # Rolling features (computed on past values only)
    for w in roll_windows:
        df[f"roll_mean_{w}"] = df[target_col].shift(1).rolling(window=w, min_periods=1).mean()
        df[f"roll_std_{w}"] = df[target_col].shift(1).rolling(window=w, min_periods=1).std().fillna(0.0)

    # If promo or price exist, add simple recent stats
    if "promo" in df.columns:
        df["promo_last_7"] = df["promo"].shift(1).rolling(window=7, min_periods=1).sum()

    if "price" in df.columns:
        df["price_change_1"] = df["price"] - df["price"].shift(1)

    # Target as-is (do not shift target here; target remains for supervised learning)
    # Drop rows with NA after lags/rolling
    df = df.dropna().reset_index(drop=True)
    return df
