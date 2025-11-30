"""data_load.py
Utilities to load data. If no CSV path is provided, a synthetic daily sales timeseries is generated.
Columns: `date` (YYYY-MM-DD), `sales` (float), optional `price`, `promo` (0/1)
"""

from pathlib import Path
import numpy as np
import pandas as pd


def load_data(csv_path: str | None = None, seed: int = 42) -> pd.DataFrame:
    """Load a CSV with a date column named `date` and numeric `sales` column.

    If `csv_path` is None or doesn't exist, generate a synthetic daily dataset.
    Returns a DataFrame sorted by date ascending.
    """
    if csv_path:
        p = Path(csv_path)
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df

    # Generate synthetic data
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2019-01-01", periods=400, freq="D")

    # Trend + yearly seasonality + weekly seasonality + noise
    day_of_year = dates.dayofyear.values
    weekly = (np.sin(2 * np.pi * (dates.dayofweek) / 7) * 10)
    yearly = (np.sin(2 * np.pi * day_of_year / 365) * 20)
    trend = np.linspace(0, 30, len(dates))
    base = 200 + trend + yearly + weekly

    # Promo flags: occasional promotions that increase sales
    promo = (rng.random(len(dates)) < 0.05).astype(int)
    sales = base + promo * 40 + rng.normal(0, 8, len(dates))
    price = 50 + rng.normal(0, 1, len(dates))  # minor price noise

    df = pd.DataFrame({"date": dates, "sales": sales, "price": price, "promo": promo})
    df = df.sort_values("date").reset_index(drop=True)
    return df
