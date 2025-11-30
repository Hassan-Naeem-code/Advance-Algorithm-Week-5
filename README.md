# Week 5 — Neural Networks for E-Commerce Sales Forecasting

This repository contains a minimal, runnable skeleton for the assignment: a time-aware tabular regression pipeline using an MLP regressor.

Contents
- `src/` : Python scripts (data loading, features, model, evaluation, main)
- `data/` : (generated synthetic data if no raw CSV is provided)
- `figures/` : saved plots (predicted vs actual, residuals)

Quick start
1. Create and activate a virtual environment and install dependencies (recommended):

```bash
make venv
make install
```

2. Run the pipeline (generates synthetic data if none provided):

```bash
make run
```

Recommended Python
------------------
This project is developed and tested with Python 3.11. Using Python 3.11 ensures prebuilt binary wheels are available for key packages (notably `scikit-learn`) and avoids slow or failing source builds.

Install Python 3.11 (macOS examples):

- Homebrew:

```bash
brew install python@3.11
# then you can run: python3.11 -m venv .venv
```

- pyenv (if you prefer per-project versions):

```bash
pyenv install 3.11.14
pyenv local 3.11.14
python -m venv .venv
```

Create the venv and install dependencies:

```bash
make venv
make install
```

If you run into a scikit-learn build error, confirm your active Python is 3.11 and re-create the venv using that interpreter.

What this repo provides
- A leakage-safe temporal split (70/15/15 chronological)
- Calendar, lag, and rolling-window features (no future leakage)
- A scikit-learn `Pipeline` that fits transforms on train only
- Baselines (naïve t-1 and t-7) and an `MLPRegressor` with TimeSeriesSplit grid search
- Evaluation metrics: MAE, RMSE, R^2, MAPE and saved plots

Notes
- This is a scaffold for the assignment. Replace the synthetic data loader with a downloaded CSV in `data/raw/` (same columns: `date`, `sales`, optionally `price`, `promo`).
- The code is intentionally simple and documented so you can adapt features, hyperparameter grids, and dataset.

References
- scikit-learn docs: `MLPRegressor`, `TimeSeriesSplit`, `Pipeline` (scikit-learn.org)
- Example datasets: Kaggle retail/e-commerce datasets (cite chosen dataset in your final README/PPT)
