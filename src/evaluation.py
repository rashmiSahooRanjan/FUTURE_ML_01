"""Forecast accuracy metrics (Module 9) and chronological splitting.

Sales can be zero, so percentage metrics are handled explicitly:
* MAPE skips rows where the actual value is zero and reports how many were skipped.
* WAPE (sum of absolute errors divided by sum of actuals) is the stable alternative.
* RMSLE is the competition metric; predictions are clipped at zero before the log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[mask], y_pred[mask]


def mae(y_true, y_pred) -> float:
    t, p = _clean_pair(y_true, y_pred)
    return float(np.mean(np.abs(t - p))) if t.size else float("nan")


def mse(y_true, y_pred) -> float:
    t, p = _clean_pair(y_true, y_pred)
    return float(np.mean((t - p) ** 2)) if t.size else float("nan")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def rmsle(y_true, y_pred) -> float:
    """Root mean squared logarithmic error, as used by the Kaggle competition."""
    t, p = _clean_pair(y_true, y_pred)
    if t.size == 0:
        return float("nan")
    t = np.clip(t, 0, None)
    p = np.clip(p, 0, None)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(t)) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error in percent, excluding zero actuals."""
    t, p = _clean_pair(y_true, y_pred)
    nonzero = t != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((t[nonzero] - p[nonzero]) / t[nonzero])) * 100)


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error in percent: sum|error| / sum|actual|."""
    t, p = _clean_pair(y_true, y_pred)
    denominator = np.sum(np.abs(t))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(t - p)) / denominator * 100)


def r_squared(y_true, y_pred) -> float:
    t, p = _clean_pair(y_true, y_pred)
    if t.size < 2:
        return float("nan")
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def all_metrics(y_true, y_pred) -> dict:
    """Compute every metric from Module 9 for one set of predictions."""
    return {
        "MAE": mae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
        "RMSLE": rmsle(y_true, y_pred),
        "R2": r_squared(y_true, y_pred),
        "WAPE_%": wape(y_true, y_pred),
    }


def chronological_split(df: pd.DataFrame, date_col: str = "date", validation_days: int = 30):
    """Split by time: the last `validation_days` calendar days form the validation set.

    Random splitting is never used for this project because it leaks future information.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    cutoff = df[date_col].max() - pd.Timedelta(days=validation_days)
    train = df[df[date_col] <= cutoff]
    valid = df[df[date_col] > cutoff]
    return train, valid


def rolling_origin_folds(df: pd.DataFrame, date_col: str = "date", horizon: int = 16, n_folds: int = 3):
    """Yield (train, valid) pairs with expanding training windows that end at different dates."""
    end = df[date_col].max()
    for fold in range(n_folds, 0, -1):
        valid_end = end - pd.Timedelta(days=horizon * (fold - 1))
        valid_start = valid_end - pd.Timedelta(days=horizon - 1)
        train = df[df[date_col] < valid_start]
        valid = df[(df[date_col] >= valid_start) & (df[date_col] <= valid_end)]
        if not train.empty and not valid.empty:
            yield train, valid
