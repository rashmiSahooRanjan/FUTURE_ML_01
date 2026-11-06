"""Time-series feature construction (Module 4).

Every lag and rolling feature for a row is computed from rows with earlier dates only.
Rolling statistics are shifted by one day before the window is applied, so the current
day's sales never enter the feature for that day.
"""
from __future__ import annotations

import pandas as pd

from config import LAG_DAYS, ROLLING_WINDOWS

SERIES_KEYS = ["store_nbr", "family"]


def add_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar features derived from the date column."""
    out = df.copy()
    d = out[date_col]
    out["year"] = d.dt.year.astype("int16")
    out["month"] = d.dt.month.astype("int8")
    out["day"] = d.dt.day.astype("int8")
    out["quarter"] = d.dt.quarter.astype("int8")
    out["week_of_year"] = d.dt.isocalendar().week.astype("int16")
    out["day_of_week"] = d.dt.dayofweek.astype("int8")
    out["is_weekend"] = (d.dt.dayofweek >= 5).astype("int8")
    out["is_month_start"] = d.dt.is_month_start.astype("int8")
    out["is_month_end"] = d.dt.is_month_end.astype("int8")
    out["is_year_start"] = d.dt.is_year_start.astype("int8")
    out["is_year_end"] = d.dt.is_year_end.astype("int8")
    # Payday-related seasonality: sales often rise around the 15th and month end.
    out["is_payday_window"] = ((d.dt.day >= 14) & (d.dt.day <= 16) | (d.dt.is_month_end)).astype("int8")
    return out


def check_daily_continuity(df: pd.DataFrame, date_col: str = "date") -> int:
    """Return how many series have gaps in their daily history.

    Row-based shifts are only correct when each series has one row per calendar day.
    """
    gaps = 0
    for _, group in df.groupby(SERIES_KEYS, observed=True):
        expected = pd.date_range(group[date_col].min(), group[date_col].max(), freq="D")
        if len(expected) != group[date_col].nunique():
            gaps += 1
    return gaps


def add_lag_features(
    df: pd.DataFrame,
    target: str = "sales",
    lags: tuple[int, ...] = LAG_DAYS,
) -> pd.DataFrame:
    """Add lagged target values per store-family series. Data must be sorted by date within each series."""
    out = df.sort_values(SERIES_KEYS + ["date"]).copy()
    grouped = out.groupby(SERIES_KEYS, observed=True)[target]
    for lag in lags:
        out[f"{target}_lag_{lag}"] = grouped.shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target: str = "sales",
    windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Add rolling mean, std, min and max over past values only (shifted by one day)."""
    out = df.sort_values(SERIES_KEYS + ["date"]).reset_index(drop=True)
    # Shift by one day inside each series, then roll over the shifted values within each series.
    out["_shifted"] = out.groupby(SERIES_KEYS, observed=True)[target].shift(1)
    grouped = out.groupby(SERIES_KEYS, observed=True)["_shifted"]

    def _rolled(stat: str, window: int) -> pd.Series:
        roll = grouped.rolling(window, min_periods=max(2, window // 2))
        result = getattr(roll, stat)()
        # Drop the group keys added by the groupby so the result aligns with `out` by row position.
        return result.reset_index(level=list(range(len(SERIES_KEYS))), drop=True).reindex(out.index)

    for window in windows:
        out[f"{target}_roll_mean_{window}"] = _rolled("mean", window)
        if window in (7, 30):
            out[f"{target}_roll_std_{window}"] = _rolled("std", window)
        out[f"{target}_roll_min_{window}"] = _rolled("min", window)
        out[f"{target}_roll_max_{window}"] = _rolled("max", window)
    return out.drop(columns="_shifted")


def add_growth_features(df: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
    """Day-over-day sales growth using only past values (lag 1 vs lag 2)."""
    out = df.copy()
    lag1 = f"{target}_lag_1"
    lag7 = f"{target}_lag_7"
    if lag1 in out.columns and lag7 in out.columns:
        out["sales_growth_vs_last_week"] = (out[lag1] - out[lag7]) / out[lag7].where(out[lag7] != 0)
    return out


def build_feature_frame(df: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
    """Full feature pipeline used by the models and the forecasting page."""
    out = add_date_features(df)
    out = add_lag_features(out, target=target)
    out = add_rolling_features(out, target=target)
    out = add_growth_features(out, target=target)
    return out
