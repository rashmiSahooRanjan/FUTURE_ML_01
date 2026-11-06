"""Feature engineering tests.

The fixtures below are small hand-written unit-test inputs that check the arithmetic and
the no-leakage rule. They are not sales data and are never written to the dataset folder.
"""
import numpy as np
import pandas as pd

from src.feature_engineering import (
    add_date_features, add_lag_features, add_rolling_features, build_feature_frame, check_daily_continuity,
)


def _toy_series() -> pd.DataFrame:
    """Two store-family series, 40 consecutive days each, with an increasing target (test fixture)."""
    rows = []
    for store, family in [(1, "A"), (2, "B")]:
        for i, date in enumerate(pd.date_range("2017-01-01", periods=40, freq="D")):
            rows.append({"date": date, "store_nbr": store, "family": family, "sales": float(i + store * 100)})
    return pd.DataFrame(rows)


def test_lag_one_equals_previous_day_value():
    out = add_lag_features(_toy_series(), lags=(1, 7))
    s1 = out[(out["store_nbr"] == 1) & (out["family"] == "A")].sort_values("date")
    assert np.isnan(s1["sales_lag_1"].iloc[0])
    assert s1["sales_lag_1"].iloc[5] == s1["sales"].iloc[4]
    assert s1["sales_lag_7"].iloc[10] == s1["sales"].iloc[3]


def test_lags_do_not_cross_series_boundaries():
    out = add_lag_features(_toy_series(), lags=(1,))
    first_of_second = out[(out["store_nbr"] == 2) & (out["family"] == "B")].sort_values("date").iloc[0]
    assert np.isnan(first_of_second["sales_lag_1"])


def test_rolling_mean_excludes_current_day():
    out = add_rolling_features(_toy_series(), windows=(7,))
    s1 = out[(out["store_nbr"] == 1) & (out["family"] == "A")].sort_values("date").reset_index(drop=True)
    # The mean for day index 10 must be the mean of the seven days before it (indices 3..9).
    expected = s1["sales"].iloc[3:10].mean()
    assert np.isclose(s1["sales_roll_mean_7"].iloc[10], expected)


def test_changing_today_does_not_change_today_feature():
    base = _toy_series()
    modified = base.copy()
    mask = (modified["store_nbr"] == 1) & (modified["family"] == "A") & (modified["date"] == "2017-02-05")
    modified.loc[mask, "sales"] = 999999.0
    a = add_rolling_features(base, windows=(7,))
    b = add_rolling_features(modified, windows=(7,))
    key = (a["store_nbr"] == 1) & (a["family"] == "A") & (a["date"] == "2017-02-05")
    assert np.isclose(a.loc[key, "sales_roll_mean_7"].iloc[0], b.loc[key, "sales_roll_mean_7"].iloc[0])


def test_date_features_are_correct():
    df = pd.DataFrame({"date": pd.to_datetime(["2017-08-16", "2017-12-31"])})
    out = add_date_features(df)
    assert out.loc[0, "day_of_week"] == 2          # Wednesday
    assert out.loc[0, "is_weekend"] == 0
    assert out.loc[1, "is_year_end"] == 1
    assert out.loc[1, "is_month_end"] == 1


def test_continuity_check_detects_gaps():
    df = _toy_series()
    assert check_daily_continuity(df) == 0
    gapped = df.drop(index=5)
    assert check_daily_continuity(gapped) == 1


def test_full_pipeline_runs_on_toy_series():
    out = build_feature_frame(_toy_series())
    assert "sales_lag_30" in out.columns
    assert "sales_roll_std_30" in out.columns
    assert "is_payday_window" in out.columns
