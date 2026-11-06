"""Metric and split tests with hand-checked values."""
import math

import pandas as pd
import pytest

from src.evaluation import (
    all_metrics, chronological_split, mae, mape, rmse, rmsle, r_squared, rolling_origin_folds, wape,
)


def test_mae_and_rmse_on_known_values():
    y, p = [1.0, 2.0, 3.0], [2.0, 2.0, 5.0]
    assert mae(y, p) == pytest.approx(1.0)
    assert rmse(y, p) == pytest.approx(math.sqrt((1 + 0 + 4) / 3))


def test_rmsle_matches_formula():
    y, p = [10.0, 0.0], [12.0, 1.0]
    expected = math.sqrt(((math.log1p(12) - math.log1p(10)) ** 2 + (math.log1p(1) - math.log1p(0)) ** 2) / 2)
    assert rmsle(y, p) == pytest.approx(expected)


def test_rmsle_clips_negative_predictions():
    assert rmsle([0.0], [-5.0]) == pytest.approx(0.0)


def test_mape_skips_zero_actuals():
    y, p = [0.0, 100.0], [5.0, 110.0]
    assert mape(y, p) == pytest.approx(10.0)


def test_mape_is_nan_when_all_actuals_zero():
    assert math.isnan(mape([0.0, 0.0], [1.0, 2.0]))


def test_wape_is_sum_error_over_sum_actual():
    y, p = [10.0, 30.0], [12.0, 27.0]
    assert wape(y, p) == pytest.approx((2 + 3) / 40 * 100)


def test_r_squared_perfect_fit_is_one():
    y = [1.0, 2.0, 3.0, 4.0]
    assert r_squared(y, y) == pytest.approx(1.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        mae([1, 2], [1])


def test_all_metrics_returns_every_metric():
    keys = set(all_metrics([1.0, 2.0], [1.5, 2.5]).keys())
    assert keys == {"MAE", "MSE", "RMSE", "MAPE_%", "RMSLE", "R2", "WAPE_%"}


def test_chronological_split_has_no_overlap_and_keeps_order():
    df = pd.DataFrame({"date": pd.date_range("2017-01-01", periods=100, freq="D")})
    train, valid = chronological_split(df, validation_days=30)
    assert train["date"].max() < valid["date"].min()
    assert len(train) + len(valid) == len(df)


def test_rolling_origin_folds_never_train_on_future_dates():
    df = pd.DataFrame({"date": pd.date_range("2017-01-01", periods=120, freq="D")})
    folds = list(rolling_origin_folds(df, horizon=10, n_folds=3))
    assert len(folds) == 3
    for train, valid in folds:
        assert train["date"].max() < valid["date"].min()
