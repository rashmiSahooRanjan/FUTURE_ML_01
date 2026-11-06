"""Preprocessing tests on the real auxiliary files, plus validation rules."""
import pandas as pd
import pytest

from config import DATA_DIR
from src.data_loader import find_dataset_files, load_dataset
from src.preprocessing import (
    build_holiday_flags, check_duplicate_keys, clean_oil, find_negative_or_zero_sales,
    merge_store_and_auxiliary, validate_stores, validate_transactions,
)

AVAILABLE = find_dataset_files(DATA_DIR)
needs_real = pytest.mark.skipif(not AVAILABLE, reason="No dataset files found")


@needs_real
def test_oil_fill_never_uses_a_later_price():
    raw = load_dataset("oil", DATA_DIR)
    cleaned, report = clean_oil(raw)
    assert cleaned["dcoilwtico"].isna().sum() == 0
    # Every originally blank row must equal the most recent known price before it,
    # except the leading rows, which are back-filled (reported separately).
    known = raw.sort_values("date").dropna(subset=["dcoilwtico"]).set_index("date")["dcoilwtico"]
    blanks = cleaned[cleaned["oil_missing_original"]]
    for _, row in blanks.iterrows():
        earlier = known[known.index < row["date"]]
        if len(earlier):
            assert row["dcoilwtico"] == earlier.iloc[-1]
    assert report["oil_missing_after_fill"] == 0


@needs_real
def test_national_holiday_flags_exclude_transferred_holidays():
    hol = load_dataset("holidays", DATA_DIR)
    flags = build_holiday_flags(hol)
    transferred_dates = hol.loc[(hol["type"] == "Holiday") & hol["transferred"] & (hol["locale"] == "National"), "date"]
    for date in transferred_dates:
        row = flags[flags["date"] == date]
        if not row.empty:
            # Only a day that is independently a Transfer/Bridge/Holiday should be flagged.
            other_valid = hol[(hol["date"] == date) & (hol["locale"] == "National") &
                              hol["type"].isin(["Holiday", "Transfer", "Bridge"]) & ~hol["transferred"]]
            assert row["is_national_holiday"].iloc[0] == (not other_valid.empty)


@needs_real
def test_transactions_have_no_duplicate_store_days():
    if "transactions" not in AVAILABLE:
        pytest.skip("transactions.csv not present")
    tx = load_dataset("transactions", DATA_DIR)
    assert check_duplicate_keys(tx, ["date", "store_nbr"]) == 0
    info = validate_transactions(tx)
    assert info["transactions_stores"] == 54


@needs_real
def test_stores_are_unique():
    if "stores" not in AVAILABLE:
        pytest.skip("stores.csv not present")
    validate_stores(load_dataset("stores", DATA_DIR))


def test_duplicate_keys_detected_on_small_frame():
    df = pd.DataFrame({"date": ["2017-01-01", "2017-01-01"], "store_nbr": [1, 1]})
    assert check_duplicate_keys(df, ["date", "store_nbr"]) == 1


def test_validate_stores_rejects_duplicates():
    with pytest.raises(ValueError):
        validate_stores(pd.DataFrame({"store_nbr": [1, 1]}))


def test_zero_and_negative_sales_are_counted_not_removed():
    df = pd.DataFrame({"sales": [0.0, -1.0, 5.0, 0.0]})
    assert find_negative_or_zero_sales(df) == {"zero": 2, "negative": 1}
    assert len(df) == 4


def test_merge_rejects_duplicate_store_rows():
    base = pd.DataFrame({"date": pd.to_datetime(["2017-01-01"]), "store_nbr": [1]})
    stores = pd.DataFrame({"store_nbr": [1, 1], "city": ["A", "B"], "state": ["X", "X"], "type": ["D", "D"], "cluster": [1, 1]})
    oil = pd.DataFrame({"date": pd.to_datetime(["2017-01-01"]), "dcoilwtico": [50.0]})
    holiday_flags = pd.DataFrame({"date": pd.to_datetime(["2017-01-01"]), "is_national_holiday": [True], "is_event": [False]})
    with pytest.raises(ValueError):
        merge_store_and_auxiliary(base, stores, oil, holiday_flags)
