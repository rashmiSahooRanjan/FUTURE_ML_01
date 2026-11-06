"""Cleaning, validation and merging of the Kaggle auxiliary datasets.

Leakage rules followed here:
* Oil prices are forward-filled only (a missing day takes the last known price).
  The one exception is the very first rows, which have no earlier value; they are
  back-filled from the first known price and this is reported in the summary.
* Transactions are never used to fill sales or features for dates they do not cover.
"""
from __future__ import annotations

import pandas as pd

# Holiday rows that represent a real non-working day at national level.
# "Transfer" is the day a holiday was moved to; "Bridge" is a long-weekend day.
# "Holiday" rows flagged transferred=True were moved away, so they are excluded.
NATIONAL_HOLIDAY_TYPES = ("Holiday", "Transfer", "Bridge")


def check_duplicate_keys(df: pd.DataFrame, keys: list[str]) -> int:
    """Return the number of rows that repeat an existing key combination."""
    return int(df.duplicated(subset=keys).sum())


def clean_oil(oil: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sort by date, forward-fill missing prices and report what was done."""
    oil = oil.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    missing_before = int(oil["dcoilwtico"].isna().sum())
    oil["oil_missing_original"] = oil["dcoilwtico"].isna()
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill()
    leading_missing = int(oil["dcoilwtico"].isna().sum())
    oil["dcoilwtico"] = oil["dcoilwtico"].bfill()  # only affects leading rows, see module docstring
    report = {
        "oil_missing_before_fill": missing_before,
        "oil_leading_rows_backfilled": leading_missing,
        "oil_missing_after_fill": int(oil["dcoilwtico"].isna().sum()),
    }
    return oil, report


def build_holiday_flags(holidays: pd.DataFrame) -> pd.DataFrame:
    """Create one row per date with national holiday and event indicators.

    Local and regional holidays are kept in the raw data but not flagged here, because
    they affect only specific cities. They can be added later as city-level features.
    """
    h = holidays.copy()
    national = h["locale"].eq("National")
    is_holiday_row = national & h["type"].isin(NATIONAL_HOLIDAY_TYPES) & ~(h["type"].eq("Holiday") & h["transferred"])
    is_event_row = national & h["type"].eq("Event")

    flags = pd.DataFrame({
        "date": h["date"],
        "is_national_holiday": is_holiday_row,
        "is_event": is_event_row,
    })
    flags = flags.groupby("date", as_index=False).max()
    return flags


def validate_stores(stores: pd.DataFrame) -> None:
    """Stores must have one row per store number, otherwise merges would duplicate sales rows."""
    if stores["store_nbr"].duplicated().any():
        raise ValueError("stores.csv contains duplicate store_nbr values.")


def validate_transactions(transactions: pd.DataFrame) -> dict:
    """Check (date, store) uniqueness and report the coverage window of transactions."""
    dupes = check_duplicate_keys(transactions, ["date", "store_nbr"])
    if dupes:
        raise ValueError(f"transactions.csv has {dupes} duplicate (date, store_nbr) rows.")
    return {
        "transactions_start": transactions["date"].min(),
        "transactions_end": transactions["date"].max(),
        "transactions_stores": int(transactions["store_nbr"].nunique()),
    }


def find_negative_or_zero_sales(df: pd.DataFrame, target: str = "sales") -> dict:
    """Count zero and negative sales. Rows are reported, not removed, because zeros can be real
    (a product not stocked, store closed) and negatives can be returns. See the README."""
    if target not in df.columns:
        return {"zero": 0, "negative": 0}
    return {
        "zero": int((df[target] == 0).sum()),
        "negative": int((df[target] < 0).sum()),
    }


def merge_store_and_auxiliary(
    base: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holiday_flags: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-join store, oil, holiday and (optional) transaction information onto base rows.

    Each join is validated for many-to-one cardinality so accidental row duplication fails loudly.
    """
    validate_stores(stores)
    merged = base.merge(stores, on="store_nbr", how="left", validate="many_to_one")
    merged = merged.merge(oil[["date", "dcoilwtico"]], on="date", how="left", validate="many_to_one")
    merged = merged.merge(holiday_flags, on="date", how="left", validate="many_to_one")
    merged["is_national_holiday"] = merged["is_national_holiday"].fillna(False).astype(bool)
    merged["is_event"] = merged["is_event"].fillna(False).astype(bool)

    if transactions is not None:
        validate_transactions(transactions)
        merged = merged.merge(
            transactions[["date", "store_nbr", "transactions"]],
            on=["date", "store_nbr"],
            how="left",
            validate="many_to_one",
        )
    return merged


def preprocessing_summary(
    original_rows: int,
    final_rows: int,
    duplicate_rows: int,
    missing_handled: dict,
    features_created: list[str],
    status: str,
) -> dict:
    """Return the preprocessing report required by Module 2."""
    return {
        "original_rows": original_rows,
        "final_rows": final_rows,
        "duplicate_rows": duplicate_rows,
        "missing_values_handled": missing_handled,
        "features_created": features_created,
        "processing_status": status,
    }
