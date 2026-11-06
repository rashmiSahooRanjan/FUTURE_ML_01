"""Locate, load and summarise the Kaggle Store Sales CSV files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DATA_DIR, DATASET_FILES

# Columns that must exist for each file before the project can use it.
REQUIRED_COLUMNS = {
    "train": ["date", "store_nbr", "family", "sales", "onpromotion"],
    "test": ["id", "date", "store_nbr", "family", "onpromotion"],
    "stores": ["store_nbr", "city", "state", "type", "cluster"],
    "oil": ["date", "dcoilwtico"],
    "holidays": ["date", "type", "locale", "locale_name", "description", "transferred"],
    "transactions": ["date", "store_nbr", "transactions"],
    "sample_submission": ["id", "sales"],
}

TARGET_COLUMN = "sales"


class DatasetError(Exception):
    """Raised when the dataset folder or a file is unusable."""


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Return the dataset folder, preferring an explicit argument over the configured default."""
    return Path(data_dir) if data_dir else DATA_DIR


def find_dataset_files(data_dir: str | Path | None = None) -> dict[str, Path]:
    """Return {dataset_key: path} for every expected file that exists in the folder."""
    folder = resolve_data_dir(data_dir)
    found = {}
    for key, filename in DATASET_FILES.items():
        candidate = folder / filename
        if candidate.is_file():
            found[key] = candidate
    return found


def missing_dataset_files(data_dir: str | Path | None = None) -> list[str]:
    """List the expected dataset files that are not present."""
    present = find_dataset_files(data_dir)
    return [name for key, name in DATASET_FILES.items() if key not in present]


def load_dataset(key: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load one dataset by key with memory-efficient dtypes and validated columns."""
    files = find_dataset_files(data_dir)
    if key not in files:
        raise DatasetError(f"{DATASET_FILES.get(key, key)} was not found in {resolve_data_dir(data_dir)}")

    df = pd.read_csv(files[key])
    if df.empty:
        raise DatasetError(f"{DATASET_FILES[key]} is empty.")

    missing_cols = [c for c in REQUIRED_COLUMNS[key] if c not in df.columns]
    if missing_cols:
        raise DatasetError(f"{DATASET_FILES[key]} is missing columns: {missing_cols}")

    return optimise_dtypes(df, key)


def optimise_dtypes(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Parse dates and downcast numeric columns to reduce memory use."""
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "store_nbr" in df.columns:
        df["store_nbr"] = df["store_nbr"].astype("int16")
    if "family" in df.columns:
        df["family"] = df["family"].astype("category")
    if "onpromotion" in df.columns:
        df["onpromotion"] = pd.to_numeric(df["onpromotion"], errors="coerce").fillna(0).astype("int32")
    if key == "train" and "sales" in df.columns:
        df["sales"] = df["sales"].astype("float32")
    if key == "transactions":
        df["transactions"] = df["transactions"].astype("int32")
    if key == "oil":
        df["dcoilwtico"] = pd.to_numeric(df["dcoilwtico"], errors="coerce").astype("float32")
    if key == "holidays" and "transferred" in df.columns:
        df["transferred"] = df["transferred"].astype(str).str.lower().eq("true")
    return df


def summarise_dataset(df: pd.DataFrame) -> dict:
    """Return the structural summary shown on the Data Explorer page (Module 1)."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    date_range = None
    if "date" in df.columns and df["date"].notna().any():
        date_range = (df["date"].min(), df["date"].max())
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 4),
        "date_range": date_range,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
