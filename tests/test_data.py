"""Data loading tests. These run against the real CSV files in data/raw (or SMARTSALES_DATA_DIR)."""
import pytest

from config import DATA_DIR, DATASET_FILES
from src.data_loader import (
    DatasetError, find_dataset_files, load_dataset, missing_dataset_files, summarise_dataset,
)

AVAILABLE = find_dataset_files(DATA_DIR)
pytestmark = pytest.mark.skipif(not AVAILABLE, reason="No dataset files found in the dataset folder")


def test_expected_file_names_are_defined():
    assert set(DATASET_FILES) == {"train", "test", "stores", "oil", "holidays", "transactions", "sample_submission"}


def test_missing_list_matches_absent_files():
    missing = missing_dataset_files(DATA_DIR)
    for key, name in DATASET_FILES.items():
        assert (name in missing) == (key not in AVAILABLE)


@pytest.mark.parametrize("key", ["stores", "oil", "holidays", "transactions", "test", "sample_submission"])
def test_available_files_load_with_required_columns(key):
    if key not in AVAILABLE:
        pytest.skip(f"{DATASET_FILES[key]} not present")
    df = load_dataset(key, DATA_DIR)
    assert len(df) > 0
    summary = summarise_dataset(df)
    assert summary["rows"] == len(df)
    assert summary["memory_mb"] > 0


def test_stores_has_54_unique_stores():
    if "stores" not in AVAILABLE:
        pytest.skip("stores.csv not present")
    df = load_dataset("stores", DATA_DIR)
    assert df["store_nbr"].nunique() == 54


def test_test_file_covers_sixteen_days_after_training_window():
    if "test" not in AVAILABLE:
        pytest.skip("test.csv not present")
    df = load_dataset("test", DATA_DIR)
    assert df["date"].min().strftime("%Y-%m-%d") == "2017-08-16"
    assert df["date"].max().strftime("%Y-%m-%d") == "2017-08-31"
    assert df["date"].nunique() == 16


def test_loading_missing_dataset_raises_clear_error(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset("train", tmp_path)
