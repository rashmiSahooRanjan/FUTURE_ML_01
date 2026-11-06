"""Central configuration for SmartSales AI.

Paths are resolved relative to the project root or taken from environment
variables, so the project runs unchanged on any machine or cloud host.
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent

# Override with the SMARTSALES_DATA_DIR environment variable to point at any folder
# that holds the Kaggle CSV files (for example a Windows path on a local machine).
DATA_DIR = Path(os.environ.get("SMARTSALES_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
EVALUATION_DIR = PROJECT_ROOT / "models" / "evaluation"
REPORTS_DIR = PROJECT_ROOT / "reports"

APP_NAME = "SmartSales AI"
TAGLINE = "Predict Sales. Understand Demand. Make Smarter Business Decisions."

# Canonical file names of the Kaggle "Store Sales - Time Series Forecasting" dataset.
DATASET_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays": "holidays_events.csv",
    "transactions": "transactions.csv",
    "sample_submission": "sample_submission.csv",
}

# Forecast horizons offered in the app (days).
FORECAST_HORIZONS = (7, 14, 30)

# Lag and rolling windows used for feature engineering (days).
LAG_DAYS = (1, 7, 14, 21, 28, 30)
ROLLING_WINDOWS = (7, 14, 30)

# Demand classification thresholds: average daily sales per store-family series.
# These are configurable defaults; they are documented in the app.
DEMAND_THRESHOLDS = {"high": 500.0, "medium": 50.0}

RANDOM_STATE = 42
