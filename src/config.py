"""
config.py
---------
Project-wide configuration settings.
"""

from pathlib import Path

# ==========================================================
# Reproducibility
# ==========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==========================================================
# Data Directories
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

EXTERNAL_DATA_DIR = DATA_DIR / "external"

# ==========================================================
# Data Files
# ==========================================================

RAW_DATA_FILE = RAW_DATA_DIR / "s1.csv"

CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "diabetes_clean.csv"

ENGINEERED_DATA_FILE = (
    PROCESSED_DATA_DIR /
    "diabetes_engineered.csv"
)

TARGET_COLUMN = "diabetes"

# ==========================================================
# Models
# ==========================================================

MODELS_DIR = PROJECT_ROOT / "models"

SAVED_MODELS_DIR = MODELS_DIR / "saved_models"

BASELINE_MODELS_DIR = (
    SAVED_MODELS_DIR /
    "baseline"
)

TUNED_MODELS_DIR = (
    SAVED_MODELS_DIR /
    "tuned"
)

# ==========================================================
# Results
# ==========================================================

RESULTS_DIR = PROJECT_ROOT / "results"

BASELINE_RESULTS_DIR = (
    RESULTS_DIR /
    "baseline"
)

TUNING_RESULTS_DIR = (
    RESULTS_DIR /
    "tuning"
)

SMOTE_RESULTS_DIR = (
    RESULTS_DIR /
    "smote"
)

STACKING_RESULTS_DIR = (
    RESULTS_DIR /
    "stacking"
)

EXPLAINABILITY_RESULTS_DIR = (
    RESULTS_DIR /
    "explainability"
)

# ==========================================================
# Create Required Directories
# ==========================================================

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SAVED_MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BASELINE_MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TUNED_MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BASELINE_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TUNING_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SMOTE_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STACKING_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPLAINABILITY_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)