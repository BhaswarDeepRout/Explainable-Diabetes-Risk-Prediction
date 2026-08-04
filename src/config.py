"""Project-wide paths and reproducibility settings."""

from pathlib import Path

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
