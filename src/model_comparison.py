"""
Model comparison
----------------
Combines baseline, tuned, SMOTE, and TabNet model results.
"""

import pandas as pd

from config import RESULTS_DIR


# ==========================================================
# Paths
# ==========================================================

BASELINE_FILE = (
    RESULTS_DIR /
    "training" /
    "baseline_results.csv"
)

TUNED_FILE = (
    RESULTS_DIR /
    "comparison" /
    "tuned_results.csv"
)

SMOTE_FILE = (
    RESULTS_DIR /
    "comparison" /
    "smote_results.csv"
)

TABNET_FILE = (
    RESULTS_DIR /
    "comparison" /
    "tabnet_results.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR /
    "comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "model_comparison.csv"
)


# ==========================================================
# Load Results
# ==========================================================

print("=" * 60)
print("Loading Model Results")
print("=" * 60)

baseline = pd.read_csv(
    BASELINE_FILE
)

tuned = pd.read_csv(
    TUNED_FILE
)

smote = pd.read_csv(
    SMOTE_FILE
)

tabnet = pd.read_csv(
    TABNET_FILE
)


# ==========================================================
# Add Experiment Type
# ==========================================================

baseline["Experiment"] = "Baseline"

tuned["Experiment"] = "Tuned"

smote["Experiment"] = "SMOTE"

tabnet["Experiment"] = "TabNet"


# ==========================================================
# Combine Results
# ==========================================================

comparison = pd.concat(
    [
        baseline,
        tuned,
        smote,
        tabnet,
    ],
    ignore_index=True,
)


# ==========================================================
# Remove Duplicate Results
# ==========================================================

comparison = comparison.drop_duplicates(
    subset=[
        "Model",
        "Experiment",
    ],
    keep="last",
)


# ==========================================================
# Reorder Columns
# ==========================================================

columns = [
    "Model",
    "Experiment",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]

comparison = comparison[columns]


# ==========================================================
# Sort by ROC-AUC
# ==========================================================

comparison = comparison.sort_values(
    by="roc_auc",
    ascending=False,
)


# ==========================================================
# Save Final Comparison
# ==========================================================

comparison.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ==========================================================
# Display
# ==========================================================

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    comparison.to_string(
        index=False
    )
)

print("\n" + "=" * 60)
print(f"Total Experiments: {len(comparison)}")
print("=" * 60)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("MODEL COMPARISON COMPLETED")
print("=" * 60)