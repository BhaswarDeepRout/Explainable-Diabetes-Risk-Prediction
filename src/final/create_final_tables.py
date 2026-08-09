from pathlib import Path

import pandas as pd

from config import RESULTS_DIR


# ==========================================================
# Paths
# ==========================================================

COMPARISON_FILE = (
    RESULTS_DIR
    / "comparison"
    / "model_comparison.csv"
)

STACKING_FILE = (
    RESULTS_DIR
    / "comparison"
    / "stacking_results.csv"
)

THRESHOLD_FILE = (
    RESULTS_DIR
    / "comparison"
    / "stacking_threshold_results.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# Load Model Comparison
# ==========================================================

print("=" * 70)
print("CREATING FINAL RESEARCH TABLES")
print("=" * 70)

comparison = pd.read_csv(
    COMPARISON_FILE
)


# ==========================================================
# Final Ranking
# ==========================================================

ranking = (
    comparison
    .sort_values(
        by="roc_auc",
        ascending=False,
    )
    .reset_index(drop=True)
)

ranking.insert(
    0,
    "ROC_AUC_Rank",
    range(1, len(ranking) + 1),
)


ranking_file = (
    OUTPUT_DIR
    / "final_model_ranking.csv"
)

ranking.to_csv(
    ranking_file,
    index=False,
)


# ==========================================================
# Best Model
# ==========================================================

best_model = ranking.iloc[0]


best_summary = pd.DataFrame({
    "Model": [
        best_model["Model"]
    ],

    "Experiment": [
        best_model["Experiment"]
    ],

    "Accuracy": [
        best_model["accuracy"]
    ],

    "Precision": [
        best_model["precision"]
    ],

    "Recall": [
        best_model["recall"]
    ],

    "F1": [
        best_model["f1"]
    ],

    "ROC_AUC": [
        best_model["roc_auc"]
    ],

    "PR_AUC": [
        best_model["pr_auc"]
    ],
})


best_file = (
    OUTPUT_DIR
    / "best_model_summary.csv"
)

best_summary.to_csv(
    best_file,
    index=False,
)


# ==========================================================
# Stacking Results
# ==========================================================

if STACKING_FILE.exists():

    stacking = pd.read_csv(
        STACKING_FILE
    )

    stacking_file = (
        OUTPUT_DIR
        / "stacking_summary.csv"
    )

    stacking.to_csv(
        stacking_file,
        index=False,
    )

else:

    print(
        "\nWARNING:"
        " stacking_results.csv not found."
    )


# ==========================================================
# Threshold Results
# ==========================================================

if THRESHOLD_FILE.exists():

    threshold = pd.read_csv(
        THRESHOLD_FILE
    )

    threshold_file = (
        OUTPUT_DIR
        / "threshold_search.csv"
    )

    threshold.to_csv(
        threshold_file,
        index=False,
    )

    best_threshold_row = (
        threshold
        .loc[
            threshold["f1"].idxmax()
        ]
    )

    threshold_summary = pd.DataFrame({

        "Best_Threshold": [
            best_threshold_row[
                "threshold"
            ]
        ],

        "Accuracy": [
            best_threshold_row[
                "accuracy"
            ]
        ],

        "Precision": [
            best_threshold_row[
                "precision"
            ]
        ],

        "Recall": [
            best_threshold_row[
                "recall"
            ]
        ],

        "F1": [
            best_threshold_row[
                "f1"
            ]
        ],
    })

    threshold_summary.to_csv(
        OUTPUT_DIR
        / "best_threshold_summary.csv",
        index=False,
    )

else:

    print(
        "\nWARNING:"
        " stacking_threshold_results.csv"
        " not found."
    )


# ==========================================================
# Dataset Summary
# ==========================================================

dataset_summary = pd.DataFrame({

    "Item": [

        "Final dataset rows",

        "Final dataset columns",

        "Training samples",

        "Testing samples",

        "Missing values",

        "Duplicate rows",

        "Positive samples",

        "Negative samples",

        "Positive class rate",

        "Train/test feature overlap",
    ],

    "Value": [

        95964,

        16,

        76771,

        19193,

        0,

        0,

        8391,

        87573,

        0.0874,

        0,
    ],
})


dataset_file = (
    OUTPUT_DIR
    / "dataset_summary.csv"
)

dataset_summary.to_csv(
    dataset_file,
    index=False,
)


# ==========================================================
# Display Final Ranking
# ==========================================================

print("\n" + "=" * 70)
print("FINAL MODEL RANKING")
print("=" * 70)

print(
    ranking.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Model      : {best_model['Model']}"
)

print(
    f"Experiment : {best_model['Experiment']}"
)

print(
    f"ROC-AUC    : {best_model['roc_auc']:.4f}"
)

print(
    f"PR-AUC     : {best_model['pr_auc']:.4f}"
)

print(
    f"F1         : {best_model['f1']:.4f}"
)

print(
    f"Recall     : {best_model['recall']:.4f}"
)

print(
    f"Precision  : {best_model['precision']:.4f}"
)


# ==========================================================
# Output Files
# ==========================================================

print("\n" + "=" * 70)
print("FINAL TABLES CREATED")
print("=" * 70)

print(
    f"\nOutput directory:\n"
    f"{OUTPUT_DIR}"
)

print(
    "\nCreated:"
)

print(
    f" - {ranking_file.name}"
)

print(
    f" - {best_file.name}"
)

print(
    f" - {dataset_file.name}"
)

if STACKING_FILE.exists():

    print(
        f" - {stacking_file.name}"
    )

if THRESHOLD_FILE.exists():

    print(
        f" - threshold_search.csv"
    )

    print(
        f" - best_threshold_summary.csv"
    )

print("\nFINAL TABLE GENERATION COMPLETED")
print("=" * 70)
