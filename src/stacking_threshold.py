"""
Leakage-free threshold optimization for OOF stacking.

The optimal threshold is selected using OOF training
predictions only.

The untouched test set is used only once for final
evaluation.
"""

import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from config import RESULTS_DIR

from data_utils import (
    load_dataset,
    prepare_data,
    split_dataset,
)

from utils import (
    print_metrics,
    save_results,
)

# ==========================================================
# Paths
# ==========================================================

OOF_FILE = (
    RESULTS_DIR /
    "comparison" /
    "stacking_oof_meta_predictions.csv"
)

TEST_FILE = (
    RESULTS_DIR /
    "comparison" /
    "stacking_predictions.csv"
)

THRESHOLD_SEARCH_FILE = (
    RESULTS_DIR /
    "comparison" /
    "stacking_threshold_results.csv"
)

FINAL_RESULT_FILE = (
    RESULTS_DIR /
    "comparison" /
    "stacking_threshold_final_results.csv"
)

THRESHOLD_FILE = (
    RESULTS_DIR /
    "comparison" /
    "stacking_best_threshold.json"
)
# ==========================================================
# Load OOF Predictions
# ==========================================================

print("=" * 60)
print("Loading OOF Meta Predictions")
print("=" * 60)

oof = pd.read_csv(
    OOF_FILE
)


# ==========================================================
# OOF Threshold Optimization
# ==========================================================

y_oof = oof[
    "y_true"
].to_numpy()

oof_probability = oof[
    "stacking_oof_probability"
].to_numpy()


print("\n" + "=" * 60)
print("Optimizing Threshold Using OOF Predictions")
print("=" * 60)


thresholds = np.arange(
    0.10,
    0.91,
    0.01,
)


records = []


for threshold in thresholds:

    oof_prediction = (
        oof_probability >= threshold
    ).astype(int)

    records.append({

        "threshold":
            threshold,

        "accuracy":
            accuracy_score(
                y_oof,
                oof_prediction,
            ),

        "precision":
            precision_score(
                y_oof,
                oof_prediction,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_oof,
                oof_prediction,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_oof,
                oof_prediction,
                zero_division=0,
            ),
    })


threshold_df = pd.DataFrame(
    records
)


# ==========================================================
# Select Best Threshold
# ==========================================================

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)


print(
    f"\nBest OOF threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"OOF F1 at threshold: "
    f"{best_row['f1']:.4f}"
)


# ==========================================================
# Load Untouched Test Predictions
# ==========================================================

print("\n" + "=" * 60)
print("Loading Untouched Test Predictions")
print("=" * 60)

test_predictions = pd.read_csv(
    TEST_FILE
)


# ==========================================================
# Load Original Test Labels
# ==========================================================

dataset = load_dataset()

X, y = prepare_data(
    dataset
)

X_train, X_test, y_train, y_test = (
    split_dataset(
        X,
        y,
    )
)

y_test = np.asarray(
    y_test,
    dtype=int,
)


# ==========================================================
# Test Probabilities
# ==========================================================

test_probability = test_predictions[
    "stacking_probability"
].to_numpy()


# ==========================================================
# Apply Fixed OOF Threshold
# ==========================================================

test_prediction = (
    test_probability >= best_threshold
).astype(int)


# ==========================================================
# Final Test Evaluation
# ==========================================================

final_metrics = {

    "accuracy":
        accuracy_score(
            y_test,
            test_prediction,
        ),

    "precision":
        precision_score(
            y_test,
            test_prediction,
            zero_division=0,
        ),

    "recall":
        recall_score(
            y_test,
            test_prediction,
            zero_division=0,
        ),

    "f1":
        f1_score(
            y_test,
            test_prediction,
            zero_division=0,
        ),

    "roc_auc":
        roc_auc_score(
            y_test,
            test_probability,
        ),

    "pr_auc":
        average_precision_score(
            y_test,
            test_probability,
        ),

    "confusion_matrix":
        confusion_matrix(
            y_test,
            test_prediction,
        ),
}


# ==========================================================
# Print Final Results
# ==========================================================

print_metrics(
    "OOF Stacking - Leakage-Free Threshold",
    final_metrics,
)


# ==========================================================
# Save Threshold Search
# ==========================================================

threshold_df.to_csv(
    THRESHOLD_SEARCH_FILE,
    index=False,
)

print(
    f"\nThreshold search saved to:\n"
    f"{THRESHOLD_SEARCH_FILE}"
)

# ==========================================================
# Save Best Threshold
# ==========================================================

with open(
    THRESHOLD_FILE,
    "w",
) as f:

    json.dump(
        {
            "best_threshold":
                best_threshold,

            "selection_metric":
                "f1",

            "selection_data":
                "OOF training predictions",
        },
        f,
        indent=4,
    )


print(
    f"Best threshold saved to:\n"
    f"{THRESHOLD_FILE}"
)


# ==========================================================
# Save Final Result
# ==========================================================

save_results(
    "OOF Stacking Threshold",
    final_metrics,
    FINAL_RESULT_FILE,
)

print(
    f"\nFinal metrics saved to:\n"
    f"{FINAL_RESULT_FILE}"
)

# ==========================================================
# Completed
# ==========================================================

print("\n" + "=" * 60)
print("LEAKAGE-FREE THRESHOLD OPTIMIZATION COMPLETED")
print("=" * 60)