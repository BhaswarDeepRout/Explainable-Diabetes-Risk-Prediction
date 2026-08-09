"""
Final Performance Analysis
--------------------------
Evaluation plots for the selected tuned XGBoost model:

1. ROC curve
2. Precision-Recall curve
3. Confusion matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

from config import (
    RESULTS_DIR,
    SAVED_MODELS_DIR,
)

from data_utils import (
    load_dataset,
    prepare_data,
    split_dataset,
)


# ==========================================================
# Paths
# ==========================================================

MODEL_FILE = (
    SAVED_MODELS_DIR /
    "tuned" /
    "xgboost_tuned.json"
)

OUTPUT_DIR = (
    RESULTS_DIR /
    "performance"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

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


# ==========================================================
# Load Final XGBoost
# ==========================================================

print("\n" + "=" * 60)
print("Loading Tuned XGBoost")
print("=" * 60)

model = XGBClassifier()

model.load_model(
    MODEL_FILE
)

print(
    f"Model loaded from:\n"
    f"{MODEL_FILE}"
)


# ==========================================================
# Predictions
# ==========================================================

print("\nGenerating predictions...")

y_probability = (
    model
    .predict_proba(X_test)[:, 1]
)

y_prediction = (
    y_probability >= 0.5
).astype(int)


# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_prediction,
)

precision = precision_score(
    y_test,
    y_prediction,
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_prediction,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_prediction,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    y_probability,
)

pr_auc = average_precision_score(
    y_test,
    y_probability,
)

cm = confusion_matrix(
    y_test,
    y_prediction,
)


# ==========================================================
# Print Metrics
# ==========================================================

print("\n" + "=" * 60)
print("FINAL TUNED XGBOOST PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy            : {accuracy:.4f}"
)

print(
    f"Precision           : {precision:.4f}"
)

print(
    f"Recall              : {recall:.4f}"
)

print(
    f"F1                  : {f1:.4f}"
)

print(
    f"ROC-AUC             : {roc_auc:.4f}"
)

print(
    f"PR-AUC              : {pr_auc:.4f}"
)

print("\nConfusion Matrix")
print(cm)


# ==========================================================
# ROC Curve
# ==========================================================

print("\nCreating ROC curve...")

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    y_probability,
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc:.4f})",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Tuned XGBoost"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

roc_file = (
    OUTPUT_DIR /
    "xgboost_roc_curve.png"
)

plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"ROC curve saved to:\n"
    f"{roc_file}"
)


# ==========================================================
# Precision-Recall Curve
# ==========================================================

print("\nCreating Precision-Recall curve...")

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_test,
        y_probability,
    )
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall_curve,
    precision_curve,
    label=f"XGBoost (AP = {pr_auc:.4f})",
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve - Tuned XGBoost"
)

plt.legend(
    loc="lower left"
)

plt.tight_layout()

pr_file = (
    OUTPUT_DIR /
    "xgboost_precision_recall_curve.png"
)

plt.savefig(
    pr_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"Precision-Recall curve saved to:\n"
    f"{pr_file}"
)


# ==========================================================
# Confusion Matrix
# ==========================================================

print("\nCreating confusion matrix...")

fig, ax = plt.subplots(
    figsize=(7, 6)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Diabetes",
        "Diabetes",
    ],
)

display.plot(
    ax=ax,
)

ax.set_title(
    "Confusion Matrix - Tuned XGBoost"
)

plt.tight_layout()

cm_file = (
    OUTPUT_DIR /
    "xgboost_confusion_matrix.png"
)

plt.savefig(
    cm_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"Confusion matrix saved to:\n"
    f"{cm_file}"
)


# ==========================================================
# Save Final Metrics
# ==========================================================

metrics_df = pd.DataFrame({

    "Model": [
        "XGBoost Tuned"
    ],

    "accuracy": [
        accuracy
    ],

    "precision": [
        precision
    ],

    "recall": [
        recall
    ],

    "f1": [
        f1
    ],

    "roc_auc": [
        roc_auc
    ],

    "pr_auc": [
        pr_auc
    ],
})

metrics_file = (
    OUTPUT_DIR /
    "xgboost_final_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False,
)

print(
    f"\nFinal metrics saved to:\n"
    f"{metrics_file}"
)


# ==========================================================
# Completed
# ==========================================================

print("\n" + "=" * 60)
print("PERFORMANCE ANALYSIS COMPLETED")
print("=" * 60)