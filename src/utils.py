"""
utils.py
--------
Shared utility functions for model training and evaluation.
"""

from pathlib import Path
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

import pandas as pd


# ==========================================================
# Evaluate Classifier
# ==========================================================

def evaluate_classifier(model, x_test, y_test):
    """
    Evaluate a binary classification model.

    Returns
    -------
    y_pred : ndarray
    y_prob : ndarray
    metrics : dict
    """

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return y_pred, y_prob, metrics


# ==========================================================
# Print Metrics
# ==========================================================

def print_metrics(model_name, metrics):
    """
    Display evaluation metrics.
    """

    print("\n" + "=" * 60)
    print(f"{model_name} Performance")
    print("=" * 60)

    for metric, value in metrics.items():

        if metric == "confusion_matrix":
            print(f"\n{metric}")
            print(value)

        else:
            print(f"{metric:<20}: {value:.4f}")


# ==========================================================
# Save Model
# ==========================================================

def save_model(model, filepath: Path):
    """
    Save a trained model.

    Supports:
        - CatBoost (.cbm)
        - XGBoost (.json)
        - Scikit-learn (.joblib)
    """
    print(filepath)
    print(filepath.suffix)

    suffix = filepath.suffix.lower()

    if suffix == ".cbm":
        model.save_model(filepath)

    elif suffix == ".json":
        model.save_model(filepath)

    elif suffix == ".joblib":
        joblib.dump(model, filepath)

    else:
        raise ValueError(
            f"Unsupported model format: {suffix}"
        )

    print(f"\nModel saved to:\n{filepath}")

# ==========================================================
# Save Model Results
# ==========================================================

def save_results(model_name, metrics, filepath):
    """
    Save model evaluation metrics to a CSV file.
    """

    row = {
        "Model": model_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
    }

    df = pd.DataFrame([row])

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if filepath.exists():

        df.to_csv(
            filepath,
            mode="a",
            header=False,
            index=False,
        )

    else:

        df.to_csv(
            filepath,
            index=False,
        )

    print(f"\nResults saved to:\n{filepath}")