"""
train.py
--------
Main training pipeline.

Responsibilities:
1. Load engineered dataset
2. Prepare data
3. Train/Test split
4. Scale continuous features
5. Train baseline models
6. Save experiment results
"""

import pandas as pd

from data_utils import (
    load_dataset,
    prepare_data,
    split_dataset,
    scale_features,
    save_split_indices,
)

from config import RESULTS_DIR

from models.catboost_model import train_catboost
from models.xgboost_model import train_xgboost
from models.mlp_model import train_mlp


# ==========================================================
# Results Directory
# ==========================================================

TRAINING_RESULTS_DIR = RESULTS_DIR / "training"
TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Save Metrics
# ==========================================================

def save_results(results):
    """
    Save baseline model performance.
    """

    rows = []

    for result in results:

        metrics = result["metrics"].copy()

        metrics.pop("confusion_matrix", None)

        metrics["Model"] = result["name"]

        rows.append(metrics)

    df = pd.DataFrame(rows)

    columns = [
        "Model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    df = df[columns]

    output = TRAINING_RESULTS_DIR / "baseline_results.csv"

    df.to_csv(output, index=False)

    print(f"\nBaseline results saved to:\n{output}")


# ==========================================================
# Main
# ==========================================================

def main():

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    dataset = load_dataset()

    X, y = prepare_data(dataset)

    # ------------------------------------------------------
    # Train/Test Split
    # ------------------------------------------------------

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
    )

    # ------------------------------------------------------
    # Scale Continuous Features
    # ------------------------------------------------------

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test,
    )

    save_split_indices(
        X_train,
        X_test,
    )

    results = []

    # ======================================================
    # CatBoost
    # ======================================================

    catboost_results = train_catboost(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    results.append(catboost_results)

    # ======================================================
    # XGBoost
    # ======================================================

    xgboost_results = train_xgboost(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    results.append(xgboost_results)

    # ======================================================
    # MLP
    # ======================================================

    mlp_results = train_mlp(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )

    results.append(mlp_results)

    # ======================================================
    # Save Results
    # ======================================================

    save_results(results)

    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETED")
    print("=" * 60)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()