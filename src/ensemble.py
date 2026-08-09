"""
Phase 7 - Soft Voting Ensemble

Combines probability predictions from the tuned:
    - CatBoost
    - XGBoost
    - MLP
    - TabNet

No models are retrained.
The existing untouched test set is used only for final evaluation.
"""

import joblib
import numpy as np

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from config import (
    RANDOM_STATE,
    SAVED_MODELS_DIR,
    RESULTS_DIR,
)

from data_utils import (
    load_dataset,
    prepare_data,
    split_dataset,
    scale_features,
)

from utils import (
    print_metrics,
    save_results,
)


# ==========================================================
# Paths
# ==========================================================

TUNED_DIR = (
    SAVED_MODELS_DIR /
    "tuned"
)

SCALER_FILE = (
    SAVED_MODELS_DIR /
    "scaler.joblib"
)

ENSEMBLE_RESULTS_FILE = (
    RESULTS_DIR /
    "comparison" /
    "ensemble_results.csv"
)

ENSEMBLE_RESULTS_FILE.parent.mkdir(
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

X, y = prepare_data(dataset)

X_train, X_test, y_train, y_test = split_dataset(
    X,
    y,
)


# ==========================================================
# Load Scaler
# ==========================================================

print("\nLoading Scaler...")

scaler = joblib.load(
    SCALER_FILE
)

X_test_scaled = scaler.transform(
    X_test
)

X_test_scaled = np.asarray(
    X_test_scaled,
    dtype=np.float32,
)


# ==========================================================
# Load CatBoost
# ==========================================================

print("\nLoading Tuned CatBoost...")

catboost_model = CatBoostClassifier()

catboost_model.load_model(
    TUNED_DIR /
    "catboost_tuned.cbm"
)


# ==========================================================
# Load XGBoost
# ==========================================================

print("Loading Tuned XGBoost...")

xgb_model = XGBClassifier()

xgb_model.load_model(
    TUNED_DIR /
    "xgboost_tuned.json"
)


# ==========================================================
# Load MLP
# ==========================================================

print("Loading Tuned MLP...")

mlp_model = joblib.load(
    TUNED_DIR /
    "mlp_tuned.joblib"
)


# ==========================================================
# Load TabNet
# ==========================================================

print("Loading TabNet...")

tabnet_model = TabNetClassifier()

tabnet_model.load_model(
    str(
        SAVED_MODELS_DIR /
        "tabnet_model.zip"
    )
)


# ==========================================================
# Generate Probability Predictions
# ==========================================================

print("\nGenerating Probability Predictions...")

catboost_prob = (
    catboost_model
    .predict_proba(X_test)[:, 1]
)

xgb_prob = (
    xgb_model
    .predict_proba(X_test)[:, 1]
)

mlp_prob = (
    mlp_model
    .predict_proba(X_test_scaled)[:, 1]
)

tabnet_prob = (
    tabnet_model
    .predict_proba(X_test_scaled)[:, 1]
)


# ==========================================================
# 3-Model Ensemble
# ==========================================================

print("\n" + "=" * 60)
print("3-MODEL ENSEMBLE")
print("=" * 60)

ensemble_3_prob = (
    catboost_prob
    + xgb_prob
    + mlp_prob
) / 3.0


# ==========================================================
# Convert Probability to Class
# ==========================================================

ensemble_3_pred = (
    ensemble_3_prob >= 0.5
).astype(int)


# ==========================================================
# Evaluate 3-Model Ensemble
# ==========================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


ensemble_3_metrics = {
    "accuracy": accuracy_score(
        y_test,
        ensemble_3_pred,
    ),

    "precision": precision_score(
        y_test,
        ensemble_3_pred,
        zero_division=0,
    ),

    "recall": recall_score(
        y_test,
        ensemble_3_pred,
        zero_division=0,
    ),

    "f1": f1_score(
        y_test,
        ensemble_3_pred,
        zero_division=0,
    ),

    "roc_auc": roc_auc_score(
        y_test,
        ensemble_3_prob,
    ),

    "pr_auc": average_precision_score(
        y_test,
        ensemble_3_prob,
    ),

    "confusion_matrix": confusion_matrix(
        y_test,
        ensemble_3_pred,
    ),
}

print_metrics(
    "3-Model Ensemble",
    ensemble_3_metrics,
)

save_results(
    "3-Model Ensemble",
    ensemble_3_metrics,
    ENSEMBLE_RESULTS_FILE,
)


# ==========================================================
# 4-Model Ensemble
# ==========================================================

print("\n" + "=" * 60)
print("4-MODEL ENSEMBLE")
print("=" * 60)

ensemble_4_prob = (
    catboost_prob
    + xgb_prob
    + mlp_prob
    + tabnet_prob
) / 4.0


# ==========================================================
# Convert Probability to Class
# ==========================================================

ensemble_4_pred = (
    ensemble_4_prob >= 0.5
).astype(int)


# ==========================================================
# Evaluate 4-Model Ensemble
# ==========================================================

ensemble_4_metrics = {
    "accuracy": accuracy_score(
        y_test,
        ensemble_4_pred,
    ),

    "precision": precision_score(
        y_test,
        ensemble_4_pred,
        zero_division=0,
    ),

    "recall": recall_score(
        y_test,
        ensemble_4_pred,
        zero_division=0,
    ),

    "f1": f1_score(
        y_test,
        ensemble_4_pred,
        zero_division=0,
    ),

    "roc_auc": roc_auc_score(
        y_test,
        ensemble_4_prob,
    ),

    "pr_auc": average_precision_score(
        y_test,
        ensemble_4_prob,
    ),

    "confusion_matrix": confusion_matrix(
        y_test,
        ensemble_4_pred,
    ),
}

print_metrics(
    "4-Model Ensemble",
    ensemble_4_metrics,
)

save_results(
    "4-Model Ensemble",
    ensemble_4_metrics,
    ENSEMBLE_RESULTS_FILE,
)


# ==========================================================
# Save Ensemble Predictions
# ==========================================================

predictions_file = (
    RESULTS_DIR /
    "comparison" /
    "ensemble_predictions.csv"
)

predictions = np.column_stack(
    [
        y_test.to_numpy(),
        catboost_prob,
        xgb_prob,
        mlp_prob,
        tabnet_prob,
        ensemble_3_prob,
        ensemble_4_prob,
    ]
)

import pandas as pd

predictions_df = pd.DataFrame(
    predictions,
    columns=[
        "y_true",
        "catboost_prob",
        "xgboost_prob",
        "mlp_prob",
        "tabnet_prob",
        "ensemble_3_prob",
        "ensemble_4_prob",
    ],
)

predictions_df.to_csv(
    predictions_file,
    index=False,
)


# ==========================================================
# Completed
# ==========================================================

print("\n" + "=" * 60)
print("ENSEMBLE EXPERIMENT COMPLETED")
print("=" * 60)

print(
    f"\nResults saved to:\n"
    f"{ENSEMBLE_RESULTS_FILE}"
)

print(
    f"\nPredictions saved to:\n"
    f"{predictions_file}"
)