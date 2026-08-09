"""
Save evaluation results for already-trained tuned models.

This script does NOT retrain or retune any model.
It loads the saved tuned models, evaluates them on the
same untouched test set, and saves their metrics.
"""

import joblib

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from config import (
    RANDOM_STATE,
    RESULTS_DIR,
    TUNED_MODELS_DIR,
    SAVED_MODELS_DIR,
)

from data_utils import (
    load_dataset,
    prepare_data,
    split_dataset,
)

from utils import (
    evaluate_classifier,
    print_metrics,
    save_results,
)


# ==========================================================
# Result File
# ==========================================================

TUNED_RESULTS_FILE = (
    RESULTS_DIR /
    "comparison" /
    "tuned_results.csv"
)

TUNED_RESULTS_FILE.parent.mkdir(
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
# CatBoost
# ==========================================================

print("\n" + "=" * 60)
print("Evaluating Tuned CatBoost")
print("=" * 60)

catboost_model = CatBoostClassifier()

catboost_model.load_model(
    TUNED_MODELS_DIR /
    "catboost_tuned.cbm"
)

y_pred, y_prob, metrics = evaluate_classifier(
    catboost_model,
    X_test,
    y_test,
)

print_metrics(
    "Tuned CatBoost",
    metrics,
)

save_results(
    "CatBoost Tuned",
    metrics,
    TUNED_RESULTS_FILE,
)


# ==========================================================
# XGBoost
# ==========================================================

print("\n" + "=" * 60)
print("Evaluating Tuned XGBoost")
print("=" * 60)

xgb_model = XGBClassifier()

xgb_model.load_model(
    TUNED_MODELS_DIR /
    "xgboost_tuned.json"
)

y_pred, y_prob, metrics = evaluate_classifier(
    xgb_model,
    X_test,
    y_test,
)

print_metrics(
    "Tuned XGBoost",
    metrics,
)

save_results(
    "XGBoost Tuned",
    metrics,
    TUNED_RESULTS_FILE,
)


# ==========================================================
# MLP
# ==========================================================

print("\n" + "=" * 60)
print("Evaluating Tuned MLP")
print("=" * 60)

mlp_model = joblib.load(
    TUNED_MODELS_DIR /
    "mlp_tuned.joblib"
)


# ----------------------------------------------------------
# Load the scaler used for MLP
# ----------------------------------------------------------

scaler = joblib.load(
    SAVED_MODELS_DIR /
    "scaler.joblib"
)

X_test_scaled = scaler.transform(
    X_test
)


# ----------------------------------------------------------
# Evaluate MLP
# ----------------------------------------------------------

y_pred, y_prob, metrics = evaluate_classifier(
    mlp_model,
    X_test_scaled,
    y_test,
)

print_metrics(
    "Tuned MLP",
    metrics,
)

save_results(
    "MLP Tuned",
    metrics,
    TUNED_RESULTS_FILE,
)


# ==========================================================
# Completed
# ==========================================================

print("\n" + "=" * 60)
print("TUNED MODEL RESULTS SAVED")
print("=" * 60)

print(
    f"\nResults saved to:\n"
    f"{TUNED_RESULTS_FILE}"
)