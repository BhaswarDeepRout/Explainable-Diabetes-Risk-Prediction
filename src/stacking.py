"""
Phase 7 - Out-of-Fold Stacking

Base models:
    - Tuned CatBoost
    - Tuned XGBoost
    - Tuned MLP

Meta learner:
    - Logistic Regression

The test set is kept completely untouched until final evaluation.
"""

import json
import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import (
    RANDOM_STATE,
    RESULTS_DIR,
    TUNED_MODELS_DIR,
)

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

STACKING_RESULTS_FILE = (
    RESULTS_DIR /
    "comparison" /
    "stacking_results.csv"
)

STACKING_PREDICTIONS_FILE = (
    RESULTS_DIR /
    "comparison" /
    "stacking_predictions.csv"
)

STACKING_MODEL_FILE = (
    TUNED_MODELS_DIR /
    "stacking_meta_model.joblib"
)

STACKING_RESULTS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Dataset for Stacking")
print("=" * 60)

dataset = load_dataset()

X, y = prepare_data(dataset)

X_train, X_test, y_train, y_test = split_dataset(
    X,
    y,
)


# ==========================================================
# Convert Labels
# ==========================================================

y_train = np.asarray(
    y_train,
    dtype=np.int64,
)

y_test = np.asarray(
    y_test,
    dtype=np.int64,
)


# ==========================================================
# Feature Scaling for MLP
# ==========================================================

print("\nPreparing MLP Scaling...")

mlp_scaler = StandardScaler()

X_train_mlp = mlp_scaler.fit_transform(
    X_train
)

X_test_mlp = mlp_scaler.transform(
    X_test
)


# ==========================================================
# Cross Validation
# ==========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ==========================================================
# OOF Prediction Arrays
# ==========================================================

oof_catboost = np.zeros(
    len(X_train)
)

oof_xgboost = np.zeros(
    len(X_train)
)

oof_mlp = np.zeros(
    len(X_train)
)


# ==========================================================
# Test Prediction Arrays
# ==========================================================

test_catboost = np.zeros(
    len(X_test)
)

test_xgboost = np.zeros(
    len(X_test)
)

test_mlp = np.zeros(
    len(X_test)
)


# ==========================================================
# Load Best Parameters
# ==========================================================

print("\nLoading tuned parameters...")

with open(
    RESULTS_DIR /
    "tuning" /
    "catboost_best_params.json",
    "r",
) as f:

    catboost_params = json.load(f)


with open(
    RESULTS_DIR /
    "tuning" /
    "xgboost_best_params.json",
    "r",
) as f:

    xgboost_params = json.load(f)


# ==========================================================
# Cross-Validation Training
# ==========================================================

print("\n" + "=" * 60)
print("Generating Out-of-Fold Predictions")
print("=" * 60)


for fold, (train_idx, val_idx) in enumerate(
    cv.split(X_train, y_train),
    start=1,
):

    print("\n" + "-" * 60)
    print(f"STACKING FOLD {fold}/5")
    print("-" * 60)


    # ------------------------------------------------------
    # Split Fold
    # ------------------------------------------------------

    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]

    y_fold_train = y_train[train_idx]
    y_fold_val = y_train[val_idx]


    # ======================================================
    # CatBoost
    # ======================================================

    print("Training CatBoost...")

    catboost_model = CatBoostClassifier(
        **catboost_params,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_seed=RANDOM_STATE,
    )

    catboost_model.fit(
        X_fold_train,
        y_fold_train,
    )

    oof_catboost[val_idx] = (
        catboost_model
        .predict_proba(X_fold_val)[:, 1]
    )

    test_catboost += (
        catboost_model
        .predict_proba(X_test)[:, 1]
        / cv.n_splits
    )


    # ======================================================
    # XGBoost
    # ======================================================

    print("Training XGBoost...")

    xgb_model = XGBClassifier(
        **xgboost_params,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )

    xgb_model.fit(
        X_fold_train,
        y_fold_train,
    )

    oof_xgboost[val_idx] = (
        xgb_model
        .predict_proba(X_fold_val)[:, 1]
    )

    test_xgboost += (
        xgb_model
        .predict_proba(X_test)[:, 1]
        / cv.n_splits
    )


    # ======================================================
    # MLP
    # ======================================================

    print("Training MLP...")

    fold_scaler = StandardScaler()

    X_fold_train_mlp = (
        fold_scaler.fit_transform(
            X_fold_train
        )
    )

    X_fold_val_mlp = (
        fold_scaler.transform(
            X_fold_val
        )
    )

    X_test_fold_mlp = (
        fold_scaler.transform(
            X_test
        )
    )

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=0.0001,
        learning_rate_init=0.001,
        batch_size=128,
        max_iter=500,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )

    mlp_model.fit(
        X_fold_train_mlp,
        y_fold_train,
    )

    oof_mlp[val_idx] = (
        mlp_model
        .predict_proba(X_fold_val_mlp)[:, 1]
    )

    test_mlp += (
        mlp_model
        .predict_proba(X_test_fold_mlp)[:, 1]
        / cv.n_splits
    )


# ==========================================================
# Create OOF Meta Features
# ==========================================================

print("\n" + "=" * 60)
print("Creating Meta Features")
print("=" * 60)

X_meta_train = np.column_stack(
    [
        oof_catboost,
        oof_xgboost,
        oof_mlp,
    ]
)

X_meta_test = np.column_stack(
    [
        test_catboost,
        test_xgboost,
        test_mlp,
    ]
)
# ==========================================================
# Train Meta Learner
# ==========================================================

print("\nTraining Logistic Regression Meta-Learner...")

meta_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000,
)

meta_model.fit(
    X_meta_train,
    y_train,
)
# ==========================================================
# Generate OOF Meta Probabilities
# ==========================================================

oof_meta_probability = (
    meta_model
    .predict_proba(X_meta_train)[:, 1]
)

# ==========================================================
# Train Meta Learner
# ==========================================================

print("\nTraining Logistic Regression Meta-Learner...")

meta_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000,
)

meta_model.fit(
    X_meta_train,
    y_train,
)


# ==========================================================
# Final Stacking Predictions
# ==========================================================

stacking_prob = (
    meta_model
    .predict_proba(X_meta_test)[:, 1]
)

stacking_pred = (
    stacking_prob >= 0.5
).astype(int)


# ==========================================================
# Evaluate Stacking
# ==========================================================

stacking_metrics = {

    "accuracy": accuracy_score(
        y_test,
        stacking_pred,
    ),

    "precision": precision_score(
        y_test,
        stacking_pred,
        zero_division=0,
    ),

    "recall": recall_score(
        y_test,
        stacking_pred,
        zero_division=0,
    ),

    "f1": f1_score(
        y_test,
        stacking_pred,
        zero_division=0,
    ),

    "roc_auc": roc_auc_score(
        y_test,
        stacking_prob,
    ),

    "pr_auc": average_precision_score(
        y_test,
        stacking_prob,
    ),

    "confusion_matrix": confusion_matrix(
        y_test,
        stacking_pred,
    ),
}


# ==========================================================
# Print Results
# ==========================================================

print_metrics(
    "OOF Stacking",
    stacking_metrics,
)


# ==========================================================
# Save Results
# ==========================================================

save_results(
    "OOF Stacking",
    stacking_metrics,
    STACKING_RESULTS_FILE,
)


# ==========================================================
# Save Meta Model
# ==========================================================

joblib.dump(
    meta_model,
    STACKING_MODEL_FILE,
)

print(
    f"\nMeta-model saved to:\n"
    f"{STACKING_MODEL_FILE}"
)


# ==========================================================
# Save Predictions
# ==========================================================
stacking_predictions = pd.DataFrame({

    "y_true": y_test,

    "catboost_test_prob":
        test_catboost,

    "xgboost_test_prob":
        test_xgboost,

    "mlp_test_prob":
        test_mlp,

    "stacking_probability":
        stacking_prob,

    "stacking_prediction":
        stacking_pred,
})
stacking_predictions.to_csv(
    STACKING_PREDICTIONS_FILE,
    index=False,
)

print(
    f"\nTest predictions saved to:\n"
    f"{STACKING_PREDICTIONS_FILE}"
)
# ==========================================================
# Save OOF Meta Predictions
# ==========================================================

OOF_META_FILE = (
    RESULTS_DIR /
    "comparison" /
    "stacking_oof_meta_predictions.csv"
)

oof_meta_predictions = pd.DataFrame({

    "y_true":
        y_train,

    "stacking_oof_probability":
        oof_meta_probability,
})

oof_meta_predictions.to_csv(
    OOF_META_FILE,
    index=False,
)

print(
    f"\nOOF meta predictions saved to:\n"
    f"{OOF_META_FILE}"
)


# ==========================================================
# Completed
# ==========================================================

print("\n" + "=" * 60)
print("OOF STACKING COMPLETED")
print("=" * 60)

print(
    f"\nResults saved to:\n"
    f"{STACKING_RESULTS_FILE}"
)

print(
    f"\nPredictions saved to:\n"
    f"{STACKING_PREDICTIONS_FILE}"
)