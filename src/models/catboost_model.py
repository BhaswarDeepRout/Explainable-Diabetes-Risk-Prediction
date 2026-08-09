"""
catboost_model.py
-----------------
Baseline CatBoost model.
"""

import joblib

from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from config import (
    RANDOM_STATE,
    BASELINE_MODELS_DIR,
)
from utils import (
    evaluate_classifier,
    print_metrics,
    save_model,
)

# ==========================================================
# Build Model
# ==========================================================

def build_catboost():

    model = CatBoostClassifier(

        iterations=300,

        learning_rate=0.05,

        depth=6,

        loss_function="Logloss",

        eval_metric="AUC",

        random_seed=RANDOM_STATE,

        verbose=False,
    )

    return model





# ==========================================================
# Train Model
# ==========================================================

def train_catboost(

    X_train,

    X_test,

    y_train,

    y_test,
):

    print("=" * 60)

    print("Training CatBoost")

    print("=" * 60)

    model = build_catboost()

    model.fit(

        X_train,

        y_train,
    )

    y_pred, y_prob, metrics = evaluate_classifier(
        model,
        X_test,
        y_test,
    )

    save_model(
        model,
        BASELINE_MODELS_DIR/ "catboost_model.cbm"
    )

    print_metrics(
        "CatBoost",
        metrics,
    )

    return {
        "name": "CatBoost",
        "model": model,
        "predictions": y_pred,
        "probabilities": y_prob,
        "metrics": metrics,
    }