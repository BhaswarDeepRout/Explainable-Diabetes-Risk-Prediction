"""
xgboost_model.py
----------------
Baseline XGBoost model.
"""

from xgboost import XGBClassifier

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


# ==========================================================
# Build Model
# ==========================================================

def build_xgboost(scale_pos_weight):

    model = XGBClassifier(

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        eval_metric="logloss",

        random_state=RANDOM_STATE,

        scale_pos_weight=scale_pos_weight,

        n_jobs=-1,
    )

    return model

from utils import (
    evaluate_classifier,
    print_metrics,
    save_model,
)
# ==========================================================
# Evaluate Model
# ==========================================================


# ==========================================================
# Train Model
# ==========================================================

def train_xgboost(

    X_train,

    X_test,

    y_train,

    y_test,
):

    print("=" * 60)

    print("Training XGBoost")

    print("=" * 60)

    negatives = (y_train == 0).sum()

    positives = (y_train == 1).sum()

    scale_pos_weight = negatives / positives

    model = build_xgboost(scale_pos_weight)

    model.fit(

        X_train,

        y_train,
    )

    y_pred, y_prob, metrics = evaluate_classifier(

        model,

        X_test,

        y_test,
    )

    save_model(model,BASELINE_MODELS_DIR / "xgboost_model.json")

    print("\nXGBoost Performance")

    print("-" * 60)

    for metric, value in metrics.items():

        if metric == "confusion_matrix":

            print(f"\n{metric}")

            print(value)

        else:

            print(f"{metric:<20}: {value:.4f}")

    return {

        "name": "XGBoost",

        "model": model,

        "predictions": y_pred,

        "probabilities": y_prob,

        "metrics": metrics,
    }