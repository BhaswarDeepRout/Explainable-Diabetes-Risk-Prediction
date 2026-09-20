"""
mlp_model.py
------------
Baseline MLP Classifier.
"""

from sklearn.neural_network import MLPClassifier

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

def build_mlp():

    model = MLPClassifier(

        hidden_layer_sizes=(128, 64),

        activation="relu",

        solver="adam",

        alpha=0.0001,

        batch_size=256,

        learning_rate="adaptive",

        learning_rate_init=0.001,

        max_iter=500,

        random_state=RANDOM_STATE,

        early_stopping=False,

        validation_fraction=0.1,
    )

    return model


# ==========================================================
# Train Model
# ==========================================================

def train_mlp(
    X_train,
    X_test,
    y_train,
    y_test,
):

    print("=" * 60)
    print("Training MLP")
    print("=" * 60)

    model = build_mlp()

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
        BASELINE_MODELS_DIR / "mlp_model.joblib",
    )

    print_metrics(
        "MLP",
        metrics,
    )

    return {
        "name": "MLP",
        "model": model,
        "predictions": y_pred,
        "probabilities": y_prob,
        "metrics": metrics,
    }