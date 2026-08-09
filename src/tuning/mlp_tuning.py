"""
mlp_tuning.py
------------------
Hyperparameter tuning for XGBoost using Optuna.
"""


import optuna
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)

from config import (
    RANDOM_STATE,
    TUNED_MODELS_DIR,
)

from data_utils import (
    load_dataset,
    prepare_data,
    split_dataset,
    scale_features,
)
from tuning.tuning_utils import (
    create_study,
    save_best_params,
    save_trials,
    save_study,
    plot_optimization_history,
    plot_parameter_importance,
)
from config import RESULTS_DIR

TUNED_RESULTS_FILE = (
    RESULTS_DIR /
    "comparison" /
    "tuned_results.csv"
)


from utils import (
    evaluate_classifier,
    print_metrics,
    save_model,
    save_results,
)


# ==========================================================
# Objective Function
# ==========================================================


# ==========================================================
# MLP Tuning
# ==========================================================

def tune_mlp():

    # ---------------------------------------------
    # Load Dataset
    # ---------------------------------------------

    dataset = load_dataset()

    X, y = prepare_data(dataset)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
    )
    X_train, X_test, scaler = scale_features(
        X_train,
        X_test,
    )
    # ---------------------------------------------
    # Cross Validation
    # ---------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(trial):
        params = {

            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes",
                [
                    (64,),
                    (128,),
                    (64, 32),
                    (128, 64),
                    (256, 128),
                ],
            ),

            "activation": trial.suggest_categorical(
                "activation",
                [
                    "relu",
                    "tanh",
                ],
            ),

            "alpha": trial.suggest_float(
                "alpha",
                1e-5,
                1e-2,
                log=True,
            ),

            "learning_rate_init": trial.suggest_float(
                "learning_rate_init",
                1e-4,
                1e-1,
                log=True,
            ),

            "batch_size": trial.suggest_categorical(
                "batch_size",
                [
                    32,
                    64,
                    128,
                    256,
                ],
            ),

            "max_iter": trial.suggest_int(
                "max_iter",
                300,
                800,
            ),

            "early_stopping": True,

            "random_state": RANDOM_STATE,
        }

        model = MLPClassifier(**params)

        scores = cross_validate(

            estimator=model,

            X=X_train,

            y=y_train,

            cv=cv,

            scoring={
                "roc_auc": "roc_auc",
            },

            n_jobs=-1,

            return_train_score=False,
        )

        return scores["test_roc_auc"].mean()
    # ------------------------------------------------------
    # Create Study
    # ------------------------------------------------------

    study = create_study()

    study.optimize(
        objective,
        n_trials=20,
        show_progress_bar=True,
    )

    print("\n" + "=" * 60)
    print("Best ROC-AUC:", study.best_value)
    print("=" * 60)

    print("\nBest Parameters:")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    # ------------------------------------------------------
    # Save Study Results
    # ------------------------------------------------------

    save_best_params(
        study,
        "mlp_best_params.json",
    )

    save_trials(
        study,
        "mlp_trials.csv",
    )

    save_study(
        study,
        "mlp_study.pkl",
    )

    plot_optimization_history(
        study,
        "mlp_history.png",
    )

    plot_parameter_importance(
        study,
        "mlp_importance.png",
    )

    # ------------------------------------------------------
    # Retrain Best Model
    # ------------------------------------------------------

    best_params = study.best_params

    best_params.update({

        "early_stopping": True,

        "random_state": RANDOM_STATE,

    })

    best_model = MLPClassifier(
        **best_params
    )

    best_model.fit(
        X_train,
        y_train,
    )

    y_pred, y_prob, metrics = evaluate_classifier(

        best_model,

        X_test,

        y_test,

    )

    print_metrics(
        "Tuned MLPBOOST",
        metrics,
    )
    save_results(
        "MLP Tuned",
        metrics,
        TUNED_RESULTS_FILE,
    )
    save_model(

        best_model,

        TUNED_MODELS_DIR / "mlp_tuned.joblib",

    )

if __name__ == "__main__":

    tune_mlp()
