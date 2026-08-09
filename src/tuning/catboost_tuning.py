"""
catboost_tuning.py
------------------
Hyperparameter tuning for CatBoost using Optuna.
"""


import optuna

from catboost import CatBoostClassifier

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
)
from config import RESULTS_DIR

TUNED_RESULTS_FILE = (
    RESULTS_DIR /
    "comparison" /
    "tuned_results.csv"
)


from tuning.tuning_utils import (
    create_study,
    save_best_params,
    save_trials,
    save_study,
    plot_optimization_history,
    plot_parameter_importance,
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
# CatBoost Tuning
# ==========================================================

def tune_catboost():

    # ---------------------------------------------
    # Load Dataset
    # ---------------------------------------------

    dataset = load_dataset()

    X, y = prepare_data(dataset)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
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

            "iterations": trial.suggest_int(
                "iterations",
                200,
                800,
            ),

            "depth": trial.suggest_int(
                "depth",
                4,
                10,
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.30,
                log=True,
            ),

            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg",
                1,
                15,
            ),

            "bagging_temperature": trial.suggest_float(
                "bagging_temperature",
                0,
                10,
            ),

            "random_strength": trial.suggest_float(
                "random_strength",
                0,
                10,
            ),

            "border_count": trial.suggest_int(
                "border_count",
                32,
                255,
            ),

            "loss_function": "Logloss",

            "eval_metric": "AUC",

            "verbose": False,

            "random_seed": RANDOM_STATE,
        }

        model = CatBoostClassifier(
            **params
        )

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
        "catboost_best_params.json",
    )

    save_trials(
        study,
        "catboost_trials.csv",
    )

    save_study(
        study,
        "catboost_study.pkl",
    )

    plot_optimization_history(
        study,
        "catboost_history.png",
    )

    plot_parameter_importance(
        study,
        "catboost_importance.png",
    )

    # ------------------------------------------------------
    # Retrain Best Model
    # ------------------------------------------------------

    best_params = study.best_params

    best_params.update({

        "loss_function": "Logloss",

        "eval_metric": "AUC",

        "verbose": False,

        "random_seed": RANDOM_STATE,

    })

    best_model = CatBoostClassifier(
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
        "Tuned CatBoost",
        metrics,
    )
    save_results(
        "CatBoost Tuned",
        metrics,
        TUNED_RESULTS_FILE,
    )
    save_model(

        best_model,

        TUNED_MODELS_DIR / "catboost_tuned.cbm",

    )

if __name__ == "__main__":

    tune_catboost()
