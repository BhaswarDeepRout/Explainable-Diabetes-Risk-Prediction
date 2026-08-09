"""
xgb_tuning.py
------------------
Hyperparameter tuning for XGBoost using Optuna.
"""


import optuna

from xgboost import XGBClassifier

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)

from config import (
    RANDOM_STATE,
    TUNED_MODELS_DIR,
)
from config import RESULTS_DIR

TUNED_RESULTS_FILE = (
    RESULTS_DIR /
    "comparison" /
    "tuned_results.csv"
)

from data_utils import (
    load_dataset,
    prepare_data,
    split_dataset,
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

def tune_xgboost():

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

            "n_estimators": trial.suggest_int(
                "n_estimators",
                200,
                800,
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                10,
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.30,
                log=True,
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),

            "gamma": trial.suggest_float(
                "gamma",
                0,
                5,
            ),

            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                1,
                10,
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                0,
                5,
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1,
                10,
            ),

            "objective": "binary:logistic",

            "eval_metric": "auc",

            "random_state": RANDOM_STATE,

            "tree_method": "hist",

            "verbosity": 0,

            "n_jobs": -1,
        }

        model = XGBClassifier(**params)

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
        "xgboost_best_params.json",
    )

    save_trials(
        study,
        "xgboost_trials.csv",
    )

    save_study(
        study,
        "xgboost_study.pkl",
    )

    plot_optimization_history(
        study,
        "xgboost_history.png",
    )

    plot_parameter_importance(
        study,
        "xgboost_importance.png",
    )

    # ------------------------------------------------------
    # Retrain Best Model
    # ------------------------------------------------------

    best_params = study.best_params

    best_params.update({

        "objective": "binary:logistic",

        "eval_metric": "auc",

        "random_state": RANDOM_STATE,

        "tree_method": "hist",

        "verbosity": 0,

        "n_jobs": -1,

    })

    best_model = XGBClassifier(
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
        "Tuned XGBOOST",
        metrics,
    )
    save_results(
        "XGBoost Tuned",
        metrics,
        TUNED_RESULTS_FILE,
    )
    save_model(

        best_model,

        TUNED_MODELS_DIR / "xgboost_tuned.json",

    )

if __name__ == "__main__":

    tune_xgboost()
