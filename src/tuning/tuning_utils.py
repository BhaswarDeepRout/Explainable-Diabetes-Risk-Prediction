"""
tuning_utils.py
---------------
Shared utilities for Optuna hyperparameter tuning.
"""

import json
import joblib
import optuna
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from config import TUNING_RESULTS_DIR


# ==========================================================
# Create Study
# ==========================================================

def create_study(direction="maximize"):

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    return study


# ==========================================================
# Save Best Parameters
# ==========================================================

def save_best_params(study, filename):

    output = TUNING_RESULTS_DIR / filename

    with open(output, "w") as f:
        json.dump(study.best_params, f, indent=4)

    print(f"Best parameters saved to:\n{output}")


# ==========================================================
# Save Trials
# ==========================================================

def save_trials(study, filename):

    output = TUNING_RESULTS_DIR / filename

    trials = study.trials_dataframe()

    trials.to_csv(output, index=False)

    print(f"Trials saved to:\n{output}")


# ==========================================================
# Save Study
# ==========================================================

def save_study(study, filename):

    output = TUNING_RESULTS_DIR / filename

    joblib.dump(study, output)

    print(f"Study saved to:\n{output}")


# ==========================================================
# Optimization History
# ==========================================================

def plot_optimization_history(study, filename):

    values = [
        trial.value
        for trial in study.trials
        if trial.value is not None
    ]

    plt.figure(figsize=(8,5))

    plt.plot(values)

    plt.xlabel("Trial")

    plt.ylabel("ROC-AUC")

    plt.title("Optimization History")

    plt.grid(True)

    output = TUNING_RESULTS_DIR / filename

    plt.tight_layout()

    plt.savefig(output, dpi=300)

    plt.close()


# ==========================================================
# Parameter Importance
# ==========================================================

def plot_parameter_importance(study, filename):

    importance = optuna.importance.get_param_importances(
        study
    )

    names = list(importance.keys())

    scores = list(importance.values())

    plt.figure(figsize=(8,5))

    plt.barh(names, scores)

    plt.xlabel("Importance")

    plt.title("Hyperparameter Importance")

    plt.tight_layout()

    output = TUNING_RESULTS_DIR / filename

    plt.savefig(output, dpi=300)

    plt.close()