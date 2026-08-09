"""Explainability utilities for the diabetes prediction pipeline."""
"""
Model Explainability
--------------------
Explain the final tuned XGBoost model using:

1. Built-in XGBoost feature importance
2. SHAP global feature importance
3. SHAP summary plot
4. SHAP bar plot
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from xgboost import XGBClassifier

from config import (
    RESULTS_DIR,
    SAVED_MODELS_DIR,
)

from data_utils import (
    load_dataset,
    prepare_data,
    split_dataset,
)


# ==========================================================
# Paths
# ==========================================================

MODEL_FILE = (
    SAVED_MODELS_DIR /
    "tuned" /
    "xgboost_tuned.json"
)

EXPLAINABILITY_DIR = (
    RESULTS_DIR /
    "explainability"
)

EXPLAINABILITY_DIR.mkdir(
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

X, y = prepare_data(
    dataset
)

X_train, X_test, y_train, y_test = (
    split_dataset(
        X,
        y,
    )
)


# ==========================================================
# Load Tuned XGBoost
# ==========================================================

print("\n" + "=" * 60)
print("Loading Tuned XGBoost")
print("=" * 60)

model = XGBClassifier()

model.load_model(
    MODEL_FILE
)

print(
    f"Model loaded from:\n"
    f"{MODEL_FILE}"
)


# ==========================================================
# Feature Names
# ==========================================================

feature_names = list(
    X_train.columns
)


# ==========================================================
# Built-in XGBoost Feature Importance
# ==========================================================

print("\n" + "=" * 60)
print("XGBoost Feature Importance")
print("=" * 60)

importance = model.feature_importances_

importance_df = pd.DataFrame({

    "feature":
        feature_names,

    "importance":
        importance,

})

importance_df = (
    importance_df
    .sort_values(
        by="importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


print(
    importance_df.to_string(
        index=False
    )
)


# ==========================================================
# Save Feature Importance
# ==========================================================

importance_file = (
    EXPLAINABILITY_DIR /
    "xgboost_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False,
)

print(
    f"\nFeature importance saved to:\n"
    f"{importance_file}"
)


# ==========================================================
# Feature Importance Plot
# ==========================================================

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    importance_df["feature"],
    importance_df["importance"],
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Tuned XGBoost Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

importance_plot = (
    EXPLAINABILITY_DIR /
    "xgboost_feature_importance.png"
)

plt.savefig(
    importance_plot,
    dpi=300,
)

plt.close()

print(
    f"Feature importance plot saved to:\n"
    f"{importance_plot}"
)


# ==========================================================
# SHAP Explainer
# ==========================================================

print("\n" + "=" * 60)
print("Calculating SHAP Values")
print("=" * 60)

explainer = shap.TreeExplainer(
    model
)


# ==========================================================
# Use Test Set for Explanation
# ==========================================================

# The test set is used here for explanation only.
# It is NOT used to train or tune the model.

shap_values = explainer.shap_values(
    X_test
)


# ==========================================================
# SHAP Summary Plot
# ==========================================================

print("\nCreating SHAP Summary Plot...")

plt.figure(
    figsize=(10, 7)
)

shap.summary_plot(
    shap_values,
    X_test,
    show=False,
)

plt.tight_layout()

shap_summary_file = (
    EXPLAINABILITY_DIR /
    "shap_summary.png"
)

plt.savefig(
    shap_summary_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"SHAP summary saved to:\n"
    f"{shap_summary_file}"
)


# ==========================================================
# SHAP Bar Plot
# ==========================================================

print("\nCreating SHAP Importance Bar Plot...")

plt.figure(
    figsize=(10, 7)
)

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False,
)

plt.tight_layout()

shap_bar_file = (
    EXPLAINABILITY_DIR /
    "shap_feature_importance.png"
)

plt.savefig(
    shap_bar_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"SHAP importance saved to:\n"
    f"{shap_bar_file}"
)
# ==========================================================
# SHAP Dependence Plots
# ==========================================================

print("\n" + "=" * 60)
print("Creating SHAP Dependence Plots")
print("=" * 60)

TOP_FEATURES = [
    "HbA1c_level",
    "blood_glucose_level",
    "age",
    "bmi",
]


for feature in TOP_FEATURES:

    print(
        f"\nCreating dependence plot for: "
        f"{feature}"
    )

    plt.figure(
        figsize=(9, 6)
    )

    shap.dependence_plot(
        feature,
        shap_values,
        X_test,
        show=False,
    )

    plt.tight_layout()

    safe_name = (
        feature
        .lower()
        .replace(" ", "_")
    )

    output_file = (
        EXPLAINABILITY_DIR /
        f"shap_dependence_{safe_name}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved to:\n"
        f"{output_file}"
    )

# ==========================================================
# Mean Absolute SHAP Importance
# ==========================================================

mean_abs_shap = np.abs(
    shap_values
).mean(axis=0)

shap_importance_df = pd.DataFrame({

    "feature":
        feature_names,

    "mean_abs_shap":
        mean_abs_shap,

})

shap_importance_df = (
    shap_importance_df
    .sort_values(
        by="mean_abs_shap",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ==========================================================
# Save SHAP Importance
# ==========================================================

shap_importance_file = (
    EXPLAINABILITY_DIR /
    "shap_feature_importance.csv"
)

shap_importance_df.to_csv(
    shap_importance_file,
    index=False,
)

print(
    f"\nSHAP feature importance saved to:\n"
    f"{shap_importance_file}"
)


# ==========================================================
# Display Top Features
# ==========================================================

print("\n" + "=" * 60)
print("TOP SHAP FEATURES")
print("=" * 60)

print(
    shap_importance_df.head(10).to_string(
        index=False
    )
)


# ==========================================================
# Completed
# ==========================================================

print("\n" + "=" * 60)
print("EXPLAINABILITY ANALYSIS COMPLETED")
print("=" * 60)