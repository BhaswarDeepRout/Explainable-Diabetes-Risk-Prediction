"""Explainable ensemble framework for early diabetes-risk prediction.

The script uses the Pima Indians clinical dataset in ``data/raw`` and
compares interpretable baselines, tree ensembles, and a stacking ensemble. It
writes reproducible results into ``results``.

Install requirements first:
    pip install -r requirements.txt

Run:
    python src/main.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET = "Outcome"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODEL_OUTPUTS_DIR = RESULTS_DIR / "model_outputs"
REPORTS_DIR = RESULTS_DIR / "reports"

# A value of zero is physiologically implausible for these measurements.  It is
# treated as missing instead of as a genuine low clinical value.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load data and add clinically motivated, leakage-safe derived features."""
    frame = pd.read_csv(path)
    expected = {
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
        "BMI", "DiabetesPedigreeFunction", "Age", TARGET,
    }
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    for column in ZERO_AS_MISSING:
        frame[f"{column}_Missing"] = (frame[column] == 0).astype(int)
        frame[column] = frame[column].replace(0, np.nan)

    # These interactions retain clinically meaningful relationships without
    # introducing external variables. The imputer in each CV fold prevents leak.
    frame["Glucose_BMI"] = frame["Glucose"] * frame["BMI"]
    frame["Age_Glucose"] = frame["Age"] * frame["Glucose"]

    y = frame.pop(TARGET).astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only binary values 0 and 1.")
    return frame, y


def make_preprocessor(columns: list[str], scale: bool) -> ColumnTransformer:
    """Median imputation, optional standardisation, and transparent feature names."""
    # The pipeline deliberately mixes two transformer classes, so the explicit
    # annotation prevents static analysers from inferring only SimpleImputer.
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [("clinical", Pipeline(numeric_steps), columns)],
        verbose_feature_names_out=False,
    )


def build_models(columns: list[str]) -> dict[str, Any]:
    """Return the specified study models with conservative reproducible settings."""
    linear = Pipeline([
        ("preprocessor", make_preprocessor(columns, scale=True)),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    tree = Pipeline([
        ("preprocessor", make_preprocessor(columns, scale=False)),
        ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=12, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    forest = Pipeline([
        ("preprocessor", make_preprocessor(columns, scale=False)),
        ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=4, max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    xgb = Pipeline([
        ("preprocessor", make_preprocessor(columns, scale=False)),
        ("model", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    lgbm = Pipeline([
        ("preprocessor", make_preprocessor(columns, scale=False)),
        ("model", LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=4, subsample=0.8, colsample_bytree=0.8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
    ])
    stack = StackingClassifier(
        estimators=[("lr", linear), ("rf", forest), ("xgb", xgb), ("lgbm", lgbm)],
        final_estimator=LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        passthrough=False,
    )
    return {
        "Logistic Regression": linear,
        "Decision Tree": tree,
        "Random Forest": forest,
        "XGBoost": xgb,
        "LightGBM": lgbm,
        "Stacking Ensemble": stack,
    }


def evaluate_models(
    models: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate using five-fold stratified CV plus an untouched test set."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "average_precision": "average_precision", "f1": "f1", "accuracy": "accuracy"}
    rows, fitted = [], {}
    for name, model in models.items():
        cv = cross_validate(model, X_train, y_train, cv=folds, scoring=scoring, n_jobs=1)
        # sklearn's clone() type stub returns object even when cloning a
        # classifier. The models supplied by build_models all implement fit
        # and predict_proba, so retain that runtime interface at this boundary.
        estimator: Any = clone(model)
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.50).astype(int)
        rows.append({
            "Model": name,
            "CV ROC-AUC (mean)": cv["test_roc_auc"].mean(),
            "CV ROC-AUC (std)": cv["test_roc_auc"].std(),
            "Test ROC-AUC": roc_auc_score(y_test, probabilities),
            "Test PR-AUC": average_precision_score(y_test, probabilities),
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1": f1_score(y_test, predictions, zero_division=0),
            "Brier score": brier_score_loss(y_test, probabilities),
        })
        fitted[name] = estimator
    return pd.DataFrame(rows).sort_values("Test ROC-AUC", ascending=False), fitted


def save_diagnostics(fitted: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save ROC, confusion matrix, report, and SHAP global explanations."""
    plt.figure(figsize=(8, 6))
    for name, estimator in fitted.items():
        RocCurveDisplay.from_predictions(y_test, estimator.predict_proba(X_test)[:, 1], name=name, ax=plt.gca())
    plt.title("Held-out test ROC curves")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=180)
    plt.close()

    final_model = fitted["Stacking Ensemble"]
    probabilities = final_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    report = {
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "decision_threshold": 0.50,
    }
    (REPORTS_DIR / "stacking_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Kernel SHAP is model-agnostic, so it faithfully explains the final stack.
    # A small background/sample set keeps the explanation practical on a laptop.
    background = shap.sample(X_test, min(50, len(X_test)), random_state=RANDOM_STATE)
    explain_rows = X_test.iloc[: min(100, len(X_test))]
    def predict_diabetes_probability(values: np.ndarray) -> np.ndarray:
        """Adapt SHAP's array input to the DataFrame expected by the pipeline."""
        clinical_rows = pd.DataFrame(values, columns=X_test.columns)
        return final_model.predict_proba(clinical_rows)[:, 1]

    explainer = shap.KernelExplainer(predict_diabetes_probability, background)
    shap_values = explainer.shap_values(explain_rows, nsamples=150)
    plt.figure()
    shap.summary_plot(shap_values, explain_rows, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stacking_shap_global_importance.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    for directory in (FIGURES_DIR, METRICS_DIR, MODEL_OUTPUTS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    results, fitted = evaluate_models(build_models(list(X.columns)), X_train, y_train, X_test, y_test)
    results.to_csv(METRICS_DIR / "model_comparison.csv", index=False, float_format="%.4f")
    save_diagnostics(fitted, X_test, y_test)
    joblib.dump(fitted["Stacking Ensemble"], MODEL_OUTPUTS_DIR / "stacking_ensemble.joblib")
    print("\nModel comparison (held-out test set):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved results, visualizations, and final model to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
