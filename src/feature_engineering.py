"""Feature-engineering utilities for the diabetes prediction pipeline."""
"""
feature_engineering.py
----------------------
Feature Engineering Pipeline

This module performs:

1. Dataset loading
2. Dataset statistics
3. Correlation analysis
4. Feature engineering reports

NOTE:
Feature selection is NOT performed in this part.
"""

from pathlib import Path

from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    CLEAN_DATA_FILE,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    TARGET_COLUMN,
)

# ==========================================================
# Results Directory
# ==========================================================

FEATURE_RESULTS_DIR = RESULTS_DIR / "feature_engineering"
FEATURE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset(filepath: Path) -> pd.DataFrame:
    """
    Load the cleaned dataset.
    """

    print("=" * 60)
    print("Loading Engineered Dataset")
    print("=" * 60)

    df = pd.read_csv(filepath)

    print(f"Dataset Loaded Successfully")
    print(f"Shape : {df.shape}")

    return df


# ==========================================================
# Dataset Statistics
# ==========================================================

def generate_statistics(df: pd.DataFrame):
    """
    Generate descriptive statistics.
    """

    print("\nGenerating Dataset Statistics...")

    stats = df.describe(include="all").transpose()

    output_file = FEATURE_RESULTS_DIR / "dataset_statistics.csv"

    stats.to_csv(output_file)

    print(f"Statistics saved to:\n{output_file}")

    return stats


# ==========================================================
# Correlation Analysis
# ==========================================================

def correlation_analysis(df: pd.DataFrame):
    """
    Generate feature correlation matrix.
    """

    print("\nGenerating Correlation Matrix...")

    correlation_matrix = df.corr(numeric_only=True)

    correlation_csv = (
        FEATURE_RESULTS_DIR /
        "feature_correlation.csv"
    )

    correlation_matrix.to_csv(correlation_csv)

    print(f"Correlation matrix saved to:\n{correlation_csv}")

    return correlation_matrix


# ==========================================================
# Correlation Heatmap
# ==========================================================

def plot_correlation_heatmap(correlation_matrix):
    """
    Save correlation heatmap.
    """

    print("\nGenerating Correlation Heatmap...")

    plt.figure(figsize=(12, 10))

    plt.imshow(
        correlation_matrix,
        interpolation="nearest",
        aspect="auto",
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=90,
        fontsize=8,
    )

    plt.yticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        fontsize=8,
    )

    plt.title("Feature Correlation Matrix")

    plt.tight_layout()

    output_file = (
        FEATURE_RESULTS_DIR /
        "correlation_heatmap.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Heatmap saved to:\n{output_file}")

# ==========================================================
# Variance Threshold
# ==========================================================

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE

def get_train_split(df):
    """Helper to isolate training split to avoid target leakage during supervised analysis."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return X_train, y_train


def variance_threshold_analysis(df):
    """
    Identify low-variance features using training data only.

    NOTE:
    Features are NOT removed automatically.
    """

    print("\nRunning Variance Threshold Analysis...")

    X_train, _ = get_train_split(df)

    selector = VarianceThreshold(threshold=0.01)

    selector.fit(X_train)

    variance = pd.DataFrame({
        "Feature": X_train.columns,
        "Variance": X_train.var().values,
        "Selected": selector.get_support()
    })

    output_file = (
        FEATURE_RESULTS_DIR /
        "variance_threshold.csv"
    )

    variance.to_csv(output_file, index=False)

    print(f"Variance report saved to:\n{output_file}")

    return variance


# ==========================================================
# Mutual Information
# ==========================================================

def mutual_information_analysis(df):
    """
    Compute Mutual Information scores using training data only.
    """

    print("\nCalculating Mutual Information...")

    X_train, y_train = get_train_split(df)

    mi_scores = mutual_info_classif(
        X_train,
        y_train,
        random_state=RANDOM_STATE
    )

    mi = pd.DataFrame({
        "Feature": X_train.columns,
        "Mutual Information": mi_scores
    })

    mi = mi.sort_values(
        by="Mutual Information",
        ascending=False
    )

    output_file = (
        FEATURE_RESULTS_DIR /
        "mutual_information.csv"
    )

    mi.to_csv(output_file, index=False)

    print(f"Mutual Information saved to:\n{output_file}")

    return mi


# ==========================================================
# Mutual Information Plot
# ==========================================================

def plot_mutual_information(mi):
    """
    Save Mutual Information plot.
    """

    print("\nGenerating Mutual Information Plot...")

    plt.figure(figsize=(10, 6))

    plt.barh(
        mi["Feature"],
        mi["Mutual Information"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Mutual Information")

    plt.title("Feature Importance using Mutual Information")

    plt.tight_layout()

    output_file = (
        FEATURE_RESULTS_DIR /
        "mutual_information.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Plot saved to:\n{output_file}")


# ==========================================================
# Random Forest Feature Importance
# ==========================================================

def random_forest_importance(df):
    """
    Compute Random Forest feature importance using training data only.
    """

    print("\nTraining Random Forest for Feature Importance...")

    X_train, y_train = get_train_split(df)

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": rf.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    output_file = (
        FEATURE_RESULTS_DIR /
        "random_forest_importance.csv"
    )

    importance.to_csv(
        output_file,
        index=False
    )

    print(f"Feature importance saved to:\n{output_file}")

    return importance


# ==========================================================
# Random Forest Importance Plot
# ==========================================================

def plot_random_forest_importance(importance):
    """
    Save Random Forest Feature Importance plot.
    """

    print("\nGenerating Random Forest Importance Plot...")

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Importance")

    plt.title("Random Forest Feature Importance")

    plt.tight_layout()

    output_file = (
        FEATURE_RESULTS_DIR /
        "random_forest_importance.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Plot saved to:\n{output_file}")
# ==========================================================
# Variance Inflation Factor (VIF)
# ==========================================================




def calculate_vif(df):
    """
    Calculate Variance Inflation Factor (VIF).

    Binary one-hot encoded columns are excluded because
    VIF is intended for continuous variables.
    """

    print("\nCalculating VIF...")

    continuous_features = [
        "age",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
    ]

    X = df[continuous_features]

    vif = pd.DataFrame()

    vif["Feature"] = X.columns

    vif["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    output_file = (
        FEATURE_RESULTS_DIR /
        "vif_scores.csv"
    )

    vif.to_csv(output_file, index=False)

    print(f"VIF scores saved to:\n{output_file}")

    return vif


# ==========================================================
# Feature Summary
# ==========================================================

def generate_feature_summary(
    variance,
    mi,
    importance,
    vif
):
    """
    Combine all feature analysis results into one table.
    """

    print("\nGenerating Feature Summary...")

    summary = variance.merge(
        mi,
        on="Feature",
        how="left",
    )

    summary = summary.merge(
        importance,
        on="Feature",
        how="left",
    )

    summary = summary.merge(
        vif,
        on="Feature",
        how="left",
    )

    summary["Recommendation"] = "KEEP"

    output_file = (
        FEATURE_RESULTS_DIR /
        "feature_summary.csv"
    )

    summary.to_csv(
        output_file,
        index=False
    )

    print(f"Summary saved to:\n{output_file}")

    return summary


# ==========================================================
# Feature Engineering Report
# ==========================================================

def generate_report(summary):
    """
    Generate text report.
    """

    report_file = (
        FEATURE_RESULTS_DIR /
        "feature_engineering_report.txt"
    )

    with open(report_file, "w") as f:

        f.write("=" * 60 + "\n")
        f.write("FEATURE ENGINEERING REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(
            f"Total Features : {len(summary)}\n\n"
        )

        f.write("Feature Summary\n")
        f.write("-" * 60 + "\n")

        f.write(
            summary.to_string(index=False)
        )

    print(f"Report saved to:\n{report_file}")


# ==========================================================
# Save Engineered Dataset
# ==========================================================

def save_engineered_dataset(df):
    """
    Save engineered dataset.

    No features are removed automatically.
    """

    output_file = (
        PROCESSED_DATA_DIR /
        "diabetes_engineered.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(f"\nEngineered dataset saved to:\n{output_file}")


# ==========================================================
# Main
# ==========================================================

def main():

    dataset = load_dataset(CLEAN_DATA_FILE)

    generate_statistics(dataset)

    correlation_matrix = correlation_analysis(dataset)

    plot_correlation_heatmap(
        correlation_matrix
    )

    variance = variance_threshold_analysis(
        dataset
    )

    mi = mutual_information_analysis(
        dataset
    )

    plot_mutual_information(mi)

    importance = random_forest_importance(
        dataset
    )

    plot_random_forest_importance(
        importance
    )

    vif = calculate_vif(dataset)

    summary = generate_feature_summary(
        variance,
        mi,
        importance,
        vif,
    )

    generate_report(summary)

    save_engineered_dataset(dataset)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()