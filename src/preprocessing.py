"""
preprocessing.py
----------------
Preprocess the diabetes dataset.

Pipeline:
1. Load raw dataset
2. Inspect dataset
3. Remove duplicates
4. Handle missing values
5. Validate data types
6. One-Hot Encode categorical variables
7. Data quality audit
8. Save cleaned dataset
"""

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE = RAW_DATA_DIR / "s1.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "diabetes_clean.csv"
TARGET_COLUMN = "diabetes"


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset(filepath: Path) -> pd.DataFrame:
    """Load dataset from CSV."""

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    df = pd.read_csv(filepath)

    print(f"Dataset Loaded Successfully")
    print(f"Shape : {df.shape}")

    return df


# ==========================================================
# Inspect Dataset
# ==========================================================

def inspect_dataset(df: pd.DataFrame) -> None:
    """Display dataset information."""

    print("\n" + "=" * 60)
    print("Dataset Inspection")
    print("=" * 60)

    print("\nShape")
    print(df.shape)

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nTarget Distribution")
    print(df[TARGET_COLUMN].value_counts())


# ==========================================================
# Remove Duplicates
# ==========================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""

    duplicates = df.duplicated().sum()

    print("\nDuplicate Rows :", duplicates)

    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows.")
    else:
        print("No duplicate rows found.")

    return df

# ==========================================================
# Remove Conflicting Feature/Target Duplicates
# ==========================================================

def remove_conflicting_duplicates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove observations where the exact same feature values
    occur with different target labels.

    Both observations are removed so that identical feature
    patterns cannot appear across train and test sets.
    """

    print(
        "\nChecking for conflicting "
        "feature/target duplicates..."
    )

    feature_columns = [
        column
        for column in df.columns
        if column != TARGET_COLUMN
    ]

    conflicting_rows = (
        df.groupby(
            feature_columns,
            dropna=False,
            group_keys=False,
        )
        .filter(
            lambda group:
            group[TARGET_COLUMN].nunique() > 1
        )
    )

    if conflicting_rows.empty:

        print(
            "No conflicting feature/target "
            "duplicates found."
        )

        return df

    conflicting_groups = (
        conflicting_rows[
            feature_columns
        ]
        .drop_duplicates()
        .shape[0]
    )

    observations_removed = len(
        conflicting_rows
    )

    print(
        f"Conflicting feature groups : "
        f"{conflicting_groups}"
    )

    print(
        f"Observations removed        : "
        f"{observations_removed}"
    )

    df = df.drop(
        index=conflicting_rows.index
    ).reset_index(
        drop=True
    )

    print(
        "Conflicting observations removed."
    )

    return df


# ==========================================================
# Handle Missing Values
# ==========================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values."""

    print("\nHandling Missing Values...")

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    if len(numerical_columns) > 0:
        numeric_imputer = SimpleImputer(strategy="median")

        df[numerical_columns] = numeric_imputer.fit_transform(
            df[numerical_columns]
        )

    if len(categorical_columns) > 0:
        categorical_imputer = SimpleImputer(
            strategy="most_frequent"
        )

        df[categorical_columns] = categorical_imputer.fit_transform(
            df[categorical_columns]
        )

    print("Missing values handled.")

    return df


# ==========================================================
# Validate Data Types
# ==========================================================

def validate_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct data types."""

    print("\nValidating Data Types...")

    df["age"] = df["age"].astype(float)
    df["bmi"] = df["bmi"].astype(float)
    df["HbA1c_level"] = df["HbA1c_level"].astype(float)
    df["blood_glucose_level"] = df["blood_glucose_level"].astype(float)

    df["hypertension"] = df["hypertension"].astype(int)
    df["heart_disease"] = df["heart_disease"].astype(int)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df


# ==========================================================
# Encode Categorical Variables
# ==========================================================

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """One-Hot Encode categorical variables."""

    print("\nEncoding categorical variables...")

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if len(categorical_columns) == 0:
        print("No categorical variables found.")
        return df

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=False,
        dtype=int,
    )

    print("Encoding completed.")

    return df


# ==========================================================
# Data Quality Audit
# ==========================================================

def audit_dataset(df: pd.DataFrame) -> None:
    """Run quality checks."""

    print("\n" + "=" * 60)
    print("DATA QUALITY AUDIT")
    print("=" * 60)

    print(f"Shape                 : {df.shape}")
    print(f"Missing Values        : {df.isnull().sum().sum()}")
    print(f"Duplicate Rows        : {df.duplicated().sum()}")

    object_columns = len(
        df.select_dtypes(include=["object"]).columns
    )

    print(f"Object Columns        : {object_columns}")

    infinite_values = (
        df.replace([float("inf"), float("-inf")], pd.NA)
        .isna()
        .sum()
        .sum()
    )

    print(f"Infinite Values       : {infinite_values}")

    print(
        f"Target Present        : {TARGET_COLUMN in df.columns}"
    )

    print("=" * 60)


# ==========================================================
# Save Dataset
# ==========================================================

def save_dataset(df: pd.DataFrame, filepath: Path) -> None:
    """Save processed dataset."""

    df.to_csv(filepath, index=False)

    print(f"\nDataset saved to:\n{filepath}")


# ==========================================================
# Main
# ==========================================================

def main():

    dataset = load_dataset(INPUT_FILE)

    inspect_dataset(dataset)

    dataset = remove_duplicates(dataset)

    dataset = remove_conflicting_duplicates(
        dataset
    )

    dataset = handle_missing_values(dataset)

    dataset = validate_dtypes(dataset)

    dataset = encode_categorical(dataset)

    audit_dataset(dataset)

    save_dataset(dataset, OUTPUT_FILE)

    print("\nPreprocessing Completed Successfully.")


if __name__ == "__main__":
    main()