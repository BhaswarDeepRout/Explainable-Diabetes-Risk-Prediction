"""
preprocessing.py
----------------
Preprocess the diabetes dataset.

Pipeline:
1. Load raw dataset
2. Inspect dataset
3. Remove duplicates
4. Handle missing values & One-Hot Encode (fitted on train split only)
5. Validate data types
6. Data quality audit
7. Save cleaned dataset
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Import project settings for train-test split consistency
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE


# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE = RAW_DATA_DIR / "s1.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "diabetes_clean.csv"


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
# Handle Missing Values & Encode Categorical Variables
# ==========================================================

def process_features_leakage_free(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values and encode categorical variables,
    fitting transformers strictly on the training partition
    to prevent procedural data leakage.
    """
    print("\nProcessing Features (Leakage-Free)...")

    # 1. Determine train indices matching the exact split in data_utils.py
    train_idx, _ = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN]
    )

    numerical_columns = df.drop(columns=[TARGET_COLUMN]).select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = df.drop(columns=[TARGET_COLUMN]).select_dtypes(include=["object", "category"]).columns.tolist()

    # Define pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median"))
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('encoder', OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"))
    ])

    # Assemble ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_columns),
            ('cat', cat_pipeline, categorical_columns)
        ],
        remainder='passthrough'
    )

    # 2. Fit ONLY on the training partition
    preprocessor.fit(df.loc[train_idx])

    # 3. Transform the entire dataset
    transformed = preprocessor.transform(df)

    # 4. Extract feature names to match get_dummies exactly
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_columns)

    # Remainder is the TARGET_COLUMN
    feature_names = numerical_columns + list(cat_feature_names) + [TARGET_COLUMN]

    df_transformed = pd.DataFrame(transformed, columns=feature_names, index=df.index)

    # 5. Restore original column order (mimic get_dummies)
    # get_dummies keeps all non-categorical cols first (in their original relative order),
    # then appends categorical ones at the end.
    original_non_cat = [c for c in df.columns if c not in categorical_columns]
    desired_order = original_non_cat + list(cat_feature_names)

    # Reorder columns to exactly match how get_dummies did it
    df_transformed = df_transformed[desired_order]

    print("Missing values imputed and categorical variables encoded.")
    return df_transformed


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

    # Cast encoded columns to int
    cat_cols = [c for c in df.columns if c.startswith('gender_') or c.startswith('smoking_history_')]
    for col in cat_cols:
        df[col] = df[col].astype(int)

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

    dataset = process_features_leakage_free(dataset)

    dataset = validate_dtypes(dataset)

    audit_dataset(dataset)

    save_dataset(dataset, OUTPUT_FILE)

    print("\nPreprocessing Completed Successfully.")


if __name__ == "__main__":
    main()