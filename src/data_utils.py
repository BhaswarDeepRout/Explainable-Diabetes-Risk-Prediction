"""
data_utils.py
-------------
Shared data preparation utilities.

Used by:
- train.py
- hyperparameter tuning
- SMOTE experiments
- stacking ensemble
- SHAP
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    ENGINEERED_DATA_FILE,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    SAVED_MODELS_DIR,
    RESULTS_DIR,
)

# ==========================================================
# Directories
# ==========================================================

TRAINING_RESULTS_DIR = RESULTS_DIR / "training"
TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Continuous Features
# ==========================================================

CONTINUOUS_FEATURES = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    print("=" * 60)
    print("Loading Engineered Dataset")
    print("=" * 60)

    df = pd.read_csv(ENGINEERED_DATA_FILE)

    print(f"Dataset Shape : {df.shape}")

    return df


# ==========================================================
# Prepare Features
# ==========================================================

def prepare_data(df):

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    return X, y


# ==========================================================
# Train/Test Split
# ==========================================================

def split_dataset(X, y):

    print("\nSplitting Dataset...")

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y,
    )

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    return (

        X_train,

        X_test,

        y_train,

        y_test,
    )


# ==========================================================
# Scale Continuous Features
# ==========================================================

def scale_features(

    X_train,

    X_test,

    save_scaler=True,
):

    print("\nScaling Continuous Features...")

    scaler = ColumnTransformer(

        transformers=[

            (

                "continuous",

                StandardScaler(),

                CONTINUOUS_FEATURES,

            )

        ],

        remainder="passthrough",
    )

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    if save_scaler:

        joblib.dump(

            scaler,

            SAVED_MODELS_DIR / "scaler.joblib",

        )

        print("Scaler saved.")

    return (

        X_train_scaled,

        X_test_scaled,

        scaler,
    )


# ==========================================================
# Save Train/Test Indices
# ==========================================================

def save_split_indices(

    X_train,

    X_test,
):

    train_file = (
        TRAINING_RESULTS_DIR /
        "train_indices.csv"
    )

    test_file = (
        TRAINING_RESULTS_DIR /
        "test_indices.csv"
    )

    pd.DataFrame(

        {"index": X_train.index}

    ).to_csv(

        train_file,

        index=False,
    )

    pd.DataFrame(

        {"index": X_test.index}

    ).to_csv(

        test_file,

        index=False,
    )

    print("Train/Test indices saved.")