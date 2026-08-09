from pathlib import Path

import pandas as pd

from data_utils import (
    load_dataset,
    prepare_data,
    split_dataset,
)


print("=" * 70)
print("FINAL DATA INTEGRITY AUDIT")
print("=" * 70)


# ==========================================================
# Load Dataset
# ==========================================================

dataset = load_dataset()

print("\nDataset shape:")
print(dataset.shape)


# ==========================================================
# Basic Quality Checks
# ==========================================================

print("\n" + "=" * 70)
print("DATA QUALITY")
print("=" * 70)

missing = dataset.isna().sum().sum()

duplicates = dataset.duplicated().sum()

infinite = (
    dataset
    .replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )
    .isna()
    .sum()
    .sum()
)

print(f"Missing values       : {missing}")
print(f"Duplicate rows       : {duplicates}")
print(f"Infinite values      : {infinite}")


# ==========================================================
# Target Distribution
# ==========================================================

X, y = prepare_data(dataset)

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(
    y.value_counts()
    .sort_index()
    .to_string()
)

print(
    f"\nPositive class rate : "
    f"{y.mean():.4f}"
)


# ==========================================================
# Train/Test Split
# ==========================================================

X_train, X_test, y_train, y_test = (
    split_dataset(
        X,
        y,
    )
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    f"Training samples : {len(X_train)}"
)

print(
    f"Testing samples  : {len(X_test)}"
)


# ==========================================================
# Feature Overlap
# ==========================================================

train_keys = X_train.apply(
    lambda row: tuple(row),
    axis=1,
)

test_keys = X_test.apply(
    lambda row: tuple(row),
    axis=1,
)

overlap = set(train_keys).intersection(
    set(test_keys)
)


print("\n" + "=" * 70)
print("TRAIN / TEST FEATURE OVERLAP")
print("=" * 70)

print(
    f"Identical feature patterns "
    f"in both sets : {len(overlap)}"
)


# ==========================================================
# Assertions
# ==========================================================

assert missing == 0, (
    "Missing values detected."
)

assert duplicates == 0, (
    "Duplicate rows detected."
)

assert infinite == 0, (
    "Infinite values detected."
)

assert len(overlap) == 0, (
    "Train/test feature overlap detected."
)

assert len(X_train) == len(y_train)

assert len(X_test) == len(y_test)


# ==========================================================
# Final Status
# ==========================================================

print("\n" + "=" * 70)
print("FINAL DATA INTEGRITY STATUS")
print("=" * 70)

print("PASS")

print("\nAll critical data-integrity checks passed.")

print("=" * 70)