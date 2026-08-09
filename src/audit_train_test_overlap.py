import pandas as pd

from data_utils import (
    load_dataset,
    prepare_data,
    split_dataset,
)


print("=" * 60)
print("CONFLICTING FEATURE / TARGET AUDIT")
print("=" * 60)

dataset = load_dataset()

X, y = prepare_data(dataset)

X_train, X_test, y_train, y_test = split_dataset(
    X,
    y,
)

# ----------------------------------------------------------
# Create feature keys
# ----------------------------------------------------------

train_keys = X_train.apply(
    lambda row: tuple(row),
    axis=1,
)

test_keys = X_test.apply(
    lambda row: tuple(row),
    axis=1,
)

overlap_keys = set(train_keys).intersection(
    set(test_keys)
)

print(
    f"\nConflicting feature patterns: "
    f"{len(overlap_keys)}"
)


# ----------------------------------------------------------
# Find the actual records
# ----------------------------------------------------------

for key in overlap_keys:

    train_indices = train_keys[
        train_keys == key
    ].index

    test_indices = test_keys[
        test_keys == key
    ].index

    for train_index in train_indices:

        for test_index in test_indices:

            print("\n" + "-" * 60)

            print(
                f"TRAIN index: {train_index}"
            )

            print(
                f"TEST index : {test_index}"
            )

            print(
                f"TRAIN target: "
                f"{y_train.loc[train_index]}"
            )

            print(
                f"TEST target : "
                f"{y_test.loc[test_index]}"
            )

            print("\nFeature values:")

            print(
                X_train.loc[
                    train_index
                ].to_string()
            )

print("\n" + "=" * 60)
print("CONFLICT AUDIT COMPLETED")
print("=" * 60)