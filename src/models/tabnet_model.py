"""
TabNet model for diabetes prediction.
"""

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier

from config import (
    RANDOM_STATE,
    SAVED_MODELS_DIR,
    RESULTS_DIR,
)

from data_utils import (
    load_dataset,
    prepare_data,
    split_dataset,
    scale_features,
)

from utils import (
    evaluate_classifier,
    print_metrics,
    save_results,
)


# ==========================================================
# Paths
# ==========================================================

TABNET_MODEL_FILE = (
    SAVED_MODELS_DIR /
    "tabnet_model"
)

TABNET_RESULTS_FILE = (
    RESULTS_DIR /
    "comparison" /
    "tabnet_results.csv"
)

TABNET_RESULTS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Dataset for TabNet")
print("=" * 60)

dataset = load_dataset()

X, y = prepare_data(dataset)

X_train, X_test, y_train, y_test = split_dataset(
    X,
    y,
)


# ==========================================================
# Scale Features
# ==========================================================

print("\nScaling Features...")

X_train_scaled, X_test_scaled, scaler = scale_features(
    X_train,
    X_test,
    save_scaler=False,
)


# ==========================================================
# Convert to NumPy
# ==========================================================

X_train_scaled = np.asarray(
    X_train_scaled,
    dtype=np.float32,
)

X_test_scaled = np.asarray(
    X_test_scaled,
    dtype=np.float32,
)

y_train_np = np.asarray(
    y_train,
    dtype=np.int64,
)

y_test_np = np.asarray(
    y_test,
    dtype=np.int64,
)


# ==========================================================
# Create Validation Set
# ==========================================================

print("\nCreating TabNet Validation Set...")

(
    X_train_tabnet,
    X_val_tabnet,
    y_train_tabnet,
    y_val_tabnet,
) = train_test_split(

    X_train_scaled,

    y_train_np,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y_train_np,
)


print(
    f"TabNet Training Samples   : "
    f"{len(X_train_tabnet)}"
)

print(
    f"TabNet Validation Samples : "
    f"{len(X_val_tabnet)}"
)

print(
    f"TabNet Test Samples       : "
    f"{len(X_test_scaled)}"
)


# ==========================================================
# Train TabNet
# ==========================================================

print("\n" + "=" * 60)
print("Training TabNet")
print("=" * 60)

model = TabNetClassifier(

    n_d=16,

    n_a=16,

    n_steps=5,

    gamma=1.5,

    lambda_sparse=1e-4,

    optimizer_fn=torch.optim.Adam,

    optimizer_params={
        "lr": 2e-3,
    },

    seed=RANDOM_STATE,

    verbose=10,
)


# ==========================================================
# Fit
# ==========================================================

model.fit(

    X_train=X_train_tabnet,

    y_train=y_train_tabnet,

    eval_set=[
        (
            X_val_tabnet,
            y_val_tabnet,
        ),
    ],

    eval_name=[
        "validation",
    ],

    eval_metric=[
        "auc",
    ],

    max_epochs=100,

    patience=15,

    batch_size=1024,

    virtual_batch_size=128,

    num_workers=0,

    drop_last=False,
)


# ==========================================================
# Final Evaluation
# ==========================================================

print("\n" + "=" * 60)
print("Evaluating TabNet on Untouched Test Set")
print("=" * 60)

y_pred, y_prob, metrics = evaluate_classifier(

    model,

    X_test_scaled,

    y_test_np,

)

print_metrics(
    "TabNet",
    metrics,
)


# ==========================================================
# Save Results
# ==========================================================

save_results(

    "TabNet",

    metrics,

    TABNET_RESULTS_FILE,

)


# ==========================================================
# Save Model
# ==========================================================

model.save_model(
    str(TABNET_MODEL_FILE)
)

print(
    f"\nTabNet model saved to:\n"
    f"{TABNET_MODEL_FILE}"
)


# ==========================================================
# Completed
# ==========================================================

print("\n" + "=" * 60)
print("TABNET TRAINING COMPLETED")
print("=" * 60)