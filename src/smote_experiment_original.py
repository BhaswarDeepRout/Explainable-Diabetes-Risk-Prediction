"""
SMOTE experiments for diabetes prediction.
"""

from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier

from config import RANDOM_STATE
from config import RESULTS_DIR

SMOTE_RESULTS_FILE = (
    RESULTS_DIR /
    "comparison" /
    "smote_results.csv"
)

from data_utils import (
    load_dataset,
    prepare_data,
    split_dataset,
)
from catboost import CatBoostClassifier

from utils import (
    evaluate_classifier,
    print_metrics,
    save_model,
    save_results,
)

from config import (
    RANDOM_STATE,
    TUNED_MODELS_DIR,
)


# ==========================================================
# Load Dataset
# ==========================================================

dataset = load_dataset()

X, y = prepare_data(dataset)

X_train, X_test, y_train, y_test = split_dataset(
    X,
    y,
)


# ==========================================================
# Check Original Class Distribution
# ==========================================================

print("\n" + "=" * 60)
print("Original Training Class Distribution")
print("=" * 60)

print(y_train.value_counts())


# ==========================================================
# Apply SMOTE ONLY to Training Data
# ==========================================================

smote = SMOTE(
    random_state=RANDOM_STATE,
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train,
)


# ==========================================================
# Check SMOTE Class Distribution
# ==========================================================

print("\n" + "=" * 60)
print("SMOTE Training Class Distribution")
print("=" * 60)

print(y_train_smote.value_counts())


print("\nOriginal Training Shape:")
print(X_train.shape)

print("\nSMOTE Training Shape:")
print(X_train_smote.shape)

print("\nTest Shape:")
print(X_test.shape)
# ==========================================================
# Train CatBoost with SMOTE
# ==========================================================

print("\n" + "=" * 60)
print("Training CatBoost with SMOTE")
print("=" * 60)

model = CatBoostClassifier(
    iterations=689,
    depth=6,
    learning_rate=0.018099203881912614,
    l2_leaf_reg=11.93369272947072,
    bagging_temperature=7.36226537450141,
    random_strength=1.3039002180458603,
    border_count=114,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=False,
)

model.fit(
    X_train_smote,
    y_train_smote,
)
# ==========================================================
# Evaluate SMOTE CatBoost
# ==========================================================

y_pred, y_prob, metrics = evaluate_classifier(
    model,
    X_test,
    y_test,
)

print_metrics(
    "CatBoost + SMOTE",
    metrics,
)
save_results(
    "CatBoost + SMOTE",
    metrics,
    SMOTE_RESULTS_FILE,
)

# ==========================================================
# Save SMOTE CatBoost
# ==========================================================

save_model(
    model,
    TUNED_MODELS_DIR / "catboost_smote.cbm",
)

# ==========================================================
# Train XGBoost with SMOTE
# ==========================================================

from xgboost import XGBClassifier

print("\n" + "=" * 60)
print("Training XGBoost with SMOTE")
print("=" * 60)

xgb_model = XGBClassifier(
    n_estimators=694,
    max_depth=6,
    learning_rate=0.03130937895909436,
    subsample=0.6982603672990167,
    colsample_bytree=0.7985651082248088,
    gamma=1.661655371082837,
    min_child_weight=4,
    reg_alpha=3.057235139701773,
    reg_lambda=3.015763533142166,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=RANDOM_STATE,
    tree_method="hist",
    verbosity=0,
    n_jobs=-1,
)

xgb_model.fit(
    X_train_smote,
    y_train_smote,
)

# ==========================================================
# Evaluate SMOTE XGBoost
# ==========================================================

y_pred, y_prob, metrics = evaluate_classifier(
    xgb_model,
    X_test,
    y_test,
)

print_metrics(
    "XGBoost + SMOTE",
    metrics,
)

# ==========================================================
# Save SMOTE XGBoost
# ==========================================================

save_model(
    xgb_model,
    TUNED_MODELS_DIR / "xgboost_smote.json",
)
save_results(
    "XGBoost + SMOTE",
    metrics,
    SMOTE_RESULTS_FILE,
)

# ==========================================================
# Scale Data for MLP
# ==========================================================

from data_utils import scale_features

X_train_smote_scaled, X_test_scaled, scaler = scale_features(
    X_train_smote,
    X_test,
    save_scaler=False,
)


# ==========================================================
# Train MLP with SMOTE
# ==========================================================

print("\n" + "=" * 60)
print("Training MLP with SMOTE")
print("=" * 60)

mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    alpha=0.0001,
    learning_rate_init=0.001,
    batch_size=128,
    max_iter=500,
    early_stopping=True,
    random_state=RANDOM_STATE,
)

mlp_model.fit(
    X_train_smote_scaled,
    y_train_smote,
)


# ==========================================================
# Evaluate SMOTE MLP
# ==========================================================

y_pred, y_prob, metrics = evaluate_classifier(
    mlp_model,
    X_test_scaled,
    y_test,
)

print_metrics(
    "MLP + SMOTE",
    metrics,
)
save_results(
    "MLP + SMOTE",
    metrics,
    SMOTE_RESULTS_FILE,
)


# ==========================================================
# Save SMOTE MLP
# ==========================================================

save_model(
    mlp_model,
    TUNED_MODELS_DIR / "mlp_smote.joblib",
)