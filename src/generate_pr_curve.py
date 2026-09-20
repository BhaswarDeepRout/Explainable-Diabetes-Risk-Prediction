import sys
sys.path.append("src")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import joblib
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from data_utils import load_dataset, prepare_data, split_dataset

# Load data identically to the pipeline
df = load_dataset()
X, y = prepare_data(df)
X_train, X_test, y_train, y_test = split_dataset(X, y)

plt.figure(figsize=(9, 7))

# 1. XGBoost Tuned
xgb = XGBClassifier()
xgb.load_model("models/saved_models/tuned/xgboost_tuned.json")
xgb_prob = xgb.predict_proba(X_test)[:, 1]
p, r, _ = precision_recall_curve(y_test, xgb_prob)
plt.plot(r, p, label=f"Tuned XGBoost (PR-AUC = {auc(r, p):.4f})", linewidth=3, color='#1f77b4')

# 2. CatBoost Tuned
cat = CatBoostClassifier()
cat.load_model("models/saved_models/tuned/catboost_tuned.cbm")
cat_prob = cat.predict_proba(X_test)[:, 1]
p, r, _ = precision_recall_curve(y_test, cat_prob)
plt.plot(r, p, label=f"Tuned CatBoost (PR-AUC = {auc(r, p):.4f})", linewidth=2.5, color='#ff7f0e', linestyle='--')

# 3. MLP Tuned
mlp = joblib.load("models/saved_models/tuned/mlp_tuned.joblib")
scaler = joblib.load("models/saved_models/scaler.joblib")
X_test_scaled_for_mlp = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
mlp_prob = mlp.predict_proba(X_test_scaled_for_mlp.values)[:, 1]
p, r, _ = precision_recall_curve(y_test, mlp_prob)
plt.plot(r, p, label=f"Tuned MLP (PR-AUC = {auc(r, p):.4f})", linewidth=2, color='#2ca02c')

# 4. Stacking Model
meta = joblib.load("models/saved_models/tuned/stacking_meta_model.joblib")
try:
    X_test_meta = np.column_stack((xgb_prob, cat_prob, mlp_prob))
    stack_prob = meta.predict_proba(X_test_meta)[:, 1]
    p, r, _ = precision_recall_curve(y_test, stack_prob)
    plt.plot(r, p, label=f"Stacking Ensemble (PR-AUC = {auc(r, p):.4f})", linewidth=2.5, color='#d62728', linestyle='-.')
except Exception as e:
    pass

plt.title("Comparative Precision-Recall (PR) Curves (Zoomed)", fontsize=15, pad=15)
plt.xlabel("Recall (Sensitivity)", fontsize=12)
plt.ylabel("Precision (Positive Predictive Value)", fontsize=12)
plt.legend(loc="lower left", fontsize=11)
plt.grid(True, alpha=0.5, linestyle='--')

# Adjust scale to zoom into the region of difference
plt.xlim([0.55, 1.0])
plt.ylim([0.45, 1.0])

plt.tight_layout()
plt.savefig("paper/pr_curves_comparative.png", dpi=300, bbox_inches='tight')
print("Successfully regenerated zoomed paper/pr_curves_comparative.png")
