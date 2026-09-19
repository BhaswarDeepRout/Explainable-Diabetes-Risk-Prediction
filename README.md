# Diabetes Risk Stratification Research

An explainable machine-learning analysis for early diabetes risk prediction using hybrid deep learning, tree-based ensembles, and SHAP explainability on a 100,000-observation clinical dataset.

This repository contains the full end-to-end code used to generate the manuscript:
**"Hybrid Deep Learning and Ensemble Learning with Explainable AI for Diabetes Risk Stratification"**

## Setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the Pipeline

The methodology is modularized to prevent data leakage and provide clear boundaries between preprocessing, standalone models, tuning, and Out-Of-Fold (OOF) Stacking.

Execute the pipeline in the following order:

```powershell
# 1. Preprocessing & Feature Engineering
py -3 src\preprocessing.py
py -3 src\feature_engineering.py

# 2. Base Model Training & Evaluation
py -3 src\train.py

# 3. Final Comparison & Stacking
py -3 src\stacking.py
py -3 src\stacking_threshold.py
py -3 src\final_comparison.py

# 4. Model Interpretation
py -3 src\explainability.py
py -3 src\performance_analysis.py
```

## Project Layout

- `data/raw`: Includes `s1.csv` (~100,000 observations).
- `data/processed`: Encoded, imputed datasets safe from test-set leakage.
- `src`: Core machine learning codebase (hyperparameter tuning, ensembling, eval).
- `models`: Saved Optuna models and `joblib` binaries.
- `results`: Metrics, comparison outputs, and SHAP visualizations.
- `paper`: Manuscript drafts, LaTeX source, and references.
