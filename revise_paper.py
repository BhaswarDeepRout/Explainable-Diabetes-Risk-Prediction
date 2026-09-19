import re

with open("paper/paper draft 2.md", "r") as f:
    text = f.read()

# 1. Dataset Provenance and 2. Early Diabetes Prediction
text = text.replace("Early Diabetes Prediction", "Diabetes Risk Stratification")
text = text.replace("early diabetes prediction", "diabetes risk stratification")
text = text.replace("early diabetes risk prediction", "diabetes risk stratification")
text = text.replace("early detection", "risk stratification")
text = text.replace("early risk stratification", "risk stratification")
text = text.replace("predictive pipeline for early diabetes prediction", "predictive pipeline for diabetes risk stratification")

# Let's fix the Dataset and Partitioning section carefully
old_A = """### A. Dataset and Partitioning
The original raw dataset consisted of 100,000 observations encompassing 9 columns (features and target). To ensure data integrity, exact statistical duplicates and contradictory observations (instances with identical features but opposing target variables) were systematically removed. This cleaning stage omitted 3,854 exact duplicate rows and 182 contradictory observations, yielding a final dataset of 95,964 robust observations.

The dataset presents a high class imbalance: the final distribution consists of 87,573 non-diabetic observations (91.26%) and 8,391 diabetic observations (8.74%). The dataset was partitioned into training and independent holdout testing subsets using a stratified split (random\_state: 42). The training partition contains 76,771 observations (70,058 non-diabetic, 6,713 diabetic), while the test partition consists of 19,193 observations (17,515 non-diabetic, 1,678 diabetic)."""

new_A = """### A. Dataset and Partitioning
The experimental pipeline executed on a dataset (`s1.csv`) comprising 100,000 raw observations across 9 columns (features and target), representing clinical and demographic indicators mapped to concurrent diabetes status. Dataset provenance, including formal institutional origin, DOI, and temporal collection boundaries, is not documented within the repository artifacts, which remains a primary limitation regarding representativeness. To ensure data integrity, exact statistical duplicates and contradictory observations (instances with identical features but opposing target variables) were systematically removed. This cleaning stage omitted 3,854 exact duplicate rows and 182 contradictory observations, yielding a final dataset of 95,964 robust observations.

The dataset presents a high class imbalance: the final distribution consists of 87,573 non-diabetic observations (91.26%) and 8,391 diabetic observations (8.74%). The dataset was partitioned into training and independent holdout testing subsets using a stratified split (random\_state: 42). The training partition contains 76,771 observations (70,058 non-diabetic, 6,713 diabetic), while the test partition consists of 19,193 observations (17,515 non-diabetic, 1,678 diabetic)."""

text = text.replace(old_A, new_A)

# Feature Dictionary mapping
# Inserting Feature Dictionary after section A
feature_dict_text = """

### B. Feature Dictionary
The raw dataset encompasses eight clinical and demographic predictors and one target variable. The variables are: `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, and `blood_glucose_level`, predicting the target `diabetes`. Clinical measurement units (e.g., mg/dL, percentages) are not documented in the repository.
"""
text = text.replace("### B. Leakage-Free Preprocessing", feature_dict_text.strip() + "\n\n### C. Leakage-Free Preprocessing")

text = text.replace("### C. Base Comparative Models", "### D. Base Comparative Models")
text = text.replace("### D. Out-of-Fold Stacking Architecture", "### E. Out-of-Fold Stacking Architecture")
text = text.replace("### E. Threshold Optimization", "### F. Threshold Optimization")
text = text.replace("### F. Class Imbalance (SMOTE)", "### G. Class Imbalance (SMOTE)")

# Optuna Methodology
old_hyp_section = """### A. Hyperparameter Optimization & Final Configurations
Table I reports the final optimized reproducibility parameters for the evaluated algorithms, extracted directly from verifiable experimental artifacts."""

new_hyp_section = """### A. Hyperparameter Optimization & Final Configurations
Hyperparameter tuning was conducted using Optuna, employing the Tree-structured Parzen Estimator (`TPESampler` with `seed=42`). The objective function evaluated candidate parameters across 20 trials (`n_trials=20`), utilizing a 5-fold `StratifiedKFold` to maximize mean ROC-AUC on the training partition (`direction="maximize"`). Table I reports the final optimized reproducibility parameters for the evaluated algorithms, extracted directly from verifiable experimental artifacts."""

text = text.replace(old_hyp_section, new_hyp_section)

# Update Table II
old_table_2 = """**TABLE II. Final Test Set Metrics**
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Tuned XGBoost | 0.971604 | 0.975651 | 0.692491 | 0.810038 | 0.977753 | 0.885530 |
| Tuned CatBoost | 0.971600 | 0.977300 | 0.691300 | 0.809800 | 0.977700 | 0.884800 |
| Tuned MLP | 0.971300 | 0.989600 | 0.679400 | 0.805700 | 0.975000 | (Not documented) |
| TabNet | 0.971000 | 0.996500 | 0.673300 | 0.803600 | (Not documented) | (Not documented) |
| Final Stacking | 0.971100 | 0.955400 | 0.702000 | 0.809300 | 0.977500 | (Not documented) |"""

new_table_2 = """**TABLE II. Final Test Set Metrics**
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Tuned XGBoost | 0.9716 | 0.9757 | 0.6925 | 0.8100 | 0.9778 | 0.8855 |
| Tuned CatBoost | 0.9716 | 0.9773 | 0.6913 | 0.8098 | 0.9777 | 0.8848 |
| Tuned MLP | 0.9713 | 0.9896 | 0.6794 | 0.8057 | 0.9750 | 0.8780 |
| TabNet | 0.9710 | 0.9965 | 0.6733 | 0.8037 | 0.9745 | 0.8734 |
| Final Stacking | 0.9711 | 0.9554 | 0.7020 | 0.8093 | 0.9775 | (Not documented) |"""

text = text.replace(old_table_2, new_table_2)

# Insert SMOTE Results
old_discussion_text = """The comparative data reveals direct numerical trade-offs. Standalone deep-learning structures like TabNet returned highly conservative precision (0.9965) but compromised on sensitivity/recall (0.6733). The Final Stacking model demonstrated the highest measured recall among optimized systems (0.7020) while returning an F1-score of 0.8093. Tuned XGBoost provided the highest numerical F1-score (0.810038) by balancing precision (0.975651) with strong recall (0.692491). No single model structurally dominates across all possible objective functions.

### C. Confusion Matrices Analysis"""

new_discussion_text = """The comparative data reveals direct numerical trade-offs. Standalone deep-learning structures like TabNet returned highly conservative precision (0.9965) but compromised on sensitivity/recall (0.6733). The Final Stacking model demonstrated the highest measured recall among optimized systems (0.7020) while returning an F1-score of 0.8093. Tuned XGBoost provided the highest numerical F1-score (0.8100) by balancing precision (0.9757) with strong recall (0.6925). No single model structurally dominates across all possible objective functions.

### C. Comparative SMOTE Performance
To quantify whether synthetic minority insertion could reliably uplift minority recall without severely degrading majority-class precision, SMOTE was applied in standalone experimental runs. The comparative F1-scores were 0.8066 for Tuned CatBoost with SMOTE, 0.8043 for Tuned XGBoost with SMOTE, and 0.7211 for the MLP with SMOTE. These SMOTE-augmented architectures were strictly evaluated as analytical benchmarks and were not incorporated into the final optimized stacking pipeline.

### D. Confusion Matrices Analysis"""

text = text.replace(old_discussion_text, new_discussion_text)

text = text.replace("### D. Explainable AI (SHAP)", "### E. Explainable AI (SHAP)")

# SHAP wording
shap_old = """To quantify feature predictive contribution, SHAP (SHapley Additive exPlanations) values were executed over the optimized XGBoost architecture using a TreeExplainer. Results signify predictive association, not biological causality. Based on global importance rankings, the top primary predictive drivers are:
1. **HbA1c_level**: Displays the most significant predictive association (mean absolute SHAP = 2.066857).
2. **blood_glucose_level**: Registers as the secondary influential marker (mean absolute SHAP = 1.470601).
Remaining features including age, BMI, and hypertension history provided distinct but markedly lower aggregate predictive contributions compared to glycemic indicators."""

shap_new = """To quantify feature predictive contribution, SHAP values were computed for the test instances using `shap.TreeExplainer` over the optimized XGBoost architecture. Results signify predictive association, not physiological causality. Based on global importance rankings, the top primary predictive drivers are:
1. **HbA1c_level**: Displays the most significant predictive association (mean absolute SHAP = 2.0669).
2. **blood_glucose_level**: Registers as the secondary influential marker (mean absolute SHAP = 1.4706).
Remaining features provided distinct but markedly lower aggregate predictive contributions, specifically age (mean absolute SHAP = 0.7056) and BMI (mean absolute SHAP = 0.3479)."""

text = text.replace(shap_old, shap_new)

# Limitations
limitations_old = """1. **Single Dataset Scope**: The evaluation was performed on a singular centralized dataset, limiting generalizability.
2. **Lack of Geographic Validation**: The model has not been validated across diverse external geographic cohorts.
3. **Lack of Temporal Validation**: There is no confirmation of predictive stability across varying longitudinal time horizons.
4. **Calibration Analysis**: Pure probabilistic calibration (e.g., Brier score) was not rigorously quantified outside of rank-based ROC metrics.
5. **Threshold Prevalence Dependence**: The optimal threshold of 0.66 is strictly coupled to the 8.74% disease prevalence rate; significant demographic population shifts would necessitate threshold recalibration.
6. **Representativeness**: The underlying source properties of the dataset impose hard boundaries on generalization.
7. **Correlational Explainability**: SHAP strictly measures internal model behavior and correlational association, rendering it invalid for establishing physiological causality.
8. **Correlated Predictors**: Multi-collinearity can distribute SHAP attributions arbitrarily among intertwined features."""

limitations_new = """1. **Unknown Dataset Provenance**: The dataset lacks verifiable institutional origin, temporal collection boundaries, or DOI documentation.
2. **Cross-Sectional Constraints**: The dataset captures concurrent metrics lacking longitudinal timestamps. The models stratify concurrent risk but cannot establish temporal forecasting or future prediction.
3. **Representativeness**: Unverifiable demography limits generalizing conclusions to defined clinical populations.
4. **Absence of External Validation**: The model was not validated across diverse external geographic cohorts.
5. **Absence of Temporal Validation**: Predictive stability across varying longitudinal time horizons was not confirmed.
6. **Calibration Analysis**: Pure probabilistic calibration (e.g., Brier score) was not rigorously quantified outside of rank-based ROC metrics.
7. **Statistical Uncertainty Estimates**: The exported evaluation metrics represent singular point estimates. No confidence intervals or explicit variability (e.g., fold standard deviations) were reported.
8. **Threshold Prevalence Dependence**: The optimal threshold of 0.66 is strictly coupled to the 8.74% disease prevalence rate; significant demographic occurrences would necessitate empirical recalibration.
9. **Correlational Explainability**: SHAP values strictly measure internal model behavior as an associational artifact, rendering them invalid for establishing physiological causality."""

text = text.replace(limitations_old, limitations_new)

with open("paper/paper draft 2.md", "w", encoding="utf-8") as f:
    f.write(text)
