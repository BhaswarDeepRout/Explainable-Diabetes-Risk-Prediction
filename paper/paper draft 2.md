# Hybrid Deep Learning and Ensemble Learning with Explainable AI for Diabetes Risk Stratification

## Abstract
Diabetes mellitus is a significant chronic health condition for which risk stratification can facilitate appropriate clinical management. Machine learning and deep learning methodologies offer analytical approaches to identify individuals at elevated risk based on patient health indicators. Challenges such as class imbalance, methodological leakage, and model interpretability can impair the reliability of predictive algorithms. This study proposes and strictly evaluates a hybrid framework incorporating machine learning, ensemble learning, and Explainable Artificial Intelligence (XAI) for diabetes risk stratification. We utilized a dataset of 100,000 raw observations, rigorously deduplicated to 95,964 clean observations. The methodology employs a leakage-free preprocessing pipeline utilizing scikit-learn's ColumnTransformer, followed by a 5-fold Stratified cross-validation out-of-fold (OOF) stacking architecture. The Level-0 models include Tuned XGBoost, Tuned CatBoost, and Scaled MLP, integrated via a Logistic Regression meta-learner. Threshold optimization was conducted exclusively on OOF predictions (optimal F1-score threshold: 0.66) and applied to untouched holdout test data. The final stacking model achieved a test accuracy of 0.9711, precision of 0.9554, recall of 0.7020, and F1-score of 0.8093. SHAP (SHapley Additive exPlanations) analysis identified HbA1c and blood glucose levels as the primary features contributing to model predictions. Finally, we establish that while the stack exhibits strong predictive metrics, the presence of 500 false negatives in independent testing emphasizes its role as a risk-stratification or decision-support tool rather than an autonomous diagnostic system.

## 1. Introduction
Diabetes mellitus represents a major public health challenge globally. The capacity to analyze large-scale tabular health data using advanced predictive models offers a pathway for risk stratification. Contemporary predictive architectures frequently deploy decision-tree ensembles, deep learning networks, and stacking mechanisms to capture complex non-linear relationships in patient data. However, the evaluation of these systems must be methodologically robust, guarding against data leakage, class imbalance, and performance exaggeration.

This study systematically develops and audits a predictive pipeline for diabetes risk stratification. Rather than asserting clinical autonomy, this work contributes a strictly verified methodological approach comparing standalone models (including XGBoost, CatBoost, Multi-Layer Perceptrons, and TabNet) against a Level-1 stacking ensemble. The core contributions of this study are:
1. Application of a strictly isolated, leakage-free preprocessing pipeline fitted exclusively on training data.
2. Construction of an Out-Of-Fold (OOF) Level-1 stacking ensemble.
3. Separation of continuous probability output and discrete classification through OOF-based threshold optimization.
4. Interpretation of the final gradient boosting model using global SHAP analysis to quantify predictive contribution.

## 2. Related Work
Contemporary studies have highlighted the utility of machine learning in diabetes risk prediction. Elgendy et al. [1] proposed a dual-stage explainable stacking framework using multiple base learners and autoencoder-based reconstruction, leveraging SHAP for explainability on the MIMIC-IV and Pima Indians datasets to demonstrate the viability of stacking. In another study, Aulia [2] focused on the Pima Indians dataset using an optimized Random Forest classifier and SHAP, exploring SMOTE and ADASYN to observe how oversampling impacts recall and overall predictive capability. Rafie et al. [3] successfully leveraged XGBoost and explainable AI to construct robust modeling for Type 2 diabetes prediction, emphasizing the balance between predictive strength and interpretability. Zaferani et al. [4] investigated transparent ensemble architectures combining Random Forests, K-Nearest Neighbors, and Neural Networks. Similarly, Ganguly and Singh [5] established ensemble approaches integrated with XAI for the broader context of diabetes management. These works collectively affirm the value of structured ensemble architectures augmented by interpretability techniques.

## 3. Methodology

### A. Dataset and Partitioning
The experimental pipeline executes on a dataset (`s1.csv`) comprising 100,000 raw observations and 9 columns representing clinical and demographic indicators, with the target variable indicating concurrent diabetes status. Dataset provenance, including formal institutional origin, DOI, and temporal collection boundaries, is not documented within the repository artifacts, which constitutes a primary limitation regarding representativeness. To ensure data integrity, exact statistical duplicates and contradictory observations (instances with identical features but opposing target variables) were systematically removed. This cleaning stage omitted 3,854 exact duplicate rows and 182 contradictory observations, yielding a final dataset of 95,964 robust observations.

The dataset presents a high class imbalance: the final distribution consists of 87,573 non-diabetic observations (91.26%) and 8,391 diabetic observations (8.74%). The dataset was partitioned into training and independent holdout testing subsets using a stratified split (random\_state: 42). The training partition contains 76,771 observations (70,058 non-diabetic, 6,713 diabetic), while the test partition consists of 19,193 observations (17,515 non-diabetic, 1,678 diabetic).

### B. Feature Dictionary
The raw dataset encompasses eight clinical and demographic predictors and one target variable. The variables are: `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, and `blood_glucose_level`, predicting the target `diabetes`. Clinical measurement units (e.g., mg/dL, percentages) are not documented in the repository.

### C. Leakage-Free Preprocessing
To strictly prevent procedural data leakage, all preprocessing operations were encapsulated within a scikit-learn `ColumnTransformer`. Imputation functions (median for numerical and most-frequent for categorical inputs) and categorical encoding definitions (`OneHotEncoder`) were fitted exclusively on the 76,771 instances in the training partition. Following the fitting process, the holdout test set was transformed using exclusively training-derived parameters. At no point was the test set utilized to estimate preprocessing parameters, guaranteeing complete segregation between model building and validation data.

### D. Base Comparative Models
To capture heterogeneous predictive signals, we constructed several Level-0 base tabular models:
1. **Tuned XGBoost**: A highly optimized gradient-boosting implementation.
2. **Tuned CatBoost**: A gradient-boosting algorithm uniquely resistant to overfitting on categorical features.
3. **Scaled MLP**: A deep Multi-Layer Perceptron neural network functioning as a baseline deep-learning component within the ensemble.

Additionally, a standalone optimized **Tuned MLP** and **TabNet** (an attentive tabular deep learning architecture) were evaluated as independent comparative benchmarks to measure baseline tabular deep-learning efficacy. Due to differences in architectural integration, TabNet and the independently Tuned MLP were not included in the final meta-learner stack but provide robust deep-learning comparisons.

### E. Out-of-Fold Stacking Architecture
To mitigate individual model bias, a stacking architecture was designed using a 5-fold Stratified cross-validation (`StratifiedKFold`) approach on the training subset. In this scheme, the three designated Level-0 models (XGBoost, CatBoost, and MLP) generated out-of-fold (OOF) prediction probabilities. These continuous probabilities formed intermediate meta-features. A Logistic Regression meta-learner (Level-1) was subsequently trained exclusively on this probability matrix to generate final aggregate predictions. Holdout test predictions were generated only at the final evaluation checkpoint.

### F. Threshold Optimization
Because default classification thresholds (e.g., 0.50) are frequently suboptimal under severe minority-class conditions (8.74% positive prevalence), a grid search threshold optimization was performed. We evaluated candidate thresholds from 0.10 to 0.90 in increments of 0.01 against the OOF predictions, utilizing the F1-score as the target metric. The optimal threshold derived from OOF training data was identified as 0.66, which yielded an OOF F1-score of 0.8135. This classification threshold (0.66) was subsequently locked and applied identically to the untouched holdout test probabilities.

### G. Class Imbalance (SMOTE)
In addition to native classification, SMOTE (Synthetic Minority Over-sampling Technique) was implemented strictly as a comparative analytic benchmark to measure empirical trade-offs. The benchmark sought to quantify whether synthetic minority insertion could reliably uplift minority recall without severely degrading majority-class precision. SMOTE integration was restricted to experimental runs and was not incorporated into the final optimized stacking pipeline.

## 4. Results and Discussion

### A. Hyperparameter Optimization & Final Configurations
Hyperparameter tuning was conducted using Optuna, employing the Tree-structured Parzen Estimator (`TPESampler` with `seed=42`). The objective function evaluated candidate parameters across 20 trials (`n_trials=20`), utilizing a 5-fold `StratifiedKFold` to maximize mean ROC-AUC on the training partition (`direction="maximize"`). Table I reports the final optimized reproducibility parameters for the evaluated algorithms, extracted directly from verifiable experimental artifacts.

**TABLE I. Final Verified Hyperparameter Configurations**

| Model | Hyperparameters |
|---|---|
| **XGBoost** | `n_estimators`: 697, `max_depth`: 6, `learning_rate`: 0.04118, `subsample`: 0.6010, `colsample_bytree`: 0.7545, `gamma`: 2.4144, `min_child_weight`: 4, `reg_alpha`: 1.9438, `reg_lambda`: 1.1186 |
| **CatBoost** | `iterations`: 665, `depth`: 6, `learning_rate`: 0.02176, `l2_leaf_reg`: 10.9945, `bagging_temperature`: 6.6177, `random_strength`: 1.4730, `border_count`: 118 |
| **MLP (Standalone Benchmark)** | `hidden_layer_sizes`: (128, 64), `activation`: tanh, `alpha`: 2.2622e-05, `learning_rate_init`: 0.00177, `batch_size`: 64, `max_iter`: 454 |
| **MLP (Level-0 Stacking)** | `hidden_layer_sizes`: (128, 64), `activation`: relu, `alpha`: 0.0001, `learning_rate_init`: 0.001, `batch_size`: 128, `max_iter`: 500 |
| **TabNet (Standalone Benchmark)** | `n_d`: 16, `n_a`: 16, `n_steps`: 5, `gamma`: 1.5, `optimizer_lr`: 2e-3, `batch_size`: 1024, `virtual_batch_size`: 128 |
| **Logistic Regression (Meta)**| `max_iter`: 1000 |

### B. Comparative Model Performance
Table II provides the evaluation metrics for all primary models on the 19,193 untouched holdout observations. Because accuracy is heavily distorted by the 91.26% negative class baseline, Precision, Recall, F1, PR-AUC, and ROC-AUC serve as the primary metrics for evaluating predictive strength.

**TABLE II. Final Test Set Metrics**
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Tuned XGBoost | 0.971604 | 0.975651 | 0.692491 | 0.810038 | 0.977753 | 0.885530 |
| Tuned CatBoost | 0.9716 | 0.9773 | 0.6913 | 0.8098 | 0.9777 | 0.8848 |
| Tuned MLP | 0.9713 | 0.9896 | 0.6794 | 0.8057 | 0.974979 | 0.877987 |
| TabNet | 0.9710 | 0.9965 | 0.6733 | 0.8037 | 0.974507 | 0.873419 |
| Final Stacking | 0.9711 | 0.9554 | 0.7020 | 0.8093 | 0.9775 | (Not documented) |

The comparative data reveals direct numerical trade-offs. Standalone deep-learning structures like TabNet returned highly conservative precision (0.9965) but compromised on sensitivity/recall (0.6733). The Final Stacking model demonstrated the highest measured recall among optimized systems (0.7020) while returning an F1-score of 0.8093. Tuned XGBoost provided the highest numerical F1-score (0.810038) by balancing precision (0.975651) with strong recall (0.692491). No single model structurally dominates across all possible objective functions.

### C. Comparative SMOTE Performance
To quantify whether synthetic minority insertion could reliably uplift minority recall without severely degrading majority-class precision, SMOTE was applied in standalone experimental runs. The comparative F1-scores were 0.8066 for Tuned CatBoost with SMOTE, 0.8043 for Tuned XGBoost with SMOTE, and 0.7211 for the MLP with SMOTE. These SMOTE-augmented architectures were strictly evaluated as analytical benchmarks and were not incorporated into the final optimized stacking pipeline.

### D. Confusion Matrices Analysis
Model decision matrices clearly illustrate the applied boundaries of the threshold predictions over the 19,193 test cases:
* **Tuned XGBoost**: Produced 17,486 True Negatives (TN), 29 False Positives (FP), 516 False Negatives (FN), and 1,162 True Positives (TP).
* **Final Stacking (Threshold: 0.66)**: Produced 17,460 True Negatives (TN), 55 False Positives (FP), 500 False Negatives (FN), and 1,178 True Positives (TP).

Both pipelines successfully identified the vast majority of non-diabetic and diabetic patients. However, the exact occurrence of 500 to 516 false negatives dictates that these pipelines possess blind spots and thereby cannot safely operate as fully autonomous diagnostic determinants in a live clinical setting.

### E. Explainable AI (SHAP)
To quantify feature predictive contribution, SHAP values were computed for the test instances using `shap.TreeExplainer` over the optimized XGBoost architecture. Results signify predictive association, not physiological causality. Based on global importance rankings, the top primary predictive drivers are:
1. **HbA1c_level**: Displays the most significant predictive association (mean absolute SHAP = 2.0669).
2. **blood_glucose_level**: Registers as the secondary influential marker (mean absolute SHAP = 1.4706).
Remaining features provided distinct but markedly lower aggregate predictive contributions, specifically age (mean absolute SHAP = 0.7056) and BMI (mean absolute SHAP = 0.3479).

## 5. Limitations and Future Work
While this study establishes a strictly verified testing protocol and reports highly optimized test statistics, significant limitations dictate cautious interpretation:
1. **Unknown Dataset Provenance**: The dataset lacks verifiable institutional origin, temporal collection boundaries, or DOI documentation. 
2. **Cross-Sectional Constraints**: The dataset captures concurrent metrics lacking longitudinal timestamps. The models stratify concurrent risk but cannot establish temporal forecasting or future prediction.
3. **Representativeness**: Unverifiable demography limits generalizing conclusions to defined clinical populations. 
4. **Absence of External Validation**: The model was not validated across diverse external geographic cohorts.
5. **Absence of Temporal Validation**: Predictive stability across varying longitudinal time horizons was not confirmed.
6. **Calibration Analysis**: Pure probabilistic calibration (e.g., Brier score) was not rigorously quantified outside of rank-based ROC metrics.
7. **Statistical Uncertainty Estimates**: The exported evaluation metrics represent singular point estimates. No confidence intervals or explicit variability (e.g., fold standard deviations) were reported.
8. **Threshold Prevalence Dependence**: The optimal threshold of 0.66 is strictly coupled to the 8.74% disease prevalence rate; significant demographic occurrences would necessitate empirical recalibration.
9. **Correlational Explainability**: SHAP values strictly measure internal model behavior as an associational artifact, rendering them invalid for establishing physiological causality.

## 6. Conclusion
This study developed an extensively verified, leakage-free pipeline integrating ensemble machine learning and SHAP interpretable analysis for diabetes risk stratification. Comparative analyses spanning gradient boosting, neural topologies (MLP, TabNet), and Out-Of-Fold stacking successfully identified high-risk populations, constrained by empirical threshold optimization (OOF F1: 0.8135). The Final Stacking classifier achieved an accuracy of 0.9711 and an F1-score of 0.8093 via strict compartmentalization of preprocessing logic. The analysis confirmed the dominant predictive contribution of HbA1c and blood glucose features without requiring biologically causal hypotheses. Recognizing the persistence of minority-class false negatives (500 cases), these systems represent robust risk-stratification aids rather than autonomous diagnostic engines.

## References
[1] A. Elgendy et al., "Dual-stage explainable ensemble learning model for diabetes diagnosis," *Expert Systems with Applications*, vol. 274, p. 126899, 2025, doi: 10.1016/j.eswa.2025.126899.
[2] H. Aulia, A. Wibowo, and Sutrisno, "Explaining Diabetes Prediction Under Oversampling Strategies: A SHAP Based Comparative Study of SMOTE and ADASYN," *2025 8th International Conference on Informatics and Computational Sciences (ICICoS)*, 2025, pp. 1-6, doi: 10.1109/icicos68590.2025.11329981.
[3] Z. Rafie, M. S. Talab, B. E. Z. Koor, et al., "Leveraging XGBoost and explainable AI for accurate prediction of type 2 diabetes," *BMC Public Health*, vol. 25, p. 3688, 2025, doi: 10.1186/s12889-025-24953-w.
[4] N. Zaferani, M. R. Afrash, and K. Moulaei, "Predicting and classifying type 2 diabetes using a transparent ensemble model combining random forest, k-nearest neighbor, and neural networks," *Scientific Reports*, vol. 16, p. 1892, 2026, doi: 10.1038/s41598-025-31562-5.
[5] R. Ganguly and D. Singh, "Explainable Artificial Intelligence (XAI) for the Prediction of Diabetes Management: An Ensemble Approach," *International Journal of Advanced Computer Science and Applications*, vol. 14, no. 7, 2023, doi: 10.14569/IJACSA.2023.0140717.
