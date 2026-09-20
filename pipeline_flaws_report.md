# Pipeline Architecture Flaws: Why XGBoost Unfairly Dominated Deep Learning

During the automated audit of the training and testing pipelines for the 100k-dataset diabetes risk stratification project, four critical methodological flaws were uncovered. These flaws systematically crippled the performance of the Deep Learning architectures (Multi-Layer Perceptron and TabNet) and the SMOTE oversampling pipeline, while granting native advantages to XGBoost. 

Consequently, the conclusion that Tuned XGBoost legitimately outperforms Deep Learning and SMOTE on this tabular dataset is mathematically invalid.

Below is a detailed breakdown of the four flaws.

## 1. SMOTE Distance Bias & Categorical Interpolation
SMOTE (Synthetic Minority Over-sampling Technique) generates realistic "fake" patients by finding a diabetic patient, locating their nearest diabetic neighbors via Euclidean Distance, and plotting a point randomly along the line connecting them. 

* **The Distance Calculation Flaw:** In `src/smote_experiment.py`, SMOTE was executed **before** continuous features were scaled. Because Euclidean Distance is scale-dependent, unscaled variables with massive numeric ranges (like `blood_glucose_level` spanning up to 300) completely dominated the distance calculations ($300^2$). The algorithm mathematically ignored boolean conditions (like hypertension or gender flags) because their tiny 0-to-1 variance was invisible compared to large continuous bounds.
* **The Interpolation Flaw:** The script used vanilla `SMOTE` instead of `SMOTENC` (Nominal and Continuous). Because nominal one-hot categorical flags were treated as continuous numbers, the algorithm geometrically interpolated them. If it connected a patient with `hypertension=1` to a neighbor with `hypertension=0`, it generated a synthetic patient with `hypertension=0.5`. Feeding physically impossible fractional categories into downstream neural networks destroys the embedding manifold the models are trying to map.

## 2. TabNet's Missing Categorical Embeddings
TabNet is an attentive deep learning architecture designed to rival Gradient Boosters on tabular data by utilizing "Sequential Attention"—learning to ignore useless features at different decision steps. 

* **The Categorical Embedding Flaw:** To handle categorical string variations, TabNet relies on Entity Embeddings (mapping a discrete integer to a dense tensor array). However, in `preprocessing.py`, all categorical columns were transformed globally into sparse One-Hot Encoded matrices. 
* Furthermore, in `src/models/tabnet_model.py`, TabNet was instantiated without defining `cat_idxs` (categorical indices) or `cat_dims`. Thus, TabNet was completely blinded. It assumed every one-hot boolean column was an entirely independent, continuous numerical variable. The model was forced to waste its attention mechanism bouncing across scattered zeros and ones, completely bypassing the embedding architecture that grants TabNet its primary power.

## 3. MLP's Accuracy-Based Early Stopping (The "Lazy Network")
Deep learning models optimize along the path of least resistance during gradient descent. In `src/tuning/mlp_tuning.py`, Scikit-Learn’s `MLPClassifier` was configured with `early_stopping=True`.

* **The Metric Tracking Flaw:** Scikit-Learn hardcodes its early stopping callback to monitor mean **Accuracy**. Given that the dataset is roughly **91.2% non-diabetic**, the absolute easiest way for the network to rapidly achieve 91.2% accuracy is to become lazy and predict `0` (Negative) for every single patient. 
* When the network attempts to map the complex gradients of the minority class, its overall accuracy often drops temporarily (e.g., to 89%) before recovering. The built-in early stopping mechanism perceives this drop as overfitting and halts training prematurely. The MLP never received sufficient epochs to map the positive diabetic boundary. 

## 4. XGBoost's Cost-Sensitive `scale_pos_weight`
This structural difference rendered the baseline comparisons fundamentally unfair.

* **The Native Weighting Bias:** In `src/models/xgboost_model.py`, the model explicitly utilized `scale_pos_weight = negatives / positives` (roughly a 10.4 multiplier). This explicitly penalized the XGBoost loss function. It instructed XGBoost that misclassifying a Diabetic patient (Positive class) incurs a mathematical penalty **10.4 times worse** than misclassifying a healthy patient. 
* Scikit-Learn's MLP and PyTorch's TabNet do not possess an automated `class_weights` argument for continuous cross-entropy loss tracking natively. When trained, they treated a missed Diabetic patient with the exact same weight as a missed Healthy patient. XGBoost was artificially forced to care deeply about the minority class, while the neural networks were left mathematically indifferent. 

***

**Required Actions (Pending Execution):**
1. **SMOTE**: Rewrite `smote_experiment.py` to scale continuous features *prior* to oversampling, and pass categorical indices directly into `SMOTENC`.
2. **MLP**: Disable accuracy-based early stopping, and implement custom Optuna callbacks targeting ROC-AUC or F1-Score for validation. 
3. **TabNet**: Refactor `tabnet_model.py` to intercept ordinal categoricals before they hit the One-Hot Encoder, enabling true entity embeddings.
