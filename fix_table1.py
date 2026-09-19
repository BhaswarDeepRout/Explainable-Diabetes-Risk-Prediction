with open("paper/paper draft 2.md", "r") as f:
    text = f.read()

old_table_1 = """**TABLE I. Final Verified Hyperparameter Configurations**
* **XGBoost**: `n_estimators`: 697, `max_depth`: 6, `learning_rate`: 0.04118, `subsample`: 0.6010, `colsample_bytree`: 0.7545, `gamma`: 2.4144, `min_child_weight`: 4, `reg_alpha`: 1.9438, `reg_lambda`: 1.1186.
* **CatBoost**: `iterations`: 665, `depth`: 6, `learning_rate`: 0.02176, `l2_leaf_reg`: 10.9945, `bagging_temperature`: 6.6177, `random_strength`: 1.4730, `border_count`: 118.
* **MLP (Standalone Benchmark)**: `hidden_layer_sizes`: (128, 64), `activation`: tanh, `alpha`: 2.2622e-05, `learning_rate_init`: 0.00177, `batch_size`: 64, `max_iter`: 454.
* **MLP (Level-0 Stacking Learner)**: `hidden_layer_sizes`: (128, 64), `activation`: relu, `alpha`: 0.0001, `learning_rate_init`: 0.001, `batch_size`: 128, `max_iter`: 500.
* **TabNet (Standalone Benchmark)**: `n_d`: 16, `n_a`: 16, `n_steps`: 5, `gamma`: 1.5, `optimizer_lr`: 2e-3, `batch_size`: 1024, `virtual_batch_size`: 128.
* **Logistic Regression (Meta-Learner)**: `max_iter`: 1000."""

new_table_1 = """**TABLE I. Final Verified Hyperparameter Configurations**

| Model | Hyperparameters |
|---|---|
| **XGBoost** | `n_estimators`: 697, `max_depth`: 6, `learning_rate`: 0.04118, `subsample`: 0.6010, `colsample_bytree`: 0.7545, `gamma`: 2.4144, `min_child_weight`: 4, `reg_alpha`: 1.9438, `reg_lambda`: 1.1186 |
| **CatBoost** | `iterations`: 665, `depth`: 6, `learning_rate`: 0.02176, `l2_leaf_reg`: 10.9945, `bagging_temperature`: 6.6177, `random_strength`: 1.4730, `border_count`: 118 |
| **MLP (Standalone Benchmark)** | `hidden_layer_sizes`: (128, 64), `activation`: tanh, `alpha`: 2.2622e-05, `learning_rate_init`: 0.00177, `batch_size`: 64, `max_iter`: 454 |
| **MLP (Level-0 Stacking)** | `hidden_layer_sizes`: (128, 64), `activation`: relu, `alpha`: 0.0001, `learning_rate_init`: 0.001, `batch_size`: 128, `max_iter`: 500 |
| **TabNet (Standalone Benchmark)** | `n_d`: 16, `n_a`: 16, `n_steps`: 5, `gamma`: 1.5, `optimizer_lr`: 2e-3, `batch_size`: 1024, `virtual_batch_size`: 128 |
| **Logistic Regression (Meta)**| `max_iter`: 1000 |"""

text = text.replace(old_table_1, new_table_1)

with open("paper/paper draft 2.md", "w", encoding="utf-8") as f:
    f.write(text)
