with open("paper/paper draft 2.md", "r") as f:
    text = f.read()

import re
table_pattern = r"(?s)\*\*TABLE II\. Final Test Set Metrics\*\*\s*\| Model.*?\n\n"
match = re.search(table_pattern, text)
if match:
    old_table2 = match.group(0)

new_table2 = """**TABLE II. Final Test Set Metrics**
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Tuned XGBoost | 0.971604 | 0.975651 | 0.692491 | 0.810038 | 0.977753 | 0.885530 |
| Tuned CatBoost | 0.9716 | 0.9773 | 0.6913 | 0.8098 | 0.9777 | 0.8848 |
| Tuned MLP | 0.9713 | 0.9896 | 0.6794 | 0.8057 | 0.974979 | 0.877987 |
| TabNet | 0.9710 | 0.9965 | 0.6733 | 0.8037 | 0.974507 | 0.873419 |
| Final Stacking | 0.9711 | 0.9554 | 0.7020 | 0.8093 | 0.9775 | (Not documented) |

"""
if match:
    text = text.replace(old_table2, new_table2)
else:
    print("Could not find Table II")

# I also need to make sure the inline text matches
text = text.replace("Tuned XGBoost provided the highest numerical F1-score (0.8100) by balancing precision (0.9757) with strong recall (0.6925).", 
                    "Tuned XGBoost provided the highest numerical F1-score (0.810038) by balancing precision (0.975651) with strong recall (0.692491).")

with open("paper/paper draft 2.md", "w") as f:
    f.write(text)
