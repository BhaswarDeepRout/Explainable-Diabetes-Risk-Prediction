import re

with open("paper/paper draft 2.md", "r", encoding="utf-8") as f:
    text = f.read()

limitations_new = """## 5. Limitations and Future Work
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

## 6. Conclusion"""

text = re.sub(r'## 5\. Limitations and Future Work.*?## 6\. Conclusion', limitations_new, text, flags=re.DOTALL)

with open("paper/paper draft 2.md", "w", encoding="utf-8") as f:
    f.write(text)
