#!/usr/bin/env python3
"""Final finishing pass for paper/mainnnn.tex"""

import re
import json
from pathlib import Path

# Load the manuscript
with open('paper/mainnnn.tex', 'r', encoding='utf-8') as f:
    content = f.read()

original_content = content

# 1. Fix the last project-specific "SMOTE" reference
print("1. Fixing last project-specific SMOTE reference...")
old = "We also conduct experiments based on SMOTE to check"
new = "We also conduct experiments based on SMOTENC to check"
if old in content:
    content = content.replace(old, new)
    print(f"  ✓ Changed: {old}")
else:
    print(f"  ✗ Not found: {old}")

# 2. Standardize threshold terminology
print("\n2. Standardizing threshold terminology...")

# Replace ambiguous phrases
replacements = [
    ("out-of-fold training predictions", "meta-level cross-fitted OOF probabilities"),
    ("OOF training predictions", "meta-level cross-fitted OOF probabilities"),
    ("training-side OOF predictions", "meta-level cross-fitted OOF probabilities"),
    ("OOF F1-score", "meta-OOF F1-score"),
    ("the OOF predictions", "the meta-level cross-fitted OOF probabilities"),
    ("an meta-OOF", "a meta-OOF"),  # Also fix the grammar issue
]

for old_phrase, new_phrase in replacements:
    if old_phrase in content:
        content = content.replace(old_phrase, new_phrase)
        print(f"  ✓ {old_phrase} → {new_phrase}")

# 3. Fix the one grammar error specifically
print("\n3. Fixing grammar error...")
if "with an meta-OOF" in content:
    content = content.replace("with an meta-OOF", "with a meta-OOF")
    print("  ✓ Fixed 'an meta-OOF' → 'a meta-OOF'")
else:
    print("  ✗ Grammar error not found")

# 4. Fix Figure 1 branch direction
print("\n4. Fixing Figure 1 branch direction...")
# Find the Figure 1 content
fig_pattern = r'\\begin\{figure\*\}.*?\\end\{figure\*\}'
fig_match = re.search(fig_pattern, content, re.DOTALL)
if fig_match:
    fig_content = fig_match.group(0)
    original_fig = fig_content

    # Remove the problematic circular arrows
    new_fig_content = re.sub(
        r'\\\\draw\[arrow\] \(smote\.east\) -- \+\+\(0\.6cm,0\) \|- \(\[yshift=0\.3cm\]compare\.north east\) -- \(compare\.north\);',
        '',  # Remove the arrow from SMOTENC back to compare
        fig_content
    )
    new_fig_content = re.sub(
        r'\\\\draw\[arrow\] \(optuna\.east\) -- \+\+\(0\.6cm,0\) \|- \(\[yshift=-0\.3cm\]compare\.south east\) -- \(compare\.south\);',
        '',  # Remove the arrow from Optuna back to compare
        new_fig_content
    )

    # Add arrows from SMOTENC/Optuna to Model Performance Comparison
    # We need to find the correct structure
    # Let's rebuild the comparison node properly
    # Search for "Model Performance Comparison" in the figure
    if "Model Performance Comparison" in new_fig_content:
        # We'll replace the problematic section entirely
        # The problematic part starts with SMOTENC Experiments and Optuna branches
        # Let's get the TikZ code between the arrows
        new_fig_content = re.sub(
            r'% Experimental branches \(right of compare\).*?\\\\draw\[arrow\] \(compare\.east\) -- \+\+\(0\.8cm,0\) \|- \(base\.west\);',
            r"""% Experimental branches (right of compare)
\\node[smallbox, right=2.0cm of compare, yshift=0.8cm, fill=orange!10] (smote) {SMOTENC Experiments};
\\node[smallbox, right=2.0cm of compare, yshift=-0.8cm, fill=purple!10] (optuna) {Optuna Hyperparameter Optimization};
\\node[box, below=34mm of models] (comparison) {
    Model Performance Comparison\\
    Accuracy, Precision, Recall,\\
    F1-score, ROC-AUC, PR-AUC
};

% Arrows for experimental branches
\\draw[arrow] (compare) -- ++(0.6cm,0) |- (smote.west);
\\draw[arrow] (compare) -- ++(0.6cm,0) |- (optuna.west);
\\draw[arrow] (smote.east) -- ++(0.6cm,0) |- (comparison.north);
\\draw[arrow] (optuna.east) -- ++(0.6cm,0) |- (comparison.south);
\\draw[arrow] (models) -- (comparison);

% Primary final model branch (below compare, left)
\\node[box, below=1.2cm of comparison, fill=cyan!10] (tuned) {Tuned XGBoost\\(Primary Final Model)};
\\node[box, below=0.5cm of tuned] (eval) {Independent Test Evaluation};
\\node[box, below=0.5cm of eval, fill=gray!10] (shap) {SHAP Explainability};
\\draw[arrow] (comparison) -- (tuned);
\\draw[arrow] (tuned) -- (eval);
\\draw[arrow] (eval) -- (shap);

% Stacking branch (right, aligned with compare -> tuned)
\\node[smallbox, right=2.8cm of compare, yshift=-2.2cm, fill=red!10] (base) {Base-Level OOF Predictions\\(XGBoost + CatBoost + MLP\\5-Fold Cross-Validation)};
\\node[box, below=0.5cm of base] (xmeta) {$X_{\\text{meta\\_train}}$};
\\node[box, below=0.5cm of xmeta, fill=red!15] (cross) {Meta-Level Cross-Fitting\\(5-Fold Logistic Regression)};
\\node[box, below=0.5cm of cross] (metaoof) {Meta-OOF Probabilities};
\\node[box, below=0.5cm of metaoof] (thresh) {Threshold Optimization};
\\node[box, below=0.5cm of thresh, fill=blue!15] (final) {Final Logistic Regression\\Fitted on Full $X_{\\text{meta\\_train}}$};
\\node[box, below=0.5cm of final] (xmetatest) {$X_{\\text{meta\\_test}}$};
\\node[box, below=0.5cm of xmetatest] (eval2) {Independent Test Evaluation};

\\draw[arrow] (base) -- (xmeta);
\\draw[arrow] (xmeta) -- (cross);
\\draw[arrow] (cross) -- (metaoof);
\\draw[arrow] (metaoof) -- (thresh);
\\draw[arrow] (thresh) -- (final);
\\draw[arrow] (final) -- (xmetatest);
\\draw[arrow] (xmetatest) -- (eval2);
\\draw[arrow] (compare.east) -- ++(0.8cm,0) |- (base.west);""",
            new_fig_content,
            flags=re.DOTALL
        )
        print("  ✓ Rebuilt Figure 1 with proper branch direction")
    else:
        print("  ✗ Could not locate Model Performance Comparison node")

    # Replace the figure content
    content = content.replace(original_fig, new_fig_content)
else:
    print("  ✗ Figure 1 not found")

# Write changes
with open('paper/mainnnn.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n5. Verifying stacking result artifacts...")
# Check the stacking predictions file
stacking_csv = Path("results/comparison/stacking_predictions.csv")
if stacking_csv.exists():
    with open(stacking_csv, 'r') as f:
        lines = f.readlines()
        if len(lines) > 1:
            metrics_line = lines[1]
            parts = metrics_line.strip().split(',')
            if len(parts) >= 7:
                accuracy, precision, recall, f1, roc_auc, pr_auc = parts[1:7]
                print(f"  ✓ Found stacking predictions:")
                print(f"    Accuracy:  {accuracy}")
                print(f"    Precision: {precision}")
                print(f"    Recall:    {recall}")
                print(f"    F1:        {f1}")
                print(f"    ROC-AUC:   {roc_auc}")
                print(f"    PR-AUC:    {pr_auc}")
            else:
                print("  ✗ Stacking CSV format unexpected")
else:
    print("  ✗ Stacking predictions file not found")

# Check the threshold artifact
threshold_json = Path("results/comparison/stacking_best_threshold.json")
if threshold_json.exists():
    with open(threshold_json, 'r') as f:
        threshold_data = json.load(f)
        print(f"  ✓ Found threshold artifact:")
        print(f"    Threshold: {threshold_data.get('selected_threshold', 'N/A')}")
        print(f"    Meta-OOF F1: {threshold_data.get('meta_oof_f1', 'N/A')}")
        print(f"    Range: {threshold_data.get('range', '0.10 to 0.90')}")
        print(f"    Step: {threshold_data.get('step', '0.01')}")
        print(f"    Verified: {threshold_data.get('verified', 'N/A')}")
else:
    print("  ✗ Threshold JSON file not found")

print("\n6. Final source search...")
search_terms = [
    r'\\textbfX',
    r'\\textbfNote',
    r'\\textbfModel',
    r'\\t',
    r'\\input\{figure1_workflow\.tex\}',
    'SMOTE',
    'OOF F1-score',
    'OOF training predictions',
    'training-side OOF'
]

for term in search_terms:
    if term == 'SMOTE':
        # Count SMOTE occurrences
        count = len(re.findall(r'\bSMOTE\b', content))
        print(f"  SMOTE occurrences: {count}")
        if count > 0:
            # Show context
            matches = re.finditer(r'\bSMOTE\b', content)
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]
                if 'cite' not in context and 'literature' not in context:
                    print(f"    Warning: Project-specific SMOTE at ~pos {match.start()}: ...{context}...")
    elif term.startswith('\\'):
        count = len(re.findall(re.escape(term), content))
        if count > 0:
            print(f"  ✗ Found {term}: {count} occurrences")
        else:
            print(f"  ✓ No {term} found")
    else:
        if term in content:
            print(f"  ✗ Found '{term}'")
        else:
            print(f"  ✓ No '{term}' found")

print("\n✅ Final finishing pass complete")
print("\nNow please compile the manuscript and verify the final PDF.")