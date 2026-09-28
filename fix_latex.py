#!/usr/bin/env python3
"""Fix malformed LaTeX in paper/mainnnn.tex"""
import re

path = 'paper/mainnnn.tex'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# -------------------------------------------------------
# 1. Fix \\t\t\t\\textbfX\_meta\_train} → \texttt{X\_meta\_train}
#    and \\t\t\t\\textbfX\_meta\_test}  → \texttt{X\_meta\_test}
# -------------------------------------------------------
content = re.sub(
    r'\\t\t+\\textbfX\\_meta\\_train\}',
    r'\\texttt{X\\_meta\\_train}',
    content
)
content = re.sub(
    r'\\t\t+\\textbfX\\_meta\\_test\}',
    r'\\texttt{X\\_meta\\_test}',
    content
)
# In case there is a remaining bare \textbfX\_meta pattern
content = re.sub(
    r'\\textbfX\\_meta\\_train\}',
    r'\\texttt{X\\_meta\\_train}',
    content
)
content = re.sub(
    r'\\textbfX\\_meta\\_test\}',
    r'\\texttt{X\\_meta\\_test}',
    content
)

# -------------------------------------------------------
# 2. Fix table header \\\textbf → \textbf  (spurious leading backslash)
# -------------------------------------------------------
content = content.replace('\\\textbf{Model}', '\\textbf{Model}')
content = content.replace('\\\textbf{Hyperparameters}', '\\textbf{Hyperparameters}')

# -------------------------------------------------------
# 3. Fix malformed \textbfXGBoost Tuned} etc.  (missing opening brace)
# -------------------------------------------------------
content = content.replace('\\textbfXGBoost Tuned}', '\\textbf{XGBoost Tuned}')
content = content.replace('\\textbfCatBoost Tuned}', '\\textbf{CatBoost Tuned}')
content = content.replace('\\textbfMLP Tuned}', '\\textbf{MLP Tuned}')

# -------------------------------------------------------
# 4. Add early_stopping: False to standalone MLP row if missing
# -------------------------------------------------------
old_mlp = 'batch_size: 128, max_iter: 406 \\\\'
new_mlp = 'batch_size: 128, max_iter: 406, early\\_stopping: False \\\\'
content = content.replace(old_mlp, new_mlp)

# -------------------------------------------------------
# 5. SMOTENC standardization – project-specific occurrences
# -------------------------------------------------------
# Only change project-specific text, not citations
replacements = [
    ('SMOTE experiments',                   'SMOTENC experiments'),
    ('SMOTE-based configurations',          'SMOTENC-based configurations'),
    ('Effect of SMOTE',                     'Effect of SMOTENC'),
    ('XGBoost + SMOTE',                     'XGBoost + SMOTENC'),
    ('CatBoost + SMOTE',                    'CatBoost + SMOTENC'),
    ('MLP + SMOTE',                         'MLP + SMOTENC'),
    ('separate SMOTE and hyperparameter',   'separate SMOTENC and hyperparameter'),
    ('with SMOTE',                          'with SMOTENC'),
    # Table column headings
    ('XGBoost + SMOTE }',                   'XGBoost + SMOTENC }'),
    ('CatBoost + SMOTE }',                  'CatBoost + SMOTENC }'),
    ('MLP + SMOTE }',                       'MLP + SMOTENC }'),
]
for old, new in replacements:
    content = content.replace(old, new)

# -------------------------------------------------------
# 6. Threshold/OOF terminology fixes
# -------------------------------------------------------
# "OOF F1-score" → "meta-OOF F1-score"  (only in threshold context)
content = re.sub(r'\bOOF F1-score\b', 'meta-OOF F1-score', content)
# "OOF training predictions" → "meta-level cross-fitted OOF probabilities"
content = re.sub(
    r'\bOOF training predictions\b',
    'meta-level cross-fitted OOF probabilities',
    content
)

# -------------------------------------------------------
# Write back
# -------------------------------------------------------
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixes applied successfully.")

# -------------------------------------------------------
# Verify
# -------------------------------------------------------
print("\n--- Remaining bad patterns ---")
bad = [
    r'\\t\t',
    r'\\textbfX',
    r'\\textbfNote',
    r'\\textbfModel',
    r'\\textbfMLP',
    r'\\textbfCatBoost',
    r'\\textbfXGBoost',
    r'\\input{figure1_workflow',
    r'OOF F1-score',  # must now be meta-OOF
]
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

for pattern in bad:
    matches = [(m.start(), text[max(0,m.start()-20):m.end()+30]) for m in re.finditer(pattern, text)]
    if matches:
        print(f"STILL PRESENT: {pattern}")
        for pos, ctx in matches[:3]:
            print(f"  line ~{text[:pos].count(chr(10))+1}: ...{ctx}...")
    else:
        print(f"OK: {pattern}")
