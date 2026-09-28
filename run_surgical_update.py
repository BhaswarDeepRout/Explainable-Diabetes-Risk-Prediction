#!/usr/bin/env python3
"""Surgical manuscript update script for paper/mainnnn.tex"""

import re
import os

path = 'paper/mainnnn.tex'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# -------------------------------------------------------------------
# 1. Replace Figure 1 (lines ~117 to ~294) with complete TikZ diagram
# -------------------------------------------------------------------

figure1_tikz = r"""\begin{figure*}[t]
\centering
\begin{tikzpicture}[font=\small, node distance=0.9cm and 1.2cm,
    box/.style={rectangle, draw, rounded corners, fill=blue!8, minimum width=3.4cm, minimum height=0.6cm, align=center},
    arrow/.style={->, >=stealth, thick},
    smallbox/.style={rectangle, draw, rounded corners, fill=green!8, minimum width=3.0cm, minimum height=0.55cm, align=center}]

% Main pipeline (left column)
\node[box] (raw) {Raw Diabetes Dataset};
\node[box, below=0.5cm of raw] (inspect) {Dataset Inspection};
\node[box, below=0.5cm of inspect] (clean) {Data Quality Assessment\\(Duplicate Removal / Conflict Detection)};
\node[box, below=0.5cm of clean] (split) {Stratified Train/Test Split\\(80\% Train / 20\% Test)};
\node[box, below=0.5cm of split, fill=yellow!10] (locked) {Test Set Locked};
\node[box, below=0.5cm of locked] (fitpre) {Fit Learned Preprocessing\\on Training Data ONLY};
\node[box, below=0.5cm of fitpre] (transform) {Transform Train and Test\\(Training-Fitted Scaler/Encoder)};
\node[box, below=0.5cm of transform] (compare) {Comparative Model Development\\(Model Performance Comparison)};

\draw[arrow] (raw) -- (inspect);
\draw[arrow] (inspect) -- (clean);
\draw[arrow] (clean) -- (split);
\draw[arrow] (split) -- (locked);
\draw[arrow] (locked) -- (fitpre);
\draw[arrow] (fitpre) -- (transform);
\draw[arrow] (transform) -- (compare);

% Experimental branches (right of compare)
\node[smallbox, right=2.0cm of compare, yshift=0.8cm, fill=orange!10] (smote) {SMOTENC Experiments};
\node[smallbox, right=2.0cm of compare, yshift=-0.8cm, fill=purple!10] (optuna) {Optuna Hyperparameter Optimization};
\draw[arrow] (compare.east) -- ++(0.6cm,0) |- (smote.west);
\draw[arrow] (compare.east) -- ++(0.6cm,0) |- (optuna.west);
\draw[arrow] (smote.east) -- ++(0.6cm,0) |- ([yshift=0.3cm]compare.north east) -- (compare.north);
\draw[arrow] (optuna.east) -- ++(0.6cm,0) |- ([yshift=-0.3cm]compare.south east) -- (compare.south);

% Primary final model branch (below compare, left)
\node[box, below=1.2cm of compare, fill=cyan!10] (tuned) {Tuned XGBoost\\(Primary Final Model)};
\node[box, below=0.5cm of tuned] (eval) {Independent Test Evaluation};
\node[box, below=0.5cm of eval, fill=gray!10] (shap) {SHAP Explainability};
\draw[arrow] (compare) -- (tuned);
\draw[arrow] (tuned) -- (eval);
\draw[arrow] (eval) -- (shap);

% Stacking branch (right, aligned with compare -> tuned)
\node[smallbox, right=2.8cm of compare, yshift=-2.2cm, fill=red!10] (base) {Base-Level OOF Predictions\\(XGBoost + CatBoost + MLP\\5-Fold Cross-Validation)};
\node[box, below=0.5cm of base] (xmeta) {$X_{\text{meta\_train}}$};
\node[box, below=0.5cm of xmeta, fill=red!15] (cross) {Meta-Level Cross-Fitting\\(5-Fold Logistic Regression)};
\node[box, below=0.5cm of cross] (metaoof) {Meta-OOF Probabilities};
\node[box, below=0.5cm of metaoof] (thresh) {Threshold Optimization};
\node[box, below=0.5cm of thresh, fill=blue!15] (final) {Final Logistic Regression\\Fitted on Full $X_{\text{meta\_train}}$};
\node[box, below=0.5cm of final] (xmetatest) {$X_{\text{meta\_test}}$};
\node[box, below=0.5cm of xmetatest] (eval2) {Independent Test Evaluation};

\draw[arrow] (base) -- (xmeta);
\draw[arrow] (xmeta) -- (cross);
\draw[arrow] (cross) -- (metaoof);
\draw[arrow] (metaoof) -- (thresh);
\draw[arrow] (thresh) -- (final);
\draw[arrow] (final) -- (xmetatest);
\draw[arrow] (xmetatest) -- (eval2);
\draw[arrow] (compare.east) -- ++(0.8cm,0) |- (base.west);
\end{tikzpicture}
\caption{Complete workflow of the proposed explainable diabetes risk-prediction framework: dataset inspection, deterministic data-quality cleaning, stratified split, test-set isolation, training-only learned preprocessing, comparative model development (with SMOTENC and Optuna branches), base-level OOF stacking with meta-level cross-fitting, meta-OOF threshold selection, final Logistic Regression fit on full $X_{\text{meta\_train}}$, independent test evaluation, and SHAP explainability.}
\label{fig:workflow}
\end{figure*}"""

fig_pattern = r'\\begin\{figure\*\}[\s\S]*?\\caption\{[\s\S]*?\\end\{figure\*\}'
content = re.sub(fig_pattern, lambda m: figure1_tikz, content, count=1)

# -------------------------------------------------------------------
# 2. Delete paper/figure1_workflow.tex if present to prevent accidental include
# -------------------------------------------------------------------
if os.path.exists('paper/figure1_workflow.tex'):
    os.remove('paper/figure1_workflow.tex')

# -------------------------------------------------------------------
# 3. Clean up malformed LaTeX & Tab/Slash issues
# -------------------------------------------------------------------
content = content.replace('meta-meta-OOF', 'meta-OOF')
content = content.replace('Meta-meta-OOF', 'Meta-OOF')
content = content.replace('SMOTENCNC', 'SMOTENC')

# Fix Hyperparameters table row 451
old_mlp_row = r'\textbf{MLP Tuned} & hidden\_layer\_sizes: (128,), activation: relu, alpha: 0.004, learning\_rate_init: 0.006, batch\_size: 128, max\_iter: 406 \\'
new_mlp_row = r'\textbf{MLP Tuned} & hidden\_layer\_sizes: (128,), activation: relu, alpha: 0.004, learning\_rate_init: 0.006, batch\_size: 128, max\_iter: 406, early_stopping: False \\'
content = content.replace(old_mlp_row, new_mlp_row)

if 'early_stopping' not in content[content.find('MLP Tuned'):content.find('MLP Tuned')+200]:
    content = re.sub(
        r'(\\textbf\{MLP Tuned\}[\s\S]*?max\_iter:\s*406)(\s*\\\\)',
        r'\1, early_stopping: False\2',
        content
    )

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Surgical updates completed.")
