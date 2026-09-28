#!/usr/bin/env python3
"""Targeted text cleanups in paper/mainnnn.tex"""

with open('paper/mainnnn.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix Note line at ~239
bad_note = '\textbf{Note:} The Level-0 stacking MLP uses \texttt{hidden\\_layer\\_sizes=(128, 64)} with \texttt{early\\_stopping=False}, which is a separate configuration from the standalone tuned MLP \texttt{(128,)}, used in the individual model comparison.'
# Let's match whatever is on line 239 by regex
import re
content = re.sub(
    r'.*?The Level-0 stacking MLP uses.*?used in the individual model comparison\.',
    r'\\textbf{Note:} The Level-0 stacking MLP uses \\texttt{hidden\\_layer\\_sizes=(128, 64)} with \\texttt{early\\_stopping=False}, which is a separate configuration from the standalone tuned MLP \\texttt{(128,)}, used in the individual model comparison.',
    content
)

# Fix SMOTENC in section 3.5 & 4.3 & conclusion
content = content.replace(
    'Synthetic Minority Over-sampling Technique (SMOTE) is investigated as a separate experimental condition.\n\nSMOTE generates synthetic minority-class observations based on existing minority samples. Rather than simply duplicating positive observations, it creates additional synthetic samples in the feature space.\n\nIn this study, SMOTE is evaluated with XGBoost, CatBoost, and MLP.',
    'Synthetic Minority Over-sampling Technique for Nominal and Continuous (SMOTENC) is investigated as a separate experimental condition.\n\nSMOTENC generates synthetic minority-class observations based on existing minority samples. Rather than simply duplicating positive observations, it creates additional synthetic samples in the feature space while handling categorical features.\n\nIn this study, SMOTENC is evaluated with XGBoost, CatBoost, and MLP.'
)

content = content.replace(
    'An important aspect of the proposed methodology is that SMOTE is treated as an experiment',
    'An important aspect of the proposed methodology is that SMOTENC is treated as an experiment'
)

content = content.replace(
    '\\subsection{Effect of SMOTENC}\n\nSMOTE was investigated',
    '\\subsection{Effect of SMOTENC}\n\nSMOTENC was investigated'
)

content = content.replace(
    'These findings indicate that SMOTE did not provide a consistent improvement',
    'These findings indicate that SMOTENC did not provide a consistent improvement'
)

content = content.replace(
    'the experimental results do not support the conclusion that SMOTE is universally beneficial',
    'the experimental results do not support the conclusion that SMOTENC is universally beneficial'
)

content = content.replace(
    'SMOTE improved recall in selected configurations',
    'SMOTENC improved recall in selected configurations'
)

content = content.replace(
    'SMOTE-based balancing, or stacking',
    'SMOTENC-based balancing, or stacking'
)

# Check table 3 models
content = content.replace('XGBoost + SMOTE &', 'XGBoost + SMOTENC &')
content = content.replace('CatBoost + SMOTE &', 'CatBoost + SMOTENC &')
content = content.replace('MLP + SMOTE &', 'MLP + SMOTENC &')

with open('paper/mainnnn.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Targeted text cleanups complete.")
