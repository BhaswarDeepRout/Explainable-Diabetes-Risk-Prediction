import re

with open("paper/paper draft 2.md", "r", encoding="utf-8") as f:
    text = f.read()

latex_preamble = r"""\documentclass[conference]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{array}

\begin{document}

\title{Hybrid Deep Learning and Ensemble Learning with Explainable AI for Diabetes Risk Stratification}

\author{\IEEEauthorblockN{Anonymous Authors}
\IEEEauthorblockA{\textit{Department of Research} \\
\textit{Institution}\\
City, Country \\
email@domain.edu}
}

\maketitle
"""

# Extract Abstract
abstract_match = re.search(r'## Abstract\s+(.*?)(?=\n## 1\. Introduction)', text, re.DOTALL)
abstract_text = abstract_match.group(1).strip() if abstract_match else ""

# Extract References
ref_match = re.search(r'## References\s+(.*)', text, re.DOTALL)
ref_text = ref_match.group(1).strip() if ref_match else ""
references = []
for line in ref_text.split('\n'):
    line = line.strip()
    if line:
        m = re.match(r'\[(\d+)\]\s+(.*)', line)
        if m:
            num = m.group(1)
            content = m.group(2)
            content = content.replace('&', r'\&').replace('%', r'\%')
            content = re.sub(r'\*(.*?)\*', r'\\textit{\1}', content)
            references.append((num, content))

# Extract body
body_match = re.search(r'## 1\. Introduction\s+(.*?)(?=\n## References)', text, re.DOTALL)
body_text = body_match.group(1).strip() if body_match else ""

def clean_text(t):
    t = t.replace('%', r'\%')
    t = t.replace('&', r'\&')
    t = re.sub(r'`(.*?)`', r'\\texttt{\1}', t)
    t = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', t)
    t = re.sub(r'\*(.*?)\*', r'\\textit{\1}', t)
    t = re.sub(r'\[([1-5])\]', r'\\cite{ref\1}', t)
    t = t.replace('_', r'\_')
    return t

abstract_clean = clean_text(abstract_text)

latex_doc = latex_preamble
latex_doc += r"\begin{abstract}\n".replace(r"\n", "\n")
latex_doc += abstract_clean + "\n"
latex_doc += r"\end{abstract}\n\n".replace(r"\n", "\n")
latex_doc += r"\begin{IEEEkeywords}\nDiabetes, Risk Stratification, Machine Learning, Ensemble Learning, Explainable AI, SHAP\n\end{IEEEkeywords}\n\n".replace(r"\n", "\n")

lines = body_text.split('\n')
i = 0

in_itemize = False
in_enumerate = False

while i < len(lines):
    line = lines[i]

    if "| Model | Hyperparameters |" in line:
        latex_doc += r"""\begin{table}[htbp]
\caption{Final Verified Hyperparameter Configurations}
\begin{center}
\begin{tabular}{|p{3cm}|p{5cm}|}
\hline
\textbf{Model} & \textbf{Hyperparameters} \\
\hline
"""
        i += 2
        while i < len(lines) and lines[i].startswith('|'):
            cells = [clean_text(c.strip()) for c in lines[i].split('|')[1:-1]]
            if len(cells) == 2:
                latex_doc += cells[0] + " & " + cells[1] + r" \\ \hline" + "\n"
            i += 1
        latex_doc += r"""\end{tabular}
\label{tab1}
\end{center}
\end{table}
"""
        continue

    if "| Model | Accuracy | Precision |" in line:
        latex_doc += r"""\begin{table*}[htbp]
\caption{Final Test Set Metrics}
\begin{center}
\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{ROC-AUC} & \textbf{PR-AUC} \\
\hline
"""
        i += 2
        while i < len(lines) and lines[i].startswith('|'):
            cells = [clean_text(c.strip()) for c in lines[i].split('|')[1:-1]]
            if len(cells) == 7:
                latex_doc += " & ".join(cells) + r" \\ \hline" + "\n"
            i += 1
        latex_doc += r"""\end{tabular}
\label{tab2}
\end{center}
\end{table*}
"""
        continue

    h2_match = re.match(r'^##\s+(.*)', line)
    if h2_match:
        if in_itemize: latex_doc += r"\end{itemize}\n\n".replace(r"\n", "\n"); in_itemize = False
        if in_enumerate: latex_doc += r"\end{enumerate}\n\n".replace(r"\n", "\n"); in_enumerate = False
        latex_doc += "\n" + r"\section{" + clean_text(h2_match.group(1)) + "}\n"
        i += 1
        continue

    h3_match = re.match(r'^###\s+[A-Z]\.\s+(.*)', line)
    if h3_match:
        if in_itemize: latex_doc += r"\end{itemize}\n\n".replace(r"\n", "\n"); in_itemize = False
        if in_enumerate: latex_doc += r"\end{enumerate}\n\n".replace(r"\n", "\n"); in_enumerate = False
        latex_doc += "\n" + r"\subsection{" + clean_text(h3_match.group(1)) + "}\n"
        i += 1
        continue

    if line.startswith('**TABLE'):
        i += 1
        continue

    if line.startswith('* '):
        if not in_itemize:
            latex_doc += r"\begin{itemize}\n".replace(r"\n", "\n")
            in_itemize = True
        latex_doc += r"\item ".replace(r"\n", "\n") + clean_text(line[2:]) + "\n"
        i += 1
        continue
    else:
        if in_itemize:
            latex_doc += r"\end{itemize}\n\n".replace(r"\n", "\n")
            in_itemize = False

    num_match = re.match(r'^(\d+)\.\s+(.*)', line)
    if num_match:
        if not in_enumerate:
            latex_doc += r"\begin{enumerate}\n".replace(r"\n", "\n")
            in_enumerate = True
        latex_doc += r"\item ".replace(r"\n", "\n") + clean_text(num_match.group(2)) + "\n"
        i += 1
        continue
    else:
        if in_enumerate:
            latex_doc += r"\end{enumerate}\n\n".replace(r"\n", "\n")
            in_enumerate = False

    if not line.strip():
        latex_doc += "\n"
        i += 1
        continue

    latex_doc += clean_text(line) + "\n"
    i += 1

if in_itemize: latex_doc += r"\end{itemize}\n\n".replace(r"\n", "\n")
if in_enumerate: latex_doc += r"\end{enumerate}\n\n".replace(r"\n", "\n")

latex_doc += "\n" + r"\begin{thebibliography}{00}" + "\n"
for num, content in references:
    latex_doc += r"\bibitem{ref" + num + r"} " + content + "\n"
latex_doc += r"\end{thebibliography}" + "\n"

latex_doc += r"\end{document}" + "\n"

with open("paper/final_manuscript.tex", "w", encoding="utf-8") as f:
    f.write(latex_doc)
