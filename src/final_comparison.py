"""
Final Model Comparison
----------------------

Creates separate publication-ready comparison plots for:
    - ROC-AUC
    - PR-AUC
    - F1
    - Recall
    - Precision

Also saves a ranked model table.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR


# ==========================================================
# Paths
# ==========================================================

INPUT_FILE = (
    RESULTS_DIR /
    "comparison" /
    "model_comparison.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR /
    "comparison" /
    "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# Load Results
# ==========================================================

print("=" * 60)
print("Loading Final Model Comparison")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE
)


# ==========================================================
# Create Labels
# ==========================================================

df["Label"] = (
    df["Model"]
    + " ("
    + df["Experiment"]
    + ")"
)


# ==========================================================
# Metrics
# ==========================================================

metrics = [

    (
        "roc_auc",
        "ROC-AUC",
        "roc_auc_comparison.png",
    ),

    (
        "pr_auc",
        "PR-AUC",
        "pr_auc_comparison.png",
    ),

    (
        "f1",
        "F1 Score",
        "f1_comparison.png",
    ),

    (
        "recall",
        "Recall",
        "recall_comparison.png",
    ),

    (
        "precision",
        "Precision",
        "precision_comparison.png",
    ),
]


# ==========================================================
# Create Ranked Table
# ==========================================================

ranking = (
    df
    .sort_values(
        by="roc_auc",
        ascending=False,
    )
    .reset_index(drop=True)
)

ranking.insert(
    0,
    "ROC_AUC_Rank",
    range(
        1,
        len(ranking) + 1,
    ),
)


# ==========================================================
# Save Ranking
# ==========================================================

ranking_file = (
    OUTPUT_DIR /
    "final_model_ranking.csv"
)

ranking.to_csv(
    ranking_file,
    index=False,
)


# ==========================================================
# Create Comparison Figures
# ==========================================================

for (
    metric,
    title,
    filename,
) in metrics:

    print(
        f"\nCreating {title} comparison..."
    )

    plot_df = (
        df
        .sort_values(
            by=metric,
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_df["Label"],
        plot_df[metric],
    )

    plt.xlabel(
        title
    )

    plt.ylabel(
        "Model / Experiment"
    )

    plt.title(
        f"Final Model Comparison - {title}"
    )

    plt.xlim(
        0,
        1.0,
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR /
        filename
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved to:\n"
        f"{output_file}"
    )


# ==========================================================
# Best Model
# ==========================================================

best_model = df.loc[
    df["roc_auc"].idxmax()
]


# ==========================================================
# Print Final Ranking
# ==========================================================

print("\n" + "=" * 60)
print("FINAL MODEL RANKING")
print("=" * 60)

print(
    ranking[
        [
            "ROC_AUC_Rank",
            "Model",
            "Experiment",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ]
    ].to_string(
        index=False
    )
)


# ==========================================================
# Best Model Summary
# ==========================================================

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"Model      : "
    f"{best_model['Model']}"
)

print(
    f"Experiment : "
    f"{best_model['Experiment']}"
)

print(
    f"ROC-AUC    : "
    f"{best_model['roc_auc']:.4f}"
)

print(
    f"PR-AUC     : "
    f"{best_model['pr_auc']:.4f}"
)

print(
    f"F1         : "
    f"{best_model['f1']:.4f}"
)

print(
    f"Recall     : "
    f"{best_model['recall']:.4f}"
)

print(
    f"Precision  : "
    f"{best_model['precision']:.4f}"
)


# ==========================================================
# Completed
# ==========================================================

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON COMPLETED")
print("=" * 60)

print(
    f"\nFigures saved to:\n"
    f"{OUTPUT_DIR}"
)

print(
    f"\nRanking saved to:\n"
    f"{ranking_file}"
)