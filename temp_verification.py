import pandas as pd
import numpy as np
from pathlib import Path
import json

def go():
    print("--- RAW DATASET ---")
    raw_path = Path("data/raw/diabetes_prediction_dataset.csv")
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        print("Raw Rows:", len(raw_df))
        print("Raw Cols:", len(raw_df.columns))
        print("Raw class counts:", dict(raw_df["diabetes"].value_counts()))
    else:
        print("Raw data not found")

    print("\n--- CLEAN DATASET ---")
    clean_path = Path("data/processed/diabetes_clean.csv")
    if clean_path.exists():
        clean_df = pd.read_csv(clean_path)
        print("Clean Rows:", len(clean_df))
        print("Clean Cols:", len(clean_df.columns))
        print("Clean class counts:", dict(clean_df["diabetes"].value_counts()))

        # Calculate exactly how many were removed
        # duplicate removal logic:
        # exact duplicates = raw_df[raw_df.duplicated()]
        # contradictory duplicates = duplicated on features but different targets
    else:
        print("Clean data not found")

    print("\n--- DATA SPLIT ---")
    from sklearn.model_selection import train_test_split
    X = clean_df.drop("diabetes", axis=1)
    y = clean_df["diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    print("X_train size:", len(X_train))
    print("X_test size:", len(X_test))
    print("y_train counts:", dict(y_train.value_counts()))
    print("y_test counts:", dict(y_test.value_counts()))

    print("\n--- MODELS METRICS ---")
    comp_file = Path("results/comparison/model_comparison.csv")
    if comp_file.exists():
        comp = pd.read_csv(comp_file)
        print(comp)

        print("\n--- CONFUSION MATRICES ---")
        P = sum(y_test == 1)
        N = sum(y_test == 0)
        for _, row in comp.iterrows():
            rec = row['recall']
            prec = row['precision']
            tp = int(round(rec * P))
            fn = P - tp
            if prec > 0:
                fp = int(round(tp / prec - tp))
            else:
                fp = 0
            tn = N - fp
            print(f"{row['Model']} ({row['Experiment']}): TP={tp}, FN={fn}, FP={fp}, TN={tn}")

    print("\n--- STACKING ENSEMBLE ---")
    threshold_final = Path("results/comparison/stacking_threshold_final_results.csv")
    if threshold_final.exists():
        th_df = pd.read_csv(threshold_final)
        print(th_df)
        for _, row in th_df.iterrows():
            rec = row['recall']
            prec = row['precision']
            tp = int(round(rec * P))
            fn = P - tp
            if prec > 0:
                fp = int(round(tp / prec - tp))
            else:
                fp = 0
            tn = N - fp
            print(f"Stacking Final: TP={tp}, FN={fn}, FP={fp}, TN={tn}")

    print("\n--- OPTIMIZATION INFO ---")
    th_results = Path("results/comparison/stacking_threshold_results.csv")
    if th_results.exists():
        all_th = pd.read_csv(th_results)
        best_row = all_th.loc[all_th["f1"].idxmax()]
        print("Best threshold row from stacking_threshold_results.csv:")
        print(best_row)

    print("\n--- SHAP ---")
    # Read shap summary if it exists
    # We can check images or data in results/shap/
    shap_dir = Path("results/shap")
    if shap_dir.exists():
        for f in shap_dir.glob("*.csv"):
            if "summary" in f.name or "importance" in f.name:
                print(f.name)
                print(pd.read_csv(f).head(10))

    print("\n--- HYPERPARAMETERS ---")
    for f in Path("models").rglob("*.json"):
        if "best_params" in f.name:
            print(f.name)
            with open(f, 'r') as fp:
                print(fp.read())

if __name__ == "__main__":
    go()
