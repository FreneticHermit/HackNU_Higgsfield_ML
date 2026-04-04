"""
Churn Prediction — Full Dataset Training with CatBoost
======================================================
Trains a high-performance CatBoost classifier on the FULL merged dataset.
Target: 0=not_churned, 1=vol_churn, 2=invol_churn

Outputs (to ./output/):
  1. catboost_churn_model.cbm       — The trained CatBoost model
  2. feature_importance_catboost.csv— Numerical importance of each feature
  3. final_model_report.txt         — Metrics (on training data) and top drivers
  4. feature_importance.png         — Visual bar chart for your presentation
"""

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    print("CatBoost is not installed. Please run: pip install catboost")
    exit()

# --- Set Working Directory ---
try:
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hack NU", "Train data"))
except:
    pass

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TARGET = "churn_label"  # 0, 1, 2
TARGET_STR = "churn_status"  # 'not_churned', etc.
LABEL_NAMES = ["not_churned", "vol_churn", "invol_churn"]
DROP_COLS = ["user_id", TARGET_STR, TARGET]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="merged_churn_dataset_numeric.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # 1. LOAD DATA
    print(f"Reading {args.data}...")
    df = pd.read_csv(args.data, low_memory=False)

    # Target Recovery
    label_map = {"not_churned": 0, "vol_churn": 1, "invol_churn": 2}
    if TARGET not in df.columns and TARGET_STR in df.columns:
        df[TARGET] = df[TARGET_STR].map(label_map)

    # 2. PREPROCESS FOR CATBOOST
    features = [c for c in df.columns if c not in DROP_COLS and not c.startswith("Unnamed")]
    train_df = df.copy()

    print(f"Training on 100% of the dataset: {len(train_df):,} rows")

    # CatBoost handles categoricals natively, but we need to identify them
    # and replace NaNs in text columns with a placeholder string.
    cat_cols = train_df[features].select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        train_df[col] = train_df[col].fillna("missing_value").astype(str)

    X_train = train_df[features]  # CatBoost works perfectly with Pandas DataFrames
    y_train = train_df[TARGET].values.astype(int)

    # 3. TRAIN CATBOOST MODEL
    print("\nTraining Final CatBoost Model...")
    print(f"Categorical features detected: {len(cat_cols)}")

    model = CatBoostClassifier(
        iterations=300,
        depth=7,
        learning_rate=0.05,
        loss_function='MultiClass',
        eval_metric='TotalF1',  # CatBoost has a built-in weighted F1 metric
        random_seed=args.seed,
        verbose=25,  # Print progress every 50 trees
        thread_count=-1,
        auto_class_weights='Balanced',
        l2_leaf_reg=5
    )

    # We pass the categorical columns directly to fit()
    model.fit(X_train, y_train, cat_features=cat_cols)

    # 4. EVALUATE ON TRAINING DATA
    # CatBoost returns predictions as shape (N, 1), so we flatten it
    y_pred = model.predict(X_train).flatten()

    f1_weighted = f1_score(y_train, y_pred, average="weighted")
    acc = accuracy_score(y_train, y_pred)

    print(f"\nTraining Results (In-Sample):")
    print(f"F1 (Weighted): {f1_weighted:.4f} | Accuracy: {acc:.4f}")

    # 5. SAVE OUTPUTS & MODEL FOR LATER PREDICTION

    # A. Save Model natively (much safer and lighter than joblib for CatBoost)
    model_path = out_dir / "catboost_churn_model.cbm"
    model.save_model(str(model_path))
    print(f"  -> Saved CatBoost model to {model_path.name} for final submission.")

    # B. Feature Importance
    # CatBoost calculates feature importance natively
    fi_df = pd.DataFrame({
        "feature": features,
        "importance": model.get_feature_importance()
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(out_dir / "feature_importance_catboost.csv", index=False)

    # C. Best Model Report
    report_text = f"""
    FINAL MODEL REPORT (CatBoost) - Trained on 100% Data
    ====================================================
    Train Weighted F1: {f1_weighted:.4f}
    Train Accuracy:    {acc:.4f}

    Classification Report (On Training Data):
    {classification_report(y_train, y_pred, target_names=LABEL_NAMES)}

    Top 10 Drivers of Churn:
    {fi_df.head(10).to_string(index=False)}
    """
    (out_dir / "final_model_report.txt").write_text(report_text)

    # D. Visuals
    plt.figure(figsize=(10, 6))
    top_fi = fi_df.head(20)
    plt.barh(top_fi["feature"][::-1], top_fi["importance"][::-1], color='coral')
    plt.title("Top 20 Indicators of User Churn (CatBoost)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png")

    print(f"\nDone! Files saved in the '{out_dir}' folder.")


if __name__ == "__main__":
    main()