"""
Churn Prediction — Fully Numerical Data Merge Pipeline
======================================================
Merges all 6 source tables into a single, purely numerical DataFrame.
- Converts all dates to "days since" or "tenure".
- One-Hot Encodes all categorical variables (creates 0/1 columns).
- Computes advanced ratios (failure rates, spend per generation).
- Fills missing counts/sums with 0.

Usage:
    python prepare_churn_data.py
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Set Working Directory
try:
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hack NU", "Train data"))
except:
    pass


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_csv(path: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, **kwargs)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    return df.drop(columns=unnamed)


def safe_parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def days_since(ts: pd.Series, ref: pd.Timestamp) -> pd.Series:
    return (ref - ts).dt.total_seconds() / 86400


# ---------------------------------------------------------------------------
# 1. BASE: Users
# ---------------------------------------------------------------------------

def load_users(path: str) -> pd.DataFrame:
    df = load_csv(path)
    label_map = {"not_churned": 0, "vol_churn": 1, "invol_churn": 2}

    if "churn_status" in df.columns:
        df["churn_label"] = df["churn_status"].map(label_map)
        df = df.drop(columns=["churn_status"])

    print(f"[Users] Loaded {len(df):,} rows.")
    return df


# ---------------------------------------------------------------------------
# 2. PROPERTIES
# ---------------------------------------------------------------------------

def process_properties(path: str, ref_date: pd.Timestamp) -> pd.DataFrame:
    df = load_csv(path)
    df["subscription_start_date"] = safe_parse_dates(df["subscription_start_date"])
    df["subscription_tenure_days"] = days_since(df["subscription_start_date"], ref_date)

    # One-Hot Encode Categoricals (Plan & Country)
    df = pd.get_dummies(df, columns=["subscription_plan", "country_code"], dummy_na=True)
    df = df.drop(columns=["subscription_start_date"])
    print(f"[Properties] Processed {len(df):,} rows.")
    return df


# ---------------------------------------------------------------------------
# 3. QUIZZES
# ---------------------------------------------------------------------------

def process_quizzes(path: str) -> pd.DataFrame:
    df = load_csv(path)
    df = df.sort_index().groupby("user_id", as_index=False).last()

    # Calculate completeness
    survey_cols = ["source", "flow_type", "team_size", "experience",
                   "usage_plan", "frustration", "first_feature", "role"]
    df["quiz_completeness"] = df[survey_cols].notna().mean(axis=1)

    # One-Hot Encode all quiz text answers
    df = pd.get_dummies(df, columns=survey_cols, dummy_na=True)
    print(f"[Quizzes] Processed {len(df):,} rows.")
    return df


# ---------------------------------------------------------------------------
# 4. PURCHASES
# ---------------------------------------------------------------------------

def process_purchases(path: str, ref_date: pd.Timestamp):
    df = load_csv(path)
    df["purchase_time"] = safe_parse_dates(df["purchase_time"])

    # One-Hot Encode purchase types
    type_dummies = pd.get_dummies(df["purchase_type"], prefix="purch_type")
    df = pd.concat([df, type_dummies], axis=1)

    type_cols = type_dummies.columns.tolist()

    # Aggregate per user
    agg_funcs = {
        "transaction_id": "count",
        "purchase_amount_dollars": ["sum", "max", "mean"],
        "purchase_time": ["min", "max"]
    }
    for col in type_cols:
        agg_funcs[col] = "sum"

    purch_agg = df.groupby("user_id").agg(agg_funcs).reset_index()

    # Flatten multi-level columns
    purch_agg.columns = ["user_id", "n_purchases", "total_spend_usd", "max_single_purchase_usd",
                         "avg_purchase_usd", "first_purchase_time", "last_purchase_time"] + type_cols

    # Time features
    purch_agg["days_since_last_purchase"] = days_since(purch_agg["last_purchase_time"], ref_date)
    purch_agg["purchase_tenure_days"] = days_since(purch_agg["first_purchase_time"], ref_date)

    purch_agg["avg_days_between_purchases"] = np.where(
        purch_agg["n_purchases"] > 1,
        purch_agg["purchase_tenure_days"] / (purch_agg["n_purchases"] - 1),
        0
    )

    purch_agg = purch_agg.drop(columns=["last_purchase_time", "first_purchase_time"])

    # Mapping for Attempts table
    user_txn_map = df[["user_id", "transaction_id"]].dropna().drop_duplicates()

    print(f"[Purchases] Aggregated into {len(purch_agg):,} users.")
    return purch_agg, user_txn_map


# ---------------------------------------------------------------------------
# 5. ATTEMPTS
# ---------------------------------------------------------------------------

def process_attempts(path: str, user_txn_map: pd.DataFrame) -> pd.DataFrame:
    df = load_csv(path)

    # Join to get user_id
    df = df.merge(user_txn_map, on="transaction_id", how="inner")

    # Convert bools properly
    for col in ["is_prepaid", "is_virtual", "is_business"]:
        if col in df.columns:
            df[col] = df[col].replace({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)

    df["attempt_failed"] = df["failure_code"].notna().astype(int)
    df["attempt_success"] = 1 - df["attempt_failed"]
    df["card_declined"] = (df["failure_code"] == "card_declined").astype(int)

    # Aggregate
    att_agg = df.groupby("user_id").agg(
        n_transaction_attempts=("transaction_id", "count"),
        n_failed_attempts=("attempt_failed", "sum"),
        n_successful_attempts=("attempt_success", "sum"),
        n_card_declined=("card_declined", "sum"),
        total_attempted_usd=("amount_in_usd", "sum"),
        is_prepaid_card=("is_prepaid", "max"),
        is_virtual_card=("is_virtual", "max"),
        is_business_card=("is_business", "max"),
    ).reset_index()

    # Ratios
    att_agg["payment_failure_rate"] = att_agg["n_failed_attempts"] / att_agg["n_transaction_attempts"]

    print(f"[Attempts] Aggregated into {len(att_agg):,} users.")
    return att_agg


# ---------------------------------------------------------------------------
# 6. GENERATIONS
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6. GENERATIONS (Memory Safe Chunking)
# ---------------------------------------------------------------------------

def process_generations(path: str, ref_date: pd.Timestamp, chunk_size: int = 250_000) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[Generations] WARNING: {path} not found. Skipping generations.")
        return pd.DataFrame(columns=["user_id"])

    print(f"[Generations] Large file detected. Reading in memory-safe chunks of {chunk_size:,} rows...")

    # Check if 'generation_type' column exists by peeking at the first row
    sample_df = pd.read_csv(path, nrows=1)
    has_gen_type = "generation_type" in sample_df.columns

    chunk_aggs = []

    # Process the file chunk by chunk
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        # 1. Parse dates and booleans for this chunk
        chunk["created_at"] = safe_parse_dates(chunk["created_at"])
        chunk["is_success"] = (chunk["status"] == "success").astype(int)
        chunk["is_failed"] = (chunk["status"] == "failed").astype(int)
        chunk["is_nsfw"] = (chunk["status"] == "nsfw").astype(int)

        type_cols = []
        if has_gen_type:
            type_dummies = pd.get_dummies(chunk["generation_type"], prefix="gen_type")
            chunk = pd.concat([chunk, type_dummies], axis=1)
            type_cols = type_dummies.columns.tolist()

        # 2. Setup aggregations for this chunk
        agg_funcs = {
            "generation_id": "count",
            "is_success": "sum",
            "is_failed": "sum",
            "is_nsfw": "sum",
            "created_at": ["min", "max"]
        }
        for col in type_cols:
            agg_funcs[col] = "sum"

        # 3. Aggregate just this chunk
        chunk_agg = chunk.groupby("user_id").agg(agg_funcs).reset_index()

        # Flatten columns
        chunk_agg.columns = ["user_id", "n_generations", "n_gen_completed", "n_gen_failed", "n_gen_nsfw",
                             "first_gen_time", "last_gen_time"] + type_cols

        chunk_aggs.append(chunk_agg)

    print("[Generations] Finished reading chunks. Merging intermediate results...")

    # Concatenate all aggregated chunks
    full_df = pd.concat(chunk_aggs, ignore_index=True)

    # Second pass aggregation: Combine users that were split across different chunks
    final_agg_funcs = {
        "n_generations": "sum",
        "n_gen_completed": "sum",
        "n_gen_failed": "sum",
        "n_gen_nsfw": "sum",
        "first_gen_time": "min",
        "last_gen_time": "max"
    }

    # Make sure we grab all one-hot encoded gen_types from all chunks
    final_type_cols = [c for c in full_df.columns if c.startswith("gen_type_")]
    for col in final_type_cols:
        full_df[col] = full_df[col].fillna(0)  # Fill NaNs for chunks missing a type
        final_agg_funcs[col] = "sum"

    gen_agg = full_df.groupby("user_id").agg(final_agg_funcs).reset_index()

    # Calculate Ratios and time based on the absolute final merged data
    gen_agg["gen_success_rate"] = gen_agg["n_gen_completed"] / gen_agg["n_generations"]
    gen_agg["days_since_last_generation"] = days_since(gen_agg["last_gen_time"], ref_date)
    gen_agg["generation_tenure_days"] = days_since(gen_agg["first_gen_time"], ref_date)

    gen_agg = gen_agg.drop(columns=["last_gen_time", "first_gen_time"])

    print(f"[Generations] Successfully Aggregated into {len(gen_agg):,} users.")
    return gen_agg


# ---------------------------------------------------------------------------
# MAIN MERGE
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", default="train_users.csv")
    parser.add_argument("--properties", default="train_users_properties.csv")
    parser.add_argument("--quizzes", default="train_users_quizzes.csv")
    parser.add_argument("--purchases", default="train_users_purchases.csv")
    parser.add_argument("--attempts", default="train_users_transaction_attempts.csv")
    parser.add_argument("--generations", default="train_users_generations.csv")
    parser.add_argument("--out", default="merged_churn_dataset_numeric.csv")
    args = parser.parse_args()

    # Anchor date for computing "days since"
    # We will use a hardcoded reference in the future so train and test match perfectly
    # For now, let's use a fixed far-future date to represent "now" in the dataset timeline
    REF_DATE = pd.to_datetime("1068-01-01", utc=True)

    # 1. Load users
    df = load_users(args.users)

    # 2. Merge Properties
    df_props = process_properties(args.properties, REF_DATE)
    df = df.merge(df_props, on="user_id", how="left")

    # 3. Merge Quizzes
    df_quizzes = process_quizzes(args.quizzes)
    df = df.merge(df_quizzes, on="user_id", how="left")

    # 4. Merge Purchases
    df_purchases, user_txn_map = process_purchases(args.purchases, REF_DATE)
    df = df.merge(df_purchases, on="user_id", how="left")

    # 5. Merge Attempts
    df_attempts = process_attempts(args.attempts, user_txn_map)
    df = df.merge(df_attempts, on="user_id", how="left")

    # 6. Merge Generations
    df_generations = process_generations(args.generations, REF_DATE)
    df = df.merge(df_generations, on="user_id", how="left")

    print("\n[Merging] All tables joined.")

    # -----------------------------------------------------------------------
    # FINAL NUMERICAL CLEANUP & CROSS-FEATURE ENGINEERING
    # -----------------------------------------------------------------------

    # Fill missing values for counts, sums, and binary flags with 0
    # For dates/tenure, we can fill with -1 or 0. We'll use 0.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Ensure booleans (from get_dummies) are int
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Cross-table Engineering
    # 1. Spend per generation
    df["spend_per_generation"] = np.where(
        df["n_generations"] > 0,
        df["total_spend_usd"] / df["n_generations"],
        0
    )

    # 2. Spend per purchase (Average order value overall)
    df["spend_per_purchase_overall"] = np.where(
        df["n_purchases"] > 0,
        df["total_spend_usd"] / df["n_purchases"],
        0
    )

    # 3. Binary Engagement Flags
    df["has_made_purchase"] = (df["n_purchases"] > 0).astype(int)
    df["has_made_generation"] = (df["n_generations"] > 0).astype(int)
    df["has_failed_payment"] = (df["n_failed_attempts"] > 0).astype(int)

    # Final Check - Drop any remaining non-numeric columns except user_id
    cols_to_keep = ["user_id"] + [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df = df[cols_to_keep]

    print(f"\nFinal Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("Sample Columns:", df.columns[:10].tolist())

    df.to_csv(args.out, index=False)
    print(f"\nSuccessfully saved fully numerical dataset to: {args.out}")


if __name__ == "__main__":
    main()