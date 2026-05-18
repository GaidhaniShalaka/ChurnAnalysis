"""
STEP 1 — DATA CLEANING
=======================
Goal: Load raw data, fix types, handle missing values, remove duplicates.
Output: data/processed/cleaned.csv
"""

import pandas as pd
import numpy as np
import os

RAW_PATH       = "data/raw/churn_dataset.csv"
CLEANED_PATH   = "data/processed/cleaned.csv"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  {len(df)} rows, {df.shape[1]} columns")
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # Parse dates
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["churn_date"]  = pd.to_datetime(df["churn_date"], errors="coerce")

    # Lowercase + strip whitespace on all string columns
    str_cols = df.select_dtypes("object").columns
    for col in str_cols:
        df[col] = df[col].str.strip().str.lower()

    print("[dtypes] Dates parsed, strings normalised")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="customer_id")
    print(f"[dupes]  Removed {before - len(df)} duplicate rows")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[missing] Before imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

    # nps_score  → median imputation (numerical, skewed)
    median_nps = df["nps_score"].median()
    df["nps_score"] = df["nps_score"].fillna(median_nps)
    df["nps_missing"] = df["nps_score"].isna().astype(int)   # flag before fill

    # team_size  → median by plan; fall back to global median if group is too small
    global_median = df["team_size"].median()
    df["team_size"] = df.groupby("plan")["team_size"].transform(
        lambda x: x.fillna(x.median() if not pd.isna(x.median()) else global_median)
    )
    df["team_size"] = df["team_size"].fillna(global_median)  # catch any remaining

    # industry   → fill with "unknown" (categorical)
    df["industry"] = df["industry"].fillna("unknown")

    print("\n[missing] After imputation:")
    remaining = df.isnull().sum()[df.isnull().sum() > 0]
    print(remaining.to_string() if len(remaining) else "  None — all clean!")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple features derivable from raw columns (more in step 3)."""
    # Tenure in months
    df["tenure_months"] = (df["tenure_days"] / 30).round(1)

    # Annual contract flag
    df["is_annual_contract"] = (df["contract_length_months"] >= 12).astype(int)

    # Engagement score (0-100)
    df["engagement_score"] = (
        (df["logins_per_month"].clip(0, 30) / 30) * 50
        + (df["features_used"].clip(0, 20) / 20) * 30
        + (df["nps_score"] / 10) * 20
    ).round(2)

    print("[derived] tenure_months, is_annual_contract, engagement_score added")
    return df


def save(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n[save]  Cleaned data → {path}  ({len(df)} rows, {df.shape[1]} cols)")


def run():
    print("=" * 50)
    print("STEP 1: DATA CLEANING")
    print("=" * 50)
    df = load_data(RAW_PATH)
    df = fix_dtypes(df)
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = add_derived_features(df)
    save(df, CLEANED_PATH)
    return df


if __name__ == "__main__":
    run()
