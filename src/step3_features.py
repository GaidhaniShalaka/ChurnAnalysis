"""
STEP 3 — FEATURE ENGINEERING
==============================
Goal: Transform cleaned data into ML-ready features.
Output: data/processed/features.csv  +  data/processed/feature_columns.txt
"""

import pandas as pd
import numpy as np
import os

CLEANED_PATH  = "data/processed/cleaned.csv"
FEATURES_PATH = "data/processed/features.csv"
COLS_PATH     = "data/processed/feature_columns.txt"


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["signup_date", "churn_date"])


# ── Numeric features ──────────────────────────────────────────────────────────

def engineer_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Spend per feature used (value density)
    df["spend_per_feature"] = (df["monthly_spend_usd"] / (df["features_used"] + 1)).round(2)

    # Tickets per month (normalised support load)
    df["tickets_per_month"] = (df["support_tickets_last_90d"] / 3).round(3)

    # Login frequency bucket: 0=inactive, 1=low, 2=medium, 3=high
    df["login_bucket"] = pd.cut(
        df["logins_per_month"],
        bins=[-1, 2, 8, 20, 999],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Recency risk: days since login / tenure_days
    df["recency_ratio"] = (df["days_since_last_login"] / (df["tenure_days"] + 1)).round(4)

    # NPS sentiment bucket
    df["nps_bucket"] = pd.cut(
        df["nps_score"],
        bins=[-1, 6, 8, 10],
        labels=["detractor", "passive", "promoter"]
    ).astype(str)

    print("[numeric] spend_per_feature, tickets_per_month, login_bucket, recency_ratio, nps_bucket added")
    return df


# ── Categorical encoding ──────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # Ordinal: plan tier (meaningful order)
    plan_order = {"free": 0, "basic": 1, "pro": 2, "enterprise": 3}
    df["plan_rank"] = df["plan"].map(plan_order).fillna(1)

    # One-hot encode low-cardinality categoricals
    ohe_cols = ["acquisition_channel", "industry", "support_tier", "nps_bucket"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=int)

    print(f"[encode] plan_rank + one-hot for {ohe_cols}")
    return df


# ── Drop columns not used for modelling ──────────────────────────────────────

DROP_COLS = [
    "customer_id", "name", "email",          # identifiers / PII
    "signup_date", "churn_date",              # raw dates (already encoded as tenure)
    "plan",                                   # replaced by plan_rank
    "country",                                # too many levels for demo; add back with target-encoding later
    "contract_length_months",                 # captured by is_annual_contract
]

def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing)
    print(f"[drop]   Removed {len(existing)} non-feature columns")
    return df


# ── Save ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame):
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)

    feature_cols = [c for c in df.columns if c != "churned"]
    with open(COLS_PATH, "w") as f:
        f.write("\n".join(feature_cols))

    print(f"\n[save] Features → {FEATURES_PATH}")
    print(f"[save] Feature list → {COLS_PATH}")
    print(f"       {len(feature_cols)} features  |  target: churned")
    print(f"\nFinal feature columns:\n  " + "\n  ".join(feature_cols))


def run():
    print("=" * 50)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 50)
    df = load(CLEANED_PATH)
    df = engineer_numeric(df)
    df = encode_categoricals(df)
    df = drop_unused(df)
    save(df)
    return df


if __name__ == "__main__":
    run()
