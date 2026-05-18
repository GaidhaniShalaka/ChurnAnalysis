"""
TESTS — Unit tests for each pipeline step
Run with:  pytest tests/ -v
"""

import pandas as pd
import numpy as np
import pytest, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import step1_clean
import step3_features


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_sample():
    """Minimal raw DataFrame that mimics the real data schema."""
    return pd.DataFrame({
        "customer_id":              ["CUST-00001", "CUST-00002", "CUST-00001"],  # duplicate
        "name":                     ["Alice Smith", "Bob Jones", "Alice Smith"],
        "email":                    ["alice@ex.com", "bob@ex.com", "alice@ex.com"],
        "country":                  ["us", "uk", "us"],
        "industry":                 ["saas", None, "saas"],
        "acquisition_channel":      ["organic", "paid search", "organic"],
        "plan":                     ["pro", "basic", "pro"],
        "support_tier":             ["email", "none", "email"],
        "signup_date":              ["2022-01-15", "2023-06-01", "2022-01-15"],
        "churn_date":               [None, None, None],
        "tenure_days":              [700, 200, 700],
        "contract_length_months":   [12, 1, 12],
        "team_size":                [10.0, None, 10.0],
        "monthly_spend_usd":        [99.0, 29.0, 99.0],
        "logins_per_month":         [15, 2, 15],
        "features_used":            [8, 3, 8],
        "support_tickets_last_90d": [1, 4, 1],
        "days_since_last_login":    [5, 45, 5],
        "nps_score":                [8.0, None, 8.0],
        "churned":                  [0, 1, 0],
    })


# ── Step 1 tests ──────────────────────────────────────────────────────────────

def test_remove_duplicates(raw_sample):
    df = step1_clean.remove_duplicates(raw_sample)
    assert len(df) == 2, "Should remove 1 duplicate row"


def test_fix_dtypes(raw_sample):
    df = step1_clean.remove_duplicates(raw_sample)
    df = step1_clean.fix_dtypes(df)
    assert pd.api.types.is_datetime64_any_dtype(df["signup_date"])


def test_handle_missing_fills_nps(raw_sample):
    df = step1_clean.remove_duplicates(raw_sample)
    df = step1_clean.fix_dtypes(df)
    df = step1_clean.handle_missing(df)
    assert df["nps_score"].isnull().sum() == 0, "nps_score should have no nulls after imputation"


def test_handle_missing_fills_team_size(raw_sample):
    df = step1_clean.remove_duplicates(raw_sample)
    df = step1_clean.fix_dtypes(df)
    df = step1_clean.handle_missing(df)
    assert df["team_size"].isnull().sum() == 0


def test_industry_filled_unknown(raw_sample):
    df = step1_clean.remove_duplicates(raw_sample)
    df = step1_clean.fix_dtypes(df)
    df = step1_clean.handle_missing(df)
    assert "unknown" in df["industry"].values


def test_derived_features(raw_sample):
    df = step1_clean.remove_duplicates(raw_sample)
    df = step1_clean.fix_dtypes(df)
    df = step1_clean.handle_missing(df)
    df = step1_clean.add_derived_features(df)
    assert "engagement_score" in df.columns
    assert "tenure_months" in df.columns
    assert df["engagement_score"].between(0, 100).all()


# ── Step 3 tests ──────────────────────────────────────────────────────────────

@pytest.fixture
def cleaned_sample(raw_sample):
    df = step1_clean.remove_duplicates(raw_sample)
    df = step1_clean.fix_dtypes(df)
    df = step1_clean.handle_missing(df)
    df = step1_clean.add_derived_features(df)
    return df


def test_plan_rank_range(cleaned_sample):
    df = step3_features.engineer_numeric(cleaned_sample)
    df = step3_features.encode_categoricals(df)
    assert df["plan_rank"].between(0, 3).all()


def test_no_negative_spend_per_feature(cleaned_sample):
    df = step3_features.engineer_numeric(cleaned_sample)
    assert (df["spend_per_feature"] >= 0).all()


def test_drop_pii(cleaned_sample):
    df = step3_features.engineer_numeric(cleaned_sample)
    df = step3_features.encode_categoricals(df)
    df = step3_features.drop_unused(df)
    for col in ["name", "email", "customer_id"]:
        assert col not in df.columns, f"PII column {col} should be dropped"


def test_no_nulls_after_features(cleaned_sample):
    df = step3_features.engineer_numeric(cleaned_sample)
    df = step3_features.encode_categoricals(df)
    df = step3_features.drop_unused(df)
    assert df.isnull().sum().sum() == 0, "No nulls should remain after feature engineering"
