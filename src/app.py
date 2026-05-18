"""
STEP 6 — API (DEPLOYMENT)
===========================
A FastAPI app that serves the trained churn model.
Run locally:  uvicorn src.app:app --reload
Deploy:       Docker → Render / Railway / Hugging Face Spaces
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib, os
import pandas as pd
import numpy as np

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")

try:
    artifact     = joblib.load(MODEL_PATH)
    MODEL        = artifact["model"]
    FEATURE_COLS = artifact["feature_cols"]
    MODEL_NAME   = artifact["model_name"]
    print(f"[startup] Model loaded: {MODEL_NAME}")
except Exception as e:
    print(f"[startup] WARNING: Could not load model: {e}")
    MODEL, FEATURE_COLS, MODEL_NAME = None, [], "none"

app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn probability from customer features.",
    version="1.0.0",
)

# ── Request / Response schemas ────────────────────────────────────────────────

class CustomerInput(BaseModel):
    """All fields the model needs. Defaults make it easy to test."""
    tenure_days:               int     = Field(365,  description="Days since signup")
    monthly_spend_usd:         float   = Field(99.0, description="Monthly spend in USD")
    logins_per_month:          int     = Field(10,   description="Average logins per month")
    features_used:             int     = Field(8,    description="Number of features used")
    support_tickets_last_90d:  int     = Field(1,    description="Support tickets in last 90 days")
    days_since_last_login:     int     = Field(5,    description="Days since most recent login")
    nps_score:                 float   = Field(7.0,  description="NPS score 0-10")
    team_size:                 float   = Field(10.0, description="Company team size")
    plan:                      str     = Field("pro", description="free | basic | pro | enterprise")
    acquisition_channel:       str     = Field("organic")
    industry:                  str     = Field("saas")
    support_tier:              str     = Field("email")
    is_annual_contract:        int     = Field(1,    description="1 if annual contract, else 0")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure_days": 180, "monthly_spend_usd": 29.0,
                "logins_per_month": 2, "features_used": 3,
                "support_tickets_last_90d": 4, "days_since_last_login": 45,
                "nps_score": 4.0, "team_size": 5.0,
                "plan": "basic", "acquisition_channel": "paid search",
                "industry": "retail", "support_tier": "none",
                "is_annual_contract": 0
            }
        }


class PredictionOutput(BaseModel):
    churn_probability: float
    churn_predicted:   bool
    risk_tier:         str
    model_used:        str


# ── Helper: raw input → model features ───────────────────────────────────────

PLAN_MAP = {"free": 0, "basic": 1, "pro": 2, "enterprise": 3}

def build_features(inp: CustomerInput) -> pd.DataFrame:
    d = inp.dict()

    # Derived features (mirrors step3_features.py)
    d["tenure_months"]       = round(d["tenure_days"] / 30, 1)
    d["spend_per_feature"]   = round(d["monthly_spend_usd"] / (d["features_used"] + 1), 2)
    d["tickets_per_month"]   = round(d["support_tickets_last_90d"] / 3, 3)
    d["login_bucket"]        = int(pd.cut([d["logins_per_month"]], bins=[-1,2,8,20,999], labels=[0,1,2,3])[0])
    d["recency_ratio"]       = round(d["days_since_last_login"] / (d["tenure_days"] + 1), 4)
    d["engagement_score"]    = round(
        (min(d["logins_per_month"], 30) / 30) * 50
        + (min(d["features_used"], 20) / 20) * 30
        + (d["nps_score"] / 10) * 20, 2
    )
    d["plan_rank"]           = PLAN_MAP.get(d["plan"].lower(), 1)
    d["nps_missing"]         = 0

    # NPS bucket one-hot
    nps = d["nps_score"]
    d["nps_bucket_detractor"] = int(nps < 7)
    d["nps_bucket_passive"]   = int(7 <= nps <= 8)
    d["nps_bucket_promoter"]  = int(nps >= 9)

    # Acquisition channel one-hot
    for ch in ["email", "organic", "paid search", "referral", "social"]:
        d[f"acquisition_channel_{ch}"] = int(d["acquisition_channel"].lower() == ch)

    # Industry one-hot
    for ind in ["education", "finance", "healthcare", "other", "retail", "saas", "unknown"]:
        d[f"industry_{ind}"] = int(d["industry"].lower() == ind)

    # Support tier one-hot
    for tier in ["dedicated", "email", "none", "priority"]:
        d[f"support_tier_{tier}"] = int(d["support_tier"].lower() == tier)

    df = pd.DataFrame([d])
    # Align to model's expected columns
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_COLS]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionOutput, summary="Predict churn for one customer")
def predict(customer: CustomerInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X     = build_features(customer)
    proba = float(MODEL.predict_proba(X)[0, 1])
    label = proba >= 0.5

    if proba < 0.2:   tier = "Low"
    elif proba < 0.5: tier = "Medium"
    elif proba < 0.8: tier = "High"
    else:             tier = "Critical"

    return PredictionOutput(
        churn_probability=round(proba, 4),
        churn_predicted=label,
        risk_tier=tier,
        model_used=MODEL_NAME,
    )


@app.post("/predict/batch", summary="Predict churn for multiple customers")
def predict_batch(customers: list[CustomerInput]):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return [predict(c) for c in customers]
