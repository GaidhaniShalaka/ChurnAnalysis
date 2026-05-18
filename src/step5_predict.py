"""
STEP 5 — PREDICT (INFERENCE)
==============================
Goal: Load trained model, run predictions on new data.
Usage:
    python src/step5_predict.py                          # runs on sample rows
    python src/step5_predict.py path/to/new_data.csv    # runs on a file
"""

import pandas as pd
import numpy as np
import joblib, sys, os

MODEL_PATH    = "models/best_model.pkl"
FEATURES_PATH = "data/processed/features.csv"


def load_model():
    artifact = joblib.load(MODEL_PATH)
    print(f"[model] Loaded: {artifact['model_name']}")
    return artifact["model"], artifact["feature_cols"]


def prepare_input(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Align a raw DataFrame to the exact columns the model expects."""
    # Add any missing columns as 0 (handles one-hot columns not in new data)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


def predict(model, X: pd.DataFrame) -> pd.DataFrame:
    proba  = model.predict_proba(X)[:, 1]
    labels = (proba >= 0.5).astype(int)

    results = pd.DataFrame({
        "churn_probability": proba.round(4),
        "churn_predicted":   labels,
        "risk_tier": pd.cut(proba, bins=[0, 0.2, 0.5, 0.8, 1.0],
                            labels=["Low", "Medium", "High", "Critical"])
    })
    return results


def run(input_path: str = None):
    print("=" * 50)
    print("STEP 5: PREDICT")
    print("=" * 50)

    model, feature_cols = load_model()

    if input_path:
        df = pd.read_csv(input_path)
        print(f"[input] {len(df)} rows from {input_path}")
    else:
        # Use last 10 rows of the feature file as demo
        df = pd.read_csv(FEATURES_PATH).tail(10).drop(columns=["churned"], errors="ignore")
        print("[input] Using 10 sample rows from features.csv")

    X       = prepare_input(df.copy(), feature_cols)
    results = predict(model, X)

    output  = pd.concat([df[["logins_per_month", "plan_rank", "engagement_score"]].reset_index(drop=True),
                         results], axis=1)

    print("\nPredictions:")
    print(output.to_string(index=False))

    out_path = "data/processed/predictions.csv"
    output.to_csv(out_path, index=False)
    print(f"\n[save] Predictions → {out_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run(path)
