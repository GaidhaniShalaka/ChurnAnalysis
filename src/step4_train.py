"""
STEP 4 — MODEL TRAINING & EVALUATION
======================================
Goal: Train, compare, and select best model. Save it for serving.
Output: models/best_model.pkl  +  models/metrics.json
"""

import pandas as pd
import numpy as np
import json, os, joblib

from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics           import (
    classification_report, roc_auc_score,
    precision_recall_curve, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES_PATH = "data/processed/features.csv"
COLS_PATH     = "data/processed/feature_columns.txt"
MODEL_PATH    = "models/best_model.pkl"
METRICS_PATH  = "models/metrics.json"
PLOT_DIR      = "notebooks/eda_plots"
os.makedirs("models", exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ── Load ──────────────────────────────────────────────────────────────────────

def load() -> tuple:
    df = pd.read_csv(FEATURES_PATH)
    with open(COLS_PATH) as f:
        feature_cols = f.read().splitlines()

    X = df[feature_cols]
    y = df["churned"]
    print(f"[load]  X: {X.shape}   Churn rate: {y.mean():.1%}")
    return X, y, feature_cols


# ── Train / test split ────────────────────────────────────────────────────────

def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[split] Train: {len(X_train)}  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ── Model candidates ──────────────────────────────────────────────────────────

def build_candidates():
    """Each candidate is a sklearn Pipeline: scaler → model."""
    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                              random_state=42, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                   max_depth=4, random_state=42))
        ]),
    }
    return candidates


# ── Compare models with cross-validation ─────────────────────────────────────

def compare(candidates: dict, X_train, y_train) -> str:
    print("\n" + "=" * 50)
    print("MODEL COMPARISON  (5-fold cross-validation on train set)")
    print("=" * 50)

    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_auc = None, 0

    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        mean, std = scores.mean(), scores.std()
        bar = "█" * int(mean * 30)
        print(f"  {name:<25}  AUC {mean:.4f} ± {std:.4f}   {bar}")
        if mean > best_auc:
            best_auc, best_name = mean, name

    print(f"\n  → Best model: {best_name}  (AUC {best_auc:.4f})")
    return best_name


# ── Final evaluation ──────────────────────────────────────────────────────────

def evaluate(name: str, pipe, X_train, X_test, y_train, y_test, feature_cols):
    pipe.fit(X_train, y_train)
    y_pred      = pipe.predict(X_test)
    y_proba     = pipe.predict_proba(X_test)[:, 1]
    auc         = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 50)
    print(f"FINAL EVALUATION — {name}")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print(f"ROC-AUC: {auc:.4f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns_heatmap_data = pd.DataFrame(cm, index=["Retained","Churned"], columns=["Pred Retained","Pred Churned"])
    ax.matshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticklabels(["", "Pred Retained", "Pred Churned"])
    ax.set_yticklabels(["", "Actual Retained", "Actual Churned"])
    ax.set_title(f"Confusion Matrix — {name}", pad=15)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/confusion_matrix.png", dpi=120)
    plt.close()

    # Feature importance (if tree-based)
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = pd.Series(clf.feature_importances_, index=feature_cols)
        top = importances.nlargest(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        top.sort_values().plot(kind="barh", ax=ax, color="#1565C0")
        ax.set_title("Top 15 Feature Importances", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/feature_importance.png", dpi=120)
        plt.close()
        print(f"\n[plot] Feature importance saved")

    print(f"[plot] Confusion matrix saved")
    return auc, y_pred


# ── Save ──────────────────────────────────────────────────────────────────────

def save_model(pipe, name: str, auc: float, feature_cols: list):
    artifact = {"model": pipe, "feature_cols": feature_cols, "model_name": name}
    joblib.dump(artifact, MODEL_PATH)
    metrics = {"model": name, "roc_auc": round(auc, 4)}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[save] Model → {MODEL_PATH}")
    print(f"[save] Metrics → {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))


# ── Run ───────────────────────────────────────────────────────────────────────

def run():
    print("=" * 50)
    print("STEP 4: MODEL TRAINING")
    print("=" * 50)
    X, y, feature_cols          = load()
    X_train, X_test, y_train, y_test = split(X, y)
    candidates                  = build_candidates()
    best_name                   = compare(candidates, X_train, y_train)
    best_pipe                   = candidates[best_name]
    auc, _                      = evaluate(best_name, best_pipe, X_train, X_test, y_train, y_test, feature_cols)
    save_model(best_pipe, best_name, auc, feature_cols)


if __name__ == "__main__":
    run()
