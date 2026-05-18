"""
STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
==========================================
Goal: Understand distributions, correlations, churn patterns.
Output: Printed summaries + saved plots in notebooks/eda_plots/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

CLEANED_PATH = "data/processed/cleaned.csv"
PLOT_DIR     = "notebooks/eda_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["signup_date", "churn_date"])
    print(f"[load] {len(df)} rows")
    return df


def overview(df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape          : {df.shape}")
    print(f"Churn rate     : {df['churned'].mean():.1%}")
    print(f"Date range     : {df['signup_date'].min().date()} → {df['signup_date'].max().date()}")
    print(f"\nNumeric summary:\n{df.describe(include='number').round(2).to_string()}")


def churn_by_category(df: pd.DataFrame):
    """Show churn rate for each categorical feature."""
    print("\n" + "=" * 50)
    print("CHURN RATE BY CATEGORY")
    print("=" * 50)
    cats = ["plan", "country", "acquisition_channel", "industry", "support_tier"]
    for col in cats:
        rates = df.groupby(col)["churned"].mean().sort_values(ascending=False)
        print(f"\n  {col.upper()}:")
        for k, v in rates.items():
            bar = "█" * int(v * 40)
            print(f"    {k:<18} {v:5.1%}  {bar}")


def plot_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Feature Distributions by Churn Status", fontsize=14, fontweight="bold")

    numeric_cols = [
        ("logins_per_month",      "Logins / Month"),
        ("monthly_spend_usd",     "Monthly Spend ($)"),
        ("tenure_months",         "Tenure (Months)"),
        ("nps_score",             "NPS Score"),
        ("days_since_last_login", "Days Since Last Login"),
        ("engagement_score",      "Engagement Score"),
    ]
    for ax, (col, label) in zip(axes.flat, numeric_cols):
        for churn_val, color, lbl in [(0, "#2196F3", "Retained"), (1, "#F44336", "Churned")]:
            subset = df[df["churned"] == churn_val][col].dropna()
            ax.hist(subset, bins=25, alpha=0.6, color=color, label=lbl, density=True)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_xlabel("")

    plt.tight_layout()
    path = f"{PLOT_DIR}/distributions.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plot] Distributions saved → {path}")


def plot_churn_by_plan(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Churn Analysis by Plan", fontsize=13, fontweight="bold")

    # Count per plan
    plan_churn = df.groupby(["plan", "churned"]).size().unstack(fill_value=0)
    plan_churn.plot(kind="bar", ax=axes[0], color=["#2196F3", "#F44336"], edgecolor="white")
    axes[0].set_title("Customer Count")
    axes[0].set_xlabel("")
    axes[0].legend(["Retained", "Churned"])
    axes[0].tick_params(axis="x", rotation=0)

    # Churn rate
    rates = df.groupby("plan")["churned"].mean().sort_values(ascending=False)
    bars = axes[1].bar(rates.index, rates.values, color="#FF7043", edgecolor="white")
    axes[1].set_title("Churn Rate by Plan")
    axes[1].set_ylabel("Churn Rate")
    for bar, val in zip(bars, rates.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.0%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = f"{PLOT_DIR}/churn_by_plan.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plot] Churn by plan saved → {path}")


def plot_correlation(df: pd.DataFrame):
    num_df = df.select_dtypes("number").drop(columns=["churned"], errors="ignore")
    corr   = num_df.corrwith(df["churned"]).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#F44336" if v > 0 else "#2196F3" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Feature Correlation with Churn", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pearson Correlation")
    plt.tight_layout()
    path = f"{PLOT_DIR}/correlation.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plot] Correlation chart saved → {path}")


def run():
    print("=" * 50)
    print("STEP 2: EDA")
    print("=" * 50)
    df = load(CLEANED_PATH)
    overview(df)
    churn_by_category(df)
    plot_distributions(df)
    plot_churn_by_plan(df)
    plot_correlation(df)
    print("\n[done] All EDA complete. Check notebooks/eda_plots/")


if __name__ == "__main__":
    run()
