"""
PIPELINE ORCHESTRATOR
======================
Runs all steps end-to-end in order.
Usage:  python src/pipeline.py
"""

import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))

import step1_clean
import step2_eda
import step3_features
import step4_train
import step5_predict


STEPS = [
    ("1  Data Cleaning",       step1_clean.run),
    ("2  EDA",                 step2_eda.run),
    ("3  Feature Engineering", step3_features.run),
    ("4  Model Training",      step4_train.run),
    ("5  Predict (demo)",      step5_predict.run),
]


def run_pipeline():
    total_start = time.time()
    results = {}

    print("\n" + "╔" + "═" * 56 + "╗")
    print("║  CHURN PREDICTION PIPELINE — FULL RUN" + " " * 18 + "║")
    print("╚" + "═" * 56 + "╝\n")

    for name, fn in STEPS:
        print(f"\n{'─' * 56}")
        print(f"  RUNNING STEP {name}")
        print(f"{'─' * 56}")
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            results[name] = ("✓", elapsed)
            print(f"\n  ✓ Done in {elapsed:.1f}s")
        except Exception as e:
            results[name] = ("✗", str(e))
            print(f"\n  ✗ FAILED: {e}")
            raise

    total = time.time() - total_start
    print("\n\n" + "╔" + "═" * 56 + "╗")
    print("║  PIPELINE SUMMARY" + " " * 38 + "║")
    print("╠" + "═" * 56 + "╣")
    for name, (status, info) in results.items():
        if isinstance(info, float):
            print(f"║  {status}  {name:<35}  {info:>4.1f}s  ║")
        else:
            print(f"║  {status}  {name:<41}  ║")
    print("╠" + "═" * 56 + "╣")
    print(f"║  Total time: {total:.1f}s" + " " * (42 - len(f"{total:.1f}")) + "║")
    print("╚" + "═" * 56 + "╝\n")


if __name__ == "__main__":
    run_pipeline()
