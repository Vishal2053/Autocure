"""
Test script for Autocure Advanced Features:
- diagnose(): Data health score & leakage/ID trap detection
- autopilot(): 1-function end-to-end data science & ML engine
- explain(): Model drivers & feature importance
"""

import autocure as ac
import pandas as pd
import numpy as np
from sklearn import datasets

def test_advanced_suite():
    print("=" * 60)
    print("Testing Autocure Advanced Features Suite")
    print("=" * 60)

    # 1. Prepare a synthetic dataset with common real-world flaws:
    # - Wine features
    # - Missing values
    # - An ID column (trap)
    # - A constant column (trap)
    wine = datasets.load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["label"] = wine.target

    # Introduce flaws to test the diagnostic engine
    np.random.seed(42)
    df["customer_uuid"] = [f"USR_{i:04d}" for i in range(len(df))]  # ID trap
    df["constant_flag"] = 1                                         # Constant column
    # Add some random missing values in 'alcohol'
    mask = np.random.rand(len(df)) < 0.1
    df.loc[mask, "alcohol"] = np.nan

    print("\n[STEP 1] Testing ac.diagnose() on dataset with intentional flaws...")
    diag_report = ac.diagnose(df, target="label")
    assert "health_score" in diag_report
    assert "customer_uuid" in diag_report["id_traps"]
    assert "constant_flag" in diag_report["constant_columns"]
    print("Diagnostics successfully flagged ID traps, constant columns, and calculated Health Score!")

    print("\n[STEP 2] Testing ac.autopilot() (1-function full workflow)...")
    # Autopilot will diagnose -> auto-drop ID traps & constants -> cure -> train -> explain
    result = ac.autopilot(df, target="label", model_dir="models")

    assert result.best_model is not None
    assert result.best_score > 0.8
    assert "customer_uuid" in result.dropped_cols
    assert "constant_flag" in result.dropped_cols

    print("\n[STEP 3] Testing direct predictions on test rows...")
    test_samples = df.drop(columns=["label"]).iloc[:3]
    preds = result.predict(test_samples)
    probs = result.predict_proba(test_samples)
    print("Predictions :", preds)
    print("Probabilities shape:", probs.shape)

    print("\n[STEP 4] Testing ac.explain()...")
    explanation = result.explain(top_n=5)
    assert not explanation.empty
    print(explanation)

    print("\n" + "=" * 60)
    print("ALL ADVANCED FEATURES TESTED AND PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_advanced_suite()
