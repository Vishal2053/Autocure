"""
Unit test for accuracy-boosting features:
1. Outlier Winsorization (cure)
2. Datetime Extraction (cure)
3. Yeo-Johnson PowerTransformer (pipeline & train)
4. High-cardinality TargetEncoder (pipeline & train)
"""

import autocure as ac
import pandas as pd
import numpy as np

def test_accuracy_suite():
    print("=" * 65)
    print("Testing Accuracy-Boosting Preprocessing Suite")
    print("=" * 65)

    np.random.seed(42)
    n = 100

    # Create dataset with:
    # - Skewed numerical data (exponential)
    # - Extreme outliers in 'income'
    # - High-cardinality categorical 'city' (20 unique cities)
    # - Low-cardinality categorical 'gender' ('M', 'F')
    # - Datetime string 'signup_date'
    # - Target 'subscribed' (0 or 1)
    df = pd.DataFrame({
        "income": np.random.exponential(scale=5000, size=n),
        "age": np.random.randint(18, 70, size=n).astype(float),
        "city": [f"City_{i % 20}" for i in range(n)],
        "gender": ["M" if i % 2 == 0 else "F" for i in range(n)],
        "signup_date": [f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)],
        "subscribed": [1 if i % 3 == 0 else 0 for i in range(n)]
    })

    # Add massive outliers to 'income'
    df.loc[0, "income"] = 1_000_000.0
    df.loc[1, "income"] = -50_000.0

    print("\n[TEST 1] Testing Outlier Winsorization & Datetime Extraction in ac.cure()...")
    df_cured = ac.cure(df, clip_outliers=True, extract_dates=True)

    # Verify outlier was clamped
    assert df_cured["income"].max() < 1_000_000.0, "Outlier was not clamped!"
    assert df_cured["income"].min() > -50_000.0, "Negative outlier was not clamped!"
    print(f"  -> Outliers successfully clamped! Max income before: 1,000,000 | After: {df_cured['income'].max():.2f}")

    # Verify datetime was decomposed
    assert "signup_date_month" in df_cured.columns, "Datetime month was not extracted!"
    assert "signup_date_is_weekend" in df_cured.columns, "Datetime is_weekend was not extracted!"
    assert "signup_date" not in df_cured.columns, "Raw datetime string was not removed!"
    print("  -> Datetime features successfully extracted (month, day, dayofweek, is_weekend)!")

    print("\n[TEST 2] Testing make_pipeline() with PowerTransformer & TargetEncoder...")
    pipe = ac.make_pipeline(df_cured, target="subscribed", power_transform=True, high_cardinality_threshold=10)
    assert pipe is not None
    print("  -> Pipeline with Yeo-Johnson and TargetEncoder constructed successfully!")

    print("\n[TEST 3] Testing ac.train() with auto PowerTransformer and TargetEncoder...")
    train_res = ac.train(
        df_cured,
        target="subscribed",
        power_transform="auto",
        model_name="random_forest_classifier",
        model_params={"n_estimators": 50, "max_depth": 5}
    )
    assert train_res["best_model"] is not None
    print(f"  -> Model trained successfully! Score: {train_res['best_score']:.4f}")

    print("\n[TEST 4] Testing ac.autopilot() on challenging real-world dataset...")
    auto_res = ac.autopilot(df, target="subscribed")
    assert auto_res.best_model is not None

    # Test predict on new sample
    sample = df.drop(columns=["subscribed"]).iloc[:3]
    preds = auto_res.predict(sample)
    print("  -> Autopilot predictions:", preds)

    print("\n" + "=" * 65)
    print("ALL ACCURACY-BOOSTING PREPROCESSING TESTS PASSED!")
    print("=" * 65)

if __name__ == "__main__":
    test_accuracy_suite()
