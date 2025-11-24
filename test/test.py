"""
Autocure test script using sklearn datasets (iris or wine)
"""

from autocure import scan, cure, make_pipeline, train
import pandas as pd
from sklearn import datasets

# Choose dataset: "iris" or "wine"
DATASET = "wine"


def load_sklearn_dataset(name: str) -> pd.DataFrame:
    if name == "wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_iris()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["label"] = data.target
    return df


def main():
    print(f"Loading sklearn '{DATASET}' dataset...")
    df = load_sklearn_dataset(DATASET)

    print(f"Dataset shape: {df.shape}")
    print(df.head(2))

    print("\nRunning scan()...")
    issues = scan(df)
    print("Issues found:", issues)

    print("\nRunning cure()...")
    df_clean = cure(df)
    print(f"After cleaning: {df_clean.shape}")
    print(df_clean.head(2))

    print("\nCreating pipeline...")
    pipeline = make_pipeline(df_clean, target="label")
    print("Pipeline created successfully!")

    # ========================================================
    # Train (uses autocure.train defaults)
    # ========================================================
    print("\n" + "="*60)
    print("Training (using autocure.train defaults)")

    result_custom = train(
        df_clean,
        target="label"
    )

    print("\nModel Training Complete!")
    print(f"Model: {result_custom.get('best_model_name')}")
    print(f"Test Accuracy: {result_custom.get('best_score')}")
    print(f"Used Parameters: {result_custom.get('best_params')}")

    # ========================================================
    # Make predictions on a few samples
    # ========================================================
    print("\nPredictions on sample rows:")
    sample_data = df_clean.drop("label", axis=1).iloc[:5].copy()
    print(sample_data)

    best_model = result_custom["best_model"]
    predictions = best_model.predict(sample_data)
    proba = None
    try:
        # handle pipeline or estimator
        estimator = best_model[-1] if hasattr(best_model, "__len__") and not hasattr(best_model, "predict") else best_model
        if hasattr(estimator, "predict_proba"):
            proba = best_model.predict_proba(sample_data)
    except Exception:
        proba = None

    print("\n→ Predictions (label indices):")
    for i, pred in enumerate(predictions):
        confidence = f"{proba[i].max():.3f}" if proba is not None else "N/A"
        print(f"   Row {i}: Predicted label = {pred} (confidence: {confidence})")
        print(f"   → features: {sample_data.iloc[i].to_dict()}\n")

    print("Test completed successfully!")


if __name__ == "__main__":
    main()