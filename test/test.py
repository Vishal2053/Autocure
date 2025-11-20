"""
Autocure test script using IMDB Sentiment Dataset
(review, sentiment)

Steps:
- load dataset from URL
- scan
- cure
- make_pipeline
- train
- predict
"""

from autocure import scan, cure, make_pipeline, train
import pandas as pd


def main():

    print("🎬 Loading IMDB Sentiment Dataset...")
    url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
    df = pd.read_csv(url)

    print(df.head())

    print("\n📌 Rename columns for consistency...")
    df["content"] = df["review"]
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    df = df[["content", "label"]]

    print(df.head())

    print("\n📌 Running scan()...")
    issues = scan(df)
    print("Issues found:", issues)

    print("\n🧹 Running cure()...")
    df_clean = cure(df)
    print("Cleaned data sample:")
    print(df_clean.head())

    print("\n🔗 Creating pipeline...")
    pipeline = make_pipeline(df_clean, target="label")
    print("Pipeline created:")
    print(pipeline)

    print("\n🤖 Training using train()...")
    result = train(df_clean, target="label")

    best_model = result["best_model"]
    best_score = result["best_score"]

    print("\n🎉 Best model:")
    print(result["best_model_name"])
    print(f"Best score: {best_score:.4f}")
    print("Best parameters:", result["best_params"])

    print("\n🔮 Predictions on sample data:")
    sample_data = df_clean.drop("label", axis=1).iloc[:5]
    preds = best_model.predict(sample_data)
    print(preds)


if __name__ == "__main__":
    main()
