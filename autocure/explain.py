import pandas as pd
import numpy as np

def explain(result_or_model, top_n: int = 10) -> pd.DataFrame:
    """
    Explain the drivers of a trained model:
    - Extracts feature importance or coefficients from the fitted pipeline
    - Calculates relative percentage contribution
    - Returns a sorted DataFrame of top features
    - Prints top model drivers in plain English
    """
    # Extract pipeline/model
    if isinstance(result_or_model, dict) and "best_model" in result_or_model:
        model = result_or_model["best_model"]
        model_name = result_or_model.get("best_model_name", "Model")
    else:
        model = result_or_model
        model_name = getattr(model, "__class__", type(model)).__name__

    # Look for preprocessing step and clf step
    estimator = None
    feature_names = None

    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("clf", model[-1])
        if "preprocess" in model.named_steps:
            preprocessor = model.named_steps["preprocess"]
            try:
                feature_names = list(preprocessor.get_feature_names_out())
                # Clean up prefix names like "num__", "cat__"
                feature_names = [f.split("__")[-1] for f in feature_names]
            except Exception:
                feature_names = None
    else:
        estimator = model

    importances = None

    # Tree-based feature importances
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    # Linear model coefficients
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)

    if importances is None:
        print(f"Explainability: Model '{model_name}' does not expose native feature importances or coefficients.")
        return pd.DataFrame(columns=["feature", "importance", "impact_pct"])

    importances = np.array(importances, dtype=float)
    total = np.sum(importances)
    if total > 0:
        impact_pct = (importances / total) * 100
    else:
        impact_pct = np.zeros_like(importances)

    if feature_names is None or len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "impact_pct": impact_pct
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    top_df = df_importance.head(top_n)

    print("\n" + "=" * 65)
    print(f"             MODEL EXPLAINABILITY & TOP DRIVERS ({model_name})")
    print("=" * 65)
    for idx, row in top_df.iterrows():
        bar_len = int(row['impact_pct'] / 2)  # 50% = 25 chars
        bar = "#" * max(bar_len, 1)
        print(f" {idx+1:>2}. {row['feature']:<25} | {row['impact_pct']:>5.1f}% | {bar}")
    print("=" * 65 + "\n")

    return top_df
