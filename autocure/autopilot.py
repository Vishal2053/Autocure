import pandas as pd
from .diagnose import diagnose
from .cure import cure
from .train import train
from .explain import explain

class AutopilotResult:
    """
    Unified result object returned by autocure.autopilot():
    Provides direct .predict(), .predict_proba(), .explain(), and access to full metrics.
    """
    def __init__(self, train_result: dict, diagnostics: dict, dropped_cols: list):
        self.best_model = train_result["best_model"]
        self.best_model_name = train_result["best_model_name"]
        self.best_score = train_result["best_score"]
        self.best_params = train_result["best_params"]
        self.leaderboard = train_result["leaderboard"]
        self.model_path = train_result.get("model_path")
        self.task = train_result.get("task", "classification")
        self.diagnostics = diagnostics
        self.dropped_cols = dropped_cols
        self.mlflow = train_result.get("mlflow")
        self._raw_result = train_result

    def _prepare_features(self, X):
        if isinstance(X, pd.DataFrame):
            X_clean = X.drop(columns=[c for c in self.dropped_cols if c in X.columns], errors="ignore")
            try:
                X_clean = cure(X_clean, drop_duplicates=False, clip_outliers=False)
            except Exception:
                pass
            return X_clean
        return X

    def predict(self, X: pd.DataFrame):
        """Predict on new samples, automatically handling preprocessing."""
        X_clean = self._prepare_features(X)
        return self.best_model.predict(X_clean)

    def predict_proba(self, X: pd.DataFrame):
        """Predict class probabilities (for classification tasks)."""
        X_clean = self._prepare_features(X)

        if hasattr(self.best_model, "predict_proba"):
            return self.best_model.predict_proba(X_clean)
        
        # Check estimator inside pipeline
        if hasattr(self.best_model, "named_steps") and hasattr(self.best_model.named_steps.get("clf"), "predict_proba"):
            return self.best_model.predict_proba(X_clean)

        raise AttributeError(f"Model '{self.best_model_name}' does not support predict_proba.")

    def explain(self, top_n: int = 10) -> pd.DataFrame:
        """Explain the top drivers of the trained model."""
        return explain(self.best_model, top_n=top_n)

    def __getitem__(self, item):
        return self._raw_result.get(item)

    def __repr__(self):
        return f"<AutopilotResult: Best Model='{self.best_model_name}' | Score={self.best_score:.4f} | Task='{self.task}'>"


def autopilot(
    df: pd.DataFrame,
    target: str,
    task: str = "auto",
    power_transform: str = "auto",
    clip_outliers: bool = True,
    auto_drop_traps: bool = True,
    model_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42,
    track_mlflow: bool = False,
    experiment_name: str = "autocure",
    registered_model_name: str = None,
) -> AutopilotResult:
    """
    THE 1-FUNCTION ALL-IN-ONE DATA SCIENCE & ML ENGINE:
    1. Runs diagnostics (Target Leakage, ID Traps, Constant features, Health Score)
    2. Auto-drops high-cardinality ID traps and zero-variance columns
    3. Runs smart data cleaning (Outlier Winsorization, Datetime extraction, Imputation)
    4. Automatically trains and compares modern ML models with hyperparameter tuning
       and Yeo-Johnson power transforms on skewed features
    5. Saves production pipeline and calculates explainability drivers

    Args:
        df: Raw input pandas DataFrame
        target: Target column name
        task: "auto", "classification", or "regression"
        power_transform: "auto", True, or False
        clip_outliers: If True, applies Winsorization to clamp extreme outliers
        auto_drop_traps: If True, automatically drops detected ID columns and constant features
        model_dir: Directory where the best model pickle will be saved
        test_size: Fraction of data reserved for evaluation
        random_state: Random seed for reproducibility
        track_mlflow: If True, tracks experiment, metrics, and models in MLflow (disabled by default in code, enabled automatically in CLI)
        experiment_name: MLflow experiment name
        registered_model_name: Name for the MLflow Model Registry to track model versions

    Returns:
        AutopilotResult with .predict(), .predict_proba(), .explain(), and model artifacts
    """
    print("\n" + "=" * 65)
    print("           AUTOCURE AUTOPILOT: STARTING FULL PIPELINE")
    print("=" * 65)

    # 1. Run Diagnostics
    diag = diagnose(df, target=target)

    cols_to_drop = []
    if auto_drop_traps:
        # Drop ID traps
        if diag.get("id_traps"):
            cols_to_drop.extend(diag["id_traps"])
        # Drop constant columns
        if diag.get("constant_columns"):
            cols_to_drop.extend(diag["constant_columns"])

        cols_to_drop = list(set([c for c in cols_to_drop if c != target and c in df.columns]))
        if cols_to_drop:
            print(f"[Autopilot] Auto-dropping non-predictive columns: {cols_to_drop}")

    df_filtered = df.drop(columns=cols_to_drop)

    # 2. Smart Cleaning & Outlier Winsorization
    print("[Autopilot] Cleaning, winsorizing outliers, and curing features...")
    df_clean = cure(df_filtered, clip_outliers=clip_outliers)

    # 3. Train & Tune Models with Smart Preprocessing
    print("[Autopilot] Running AutoML model competition...")
    train_result = train(
        df_clean,
        target=target,
        task=task,
        power_transform=power_transform,
        model_dir=model_dir,
        test_size=test_size,
        random_state=random_state,
        track_mlflow=track_mlflow,
        experiment_name=experiment_name,
        registered_model_name=registered_model_name,
    )

    result_obj = AutopilotResult(train_result, diag, cols_to_drop)

    # 4. Instant Explainability
    print("\n[Autopilot] Generating feature drivers...")
    result_obj.explain(top_n=5)

    print(f"[Autopilot] Completed! Access with result.predict() or result.explain().\n")
    return result_obj
