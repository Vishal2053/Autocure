import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ML models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def _detect_task(y: pd.Series, task: str = "auto") -> bool:
    """Returns True if classification, False if regression."""
    if task != "auto":
        return task.lower() in ["classification", "clf"]
    if y.dtype == "object" or isinstance(y.dtype, pd.CategoricalDtype) or y.dtype == "bool":
        return True
    # If numeric, check if integer labels with low cardinality
    non_null = y.dropna()
    if len(non_null) > 0 and (non_null % 1 == 0).all() and non_null.nunique() <= 10:
        return True
    return False


def train(
    df: pd.DataFrame,
    target: str,
    model_name: str = None,
    model_params: dict = None,
    task: str = "auto",
    power_transform: str = "auto",
    high_cardinality_threshold: int = 10,
    model_dir: str = "models",
    cv: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
    track_mlflow: bool = False,
    experiment_name: str = "autocure",
    registered_model_name: str = None,
):
    """
    Train AutoML models with smart behavior:
    - If model_name + model_params -> train directly (no GridSearch)
    - If only model_name -> run GridSearch on that model
    - If neither -> run AutoML across all models and select best

    Args:
        df: Input pandas DataFrame
        target: Target column name
        model_name: Optional specific model identifier
        model_params: Optional specific parameters dictionary
        task: "auto", "classification", or "regression"
        power_transform: "auto", True, or False. If "auto", applies Yeo-Johnson to skewed features
        high_cardinality_threshold: Columns with more unique values than this use TargetEncoder
        model_dir: Directory where the best model will be saved (default: "models" in current directory)
        cv: Cross-validation folds
        test_size: Holdout test fraction
        random_state: Seed for reproducibility

    Returns:
        Dictionary with best_model, best_model_name, best_score, best_params, leaderboard, model_path
    """
    X = df.drop(columns=[target])
    y = df[target]

    # Determine task type
    is_classification = _detect_task(y, task)

    # Preprocessing with robust imputation so NaNs never crash predictions
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    # Determine whether to use Yeo-Johnson PowerTransformer
    use_power = False
    if power_transform is True:
        use_power = True
    elif power_transform == "auto" and len(X) >= 30 and len(numeric_cols) > 0:
        try:
            max_skew = X[numeric_cols].skew().abs().max()
            if max_skew > 0.75:
                use_power = True
        except Exception:
            use_power = False

    if use_power:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("power", PowerTransformer(method="yeo-johnson", standardize=True))
        ])
    else:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipeline, numeric_cols))

    # Split categoricals into low vs high cardinality
    low_card_cols = [c for c in cat_cols if X[c].nunique() <= high_cardinality_threshold]
    high_card_cols = [c for c in cat_cols if X[c].nunique() > high_cardinality_threshold]

    if low_card_cols:
        low_cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("low_cat", low_cat_pipe, low_card_cols))

    if high_card_cols:
        high_cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("target_enc", TargetEncoder(smooth="auto", cv=cv))
        ])
        transformers.append(("high_cat", high_cat_pipe, high_card_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough"
    )

    # Train-test split
    stratify = y if (is_classification and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Define base models and default param grids
    if is_classification:
        base_models = {
            "random_forest_classifier": (
                RandomForestClassifier(random_state=random_state),
                {"clf__n_estimators": [50, 100], "clf__max_depth": [5, 10, None]}
            ),
            "logistic_regression": (
                LogisticRegression(max_iter=1000),
                {"clf__C": [0.1, 1, 10], "clf__solver": ["lbfgs"]}
            ),
            "hist_gradient_boosting": (
                HistGradientBoostingClassifier(random_state=random_state),
                {"clf__learning_rate": [0.05, 0.1], "clf__max_iter": [50, 100]}
            ),
            "decision_tree_classifier": (
                DecisionTreeClassifier(random_state=random_state),
                {"clf__max_depth": [3, 5, 10, None]}
            ),
            "knn_classifier": (
                KNeighborsClassifier(),
                {"clf__n_neighbors": [3, 5, 7]}
            ),
            "svm_classifier": (
                SVC(),
                {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}
            ),
        }
    else:
        base_models = {
            "random_forest_regressor": (
                RandomForestRegressor(random_state=random_state),
                {"clf__n_estimators": [50, 100], "clf__max_depth": [5, 10, None]}
            ),
            "linear_regression": (
                LinearRegression(),
                {}
            ),
            "hist_gradient_boosting": (
                HistGradientBoostingRegressor(random_state=random_state),
                {"clf__learning_rate": [0.05, 0.1], "clf__max_iter": [50, 100]}
            ),
            "decision_tree_regressor": (
                DecisionTreeRegressor(random_state=random_state),
                {"clf__max_depth": [5, 10, None]}
            ),
            "knn_regressor": (
                KNeighborsRegressor(),
                {"clf__n_neighbors": [3, 5, 7]}
            ),
            "svm_regressor": (
                SVR(),
                {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}
            ),
        }

    scoring_metric = "accuracy" if is_classification else "r2"

    # CASE 1: Specific model + fixed parameters
    if model_name and model_params:
        if model_name not in base_models:
            raise ValueError(f"Model '{model_name}' not supported. Choose from: {list(base_models.keys())}")

        print(f"Training specified model with fixed parameters: {model_name}")
        base_model = base_models[model_name][0]
        base_model.set_params(**model_params)

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("clf", base_model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        score = accuracy_score(y_test, preds) if is_classification else r2_score(y_test, preds)

        result = {
            "best_model": pipeline,
            "best_model_name": model_name,
            "best_score": score,
            "best_params": model_params,
            "leaderboard": [{"model": model_name, "best_params": model_params, "score": score}],
            "task": "classification" if is_classification else "regression"
        }

    # CASE 2: Specific model + hyperparameter search
    elif model_name:
        if model_name not in base_models:
            raise ValueError(f"Model '{model_name}' not supported. Choose from: {list(base_models.keys())}")

        print(f"Tuning hyperparameters for: {model_name}")
        base_model, param_grid = base_models[model_name]

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("clf", base_model)
        ])

        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring_metric, n_jobs=-1)
        grid.fit(X_train, y_train)

        preds = grid.predict(X_test)
        score = accuracy_score(y_test, preds) if is_classification else r2_score(y_test, preds)

        result = {
            "best_model": grid.best_estimator_,
            "best_model_name": model_name,
            "best_score": score,
            "best_params": grid.best_params_,
            "leaderboard": [{"model": model_name, "best_params": grid.best_params_, "score": score}],
            "task": "classification" if is_classification else "regression"
        }

    # CASE 3: Full AutoML search across all models
    else:
        print("Running AutoML: Testing models...")
        results = []
        best_score = -9999
        best_model = None
        best_model_name = None
        best_params = None

        for name, (model, param_grid) in base_models.items():
            print(f"   Training {name}...", end="", flush=True)

            pipeline = Pipeline([
                ("preprocess", preprocessor),
                ("clf", model)
            ])

            if param_grid:
                grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring_metric, n_jobs=-1)
                grid.fit(X_train, y_train)
                current_model = grid.best_estimator_
                current_params = grid.best_params_
                preds = current_model.predict(X_test)
            else:
                current_model = pipeline
                current_model.fit(X_train, y_train)
                current_params = {}
                preds = current_model.predict(X_test)

            score = accuracy_score(y_test, preds) if is_classification else r2_score(y_test, preds)
            print(f" {score:.4f}")

            results.append({
                "model": name,
                "best_params": current_params,
                "score": score
            })

            if score > best_score:
                best_score = score
                best_model = current_model
                best_model_name = name
                best_params = current_params

        result = {
            "best_model": best_model,
            "best_model_name": best_model_name,
            "best_score": best_score,
            "best_params": best_params,
            "leaderboard": sorted(results, key=lambda x: x["score"], reverse=True),
            "task": "classification" if is_classification else "regression"
        }

    # Save model safely to model_dir (defaults to current working directory ./models)
    if model_dir:
        save_path = os.path.abspath(model_dir)
        os.makedirs(save_path, exist_ok=True)
        model_file = os.path.join(save_path, f"{result['best_model_name']}_model.pickle")

        with open(model_file, "wb") as f:
            pickle.dump(result["best_model"], f)

        print(f"\nModel saved to: {model_file}")
        result["model_path"] = model_file
    else:
        result["model_path"] = None

    print(f"Best model: {result['best_model_name']} | Score: {result['best_score']:.4f}")

    # Track in MLflow if requested (runs automatically in CLI commands, disabled in normal codes)
    if track_mlflow:
        from .tracking import log_autocure_run
        tracking_info = log_autocure_run(
            train_result=result,
            df=df,
            target=target,
            task="classification" if is_classification else "regression",
            experiment_name=experiment_name,
            registered_model_name=registered_model_name or f"autocure_{target}",
            X_sample=X.iloc[:5] if len(X) > 0 else None,
        )
        result["mlflow"] = tracking_info

    return result