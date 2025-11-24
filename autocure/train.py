import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ML models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.naive_bayes import GaussianNB  # Optional: add if needed


def train(df: pd.DataFrame, target: str, model_name: str = None, model_params: dict = None):
    """
    Train AutoML models with smart behavior:
    - If model_name + model_params → train directly (no GridSearch)
    - If only model_name → run GridSearch on that model
    - If neither → run AutoML on all models

    Returns:
        Dictionary with best_model, best_model_name, best_score, best_params, leaderboard, model_path
    """
    X = df.drop(columns=[target])
    y = df[target]

    # Determine task type
    is_classification = y.dtype == "object" or y.nunique() < 20

    # Preprocessing
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="passthrough"
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if is_classification else None)

    # Define base models and default param grids
    if is_classification:
        base_models = {
            "logistic_regression": (LogisticRegression(max_iter=1000), {"clf__C": [0.1, 1, 10], "clf__solver": ["lbfgs", "liblinear"]}),
            "random_forest_classifier": (RandomForestClassifier(random_state=42), {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [5, 10, None]}),
            "svm_classifier": (SVC(), {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}),
            "decision_tree_classifier": (DecisionTreeClassifier(random_state=42), {"clf__max_depth": [3, 5, 10, None]}),
            "knn_classifier": (KNeighborsClassifier(), {"clf__n_neighbors": [3, 5, 7, 9]}),
        }
    else:
        base_models = {
            "linear_regression": (LinearRegression(), {}),
            "random_forest_regressor": (RandomForestRegressor(random_state=42), {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [5, 10, None]}),
            "svm_regressor": (SVR(), {"clf__C": [0.1, 1, 10, 100], "clf__kernel": ["linear", "rbf"]}),
            "decision_tree_regressor": (DecisionTreeRegressor(random_state=42), {"clf__max_depth": [5, 10, None]}),
            "knn_regressor": (KNeighborsRegressor(), {"clf__n_neighbors": [3, 5, 7, 9]}),
        }

    scoring_metric = "accuracy" if is_classification else "r2"

    # CASE 1: User wants to train a specific model with FIXED parameters → No GridSearch
    if model_name and model_params:
        if model_name not in base_models:
            raise ValueError(f"Model '{model_name}' not supported. Choose from: {list(base_models.keys())}")

        print(f"Training specified model with fixed parameters: {model_name}")
        print(f"Parameters: {model_params}")

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
            "leaderboard": [{"model": model_name, "best_params": model_params, "score": score}]
        }

    # CASE 2: Only model_name given → run GridSearch on that model only
    elif model_name:
        if model_name not in base_models:
            raise ValueError(f"Model '{model_name}' not supported. Choose from: {list(base_models.keys())}")

        print(f"Tuning hyperparameters for: {model_name}")
        base_model, param_grid = base_models[model_name]

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("clf", base_model)
        ])

        grid = GridSearchCV(
            pipeline, param_grid, cv=3, scoring=scoring_metric, n_jobs=-1
        )
        grid.fit(X_train, y_train)

        preds = grid.predict(X_test)
        score = accuracy_score(y_test, preds) if is_classification else r2_score(y_test, preds)

        result = {
            "best_model": grid.best_estimator_,
            "best_model_name": model_name,
            "best_score": score,
            "best_params": grid.best_params_,
            "leaderboard": [{"model": model_name, "best_params": grid.best_params_, "score": score}]
        }

    # CASE 3: No model specified → AutoML mode (try all models)
    else:
        print("Running AutoML: Testing all models...")
        results = []
        best_score = -9999
        best_model = None
        best_model_name = None
        best_params = None

        for name, (model, param_grid) in base_models.items():
            print(f"   Training {name}...", end="")

            pipeline = Pipeline([
                ("preprocess", preprocessor),
                ("clf", model)
            ])

            if param_grid:  # Has tunable params
                grid = GridSearchCV(pipeline, param_grid, cv=3, scoring=scoring_metric, n_jobs=-1)
                grid.fit(X_train, y_train)
                current_model = grid.best_estimator_
                current_params = grid.best_params_
                preds = current_model.predict(X_test)
            else:  # No params to tune (e.g. LinearRegression)
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
            "leaderboard": sorted(results, key=lambda x: x["score"], reverse=True)
        }

    # Save the best/final model
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, f"{result['best_model_name']}_model.pickle")

    with open(model_file, "wb") as f:
        pickle.dump(result["best_model"], f)

    print(f"\nModel saved to: {model_file}")
    print(f"Best model: {result['best_model_name']} | Score: {result['best_score']:.4f}")

    result["model_path"] = model_file
    return result