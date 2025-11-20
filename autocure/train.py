import pandas as pd
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
from sklearn.naive_bayes import GaussianNB


def train(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]

    # Determine if classification problem
    is_classification = y.dtype == "object" or y.nunique() < 20

    # Preprocessing
    numeric_cols = X.select_dtypes(include=["int", "float"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ML Algorithms + Params
    if is_classification:
        models = {
            "logistic_regression": (
                LogisticRegression(max_iter=200),
                {"clf__C": [0.1, 1, 10]}
            ),
            "random_forest_classifier": (
                RandomForestClassifier(),
                {"clf__n_estimators": [50, 100],
                 "clf__max_depth": [5, 10, None]}
            ),
            "svm_classifier": (
                SVC(),
                {"clf__C": [0.1, 1, 10],
                 "clf__kernel": ["linear", "rbf"]}
            ),
            "decision_tree_classifier": (
                DecisionTreeClassifier(),
                {"clf__max_depth": [3, 5, 10, None]}
            ),
            "knn_classifier": (
                KNeighborsClassifier(),
                {"clf__n_neighbors": [3, 5, 7]}
            )
            
        }
    else:
        models = {
            "linear_regression": (
                LinearRegression(),
                {}
            ),
            "random_forest_regressor": (
                RandomForestRegressor(),
                {"clf__n_estimators": [50, 100],
                 "clf__max_depth": [5, 10, None]}
            ),
            "svm_regressor": (
                SVR(),
                {"clf__C": [0.1, 1, 10],
                 "clf__kernel": ["linear", "rbf"]}
            ),
            "decision_tree_regressor": (
                DecisionTreeRegressor(),
                {"clf__max_depth": [5, 10, None]}
            ),
            "knn_regressor": (
                KNeighborsRegressor(),
                {"clf__n_neighbors": [3, 5, 7]}
            )
        }

    # Train ALL models
    results = []
    best_model, best_params, best_score, best_model_name = None, None, -999, None

    for name, (model, params) in models.items():
        print(f"Training model: {name}")

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("clf", model)
        ])

        grid = GridSearchCV(
            pipeline,
            params,
            cv=3,
            n_jobs=-1,
            scoring="accuracy" if is_classification else "r2"
        )

        grid.fit(X_train, y_train)

        preds = grid.predict(X_test)
        score = accuracy_score(y_test, preds) if is_classification else r2_score(y_test, preds)

        results.append({
            "model": name,
            "best_params": grid.best_params_,
            "score": score
        })

        if score > best_score:
            best_score = score
            best_model = grid.best_estimator_
            best_model_name = name
            best_params = grid.best_params_

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_score": best_score,
        "best_params": best_params,
        "leaderboard": results
    }
