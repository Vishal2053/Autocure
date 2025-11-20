from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def make_pipeline(df, target):
    features = df.drop(columns=[target])

    numeric = features.select_dtypes(include=["int", "float"]).columns
    categorical = features.select_dtypes(include=["object"]).columns

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        ]
    )

    pipeline = Pipeline([
        ("preprocess", transformer)
    ])

    return pipeline
