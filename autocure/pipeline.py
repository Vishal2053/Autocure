from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def make_pipeline(
    df,
    target: str = None,
    power_transform: bool = False,
    high_cardinality_threshold: int = 10
):
    """
    Construct a production preprocessing pipeline:
    - Numeric: Robust imputation (median) + Yeo-Johnson Power Transform (or StandardScaler)
    - Low-cardinality Categorical (<= 10 categories): OneHotEncoder
    - High-cardinality Categorical (> 10 categories): TargetEncoder (or Frequency-capped OneHot)
    """
    if target and target in df.columns:
        features = df.drop(columns=[target])
    else:
        features = df

    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = features.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    # Split categoricals into low vs high cardinality
    low_card_cols = [c for c in categorical if features[c].nunique() <= high_cardinality_threshold]
    high_card_cols = [c for c in categorical if features[c].nunique() > high_cardinality_threshold]

    # Numeric pipeline: Yeo-Johnson Power Transform or StandardScaler
    if power_transform:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("power", PowerTransformer(method="yeo-johnson", standardize=True))
        ])
    else:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

    transformers = []
    if numeric:
        transformers.append(("num", num_transformer, numeric))

    # Low-cardinality: OneHotEncoder
    if low_card_cols:
        low_cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("low_cat", low_cat_pipeline, low_card_cols))

    # High-cardinality: TargetEncoder if target provided, else OneHotEncoder with max_categories
    if high_card_cols:
        if target:
            high_cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("target_enc", TargetEncoder(smooth="auto", cv=5))
            ])
        else:
            high_cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", max_categories=15, sparse_output=False))
            ])
        transformers.append(("high_cat", high_cat_pipeline, high_card_cols))

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ("preprocess", transformer)
    ])

    return pipeline


