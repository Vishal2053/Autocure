# Autocure

Automated data cleaning + AutoML model training with hyperparameter tuning.

## Key features

- Data scan
  - Missing values summary
  - Dtype inference
  - Outlier counts (IQR method)
  - Duplicate rows count

- Data cleaning (cure)
  - Numeric imputation (median)
  - Categorical imputation (mode / fallback "Unknown")
  - Text preprocessing (applies only to text columns):
    - remove punctuation
    - lowercase
    - tokenize
    - remove stopwords
    - lemmatize
  - Drop duplicates

- Pipeline & AutoML
  - Automatic ColumnTransformer (StandardScaler for numeric, OneHotEncoder for categorical)
  - Support for classification and regression
  - Multiple algorithms with GridSearchCV hyperparameter search
  - Model leaderboard and best-model selection
  - Save best model as a pickle into autocure/models/{model_name}_model.pickle

- CLI
  - scan-data, fix, train-model commands for quick workflows

## Installation

From project root (editable install for development):

```bash
python -m pip install -e .
```

Or install dependencies:

```bash
pip install -r requirements.txt
pip install nltk
```

NLTK data used by cure() is downloaded at runtime (punkt, stopwords, wordnet).

## Quick usage

Python:

```python
import autocure as ac
import pandas as pd

df = pd.read_csv("your_dataset.csv")

# 1) Inspect dataset
report = ac.scan(df)
print(report)

# 2) Clean dataset
clean = ac.cure(df)

# 3) Create pipeline (optional)
pipeline = ac.make_pipeline(clean, target="target_column")

# 4) Train models (returns a dict)
result = ac.train(clean, target="target_column")

print(result["best_model_name"])
print("score:", result["best_score"])
# Best model object for predict:
best_model = result["best_model"]
preds = best_model.predict(clean.drop(columns=["target_column"]).iloc[:5])
```

Returned keys from train():
- best_model (fitted sklearn estimator / pipeline)
- best_model_name
- best_score
- best_params
- leaderboard (list/dict of model results)
- model_path (path to saved pickle file under autocure/models/)

## Train a custom model (pass model_name + model_params)

You can request training of a specific model with custom (fixed) parameters by passing model_name and model_params to train(). When both are provided, train() will set the estimator parameters, fit on the data (no GridSearch), evaluate on the test split, and save the fitted pipeline to a pickle file:

- Saved path: autocure/models/{model_name}_model.pickle

Example (your requested usage):

```python
from autocure import cure, train
import pandas as pd
from sklearn import datasets

data = datasets.load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["label"] = data.target

clean = cure(df)

result_custom = train(
    clean,
    target="label",
    model_name="random_forest_classifier",
    model_params={"n_estimators": 50, "max_depth": 5}
)

print("Best model:", result_custom["best_model_name"])
print("Score:", result_custom["best_score"])
print("Saved model path:", result_custom["model_path"])
```

Notes:
- model_name must match one of the identifiers in train.py's base_models mapping (e.g. "random_forest_classifier", "logistic_regression", etc.). If you add new models to train.py's base_models, you can use the new identifier here.
- model_params should be a dict of estimator parameters (the keys are the estimator's parameter names, not pipeline step names). train() calls set_params(**model_params) on the estimator before fitting.
- The saved pickle contains the full pipeline (preprocessing + estimator) so it can be used directly for predict().

## Loading a saved model

```python
import pickle
import pandas as pd

X = pd.read_csv("my_features.csv")  # ensure same columns as training input

with open("autocure/models/random_forest_classifier_model.pickle", "rb") as f:
    model = pickle.load(f)

preds = model.predict(X)
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X)
```

## CLI

Examples:

```bash
# Scan a CSV
autocure scan-data data.csv

# Clean and save
autocure fix data.csv --out cleaned.csv

# Train via CLI
autocure train-model data.csv --target target_column
```

## Tips & notes

- cure() applies text preprocessing only to columns detected as object/string dtype.
- For custom high-cardinality categorical handling or advanced NLP, prepare features (vectorize / embed) before calling train() or extend the preprocessing in train.py.
- To add custom models to the AutoML search, add them to train.py's base_models dict (include estimator instance and param grid).

## Contributing

Bug reports and PRs welcome. Follow repository guidelines and add tests for new behavior.

## License

MIT License
