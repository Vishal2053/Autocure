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
- For small categorical cardinality detection and problem-type heuristics, train() uses simple rules; override by preparing the DataFrame appropriately.
- If you use large text datasets, consider vectorization or specialized NLP pipelines before heavy GridSearchCV.

## Contributing

Bug reports and PRs welcome. Follow repository guidelines and add tests for new behavior.

## License

MIT License
