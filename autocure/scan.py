import pandas as pd

def scan(df: pd.DataFrame):
    report = {}

    # Missing values
    report["missing_values"] = df.isnull().sum().to_dict()

    # Dtypes
    report["dtypes"] = df.dtypes.astype(str).to_dict()

    # Outliers
    outliers = {}
    for col in df.select_dtypes(include=["int", "float"]):
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        upper, lower = q3 + 1.5 * iqr, q1 - 1.5 * iqr
        outliers[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    report["outliers"] = outliers

    # Duplicates
    report["duplicate_rows"] = int(df.duplicated().sum())

    return report
