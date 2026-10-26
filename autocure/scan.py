import pandas as pd

def scan(df: pd.DataFrame):
    report = {}

    # Basic shape
    report["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}

    # Missing values
    report["missing_values"] = df.isnull().sum().to_dict()

    # Dtypes
    report["dtypes"] = df.dtypes.astype(str).to_dict()

    # Outliers (IQR method on all numeric columns)
    outliers = {}
    for col in df.select_dtypes(include=["number"]).columns:
        valid_series = df[col].dropna()
        if len(valid_series) >= 4:
            q1, q3 = valid_series.quantile([0.25, 0.75])
            iqr = q3 - q1
            upper, lower = q3 + 1.5 * iqr, q1 - 1.5 * iqr
            outliers[col] = int(((valid_series < lower) | (valid_series > upper)).sum())
        else:
            outliers[col] = 0
    report["outliers"] = outliers

    # Duplicates
    report["duplicate_rows"] = int(df.duplicated().sum())

    return report

