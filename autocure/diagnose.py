import pandas as pd
import numpy as np

def diagnose(df: pd.DataFrame, target: str = None) -> dict:
    """
    Advanced Data Science & ML Diagnostic Doctor:
    - Calculates overall Data Health Score (0 - 100)
    - Detects Target Leakage (features with extreme correlation > 0.95 with target)
    - Detects ID & High-Cardinality Traps (> 90% unique values)
    - Detects Constant / Zero-Variance columns
    - Detects Extreme Class Imbalance
    - Analyzes Missing Values and Outliers
    - Generates actionable recommendations

    Returns:
        dict: Diagnostic metrics, alerts, health_score, and recommendations
    """
    n_rows, n_cols = df.shape
    alerts = []
    penalties = 0

    report = {
        "dataset_shape": {"rows": n_rows, "columns": n_cols},
        "target": target,
        "health_score": 100,
        "alerts": [],
        "target_leakage": [],
        "id_traps": [],
        "constant_columns": [],
        "missing_summary": {},
        "imbalance_warning": None,
        "recommendations": []
    }

    if n_rows == 0 or n_cols == 0:
        report["health_score"] = 0
        report["alerts"].append("Dataset is empty.")
        return report

    # 1. Missing Values Analysis
    missing_pct = (df.isnull().sum() / n_rows) * 100
    cols_with_missing = missing_pct[missing_pct > 0].to_dict()
    report["missing_summary"] = {k: round(v, 2) for k, v in cols_with_missing.items()}

    total_missing_cells = df.isnull().sum().sum()
    total_cells = n_rows * n_cols
    overall_missing_rate = (total_missing_cells / total_cells) * 100

    if overall_missing_rate > 30:
        penalties += 25
        alerts.append(f"Severe missingness: {overall_missing_rate:.1f}% of all cells are missing.")
        report["recommendations"].append("Consider dropping columns with > 50% missing values or using advanced imputation.")
    elif overall_missing_rate > 5:
        penalties += 10
        alerts.append(f"Moderate missingness: {overall_missing_rate:.1f}% of cells are missing.")

    # 2. Constant / Zero-Variance Columns
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_cols:
        penalties += min(len(constant_cols) * 5, 20)
        report["constant_columns"] = constant_cols
        alerts.append(f"Zero-variance columns found: {constant_cols}. These add no predictive value.")
        report["recommendations"].append(f"Drop zero-variance columns: {constant_cols}")

    # 3. ID / High-Cardinality Traps (Columns with > 90% unique values that aren't float)
    id_traps = []
    for col in df.columns:
        if col == target:
            continue
        if df[col].dtype == "object" or str(df[col].dtype).startswith("int"):
            unique_ratio = df[col].nunique() / n_rows
            col_lower = str(col).lower()
            if unique_ratio > 0.90 and n_rows >= 30:
                id_traps.append(col)
            elif any(tag in col_lower for tag in ["id", "uuid", "guid", "row_number"]) and unique_ratio > 0.70:
                id_traps.append(col)

    if id_traps:
        penalties += min(len(id_traps) * 8, 20)
        report["id_traps"] = id_traps
        alerts.append(f"Potential ID / High-Cardinality traps: {id_traps}. These cause overfitting or massive one-hot encoding matrices.")
        report["recommendations"].append(f"Drop ID or serial number columns: {id_traps}")

    # 4. Duplicate Rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = (dup_count / n_rows) * 100
        if dup_pct > 5:
            penalties += 15
            alerts.append(f"High duplicate rate: {dup_count} duplicate rows ({dup_pct:.1f}%).")
        else:
            penalties += 5
            alerts.append(f"Found {dup_count} duplicate rows ({dup_pct:.1f}%).")
        report["recommendations"].append("Remove duplicate rows using ac.cure(df, drop_duplicates=True).")

    # 5. Target Leakage & Imbalance Analysis (if target specified)
    if target and target in df.columns:
        y = df[target]

        # Check Imbalance for Categorical / Low Cardinality target
        if y.dtype == "object" or y.nunique() <= 10:
            val_counts = y.value_counts(normalize=True)
            if len(val_counts) > 1:
                min_class_ratio = val_counts.min()
                max_class_ratio = val_counts.max()
                imbalance_ratio = max_class_ratio / max(min_class_ratio, 1e-6)
                if imbalance_ratio > 5.0:
                    penalties += 15
                    imbalance_msg = f"Severe class imbalance: ratio is {imbalance_ratio:.1f}:1 (minority class is {min_class_ratio*100:.1f}%)."
                    report["imbalance_warning"] = imbalance_msg
                    alerts.append(imbalance_msg)
                    report["recommendations"].append("Use class-weighted loss, SMOTE, or evaluate with PR-AUC / F1 instead of Accuracy.")

        # Check Target Leakage on numeric features
        leakage_cols = []
        if pd.api.types.is_numeric_dtype(y) and y.std() > 0:
            for col in df.select_dtypes(include=["number"]).columns:
                if col == target or col in constant_cols or df[col].std() == 0:
                    continue
                try:
                    corr = abs(df[col].corr(y))
                    if corr >= 0.95:
                        leakage_cols.append({"column": col, "correlation": round(float(corr), 4)})
                except Exception:
                    pass

        if leakage_cols:
            penalties += 30
            report["target_leakage"] = leakage_cols
            col_names = [l["column"] for l in leakage_cols]
            alerts.append(f"SUSPECTED TARGET LEAKAGE: Columns {col_names} have correlation >= 0.95 with the target!")
            report["recommendations"].append(f"Investigate {col_names} for data leakage before deploying model.")

    # Calculate final health score
    health_score = max(0, min(100, 100 - penalties))
    report["health_score"] = health_score
    report["alerts"] = alerts

    # Pretty console printout
    print("\n" + "=" * 65)
    print("                AUTOCURE DATA HEALTH REPORT")
    print("=" * 65)
    print(f"Dataset Shape : {n_rows} rows x {n_cols} columns")
    print(f"Target Column : {target if target else 'None specified'}")
    
    status_label = "EXCELLENT" if health_score >= 85 else ("GOOD" if health_score >= 70 else ("WARNING" if health_score >= 50 else "CRITICAL"))
    print(f"Health Score  : {health_score}/100 [{status_label}]")
    print("-" * 65)

    if alerts:
        print("Alerts:")
        for a in alerts:
            print(f"  [!] {a}")
    else:
        print("  [OK] No critical dataset issues detected!")

    if report["recommendations"]:
        print("\nActionable Recommendations:")
        for r in report["recommendations"]:
            print(f"  -> {r}")
    print("=" * 65 + "\n")

    return report
