import pandas as pd
import re

def _is_text_column(series: pd.Series, min_avg_words: float = 3.0, min_unique_ratio: float = 0.2) -> bool:
    """
    Detect if an object column is free-text (e.g. descriptions, reviews)
    rather than a discrete categorical column (e.g. gender, country, status).
    """
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return False
    # Check average word count
    avg_words = non_null.apply(lambda s: len(s.split())).mean()
    unique_ratio = non_null.nunique() / max(len(non_null), 1)
    return avg_words >= min_avg_words or (unique_ratio > min_unique_ratio and avg_words > 2.0)


def _try_parse_datetime(series: pd.Series, sample_size: int = 20) -> bool:
    """Check if an object series looks like date/time values."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    sample = non_null.head(sample_size).astype(str)
    # Fast check: must contain separators like -, /, or :
    if not any("-" in s or "/" in s or ":" in s for s in sample):
        return False
    try:
        pd.to_datetime(sample, errors="raise")
        return True
    except Exception:
        return False


def cure(
    df: pd.DataFrame,
    text_cols: list = None,
    drop_duplicates: bool = True,
    clip_outliers: bool = True,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    extract_dates: bool = True,
) -> pd.DataFrame:
    """
    Advanced data cleaning and preprocessing:
    - Robust numeric imputation (median) across all numeric dtypes
    - Outlier Winsorization: Clamps extreme numeric values to [1st, 99th] quantiles
    - Automated Datetime Feature Extraction (year, month, day, dayofweek, is_weekend)
    - Categorical imputation with whitespace stripping
    - NLP text preprocessing ONLY on free-text narrative columns
    - Duplicate row removal
    """
    df = df.copy()

    # 1. Automated Datetime Extraction
    if extract_dates:
        for col in list(df.select_dtypes(include=["object", "string"]).columns):
            if _try_parse_datetime(df[col]):
                try:
                    dt = pd.to_datetime(df[col], errors="coerce")
                    if dt.notnull().sum() > 0:
                        df[f"{col}_year"] = dt.dt.year
                        df[f"{col}_month"] = dt.dt.month
                        df[f"{col}_day"] = dt.dt.day
                        df[f"{col}_dayofweek"] = dt.dt.dayofweek
                        df[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
                        df = df.drop(columns=[col])
                except Exception:
                    pass

    # 2. Fill numeric columns with median & Winsorize Outliers
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notnull(median_val) else 0)

        # Outlier Winsorization (clamping extreme tails)
        if clip_outliers and len(df) >= 20:
            q_low = df[col].quantile(lower_quantile)
            q_high = df[col].quantile(upper_quantile)
            if pd.notnull(q_low) and pd.notnull(q_high) and q_low < q_high:
                df[col] = df[col].clip(lower=q_low, upper=q_high)

    # 3. Separate free-text columns from discrete categorical columns
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    
    if text_cols is None:
        actual_text_cols = [col for col in obj_cols if _is_text_column(df[col])]
    else:
        actual_text_cols = [col for col in text_cols if col in df.columns]

    cat_cols = [c for c in obj_cols if c not in actual_text_cols]

    # Fill and clean standard categorical columns (no lemmatizing/stopword stripping)
    for col in cat_cols:
        if df[col].isnull().any():
            mode_vals = df[col].mode()
            df[col] = df[col].fillna(mode_vals[0] if not mode_vals.empty else "Unknown")
        df[col] = df[col].astype(str).str.strip()

    # Clean free-text columns only if present
    if actual_text_cols:
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            from nltk.stem import WordNetLemmatizer
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

            for col in actual_text_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna("")
                df[col] = df[col].apply(lambda x: _process_text(str(x), stop_words, lemmatizer))
        except Exception:
            # Fallback if NLTK is unavailable/offline: basic regex cleanup
            for col in actual_text_cols:
                df[col] = df[col].fillna("").astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)

    # 4. Drop duplicates
    if drop_duplicates:
        df = df.drop_duplicates()

    return df


def _process_text(text: str, stop_words, lemmatizer):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase
    text = text.lower()
    # Tokenize
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)