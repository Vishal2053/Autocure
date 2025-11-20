import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def cure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fill numeric
    for col in df.select_dtypes(include=["int", "float"]):
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical
    for col in df.select_dtypes(include=["object"]):
        if df[col].isnull().any():
            mode_vals = df[col].mode()
            df[col] = df[col].fillna(mode_vals[0] if not mode_vals.empty else "Unknown")

    # Text preprocessing
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].apply(lambda x: _process_text(x, stop_words, lemmatizer) if isinstance(x, str) else x)

    # Drop duplicates
    df = df.drop_duplicates()

    return df


def _process_text(text, stop_words, lemmatizer):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)