from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import clean_text


# Find the project root automatically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "data" / "emails.csv"


def load_data(file_path):
    """Load the cleaned email dataset."""

    df = pd.read_csv(file_path)

    print(f"Loaded {len(df):,} emails.")

    return df


def preprocess_emails(df):
    """Clean the email text."""

    print("Cleaning email text...")

    df["clean_text"] = df["body"].apply(clean_text)

    return df


def create_tfidf_features(df):
    """Convert email text into TF-IDF numerical features."""

    print("Creating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(df["clean_text"])

    y = df["label"]

    print(f"Number of emails: {X.shape[0]:,}")
    print(f"Number of TF-IDF features: {X.shape[1]:,}")

    return X, y, vectorizer


if __name__ == "__main__":

    if not DATASET.exists():
        print(f"ERROR: Dataset not found: {DATASET}")
        raise SystemExit(1)

    df = load_data(DATASET)

    df = preprocess_emails(df)

    X, y, vectorizer = create_tfidf_features(df)

    print("\nFeature extraction completed successfully!")