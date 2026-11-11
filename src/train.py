from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from preprocess import clean_text
from url_features import extract_url_features


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "data" / "emails.csv"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODEL_DIR / "phishing_text_url_model.joblib"

VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer_url.joblib"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv(DATASET)

    print(f"Total emails: {len(df):,}")

    return df


# ============================================================
# PREPROCESS TEXT
# ============================================================

def prepare_text(df):

    print("Cleaning email text...")

    df["clean_text"] = (
        df["body"]
        .fillna("")
        .apply(clean_text)
    )

    return df


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_dataset(df):

    X_text = df["clean_text"]

    y = df["label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nDataset split:")

    print(f"Training emails: {len(X_train_text):,}")
    print(f"Testing emails : {len(X_test_text):,}")

    return (
        X_train_text,
        X_test_text,
        y_train,
        y_test
    )


# ============================================================
# TF-IDF FEATURES
# ============================================================

def create_tfidf_features(
    X_train_text,
    X_test_text
):

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train_text
    )

    X_test_tfidf = vectorizer.transform(
        X_test_text
    )

    print(
        "TF-IDF training features:",
        X_train_tfidf.shape
    )

    print(
        "TF-IDF testing features:",
        X_test_tfidf.shape
    )

    return (
        X_train_tfidf,
        X_test_tfidf,
        vectorizer
    )


# ============================================================
# URL FEATURES
# ============================================================

def create_url_features(
    X_train_text,
    X_test_text
):

    print("\nExtracting URL features...")

    X_train_url = extract_url_features(
        X_train_text
    )

    X_test_url = extract_url_features(
        X_test_text
    )

    print(
        "URL training features:",
        X_train_url.shape
    )

    print(
        "URL testing features:",
        X_test_url.shape
    )

    return X_train_url, X_test_url


# ============================================================
# COMBINE FEATURES
# ============================================================

def combine_features(
    X_train_tfidf,
    X_test_tfidf,
    X_train_url,
    X_test_url
):

    print("\nCombining text + URL features...")

    # Convert URL features to sparse matrices
    X_train_url_sparse = csr_matrix(
        X_train_url
    )

    X_test_url_sparse = csr_matrix(
        X_test_url
    )

    # Combine TF-IDF + URL features
    X_train = hstack(
        [
            X_train_tfidf,
            X_train_url_sparse
        ]
    )

    X_test = hstack(
        [
            X_test_tfidf,
            X_test_url_sparse
        ]
    )

    print(
        "Combined training features:",
        X_train.shape
    )

    print(
        "Combined testing features:",
        X_test.shape
    )

    return X_train, X_test


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train
):

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\nEvaluating model...")

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 50)
    print("TEXT + URL MODEL PERFORMANCE")
    print("=" * 50)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Safe",
                "Phishing"
            ]
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return predictions


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    vectorizer
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )

    print("\nModels saved successfully!")

    print(
        f"Model     : {MODEL_FILE}"
    )

    print(
        f"Vectorizer: {VECTORIZER_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "PHISHING EMAIL DETECTION"
    )

    print(
        "TEXT + URL MACHINE LEARNING MODEL"
    )

    print("=" * 60)

    # Load dataset
    df = load_dataset()

    # Clean email text
    df = prepare_text(df)

    # Split dataset
    (
        X_train_text,
        X_test_text,
        y_train,
        y_test
    ) = split_dataset(df)

    # TF-IDF
    (
        X_train_tfidf,
        X_test_tfidf,
        vectorizer
    ) = create_tfidf_features(
        X_train_text,
        X_test_text
    )

    # URL features
    (
        X_train_url,
        X_test_url
    ) = create_url_features(
        X_train_text,
        X_test_text
    )

    # Combine
    (
        X_train,
        X_test
    ) = combine_features(
        X_train_tfidf,
        X_test_tfidf,
        X_train_url,
        X_test_url
    )

    # Train
    model = train_model(
        X_train,
        y_train
    )

    # Evaluate
    evaluate_model(
        model,
        X_test,
        y_test
    )

    # Save
    save_model(
        model,
        vectorizer
    )

    print("\n" + "=" * 60)

    print(
        "TEXT + URL MODEL TRAINING COMPLETED!"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()