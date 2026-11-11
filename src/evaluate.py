from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split

from preprocess import clean_text
from url_features import extract_url_features

from scipy.sparse import hstack, csr_matrix


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "data" / "emails.csv"

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "phishing_text_url_model.joblib"
)

VECTORIZER_FILE = (
    PROJECT_ROOT
    / "models"
    / "tfidf_vectorizer_url.joblib"
)

REPORT_DIR = PROJECT_ROOT / "reports"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATASET)

    df["clean_text"] = (
        df["body"]
        .fillna("")
        .apply(clean_text)
    )

    return df


# ============================================================
# PREPARE TEST DATA
# ============================================================

def prepare_test_data(df):

    X = df["clean_text"]

    y = df["label"]

    (
        X_train_text,
        X_test_text,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return (
        X_train_text,
        X_test_text,
        y_test
    )


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(
    X_train_text,
    X_test_text,
    vectorizer
):

    X_train_tfidf = vectorizer.transform(
        X_train_text
    )

    X_test_tfidf = vectorizer.transform(
        X_test_text
    )

    X_train_url = extract_url_features(
        X_train_text
    )

    X_test_url = extract_url_features(
        X_test_text
    )

    X_train_url = csr_matrix(
        X_train_url
    )

    X_test_url = csr_matrix(
        X_test_url
    )

    X_train = hstack(
        [
            X_train_tfidf,
            X_train_url
        ]
    )

    X_test = hstack(
        [
            X_test_tfidf,
            X_test_url
        ]
    )

    return X_test


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
    y_test,
    predictions
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Safe",
            "Phishing"
        ]
    )

    display.plot()

    plt.title(
        "Phishing Email Detection - Confusion Matrix"
    )

    plt.tight_layout()

    output_file = (
        REPORT_DIR
        / "confusion_matrix.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        f"Confusion matrix saved: {output_file}"
    )


# ============================================================
# PERFORMANCE GRAPH
# ============================================================

def create_performance_graph(
    y_test,
    predictions
):

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

    metrics = [
        accuracy,
        precision,
        recall,
        f1
    ]

    labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    plt.figure(
        figsize=(8, 5)
    )

    bars = plt.bar(
        labels,
        metrics
    )

    plt.ylim(
        0,
        1.05
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Phishing Email Detection Model Performance"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    # Display values above bars
    for bar, value in zip(
        bars,
        metrics
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center"
        )

    plt.tight_layout()

    output_file = (
        REPORT_DIR
        / "performance_metrics.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        f"Performance graph saved: {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "PHISHING EMAIL DETECTION - MODEL EVALUATION"
    )

    print("=" * 60)

    print("\nLoading dataset...")

    df = load_data()

    (
        X_train_text,
        X_test_text,
        y_test
    ) = prepare_test_data(df)

    print(
        f"Test emails: {len(X_test_text):,}"
    )

    print("\nLoading trained model...")

    model = joblib.load(
        MODEL_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    print(
        "Model and vectorizer loaded."
    )

    print("\nCreating test features...")

    X_test = create_features(
        X_train_text,
        X_test_text,
        vectorizer
    )

    print(
        f"Test feature matrix: {X_test.shape}"
    )

    print("\nGenerating predictions...")

    predictions = model.predict(
        X_test
    )

    print(
        "Predictions generated."
    )

    print("\nGenerating reports...")

    create_confusion_matrix(
        y_test,
        predictions
    )

    create_performance_graph(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)

    print(
        "EVALUATION COMPLETED SUCCESSFULLY!"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()