from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

from preprocess import clean_text
from url_features import extract_url_features


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    if not VECTORIZER_FILE.exists():

        raise FileNotFoundError(
            f"Vectorizer not found: {VECTORIZER_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    return model, vectorizer


# ============================================================
# PREDICT EMAIL
# ============================================================

def predict_email(
    email_text,
    model,
    vectorizer
):

    # Clean email
    cleaned_text = clean_text(
        email_text
    )

    # TF-IDF features
    text_features = vectorizer.transform(
        [cleaned_text]
    )

    # URL features
    url_features = extract_url_features(
        [email_text]
    )

    url_features = csr_matrix(
        url_features
    )

    # Combine text + URL features
    features = hstack(
        [
            text_features,
            url_features
        ]
    )

    # Prediction
    prediction = model.predict(
        features
    )[0]

    # Probability
    probabilities = model.predict_proba(
        features
    )[0]

    confidence = probabilities[
        int(prediction)
    ]

    if prediction == 1:

        label = "PHISHING"

    else:

        label = "SAFE"

    return label, confidence


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "PHISHING EMAIL DETECTOR"
    )

    print("=" * 60)

    model, vectorizer = load_model()

    print(
        "\nModel loaded successfully."
    )

    print(
        "\nPaste the email content below."
    )

    print(
        "Type END on a new line when finished."
    )

    print(
        "\n----------------------------------------"
    )

    lines = []

    while True:

        line = input()

        if line.strip().upper() == "END":

            break

        lines.append(line)

    email_text = "\n".join(
        lines
    )

    if not email_text.strip():

        print(
            "\nNo email content provided."
        )

        return

    label, confidence = predict_email(
        email_text,
        model,
        vectorizer
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "PREDICTION RESULT"
    )

    print(
        "=" * 60
    )

    if label == "PHISHING":

        print(
            "\n🚨 PHISHING EMAIL"
        )

        print(
            "This email contains characteristics "
            "associated with phishing."
        )

    else:

        print(
            "\n✅ SAFE EMAIL"
        )

        print(
            "This email appears to be legitimate "
            "according to the model."
        )

    print(
        f"\nConfidence: {confidence * 100:.2f}%"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()