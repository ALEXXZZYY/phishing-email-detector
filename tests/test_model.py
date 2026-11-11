from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "phishing_text_url_model.joblib"
)

VECTORIZER_PATH = (
    PROJECT_ROOT
    / "models"
    / "tfidf_vectorizer_url.joblib"
)


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_vectorizer_file_exists():
    assert VECTORIZER_PATH.exists()


def test_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_vectorizer_can_be_loaded():
    vectorizer = joblib.load(VECTORIZER_PATH)

    assert vectorizer is not None