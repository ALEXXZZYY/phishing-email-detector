from pathlib import Path

import joblib
import streamlit as st
from scipy.sparse import csr_matrix, hstack

from src.preprocess import clean_text
from src.url_features import (
    extract_url_features,
    extract_urls
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    return model, vectorizer


# ============================================================
# PREDICTION
# ============================================================

def predict_email(
    email_text,
    model,
    vectorizer
):

    # Clean text
    cleaned_text = clean_text(
        email_text
    )

    # TF-IDF
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

    # Combine features
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
# HEADER
# ============================================================

st.title(
    "🛡️ Phishing Email Detector"
)

st.markdown(
    """
### Machine Learning-Based Phishing Email Detection

Analyze an email using **textual and URL-based features**
and classify it as **Safe** or **Phishing**.
"""
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, vectorizer = load_model()

except Exception as error:

    st.error(
        f"Unable to load the trained model: {error}"
    )

    st.stop()


# ============================================================
# EMAIL INPUT
# ============================================================

st.subheader(
    "📧 Email Analysis"
)

email_text = st.text_area(
    "Paste the email content below:",
    height=300,
    placeholder=(
        "Subject: Urgent Account Verification\n\n"
        "Dear Customer,\n\n"
        "Your account has been suspended. "
        "Please verify your account immediately..."
    )
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔍 Analyze Email",
    type="primary",
    use_container_width=True
)


if analyze:

    if not email_text.strip():

        st.warning(
            "Please enter an email before analyzing."
        )

    else:

        with st.spinner(
            "Analyzing email..."
        ):

            label, confidence = predict_email(
                email_text,
                model,
                vectorizer
            )

            urls = extract_urls(
                email_text
            )

        st.divider()

        # ====================================================
        # RESULT
        # ====================================================

        st.subheader(
            "🎯 Prediction Result"
        )

        col1, col2 = st.columns(2)

        with col1:

            if label == "PHISHING":

                st.error(
                    "🚨 PHISHING EMAIL"
                )

            else:

                st.success(
                    "✅ SAFE EMAIL"
                )

        with col2:

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

        # ====================================================
        # URL ANALYSIS
        # ====================================================

        st.subheader(
            "🔗 URL Analysis"
        )

        url_col1, url_col2 = st.columns(2)

        with url_col1:

            st.metric(
                "URLs Detected",
                len(urls)
            )

        with url_col2:

            if urls:

                st.info(
                    "The email contains one or more URLs."
                )

            else:

                st.success(
                    "No URLs detected."
                )

        if urls:

            st.write(
                "**Detected URLs:**"
            )

            for url in urls:

                st.code(
                    url
                )

        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.subheader(
            "🤖 Model Information"
        )

        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:

            st.metric(
                "Training Emails",
                "82,077"
            )

        with info_col2:

            st.metric(
                "TF-IDF Features",
                "10,000"
            )

        with info_col3:

            st.metric(
                "Test Accuracy",
                "98.31%"
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🛡️ About"
    )

    st.write(
        """
        This application uses a machine learning model
        to classify emails as Safe or Phishing.
        """
    )

    st.divider()

    st.subheader(
        "Model"
    )

    st.write(
        "Logistic Regression"
    )

    st.subheader(
        "Features"
    )

    st.write(
        """
        • TF-IDF text features  
        • URL-based features  
        • URL detection  
        """
    )

    st.subheader(
        "Dataset"
    )

    st.write(
        "82,077 cleaned emails"
    )

    st.divider()

    st.caption(
        "Phishing Email Detection"
    )