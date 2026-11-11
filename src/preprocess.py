import re


def clean_text(text):
    """
    Clean email text before TF-IDF vectorization.
    """

    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace URLs with a special token
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text
    )

    # Replace email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " EMAIL ",
        text
    )

    # Replace numbers
    text = re.sub(r"\d+", " NUMBER ", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()