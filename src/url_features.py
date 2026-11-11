import re
import numpy as np


# Common URL pattern
URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    re.IGNORECASE
)


# URL shortener domains
SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "buff.ly",
    "tiny.cc",
    "lnkd.in",
}


def extract_urls(text):
    """Extract URLs from email text."""

    if not isinstance(text, str):
        return []

    return URL_PATTERN.findall(text)


def has_ip_address(url):
    """Check whether a URL contains an IPv4 address."""

    ip_pattern = re.compile(
        r"(?:\d{1,3}\.){3}\d{1,3}"
    )

    return bool(ip_pattern.search(url))


def is_shortened_url(url):
    """Check whether URL belongs to a common URL shortener."""

    url_lower = url.lower()

    return any(
        domain in url_lower
        for domain in SHORTENER_DOMAINS
    )


def calculate_url_features(text):
    """
    Extract numerical URL-related features from email text.

    Returns:
        numpy array containing URL features.
    """

    urls = extract_urls(text)

    if not urls:
        return np.array([
            0,  # number of URLs
            0,  # average URL length
            0,  # maximum URL length
            0,  # IP address URL
            0,  # shortened URL
            0,  # suspicious character count
            0,  # HTTPS URL count
            0,  # HTTP URL count
        ], dtype=float)

    url_lengths = [
        len(url)
        for url in urls
    ]

    ip_count = sum(
        has_ip_address(url)
        for url in urls
    )

    shortened_count = sum(
        is_shortened_url(url)
        for url in urls
    )

    suspicious_characters = sum(
        url.count("@")
        + url.count("-")
        + url.count("_")
        + url.count("=")
        + url.count("?")
        for url in urls
    )

    https_count = sum(
        url.lower().startswith("https://")
        for url in urls
    )

    http_count = sum(
        url.lower().startswith("http://")
        for url in urls
    )

    return np.array([
        len(urls),
        np.mean(url_lengths),
        max(url_lengths),
        ip_count,
        shortened_count,
        suspicious_characters,
        https_count,
        http_count,
    ], dtype=float)


def extract_url_features(texts):
    """Extract URL features for multiple emails."""

    return np.array([
        calculate_url_features(text)
        for text in texts
    ])


if __name__ == "__main__":

    test_email = """
    Your account has been suspended.

    Click here immediately:
    http://192.168.1.100/login?verify=true

    """

    features = calculate_url_features(test_email)

    print("Test URL features:")
    print(features)