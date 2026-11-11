from src.url_features import (
    extract_urls,
    extract_url_features
)


def test_extract_single_url():
    text = "Visit https://example.com"

    urls = extract_urls(text)

    assert len(urls) == 1


def test_extract_multiple_urls():
    text = """
    Visit https://example.com
    and http://test.com
    """

    urls = extract_urls(text)

    assert len(urls) == 2


def test_no_url():
    text = "This email contains no links."

    urls = extract_urls(text)

    assert len(urls) == 0


def test_url_features_shape():
    text = "Visit https://example.com"

    features = extract_url_features([text])

    assert features.shape == (1, 8)