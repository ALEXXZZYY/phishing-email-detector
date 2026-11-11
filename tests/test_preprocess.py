from src.preprocess import clean_text


def test_clean_text_returns_string():
    text = "Hello! This is a TEST email."
    
    result = clean_text(text)

    assert isinstance(result, str)


def test_clean_text_lowercase():
    text = "HELLO WORLD"

    result = clean_text(text)

    assert result == result.lower()


def test_clean_text_empty_input():
    result = clean_text("")

    assert isinstance(result, str)