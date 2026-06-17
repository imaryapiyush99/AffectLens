import pytest
from affectlens.preprocessing import clean_text, tokenize, remove_stopwords, lemmatize, preprocess

def test_clean_text_invalid_input():
    with pytest.raises(ValueError):
        clean_text(123)
    with pytest.raises(ValueError):
        clean_text("")

def test_clean_text_valid():
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("This is a test.") == "this is a test"
    assert clean_text("12345") == "12345"
    assert clean_text("Special characters: @#$%^&*()") == "special characters"
    assert clean_text("   Extra whitespace   ") == "extra whitespace"
    assert clean_text("HTML entities &amp; &lt; &gt;") == "html entities"
    assert clean_text("Visit http://example.com today") == "visit today"

def test_tokenize_invalid_input():
    with pytest.raises(ValueError):
        tokenize(123)
    with pytest.raises(ValueError):
        tokenize("  ")
    with pytest.raises(ValueError):   
        tokenize([1, 0])
    with pytest.raises(ValueError):
        tokenize(["", " "])


def test_tokenize_valid():
    assert tokenize("hello world") == ["hello", "world"]
    assert tokenize("this is a test") == ["this", "is", "a", "test"]

def test_remove_stopwords_invalid_input():
    with pytest.raises(ValueError):
        remove_stopwords("not a list")
    with pytest.raises(ValueError):
        remove_stopwords([])
    with pytest.raises(ValueError):
        remove_stopwords([1, 2, 3])
    with pytest.raises(ValueError):
        remove_stopwords(["", " "])

def test_remove_stopwords_valid():
    assert remove_stopwords(["this", "is", "a", "test"]) == ["test"]
    assert remove_stopwords(["not", "a", "stopword"]) == ["not", "stopword"]

def test_lemmatize_invalid_input():
    with pytest.raises(ValueError):
        lemmatize("not a list")
    with pytest.raises(ValueError):
        lemmatize([])
    with pytest.raises(ValueError):
        lemmatize([1, 2, 3])
    with pytest.raises(ValueError):
        lemmatize(["", " "])

def test_lemmatize_valid():
    assert lemmatize(["cats", "mice", "geese"]) == ["cat", "mouse", "goose"]

def test_preprocess_invalid_input():
    with pytest.raises(ValueError):
        preprocess(123)
    with pytest.raises(ValueError):
        preprocess("   ")
    with pytest.raises(ValueError):
        preprocess(["not", "a", "string"])
    with pytest.raises(ValueError):
        preprocess("")

def test_preprocess_valid():
    assert preprocess("This is a test!") == "test"
    assert preprocess("Running and jogging") == "running jogging"
    assert preprocess("Not a stopword") == "not stopword"

