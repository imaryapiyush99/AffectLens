import pytest
from affectlens.preprocessing import clean_text, tokenize, remove_stopwords, lemmatize, preprocess, get_processed_text

def test_clean_text():
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("This is a test.") == "this is a test"
    assert clean_text("12345") == "12345"
    assert clean_text("Special characters: @#$%^&*()") == "special characters"
    assert clean_text("   Extra whitespace   ") == "extra whitespace"
    assert clean_text("HTML entities &amp; &lt; &gt;") == "html entities"
    assert clean_text("Visit http://example.com today") == "visit today"

def test_tokenize():
    assert tokenize("hello world") == ["hello", "world"]
    assert tokenize("this is a test") == ["this", "is", "a", "test"]

def test_remove_stopwords():
    assert remove_stopwords(["this", "is", "a", "test"]) == ["test"]
    assert remove_stopwords(["not", "a", "stopword"]) == ["not", "stopword"]

def test_lemmatize():
    assert lemmatize(["cats", "mice", "geese"]) == ["cat", "mouse", "goose"]

def test_preprocess():
    assert preprocess("This is a test!") == "test"
    assert preprocess("Running and jogging") == "running jogging"
    assert preprocess("Not a stopword") == "not stopword"

def test_get_processed_text_empty():
    with pytest.raises(ValueError):
        get_processed_text("")
    with pytest.raises(ValueError):        
        get_processed_text("   ")    

def test_get_processed_text_valid():
    assert get_processed_text("This is a test!") == "test"
    assert get_processed_text("Running and jogging") == "running jogging"
    assert get_processed_text("Not a stopword") == "not stopword"  
    assert get_processed_text("The and a") == ""
    assert get_processed_text("https://example.com") == ""
    assert get_processed_text("HTML &amp; entities") == "html entity"