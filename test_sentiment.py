import pytest
from affectlens.sentiment import get_vader_score, get_textblob_score, ensemble_score, get_processed_text

def test_get_processed_text_empty():
    with pytest.raises(ValueError):
        get_processed_text("   ")

def test_get_processed_text_valid():
    assert get_processed_text("Hello World!") == "hello world"

def test_get_vader_score():
    assert get_vader_score("I love this!") > 0
    assert get_vader_score("I hate this!") < 0

def test_get_textblob_score():
    assert get_textblob_score("I love this!") > 0
    assert get_textblob_score("I hate this!") < 0

def test_ensemble_score_zero():
    assert ensemble_score("this is it") == 0.0
    assert ensemble_score("i am ") == 0.0
    assert ensemble_score("!!!") == 0.0

def test_ensemble_score_valid():
    score = ensemble_score("I love this!")
    assert 0 < score <= 1
    score = ensemble_score("I hate this!")
    assert -1 <= score < 0    
