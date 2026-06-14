import pytest
from affectlens.sentiment import get_vader_score, get_textblob_score, ensemble_score

def test_get_vader_score():
    assert get_vader_score("love") > 0
    assert get_vader_score("hate") < 0

def test_get_textblob_score():
    assert get_textblob_score("love") > 0
    assert get_textblob_score("hate") < 0

def test_ensemble_score_empty():
    with pytest.raises(ValueError):
        ensemble_score("   ")    

def test_ensemble_score_zero():
    assert ensemble_score("this is it") == 0.0
    assert ensemble_score("i am ") == 0.0
    assert ensemble_score("!!!") == 0.0

def test_ensemble_score_valid():
    score = ensemble_score("I love this!")
    assert 0 < score <= 1
    score = ensemble_score("I hate this!")
    assert -1 <= score < 0    
