import pytest
from affectlens.sentiment import get_vader_score, get_textblob_score, ensemble_score
from affectlens.constants import WEIGHT

def test_get_vader_score_invalid_input():
    # Empty string
    with pytest.raises(ValueError):
        get_vader_score("   ")
    # Incorrect input type
    with pytest.raises(ValueError):
        get_vader_score(123)        

def test_get_vader_score_valid_input():
    assert get_vader_score("love") > 0
    assert get_vader_score("hate") < 0

def test_get_textblob_score_invalid_input():
    # Empty string
    with pytest.raises(ValueError):
        get_textblob_score("   ")
    # Incorrect input type
    with pytest.raises(ValueError):
        get_textblob_score(123)    

def test_get_textblob_score_valid_input():
    assert get_textblob_score("love") > 0
    assert get_textblob_score("hate") < 0

def test_ensemble_score_invalid_input():
    # Empty string
    with pytest.raises(ValueError):
        ensemble_score("   ", WEIGHT)
    # Incorrect input type
    with pytest.raises(ValueError):
        ensemble_score(123, WEIGHT)

def test_ensemble_score_zero():
    assert ensemble_score("this is it", WEIGHT) == 0.0
    assert ensemble_score("i am ", WEIGHT) == 0.0
    assert ensemble_score("!!!", WEIGHT) == 0.0

def test_ensemble_score_valid():
    score = ensemble_score("I love this!", WEIGHT)
    assert 0 < score <= 1
    score = ensemble_score("I hate this!", WEIGHT)
    assert -1 <= score < 0    
