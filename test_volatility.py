import pytest
from affectlens.volatility import calculate_emotional_swing, calculate_volatility_score, high_volatility_flag

def test_calculate_emotional_swing_empty_list():
    with pytest.raises(ValueError):
        calculate_emotional_swing([])

def test_calculate_emotional_swing_single_entry():
    result = calculate_emotional_swing([0.5])
    assert result == [None]

def test_calculate_emotional_swing_multiple_entries():
    sentiment_scores = [0.1, 0.4, 0.2, 0.5]
    expected_swing = [None, 0.3, 0.2, 0.3]
    result = calculate_emotional_swing(sentiment_scores)
    assert [round(x, 1) if x is not None else None for x in result] == expected_swing

def test_calculate_volatility_score_empty_list():
    with pytest.raises(ValueError):
        calculate_volatility_score([])

def test_calculate_volatility_score_insufficient_entries():
    sentiment_scores = [0.1, 0.4, 0.2]
    result = calculate_volatility_score(sentiment_scores)
    assert result == [None, None, None]

def test_calculate_volatility_score_sufficient_entries():
    sentiment_scores = [0.1, 0.4, 0.2, 0.5, 0.3, 0.6]
    result = calculate_volatility_score(sentiment_scores)
    assert len(result) == len(sentiment_scores)
    assert all(isinstance(score, (float, type(None))) for score in result)

def test_high_volatility_flag_invalid_input():
    with pytest.raises(ValueError):
        high_volatility_flag(None, threshold=0.5)
    with pytest.raises(ValueError):
        high_volatility_flag("invalid_score", threshold=0.5)
    with pytest.raises(ValueError):
        high_volatility_flag(0.6, threshold="invalid_threshold")
    with pytest.raises(ValueError):
        high_volatility_flag(0.6, threshold=-0.1)

def test_high_volatility_flag_valid_input():
    assert high_volatility_flag(0.6, threshold=0.5) == True
    assert high_volatility_flag(0.4, threshold=0.5) == False