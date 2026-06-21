import pytest
from affectlens.visualization import plot_sentiment_trend, plot_volatility_trend, plot_emotional_swing_trend, plot_emotion_heatmap, plot_emotion_transition_matrix


def test_plot_sentiment_trend_invalid_input():
    # Test that plot_sentiment_trend raises a ValueError when given invalid input
    with pytest.raises(ValueError):
        plot_sentiment_trend(None)  # Assuming None is invalid input

    with pytest.raises(ValueError):
        plot_sentiment_trend(123)  # Assuming an integer is invalid input

    with pytest.raises(ValueError):
        plot_sentiment_trend([])  # Assuming an empty list is invalid input

    with pytest.raises(ValueError):
        plot_sentiment_trend({})  # Assuming an empty dictionary is invalid input

def test_plot_sentiment_trend_valid_input():
    # Test that plot_sentiment_trend returns a matplotlib figure when given valid input
    enriched_posts = [
        {"High_volatility_flag": False, "Ensemble_score": 0.5},
        {"High_volatility_flag": True, "Ensemble_score": 0.3},
        {"High_volatility_flag": False, "Ensemble_score": 0.7},
    ]
    fig = plot_sentiment_trend(enriched_posts)
    assert fig is not None

def test_plot_volatility_trend_invalid_input():
    # Test that plot_volatility_trend raises a ValueError when given invalid input
    with pytest.raises(ValueError):
        plot_volatility_trend(None, 0.5)  # Assuming None is invalid input

    with pytest.raises(ValueError):
        plot_volatility_trend(123, 0.5)  # Assuming an integer is invalid input

    with pytest.raises(ValueError):
        plot_volatility_trend([], 0.5)  # Assuming an empty list is invalid input

    with pytest.raises(ValueError):
        plot_volatility_trend({}, 0.5)  # Assuming an empty dictionary is invalid input  

    with pytest.raises(ValueError):
        plot_volatility_trend([{"High_volatility_flag": False}], 0.5)  # Assuming a list with missing keys is invalid input   

    with pytest.raises(ValueError):
        plot_volatility_trend([{"High_volatility_flag": False, "Volatility_score": 0.5}], None)  # Assuming None threshold is invalid input

    with pytest.raises(ValueError):
        plot_volatility_trend([{"High_volatility_flag": False, "Volatility_score": 0.5}], "invalid")  # Assuming a non-numeric threshold is invalid input

    with pytest.raises(ValueError):
        plot_volatility_trend([{"High_volatility_flag": False, "Volatility_score": 0.5}], -0.1)  # Assuming a negative threshold is invalid input             

def test_plot_volatility_trend_valid_input():  
    
    # Test that plot_volatility_trend returns a matplotlib figure when given valid input
    enriched_posts = [
        {"High_volatility_flag": False, "Volatility_score": 0.5},
        {"High_volatility_flag": True, "Volatility_score": 0.3},
        {"High_volatility_flag": False, "Volatility_score": 0.7},
    ]
    fig = plot_volatility_trend(enriched_posts, threshold=0.5)
    assert fig is not None      

def test_plot_emotional_swing_trend_invalid_input():
    # Test that plot_emotional_swing_trend raises a ValueError when given invalid input
    with pytest.raises(ValueError):
        plot_emotional_swing_trend(None)  # Assuming None is invalid input

    with pytest.raises(ValueError):
        plot_emotional_swing_trend(123)  # Assuming an integer is invalid input

    with pytest.raises(ValueError):
        plot_emotional_swing_trend([])  # Assuming an empty list is invalid input

    with pytest.raises(ValueError):
        plot_emotional_swing_trend({})  # Assuming an empty dictionary is invalid input

def test_plot_emotional_swing_trend_valid_input():
    # Test that plot_emotional_swing_trend returns a matplotlib figure when given valid input
    enriched_posts = [
        {"Emotional_swing": 0.5, "High_volatility_flag": False},
        {"Emotional_swing": 0.3, "High_volatility_flag": True},
        {"Emotional_swing": 0.7, "High_volatility_flag": False},
    ]
    fig = plot_emotional_swing_trend(enriched_posts)
    assert fig is not None        


def test_plot_emotion_distribution_invalid_input():
    # Test that plot_emotion_distribution raises a ValueError when given invalid input
    with pytest.raises(ValueError):
        plot_emotion_heatmap(None)  # Assuming None is invalid input

    with pytest.raises(ValueError):
        plot_emotion_heatmap(123)  # Assuming an integer is invalid input

    with pytest.raises(ValueError):
        plot_emotion_heatmap([])  # Assuming an empty list is invalid input

    with pytest.raises(ValueError):
        plot_emotion_heatmap({})  # Assuming an empty dictionary is invalid input

def test_plot_emotion_distribution_valid_input():
    # Test that plot_emotion_distribution returns a matplotlib figure when given valid input
    enriched_posts = [
        {"Emotion": "Happy", "High_volatility_flag": False},
        {"Emotion": "Sad", "High_volatility_flag": True},
        {"Emotion": "Angry", "High_volatility_flag": False},
    ]
    fig = plot_emotion_heatmap(enriched_posts)
    assert fig is not None            

def test_plot_emotion_heatmap_invalid_input():
    # Test that plot_emotion_heatmap raises a ValueError when given invalid input
    with pytest.raises(ValueError):
        plot_emotion_heatmap(None)  # Assuming None is invalid input

    with pytest.raises(ValueError):
        plot_emotion_heatmap(123)  # Assuming an integer is invalid input

    with pytest.raises(ValueError):
        plot_emotion_heatmap([])  # Assuming an empty list is invalid input

    with pytest.raises(ValueError):
        plot_emotion_heatmap({})  # Assuming an empty dictionary is invalid input

    with pytest.raises(ValueError):
        plot_emotion_heatmap([{"emotion": "Happy"}])  # Assuming a list with missing keys is invalid input        

def test_plot_emotion_heatmap_valid_input():
    # Test that plot_emotion_heatmap returns a matplotlib figure when given valid input
    enriched_posts = [
        {"Emotion": "Happy", "High_volatility_flag": False},
        {"Emotion": "Sad", "High_volatility_flag": True},
        {"Emotion": "Angry", "High_volatility_flag": False},
    ]
    fig = plot_emotion_heatmap(enriched_posts)
    assert fig is not None        

def test_plot_emotion_transition_matrix_invalid_input():
    # Test that plot_emotion_transition_matrix raises a ValueError when given invalid input
    with pytest.raises(ValueError):
        plot_emotion_transition_matrix(None)  # Assuming None is invalid input

    with pytest.raises(ValueError):
        plot_emotion_transition_matrix(123)  # Assuming an integer is invalid input

    with pytest.raises(ValueError):
        plot_emotion_transition_matrix([])  # Assuming an empty list is invalid input

    with pytest.raises(ValueError):
        plot_emotion_transition_matrix({})  # Assuming an empty dictionary is invalid input

    with pytest.raises(ValueError):
        plot_emotion_transition_matrix([{"Emotion": "Happy"}])  # Assuming a list with missing keys is invalid input


def test_plot_emotion_transition_matrix_valid_input():
    # Test that plot_emotion_transition_matrix returns a matplotlib figure when given valid input
    enriched_posts = [
        {"Emotion": "Happy", "High_volatility_flag": False, "Emotion_scores": {"Happy": 0.8, "Sad": 0.1, "Angry": 0.1}},
        {"Emotion": "Sad", "High_volatility_flag": True, "Emotion_scores": {"Happy": 0.2, "Sad": 0.6, "Angry": 0.2}},
        {"Emotion": "Angry", "High_volatility_flag": False, "Emotion_scores": {"Happy": 0.1, "Sad": 0.1, "Angry": 0.8}},
    ]
    fig = plot_emotion_transition_matrix(enriched_posts)
    assert fig is not None            