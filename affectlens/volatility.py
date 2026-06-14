#Row order is treated as chronological order. V2 will use Reddit API timestamps.

import pandas as pd
import warnings

def calculate_emotional_swing(sentiment_scores: list[float]) -> list[float | None]:
    """
    Calculate the absolute change in sentiment scores between consecutive entries.

    Args: sentiment_scores (list[float]): List of sentiment scores in chronological order

    Returns: list[float | None]: List of absolute changes in sentiment scores between consecutive entries
    """
    if not sentiment_scores:
        raise ValueError("Sentiment scores list is empty.")
    if len(sentiment_scores) < 2:
        warnings.warn("Warning: Sentiment scores list has less than 2 entries. Emotional swing cannot be calculated.")
        return [None]
    data = pd.Series(sentiment_scores)
    emotional_swing = data.diff().abs()
    return [score if not pd.isna(score) else None for score in emotional_swing.tolist()]

def calculate_volatility_score(sentiment_scores: list[float]) -> list[float | None]:
    """
    Calculate the rolling standard deviation of sentiment scores over a window of window_size entries.

    Args: sentiment_scores (list[float]): List of sentiment scores in chronological order

    Returns: list[float | None]: List of rolling standard deviations over a window of window_size entries
    """
    if not sentiment_scores:
        raise ValueError("Sentiment scores list is empty.")
    if len(sentiment_scores) < 5:
        warnings.warn("Warning: Sentiment scores list has less than 5 entries. Volatility score may not be meaningful.")
        return [None] * len(sentiment_scores)
    window_size = 5
    volatility_score = pd.Series(sentiment_scores).rolling(window=window_size).std()
    return [score if not pd.isna(score) else None for score in volatility_score.tolist()]

