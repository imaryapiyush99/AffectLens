from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

sia = SentimentIntensityAnalyzer()

def get_vader_score(text: str) -> float:
    """
    Get the sentiment score using VADER.

    Args: text (str): Preprocessed text string

    Returns: float: Compound sentiment score between -1 (most negative) and +1 (most positive)
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    if not text.strip():
        raise ValueError("Input text is empty or whitespace only.")
    scores = sia.polarity_scores(text)
    return scores["compound"]

def get_textblob_score(text: str) -> float:
    """
    Get the sentiment score using TextBlob.

    Args: text (str): Preprocessed text string

    Returns: float: Sentiment score between -1 (most negative) and +1 (most positive)
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    if not text.strip():
        raise ValueError("Input text is empty or whitespace only.")
    blob = TextBlob(text)
    return blob.sentiment.polarity
    
def ensemble_score(text: str, weight: float) -> float:
    """
    Get the ensemble sentiment score using both VADER and TextBlob.

    Args: text (str): Preprocessed text string

    Returns: float: Ensemble sentiment score between -1 (most negative) and +1 (most positive)
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    if not isinstance(weight, (int, float)):
        raise ValueError("Weight must be a number between 0 and 1.")
    if not 0 <= weight <= 1:
        raise ValueError("Weight must be between 0 and 1.")
    if not text.strip():
        raise ValueError("Input text is empty or whitespace only.")
    vader_score = get_vader_score(text)
    textblob_score = get_textblob_score(text)
    return ((1 - weight) * vader_score) + (weight * textblob_score)





