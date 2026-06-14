from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

sia = SentimentIntensityAnalyzer()


def get_vader_score(text: str) -> float:
    """
    Get the sentiment score using VADER.

    Args: text (str): Preprocessed text string

    Returns: float: Compound sentiment score between -1 (most negative) and +1 (most positive)
    """
    scores = sia.polarity_scores(text)
    return scores["compound"]

def get_textblob_score(text: str) -> float:
    """
    Get the sentiment score using TextBlob.

    Args: text (str): Preprocessed text string

    Returns: float: Sentiment score between -1 (most negative) and +1 (most positive)
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity
    
def ensemble_score(text: str) -> float:
    """
    Get the ensemble sentiment score using both VADER and TextBlob.

    Args: text (str): Preprocessed text string

    Returns: float: Ensemble sentiment score between -1 (most negative) and +1 (most positive)
    """
    vader_score = get_vader_score(text)
    textblob_score = get_textblob_score(text)
    weight = 0.3
    return ((1 - weight) * vader_score) + (weight * textblob_score)





