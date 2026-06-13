from affectlens.preprocessing import preprocess
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

sia = SentimentIntensityAnalyzer()

def get_processed_text(text: str) -> str:
    """
    Get the cleaned and preprocessed text.

    Args: text (str): Raw text string

    Returns: str: Cleaned and preprocessed text string
    """
    if not text.strip():
        raise ValueError("Input text is empty or whitespace only.")
    return preprocess(text)

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
    Get the weighted sentiment score from both VADER and TextBlob.

    Args: text (str): Raw text string

    Returns: float: Weighted sentiment score between -1 (most negative) and +1 (most positive)
    """
    processed_text = get_processed_text(text)
    if not processed_text.strip():
        return 0.0
    vader = get_vader_score(processed_text)
    textblob = get_textblob_score(processed_text)
    weight = 0.3
    return (vader * (1 - weight) + textblob * weight)


