import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data on first run
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

def clean_text(text: str) -> str:
    """
    Removes URLs, HTML entities, punctuation and extra whitespaces and converts text to lowercase.

    Args: text (str): Raw text string from Reddit posts

    Returns: str: Lowercase cleaned string with noise removed
    """
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"((https?://)|(www\.))\S+", "", text)
    # Remove HTML entities
    text = re.sub(r"&\S+;", "", text)
    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> list[str]:
    """
    Splits cleaned text into individual words (tokens).

    Args: text (str): Cleaned text string

    Returns: list[str]: List of word tokens
    """
    return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Removes common English stop words from the list of tokens.

    Args: tokens (list[str]): List of word tokens

    Returns: list[str]: List of tokens with stop words removed
    """
    negation = set(["not", "no", "never", "n't", "ain't", "nor", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't"])

    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words or word in negation]


def lemmatize(tokens: list[str]) -> list[str]:
    """
    Reduces words to their base or root form using WordNetLemmatizer.

    Args: tokens (list[str]): List of word tokens

    Returns: list[str]: List of lemmatized tokens
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline that cleans, tokenizes, removes stop words and lemmatizes the input text.

    Args: text (str): Raw text string from Reddit posts

    Returns: str: String of preprocessed tokens ready for analysis
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    no_stopwords = remove_stopwords(tokens)
    lemmatized = lemmatize(no_stopwords)
    return " ".join(lemmatized)
  