import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from emoji import demojize
from affectlens.constants import EMOTICONS

# Download required NLTK data on first run
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("vader_lexicon", quiet=True)

def clean_text(text: str) -> str:
    """
    Removes URLs, HTML entities, punctuation and extra whitespaces and converts text to lowercase.

    Args: text (str): Raw text string

    Returns: str: Lowercase cleaned string with noise removed
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    if not text.strip():
        raise ValueError("Input text is empty or whitespace only.")
    # Replace emoticons with text
    text = demojize(text)
    for emoticon, replacement in EMOTICONS.items():
        text = text.replace(emoticon, replacement)
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
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    if not text.strip():
        raise ValueError("Input text is empty or whitespace only.")
    return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Removes common English stop words from the list of tokens.

    Args: tokens (list[str]): List of word tokens

    Returns: list[str]: List of tokens with stop words removed
    """
    if not isinstance(tokens, list):
        raise ValueError("Input tokens must be a list of strings.")
    if not tokens:
        raise ValueError("Input tokens list is empty.")
    if not all(isinstance(token, str) for token in tokens):
        raise ValueError("All items in the input tokens list must be strings.")
    if tokens[0].strip() == "":
        raise ValueError("Input tokens list contains empty strings.")
    
    negation = set(["not", "no", "never", "n't", "ain't", "nor", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't"])

    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words or word in negation]


def lemmatize(tokens: list[str]) -> list[str]:
    """
    Reduces words to their base or root form using WordNetLemmatizer.

    Args: tokens (list[str]): List of word tokens

    Returns: list[str]: List of lemmatized tokens
    """
    if not isinstance(tokens, list):
        raise ValueError("Input tokens must be a list of strings.")
    if not tokens:
        raise ValueError("Input tokens list is empty.")
    if not all(isinstance(token, str) for token in tokens):
        raise ValueError("All items in the input tokens list must be strings.")
    if tokens[0].strip() == "":
        raise ValueError("Input tokens list contains empty strings.")
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline that cleans, tokenizes, removes stop words and lemmatizes the input text.

    Args: text (str): Raw text string from Reddit posts

    Returns: str: String of preprocessed tokens ready for analysis
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    if not text.strip():
        raise ValueError("Input text is empty or whitespace only.")
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    no_stopwords = remove_stopwords(tokens)
    if not no_stopwords:
        no_stopwords = tokens  # If all tokens are stop words, keep the original tokens to avoid empty input for lemmatization
    lemmatized = lemmatize(no_stopwords)
    return " ".join(lemmatized)
  
