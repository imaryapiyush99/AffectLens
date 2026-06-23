from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from affectlens.constants import EMOTIONS


def needs_training(model_path: str, vectorizer_path: str) -> bool:
    """
    Checks if the model needs to be trained by checking if the model file and the vectorizer file exist.

    Args: model_path (str): Path to the saved model file, vectorizer_path (str): Path to the saved vectorizer file
    
    Returns: bool: True if either file is missing, False if both files exist
    """
    model_file = Path(model_path)
    vectorizer_file = Path(vectorizer_path)
    if not model_file.exists() or not vectorizer_file.exists():
        return True
    if not model_file.is_file() or not vectorizer_file.is_file():
        return True
    if model_file.stat().st_size == 0 or vectorizer_file.stat().st_size == 0:
        return True
    return False



def train(texts: list[str], labels: list[list[int]], model_path: str, vectorizer_path: str) -> None:
    """
    Trains the sentiment classifier using the provided CSV dataset and saves the model and vectorizer to disk.

    Args: texts (list[str]): List of text strings for training, labels (list[list[int]]): List of lists of corresponding labels for training, model_path (str): Path to save the trained model file, vectorizer_path (str): Path to save the trained vectorizer file
    """
    if not isinstance(model_path, str) or not isinstance(vectorizer_path, str):
        raise ValueError("Model path and vectorizer path must be strings.")
    if not model_path.strip() or not vectorizer_path.strip():
        raise ValueError("Model path and vectorizer path must be provided.")
    if not Path(model_path).parent.exists() or not Path(vectorizer_path).parent.exists():
        raise FileNotFoundError("Model path or vectorizer path directory does not exist. Please provide valid paths.")
    if not texts or not labels:
        raise ValueError("Training texts and labels must be provided.")
    if len(texts) != len(labels):
        raise ValueError("The number of training texts and labels must be the same.")
    # Implementation for training the model goes here
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(texts)

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf = MultiOutputClassifier(model)
    clf.fit(feature_matrix, labels)

    # Save the trained model and vectorizer to disk
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)


def load(model_path: str, vectorizer_path: str) -> tuple[MultiOutputClassifier, TfidfVectorizer]:
    """
    Returns the loaded model and vectorizer from disk.

    Args: model_path (str): Path to the saved model file, vectorizer_path (str): Path to the saved vectorizer file

    Returns: tuple[MultiOutputClassifier, TfidfVectorizer]: The loaded model and vectorizer objects
    """
    if not Path(model_path).exists() or not Path(vectorizer_path).exists():
        raise FileNotFoundError("Model file or vectorizer file not found. Please train the model first.")
    if not Path(model_path).is_file() or not Path(vectorizer_path).is_file():
        raise ValueError("Model path or vectorizer path is not a file. Please provide valid file paths.")
    if Path(model_path).stat().st_size == 0 or Path(vectorizer_path).stat().st_size == 0:
        raise ValueError("Model file or vectorizer file is empty. Please train the model again.")
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return clf, vectorizer



def predict(text: str, clf: MultiOutputClassifier, vectorizer: TfidfVectorizer, threshold: float) -> dict[str, float]:
    """
    Predicts the sentiment of the given text and returns the predicted label and confidence score.

    Args: text (str): The input text string to classify, clf (MultiOutputClassifier): The loaded model object, vectorizer (TfidfVectorizer): The loaded vectorizer object, threshold (float): The probability threshold for making predictions

    Returns: dict[str, float]: Dictionary mapping predicted sentiment labels to their confidence scores
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    if not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a numeric value.")
    if not text.strip():
        raise ValueError("Input text is empty or whitespace only.")
    if clf is None or vectorizer is None:
        raise ValueError("Model and vectorizer must be loaded before prediction.")
    if not isinstance(clf, MultiOutputClassifier):
        raise ValueError("Model must be an instance of MultiOutputClassifier.")
    if not isinstance(vectorizer, TfidfVectorizer):
        raise ValueError("Vectorizer must be an instance of TfidfVectorizer.")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    feature_vector = vectorizer.transform([text])
    probabilities = clf.predict_proba(feature_vector)
    predictions = {}
    best_emotion = None
    best_probability = 0.0
    for i, emotion in enumerate(EMOTIONS):
        proba = probabilities[i][0]
        if len(proba) < 2:
            positive_probability = proba[0]
        else:
            positive_probability = proba[1]
        if positive_probability >= threshold:
            predictions[emotion] = float(positive_probability)
        if positive_probability > best_probability:
            best_probability = positive_probability
            best_emotion = emotion
    if not predictions and best_emotion is not None:
        predictions[best_emotion] = float(best_probability)

    return predictions






