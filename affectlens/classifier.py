from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from data_loader import load_training_data


LABELS = {
    0: "Stress",
    1: "Depression",
    2: "Bipolar disorder",
    3: "Personality disorder",
    4: "Anxiety"
}

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
    return not (model_file.exists() and vectorizer_file.exists())



def train(csv_path: str, model_path: str, vectorizer_path: str) -> None:
    """
    Trains the sentiment classifier using the provided CSV dataset and saves the model and vectorizer to disk.

    Args: csv_path (str): Path to the CSV file containing the training data with 'text', 'target' and 'title'(optional) columns, model_path (str): Path to save the trained model file, vectorizer_path (str): Path to save the trained vectorizer file
    """
    texts, combined_labels = load_training_data(csv_path)
    if not model_path.strip() or not vectorizer_path.strip():
        raise ValueError("Model path and vectorizer path must be provided.")
    if not isinstance(model_path, str) or not isinstance(vectorizer_path, str):
        raise ValueError("Model path and vectorizer path must be strings.")
    if not Path(model_path).parent.exists() or not Path(vectorizer_path).parent.exists():
        raise FileNotFoundError("Model path or vectorizer path directory does not exist. Please provide valid paths.")
    # Implementation for training the model goes here
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(feature_matrix, combined_labels)

    # Save the trained model and vectorizer to disk
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)


def load(model_path: str, vectorizer_path: str) -> tuple[LogisticRegression, TfidfVectorizer]:
    """
    Returns the loaded model and vectorizer from disk.

    Args: model_path (str): Path to the saved model file, vectorizer_path (str): Path to the saved vectorizer file

    Returns: tuple[LogisticRegression, TfidfVectorizer]: The loaded model and vectorizer objects
    """
    if not Path(model_path).exists() or not Path(vectorizer_path).exists():
        raise FileNotFoundError("Model file or vectorizer file not found. Please train the model first.")
    if not Path(model_path).is_file() or not Path(vectorizer_path).is_file():
        raise ValueError("Model path or vectorizer path is not a file. Please provide valid file paths.")
    if Path(model_path).stat().st_size == 0 or Path(vectorizer_path).stat().st_size == 0:
        raise ValueError("Model file or vectorizer file is empty. Please train the model again.")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer



def predict(text: str, model: LogisticRegression, vectorizer: TfidfVectorizer) -> tuple[str, float]:
    """
    Predicts the sentiment of the given text and returns the predicted label and confidence score.

    Args: text (str): The input text string to classify, model (LogisticRegression): The loaded model object, vectorizer (TfidfVectorizer): The loaded vectorizer object

    Returns: tuple[str, float]: The predicted sentiment label ('Stress', 'Depression', 'Bipolar disorder', 'Personality disorder', 'Anxiety') and the confidence score (0.0 to 1.0)
    """
    if not text.strip():
        raise ValueError("Input text is empty or whitespace only.")
    if model is None or vectorizer is None:
        raise ValueError("Model and vectorizer must be loaded before prediction.")
    if not isinstance(model, LogisticRegression):
        raise ValueError("Model must be an instance of LogisticRegression.")
    if not isinstance(vectorizer, TfidfVectorizer):
        raise ValueError("Vectorizer must be an instance of TfidfVectorizer.")
    feature_vector = vectorizer.transform([text])
    predictions = model.predict(feature_vector)
    probabilities = model.predict_proba(feature_vector)
    classifier_score = max(probabilities[0])
    predicted_label = LABELS[predictions[0]]
    return predicted_label, classifier_score



