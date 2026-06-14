from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from data_loader import load_training_data
from data_loader import load_dataset


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
    return not (model_file.exists() and vectorizer_file.exists())



def train(csv_path: str, model_path: str, vectorizer_path: str) -> None:
    """
    Trains the sentiment classifier using the provided CSV dataset and saves the model and vectorizer to disk.

    Args: csv_path (str): Path to the CSV file containing the training data with 'text', 'target' and 'title'(optional) columns, model_path (str): Path to save the trained model file, vectorizer_path (str): Path to save the trained vectorizer file
    """
    texts, combined_labels = load_training_data(csv_path)

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
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer



def predict(text: str, model: LogisticRegression, vectorizer: TfidfVectorizer) -> tuple[str, float]:
    """
    Predicts the sentiment of the given text and returns the predicted label and confidence score.

    Args: text (str): The input text string to classify, model (LogisticRegression): The loaded model object, vectorizer (TfidfVectorizer): The loaded vectorizer object

    Returns: tuple[str, float]: The predicted sentiment label ('Stress', 'Depression', 'Bipolar disorder', 'Personality disorder', 'Anxiety') and the confidence score (0.0 to 1.0)
    """
    feature_vector = vectorizer.transform([text])
    predictions = model.predict(feature_vector)
    probabilities = model.predict_proba(feature_vector)
    classifier_score = max(probabilities[0])
    predicted_label = LABELS[predictions[0]]
    return predicted_label, classifier_score

def get_predictions(csv_path: str, model_path: str, vectorizer_path: str) -> list[dict[str, str | float]]:
    """
    Gets the predictions for each text in the dataset and returns a list of dictionaries with "text", "title", "predicted_label", "classifier_score" keys.

    Args: csv_path (str): Path to the CSV file containing the dataset with 'text', 'title'(optional), and 'target' columns, model_path (str): Path to the saved model file, vectorizer_path (str): Path to the saved vectorizer file

    Returns: list[dict[str, str | float]]: A list of dictionaries where each dictionary has "text", "title", "predicted_label", "classifier_score" keys
    """
    dataset = load_dataset(csv_path)
    results = []
    model, vectorizer = load(model_path, vectorizer_path)
    for item in dataset:
        text = item["text"]
        title = item.get("title", "")
        predicted_label, classifier_score = predict(text, model, vectorizer)
        results.append({"text": text, "title": title, "predicted_label": predicted_label, "classifier_score": classifier_score})
    return results


