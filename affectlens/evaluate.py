from affectlens.data_loader import validate_csv, load_labeled_data
from affectlens.classifier import load, predict 
from affectlens.sentiment import ensemble_score, get_vader_score, get_textblob_score
from affectlens.constants import EMOTIONS, THRESHOLD
from sklearn.metrics import f1_score, hamming_loss
from pathlib import Path


def _predictions_to_binary(predictions: list[dict[str, float]], emotions: list[str]) -> list[list[int]]:
    """
    Converts the predictions to binary format.

    Args: predictions (list[dict[str, float]]): List of dictionaries containing predicted probabilities for each emotion, emotions (list[str]): List of emotion categories

    Returns: list[list[int]]: List of lists representing binary predictions for each emotion category
    """
    if not isinstance(predictions, list) or not all(isinstance(prediction, dict) for prediction in predictions):
        raise ValueError("Predictions must be a list of dictionaries.")
    if not isinstance(emotions, list) or not all(isinstance(emotion, str) for emotion in emotions):
        raise ValueError("Emotions must be a list of strings.")
    if not predictions:
        raise ValueError("Predictions list cannot be empty.")
    if not emotions:
        raise ValueError("Emotions list cannot be empty.")
    binary_predictions = []
    for prediction in predictions:
        binary_prediction = [1 if emotion in prediction else 0 for emotion in emotions]
        binary_predictions.append(binary_prediction)
    return binary_predictions        

def evaluate_classifier(test_csv_path: str, required_columns: list[str], model_path: str, vectorizer_path: str, threshold: float) -> dict[str, float]:
    """
    Evaluates the performance of the trained classifier on a test dataset.

    Args: test_csv_path (str): Path to the test CSV file, required_columns (list[str]): List of column names representing the target emotions, model_path (str): Path to the saved trained model, vectorizer_path (str): Path to the saved vectorizer, threshold (float): Threshold for classifying predictions

    Returns: dict[str, float]: Dictionary containing evaluation metrics such as micro f1, macro f1, weighted f1, and hamming loss for each emotion category
    """
    if not isinstance(test_csv_path, str) or not isinstance(model_path, str) or not isinstance(vectorizer_path, str):
        raise ValueError("CSV path, model path, and vectorizer path must be strings.")
    if not test_csv_path.strip() or not model_path.strip() or not vectorizer_path.strip():
        raise ValueError("CSV path, model path, and vectorizer path cannot be empty or whitespace only.")
    if not isinstance(required_columns, list) or not all(isinstance(col, str) for col in required_columns):
        raise ValueError("Required columns must be a list of strings.")
    if not required_columns:
        raise ValueError("Required columns list cannot be empty.")
    if not all(col.strip() for col in required_columns):
        raise ValueError("Required column names cannot be empty or whitespace only.")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a number between 0 and 1.")
    if not Path(test_csv_path).is_file():
        raise ValueError("CSV path must be a valid file path.")
    # Implementation for evaluating the classifier goes here
    # This would typically involve loading the model and vectorizer, preprocessing the test data, making predictions, and calculating evaluation metrics such as accuracy, precision, recall, and F1-score for each emotion category
    validate_csv(test_csv_path, required_columns=required_columns)
    model, vectorizer = load(model_path, vectorizer_path)
    test_data = load_labeled_data(test_csv_path, required_columns)
    texts, true_labels = zip(*test_data)
    texts, true_labels = list(texts), list(true_labels)
    predictions = []
    for text in texts:
        prediction = predict(text, model, vectorizer, threshold)
        predictions.append(prediction)

    predicted_labels = _predictions_to_binary(predictions, EMOTIONS)
    
    # Calculate evaluation metrics based on predictions and true labels
    micro_f1 = f1_score(true_labels, predicted_labels, average="micro")
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
    weighted_f1 = f1_score(true_labels, predicted_labels, average="weighted")
    hamming = hamming_loss(true_labels, predicted_labels)
    evaluation_results = {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "hamming_loss": hamming
    }
    return evaluation_results
    


def optimize_threshold(validation_csv_path: str, target_columns: list[str], model_path: str, vectorizer_path: str) -> tuple[float, float]:
    """
    Optimizes the threshold for classifying predictions based on validation data.

    Args: validation_csv_path (str): Path to the validation CSV file, target_columns (list[str]): List of column names representing the target emotions, model_path (str): Path to the saved trained model, vectorizer_path (str): Path to the saved vectorizer

    Returns: tuple[float, float]: The optimal threshold value and the corresponding evaluation metric (e.g., F1-score) achieved at that threshold
    """
    thresholds = [i / 100 for i in range(0, 101, 5)]  # Thresholds from 0.0 to 1.0 with a step of 0.05
    best_threshold = 0.0
    best_micro_f1 = 0.0
    for threshold in thresholds:
        # Evaluate the model with the current threshold
        eval_results = evaluate_classifier( validation_csv_path, target_columns, model_path, vectorizer_path, threshold)
        micro_f1 = eval_results["micro_f1"]
        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            best_threshold = threshold
    return best_threshold, best_micro_f1


def optimize_ensemble_weight(dataset: list[tuple[str, float]]) -> tuple[float, float]:
    """
    Optimizes the weights for an ensemble of sentiment models based on validation data.

    Args: dataset (list[tuple[str, float]]): The validation dataset containing sentences and their true sentiment scores

    Returns: tuple[float, float]: The optimal weight value and the corresponding evaluation metric (e.g., F1-score) achieved at that weight
    """
    if not isinstance(dataset, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in dataset):
        raise ValueError("Dataset must be a list of tuples containing sentences and true sentiment scores.")
    if not dataset:
        raise ValueError("Dataset cannot be empty.")
    if not all(isinstance(sentence, str) and isinstance(score, (int, float)) for sentence, score in dataset):
        raise ValueError("Each item in the dataset must be a tuple of (sentence: str, true_score: int or float).")
    if not all(0 <= score <= 1 for _, score in dataset):
        raise ValueError("True sentiment scores must be between 0 and 1.")
    weights = [i * 0.05 for i in range(0, 21)]  # Weights from 0.0 to 1.0 with a step of 0.05
    best_weight = 0.0
    lowest_mae = float("inf")
    for weight in weights:
        total_error = 0.0
        for text, true_score in dataset:
            predicted_score = (ensemble_score(text, weight) + 1) / 2
            total_error += abs(predicted_score - true_score)
        mae = total_error / len(dataset)
        if mae < lowest_mae:
            lowest_mae = mae
            best_weight = weight
    return best_weight, lowest_mae


def compare_sentiment_models(dataset: list[tuple[str, float]]) -> dict[str, float]:
    """
    Compares the performance of different sentiment analysis models on a given dataset.

    Args: dataset (list[tuple[str, float]]): The validation dataset containing sentences and their true sentiment scores

    Returns: dict[str, float]: A dictionary mapping model names to their optimal weights and corresponding evaluation metrics
    """
    if not isinstance(dataset, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in dataset):
        raise ValueError("Dataset must be a list of tuples containing sentences and true sentiment scores.")
    if not dataset:
        raise ValueError("Dataset cannot be empty.")
    if not all(isinstance(sentence, str) and isinstance(score, (int, float)) for sentence, score in dataset):
        raise ValueError("Each item in the dataset must be a tuple of (sentence: str, true_score: int or float).")
    if not all(0 <= score <= 1 for _, score in dataset):
        raise ValueError("True sentiment scores must be between 0 and 1.")
    model_comparison_results = {}
    best_weight, _ = optimize_ensemble_weight(dataset)
    vader_errors, textblob_errors, ensemble_errors = 0.0, 0.0, 0.0
    # Code to compare different sentiment models can be implemented here
    for text, true_score in dataset:
        vader_scores = (get_vader_score(text) + 1) / 2  # Normalize VADER score to [0, 1]
        textblob_scores = (get_textblob_score(text) + 1) / 2  # Normalize TextBlob score to [0, 1]
        ensemble_score_values = (ensemble_score(text, best_weight) + 1) / 2  # Normalize ensemble score to [0, 1]

        vader_errors += abs(vader_scores - true_score)
        textblob_errors += abs(textblob_scores - true_score)
        ensemble_errors += abs(ensemble_score_values - true_score)
    vader_mae = vader_errors / len(dataset)
    textblob_mae = textblob_errors / len(dataset)
    ensemble_mae = ensemble_errors / len(dataset)
    model_comparison_results = {
        "VADER": vader_mae,
        "TextBlob": textblob_mae,
        "Ensemble": ensemble_mae
    }
    return model_comparison_results