import argparse
from email.mime import text
from pathlib import Path
from affectlens.constants import THRESHOLD, WEIGHT, TARGET_COLUMNS, PREDICTION_COLUMNS, OUTPUT_COLUMNS
from affectlens.data_loader import  edit_training_csv, load_dataset, load_training_data, save_results
from affectlens.preprocessing import preprocess
from affectlens.classifier import needs_training, train, load, predict
from affectlens.sentiment import ensemble_score
from affectlens.volatility import calculate_emotional_swing, calculate_volatility_score, high_volatility_flag


def get_preprocessed_text(dataset: list[dict[str, str | int]]) -> list[str]:
    """
    Preprocess the text data from a CSV file.

    Args:
        dataset (list[dict[str, str | int]]): List of dictionaries containing the dataset

    Returns:
        list[str]: List of preprocessed text strings.
    """
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list of dictionaries.")
    if not dataset:
        raise ValueError("Dataset is empty.")
    if not all(isinstance(item, dict) for item in dataset):
        raise ValueError("All items in the dataset must be dictionaries.")
    if not dataset[0].get("text"):
        raise ValueError("Dataset dictionaries must contain a 'text' key.")
    preprocessed_texts = []
    for item in dataset:
        try:
            preprocessed_text = preprocess(item["text"])
            preprocessed_texts.append(preprocessed_text)
        except ValueError as e:
            print("FAILED:", repr(item["text"]))
            print("ERROR:", e)
    return preprocessed_texts


def get_ensemble_scores(texts: list[str], weight: float) -> list[float]:
    """"
    Get the ensemble sentiment scores for a list of preprocessed text strings.

    Args: texts (list[str]): List of preprocessed text strings, weight (float): Weight for combining VADER and TextBlob scores

    Results: list[float]: List of ensemble sentiment scores between -1 (most negative) and +1 (most positive)
    """
    if not isinstance(texts, list):
        raise ValueError("Input texts must be a list of strings.")
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("All items in the input text list must be strings.")
    if not texts:
        raise ValueError("Input text list is empty.")
    if not isinstance(weight, (int, float)):
        raise ValueError("Weight must be a number.")
    if not 0 <= weight <= 1:
        raise ValueError("Weight must be between 0 and 1.")
    ensemble_scores = []
    for text in texts:
        score = ensemble_score(text, weight)
        ensemble_scores.append(score)
    return ensemble_scores    


def get_classifier_predictions(texts: list[str], model_path: str, vectorizer_path: str, threshold: float) -> list[list[str]]:
    """"
    Get the classifier predictions and confidence scores for a list of preprocessed text strings.

    Args: texts (list[str]): List of preprocessed text strings, model_path (str): Path to the saved model file, vectorizer_path (str): Path to the saved vectorizer file, threshold (float): Probability threshold for making predictions

    Results: list[tuple[str, float]]: List of tuples containing the predicted sentiment label and confidence score for each input text
    """
    if not isinstance(texts, list):
        raise ValueError("Input texts must be a list of strings.")
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("All items in the input text list must be strings.")
    if not texts:
        raise ValueError("Input text list is empty.")
    if any(not text.strip() for text in texts):
        raise ValueError("Input texts cannot be empty.")
    if not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a number.")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if not isinstance(model_path, str) or not isinstance(vectorizer_path, str):
        raise ValueError("Model path and vectorizer path must be strings.")
    if not model_path.strip() or not vectorizer_path.strip():
        raise ValueError("Model path and vectorizer path must be provided.")
    if not Path(model_path).is_file() or not Path(vectorizer_path).is_file():
        raise ValueError("Model path and vectorizer path must be valid file paths.")
    model, vectorizer = load(model_path, vectorizer_path)
    predictions = []
    for text in texts:
        prediction = predict(text, model, vectorizer, threshold)
        predicts = [f"{emotion} ({score:.2f})" for emotion, score in prediction.items()]
        predictions.append(predicts)
    return predictions


def get_emotional_swing(texts: list[str], sentiment_scores: list[float]) -> list[float | None]:
    """
    Get the emotional swing scores for a list of sentiment scores.

    Args: texts (list[str]): List of input texts, sentiment_scores (list[float]): List of sentiment scores for each input text

    Returns: list[float | None]: List of emotional swing scores for each input text, where None indicates insufficient data to calculate swing
    """
    if not isinstance(texts, list):
        raise ValueError("Input texts must be a list of strings.")
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("All items in the input text list must be strings.")
    if not texts:
        raise ValueError("Input text list is empty.")
    if not texts[0].strip():
        raise ValueError("Input texts cannot be empty.")
    if not isinstance(sentiment_scores, list):
        raise ValueError("Sentiment scores must be a list of floats.")
    if not sentiment_scores:
        raise ValueError("Sentiment scores list is empty.")
    if len(texts) != len(sentiment_scores):
        raise ValueError("Texts and sentiment scores must have the same length.")
    emotional_swings = calculate_emotional_swing(sentiment_scores)
    return emotional_swings


def get_volatility_scores(texts: list[str], sentiment_scores: list[float]) -> list[ float | None]:    
    """
    Get the volatility scores for a list of sentiment scores.

    Args: texts (list[str]): List of input texts, sentiment_scores (list[float]): List of sentiment scores for each input text

    Returns: list[float | None]: List of volatility scores for each input text, where None indicates insufficient data to calculate volatility
    """
    if not isinstance(texts, list):
        raise ValueError("Input texts must be a list of strings.")
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("All items in the input text list must be strings.")
    if not texts:
        raise ValueError("Input text list is empty.")
    if not texts[0].strip():
        raise ValueError("Input texts cannot be empty.")
    if not isinstance(sentiment_scores, list):
        raise ValueError("Sentiment scores must be a list of floats.")
    if not sentiment_scores:
        raise ValueError("Sentiment scores list is empty.")
    if len(texts) != len(sentiment_scores):
        raise ValueError("Texts and sentiment scores must have the same length.")
    volatility_scores = calculate_volatility_score(sentiment_scores)
    return volatility_scores


def get_high_volatility_flags(texts: list[str], volatility_scores: list[float | None], threshold: float) -> list[bool | None]:
    """
    Get the high volatility flags for a list of volatility scores based on a specified threshold.

    Args: texts (list[str]): List of input texts, volatility_scores (list[float | None]): List of volatility scores for each input text, threshold (float): Threshold for flagging high volatility

    Returns: list[bool | None]: List of high volatility flags for each input text, where None indicates insufficient data to determine high volatility
    """
    if not isinstance(texts, list):
        raise ValueError("Input texts must be a list of strings.")
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("All items in the input text list must be strings.")
    if not texts:
        raise ValueError("Input text list is empty.")
    if not texts[0].strip():
        raise ValueError("Input texts cannot be empty.")
    if not isinstance(volatility_scores, list):
        raise ValueError("Volatility scores must be a list of floats or None.")
    if not volatility_scores:
        raise ValueError("Volatility scores list is empty.")
    if len(texts) != len(volatility_scores):
        raise ValueError("Texts and volatility scores must have the same length.")
    if not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a number.")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    high_volatility_flags = []
    for i in range(len(texts)):
        volatility_flag = high_volatility_flag(volatility_scores[i], threshold)
        high_volatility_flags.append(volatility_flag)
    return high_volatility_flags


def build_enriched_posts(required_columns: list[str], model_path: str, vectorizer_path: str, weight: float, threshold: float, csv_path: str | None = None, text: str | None = None) -> list[dict[str, str | int | float | None | list[str | float]]]:
    """
    Build enriched posts with sentiment scores, classifier predictions, emotional swing, and volatility scores.

    Args: required_columns (list[str]): List of column names to include in the loaded dataset, model_path (str): Path to the saved model file, vectorizer_path (str): Path to the saved vectorizer file, weight (float): Weight for the sentiment scores, threshold (float): Threshold for the classifier predictions, csv_path (str): Path to the input CSV file containing the dataset with 'text', 'target' and 'title'(optional) columns, text (str): Single text input for processing

    Returns: list[dict[str, str | int | float | None | dict[str, float]]]: List of enriched posts with sentiment scores, classifier predictions, emotional swing, and volatility scores
    """
    # Implementation for building enriched posts goes here
    if not isinstance(required_columns, list) or not all(isinstance(col, str) for col in required_columns):
        raise ValueError("Required columns must be a list of strings.")
    if not required_columns:
        raise ValueError("Required columns list is empty.")
    if not isinstance(model_path, str) or not isinstance(vectorizer_path, str):
        raise ValueError("Model path and vectorizer path must be strings.")
    if not model_path.strip() or not vectorizer_path.strip():
        raise ValueError("Model path and vectorizer path must be provided.")
    if csv_path is None and text is None:
        raise ValueError("Either csv_path or text must be provided.")
    if csv_path is None and text is not None:
        dataset = [{"text": text}]
    else:
        dataset = load_dataset(csv_path, required_columns)
    if csv_path is not None and text is not None:
        raise ValueError("Provide either csv_path or text, not both.")
    if csv_path is not None and not isinstance(csv_path, str):
        raise ValueError("CSV path must be a string.")
    if not isinstance(model_path, str) or not isinstance(vectorizer_path, str):
        raise ValueError("Model path and vectorizer path must be strings.")

    if (csv_path is not None and not Path(csv_path).is_file()) or not Path(model_path).is_file() or not Path(vectorizer_path).is_file():
        raise FileNotFoundError("One or more specified paths are not valid files. Please provide valid file paths.")
    if not isinstance(weight, (int, float)) or not isinstance(threshold, (int, float)):
        raise ValueError("Weight and threshold must be numbers.")
    if not 0 <= weight <= 1 or not 0 <= threshold <= 1:
        raise ValueError("Weight and threshold must be between 0 and 1.")
    preprocessed_texts = get_preprocessed_text(dataset)
    ensemble_scores = get_ensemble_scores(preprocessed_texts, weight)
    classifier_predictions = get_classifier_predictions(preprocessed_texts, model_path, vectorizer_path, threshold)
    emotional_swing_results = get_emotional_swing(preprocessed_texts, ensemble_scores)
    volatility_scores = get_volatility_scores(preprocessed_texts, ensemble_scores)
    high_volatility_flags = get_high_volatility_flags(preprocessed_texts, volatility_scores, threshold)


    # Combine all the results into a single list of dictionaries
    enriched_posts = []
    for i in range(len(dataset)):
        enriched_post = {
            "Text": dataset[i]["text"],
            "Ensemble_score": ensemble_scores[i],
            "Emotion": classifier_predictions[i],
            "Emotional_swing": emotional_swing_results[i],
            "Volatility_score": volatility_scores[i],
            "High_volatility_flag": high_volatility_flags[i]
        }    
        enriched_posts.append(enriched_post)

    return enriched_posts



def main() -> None:
    """
    Main function to execute the AffectLens project workflow. This includes loading the dataset, preprocessing the text, training the classifier, saving the results, and calculating emotional volatility metrics.

    Steps:
    1. Load the dataset from a CSV file.
    2. Preprocess the text data (e.g., tokenization, stopword removal, lemmatization).
    3. Train a logistic regression classifier on the preprocessed data.
    4. Save the trained model and vectorizer to disk for future use.
    5. Optionally, evaluate the model on a test set and save the results to a CSV file.
    6. Calculate emotional swing and volatility scores for the test set and include them in the results.
    7. Handle any exceptions that may occur during the workflow and provide informative error messages to the user.
    8. Ensure that all file paths and inputs are validated before processing to prevent runtime errors.
    """
    # Implementation of the main workflow goes here

    # Argument parsing for command-line execution
    parser = argparse.ArgumentParser(description="AffectLens: A tool for analyzing emotional volatility in text data.")
    parser.add_argument("--text", type=str, required=False, help="Input text string for sentiment and volatility analysis.")
    parser.add_argument("--required_columns", type=str, nargs="+", default=TARGET_COLUMNS, help="List of target columns to include in the dataset.")
    parser.add_argument("--input_csv", type=str, required=False, help="Path to the input CSV file containing the dataset with 'text' column.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save the trained model file.")
    parser.add_argument("--vectorizer_path", type=str, required=True, help="Path to save the trained vectorizer file.")
    parser.add_argument("--output_csv", type=str, required=False, help="Path to save the results CSV file with enriched data and metrics.")
    parser.add_argument("--weight", type=float, default=WEIGHT, help="Weight for combining VADER and TextBlob sentiment scores.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Threshold for classifying high volatility.")
    args = parser.parse_args()

    # Call the main workflow function with the provided arguments
    final_input_csv = Path(args.input_csv).expanduser().resolve() if args.input_csv else None  
    final_model_path = Path(args.model_path).expanduser().resolve()
    final_vectorizer_path = Path(args.vectorizer_path).expanduser().resolve()
    final_output_csv =Path(args.output_csv).expanduser().resolve() if args.output_csv else None

    try:
        if needs_training(args.model_path, args.vectorizer_path) and args.input_csv:
            training_csv = edit_training_csv(args.input_csv, "/Users/imaryapiyush99/AffectLens/data/training_data/training_data.csv")
            training_dataset = load_training_data(training_csv, args.required_columns)
            texts, labels = map(list, zip(*training_dataset))
            train(texts, labels, args.model_path, args.vectorizer_path)
            enriched_posts = build_enriched_posts(csv_path=str(final_input_csv) if final_input_csv else None, text=args.text, required_columns=args.required_columns, model_path=str(final_model_path), vectorizer_path=str(final_vectorizer_path), weight=args.weight, threshold=args.threshold)
        elif needs_training(args.model_path, args.vectorizer_path) and not args.input_csv:
            raise ValueError("Model and vectorizer need to be trained, but no input CSV file was provided for training. Please provide a valid input CSV file.")    
        else:
            enriched_posts = build_enriched_posts(csv_path=str(final_input_csv) if final_input_csv else None, text=args.text, required_columns=PREDICTION_COLUMNS, model_path=str(final_model_path), vectorizer_path=str(final_vectorizer_path), weight=args.weight, threshold=args.threshold)
        if enriched_posts and final_output_csv:
            save_results(str(final_output_csv), OUTPUT_COLUMNS, enriched_posts)
        else:    
            print(enriched_posts)    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except FileNotFoundError as fnfe:
        print(f"FileNotFoundError: {fnfe}")

if __name__ == "__main__":
    main()