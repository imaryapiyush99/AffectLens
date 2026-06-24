import argparse, platform, os, subprocess 
from pathlib import Path
from collections import Counter
from affectlens.constants import THRESHOLD, WEIGHT, TARGET_COLUMNS, PREDICTION_COLUMNS, OUTPUT_COLUMNS
from affectlens.data_loader import  edit_training_csv, split_dataset, load_dataset, load_labeled_data, save_results, load_sst_dataset
from affectlens.preprocessing import preprocess
from affectlens.classifier import needs_training, train, load, predict
from affectlens.sentiment import ensemble_score
from affectlens.volatility import calculate_emotional_swing, calculate_volatility_score, high_volatility_flag
from affectlens.evaluate import evaluate_classifier, optimize_threshold, optimize_ensemble_weight, compare_sentiment_models, get_classification_report
from affectlens.visualization import plot_sentiment_trend, plot_volatility_trend, plot_emotional_swing_trend, plot_emotion_distribution, plot_emotion_heatmap, plot_emotion_transition_matrix



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


def get_classifier_predictions(texts: list[str], model_path: str, vectorizer_path: str, threshold: float) -> tuple[list[list[str]], list[dict[str, float]]]:
    """"
    Get the classifier predictions and confidence scores for a list of preprocessed text strings.

    Args: texts (list[str]): List of preprocessed text strings, model_path (str): Path to the saved model file, vectorizer_path (str): Path to the saved vectorizer file, threshold (float): Probability threshold for making predictions

    Results: tuple[list[list[str]], list[dict[str, float]]]: A tuple containing two lists: the first list contains the predicted emotion labels for each input text, and the second list contains the confidence scores for each emotion label for each input text.
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
    classifier_scores = []
    for text in texts:
        prediction = predict(text, model, vectorizer, threshold)
        classifier_scores.append(prediction)
        predictions.append((list(prediction.keys())))
    return predictions, classifier_scores


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


def build_enriched_posts(required_columns: list[str], model_path: str, vectorizer_path: str, weight: float, threshold: float, csv_path: str | None = None, text: str | None = None) -> list[dict[str, str | int | float | None | list[str | float] | dict[str, float]]]:
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
    classifier_predictions, classifier_scores = get_classifier_predictions(preprocessed_texts, model_path, vectorizer_path, threshold)
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
            "Emotion_scores": classifier_scores[i],
            "Emotional_swing": emotional_swing_results[i],
            "Volatility_score": volatility_scores[i],
            "High_volatility_flag": high_volatility_flags[i]
        }    
        enriched_posts.append(enriched_post)

    return enriched_posts

def dataset_is_training_data(csv_path: str) -> bool:
    """
    Check if the provided CSV file is one of the training datasets.
    
    Args: csv_path (str): Path to the CSV file to check

    Returns: bool: True if the CSV file is one of the training datasets, False otherwise
    """
    if not isinstance(csv_path, str):
        raise ValueError("CSV path must be a string.")
    if not csv_path.strip():
        raise ValueError("CSV path must be provided.")
    
    csv_file = Path(csv_path).resolve()
    training_paths = [
        (Path(__file__).parent / "data" / "go_emotions_dataset.csv").resolve(),
        (Path(__file__).parent / "data" / "converted.csv").resolve(),
        (Path(__file__).parent / "data" / "training_data" / "train.csv").resolve(),
    ]
    return csv_file in training_paths

def sst_dataset_available(data_dir: Path) -> bool:
    """
    Check if the SST dataset is available in the specified directory.

    Args: data_dir (Path): Path to the directory containing the SST dataset

    Returns: bool: True if all required files are present, False otherwise
    """
    required_files = [
        "datasetSentences.txt",
        "datasetSplit.txt",
        "dictionary.txt",
        "sentiment_labels.txt",
    ]

    return all(
        (data_dir / "SST" / file).is_file()
        for file in required_files
    )

def print_summary(enriched_posts: list[dict[str, str | int | float | None | dict[str, float]]]) -> None:
    """
    Print a summary of the enriched posts, including counts of emotions, average sentiment scores, and volatility metrics.

    Args: enriched_posts (list[dict[str, str | int | float | None | dict[str, float]]]): List of enriched posts with sentiment scores, classifier predictions, emotional swing, and volatility scores
    """
    if not isinstance(enriched_posts, list):
        raise ValueError("Enriched posts must be a list of dictionaries.")
    if not enriched_posts:
        raise ValueError("Enriched posts list is empty.")
    if not all(isinstance(post, dict) for post in enriched_posts):
        raise ValueError("All items in the enriched posts list must be dictionaries.")
    total_posts = len(enriched_posts)

    sentiments= [post["Ensemble_score"] for post in enriched_posts if post["Ensemble_score"] is not None]
    volatilities = [post["Volatility_score"] for post in enriched_posts if post["Volatility_score"] is not None]
    swings = [post["Emotional_swing"] for post in enriched_posts if post["Emotional_swing"] is not None]
    emotions = [emotion for post in enriched_posts for emotion in post["Emotion"] if emotion is not None]
    
    high_volatility_count = sum(1 for post in enriched_posts if post["High_volatility_flag"] is True)
    average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
    average_volatility = sum(volatilities) / len(volatilities) if volatilities else 0.0
    average_swing = sum(swings) / len(swings) if swings else 0.0
    most_common_emotion = Counter(emotions).most_common(1)[0][0] if emotions else None
    most_positive_post = max(enriched_posts, key=lambda post: post["Ensemble_score"])
    most_negative_post = min(enriched_posts, key=lambda post: post["Ensemble_score"])
    high_volatility_percentage = (high_volatility_count / total_posts) * 100 if total_posts > 0 else 0.0

    print("\n========== AFFECTLENS SUMMARY ==========")
    print(f"Posts analyzed: {total_posts:,}")
    print()
    print(f"Average sentiment: {average_sentiment:.2f}")
    print(f"Average volatility: {average_volatility:.2f}")
    print(f"Average emotional swing: {average_swing:.2f}")
    print()
    print(f"Most common emotion: {most_common_emotion}")
    print(f"Most positive post: {most_positive_post['Text']} (Score: {most_positive_post['Ensemble_score']:.2f})")
    print(f"Most negative post: {most_negative_post['Text']} (Score: {most_negative_post['Ensemble_score']:.2f})")
    print(
        f"High-volatility posts: "
        f"{high_volatility_count:,} "
        f"({high_volatility_percentage:.1f}%)"
    )
    print("========================================\n")


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
    parser.add_argument("--evaluate", action="store_true", help="Flag to evaluate the trained model on a test set, compare sentiment analysis models and show the results.")
    parser.add_argument("--optimize_threshold", action="store_true", help="Flag to optimize the threshold for the classifier based on validation data.")
    parser.add_argument("--optimize_weight", action="store_true", help="Flag to optimize the ensemble weight for combining VADER and TextBlob scores based on validation data.")
    parser.add_argument("--classification_report", action="store_true", help="Flag to generate a classification report for the model.")
    parser.add_argument("--visualize", action="store_true", help="Flag to visualize the results.")
    args = parser.parse_args()

    # Call the main workflow function with the provided argument
    data_dir = Path(__file__).parent / "data"
    final_input_csv = Path(args.input_csv).expanduser().resolve() if args.input_csv else None  
    final_model_path = Path(args.model_path).expanduser().resolve()
    final_vectorizer_path = Path(args.vectorizer_path).expanduser().resolve()
    final_output_csv =Path(args.output_csv).expanduser().resolve() if args.output_csv else None 


    try:
        requires_training = needs_training(args.model_path, args.vectorizer_path)
        built_enriched_posts = False
        # One Input Validation: Ensure that at least one of the required inputs (text or input CSV) is provided when the model needs training
        if not args.text and not args.input_csv:
            raise ValueError("No input provided. Please provide either an input text string or a path to an input CSV file.")
        
        if requires_training and (args.evaluate or args.optimize_threshold or args.optimize_weight):
            raise ValueError("Cannot evaluate because the model and vectorizer do not exist. Train the model first using a labeled training dataset.")  
        
        # Training Validation: If the model needs training, ensure that an input CSV file is provided for training
        if requires_training and args.input_csv:
            if not dataset_is_training_data(str(final_input_csv)):
                raise ValueError("Model and vectorizer need to be trained. Please provide a labeled training dataset.")
            elif final_input_csv == data_dir / "training_data" / "train.csv":
                training_dataset = load_labeled_data(str(final_input_csv), args.required_columns)
                texts, labels = map(list, zip(*training_dataset))
                train(texts, labels, str(final_model_path), str(final_vectorizer_path)) 
            else:    
                converted_csv_path = edit_training_csv(str(final_input_csv), str(data_dir / "converted.csv"))
                train_csv_path, _, _ = split_dataset(converted_csv_path=str(converted_csv_path), train_csv_path=str(data_dir / "training_data" / "train.csv"), validation_csv_path=str(data_dir / "training_data" / "validation.csv"), test_csv_path=str(data_dir / "training_data" / "test.csv"), random_seed=42)
                training_dataset = load_labeled_data(str(train_csv_path), args.required_columns)
                texts, labels = map(list, zip(*training_dataset))
                train(texts, labels, str(final_model_path), str(final_vectorizer_path))  
        
        # Evaluation Validation: If the evaluation flag is set, ensure that an input CSV file is provided for evaluation
        if args.evaluate and final_input_csv:
            if dataset_is_training_data(str(final_input_csv)):
                converted_csv_path = edit_training_csv(str(final_input_csv), str(data_dir / "converted.csv"))
                _, _, test_csv_path = split_dataset(converted_csv_path=str(converted_csv_path), train_csv_path=str(data_dir / "training_data" / "train.csv"), validation_csv_path=str(data_dir / "training_data" / "validation.csv"), test_csv_path=str(data_dir / "training_data" / "test.csv"), random_seed=42)
                evaluate_classifier_results = evaluate_classifier(str(test_csv_path), TARGET_COLUMNS, str(final_model_path), str(final_vectorizer_path), args.threshold)            
            else:
                evaluate_classifier_results = evaluate_classifier(str(final_input_csv), TARGET_COLUMNS, str(final_model_path), str(final_vectorizer_path), args.threshold)            
            print("Evaluation Results:", evaluate_classifier_results)
            if sst_dataset_available(data_dir):
                sst_dataset = load_sst_dataset(str(data_dir / "SST" / "datasetSentences.txt"), str(data_dir / "SST" / "datasetSplit.txt"), str(data_dir / "SST" / "dictionary.txt"), str(data_dir / "SST" / "sentiment_labels.txt"), 3)
                compared_sentiment_results = compare_sentiment_models(sst_dataset)
                print("Sentiment Model Comparison Results(Errors):", compared_sentiment_results)
            return
        if args.evaluate and not final_input_csv:
            raise ValueError("Evaluation flag is set, but no input CSV file was provided for evaluation. Please provide a valid input CSV file for evaluation.")    
        
        # Optimize Threshold Validation: If the optimize threshold flag is set, ensure that an input CSV file is provided for optimizing the threshold
        if args.optimize_threshold and final_input_csv:
            if dataset_is_training_data(str(final_input_csv)):
                converted_csv_path = edit_training_csv(str(final_input_csv), str(data_dir / "converted.csv"))
                _, validation_csv_path, _ = split_dataset(converted_csv_path=str(converted_csv_path), train_csv_path=str(data_dir / "training_data" / "train.csv"), validation_csv_path=str(data_dir / "training_data" / "validation.csv"), test_csv_path=str(data_dir / "training_data" / "test.csv"), random_seed=42)
                best_threshold, best_micro_f1 = optimize_threshold(str(validation_csv_path), TARGET_COLUMNS, str(final_model_path), str(final_vectorizer_path))
            else:
                best_threshold, best_micro_f1 = optimize_threshold(str(final_input_csv), TARGET_COLUMNS, str(final_model_path), str(final_vectorizer_path))
            print(f"Optimal Threshold: {best_threshold}, Best Micro F1-Score: {best_micro_f1}")
            return
        if args.optimize_threshold and not final_input_csv:
            raise ValueError("Optimize threshold flag is set, but no input CSV file was provided for optimization. Please provide a valid input CSV file for optimizing the threshold.")

        # Optimize Ensemble Weight Validation: If the optimize ensemble weight flag is set, ensure that an input CSV file is provided for optimizing the ensemble weight    
        if args.optimize_weight:
            sst_dataset = load_sst_dataset(str(data_dir / "SST" / "datasetSentences.txt"), str(data_dir / "SST" / "datasetSplit.txt"), str(data_dir / "SST" / "dictionary.txt"), str(data_dir / "SST" / "sentiment_labels.txt"), 3)
            best_weight, mae = optimize_ensemble_weight(sst_dataset)
            print(f"Optimal Ensemble Weight: {best_weight}, Best MAE: {mae}")
            return
        
        # Classification Report Validation: If the classification report flag is set, ensure that an input CSV file is provided for generating the classification report
        if args.classification_report and final_input_csv:
            if dataset_is_training_data(str(final_input_csv)):
                converted_csv_path = edit_training_csv(str(final_input_csv), str(data_dir / "converted.csv"))
                _, _, test_csv_path = split_dataset(converted_csv_path=str(converted_csv_path), train_csv_path=str(data_dir / "training_data" / "train.csv"), validation_csv_path=str(data_dir / "training_data" / "validation.csv"), test_csv_path=str(data_dir / "training_data" / "test.csv"), random_seed=42)
                classification_report_results = get_classification_report(str(test_csv_path), TARGET_COLUMNS, str(final_model_path), str(final_vectorizer_path), args.threshold)            
            else:
                classification_report_results = get_classification_report(str(final_input_csv), TARGET_COLUMNS, str(final_model_path), str(final_vectorizer_path), args.threshold)            
            print("Classification Report Results:", classification_report_results)
            return
        if args.classification_report and not final_input_csv:
            raise ValueError("Classification report flag is set, but no input CSV file was provided for generating the report. Please provide a valid input CSV file for generating the classification report.")

        # Visualization Validation: If the visualize flag is set, ensure that an input CSV file is provided for visualization, as visualization is intended for multi-post analysis
        if args.visualize:
            visualization_dir = Path("visualizations")
            visualization_dir.mkdir(exist_ok=True)
            enriched_posts = build_enriched_posts(csv_path=str(final_input_csv) if final_input_csv else None, text=args.text if args.text else None, required_columns=PREDICTION_COLUMNS, model_path=str(final_model_path), vectorizer_path=str(final_vectorizer_path), weight=args.weight, threshold=args.threshold)
            built_enriched_posts = True
            print_summary(enriched_posts)

            if final_input_csv:
                figures = {
                    "sentiment_trend.png": plot_sentiment_trend(enriched_posts),
                    "volatility_trend.png": plot_volatility_trend(enriched_posts, THRESHOLD),
                    "emotional_swing_trend.png": plot_emotional_swing_trend(enriched_posts),
                    "emotion_distribution.png": plot_emotion_distribution(enriched_posts),
                    "emotion_heatmap.png": plot_emotion_heatmap(enriched_posts),
                    "emotion_transition_matrix.png": plot_emotion_transition_matrix(enriched_posts),
                }
                for filename, figure in figures.items():
                    output_path = visualization_dir / filename
                    figure.savefig(output_path, dpi=300, bbox_inches="tight")
                    print(f"Saved: {output_path}")
            else:
                print("Visualization skipped. Multi-post visualizations require a CSV containing multiple posts; only a single text input was provided.")       
                return
            os_name = platform.system()
            open_files = input("\nOpen generated visualizations? (y/n): ").strip().lower()
            if open_files == "y":
                for filename in figures.keys():
                    output_path = visualization_dir / filename
                    if output_path.is_file():
                        if os_name == "Windows":
                            os.startfile(output_path)
                        elif os_name == "Darwin":  # macOS
                            subprocess.run(["open", str(output_path)])
                        elif os_name == "Linux":
                            subprocess.run(["xdg-open", str(output_path)])
                        else:
                            print(f"Unsupported OS: {os_name}. Cannot open files automatically.")
                    else:
                        print(f"File not found: {output_path}")
                        

        if not built_enriched_posts:                       
            enriched_posts = build_enriched_posts(csv_path=str(final_input_csv) if final_input_csv else None, text=args.text if args.text else None, required_columns=PREDICTION_COLUMNS, model_path=str(final_model_path), vectorizer_path=str(final_vectorizer_path), weight=args.weight, threshold=args.threshold)
        if enriched_posts and final_output_csv:
            save_results(str(final_output_csv), OUTPUT_COLUMNS, enriched_posts)
            print_summary(enriched_posts)
            print(f"Results saved to: {final_output_csv}")
        elif enriched_posts and not args.visualize:    
            print(enriched_posts)    

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except FileNotFoundError as fnfe:
        print(f"FileNotFoundError: {fnfe}")

if __name__ == "__main__":
    main()