from pathlib import Path
import csv

from affectlens.constants import EMOTION_MAPPING, GOEMOTIONS_COLUMNS, TARGET_COLUMNS, EMOTIONS

def validate_csv(csv_path: str, required_columns: list[str]) -> None:
    """
    Validates the training CSV file.

    Args: csv_path (str): Path to the training CSV file, required_columns (list[str]): List of required column names for the training CSV file
    """
    if not isinstance(required_columns, list):
        raise ValueError("Required columns must be a list of strings.")
    if not isinstance(csv_path, str):
        raise ValueError("CSV path must be a string.")
    if not csv_path.strip():
        raise ValueError("CSV path must be provided.")
    if not Path(csv_path).is_file():
        raise FileNotFoundError("CSV path is not a file. Please provide a valid file path.")
    if not required_columns:
        raise ValueError("Required columns list is empty. Please provide a list of required column names.")
    required = [column for column in required_columns if column not in ["title"]]
    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV file is missing header row.")
        if not set(required).issubset(set(reader.fieldnames)):
            raise ValueError("CSV file is missing required columns.")
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError("CSV file has no data rows.")


def edit_training_csv(csv_path: str, training_csv_path: str) -> str:
    """
    Edits the training CSV file to match the expected format for training the classifier.

    Args: csv_path (str): Path to the input CSV file, training_csv_path (str): Path to save the edited training CSV file
    """
    validate_csv(csv_path, required_columns=GOEMOTIONS_COLUMNS)
    if not isinstance(training_csv_path, str):
        raise ValueError("Training CSV path must be a string.")
    if not training_csv_path.strip():
        raise ValueError("Training CSV path must be provided.")
    with open(csv_path, "r", encoding="utf-8", newline="") as infile, open(training_csv_path, "w", encoding="utf-8", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = TARGET_COLUMNS
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, restval=0)
        writer.writeheader()    
        
        for row in reader:
            if row["example_very_unclear"] == "True":
                continue
            new_row = {"text": f"{row.get('title', '')} {row['text']}".strip()}
            for emotion, source_labels in EMOTION_MAPPING.items():
                new_row[emotion] = int(any(int(row[label]) for label in source_labels))
            writer.writerow(new_row)               
    return training_csv_path

def load_dataset(csv_path: str, required_columns: list[str]) -> list[dict[str, str | int]]:
    """
    Loads the dataset from a CSV file and returns it as a list of dictionaries.

    Args: training_csv_path (str): Path to the training CSV file, required_columns (list[str]): List of column names to include in the loaded dataset

    Returns: list[dict[str, str | int]]: List of dictionaries representing the dataset, where each dictionary corresponds to a row in the CSV file with keys as column names and values as the corresponding cell values
    """
    validate_csv(csv_path, required_columns)
    dataset = []
    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            dict = {
                "text": f"{row.get('title', '')} {row['text']}".strip()
            }
            dataset.append(dict)
    return dataset
    

def load_training_data(training_csv_path: str, required_columns: list[str]) -> list[tuple[str, list[int]]]:
    """
    Loads the training data from a CSV file and returns it as a list of dictionaries.

    Args: training_csv_path (str): Path to the training CSV file, required_columns (list[str]): List of column names to include in the loaded training data

    Returns: list[tuple[str, list[int]]]: List of "text" strings and lists of lists representing the training data, where each list corresponds to a row in the CSV file with keys as column names and values as the corresponding cell values
    """
    validate_csv(training_csv_path, required_columns)
    training_data = []
    with open(training_csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row["text"]
            labels = [int(row[emotion]) for emotion in EMOTIONS]
            training_data.append((text, labels))
    return training_data

def save_results(results_csv_path: str, required_columns: list[str], results: list[dict]) -> None:
    """
    Saves the results to a CSV file.

    Args: results_csv_path (str): Path to save the results CSV file, required_columns (list[str]): List of column names to include in the results CSV file, results (list[dict[str, str | int | float | None]]): List of dictionaries representing the results, where each dictionary corresponds to a row in the CSV file with keys as column names and values as the corresponding cell values
    """
    if not isinstance(results_csv_path, str):
        raise ValueError("Results CSV path must be a string.")
    if not isinstance(required_columns, list):
        raise ValueError("Required columns must be a list of strings.")
    if not isinstance(results, list):
        raise ValueError("Results must be a list.")
    if not all(isinstance(item, dict) for item in results):
        raise ValueError("Results must be a list of dictionaries.")
    if not Path(results_csv_path).parent.exists():
        raise FileNotFoundError("Results CSV path directory does not exist. Please provide a valid directory path.")
    if not results_csv_path.strip():
        raise ValueError("Results CSV path must be provided.")
    if not results:
        raise ValueError("Results list is empty. No results to save.")
    if not results[0]:
        raise ValueError("Results list contains an empty dictionary. No results to save.")
    with open(results_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=required_columns, extrasaction="ignore", restval="NA")
        if not set(required_columns).issubset(set(results[0].keys())):
            raise ValueError("Results dictionaries are missing required columns.")
        writer.writeheader()
        for result in results:
            writer.writerow(result)