import csv
from pathlib import Path

def validate_csv(csv_path: str) -> None:
    """
    Validates that the CSV file exists and has the required columns.

    Args: csv_path (str): Path to the CSV file to validate

    Raises: FileNotFoundError: If the CSV file does not exist, ValueError: If the CSV file is missing required columns
    """
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or malformed.")
        required_columns = {"text", "target"}
        if not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"CSV file is missing required columns. Required columns are: {required_columns}")
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError("CSV file has no data rows.")



def load_dataset(csv_path: str) -> list[dict[str, str | int]]:
    """
    Loads the dataset from a CSV file and returns a list of dictionaries with 'text', 'title'(optional), and 'label' keys.

    Args: csv_path (str): Path to the CSV file containing the dataset with 'text', 'title'(optional), and 'target' columns

    Returns: list[dict[str, str | int]]: A list of dictionaries where each dictionary has 'text', 'title'(optional), and 'label' keys
    """
    validate_csv(csv_path)
    dataset = []
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row["text"].strip()
            title = row.get("title", "").strip()
            label = int(row["target"].strip())
            dataset.append({"text": text, "title": title, "label": label})
    return dataset



def load_training_data(csv_path: str) -> tuple[list[str], list[int]]:
    """
    Loads the training data from a CSV file and returns the texts and labels as separate lists.

    Args: csv_path (str): Path to the CSV file containing the training data with 'text', 'title'(optional), and 'target' columns

    Returns: tuple[list[str], list[int]]: A tuple containing a list of text strings and a list of corresponding integer labels
    """
    texts = []
    labels = []
    dataset = load_dataset(csv_path)
    for item in dataset:
        text = item["text"]
        title = item.get("title", "")
        combined_text = f"{title} {text}".strip()
        texts.append(combined_text)
        labels.append(item["label"])
    return texts, labels



def save_results(results: list[dict[str, str | float]], output_path: str) -> None:
    """
    Saves the results to a CSV file with "text", "title", "predicted_label", "classifier_score", "ensemble_score", "final_score", "volatility_score" columns.

    Args: results (list[dict[str, str | float]]): A list of dictionaries where each dictionary has "text", "title", "predicted_label", "classifier_score", "ensemble_score", "final_score", "volatility_score" keys, output_path (str): Path to save the results CSV file
    """
    if not results:
        raise ValueError("Results list is empty. Cannot save to CSV.")
    csv_file = Path(output_path)
    if not csv_file.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {csv_file.parent}")
    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["text", "title", "predicted_label", "classifier_score", "ensemble_score", "final_score", "volatility_score"], extrasaction="ignore", restval="N/A")
        writer.writeheader()
        for result in results:
            writer.writerow(result)