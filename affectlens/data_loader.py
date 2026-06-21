from pathlib import Path
import csv, random
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


def edit_training_csv(csv_path: str, converted_csv_path: str) -> str:
    """
    Edits the training CSV file to match the expected format for training the classifier.

    Args: csv_path (str): Path to the input CSV file, converted_csv_path (str): Path to save the edited CSV file
    """
    validate_csv(csv_path, required_columns=GOEMOTIONS_COLUMNS)
    if not isinstance(converted_csv_path, str):
        raise ValueError("Converted CSV path must be a string.")
    if not converted_csv_path.strip():
        raise ValueError("Converted CSV path must be provided.")
    with open(csv_path, "r", encoding="utf-8", newline="") as infile, open(converted_csv_path, "w", encoding="utf-8", newline="") as outfile:
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
    return converted_csv_path

def split_dataset(converted_csv_path: str, train_csv_path: str, validation_csv_path: str, test_csv_path: str,  random_seed: int = 42) -> tuple[str, str, str]:
    """
    Splits a dataset into: 70% train, 15% validation, 15% test

    Args: converted_csv_path (str): Path to the converted CSV file, train_csv_path (str): Path to save the training CSV file, validation_csv_path (str): Path to save the validation CSV file, test_csv_path (str): Path to save the test CSV file, random_seed (int): Random seed for reproducibility
    """
    if not isinstance(train_csv_path, str) or not isinstance(validation_csv_path, str) or not isinstance(test_csv_path, str):
        raise ValueError("Train, validation, and test CSV paths must be strings.")
    if not train_csv_path.strip() or not validation_csv_path.strip() or not test_csv_path.strip():
        raise ValueError("Train, validation, and test CSV paths must be provided.")
    if not Path(train_csv_path).parent.exists() or not Path(validation_csv_path).parent.exists() or not Path(test_csv_path).parent.exists():
        raise FileNotFoundError("Train, validation, or test CSV path directory does not exist. Please provide valid directory paths.")
    if not isinstance(random_seed, int):
        raise ValueError("Random seed must be an integer.")
    
    validate_csv(converted_csv_path, required_columns=TARGET_COLUMNS)
    # Implementation for splitting the dataset goes here
    with open(converted_csv_path, "r", encoding="utf-8", newline="") as infile:
        rows = list(csv.reader(infile))
        header = rows[0]
        data = rows[1:]
        if not data:
            raise ValueError("Converted CSV file has no data rows to split.")
        
        rng = random.Random(random_seed)
        rng.shuffle(data)

        train_end = int(len(data) * 0.70)
        validation_end = int(len(data) * 0.85)

        train_rows = data[:train_end]
        validation_rows = data[train_end:validation_end]
        test_rows = data[validation_end:]

        with open(train_csv_path, "w", encoding="utf-8", newline="") as train_file:
            writer = csv.writer(train_file)
            writer.writerow(header)
            writer.writerows(train_rows)

        with open(validation_csv_path, "w", encoding="utf-8", newline="") as validation_file:
            writer = csv.writer(validation_file)
            writer.writerow(header)
            writer.writerows(validation_rows)

        with open(test_csv_path, "w", encoding="utf-8", newline="") as test_file:
            writer = csv.writer(test_file)
            writer.writerow(header)
            writer.writerows(test_rows)      

    return train_csv_path, validation_csv_path, test_csv_path          



def load_dataset(csv_path: str, required_columns: list[str]) -> list[dict[str, str]]:
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
            dictionary = {
                "text": f"{row.get('title', '')} {row['text']}".strip()
            }
            dataset.append(dictionary)
    return dataset
    

def load_labeled_data(labeled_csv_path: str, required_columns: list[str]) -> list[tuple[str, list[int]]]:
    """
    Loads the labeled data from a CSV file and returns it as a list of tuples.

    Args: labeled_csv_path (str): Path to the labeled CSV file, required_columns (list[str]): List of column names to include in the loaded labeled data

    Returns: list[tuple[str, list[int]]]: List of "text" strings and lists of lists representing the labeled data, where each list corresponds to a row in the CSV file with keys as column names and values as the corresponding cell values
    """
    validate_csv(labeled_csv_path, required_columns)
    labeled_data = []
    with open(labeled_csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row["text"]
            labels = [int(row[emotion]) for emotion in EMOTIONS]
            labeled_data.append((text, labels))
    return labeled_data

def load_sst_dataset(dataset_sentences_path: str, dataset_split_path: str, dictionary_path: str, sentiment_labels_path: str, split: int) -> list[tuple[str, float]]:    
    """
    Loads the SST dataset from the provided files and returns it as a list of tuples.

    Args: dataset_sentences_path (str): Path to the sentences file, dataset_split_path (str): Path to the split file, dictionary_path (str): Path to the dictionary file, sentiment_labels_path (str): Path to the sentiment labels file, split (int): Split value for loading the dataset

    Returns: list[tuple[str, float]]: List of tuples representing the SST dataset, where each tuple contains a sentence and its corresponding sentiment label
    """
    # Implementation for loading the SST dataset goes here
    for path in (
        dataset_sentences_path,
        dataset_split_path,
        dictionary_path,
        sentiment_labels_path,
    ):
        if not isinstance(path, str):
            raise ValueError("All file paths must be strings.")
        if not Path(path).is_file():
            raise FileNotFoundError(...)
    if split not in {1, 2, 3}:
        raise ValueError("Split must be 1 (train), 2 (test), or 3 (dev).")
    sentences = {}
    splits = {}
    phrase_ids = {}
    sentiment_scores = {}
    # Load the sentences
    with open(dataset_sentences_path, "r", encoding="utf-8") as sentences_file:
        next(sentences_file)  # Skip the header line
        for line in sentences_file:
            sentence_index, sentence = line.rstrip("\n").rsplit("\t", maxsplit=1)
            sentences[int(sentence_index)] = sentence

    # Load the splits
    with open(dataset_split_path, "r", encoding="utf-8") as split_file:
        next(split_file)  # Skip the header line
        for line in split_file:
            sentence_index, split_value = line.rstrip("\n").rsplit(",", maxsplit=1)
            splits[int(sentence_index)] = int(split_value)

    # Load the phrase IDs
    with open(dictionary_path, "r", encoding="utf-8") as dictionary_file:   
        for line in dictionary_file:
            phrase, phrase_id = line.rstrip("\n").rsplit("|", maxsplit=1)
            phrase_ids[phrase] = int(phrase_id)             

    # Load the sentiment scores
    with open(sentiment_labels_path, "r", encoding="utf-8") as sentiment_file:
        next(sentiment_file)  # Skip the header line
        for line in sentiment_file:
            phrase_id, sentiment_score = line.rstrip("\n").rsplit("|", maxsplit=1)
            sentiment_scores[int(phrase_id)] = float(sentiment_score)

    # Combine the data into a list of tuples
    sst_data = []
    for sentence_index, sentence in sentences.items(): 
        # Some SST sentences do not exactly match dictionary.txt
        # because of tokenization/formatting differences.
        # Unmatched sentences are skipped.
        if splits[sentence_index] != split:
            continue     
        if sentence not in phrase_ids:
           continue
        phrase_id = phrase_ids[sentence]        
        if phrase_id not in sentiment_scores:
            raise ValueError(f"Phrase ID {phrase_id} not found in sentiment labels.")
        sentiment_score = sentiment_scores[phrase_id]
        sst_data.append((sentence, sentiment_score))               
    return sst_data

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