import pytest
from affectlens.data_loader import validate_csv, edit_training_csv, split_dataset, load_dataset, load_labeled_data, save_results, load_sst_dataset
from affectlens.constants import EMOTIONS, TARGET_COLUMNS



def test_csv_validate_invalid_input(tmp_path):
    # Empty path
    with pytest.raises(ValueError):
        validate_csv("", required_columns=["text", "target"])

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        validate_csv(
            "non_existent_file.csv",
            required_columns=["text", "target"]
        )

    # Invalid required_columns type
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("text,target\nhello,1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_csv(str(csv_file), required_columns="not_a_list")

    # Empty required_columns
    with pytest.raises(ValueError):
        validate_csv(str(csv_file), required_columns=[])

    # Missing required column
    csv_file.write_text(
        "header1,header2\n"
        "data1,data2\n",
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        validate_csv(
            str(csv_file),
            required_columns=["non_existent_column"]
        )

def test_edit_training_csv_invalid_input(tmp_path):
    # Create a dummy CSV file with valid data
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "text,Happiness,Sadness,Anger,Fear,Disgust,Surprise,Neutral,Curiosity,Confusion,Realization,Desire\n"
        '"I am happy",1,0,0,0,0,0,0,0,0,0,0\n'
        '"I am sad",0,1,0,0,0,0,0,0,0,0,0\n'
        '"I am angry",0,0,1,0,0,0,0,0,0,0,0\n'
        '"I am afraid",0,0,0,1,0,0,0,0,0,0,0\n'
        '"I am disgusted",0,0,0, 0,1,0,0,0,0,0,0\n'
    )

    # Invalid training_csv_path type
    with pytest.raises(ValueError):
        edit_training_csv(str(csv_file), converted_csv_path=123)

    # Empty training_csv_path
    with pytest.raises(ValueError):
        edit_training_csv(str(csv_file), converted_csv_path="")


def test_split_dataset_invalid_input(tmp_path):
    # Create a dummy CSV file with valid data
    csv_file = tmp_path / "test.csv" 

    # Invalid converted_csv_path type
    with pytest.raises(ValueError):
        split_dataset(converted_csv_path=123, train_csv_path=str(tmp_path / "train.csv"), validation_csv_path=str(tmp_path / "validation.csv"), test_csv_path=str(tmp_path / "test.csv"), random_seed=42)

    # Empty converted_csv_path
    with pytest.raises(ValueError):
        split_dataset(converted_csv_path="", train_csv_path=str(tmp_path / "train.csv"), validation_csv_path=str(tmp_path / "validation.csv"), test_csv_path=str(tmp_path / "test.csv"), random_seed=42)

    # Non-existent converted_csv_path
    with pytest.raises(FileNotFoundError):
        split_dataset(converted_csv_path=str(tmp_path / "non_existent_file.csv"), train_csv_path=str(tmp_path / "train.csv"), validation_csv_path=str(tmp_path / "validation.csv"), test_csv_path=str(tmp_path / "test.csv"), random_seed=42)

    # Invalid train_csv_path type
    with pytest.raises(ValueError):
        split_dataset(converted_csv_path=str(csv_file), train_csv_path=123, validation_csv_path=str(tmp_path / "validation.csv"), test_csv_path=str(tmp_path / "test.csv"), random_seed=42)

    # Empty train_csv_path
    with pytest.raises(ValueError):
        split_dataset(converted_csv_path=str(csv_file), train_csv_path="", validation_csv_path=str(tmp_path / "validation.csv"), test_csv_path=str(tmp_path / "test.csv"), random_seed=42)

    # Non-existent train_csv_path directory
    non_existent_dir = tmp_path / "non_existent_dir"
    with pytest.raises(FileNotFoundError):
        split_dataset(converted_csv_path=str(csv_file), train_csv_path=str(non_existent_dir / "train.csv"), validation_csv_path=str(tmp_path / "validation.csv"), test_csv_path=str(tmp_path / "test.csv"), random_seed=42)

    # Invalid validation_csv_path type
    with pytest.raises(ValueError):
        split_dataset(converted_csv_path=str(csv_file), train_csv_path=str(tmp_path / "train.csv"), validation_csv_path=123, test_csv_path=str(tmp_path / "test.csv"), random_seed=42)

    # Empty validation_csv_path
    with pytest.raises(ValueError):
        split_dataset(converted_csv_path=str(csv_file), train_csv_path=str(tmp_path / "train.csv"), validation_csv_path="", test_csv_path=str(tmp_path / "test.csv"), random_seed=42)

    # Non-existent validation_csv_path directory
    with pytest.raises(FileNotFoundError):
        split_dataset(converted_csv_path=str(csv_file), train_csv_path=str(tmp_path / "train.csv"), validation_csv_path=str(non_existent_dir / "validation.csv"), test_csv_path=str(tmp_path / "test.csv"), random_seed=42)

    # Invalid test_csv_path type
    with pytest.raises(ValueError):
        split_dataset(converted_csv_path=str(csv_file), train_csv_path=str(tmp_path / "train.csv"), validation_csv_path=str(tmp_path / "validation.csv"), test_csv_path=123, random_seed=42)

    # Empty test_csv_path
    with pytest.raises(ValueError):
        split_dataset(converted_csv_path=str(csv_file), train_csv_path=str(tmp_path / "train.csv"), validation_csv_path=str(tmp_path / "validation.csv"), test_csv_path="", random_seed=42)

    # Non-existent test_csv_path directory
    with pytest.raises(FileNotFoundError):
        split_dataset(converted_csv_path=str(csv_file), train_csv_path=str(tmp_path / "train.csv"), validation_csv_path=str(tmp_path / "validation.csv"), test_csv_path=str(non_existent_dir / "test.csv"), random_seed=42)

    # Invalid random_seed type
    with pytest.raises(ValueError):
        split_dataset(converted_csv_path=str(csv_file), train_csv_path=str(tmp_path / "train.csv"), validation_csv_path=str(tmp_path / "validation.csv"), test_csv_path=str(tmp_path / "test.csv"), random_seed="not_an_integer")

    # Empty converted CSV file
    empty_csv_file = tmp_path / "empty.csv"
    empty_csv_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        split_dataset(converted_csv_path=str(empty_csv_file), train_csv_path=str(tmp_path / "train.csv"), validation_csv_path=str(tmp_path / "validation.csv"), test_csv_path=str(tmp_path / "test.csv"), random_seed=42)

    # Converted CSV file with only header and no data rows
    header_only_csv_file = tmp_path / "header_only.csv"
    header_only_csv_file.write_text("text,Happiness,Sadness,Anger,Fear,Disgust,Surprise,Neutral,Curiosity,Confusion,Realization,Desire\n", encoding="utf-8")
    with pytest.raises(ValueError):
        split_dataset(converted_csv_path=str(header_only_csv_file), train_csv_path=str(tmp_path / "train.csv"), validation_csv_path=str(tmp_path / "validation.csv"), test_csv_path=str(tmp_path / "test.csv"), random_seed=42)


def test_split_dataset_valid_input(tmp_path):
    # Create a dummy CSV file with valid data
    csv_file = tmp_path / "testing.csv"
    csv_file.write_text(
        '"text","Happiness","Sadness","Anger","Fear","Disgust","Surprise","Neutral","Curiosity","Confusion","Realization","Desire"\n'
        '"text1",1,0,0,0,0,0,0,0,0,0,0\n'
        '"text2",0,1,0,0,0,0,0,0,0,0,0\n'
        '"text3",0,0,1,0,0,0,0,0,0,0,0\n'
        '"text4",0,0,0,1,0,0,0,0,0,0,0\n'
        '"text5",0,0,0,0,1,0,0,0,0,0,0\n'
        '"text6",0,0,0,0,0,1,0,0,0,0,0\n'
        '"text17",0,0,0,0,0,1,1,0,0,0,0\n'
        '"text8",0,0,0,0,0,0,0,1,0,0,0\n'
        '"text9",0,0,0,0,0,0,0,0,1,0,0\n'
        '"text20",0,0,0,0,0,0,0,0,1,1,0\n'
        '"text11",0,0,0,0,0,0,0,0,0,0,1\n'
        '"text12",1,1,0,0,0,0,0,0,0,0,0\n'      
        '"text13",0,1,1,0,0,0,0,0,0,0,0\n'
        '"text14",0,0,1,1,0,0,0,0,0,0,0\n'
        '"text15",0,0,0,1,1,0,0,0,0,0,0\n'
        '"text16",0,0,0,0,1,1,0,0,0,0,0\n'
        '"text17",0,0,0,0,0,1,1,0,0,0,0\n'
        '"text18",0,0,0,0,0,0,1,1,0,0,0\n'
        '"text19",0,0,0,0,0,0,0,1,1,0,0\n'
        '"text20",0,0,0,0,0,0,0,0,1,1,0\n'
    ) 

    assert split_dataset(converted_csv_path=str(csv_file), train_csv_path=str(tmp_path / "train.csv"), validation_csv_path=str(tmp_path / "validation.csv"), test_csv_path=str(tmp_path / "test.csv"), random_seed=42) == (str(tmp_path / "train.csv"), str(tmp_path / "validation.csv"), str(tmp_path / "test.csv"))
    train_path, val_path, test_path = split_dataset(converted_csv_path=str(csv_file), train_csv_path=str(tmp_path / "train.csv"), validation_csv_path=str(tmp_path / "validation.csv"), test_csv_path=str(tmp_path / "test.csv"), random_seed=42)

    train = load_labeled_data(train_path, TARGET_COLUMNS)
    val = load_labeled_data(val_path, TARGET_COLUMNS)
    test = load_labeled_data(test_path, TARGET_COLUMNS)

    assert len(train) == 14
    assert len(val) == 3
    assert len(test) == 3
    assert len(train) + len(val) + len(test) == 20 


def test_load_dataset_invalid_input(tmp_path):
    # Create a dummy CSV file with valid data
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "text,Happiness,Sadness,Anger,Fear,Disgust,Surprise,Neutral,Curiosity,Confusion,Realization,Desire\n"
        '"I am happy",1,0,0,0,0,0,0,0,0,0,0\n'
        '"I am sad",0,1,0,0,0,0,0,0,0,0,0\n'
        '"I am angry",0,0,1,0,0,0,0,0,0,0,0\n'
        '"I am afraid",0,0,0,1,0,0,0,0,0,0,0\n'
        '"I am disgusted",0,0,0, 0,1,0,0,0,0,0,0\n'
    )
    # Invalid training_csv_path type
    with pytest.raises(ValueError):
        load_dataset(csv_path=123, required_columns=["text"] + EMOTIONS)

    # Empty training_csv_path
    with pytest.raises(ValueError):
        load_dataset(csv_path="", required_columns=["text"] + EMOTIONS)

    # Missing required column
    with pytest.raises(ValueError):
        load_dataset(csv_path=str(csv_file), required_columns=["text", "score"] + EMOTIONS)

def test_load_dataset_valid_input(tmp_path):
    # Create a dummy CSV file with valid data
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "text,Happiness,Sadness,Anger,Fear,Disgust,Surprise,Neutral,Curiosity,Confusion,Realization,Desire\n"
        '"I am happy",1,0,0,0,0,0,0,0,0,0,0\n'
        '"I am sad",0,1,0,0,0,0,0,0,0,0,0\n'
        '"I am angry",0,0,1,0,0,0,0,0,0,0,0\n'
        '"I am afraid",0,0,0,1,0,0,0,0,0,0,0\n'
        '"I am disgusted",0,0,0, 0,1,0,0,0,0,0,0\n'
    )    

    dataset = load_dataset(csv_path=str(csv_file), required_columns=["text"])
    assert isinstance(dataset, list)
    assert all(isinstance(item, dict) for item in dataset)
    assert dataset[0]["text"] == "I am happy"


def test_load_labeled_data_invalid_input(tmp_path):
    # Create a dummy CSV file with valid data
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "text,Happiness,Sadness,Anger,Fear,Disgust,Surprise,Neutral,Curiosity,Confusion,Realization,Desire\n"
        '"I am happy",1,0,0,0,0,0,0,0,0,0,0\n'
        '"I am sad",0,1,0,0,0,0,0,0,0,0,0\n'
        '"I am angry",0,0,1,0,0,0,0,0,0,0,0\n'
        '"I am afraid",0,0,0,1,0,0,0,0,0,0,0\n'
        '"I am disgusted",0,0,0, 0,1,0,0,0,0,0,0\n'
    )  

    # Invalid labeled_csv_path type
    with pytest.raises(ValueError):
        load_labeled_data(labeled_csv_path=123, required_columns=["text"] + EMOTIONS)

    # Empty labeled_csv_path
    with pytest.raises(ValueError):
        load_labeled_data(labeled_csv_path="", required_columns=["text"] + EMOTIONS)

    # Missing required column
    with pytest.raises(ValueError):
        load_labeled_data(labeled_csv_path=str(csv_file), required_columns=["non_existent_column"])

def test_load_labeled_data_valid_input(tmp_path):
    # Create a dummy CSV file with valid data
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "text,Happiness,Sadness,Anger,Fear,Disgust,Surprise,Neutral,Curiosity,Confusion,Realization,Desire\n"
        '"I am happy",1,0,0,0,0,0,0,0,0,0,0\n'
        '"I am sad",0,1,0,0,0,0,0,0,0,0,0\n'
        '"I am angry",0,0,1,0,0,0,0,0,0,0,0\n'
        '"I am afraid",0,0,0,1,0,0,0,0,0,0,0\n'
        '"I am disgusted",0,0,0, 0,1,0,0,0,0,0,0\n'
    )  

    labeled_data = load_labeled_data(labeled_csv_path=str(csv_file), required_columns=["text"] + EMOTIONS)
    assert isinstance(labeled_data, list)
    assert all(isinstance(item, tuple) for item in labeled_data)
    assert all(isinstance(item[0], str) and isinstance(item[1], list) for item in labeled_data)
    assert labeled_data[0][0] == "I am happy"
    assert labeled_data[0][1][0] == 1


def test_load_sst_dataset_invalid_input(tmp_path):
    dataset_sentences_path = tmp_path / "datasetSentences.txt"
    dataset_split_path = tmp_path / "datasetSplit.txt"
    dictionary_path = tmp_path / "dictionary.txt"
    sentiment_labels_path = tmp_path / "sentiment_labels.txt"
    
     # Non-existent dataset files (assuming the function checks for file existence)
    with pytest.raises(FileNotFoundError):
        load_sst_dataset(dataset_sentences_path=str(dataset_sentences_path), dataset_split_path=str(dataset_split_path), dictionary_path=str(dictionary_path), sentiment_labels_path=str(sentiment_labels_path), split=1)  # Assuming the required files are not present in the test environment

    dataset_sentences_path.write_text("sentence_index\tsentence\n1\tI am happy.\n2\tI am sad.\n")
    dataset_split_path.write_text("sentence_index\tsplitset_label\n1\t1\n2\t2\n")
    dictionary_path.write_text("phrase\tphrase_id\nI am happy.\t1\nI am sad.\t2\n")
    sentiment_labels_path.write_text("phrase ids\tsentiment values\n1\t0.9\n2\t0.1\n")

    # Invalid split type
    with pytest.raises(ValueError):
        load_sst_dataset(dataset_sentences_path=str(dataset_sentences_path), dataset_split_path=str(dataset_split_path), dictionary_path=str(dictionary_path), sentiment_labels_path=str(sentiment_labels_path), split="invalid_split")

    # Invalid split value
    with pytest.raises(ValueError):
        load_sst_dataset(dataset_sentences_path=str(dataset_sentences_path), dataset_split_path=str(dataset_split_path), dictionary_path=str(dictionary_path), sentiment_labels_path=str(sentiment_labels_path), split=4)  # Valid splits are 1, 2, or 3

    # Invalid dataset structure (missing required columns)
    dataset_sentences_path.write_text("invalid_column\n1\tI am happy.\n2\tI am sad.\n")
    with pytest.raises(ValueError):
        load_sst_dataset(dataset_sentences_path=str(dataset_sentences_path), dataset_split_path=str(dataset_split_path), dictionary_path=str(dictionary_path), sentiment_labels_path=str(sentiment_labels_path), split=1)
    


def test_save_results_invalid_input(tmp_path):
    # Create a dummy results list
    results = [{"text": "I am happy", "Happiness": 0.9}, {"text": "I am sad", "Sadness": 0.1}]

    # Invalid output_csv_path type
    with pytest.raises(ValueError):
        save_results(results = results, required_columns=["text"], results_csv_path=123)

    # Empty output_csv_path
    with pytest.raises(ValueError):
        save_results(results = results, required_columns=["text"], results_csv_path="")  

    # Non-existent directory
    non_existent_dir = tmp_path / "non_existent_dir"
    with pytest.raises(FileNotFoundError): 
        save_results(results= results , required_columns=["text"], results_csv_path=str(non_existent_dir / "results.csv"))

    # Invalid results type
    with pytest.raises(ValueError):
        save_results(results="not_a_list_of_ dict", required_columns=["text"], results_csv_path=str(tmp_path / "results.csv"))

    # Empty results list of dict
    with pytest.raises(ValueError):
        save_results(results=[{}], required_columns=["text"], results_csv_path=str(tmp_path / "results.csv"))

    # Invalid results format
    with pytest.raises(ValueError):    
        save_results(results=[("I am happy", "not_a_float")], required_columns=["text"], results_csv_path=str(tmp_path / "results.csv"))

    # Missing required columns in results
    with pytest.raises(ValueError):
        save_results(results=[{"title": "I am happy"}], required_columns=["text"] , results_csv_path=str(tmp_path / "results.csv"))    
 