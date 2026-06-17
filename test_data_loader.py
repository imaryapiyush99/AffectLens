import pytest
from affectlens.data_loader import validate_csv, edit_training_csv, load_dataset, load_training_data, save_results
from affectlens.constants import EMOTIONS



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
        edit_training_csv(str(csv_file), training_csv_path=123)

    # Empty training_csv_path
    with pytest.raises(ValueError):
        edit_training_csv(str(csv_file), training_csv_path="")

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


def test_load_training_data_invalid_input(tmp_path):
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
        load_training_data(training_csv_path=123, required_columns=["text"] + EMOTIONS)

    # Empty training_csv_path
    with pytest.raises(ValueError):
        load_training_data(training_csv_path="", required_columns=["text"] + EMOTIONS)

    # Missing required column
    with pytest.raises(ValueError):
        load_training_data(training_csv_path=str(csv_file), required_columns=["non_existent_column"])

def test_load_training_data_valid_input(tmp_path):
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

    training_data = load_training_data(training_csv_path=str(csv_file), required_columns=["text"] + EMOTIONS)
    assert isinstance(training_data, list)
    assert all(isinstance(item, tuple) for item in training_data)
    assert all(isinstance(item[0], str) and isinstance(item[1], list) for item in training_data)
    assert training_data[0][0] == "I am happy"
    assert training_data[0][1][0] == 1

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
 