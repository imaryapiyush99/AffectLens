import pytest 
from affectlens.data_loader import validate_csv, load_dataset, load_training_data, save_results

def test_validate_csv_empty_file(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")

    with pytest.raises(ValueError, match="empty or malformed"):
        validate_csv(csv_file)

def test_validate_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        validate_csv("does_not_exist.csv")

def test_validate_csv_missing_columns(tmp_path):
    csv_file = tmp_path / "missing_columns.csv"
    csv_file.write_text("text\nSample text")

    with pytest.raises(ValueError, match="missing required columns"):
        validate_csv(csv_file)

def test_validate_csv_no_data_rows(tmp_path):
    csv_file = tmp_path / "no_rows.csv"
    csv_file.write_text("text,target\n")

    with pytest.raises(ValueError, match="no data rows"):
        validate_csv(csv_file)   

def test_load_dataset(tmp_path):
    csv_file = tmp_path / "dataset.csv"
    csv_file.write_text("text,target,title\nSample text 1,0,Title 1\nSample text 2,1,Title 2")

    dataset = load_dataset(csv_file)
    assert len(dataset) == 2
    assert dataset[0]["text"] == "Sample text 1"
    assert dataset[0]["title"] == "Title 1"
    assert dataset[0]["label"] == 0
    assert dataset[1]["text"] == "Sample text 2"
    assert dataset[1]["title"] == "Title 2"
    assert dataset[1]["label"] == 1

def test_load_training_data(tmp_path):
    csv_file = tmp_path / "training_data.csv"
    csv_file.write_text("text,target,title\nSample text 1,0,Title 1\nSample text 2,1,Title 2")

    texts, labels = load_training_data(csv_file)
    assert len(texts) == 2
    assert len(labels) == 2
    assert texts[0] == "Title 1 Sample text 1"
    assert labels[0] == 0
    assert texts[1] == "Title 2 Sample text 2"
    assert labels[1] == 1

def test_save_results_empty(tmp_path):
    with pytest.raises(ValueError, match="Results list is empty"):
        save_results([], tmp_path / "results.csv")

def test_save_results_no_directory(tmp_path):
    results = [{"text": "Sample text", "title": "Sample title", "predicted_label": "Stress", "classifier_score": 0.85}]
    output_path = tmp_path / "non_existent_dir" / "results.csv"

    with pytest.raises(FileNotFoundError, match="Output directory does not exist"):
        save_results(results, output_path)

def test_save_results_success(tmp_path):
    results = [{"text": "Sample text", "title": "Sample title", "predicted_label": "Stress", "classifier_score": 0.85}]
    output_path = tmp_path / "results.csv"

    save_results(results, output_path)
    assert output_path.exists()
    content = output_path.read_text()
    assert "Sample text" in content
    assert "Sample title" in content
    assert "Stress" in content
    assert "0.85" in content

def test_save_results_extra_fields(tmp_path):
    results = [{"text": "Sample text", "title": "Sample title", "predicted_label": "Stress", "classifier_score": 0.85, "extra_field": "Extra value"}]
    output_path = tmp_path / "results.csv"

    save_results(results, output_path)
    assert output_path.exists()
    content = output_path.read_text()
    assert "Sample text" in content
    assert "Sample title" in content
    assert "Stress" in content
    assert "0.85" in content
    assert "Extra value" not in content

def test_save_results_missing_fields(tmp_path):
    results = [{"text": "Sample text", "predicted_label": "Stress", "classifier_score": 0.85}]
    output_path = tmp_path / "results.csv"

    save_results(results, output_path)
    assert output_path.exists()
    content = output_path.read_text()
    assert "Sample text" in content
    assert "Stress" in content
    assert "0.85" in content
    assert "N/A" in content  # Missing fields should be filled with "N/A"    