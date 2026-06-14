import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from affectlens.classifier import needs_training, train, load, predict

def test_needs_training(tmp_path):
    model_path = tmp_path / "model.joblib"
    vectorizer_path = tmp_path / "vectorizer.joblib"
    
    # Initially, both files do not exist, so needs_training should return True
    assert needs_training(model_path, vectorizer_path) == True
    
    # Create one file to simulate either existing model or vectorizer
    model_path.touch()
    
    # Only one file exist, so needs_training should return True
    assert needs_training(model_path, vectorizer_path) == True

    # Create the other file to simulate both existing model and vectorizer
    vectorizer_path.touch()

    # Both files exist, so needs_training should return False
    assert needs_training(model_path, vectorizer_path) == False

def test_needs_training_files_not_a_file(tmp_path):
    model_path = tmp_path / "model_dir"
    vectorizer_path = tmp_path / "vectorizer_dir"
    
    model_path.mkdir()
    vectorizer_path.mkdir()
    
    # Both paths exist but are directories, so needs_training should return True
    assert needs_training(model_path, vectorizer_path) == True

def test_needs_training_files_empty(tmp_path):
    model_path = tmp_path / "model.joblib"
    vectorizer_path = tmp_path / "vectorizer.joblib"
    
    model_path.touch()
    vectorizer_path.touch()
    
    # Both files exist but are empty, so needs_training should return True
    assert needs_training(model_path, vectorizer_path) == True

def test_train(tmp_path):
    # This test will check if the train function runs without errors and creates the model and vectorizer files
    csv_path = tmp_path / "training_data.csv"
    model_path = tmp_path / "model.joblib"
    vectorizer_path = tmp_path / "vectorizer.joblib"
    
    # Create a dummy CSV file with training data
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("text,target\n")
        f.write("I am feeling stressed,0\n")
        f.write("I am feeling depressed,1\n")
        f.write("I am feeling bipolar,2\n")
        f.write("I have a personality disorder,3\n")
        f.write("I am feeling anxious,4\n")
    
    # Train the model using the dummy CSV file
    train(csv_path, model_path, vectorizer_path)
    
    # Check if the model and vectorizer files were created
    assert model_path.exists() and model_path.is_file() and model_path.stat().st_size > 0
    assert vectorizer_path.exists() and vectorizer_path.is_file() and vectorizer_path.stat().st_size > 0

def test_load_files_not_found(tmp_path):
    model_path = tmp_path / "non_existent_model.joblib"
    vectorizer_path = tmp_path / "non_existent_vectorizer.joblib"
    
    with pytest.raises(FileNotFoundError):
        load(model_path, vectorizer_path)

def test_load_one_file_not_found(tmp_path):
    model_path = tmp_path / "model.joblib"
    vectorizer_path = tmp_path / "vectorizer.joblib"
    
    model_path.touch()
    
    with pytest.raises(FileNotFoundError):
        load(model_path, vectorizer_path)

def test_load_files_not_a_file(tmp_path):
    model_path = tmp_path / "model_dir"
    vectorizer_path = tmp_path / "vectorizer_dir"
    
    model_path.mkdir()
    vectorizer_path.mkdir()
    
    with pytest.raises(ValueError):
        load(model_path, vectorizer_path)

def test_load_files_empty(tmp_path):
    model_path = tmp_path / "model.joblib"
    vectorizer_path = tmp_path / "vectorizer.joblib"
    
    model_path.touch()
    vectorizer_path.touch()
    
    with pytest.raises(ValueError):
        load(model_path, vectorizer_path)

def test_predict_empty_text(model, vectorizer):
    with pytest.raises(ValueError):
        predict("", model, vectorizer)

def test_predict_whitespace_text(model, vectorizer):
    with pytest.raises(ValueError):
        predict("   ", model, vectorizer)

def test_predict_model_vectorizer_none():
    with pytest.raises(ValueError):
        predict("Test text", None, None)

def test_predict_model_not_logistic_regression(vectorizer):
    class DummyClassifier:
        def predict(self, X):
            return [0]
        def predict_proba(self, X):
            return [[0.5, 0.5]]
    
    dummy_model = DummyClassifier()
    
    with pytest.raises(ValueError):
        predict("Test text", dummy_model, vectorizer)

def test_predict_vectorizer_not_tfidf(vectorizer):
    class DummyRegressor:
        def transform(self, texts):
            return [[0.1, 0.2, 0.3]]
    
    dummy_vectorizer = DummyRegressor()
    model = LogisticRegression()
    
    with pytest.raises(ValueError):
        predict("Test text", model, dummy_vectorizer) 

def test_predict_valid_input(model, vectorizer):
    # Assuming the model and vectorizer are properly trained and loaded
    predicted_label, confidence_score = predict("Test text", model, vectorizer)
    
    assert isinstance(predicted_label, str)
    assert predicted_label in ["Stress", "Depression", "Bipolar disorder", "Personality disorder", "Anxiety"]
    assert isinstance(confidence_score, float)
    assert 0.0 <= confidence_score <= 1.0