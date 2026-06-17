import pytest
from affectlens.classifier import needs_training, train, load, predict
from affectlens.constants import THRESHOLD, EMOTIONS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier


def test_needs_training_valid(tmp_path):
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

    # Both files exist but are empty, so needs_training should return True
    assert needs_training(model_path, vectorizer_path) == True

    # Both files exist and are not empty, so needs_training should return False
    model_path.write_text("dummy model content")
    vectorizer_path.write_text("dummy vectorizer content")

    assert needs_training(model_path, vectorizer_path) == False


def test_needs_training_invalid_input(tmp_path):

    model_path = tmp_path / "model_dir"
    vectorizer_path = tmp_path / "vectorizer_dir"
    model_path.mkdir()
    vectorizer_path.mkdir()
    
    # Both paths are directories, so needs_training should return True
    assert needs_training(model_path, vectorizer_path) == True

    # Both files exist but are not valid files, so needs_training should return True
    model_path.touch()
    vectorizer_path.touch()

    assert needs_training(model_path, vectorizer_path) == True

    # Both files exist but are empty, so needs_training should return True  
    model_path = tmp_path / "model.joblib"
    vectorizer_path = tmp_path / "vectorizer.joblib"

    assert needs_training(model_path, vectorizer_path) == True

    # Both files exist and are not empty, so needs_training should return False
    model_path.write_text("dummy model content")
    vectorizer_path.write_text("dummy vectorizer content")

    assert needs_training(model_path, vectorizer_path) == False


def test_train(tmp_path):
    model_path = tmp_path / "model.joblib"
    vectorizer_path = tmp_path / "vectorizer.joblib"

    texts = ["happy", "sad"]

    labels = [
        [1] * len(EMOTIONS),
        [0] * len(EMOTIONS),
    ]

    train(
        texts,
        labels,
        str(model_path),
        str(vectorizer_path)
    )

    assert model_path.exists()
    assert vectorizer_path.exists()
    assert model_path.stat().st_size > 0
    assert vectorizer_path.stat().st_size > 0


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

 
def test_load_valid_files(tmp_path):
    model_path = tmp_path / "model.joblib"
    vectorizer_path = tmp_path / "vectorizer.joblib"

    texts = ["happy", "sad"]
    labels = [[1] * len(EMOTIONS), [0] * len(EMOTIONS)]

    train(
        texts,
        labels,
        str(model_path),
        str(vectorizer_path)
    )

    model, vectorizer = load(
        str(model_path),
        str(vectorizer_path)
    )

    assert isinstance(model, MultiOutputClassifier)
    assert isinstance(vectorizer, TfidfVectorizer)

  

def test_predict_invalid_input(tmp_path):
    # Empty text input
    with pytest.raises(ValueError):
        predict("", None, None, THRESHOLD)

    # Non-string text input
    with pytest.raises(ValueError):
        predict(123, None, None, THRESHOLD)

    # Non-numeric threshold input
    with pytest.raises(ValueError):
        predict("Test text", None, None, "invalid_threshold")

    # Text input with only whitespace
    with pytest.raises(ValueError):
        predict("   ", None, None, THRESHOLD)

    # None model and vectorizer
    with pytest.raises(ValueError):
        predict("Test text", None, None, THRESHOLD)

    # Invalid model type
    with pytest.raises(ValueError):
        predict("Test text", "invalid_model", None, THRESHOLD)

    # Invalid vectorizer type
    with pytest.raises(ValueError):
        predict("Test text", None, "invalid_vectorizer", THRESHOLD)

    # Threshold out of range
    with pytest.raises(ValueError):
        predict("Test text", None, None, -0.1)

    # Model loaded but vectorizer None
    model_path = tmp_path / "model.joblib"
    vectorizer_path = tmp_path / "vectorizer.joblib"

    texts = [

        "happy",
        "sad",
        "angry",
        "fear",
        "disgust",
        "surprise",
        "neutral",
        "curious",
        "confused",
        "realization",
        "desire",
        "none",
    ]

    labels = [
        [1,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,0,0],
        ]

    train(
        texts,
        labels,
        str(model_path),
        str(vectorizer_path)
    )

    clf, vectorizer = load(
        str(model_path),
        str(vectorizer_path)
    )
    
    with pytest.raises(ValueError):
        predict("Test text", clf, None, THRESHOLD)

    # Vectorizer loaded but model None
    with pytest.raises(ValueError):
        predict("Test text", None, vectorizer, THRESHOLD)

    # Valid model and vectorizer but threshold out of range
    with pytest.raises(ValueError):
        predict("Test text", clf, vectorizer, 1.5)



def test_predict_valid_input(tmp_path):
    model_path = tmp_path / "model.joblib"
    vectorizer_path = tmp_path / "vectorizer.joblib"

    texts = [

        "happy",
        "sad",
        "angry",
        "fear",
        "disgust",
        "surprise",
        "neutral",
        "curious",
        "confused",
        "realization",
        "desire",
        "none",
    ]

    labels = [
        [1,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,0,0],
        ]

    train(
        texts,
        labels,
        str(model_path),
        str(vectorizer_path)
    )

    clf, vectorizer = load(
        str(model_path),
        str(vectorizer_path)
    )

    predictions = predict(
        "I am happy",
        clf,
        vectorizer,
        0.0
    )

    assert isinstance(predictions, dict)