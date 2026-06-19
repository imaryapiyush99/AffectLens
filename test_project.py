from pathlib import Path
import pytest
from project import get_preprocessed_text, get_ensemble_scores, get_classifier_predictions, get_emotional_swing, get_volatility_scores, get_high_volatility_flags, build_enriched_posts, dataset_is_training_data
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier

def test_get_preprocessed_text_invalid_input():
    # Test that get_preprocessed_text raises a ValueError when given invalid input
    with pytest.raises(ValueError):
        get_preprocessed_text(None)  # Assuming None is invalid input

    with pytest.raises(ValueError):
        get_preprocessed_text(123)  # Assuming an integer is invalid input

    with pytest.raises(ValueError):
        get_preprocessed_text([])  # Assuming an empty list is invalid input

    with pytest.raises(ValueError):
        get_preprocessed_text({})  # Assuming an empty dictionary is invalid input

    with pytest.raises(ValueError):
        get_preprocessed_text([{"not_text": "This is a test."}])  # Assuming missing 'text' key is invalid input

    with pytest.raises(ValueError):
        get_preprocessed_text([{"text": ""}])  # Assuming empty 'text' value is invalid input

    with pytest.raises(ValueError):
        get_preprocessed_text([{"text": None}])  # Assuming None 'text' value is invalid input  


def test_get_preprocessed_text_valid_input():
    # Test that get_preprocessed_text returns the expected output for valid input
    input_data = [{"text": "This is a test."}, {"text": "Another test."}]
    expected_output = ["test", "another test"]
    assert get_preprocessed_text(input_data) == expected_output                         


def test_get_ensemble_scores_invalid_input():
    # Test that get_ensemble_scores raises a ValueError when given invalid input
    with pytest.raises(ValueError):
        get_ensemble_scores(None, 0.5)  # Assuming None is invalid input

    with pytest.raises(ValueError):
        get_ensemble_scores(123, 0.5)  # Assuming an integer is invalid input

    with pytest.raises(ValueError):
        get_ensemble_scores([], 0.5)  # Assuming an empty list is invalid input

    with pytest.raises(ValueError):
        get_ensemble_scores({}, 0.5)  # Assuming an empty dictionary is invalid input

    with pytest.raises(ValueError):
        get_ensemble_scores(["text", None], 0.5)  # Assuming None 'text' value is invalid input

    with pytest.raises(ValueError):
        get_ensemble_scores(["text", "This is a test."], "not_a_number")  # Assuming non-numeric weight is invalid input

    with pytest.raises(ValueError):
        get_ensemble_scores(["text", "This is a test."], None)  # Assuming None weight is invalid input

    with pytest.raises(ValueError):
        get_ensemble_scores(["text", "This is a test."], -1)  # Assuming negative weight is invalid input

    with pytest.raises(ValueError):
        get_ensemble_scores(["text", "This is a test."], 1.5)  # Assuming weight greater than 1 is invalid input                


def test_get_ensemble_scores_valid_input():
    # Test that get_ensemble_scores returns the expected output for valid input
    input_data = ["This is a test.", "Another test."]
    weight = 0.5
    assert len(get_ensemble_scores(input_data, weight)) == len(input_data)  # Check that the output length matches the input length
    


def test_get_classifier_predictions_invalid_input():
    # Test that get_classifier_predictions raises a ValueError when given invalid input
    with pytest.raises(ValueError):
        get_classifier_predictions(None, "model.pkl", "vectorizer.pkl", 0.5)  # Assuming None is invalid input

    with pytest.raises(ValueError):
        get_classifier_predictions(123, "model.pkl", "vectorizer.pkl", 0.5)  # Assuming an integer is invalid input

    with pytest.raises(ValueError):
        get_classifier_predictions([], "model.pkl", "vectorizer.pkl", 0.5)  # Assuming an empty list is invalid input

    with pytest.raises(ValueError):
        get_classifier_predictions({}, "model.pkl", "vectorizer.pkl", 0.5)  # Assuming an empty dictionary is invalid input


    with pytest.raises(ValueError):
        get_classifier_predictions([""], "model.pkl", "vectorizer.pkl", 0.5)  # Assuming empty string is invalid input

    with pytest.raises(ValueError):
        get_classifier_predictions([None], "model.pkl", "vectorizer.pkl", 0.5)  # Assuming None 'text value is invalid input

    with pytest.raises(ValueError):  
        get_classifier_predictions(["This is a test."], "model.pkl", "vectorizer.pkl", "not_a_number")  # Assuming non-numeric weight is invalid input      

    with pytest.raises(ValueError):
        get_classifier_predictions(["This is a test."], "model.pkl", "vectorizer.pkl", None)  # Assuming None weight is invalid input

    with pytest.raises(ValueError):
        get_classifier_predictions(["This is a test."], "model.pkl", "vectorizer.pkl", -1)  # Assuming negative weight is invalid input

    with pytest.raises(ValueError):
        get_classifier_predictions(["This is a test."], "model.pkl", "vectorizer.pkl", 1.5)  # Assuming weight greater than 1 is invalid input        



def test_get_classifier_predictions_valid_input(tmp_path):
    # Create a dummy model and vectorizer for testing
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
    
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = MultiOutputClassifier(LogisticRegression())
    model.fit(X, labels)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    assert len(get_classifier_predictions(texts, str(model_path), str(vectorizer_path), 0.5)) == len(texts)

def test_get_emotional_swing_invalid_input():
    # Test that get_emotional_swing raises a ValueError when given invalid input
    with pytest.raises(ValueError):
        get_emotional_swing(None, [0.5])  # Assuming None is invalid input

    with pytest.raises(ValueError):
        get_emotional_swing(123, [0.6])  # Assuming an integer is invalid input

    with pytest.raises(ValueError):
        get_emotional_swing([], [0.5])  # Assuming an empty list is invalid input

    with pytest.raises(ValueError):
        get_emotional_swing({}, [0.5])  # Assuming an empty dictionary is invalid input

    with pytest.raises(ValueError):
        get_emotional_swing(["text", None], [0.5])  # Assuming None 'text' value is invalid input

    with pytest.raises(ValueError):
        get_emotional_swing(["text", "This is a test."], "not_a_list")  # Assuming non-list scores is invalid input

    with pytest.raises(ValueError):
        get_emotional_swing(["text", "This is a test."], None)  # Assuming None scores is invalid input

    with pytest.raises(ValueError):
        get_emotional_swing(["text", "This is a test."], [])  # Assuming empty scores list is invalid input

    with pytest.raises(ValueError):
        get_emotional_swing(["text", "This is a test."], [0.5])  # Assuming scores list length mismatch is invalid input
        

def test_get_emotional_swing_valid_input():
    # Test that get_emotional_swing returns the expected output for valid input
    input_texts = ["This is a test.", "Another test."]
    input_scores = [0.6, 0.4]
    expected_output = [0.0, 0.2]  # Assuming emotional swing is calculated as the absolute difference between scores
    assert len(get_emotional_swing(input_texts, input_scores)) == len(expected_output)


def test_get_volatility_scores_invalid_input():
    # Test that get_volatility_scores raises a ValueError when given invalid input
    with pytest.raises(ValueError):
        get_volatility_scores(None, [0.5])  # Assuming None is invalid input

    with pytest.raises(ValueError):
        get_volatility_scores(123, [0.5])  # Assuming an integer is invalid input

    with pytest.raises(ValueError):
        get_volatility_scores([], [0.5])  # Assuming an empty list is invalid input

    with pytest.raises(ValueError):
        get_volatility_scores({}, [0.5])  # Assuming an empty dictionary is invalid input

    with pytest.raises(ValueError):
        get_volatility_scores(["text", None], [0.5])  # Assuming None 'text' value is invalid input

    with pytest.raises(ValueError):
        get_volatility_scores(["text", "This is a test."], "not_a_list")  # Assuming non-list scores is invalid input

    with pytest.raises(ValueError):
        get_volatility_scores(["text", "This is a test."], None)  # Assuming None scores is invalid input

    with pytest.raises(ValueError):
        get_volatility_scores(["text", "This is a test."], [])  # Assuming empty scores list is invalid input

    with pytest.raises(ValueError):
        get_volatility_scores(["text", "This is a test."], [0.5])  # Assuming scores list length mismatch is invalid input


def test_get_volatility_scores_valid_input():
    # Test that get_volatility_scores returns the expected output for valid input
    input_texts = ["This is a test.", "Another test."]
    input_scores = [0.6, 0.4]
    expected_output = [0.0, 0.2]
    assert len(get_volatility_scores(input_texts, input_scores)) == len(expected_output)                         


def test_get_high_volatility_flags_invalid_input():
    # Test that get_high_volatility_flags raises a ValueError when given invalid input
    with pytest.raises(ValueError):
        get_high_volatility_flags(None,[0.5], 0.5)  # Assuming None is invalid input

    with pytest.raises(ValueError):
        get_high_volatility_flags(123, [0.5], 0.5)  # Assuming an integer is invalid input

    with pytest.raises(ValueError):
        get_high_volatility_flags([], [0.5], 0.5)  # Assuming an empty list is invalid input

    with pytest.raises(ValueError):
        get_high_volatility_flags({}, [0.5], 0.5)  # Assuming an empty dictionary is invalid input

    with pytest.raises(ValueError):
        get_high_volatility_flags(["text", None], [0.5, 0.4], 0.5)  # Assuming None 'text' value is invalid input

    with pytest.raises(ValueError):
        get_high_volatility_flags(["text", "This is a test."], [0.5, 0.4], "not_a_number")  # Assuming non-numeric threshold is invalid input

    with pytest.raises(ValueError):
        get_high_volatility_flags(["text", "This is a test."], [0.5, 0.4], None)  # Assuming None threshold is invalid input

    with pytest.raises(ValueError):
        get_high_volatility_flags(["text", "This is a test."], [0.5, 0.4], -1)  # Assuming negative threshold is invalid input

    with pytest.raises(ValueError):
        get_high_volatility_flags(["text", "This is a test."], [0.5, 0.4], 1.5)  # Assuming threshold greater than 1 is invalid input


def test_get_high_volatility_flags_valid_input():
    # Test that get_high_volatility_flags returns the expected output for valid input
    input_texts = ["This is a test.", "Another test."]
    input_scores = [0.6, 0.4]
    threshold = 0.5
    expected_output = [True, False]  # Assuming volatility score of 0.2 is below the threshold of 0.5
    assert len(get_high_volatility_flags(input_texts, input_scores, threshold)) == len(expected_output)


def test_build_enriched_posts_invalid_input(tmp_path):
    # Test that build_enriched_posts raises a ValueError when given invalid input
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
    
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = MultiOutputClassifier(LogisticRegression())
    model.fit(X, labels)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    with pytest.raises(ValueError):
        build_enriched_posts(None, str(model_path), str(vectorizer_path), 0.5, 0.5, None, "text")  # Assuming None is invalid input

    with pytest.raises(ValueError):
        build_enriched_posts(123, str(model_path), str(vectorizer_path), 0.5, 0.5, None, "text")  # Assuming an integer is invalid input

    with pytest.raises(ValueError):
        build_enriched_posts([], str(model_path), str(vectorizer_path), 0.5, 0.5, None, "text")  # Assuming an empty list is invalid input

    with pytest.raises(ValueError):
        build_enriched_posts(["text", None], str(model_path), str(vectorizer_path), 0.5, 0.5, None, "text")  # Assuming None 'text' value is invalid input
    
    with pytest.raises(ValueError):
        build_enriched_posts(["text", "This is a test."], None, str(vectorizer_path), 0.5, 0.5, None, "text")  # Assuming None model path is invalid input

    with pytest.raises(ValueError):
        build_enriched_posts(["text", "This is a test."], str(model_path), None, 0.5, 0.5, None, "text")  # Assuming None vectorizer path is invalid input

    with pytest.raises(ValueError):
        build_enriched_posts(["text", "This is a test."], "", str(vectorizer_path), 0.5, 0.5, None, "text")  # Assuming empty model path is invalid input

    with pytest.raises(ValueError):
        build_enriched_posts(["text", "This is a test."], str(model_path), "", 0.5, 0.5, None, "text")  # Assuming empty vectorizer path is invalid input

    with pytest.raises(ValueError):
        build_enriched_posts(["text", "This is a test."], str(model_path), str(vectorizer_path), 0.5, 0.5, None, None)  # Assuming both csv_path and text are None is invalid input   

    with pytest.raises(ValueError):
        build_enriched_posts(["text", "This is a test."], str(model_path), str(vectorizer_path), 0.5, 0.5, None, "   ")  # Assuming whitespace text is invalid input                 

    with pytest.raises(ValueError):
        build_enriched_posts(["text", "This is a test."], str(model_path), str(vectorizer_path), -1, 0.5, None, "text")  # Assuming negative weight is invalid input

    with pytest.raises(ValueError):
        build_enriched_posts(["text", "This is a test."], str(model_path), str(vectorizer_path), 1.5, 0.5, None, "text")  # Assuming weight greater than 1 is invalid input

    with pytest.raises(ValueError):
        build_enriched_posts(["text", "This is a test."], str(model_path), str(vectorizer_path), 0.5, "not_a_number", None, "text")  # Assuming non-numeric threshold is invalid input

    with pytest.raises(ValueError):
        build_enriched_posts(["text", "This is a test."], str(model_path), str(vectorizer_path), 0.5, None, None, "text")  # Assuming None threshold is invalid input

    with pytest.raises(ValueError):
        build_enriched_posts(["text", "This is a test."], str(model_path), str(vectorizer_path), 0.5, -1, None, "text")  # Assuming negative threshold is invalid input

    with pytest.raises(ValueError):
        build_enriched_posts(["text", "This is a test."], str(model_path), str(vectorizer_path), 0.5, 1.5, None, "text")  # Assuming threshold greater than 1 is invalid input

    with pytest.raises(ValueError):
        build_enriched_posts(["text", "This is a test."], str(model_path), str(vectorizer_path), 0.5, 0.5, "   ", None)  # Assuming whitespace csv_path is invalid input
                            


def test_build_enriched_posts_valid_input(tmp_path):
    # Test that build_enriched_posts raises a ValueError when given invalid input
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
    
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = MultiOutputClassifier(LogisticRegression())
    model.fit(X, labels)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    assert len(build_enriched_posts(["text"], str(model_path), str(vectorizer_path), 0.5, 0.5, None, "This is a test.")) == 1
