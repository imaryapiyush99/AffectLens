import pytest, joblib
from affectlens.evaluate import _predictions_to_binary, evaluate_classifier, optimize_threshold, optimize_ensemble_weight, compare_sentiment_models
from affectlens.constants import EMOTIONS
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def test_predictions_to_binary_invalid_input():
    with pytest.raises(ValueError):
        _predictions_to_binary("invalid_input", ["joy", "sadness"])
    with pytest.raises(ValueError):
        _predictions_to_binary([{"joy": 0.8}], "invalid_emotions")
    with pytest.raises(ValueError):
        _predictions_to_binary([], ["joy", "sadness"])
    with pytest.raises(ValueError):
        _predictions_to_binary([{"joy": 0.8}], [])

def test_predictions_to_binary_valid_input():
    predictions = [
    {"Happiness": 0.8, "Curiosity": 0.7},
    {"Sadness": 0.9}
    ]
    expected_output = [
    [1,0,0,0,0,0,0,1,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0]
    ]
    assert _predictions_to_binary(predictions, EMOTIONS) == expected_output

def test_evaluate_classifier_invalid_input(tmp_path):
    # Create a temporary CSV file for testing
    test_csv = tmp_path / "test.csv"
    test_csv.write_text("text,joy,sadness\nHello world,1,0\n")
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"
    model_path.write_text("dummy model content")
    vectorizer_path.write_text("dummy vectorizer content")

    with pytest.raises(ValueError):
        evaluate_classifier("invalid_path", ["joy", "sadness"], "model_path", "vectorizer_path", 0.5)
    with pytest.raises(ValueError):
        evaluate_classifier("test_csv", "invalid_columns", "model_path", "vectorizer_path", 0.5)
    with pytest.raises(ValueError):
        evaluate_classifier("test_csv", ["joy", "sadness"], "model_path", "vectorizer_path", -0.1)
    with pytest.raises(ValueError):
        evaluate_classifier("test_csv", ["joy", "sadness"], "model_path", "vectorizer_path", 1.5)
    with pytest.raises(ValueError):
        evaluate_classifier("test_csv", [], "model_path", "vectorizer_path", 0.5)
    with pytest.raises(ValueError):
        evaluate_classifier("test_csv", ["joy", ""], "model_path", "vectorizer_path", 0.5)
    with pytest.raises(ValueError):
        evaluate_classifier("test_csv", ["joy", "sadness"], "model_path", "vectorizer_path", "invalid_threshold")
    with pytest.raises(ValueError):
        evaluate_classifier("test_pdf", ["joy", "sadness"], "model_path", "vectorizer_path", 0.5)    

def test_evaluate_classifier_valid_input(tmp_path): 
    # Create a temporary CSV file for testing
    test_csv = tmp_path / "test.csv"
    test_csv.write_text(
       "text,Happiness,Sadness,Anger,Fear,Disgust,Surprise,Neutral,Curiosity,Confusion,Realization,Desire\n"
        "happy,1,0,0,0,0,0,0,0,0,0,0\n"
        "sad,0,1,0,0,0,0,0,0,0,0,0\n"
        "angry,0,0,1,0,0,0,0,0,0,0,0\n"
        "fear,0,0,0,1,0,0,0,0,0,0,0\n"
        "disgust,0,0,0,0,1,0,0,0,0,0,0\n"
        "surprise,0,0,0,0,0,1,0,0,0,0,0\n"
        "neutral,0,0,0,0,0,0,1,0,0,0,0\n"
        "curious,0,0,0,0,0,0,0,1,0,0,0\n"
        "confused,0,0,0,0,0,0,0,0,1,0,0\n"
        "realization,0,0,0,0,0,0,0,0,0,1,0\n"
        "desire,0,0,0,0,0,0,0,0,0,0,1\n"
        "happy sad,1,1,0,0,0,0,0,0,0,0,0\n"
        "angry fear,0,0,1,1,0,0,0,0,0,0,0\n"
        "disgust surprise,0,0,0,0,1,1,0,0,0,0,0\n"
        "neutral curious,0,0,0,0,0,0,1,1,0,0,0\n"
        "confused realization,0,0,0,0,0,0,0,0,1,1,0\n"
        "realization desire,0,0,0,0,0,0,0,0,0,1,1\n"
    ) 

    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"
    # Create a simple model and vectorizer for testing
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(["happy",
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
        "none",])
    
    y = [
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
    model = LogisticRegression()
    clf = MultiOutputClassifier(model)
    clf.fit(X, y)
    # Save the model and vectorizer to disk     
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    # Assuming the evaluate_classifier function is implemented correctly, it should return a dictionary of metrics
    result = evaluate_classifier(str(test_csv), EMOTIONS, str(model_path), str(vectorizer_path), 0.5)
    assert isinstance(result, dict)
    assert all(metric in result for metric in ["micro_f1", "macro_f1", "weighted_f1", "hamming_loss"])


def test_optimize_threshold_invalid_input(tmp_path):
    validation_csv_path = tmp_path / "validation.csv"
    validation_csv_path.write_text(
       "text,Happiness,Sadness,Anger,Fear,Disgust,Surprise,Neutral,Curiosity,Confusion,Realization,Desire\n"
        "happy,1,0,0,0,0,0,0,0,0,0,0\n"
        "sad,0,1,0,0,0,0,0,0,0,0,0\n"
        "angry,0,0,1,0,0,0,0,0,0,0,0\n"
        "fear,0,0,0,1,0,0,0,0,0,0,0\n"
        "disgust,0,0,0,0,1,0,0,0,0,0,0\n"
        "surprise,0,0,0,0,0,1,0,0,0,0,0\n"
        "neutral,0,0,0,0,0,0,1,0,0,0,0\n"
        "curious,0,0,0,0,0,0,0,1,0,0,0\n"
        "confused,0,0,0,0,0,0,0,0,1,0,0\n"
        "realization,0,0,0,0,0,0,0,0,0,1,0\n"
        "desire,0,0,0,0,0,0,0,0,0,0,1\n"
        "happy sad,1,1,0,0,0,0,0,0,0,0,0\n"
        "angry fear,0,0,1,1,0,0,0,0,0,0,0\n"
        "disgust surprise,0,0,0,0,1,1,0,0,0,0,0\n"
        "neutral curious,0,0,0,0,0,0,1,1,0,0,0\n"
        "confused realization,0,0,0,0,0,0,0,0,1,1,0\n"
        "realization desire,0,0,0,0,0,0,0,0,0,1,1\n"
    ) 
    with pytest.raises(ValueError):
        optimize_threshold(str(validation_csv_path), ["joy", "sadness"], "model_path", "vectorizer_path")
    with pytest.raises(ValueError):
        optimize_threshold(str(validation_csv_path), "invalid_columns", "model_path", "vectorizer_path")
    with pytest.raises(ValueError):
        optimize_threshold(str(validation_csv_path), [], "model_path", "vectorizer_path")
    with pytest.raises(ValueError):
        optimize_threshold(str(validation_csv_path), ["joy", ""], "model_path", "vectorizer_path")
    with pytest.raises(ValueError):
        optimize_threshold(str(validation_csv_path), ["joy", "sadness"], "model_path", "vectorizer_path")

def test_optimize_threshold_valid_input(tmp_path): 
    # Create a temporary CSV file for testing
    validation_csv = tmp_path / "validation.csv"
    validation_csv.write_text(
       "text,Happiness,Sadness,Anger,Fear,Disgust,Surprise,Neutral,Curiosity,Confusion,Realization,Desire\n"
        "happy,1,0,0,0,0,0,0,0,0,0,0\n"
        "sad,0,1,0,0,0,0,0,0,0,0,0\n"
        "angry,0,0,1,0,0,0,0,0,0,0,0\n"
        "fear,0,0,0,1,0,0,0,0,0,0,0\n"
        "disgust,0,0,0,0,1,0,0,0,0,0,0\n"
        "surprise,0,0,0,0,0,1,0,0,0,0,0\n"
        "neutral,0,0,0,0,0,0,1,0,0,0,0\n"
        "curious,0,0,0,0,0,0,0,1,0,0,0\n"
        "confused,0,0,0,0,0,0,0,0,1,0,0\n"
        "realization,0,0,0,0,0,0,0,0,0,1,0\n"
        "desire,0,0,0,0,0,0,0,0,0,0,1\n"
        "happy sad,1,1,0,0,0,0,0,0,0,0,0\n"
        "angry fear,0,0,1,1,0,0,0,0,0,0,0\n"
        "disgust surprise,0,0,0,0,1,1,0,0,0,0,0\n"
        "neutral curious,0,0,0,0,0,0,1,1,0,0,0\n"
        "confused realization,0,0,0,0,0,0,0,0,1,1,0\n"
        "realization desire,0,0,0,0,0,0,0,0,0,1,1\n"
    ) 
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"
    # Create a simple model and vectorizer for testing
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(["happy",
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
        "none",])
    
    y = [
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
    model = LogisticRegression()
    clf = MultiOutputClassifier(model)
    clf.fit(X, y)
    # Save the model and vectorizer to disk     
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    optimal_threshold, best_f1 = optimize_threshold(str(validation_csv), EMOTIONS, str(model_path), str(vectorizer_path))
    assert isinstance(optimal_threshold, float)
    assert isinstance(best_f1, float)    
                        
def test_optimize_ensemble_weight_invalid_input():
    with pytest.raises(ValueError):
        optimize_ensemble_weight("invalid_dataset")
    with pytest.raises(ValueError):
        optimize_ensemble_weight([])
    with pytest.raises(ValueError):
        optimize_ensemble_weight([("sentence", "invalid_score")])
    with pytest.raises(ValueError):
        optimize_ensemble_weight([("sentence", -0.1)])
    with pytest.raises(ValueError):
        optimize_ensemble_weight([("sentence", 1.5)])


def test_optimize_ensemble_weight_valid_input():
    dataset = [
        ("I am happy", 0.8),
        ("I am sad", 0.2),
        ("I am angry", 0.1),
        ("I am fearful", 0.3),
        ("I am disgusted", 0.4),
    ]
    optimal_weight, best_f1 = optimize_ensemble_weight(dataset)
    assert isinstance(optimal_weight, float)
    assert isinstance(best_f1, float)      

def test_compare_sentiment_models_invalid_input():
    with pytest.raises(ValueError):
        compare_sentiment_models("invalid_dataset")
    with pytest.raises(ValueError):
        compare_sentiment_models([])
    with pytest.raises(ValueError):
        compare_sentiment_models([("sentence", "invalid_score")])
    with pytest.raises(ValueError):
        compare_sentiment_models([("sentence", -0.1)])
    with pytest.raises(ValueError):
        compare_sentiment_models([("sentence", 1.5)])
    with pytest.raises(ValueError):
        compare_sentiment_models([("sentence", 0.5), ("sentence2", "invalid_score")])

def test_compare_sentiment_models_valid_input():
    dataset = [
        ("I am happy", 0.8),
        ("I am sad", 0.2),
        ("I am angry", 0.1),
        ("I am fearful", 0.3),
        ("I am disgusted", 0.4),
    ]
    results = compare_sentiment_models(dataset)
    assert isinstance(results, dict)
    assert all(isinstance(model, str) and isinstance(metric, float) for model, metric in results.items())
