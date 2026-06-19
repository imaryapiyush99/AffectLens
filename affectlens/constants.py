GOEMOTIONS_COLUMNS = ["text", "title", "example_very_unclear", "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]

TARGET_COLUMNS = ["text", "Happiness", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Neutral", "Curiosity", "Confusion", "Realization", "Desire",]

PREDICTION_COLUMNS = ["text"]

OUTPUT_COLUMNS = ["Text", "Ensemble_score", "Emotion", "Emotional_swing", "Volatility_score", "High_volatility_flag"]

EMOTICONS = {
    ":)": " happy ",
    ":-)": " happy ",
    ":(": " sad ",
    ":-(": " sad ",
    ":^)": " happy ",
    ":^(": " sad ",
    ":D": " very_happy ",
    ";)": " wink ",
}


EMOTION_MAPPING: dict[str, list[str]] = {
    "Happiness": ["joy", "amusement", "excitement", "love", "optimism", "gratitude", "pride", "relief", "admiration", "approval", "caring"],
    "Sadness": ["sadness", "grief", "remorse", "disappointment", "embarrassment"],
    "Anger": ["anger", "annoyance", "disapproval"],
    "Fear": ["fear", "nervousness"],
    "Disgust": ["disgust"],
    "Surprise": ["surprise"],
    "Neutral": ["neutral"],
    "Curiosity": ["curiosity"],
    "Confusion": ["confusion"],
    "Realization": ["realization"],
    "Desire": ["desire"],
}

SENTIMENT_MAPPING: dict[str, int] = {
    "negative": 0,
    "neutral": 2,
    "positive": 4
}

EMOTIONS: list[str] = list(EMOTION_MAPPING.keys())

WEIGHT: float = 0.65

THRESHOLD: float = 0.25
