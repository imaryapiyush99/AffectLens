# AffectLens

AffectLens is a Python-based emotional analytics toolkit that combines emotion classification, sentiment analysis, volatility detection, and visual analytics to uncover emotional patterns in text datasets.

Built using traditional machine learning, ensemble sentiment analysis, and statistical emotion tracking, AffectLens transforms raw text into actionable emotional insights.

---

## Features

### Emotion Analysis
- Multi-label emotion classification
- Confidence scores for each predicted emotion
- Dominant emotion extraction
- Emotion distribution analysis

### Sentiment Analysis
- VADER sentiment scoring
- TextBlob sentiment scoring
- Weighted ensemble sentiment score

### Emotional Dynamics
- Emotional swing detection
- Emotional volatility analysis
- High-volatility flagging

### Optimization
- Classification threshold optimization
- Ensemble weight optimization
- Sentiment model comparison

### Visual Analytics
- Sentiment trend visualization
- Volatility trend visualization
- Emotional swing trend visualization
- Emotion distribution charts
- Emotion heatmaps
- Emotion transition matrices

### Engineering
- CSV dataset support
- Single-text analysis
- Command-line interface
- Comprehensive automated test suite

---

## Pipeline

```text
Input Text
    ↓
Preprocessing
    ↓
Emotion Classification
    ↓
Sentiment Analysis
    ↓
Ensemble Scoring
    ↓
Volatility Analysis
    ↓
Enriched Posts
    ↓
Visual Analytics
```

---

## Project Structure

```text
AffectLens/
│
├── affectlens/
│   ├── classifier.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── volatility.py
│   ├── visualization.py
│   └── constants.py
│
├── models/
│
├── visualizations/
│
├── tests/
│
├── project.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/AffectLens.git
cd AffectLens
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train a Model

```bash
python project.py \
    --input_csv data/go_emotions.csv \
    --model_path models/model.joblib \
    --vectorizer_path models/vectorizer.joblib
```

---

## Analyze a Dataset

```bash
python project.py \
    --input_csv data/test_data/test.csv \
    --model_path models/model.joblib \
    --vectorizer_path models/vectorizer.joblib \
    --output_csv results.csv
```

---

## Analyze a Single Text

```bash
python project.py \
    --text "I am excited about finishing AffectLens." \
    --model_path models/model.joblib \
    --vectorizer_path models/vectorizer.joblib
```

---

## Generate Visualizations

```bash
python project.py \
    --visualize \
    --input_csv data/test_data/test.csv \
    --model_path models/model.joblib \
    --vectorizer_path models/vectorizer.joblib
```

Generated visualizations are saved to:

```text
visualizations/
```

Including:

```text
sentiment_trend.png
volatility_trend.png
emotional_swing_trend.png
emotion_distribution.png
emotion_heatmap.png
emotion_transition_matrix.png
```

---

## Example Analysis Summary

```text
========== AFFECTLENS SUMMARY ==========
Posts analyzed: 31,173

Average sentiment: 0.09
Average volatility: 0.33
Average emotional swing: 0.39

Most common emotion: Neutral

Most positive post:
I wanted to romance [NAME] more than I've wanted to romance more than any other character in the series.
(Score: 0.97)

Most negative post:
That woman is simultaneously very evil and using her evil to make others regret doing evil things.
(Score: -0.98)

High-volatility posts: 22,783 (73.1%)
========================================
```

---

## Visualizations

### Sentiment Trend

Tracks changes in overall sentiment over time using a moving-average smoothing strategy for large datasets.

### Volatility Trend

Visualizes emotional instability and fluctuations across posts.

### Emotional Swing Trend

Measures emotional change between consecutive posts.

### Emotion Distribution

Displays the frequency of detected emotions.

### Emotion Heatmap

Visualizes emotion occurrence patterns across the dataset.

### Emotion Transition Matrix

Shows how dominant emotions transition from one post to the next.

---

## Example Outputs

Add screenshots after generating visualizations:

### Emotion Distribution

![Emotion Distribution](README_assets/emotion_distribution.png)

### Emotion Transition Matrix

![Emotion Transition Matrix](README_assets/emotion_transition_matrix.png)

### Volatility Trend

![Volatility Trend](README_assets/volatility_trend.png)

---

## Testing

Run all tests:

```bash
pytest -v
```

Current status:

```text
81 passed
```

---

## Tech Stack

### Core
- Python

### Machine Learning
- scikit-learn

### NLP
- TextBlob
- VADER Sentiment

### Visualization
- Matplotlib

### Model Persistence
- Joblib

### Testing
- pytest

---

## Key Metrics

- 81 automated tests
- Multi-label emotion classification
- Ensemble sentiment analysis
- Emotional volatility detection
- Emotional swing tracking
- 6 visualization types
- Command-line interface

---

## Future Improvements

- Transformer-based emotion classification
- Interactive dashboard
- Temporal emotion forecasting
- Topic-aware emotion analysis
- Adaptive volatility thresholds
- Emotion trajectory clustering
- Web application deployment

---

## Motivation

Understanding emotions at scale is challenging. While traditional sentiment analysis provides a single positive-to-negative score, real human emotions are multi-dimensional and dynamic.

AffectLens was built to move beyond simple sentiment analysis by combining emotion classification, volatility detection, emotional transitions, and visual analytics into a unified emotional intelligence pipeline.

---

## License

This project is licensed under the MIT License.
