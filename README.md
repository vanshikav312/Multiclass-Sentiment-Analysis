# Multiclass Mental Health Sentiment Analysis

A machine learning system that classifies mental health signals from text across four dimensions — emotion, depression, anxiety, and eating behavior — using TF-IDF + Logistic Regression pipelines trained on 5 real-world datasets.

---

## Results

| Model | Test Accuracy | F1 Score (macro) | 5-Fold CV | Classes |
|---|---|---|---|---|
| Depression | 96.47% | 96.28% | 97.20% ± 0.37% | 2 |
| Anxiety | 96.58% | 96.58% | 96.58% ± 0.43% | 2 |
| Emotion | 94.44% | 93.81% | 94.54% ± 0.19% | 6 |
| Eating Behavior | 100.00% | 100.00% | 100.00% ± 0.00% | 5 |
| Stress (Sensor)* | 32.83% | 32.86% | 33.87% ± 1.32% | 3 |

*Stress model uses physiological sensor data (heart rate, skin conductance, etc.) — not applicable to text input. Low accuracy reflects the difficulty of the sensor-fusion task, not a bug.

---

## What This Project Does

- Trains **5 independent classifiers** on separate mental health datasets (400k+ rows combined)
- Handles **severe class imbalance** through stratified sampling and balanced class weights
- Produces **per-class probability distributions**, not just a single label — useful for borderline cases
- Includes a **Streamlit web app** for real-time text analysis with confidence score visualizations

---

## Problems Diagnosed and Fixed During Development

This project involved debugging several non-trivial data and compatibility issues:

**Data problems:**
- Emotion dataset (422k rows) had multi-label overlap — resolved via label priority rules and deduplication
- Depression and anxiety datasets had 10:1 class imbalance — balanced via undersampling to 3,000 per class
- Eating behavior labels had inconsistent casing and whitespace — normalized during preprocessing

**Compatibility issues:**
- `scikit-learn 1.9` removed the `multi_class` parameter from `LogisticRegression` — updated pipeline accordingly
- `pandas 2.x` changed `groupby().apply()` behavior — fixed aggregation logic
- Notebook `input()` cells replaced with default fallback for non-interactive execution

---

## Project Structure

```
├── notebook/
│   ├── nlp_multiclass_sentiment_analysis.ipynb   # Full training pipeline
│   ├── model_emotion.pkl                          # Trained emotion classifier (6 classes)
│   ├── model_depression.pkl                       # Trained depression classifier
│   ├── model_anxiety.pkl                          # Trained anxiety classifier
│   ├── model_eating.pkl                           # Trained eating behavior classifier
│   ├── model_stress.pkl                           # Stress model (sensor-based)
│   └── model_performance_summary.csv              # All metrics
├── predict.py                                     # CLI inference script (all 5 models)
├── app.py                                         # Streamlit web app
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/vanshikav312/Multiclass-Sentiment-Analysis.git
cd Multiclass-Sentiment-Analysis
pip install -r requirements.txt
```

**Run CLI:**
```bash
python predict.py
```

**Run Web App:**
```bash
streamlit run app.py
```

---

## Tech Stack

- **Python 3.13**
- **scikit-learn 1.9** — TF-IDF vectorization, Logistic Regression, LinearSVC
- **pandas / numpy** — data cleaning and preprocessing
- **NLTK** — tokenization and stopword removal (used in training pipeline)
- **Streamlit** — interactive web interface
- **matplotlib / seaborn** — evaluation visualizations

---

## Datasets

| Dataset | Source | Size |
|---|---|---|
| Emotion | Kaggle (go_emotions / Twitter) | ~422k rows |
| Depression | Reddit mental health posts | ~8k rows |
| Anxiety | Reddit anxiety subreddit | ~6k rows |
| Eating Behavior | Clinical text dataset | ~500 rows |
| Stress (Sensor) | WESAD physiological dataset | ~1.5k rows |
