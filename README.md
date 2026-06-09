# Multi-Class NLP Mental Health & Emotion Classifier

A machine learning system that detects mental-health and emotional signals from text across four dimensions — **emotion, depression, anxiety, and eating behavior** — using TF-IDF feature extraction with calibrated LinearSVC classifiers, trained end-to-end on five real-world datasets. A separate sensor-based stress model is included for completeness.

> **Note on scope:** This is a research and learning project for text-signal detection. It is **not** a diagnostic tool and should not be used for clinical or medical decision-making.

---

## Results

| Model | Test Accuracy | F1 (macro) | 5-Fold CV | Classes | Data Type |
|---|---|---|---|---|---|
| Depression | 96.47% | 96.28% | 97.20% ± 0.37% | 2 | Text |
| Anxiety | 96.58% | 96.58% | 96.58% ± 0.43% | 2 | Text |
| Emotion | 94.44% | 93.81% | 94.54% ± 0.19% | 6 | Text |
| Eating Behavior | 100.00%\* | 100.00%\* | 100.00% ± 0.00% | 5 | Text |
| Stress | 32.83% | 32.86% | 33.87% ± 1.32% | 3 | Sensor |

\* **Eating-behavior 100% is reported honestly, not as a success claim.** The dataset is small (~1,000 rows, ~200 per class after cleaning) and the classes are cleanly separable, so perfect scores reflect dataset simplicity and a high overfitting risk — not real-world robustness. A larger, noisier test set would be needed to trust this number.

**On the stress model:** it uses physiological sensor features (not text), so it is not part of the text-inference demo. Its low accuracy reflects a genuinely hard sensor-fusion task — see *Limitations* below.

---

## What This Project Does

- Trains **five independent classifiers** on separate datasets covering distinct mental-health and behavioral signals.
- Handles **class imbalance** via stratified sampling and balanced class weights.
- Returns **per-class probability distributions**, not just a single label — useful for borderline cases.
- Includes a **command-line inference script** and a **Streamlit web app** for real-time text analysis.

---

## Modeling Approach

- **Text models (emotion, depression, anxiety, eating):** TF-IDF vectorization → **LinearSVC**, wrapped in `CalibratedClassifierCV` to produce calibrated probabilities alongside class labels.
- **Why LinearSVC:** the vocabulary overlap between classes (e.g. depressed vs. non-depressed text) is high, so a maximum-margin linear boundary separates the classes more reliably than a plain logistic boundary, while remaining fast and interpretable on high-dimensional sparse text features.
- **Stress model:** structured physiological/sensor features (sleep duration, skin conductance, screen time, mobility, Big Five personality scores) → **Random Forest**, chosen because the signal lives in nonlinear interactions between features rather than any single feature.

---

## Problems Diagnosed and Fixed During Development

Real data work, documented honestly:

- **Emotion dataset class imbalance** — one class dominated the raw data (~422k rows unbalanced), biasing predictions; fixed by capping each class to 15,000 samples with stratified sampling.
- **Anxiety dataset had no negative class** — the source contained only positive (anxious) examples; negatives were approximated using non-distressed text from the depression dataset. *(Honest limitation: the model partly learns "anxious vs. normal-language" — see Limitations.)*
- **Depression length bias** — depressed and non-depressed samples differed sharply in average text length, so the model was partly learning length instead of content; mitigated by truncation and balanced sampling to 3,000 per class.
- **Eating-behavior label noise** — inconsistent casing and whitespace in labels; normalized during preprocessing.
- **scikit-learn 1.9 breaking change** — `multi_class` parameter removed from `LogisticRegression`; updated pipeline accordingly.
- **pandas 2.x breaking change** — `groupby().apply()` behavior changed; fixed aggregation logic.
- **Non-interactive notebook execution** — `input()` cells replaced with default fallback for headless execution via `nbconvert`.

---

## Limitations

- **Eating-behavior 100% accuracy** is on a small, cleanly-separable dataset and almost certainly reflects overfitting / dataset simplicity rather than real-world performance.
- **Anxiety negatives are approximated** from non-distressed text (no true negatives existed in the source), which biases what the model actually learns.
- **Implicit emotion and sarcasm** are not reliably captured — e.g. sarcastic positive-sounding text can be misclassified as positive.
- **Stress detection (sensor data)** is the weakest model; structured sensor fusion is a harder problem than the text tasks with the available dataset size (~3,000 rows).
- This is a **research project**, not a validated diagnostic system.

---

## Project Structure

```
├── notebook/
│   ├── nlp_multiclass_sentiment_analysis.ipynb   # Full training pipeline
│   ├── model_emotion.pkl                          # Emotion classifier (6 classes)
│   ├── model_depression.pkl                       # Depression classifier
│   ├── model_anxiety.pkl                          # Anxiety classifier
│   ├── model_eating.pkl                           # Eating-behavior classifier
│   ├── model_stress.pkl                           # Stress model (sensor-based)
│   └── model_performance_summary.csv              # All metrics
├── predict.py                                     # CLI inference (text models)
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

- **Python 3.13.3**
- **scikit-learn 1.9.0** — TF-IDF, LinearSVC, CalibratedClassifierCV, Random Forest
- **pandas 2.3.0 / numpy 2.0.0** — data cleaning and preprocessing
- **NLTK 3.9.4** — tokenization and stopword removal (used in training pipeline)
- **Streamlit** — interactive web interface
- **matplotlib / seaborn** — evaluation visualizations

---

## Datasets

| Dataset | Source | Rows (after cleaning) | Classes |
|---|---|---|---|
| Emotion | Kaggle — tweet emotion dataset | ~87,000 | 6 (joy, sadness, anger, fear, love, surprise) |
| Depression | Reddit mental-health posts (Kaggle) | ~7,600 | 2 (depressed / not depressed) |
| Anxiety | Reddit anxiety subreddit (Kaggle) | ~3,700 | 2 (anxious / not anxious) |
| Eating Behavior | Kaggle eating disorder text dataset | ~1,000 | 5 |
| Stress (Sensor) | Physiological + behavioral sensor dataset | ~3,000 | 3 (low / medium / high) |

---

## Author

**Vanshika Valecha** — [GitHub](https://github.com/vanshikav312)

## License

MIT
