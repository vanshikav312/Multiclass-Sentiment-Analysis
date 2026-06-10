# MindScan — Multiclass Mental Health Sentiment Analyzer

A full-stack mental health sentiment analysis application that classifies text across four dimensions — **emotion, depression risk, anxiety risk, and eating behavior** — using a React + FastAPI interface backed by scikit-learn models trained on five real-world datasets.

> **Note on scope:** This is a research and learning project for text-signal detection. It is **not** a diagnostic tool and should not be used for clinical or medical decision-making.

---

## Results

| Model | Test Accuracy | F1 (macro) | 5-Fold CV | Classes | Data Type |
|---|---|---|---|---|---|
| Depression | 96.47% | 96.28% | 97.20% ± 0.37% | 2 | Text |
| Anxiety | 96.58% | 96.58% | 96.58% ± 0.43% | 2 | Text |
| Emotion | 91.53% | 91.52% | — | 6 | Text |
| Eating Behavior | 100.00%\* | 100.00%\* | 100.00% ± 0.00% | 5 | Text |
| Stress | 32.83% | 32.86% | 33.87% ± 1.32% | 3 | Sensor |

\* **Eating-behavior 100% is reported honestly, not as a success claim.** The dataset is small (~1,000 rows) and cleanly separable — this number reflects dataset simplicity and high overfitting risk, not real-world robustness.

**On the stress model:** it uses physiological sensor features (not text) so it is excluded from the text-inference UI. Its low accuracy reflects a genuinely hard sensor-fusion task.

---

## What This Project Does

- Trains **five independent classifiers** on separate datasets covering distinct mental-health signals.
- Handles **class imbalance** via stratified sampling and balanced class weights.
- Returns **per-class probability distributions**, not just a single label — useful for borderline cases.
- Includes a **CLI inference script** and a **React + FastAPI dashboard** for real-time text analysis.

---

## Preprocessing Pipeline

All text goes through 15 steps in a shared `preprocessing.py` before reaching the model — the same function used at training time and inference time:

1. Emoticon replacement (`:)` → `happy`)
2. Lowercase
3. Contraction expansion (`can't` → `cannot`)
4. Slang normalization (`lol` → `laughing`, `tbh` → `to be honest`)
5. URL and HTML tag removal
6. Repeated character normalization (`sooooo` → `soo`)
7. Negation handling (`not happy` → `not not_happy`, up to 3 words)
8. Punctuation removal (preserving `not_` prefix tokens)
9. Digit removal
10. Whitespace normalization
11. NLTK word tokenization
12. Stopword removal (preserving negation tokens)
13. POS-aware lemmatization
14. Short token filtering (length > 2)

---

## Modeling Approach

- **Emotion model:** TF-IDF + VADER sentiment scores (compound, positive, negative, neutral) → Logistic Regression. VADER features on raw text improve detection on short and conversational sentences.
- **Depression, Anxiety, Eating models:** TF-IDF → Logistic Regression with balanced class weights.
- **Stress model:** physiological/sensor features → Random Forest (not used in text inference).

---

## UI Features

- **Disclaimer banner** — prominently displayed at the top on every visit, clarifying this is not a medical tool
- **Input validation** — button disabled and amber warning shown when input is under 5 words
- **Confidence threshold** — emotion result shows **"Uncertain"** when top confidence is below 55%, preventing the model from displaying a wrong label with false confidence
- **Label corrections** — training data typo `suprise` is displayed as `Surprise` everywhere in the UI

---

## Limitations

- **Eating-behavior 100% accuracy** almost certainly reflects overfitting on a small, cleanly-separable dataset.
- **Anxiety dataset has no negative examples** in the source data — the model was trained only on anxious text, which limits its ability to distinguish anxious from non-anxious input reliably.
- **Implicit emotion and sarcasm** are not reliably captured — e.g. "I'm so done" or sarcastic positive text can be misclassified.
- **Short, vague sentences** (e.g. "I feel weird") may produce low-confidence or incorrect predictions due to sparse TF-IDF features.
- This is a **research project**, not a validated diagnostic system.

---

## Project Structure

```
├── backend/
│   ├── main.py                                    # FastAPI server — /api/analyze, /api/metrics, /api/explain
│   └── requirements.txt
├── frontend/
│   ├── package.json                               # React + Vite (npm run dev:all starts both)
│   └── src/
│       ├── App.jsx
│       ├── index.css
│       └── components/
│           ├── Analyzer.jsx                       # Text input + live results
│           └── Dashboard.jsx                      # Evaluation metrics + charts
├── notebook/
│   ├── nlp_multiclass_sentiment_analysis.ipynb   # Full training pipeline
│   ├── model_emotion.pkl                          # Emotion classifier (TF-IDF + VADER)
│   ├── model_depression.pkl
│   ├── model_anxiety.pkl
│   ├── model_eating.pkl
│   ├── model_stress.pkl
│   └── model_performance_summary.csv
├── preprocessing.py                               # Shared preprocess_text() — training and inference
├── predict.py                                     # CLI inference script
├── retrain_emotion.py                             # Retrain script for emotion model
└── dataset/                                       # Raw CSVs for all 5 tasks
```

---

## Setup & Running

### Prerequisites

- Python 3.11+
- Node.js 18+

### Install Python dependencies

```bash
pip install fastapi uvicorn scikit-learn nltk pandas numpy vaderSentiment
```

NLTK data downloads automatically on first run.

### Install frontend dependencies

```bash
cd frontend
npm install
```

### Run both together (one command)

```bash
cd frontend
npm run dev:all
```

- Frontend → [http://localhost:5173](http://localhost:5173)
- API → [http://localhost:8000](http://localhost:8000)

> The backend may take 1–2 minutes on first startup while loading models and NLTK data.

### Run separately

**Backend:**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm run dev
```

### CLI mode

```bash
python predict.py
```

---

## API Reference

### `POST /api/analyze`

```json
{ "text": "I have been feeling really anxious and overwhelmed lately." }
```

**Response:**
```json
{
  "text": "...",
  "cleaned_text": "...",
  "results": {
    "emotion":    { "fear": 0.81, "sad": 0.09, "anger": 0.04, "joy": 0.03, "love": 0.02, "surprise": 0.01 },
    "depression": { "Depressed": 0.76, "Not Depressed": 0.24 },
    "anxiety":    { "Anxious": 0.63, "Not Anxious": 0.37 },
    "eating":     { "emotional": 0.72, "normal": 0.18, "anxiety": 0.06, "obesity": 0.03, "healthy eating": 0.01 }
  }
}
```

### `GET /api/metrics`

Returns model performance summary (accuracy, F1, cross-validation scores).

### `GET /api/explain`

Returns plain-English explanation of each model's strengths and limitations.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | scikit-learn — TF-IDF, Logistic Regression, FeatureUnion |
| NLP | NLTK, VADER Sentiment |
| Backend | FastAPI, Uvicorn |
| Frontend | React 19, Vite 8, Lucide React |
| Language | Python 3.11+ |

---

## Datasets

| Dataset | Source | Rows | Classes |
|---|---|---|---|
| Emotion | Kaggle — tweet emotion dataset | ~87,000 | 6 (joy, sadness, anger, fear, love, surprise) |
| Depression | Reddit mental-health posts | ~7,600 | 2 |
| Anxiety | Reddit anxiety subreddit | ~3,700 | 2 |
| Eating Behavior | Kaggle eating disorder dataset | ~1,000 | 5 |
| Stress (Sensor) | Physiological + behavioral sensor dataset | ~3,000 | 3 |

---

## Author

**Vanshika** — [GitHub](https://github.com/vanshikav312)

## License

MIT
