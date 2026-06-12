# MindScan — Multiclass Mental Health Sentiment Analyzer

A full-stack mental health sentiment analysis application that classifies text across four dimensions — **emotion, depression risk, anxiety risk, and eating behavior** — using a React + FastAPI interface backed by scikit-learn models trained on real-world datasets.

> **This is a research and learning project.** It is not a diagnostic tool and should not be used for clinical or medical decision-making.

---

## Model Performance

| Model | Architecture | Test Accuracy | F1 (macro) | 5-Fold CV | Classes |
|---|---|---|---|---|---|
| Depression | TF-IDF (word+char FeatureUnion) + LinearSVC | 96.47% | 96.28% | 97.20% ± 0.37% | 2 |
| Anxiety | TF-IDF + LinearSVC (balanced) | 97.78% | 97.78% | 97.24% ± 0.36% | 2 |
| Emotion | TF-IDF + VADER + Logistic Regression | 91.53% | 91.52% | 89.63% ± 0.15% | 6 |
| Eating Behavior | TF-IDF + LinearSVC | 100.00%\* | 100.00%\* | 100.00% ± 0.00% | 5 |

All CV scores use 5-fold stratified cross-validation on the same dataset used for training.

\* **Eating-behavior 100% reflects dataset structure, not real-world robustness.** See Known Limitations.

---

## Architecture

### Text models

All text goes through a shared 14-step preprocessing pipeline (`preprocessing.py`) before training and at inference time — identical function in both paths:

1. Emoticon replacement (`:)` → `happy`)
2. Lowercase
3. Contraction expansion (`can't` → `cannot`)
4. Slang normalization (`lol` → `laughing out loud`)
5. URL and HTML tag removal
6. Repeated-character normalization (`sooooo` → `soo`)
7. Negation handling (`not happy` → `not_happy`, up to 3-word scope)
8. Punctuation removal (preserving `not_` prefix tokens)
9. Digit removal
10. Whitespace normalization
11. NLTK word tokenization
12. Stopword removal (preserving negation tokens)
13. POS-aware lemmatization (WordNet)
14. Short-token filtering (length > 2)

**Emotion model** additionally feeds raw text through a VADER transformer (compound, positive, negative, neutral scores), concatenated with the TF-IDF vector via `FeatureUnion`. This improves detection on short, conversational sentences where vocabulary alone is sparse.

**Anxiety model** was trained on a balanced set — 3,713 anxious posts paired with 3,713 non-anxious examples approximated from non-distressed posts in the depression dataset (label = 0). The original model had zero negative examples; its reported 96.58% was recall-only. See `experiments/EXPERIMENTS.md` Experiment 2.

---

## Experiment Log

Three experiments documented in [`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md):

- **Experiment 1 (rejected):** GoEmotions dataset as emotion model replacement — 68.51% vs current 91.53%. Rejected.
- **Experiment 2 (applied):** Anxiety zero-negatives fix — original model had no negatives, 96.58% was a recall artefact. Balanced retraining → 97.78% real classification.
- **Experiment 3 (negative result):** Sensor-based stress model — 5 model families (RandomForest, GBM, XGBoost, LightGBM, baseline) all at chance on both 3-class and binary formulations using participant-level GroupKFold. Features do not carry sufficient signal. Model retired from dashboard, archived in `experiments/`.

---

## Known Limitations

**Anxiety — implicit language gap:** sentences with implicit anxiety vocabulary ("terrified", "nervous", "dreading") without the word "anxious" are frequently misclassified as Not Anxious. The training corpus covers clinical/direct anxiety expression but not subtle or indirect forms.

**Emotion — tweet labelling artefact:** the training data labels "overwhelmed" and "exhausted" as `surprise`, following Twitter annotation conventions. This causes those states to be predicted as surprise rather than fear or sadness. The 5-fold CV (89.6%) is lower than test accuracy (91.5%), reflecting real distribution shift across folds.

**Eating behavior — near-duplicate leakage:** after deduplication the dataset has 1,004 rows across 5 classes. Near-duplicate analysis shows 73% of test texts have cosine similarity > 0.8 to a training text; 9.5% are exact matches. The 100% accuracy reflects template-like dataset structure and near-duplicate leakage, not real-world robustness.

**General:** all text models were trained on English Reddit and Twitter posts from specific mental-health communities. They may not generalise to clinical notes, other languages, or formal writing.

---

## Eval Harness

```bash
python eval/run_eval.py
```

Runs 20 labelled sentences through the same `predict()` function the API uses, prints pass/fail per case, and reports an aggregate score (80% on current models). Use this to verify predictions after any model swap.

---

## Project Structure

```
├── backend/
│   ├── main.py                  # FastAPI server — /api/analyze, /api/metrics, /api/explain
│   └── requirements.txt
├── eval/
│   └── run_eval.py              # 20-sentence evaluation harness
├── experiments/
│   ├── EXPERIMENTS.md           # Three documented experiments with full result tables
│   ├── model_anxiety_old.pkl    # Archived pre-fix anxiety model (zero-negatives version)
│   ├── model_stress.pkl         # Archived sensor stress model (negative result)
│   ├── stress_scaler.pkl        # Scaler for stress model features
│   ├── retrain_anxiety.py       # Balanced retraining script (Experiment 2)
│   ├── retrain_emotion.py       # Emotion retrain with VADER features
│   └── retrain_emotion_goemotions.py  # GoEmotions attempt (Experiment 1)
├── frontend/
│   ├── package.json             # React + Vite (npm run dev:all starts both servers)
│   └── src/
│       └── components/
│           ├── Analyzer.jsx     # Text input + results with confidence threshold
│           └── Dashboard.jsx    # Evaluation metrics + charts
├── notebook/
│   ├── nlp_multiclass_sentiment_analysis.ipynb  # Full training pipeline
│   ├── model_emotion.pkl        # TF-IDF + VADER + LR — 91.53% / CV 89.63%
│   ├── model_depression.pkl     # TF-IDF (word+char) + LinearSVC — 96.47% / CV 97.20%
│   ├── model_anxiety.pkl        # TF-IDF + LinearSVC balanced — 97.78% / CV 97.24%
│   ├── model_eating.pkl         # TF-IDF + LinearSVC — 100%* / CV 100%*
│   └── model_performance_summary.csv
├── preprocessing.py             # Shared preprocess_text() — identical at train + inference
├── predict.py                   # CLI inference + predict() used by API
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+

### Install dependencies

```bash
pip install -r requirements.txt
```

NLTK corpora download automatically on first run.

```bash
cd frontend && npm install
```

### Run (one command)

```bash
cd frontend
npm run dev:all
```

- Frontend → http://localhost:5173
- API → http://localhost:8000

> Backend takes 30–60 seconds on first start while loading models and NLTK data.

### Run separately

```bash
# Terminal 1
uvicorn backend.main:app --reload --port 8000

# Terminal 2
cd frontend && npm run dev
```

### CLI

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
    "emotion":    { "fear": 0.81, "sad": 0.09, "anger": 0.04, "joy": 0.03, "love": 0.02, "suprise": 0.01 },
    "depression": { "Depressed": 0.76, "Not Depressed": 0.24 },
    "anxiety":    { "Anxious": 0.63, "Not Anxious": 0.37 },
    "eating":     { "emotional": 0.72, "normal": 0.18, "anxiety": 0.06, "obesity": 0.03, "stress": 0.01 }
  }
}
```

### `GET /api/metrics`

Returns model performance summary — accuracy, macro F1, 5-fold CV mean and std for all four models.

### `GET /api/explain`

Returns plain-English description of each model's architecture, training data, and limitations.

---

## Data Sources

Raw datasets are **not included in the repository** (licensing and data privacy).

| Dataset | Source | Rows (after cleaning) | Notes |
|---|---|---|---|
| Emotion | [Kaggle — Tweet Emotion Dataset](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text) | ~90k (balanced) | 6 classes; balanced from 422k raw rows |
| Depression | Reddit mental-health posts (Kaggle) | ~7,649 | Binary: depressed / not depressed |
| Anxiety | [Kaggle — Anxiety and Depression Detection](https://www.kaggle.com/datasets/ahmedmoorsy/anxiety-and-depression-in-social-media) | ~3,713 positive + ~3,713 negative | Negatives approximated from depression dataset |
| Eating Behavior | [Kaggle — Eating Disorder Dataset](https://www.kaggle.com/datasets) | ~1,004 (after dedup) | 5-class; 85 duplicate rows removed |

To retrain, place the CSVs back in a `dataset/` folder at the project root and run the relevant script from `experiments/`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn 1.9.0 — TF-IDF, FeatureUnion, LinearSVC, LogisticRegression |
| NLP preprocessing | NLTK (tokenization, lemmatization, POS tagging, stopwords) |
| Sentiment features | VADER Sentiment |
| Backend | FastAPI, Uvicorn |
| Frontend | React 19, Vite, Lucide React |

---

## Author

**Vanshika** — [GitHub](https://github.com/vanshikav312)

## License

MIT
