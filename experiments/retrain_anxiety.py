"""
Retrain anxiety model with balanced data.
Negatives: "Not Depressed" rows from depression dataset (general non-distressed text).
Positives: all anxiety dataset rows (label=1).
Pipeline: TfidfVectorizer + LinearSVC wrapped in CalibratedClassifierCV.
sklearn==1.9.0 — do not change.
"""

import os, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'notebook')

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading datasets...")
anx = pd.read_csv(os.path.join(MODEL_DIR, 'anxiety_preprocessed.csv'))
dep = pd.read_csv(os.path.join(MODEL_DIR, 'depression_preprocessed.csv'))

# Positives: all anxious rows
positives = anx[['clean_text']].copy()
positives['label'] = 1
print(f"  Positives (anxious):     {len(positives)}")

# Negatives: not-depressed rows, sampled to match positives
negatives_pool = dep[dep['label'] == 0][['clean_text']].copy()
n_neg = min(len(positives), len(negatives_pool))
negatives = negatives_pool.sample(n_neg, random_state=42)
negatives['label'] = 0
print(f"  Negatives (not-anxious): {len(negatives)}")

df = pd.concat([positives, negatives], ignore_index=True)
df = df[df['clean_text'].str.strip().str.len() > 0]
print(f"  Total after merge:       {len(df)}")
print(f"\nClass distribution:")
print(df['label'].value_counts().to_string())

X = df['clean_text'].tolist()
y = df['label'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

# ── Pipeline ──────────────────────────────────────────────────────────────────

print("\nTraining pipeline (TF-IDF + LinearSVC + CalibratedClassifierCV)...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )),
    ('clf', CalibratedClassifierCV(
        LinearSVC(max_iter=2000, random_state=42, class_weight='balanced'),
        cv=3,
    )),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

acc    = accuracy_score(y_test, y_pred)
f1_mac = f1_score(y_test, y_pred, average='macro')

print(f"\nPer-class report:")
print(classification_report(y_test, y_pred, target_names=['Not Anxious', 'Anxious']))

# ── Comparison table ──────────────────────────────────────────────────────────

print(f"\n{'='*58}")
print(f"  COMPARISON")
print(f"{'='*58}")
print(f"  {'Model':<40} {'Accuracy':>8} {'Macro F1':>8}")
print(f"  {'-'*58}")
print(f"  {'Old  (TF-IDF + LR, no negatives)':<40} {'96.58%':>8} {'96.58%':>8}")
print(f"  {'New  (TF-IDF + LinearSVC, balanced)':<40} {f'{acc:.2%}':>8} {f'{f1_mac:.2%}':>8}")
print(f"{'='*58}")
print("\nNOTE: Old model trained on all-positive data — its 96.58%")
print("was measuring recall only, not real classification ability.")

# ── Sanity test ───────────────────────────────────────────────────────────────

print("\nSanity test:")
probes = [
    ("I cant stop worrying about everything",         "Anxious"),
    ("Im terrified of what might happen",             "Anxious"),
    ("Im nervous about my presentation",              "Anxious"),
    ("Everything makes me anxious I cant stop",       "Anxious"),
    ("I am very happy today",                         "Not Anxious"),
    ("I had a great day with my friends",             "Not Anxious"),
    ("Im proud of what I achieved today",             "Not Anxious"),
    ("I feel completely empty and hopeless",          "Not Anxious"),
    ("I am not feeling good today because of breakup","Uncertain"),
]
for text, expected in probes:
    probs = pipeline.predict_proba([text])[0]
    pred  = 'Anxious' if probs[1] >= 0.5 else 'Not Anxious'
    match = 'OK' if pred == expected else f'!! (expected {expected})'
    print(f"  [{match}] {text!r}")
    print(f"       -> {pred}  (Anxious: {probs[1]:.0%})")

# ── Save candidate (do NOT overwrite yet) ─────────────────────────────────────

le = LabelEncoder()
le.fit([0, 1])

tmp_path = os.path.join(MODEL_DIR, 'model_anxiety_candidate.pkl')
with open(tmp_path, 'wb') as f:
    pickle.dump({'pipeline': pipeline, 'label_encoder': le}, f)
print(f"\nCandidate saved -> {tmp_path}")
print("model_anxiety.pkl NOT replaced. Confirm to apply.")
