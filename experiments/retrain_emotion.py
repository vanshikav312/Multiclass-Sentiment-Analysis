"""
Retrain the emotion model with TF-IDF + VADER features combined.
Inputs:  notebook/emotion_preprocessed.csv  (columns: text, label, clean_text)
Output:  notebook/model_emotion.pkl  (overwrites existing)

Pipeline: raw text → FeatureUnion(TF-IDF on preprocessed, VADER on raw) → LogisticRegression
"""

import os, pickle
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from preprocessing import preprocess_text, PreprocessedTfidf, VaderTransformer

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'notebook')


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading emotion dataset...")
df = pd.read_csv(os.path.join(MODEL_DIR, 'emotion_preprocessed.csv'))
print(f"  {len(df)} rows, label distribution:")
print(df['label'].value_counts().to_string())

X = df['text'].fillna('').tolist()          # raw text — VADER uses this
y_raw = df['label'].tolist()

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}")


# ── Build pipeline ─────────────────────────────────────────────────────────────

print("\nBuilding pipeline...")
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', PreprocessedTfidf(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
        )),
        ('vader', VaderTransformer()),
    ])),
    ('clf', LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='lbfgs',
    )),
])


# ── Train ─────────────────────────────────────────────────────────────────────

print("Training...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# ── Quick sanity test ─────────────────────────────────────────────────────────

print("\nSanity test (raw text in):")
probes = [
    "I am very happy today",
    "I feel so sad and empty inside",
    "I am extremely anxious about everything",
    "I am so angry at everything",
    "I love spending time with my friends",
    "Life feels meaningless",
    "I am doing great today",
    "I have been feeling really down lately",
    "I feel overwhelmed and exhausted all the time",
    "I feel like I cannot breathe",
]
for text in probes:
    probs = pipeline.predict_proba([text])[0]
    top_idx = probs.argmax()
    print(f"  {text!r}")
    print(f"    -> {le.classes_[top_idx]} ({probs[top_idx]:.0%})")


# ── Save ──────────────────────────────────────────────────────────────────────

out_path = os.path.join(MODEL_DIR, 'model_emotion.pkl')
with open(out_path, 'wb') as f:
    pickle.dump({'pipeline': pipeline, 'label_encoder': le}, f)
print(f"\nSaved -> {out_path}")
