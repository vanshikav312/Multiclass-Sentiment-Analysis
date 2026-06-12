# Experiments

---

## Experiment 1 — GoEmotions as emotion model replacement (rejected)

Tried swapping the tweet-trained emotion model for one trained on GoEmotions (Google, Reddit-sourced, 28 fine-grained classes mapped to the same 6). The tweet model mislabels "overwhelmed/exhausted" as `surprise` because that's how the original Twitter dataset annotated those states — I wanted to see if a different training source fixed it.

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Original (tweet, TF-IDF + LR) | 94.44% | 93.81% |
| Current (TF-IDF + VADER + LR) | 91.53% | 91.52% |
| GoEmotions (TF-IDF + LinearSVC) | 68.51% | — |

GoEmotions drops to 68.5% after filtering to single-label rows and mapping 28→6 classes. The training set ends up too small and imbalanced. The "overwhelmed → surprise" issue is a labelling artefact in the source data, not an architecture problem — no replacement dataset fixes it without redefining the 6-class taxonomy. Kept current model.

---

## Experiment 2 — Anxiety model retraining (applied)

The original `model_anxiety.pkl` was trained on a dataset that only had positive examples — every row was labelled anxious. It was predicting "Anxious" for almost everything and hitting 96.58% because there was nothing to get wrong. That number was measuring recall, not classification.

Fixed by building a balanced dataset: 3,713 anxious posts paired with 3,713 non-distressed posts pulled from the depression dataset (`label = 0`). Retrained with TF-IDF + LinearSVC (balanced class weights).

| Model | Test Accuracy | Macro F1 | Notes |
|---|---|---|---|
| Old (no negatives) | 96.58% | 96.58% | Recall-only, predicted Anxious for everything |
| New (balanced) | 97.78% | 97.78% | Actual binary classification |

Old model archived at `model_anxiety_old.pkl`. The main remaining gap is implicit anxiety vocabulary — "terrified", "dreading", "nervous" without the word "anxious" still gets missed. That's a training data coverage issue, not something the architecture fixes easily.

---

## Experiment 3 — Stress model diagnostic (negative result, model retired)

The stress model (RandomForest, 17 sensor + behavioral features, 100 participants × 30 days) reported 32.83% test accuracy on a 3-class problem where random baseline is 33.3% — essentially chance. Before retiring it I ran a diagnostic to check if it was a modelling problem or a data problem.

The issue is split validity. With 30 rows per participant, a random train/test split puts rows from the same person on both sides. The model can look like it's learning by recognizing individual-level sensor patterns, not the actual stress signal. A participant-level split (GroupKFold by participant_id) removes that shortcut.

**3-class (Low / Moderate / High), 5-fold GroupKFold:**

| Model | Acc | ± | Macro F1 | ± |
|---|---|---|---|---|
| Majority-class baseline | 0.348 | 0.013 | 0.172 | 0.005 |
| RandomForest (200, depth=10) | 0.344 | 0.022 | 0.343 | 0.022 |
| RandomForest (500, full depth) | 0.341 | 0.017 | 0.340 | 0.017 |
| GradientBoosting (200) | 0.330 | 0.022 | 0.328 | 0.021 |
| XGBoost (300) | 0.343 | 0.025 | 0.341 | 0.024 |
| LightGBM (300) | 0.351 | 0.018 | 0.349 | 0.017 |
| LightGBM (tuned) | 0.335 | 0.013 | 0.333 | 0.012 |
| Chance (uniform) | 0.333 | | | |

**Binary (High vs Low only), 5-fold GroupKFold:**

| Model | Acc | ± | Macro F1 | ± |
|---|---|---|---|---|
| Majority-class baseline | 0.519 | 0.034 | 0.341 | 0.015 |
| RandomForest (500, full depth) | 0.505 | 0.016 | 0.503 | 0.015 |
| LightGBM (tuned) | 0.492 | 0.016 | 0.490 | 0.015 |
| Chance (uniform) | 0.500 | | | |

Every model family — across both 3-class and binary formulations — sits at chance once the split is honest. The per-fold variance (e.g. LightGBM 3-class: .325–.377) shows the models were picking up participant-specific noise, not a generalizable signal. Same failure mode as training a speaker-classification model with random clip-level splits instead of speaker-disjoint splits: the model memorizes identities, not the task.

Model removed from the dashboard. Artifacts archived here.
