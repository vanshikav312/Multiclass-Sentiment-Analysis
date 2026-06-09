import pickle
import os
import re

# Resolve model directory relative to this script — no hardcoded paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'notebook')

def _load(name):
    with open(os.path.join(MODEL_DIR, name), 'rb') as f:
        return pickle.load(f)

# Load all 5 models
emo  = _load('model_emotion.pkl')
dep  = _load('model_depression.pkl')
anx  = _load('model_anxiety.pkl')
eat  = _load('model_eating.pkl')
stress = _load('model_stress.pkl')

def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict(text):
    t = clean(text)

    # Emotion
    emo_label = emo['label_encoder'].inverse_transform(emo['pipeline'].predict([t]))[0]
    emo_conf  = emo['pipeline'].predict_proba([t]).max()

    # Depression
    dep_pred  = dep['pipeline'].predict([t])[0]
    dep_conf  = dep['pipeline'].predict_proba([t]).max()
    dep_label = 'Depressed' if dep_pred == 1 else 'Not Depressed'

    # Anxiety
    anx_pred  = anx['pipeline'].predict([t])[0]
    anx_conf  = anx['pipeline'].predict_proba([t]).max()
    anx_label = 'Anxious' if anx_pred == 1 else 'Not Anxious'

    # Eating
    eat_label = eat['label_encoder'].inverse_transform(eat['pipeline'].predict([t]))[0]
    eat_conf  = eat['pipeline'].predict_proba([t]).max()

    # Stress (sensor-based model — note: lower accuracy ~33% as it uses physiological features)
    stress_label = 'N/A (sensor-based model — requires physiological data)'
    stress_conf  = None

    print("\n" + "="*60)
    print(f"  INPUT : {text[:80]}")
    print("="*60)
    print(f"  Emotion    : {emo_label:<22} ({emo_conf:.0%} confidence)")
    print(f"  Depression : {dep_label:<22} ({dep_conf:.0%} confidence)")
    print(f"  Anxiety    : {anx_label:<22} ({anx_conf:.0%} confidence)")
    print(f"  Eating     : {eat_label:<22} ({eat_conf:.0%} confidence)")
    print(f"  Stress     : {stress_label}")
    print("="*60 + "\n")

if __name__ == '__main__':
    print("Mental Health Sentiment Analyzer")
    print("Type your text below. Type 'quit' to exit.\n")
    while True:
        try:
            text = input("You: ").strip()
            if text.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            if text:
                predict(text)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
