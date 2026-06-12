import pickle
import os

from preprocessing import preprocess_text

# Resolve model directory relative to this script — no hardcoded paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'notebook')

def _load(name):
    with open(os.path.join(MODEL_DIR, name), 'rb') as f:
        return pickle.load(f)

# Load text classifiers (stress model uses sensor data — not loaded here)
emo = _load('model_emotion.pkl')
dep = _load('model_depression.pkl')
anx = _load('model_anxiety.pkl')
eat = _load('model_eating.pkl')

def predict(text, silent=False):
    """Run all four text classifiers on *text*.

    Returns a dict with keys: emotion, depression, anxiety, eating.
    Each value is {'label': str, 'confidence': float}.
    Stress model requires physiological sensor data and is not text-inferrable.
    Pass silent=True to suppress printed output (used by eval harness).
    """
    t = preprocess_text(text)

    # Emotion — pipeline handles preprocessing + VADER internally; takes raw text
    emo_label = emo['label_encoder'].inverse_transform(emo['pipeline'].predict([text]))[0]
    emo_conf  = float(emo['pipeline'].predict_proba([text]).max())

    # Depression
    dep_pred  = dep['pipeline'].predict([t])[0]
    dep_conf  = float(dep['pipeline'].predict_proba([t]).max())
    dep_label = 'Depressed' if dep_pred == 1 else 'Not Depressed'

    # Anxiety
    anx_pred  = anx['pipeline'].predict([t])[0]
    anx_conf  = float(anx['pipeline'].predict_proba([t]).max())
    anx_label = 'Anxious' if anx_pred == 1 else 'Not Anxious'

    # Eating
    eat_label = eat['label_encoder'].inverse_transform(eat['pipeline'].predict([t]))[0]
    eat_conf  = float(eat['pipeline'].predict_proba([t]).max())

    result = {
        'emotion':    {'label': emo_label, 'confidence': emo_conf},
        'depression': {'label': dep_label, 'confidence': dep_conf},
        'anxiety':    {'label': anx_label, 'confidence': anx_conf},
        'eating':     {'label': eat_label, 'confidence': eat_conf},
    }

    if not silent:
        print("\n" + "="*60)
        print(f"  INPUT : {text[:80]}")
        print("="*60)
        print(f"  Emotion    : {emo_label:<22} ({emo_conf:.0%} confidence)")
        print(f"  Depression : {dep_label:<22} ({dep_conf:.0%} confidence)")
        print(f"  Anxiety    : {anx_label:<22} ({anx_conf:.0%} confidence)")
        print(f"  Eating     : {eat_label:<22} ({eat_conf:.0%} confidence)")
        print(f"  Stress     : not served (sensor model — see experiments/)")
        print("="*60 + "\n")

    return result

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
