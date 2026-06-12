import sys
import os
import json
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import preprocess_text

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'notebook')

app = FastAPI(title="MindScan API", description="Mental Health Sentiment & Multi-class Emotion Analyzer API")

# Configure CORS for React frontend (default Vite dev server is on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load machine learning models
models = {}
try:
    with open(os.path.join(MODEL_DIR, 'model_emotion.pkl'), 'rb') as f:
        models['emotion'] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'model_depression.pkl'), 'rb') as f:
        models['depression'] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'model_anxiety.pkl'), 'rb') as f:
        models['anxiety'] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'model_eating.pkl'), 'rb') as f:
        models['eating'] = pickle.load(f)
    print("All ML models loaded successfully.")
except Exception as e:
    print(f"Error loading models from {MODEL_DIR}: {e}")

# Serves saved plots (confusion matrices, ROC curves, class distributions) as static files
if os.path.exists(MODEL_DIR):
    app.mount("/static", StaticFiles(directory=MODEL_DIR), name="static")

class AnalyzeRequest(BaseModel):
    text: str

def clean_text(text: str) -> str:
    return preprocess_text(text)

@app.post("/api/analyze")
async def analyze_text(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    cleaned = clean_text(request.text)  # for dep/anx/eating

    results = {}
    try:
        # 1. Emotion — new pipeline takes raw text (preprocessing + VADER inside)
        m_emo = models['emotion']
        probs_emo = m_emo['pipeline'].predict_proba([request.text])[0]
        results['emotion'] = dict(zip(m_emo['label_encoder'].classes_, [float(p) for p in probs_emo]))
        
        # 2. Depression (binary)
        m_dep = models['depression']
        probs_dep = m_dep['pipeline'].predict_proba([cleaned])[0]
        results['depression'] = {
            'Not Depressed': float(probs_dep[0]),
            'Depressed': float(probs_dep[1])
        }
        
        # 3. Anxiety (binary)
        m_anx = models['anxiety']
        probs_anx = m_anx['pipeline'].predict_proba([cleaned])[0]
        results['anxiety'] = {
            'Not Anxious': float(probs_anx[0]),
            'Anxious': float(probs_anx[1])
        }
        
        # 4. Eating Behavior (multiclass)
        m_eat = models['eating']
        probs_eat = m_eat['pipeline'].predict_proba([cleaned])[0]
        results['eating'] = dict(zip(m_eat['label_encoder'].classes_, [float(p) for p in probs_eat]))
        
        return {
            "text": request.text,
            "cleaned_text": cleaned,
            "results": results
        }
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Model loading or prediction format mismatch: missing key {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.get("/api/metrics")
async def get_metrics():
    csv_path = os.path.join(MODEL_DIR, 'model_performance_summary.csv')
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Metrics summary file not found")
    
    try:
        df = pd.read_csv(csv_path)
        raw_records = df.to_dict(orient='records')
        
        import math
        records = []
        for r in raw_records:
            clean_r = {}
            for k, v in r.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    clean_r[k] = None
                elif isinstance(v, str):
                    clean_r[k] = v.replace('Â±', '+/-').replace('Â+/-', '+/-').replace('±', '+/-')
                else:
                    clean_r[k] = v
            records.append(clean_r)
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading performance summary: {e}")

@app.get("/api/explain")
async def get_explain():
    return {
        "Depression": {
            "summary": "Reddit and Twitter mental health support subreddits have strong, distinct semantic features.",
            "why": "Clinical depression posts contain highly localized vocabulary like 'feeling hopeless', 'can't get out of bed', 'empty', or 'numb'. There is very little vocabulary overlap with standard control groups, allowing linear/logistic classifiers with TF-IDF representations to separate depressed from non-depressed users with high accuracy (96.5%)."
        },
        "Emotion": {
            "summary": "Captures six sentiment states (anger, fear, joy, love, sadness, surprise) from tweet-sourced training data.",
            "why": "Trained on ~90k balanced tweet samples across six classes: anger, fear, joy, love, sad, surprise. A TF-IDF + VADER feature union feeds a Logistic Regression classifier, achieving 91.5% test accuracy and 89.6% 5-fold CV accuracy. The gap between test and CV reflects slight distribution shift across folds. Note: the training data labels exhaustion and overwhelm as 'surprise' following Twitter annotation conventions — this is a dataset artefact, not a model error."
        },
        "Eating Behavior": {
            "summary": "Classifies eating-related text into five categories: normal, emotional, obesity, anxiety, stress.",
            "why": "The dataset is small (~1,004 rows after deduplication), perfectly balanced (200 per class), and highly template-like. Near-duplicate analysis shows 73% of test texts have cosine similarity > 0.8 to a training text, and 9.5% are exact matches. The 100% accuracy reflects dataset separability and near-duplicate leakage, not real-world robustness. 5-fold CV is also 100% ± 0.000, confirming the pattern is consistent but likely reflects memorization."
        },
        "Anxiety": {
            "summary": "Identifies somatic panic and generalized anxiety markers.",
            "why": "Anxiety indicators are highly verbalized in peer-support forums. Frequent descriptions of physical symptoms ('panic attack', 'heart races', 'can't breathe') and cognitive loops ('racing thoughts', 'constant worry') create strong TF-IDF signals. The model was trained on a balanced set of 3,713 anxious posts and 3,713 non-anxious examples approximated from non-distressed posts in a depression dataset. LinearSVC with balanced class weights achieves 97.8% test accuracy and 97.2% 5-fold CV macro F1."
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
