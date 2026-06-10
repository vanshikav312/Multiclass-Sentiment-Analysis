import sys
import os
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
        # Handle non-standard characters like +/- sign
        df = df.replace({'±': '+/-'}, regex=True)
        # Convert df columns to lower-camel-case or friendly names
        records = df.to_dict(orient="records")
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
            "summary": "Captures standard human sentiment states (joy, sadness, anger, fear, surprise, disgust) from explicitly emotional training sets.",
            "why": "The underlying datasets consist of tweets or diary logs containing explicit emotional statements. Classifiers easily identify distinct terms (e.g. 'furious' for anger, 'scared' for fear, 'thrilled' for joy). The 94.4% accuracy is highly robust, though minor classification overlap occurs in highly nuanced text where emotions blend (e.g., fear and anger)."
        },
        "Eating Behavior": {
            "summary": "Distinguishes between selective eating disorders, healthy eating, and clinical eating concerns.",
            "why": "The model reaches 100% accuracy, which indicates the training data was highly structured, small, or template-based with very specific terminology (e.g., 'calorie count', 'binge', 'purge', 'healthy diet'). While perfect on the test split, it might show overfitting behaviors on real-world inputs that don't match these exact word lists."
        },
        "Anxiety": {
            "summary": "Identifies somatic panic and generalized anxiety markers.",
            "why": "Anxiety indicators are highly verbalized in peer-support forums. Frequent descriptions of physical stress (e.g. 'panic attack', 'chest tightens', 'shaking') and cognitive loops ('racing thoughts', 'constant worry') create prominent TF-IDF peaks. Consequently, the linear classifier achieves 96.6% accuracy."
        },
        "Stress (Sensor)": {
            "summary": "Attempts to classify physiological stress levels (3 classes) from physical sensors.",
            "why": "Sensor readings (like Heart Rate Variability, Galvanic Skin Response, and skin temperature) are highly noisy, subjective, and vary widely from person to person. Without personalized calibrations or sequential deep learning (like LSTMs), standard static classifiers achieve only 32.8% accuracy, which is close to a random 33.3% baseline."
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
