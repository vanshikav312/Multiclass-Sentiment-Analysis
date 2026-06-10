import sys
sys.path.insert(0, 'D:\\PythonPackages')

import streamlit as st
import pickle, os, re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'notebook')

st.set_page_config(
    page_title="MindScan — Mental Health Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05) !important;
    border-right: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}

[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}

.metric-label {
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #a0a0c0;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
}

.metric-sub {
    font-size: 12px;
    color: #7878a0;
    margin-top: 4px;
}

/* Title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

.main-subtitle {
    font-size: 15px;
    color: #8888aa;
    margin-bottom: 32px;
}

/* Score bars */
.score-row {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    gap: 12px;
}
.score-label {
    width: 130px;
    font-size: 13px;
    font-weight: 500;
    color: #c0c0e0;
    text-transform: capitalize;
}
.score-bar-bg {
    flex: 1;
    background: rgba(255,255,255,0.08);
    border-radius: 100px;
    height: 10px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.4s ease;
}
.score-pct {
    width: 42px;
    font-size: 13px;
    font-weight: 600;
    color: #e0e0ff;
    text-align: right;
}

/* Section headers */
.section-header {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #7878aa;
    margin-bottom: 14px;
    margin-top: 8px;
}

/* Top prediction badge */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 13px;
    font-weight: 600;
    margin-left: 8px;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 24px 0;
}

/* Text area */
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e0e0e0 !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #a78bfa, #60a5fa) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.3px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Dataframe */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    def load(name):
        with open(os.path.join(MODEL_DIR, name), 'rb') as f:
            return pickle.load(f)
    return {
        'emotion':    load('model_emotion.pkl'),
        'depression': load('model_depression.pkl'),
        'anxiety':    load('model_anxiety.pkl'),
        'eating':     load('model_eating.pkl'),
    }

def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict(text, models):
    t = clean(text)
    results = {}
    m = models['emotion']
    probs = m['pipeline'].predict_proba([t])[0]
    results['emotion'] = dict(zip(m['label_encoder'].classes_, probs))
    m = models['depression']
    probs = m['pipeline'].predict_proba([t])[0]
    results['depression'] = {'Not Depressed': probs[0], 'Depressed': probs[1]}
    m = models['anxiety']
    probs = m['pipeline'].predict_proba([t])[0]
    results['anxiety'] = {'Not Anxious': probs[0], 'Anxious': probs[1]}
    m = models['eating']
    probs = m['pipeline'].predict_proba([t])[0]
    results['eating'] = dict(zip(m['label_encoder'].classes_, probs))
    return results

def score_bar(label, score, color):
    pct = int(score * 100)
    return f"""
    <div class="score-row">
        <div class="score-label">{label}</div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
        <div class="score-pct">{pct}%</div>
    </div>"""

def get_emotion_color(emotion):
    colors = {
        'joy': '#34d399', 'happy': '#34d399', 'happiness': '#34d399',
        'sad': '#60a5fa', 'sadness': '#60a5fa',
        'anger': '#f87171', 'angry': '#f87171',
        'fear': '#fbbf24', 'fearful': '#fbbf24',
        'surprise': '#a78bfa', 'surprised': '#a78bfa',
        'disgust': '#fb923c', 'love': '#f472b6',
        'emotional': '#e879f9', 'neutral': '#94a3b8',
    }
    return colors.get(emotion.lower(), '#a78bfa')

models = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 MindScan")
    st.markdown("<div style='color:#7878aa;font-size:13px;margin-bottom:24px;'>Mental Health Sentiment Analyzer</div>", unsafe_allow_html=True)
    page = st.radio("", ["🔍  Analyze Text", "📊  Evaluation Metrics"], label_visibility="collapsed")
    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:24px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#7878aa;font-size:12px;'>Powered by ML models trained on 5 mental health datasets</div>", unsafe_allow_html=True)

# ── Page 1: Analyze ───────────────────────────────────────────────────────────
if page == "🔍  Analyze Text":
    st.markdown('<div class="main-title">MindScan</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Enter any text to get a real-time mental health sentiment analysis across 4 dimensions.</div>', unsafe_allow_html=True)

    text = st.text_area("", placeholder="e.g. I've been feeling really overwhelmed lately and can't seem to focus on anything...", height=140, label_visibility="collapsed")

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        analyze = st.button("Analyze →", type="primary")

    if analyze:
        if not text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                results = predict(text, models)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # ── Emotion ──────────────────────────────────────────────────────
            emo = results['emotion']
            top_emo = max(emo, key=emo.get)
            top_color = get_emotion_color(top_emo)
            sorted_emo = sorted(emo.items(), key=lambda x: x[1], reverse=True)

            st.markdown(f"""
            <div class='card'>
                <div class='section-header'>Emotion Detection</div>
                <div style='font-size:26px;font-weight:700;color:#fff;margin-bottom:4px;'>
                    {top_emo.title()}
                    <span style='font-size:14px;font-weight:500;color:{top_color};background:rgba(167,139,250,0.12);padding:3px 12px;border-radius:100px;margin-left:8px;'>
                        {emo[top_emo]:.0%} confidence
                    </span>
                </div>
                <div style='margin-top:18px;'>
                    {''.join(score_bar(e.title(), s, get_emotion_color(e)) for e, s in sorted_emo)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── 3 columns ────────────────────────────────────────────────────
            col1, col2, col3 = st.columns(3)

            with col1:
                dep = results['depression']
                top_dep = max(dep, key=dep.get)
                dep_color = '#f87171' if top_dep == 'Depressed' else '#34d399'
                st.markdown(f"""
                <div class='card'>
                    <div class='section-header'>Depression</div>
                    <div style='font-size:22px;font-weight:700;color:{dep_color};margin-bottom:16px;'>{top_dep}</div>
                    {''.join(score_bar(k, v, '#f87171' if k=='Depressed' else '#34d399') for k,v in dep.items())}
                </div>
                """, unsafe_allow_html=True)

            with col2:
                anx = results['anxiety']
                top_anx = max(anx, key=anx.get)
                anx_color = '#fbbf24' if top_anx == 'Anxious' else '#34d399'
                st.markdown(f"""
                <div class='card'>
                    <div class='section-header'>Anxiety</div>
                    <div style='font-size:22px;font-weight:700;color:{anx_color};margin-bottom:16px;'>{top_anx}</div>
                    {''.join(score_bar(k, v, '#fbbf24' if k=='Anxious' else '#34d399') for k,v in anx.items())}
                </div>
                """, unsafe_allow_html=True)

            with col3:
                eat = results['eating']
                top_eat = max(eat, key=eat.get)
                sorted_eat = sorted(eat.items(), key=lambda x: x[1], reverse=True)
                eat_colors = ['#a78bfa', '#60a5fa', '#34d399', '#fbbf24', '#f87171']
                st.markdown(f"""
                <div class='card'>
                    <div class='section-header'>Eating Behavior</div>
                    <div style='font-size:22px;font-weight:700;color:#a78bfa;margin-bottom:16px;'>{top_eat}</div>
                    {''.join(score_bar(k, v, eat_colors[i % len(eat_colors)]) for i, (k,v) in enumerate(sorted_eat))}
                </div>
                """, unsafe_allow_html=True)

# ── Page 2: Metrics ───────────────────────────────────────────────────────────
elif page == "📊  Evaluation Metrics":
    st.markdown('<div class="main-title">Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Model performance across all 5 mental health classifiers.</div>', unsafe_allow_html=True)

    metrics_path = os.path.join(MODEL_DIR, 'model_performance_summary.csv')
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)

        # Summary metric cards
        cols = st.columns(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            with cols[i]:
                acc = float(row['Test Accuracy'])
                color = '#34d399' if acc >= 0.9 else '#fbbf24' if acc >= 0.7 else '#f87171'
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{row['Model']}</div>
                    <div class='metric-value' style='color:{color};'>{acc:.1%}</div>
                    <div class='metric-sub'>F1: {float(row['F1 Score (macro)']):.1%}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Chart
        st.markdown("<div class='section-header'>Accuracy vs F1 Score</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        x = range(len(df))
        w = 0.35
        acc_colors = ['#34d399' if v >= 0.9 else '#fbbf24' if v >= 0.7 else '#f87171' for v in df['Test Accuracy']]
        ax.bar([i - w/2 for i in x], df['Test Accuracy'], w, label='Accuracy', color='#a78bfa', alpha=0.9)
        ax.bar([i + w/2 for i in x], df['F1 Score (macro)'], w, label='F1 Score', color='#60a5fa', alpha=0.9)
        ax.set_xticks(list(x))
        ax.set_xticklabels(df['Model'], color='#c0c0e0', fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score', color='#8888aa')
        ax.tick_params(colors='#8888aa')
        for spine in ax.spines.values():
            spine.set_color('rgba(255,255,255,0.1)')
        ax.yaxis.grid(True, color='rgba(255,255,255,0.05)', linestyle='--')
        ax.set_axisbelow(True)
        for i, v in enumerate(df['Test Accuracy']):
            ax.text(i - w/2, v + 0.02, f'{v:.0%}', ha='center', color='#e0e0ff', fontsize=10, fontweight='600')
        for i, v in enumerate(df['F1 Score (macro)']):
            ax.text(i + w/2, v + 0.02, f'{v:.0%}', ha='center', color='#e0e0ff', fontsize=10, fontweight='600')
        ax.legend(facecolor='#1a1a2e', labelcolor='#c0c0e0', edgecolor='rgba(255,255,255,0.1)')
        st.pyplot(fig)
        plt.close()

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Full table
        st.markdown("<div class='section-header'>Full Performance Table</div>", unsafe_allow_html=True)
        st.dataframe(
            df.style.format({
                'Test Accuracy': '{:.2%}',
                'F1 Score (macro)': '{:.2%}',
                '5-Fold CV Mean': '{:.2%}',
            }).background_gradient(subset=['Test Accuracy'], cmap='RdYlGn'),
            use_container_width=True
        )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Saved plots
    for title, fname in [
        ("Confusion Matrices", "confusion_matrices_all.png"),
        ("ROC AUC Curves", "roc_auc_combined.png"),
        ("Class Distribution", "class_distribution.png"),
        ("Prediction Dashboard", "prediction_dashboard.png"),
    ]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
            st.image(path, use_container_width=True)
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
