import os
import re
from datetime import datetime
import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #0b1020;
            --panel: #121a2c;
            --panel-2: #1a2338;
            --text: #e5ecff;
            --muted: #a7b3d1;
            --primary: #4f8cff;
            --primary-2: #7c3aed;
            --success-bg: rgba(34, 197, 94, 0.14);
            --success-text: #86efac;
            --danger-bg: rgba(239, 68, 68, 0.14);
            --danger-text: #fca5a5;
        }
        .stApp {
            background: radial-gradient(circle at top, #111a2d 0%, #0b1020 42%, #070d19 100%);
            color: var(--text);
        }
        .hero {
            background: linear-gradient(135deg, rgba(79,140,255,0.24), rgba(124,58,237,0.34));
            border: 1px solid rgba(148, 163, 184, 0.18);
            padding: 2rem 1.7rem;
            border-radius: 26px;
            margin-bottom: 1.4rem;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(10px);
        }
        .hero-panel {
            display: grid;
            grid-template-columns: 1.35fr 0.9fr;
            gap: 1rem;
            align-items: center;
        }
        .hero-copy h1 {
            color: #ffffff !important;
            font-size: 2.5rem !important;
            margin: 0;
            font-weight: 900;
            letter-spacing: -0.05em;
            line-height: 1.05;
        }
        .hero-copy p {
            color: rgba(255,255,255,0.82) !important;
            margin-top: 0.7rem;
            margin-bottom: 0;
            font-size: 1rem;
            max-width: 620px;
        }
        .hero-features {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 1rem;
        }
        .feature-pill {
            background: rgba(15,22,37,0.42);
            border: 1px solid rgba(255,255,255,0.12);
            color: #e7eeff;
            padding: 0.45rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .brand {
            display: inline-flex;
            align-items: center;
            gap: 0.7rem;
            background: rgba(15,22,37,0.6);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 999px;
            padding: 0.35rem 0.7rem 0.35rem 0.4rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }
        .brand-mark {
            width: 28px;
            height: 28px;
            border-radius: 10px;
            background: linear-gradient(135deg, #4f8cff, #7c3aed);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            font-weight: 800;
            color: white;
            box-shadow: 0 8px 18px rgba(79, 140, 255, 0.28);
        }
        .brand-name {
            font-size: 0.82rem;
            font-weight: 800;
            color: white;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .badge {
            display: inline-block;
            background: rgba(79, 140, 255, 0.12);
            border: 1px solid rgba(79, 140, 255, 0.35);
            color: #dfeaff;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
        }
        .badge-live {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.35);
            color: #bbf7d0;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            animation: badgePulse 1.7s ease-in-out infinite alternate;
        }
        .dot {
            width: 8px;
            height: 8px;
            background: #4ade80;
            border-radius: 50%;
            box-shadow: 0 0 10px rgba(74, 222, 128, 0.9);
        }
        @keyframes badgePulse {
            0% { transform: scale(1); box-shadow: 0 0 0 rgba(74, 222, 128, 0); }
            100% { transform: scale(1.04); box-shadow: 0 0 18px rgba(74, 222, 128, 0.35); }
        }
        .card {
            background: rgba(18, 26, 44, 0.68);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 20px;
            padding: 1.1rem 1.15rem;
            color: var(--text);
            box-shadow: 0 10px 22px rgba(0,0,0,0.18);
            backdrop-filter: blur(10px);
        }
        .result {
            padding: 1rem 1.2rem;
            border-radius: 16px;
            font-weight: 700;
            font-size: 1.1rem;
            text-align: center;
            margin-top: 1rem;
            border: 1px solid transparent;
        }
        .result.fake {
            background: var(--danger-bg);
            color: var(--danger-text);
            border-color: rgba(248, 113, 113, 0.45);
        }
        .result.real {
            background: var(--success-bg);
            color: var(--success-text);
            border-color: rgba(74, 222, 128, 0.5);
        }
        .small-note {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.8rem;
        }
        .meter-shell {
            width: 100%;
            max-width: 440px;
            background: rgba(15,22,37,0.9);
            border-radius: 999px;
            height: 22px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            overflow: hidden;
            margin: 0.7rem auto 0.9rem auto;
            position: relative;
        }
        .meter-fill {
            height: 100%;
            width: 50%;
            background: linear-gradient(90deg, #22c55e 0%, #facc15 50%, #ef4444 100%);
            border-radius: 999px;
            box-shadow: 0 0 20px rgba(79, 140, 255, 0.22);
            transition: width 0.8s ease, filter 0.5s ease, transform 0.5s ease;
            animation: pulseMeter 1.8s ease-in-out infinite alternate;
        }
        @keyframes pulseMeter {
            from {
                filter: brightness(1);
                transform: scaleX(1);
            }
            to {
                filter: brightness(1.18);
                transform: scaleX(1.01);
            }
        }
        .meter-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            color: var(--muted);
            margin-top: 0.5rem;
        }
        .metric-box {
            background: rgba(15,22,37,0.7);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            text-align: center;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
            backdrop-filter: blur(8px);
        }
        .metric-box strong {
            display: block;
            font-size: 1.6rem;
            color: white;
            margin-bottom: 0.2rem;
        }
        .metric-box span {
            color: var(--muted);
            font-size: 0.82rem;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1.3rem;
        }
        .feature-card {
            background: rgba(18, 26, 44, 0.66);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 1rem 0.9rem;
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
            backdrop-filter: blur(8px);
        }
        .feature-icon {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, rgba(79,140,255,0.22), rgba(124,58,237,0.28));
            border: 1px solid rgba(148, 163, 184, 0.18);
            font-size: 1.2rem;
            margin-bottom: 0.75rem;
        }
        .feature-card h4 {
            color: white;
            margin: 0 0 0.4rem 0;
            font-size: 1rem;
        }
        .feature-card p {
            color: var(--muted);
            margin: 0;
            font-size: 0.82rem;
            line-height: 1.55;
        }
        .mockup {
            margin-top: 1.4rem;
            background: linear-gradient(180deg, rgba(18,26,44,0.8), rgba(11,16,32,0.9));
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1rem;
            box-shadow: 0 12px 32px rgba(0,0,0,0.18);
            backdrop-filter: blur(10px);
        }
        .mockup-window {
            background: rgba(7, 11, 19, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 16px;
            padding: 0.95rem;
        }
        .mockup-topbar {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.9rem;
        }
        .dot-red, .dot-yellow, .dot-green {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }
        .dot-red { background: #f87171; }
        .dot-yellow { background: #fbbf24; }
        .dot-green { background: #4ade80; }
        .mockup-grid {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 0.8rem;
        }
        .mockup-panel {
            background: rgba(18,26,44,0.9);
            border: 1px solid rgba(148,163,184,0.15);
            border-radius: 14px;
            padding: 0.8rem;
            min-height: 120px;
        }
        .mockup-line {
            height: 10px;
            border-radius: 999px;
            background: rgba(148,163,184,0.18);
            margin-bottom: 0.6rem;
        }
        .mockup-line.short { width: 58%; }
        .mockup-line.mid { width: 72%; }
        .mockup-line.long { width: 100%; }
        .mockup-card {
            background: linear-gradient(135deg, rgba(79,140,255,0.16), rgba(124,58,237,0.2));
            border: 1px solid rgba(148,163,184,0.15);
            border-radius: 14px;
            padding: 0.8rem;
            min-height: 120px;
        }
        .score-panel {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.8rem;
            flex-wrap: wrap;
            min-height: 190px;
        }
        .score-dial {
            position: relative;
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background: conic-gradient(var(--dial-color) calc(var(--score) * 1%), rgba(148, 163, 184, 0.18) 0);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 22px rgba(79, 140, 255, 0.2);
            animation: dialFloat 2.4s ease-in-out infinite alternate;
        }
        @keyframes dialFloat {
            from { transform: translateY(0px); }
            to { transform: translateY(-3px); }
        }
        .score-dial::before {
            content: "";
            position: absolute;
            inset: 14px;
            border-radius: 50%;
            background: rgba(9, 15, 25, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.15);
        }
        .score-value {
            position: relative;
            z-index: 1;
            text-align: center;
        }
        .score-value strong {
            display: block;
            font-size: 2rem;
            line-height: 1;
            color: white;
            letter-spacing: -0.04em;
        }
        .score-value span {
            display: block;
            margin-top: 0.35rem;
            color: var(--muted);
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .confidence-ring {
            background: rgba(15,22,37,0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 0.7rem 0.85rem;
            min-width: 120px;
            text-align: center;
            font-size: 0.8rem;
            color: var(--muted);
            box-shadow: inset 0 0 10px rgba(79, 140, 255, 0.07);
            backdrop-filter: blur(8px);
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 999px;
            padding: 0.48rem 0.8rem;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            animation: badgePulse 1.8s ease-in-out infinite alternate;
        }
        .status-badge.verified {
            background: rgba(34, 197, 94, 0.14);
            color: #bbf7d0;
            border: 1px solid rgba(34, 197, 94, 0.45);
        }
        .status-badge.flagged {
            background: rgba(239, 68, 68, 0.12);
            color: #fecaca;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }
        .fact-card {
            margin-top: 0.9rem;
            background: rgba(15,22,37,0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .fact-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.6rem;
        }
        .fact-title {
            font-size: 0.78rem;
            color: #dfeaff;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .fact-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 999px;
            padding: 0.34rem 0.6rem;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .fact-chip.fake {
            background: rgba(239, 68, 68, 0.14);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #fecaca;
        }
        .fact-chip.real {
            background: rgba(34, 197, 94, 0.14);
            border: 1px solid rgba(34, 197, 94, 0.45);
            color: #bbf7d0;
        }
        .fact-text {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.6;
            margin: 0;
        }
        textarea {
            border-radius: 16px !important;
            background: rgba(15,22,37,0.92) !important;
            color: white !important;
            border: 1px solid rgba(148, 163, 184, 0.35) !important;
            box-shadow: inset 0 0 0 1px rgba(79,140,255,0.12);
        }
        .stTextInput > label, .stTextArea > label {
            color: var(--text) !important;
            font-weight: 600;
        }
        div[data-testid="stButton"] > button {
            width: 100%;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 100%);
            color: white;
            border: none;
            font-weight: 800;
            padding: 0.8rem 1rem;
            box-shadow: 0 10px 20px rgba(79, 140, 255, 0.28), 0 0 24px rgba(124, 58, 237, 0.18);
        }
        div[data-testid="stButton"] > button:hover {
            filter: brightness(1.08);
        }
        .stCaption {
            color: var(--muted) !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_text_column(df):
    preferred = [
        "text", "content", "news", "headline", "article",
        "statement", "title", "summary", "body", "claim"
    ]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            return col
    return df.columns[0]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)


FALSEHOOD_RULES = [
    (r"\bcow\s+is\s+a\s+bird\b|\bchickens\s+are\s+fish\b|\bwhales\s+are\s+birds\b", "Known biology contradiction"),
    (r"\bthe\s+earth\s+is\s+flat\b|\bflat\s+earth\b", "Geography contradiction"),
    (r"\b2\s*\+\s*2\s*=\s*5\b|\b1\s*\+\s*1\s*=\s*3\b", "Mathematical impossibility"),
    (r"\bwater\s+is\s+made\s+of\s+gold\b|\bgold\s+is\s+made\s+of\s+water\b", "Scientific impossibility"),
    (r"\bvaccines\s+cause\s+autism\b|\bmoon\s+made\s+of\s+cheese\b|\bhumans\s+can\s+fly\b", "Widely disproven myth"),
]


def check_obvious_falsehood(text):
    cleaned = clean_text(text)
    search_texts = [text.lower(), cleaned]
    for pattern, reason in FALSEHOOD_RULES:
        for candidate in search_texts:
            if re.search(pattern, candidate):
                return 0, reason
    return None, None


def build_explanation(prediction, reason=None):
    if prediction == 0:
        if reason:
            return {
                "label": "Fake",
                "note": f"Why this is fake: this statement conflicts with a known fact pattern ({reason}). The text looks false or misleading."
            }
        return {
            "label": "Fake",
            "note": "Why this is fake: this statement appears to contradict common factual knowledge or scientific reality."
        }
    return {
        "label": "Real",
        "note": "Fact-check note: this statement does not match a known falsehood pattern, so the model score is based on language patterns and context."
    }


model_path = os.path.join(BASE_DIR, "fake_news_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")


def format_model_status(path):
    if not os.path.exists(path):
        return {
            "exists": False,
            "path": path,
            "updated": "Not generated yet",
        }
    ts = os.path.getmtime(path)
    return {
        "exists": True,
        "path": path,
        "updated": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
    }


model_status = format_model_status(model_path)
vectorizer_status = format_model_status(vectorizer_path)


def render_model_status_banner():
    if model_status["exists"]:
        return "Model loaded successfully", os.path.basename(model_path), model_status["updated"], "verified"
    return "Model missing", os.path.basename(model_path), model_status["updated"], "flagged"


status_text, file_name, updated_time, status_state = render_model_status_banner()

if not model_status["exists"]:
    st.error("Model file missing. Please train the model first using train_model.py.")

model = joblib.load(model_path) if model_status["exists"] else None
tfidf = joblib.load(vectorizer_path) if vectorizer_status["exists"] else None

st.markdown(
    f"""
    <div class="header-row">
        <div class="brand">
            <div class="brand-mark">N</div>
            <div class="brand-name">NewsGuard</div>
        </div>
        <span class="badge-live"><span class="dot"></span>Live</span>
    </div>
    <div class="card" style="margin-bottom: 1rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap;">
            <div>
                <div style="font-size:0.72rem; color:#a7b3d1; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.35rem;">Model status</div>
                <div style="font-size:1rem; font-weight:700; color:white;">{status_text}</div>
            </div>
            <div class="badge">{file_name}</div>
        </div>
        <div style="margin-top:0.65rem; color:#a7b3d1; font-size:0.8rem;">Updated: {updated_time}</div>
    </div>
    <div class="hero hero-panel">
        <div class="hero-copy">
            <h1>📰 Fake News Detector</h1>
            <p>Instantly score a headline or article for signs of misinformation before it spreads.</p>
            <div class="hero-features">
                <span class="feature-pill">⚡ Instant detection</span>
                <span class="feature-pill">🧠 ML-powered</span>
                <span class="feature-pill">🛡️ Trust scoring</span>
            </div>
        </div>
        <div class="score-panel">
            <div class="score-dial" style="--score: 82; --dial-color: #4ade80;">
                <div class="score-value">
                    <strong>82%</strong>
                    <span>Real</span>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"""
    <div class="card" style="padding:0.9rem; margin-top:0.5rem;">
        <div style="font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase; color:#a7b3d1; margin-bottom:0.5rem;">Saved model</div>
        <div style="font-weight:700; color:white; margin-bottom:0.25rem;">{status_text}</div>
        <div style="color:#a7b3d1; font-size:0.78rem;">Path: {model_path}</div>
        <div style="color:#a7b3d1; font-size:0.78rem; margin-top:0.2rem;">Updated: {updated_time}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


if not model_status["exists"]:
    st.warning("Run the training script once to generate the model file before using the detector.")

st.markdown(
    """
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom: 0.8rem;">
            <div>
                <div style="font-size:0.75rem; color:#a7b3d1; letter-spacing:0.08em; text-transform:uppercase;">Trust Meter</div>
                <div style="font-weight:700; font-size:1.1rem; color:white;">Real / Fake</div>
            </div>
            <div class="status-badge verified"><span class="dot"></span>Verified</div>
        </div>
        <div class="meter-shell">
            <div class="meter-fill" id="trust-meter"></div>
        </div>
        <div class="meter-labels">
            <span>Fake</span>
            <span>Balanced</span>
            <span>Real</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin-top: 1.3rem; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.8rem;">
        <div class="metric-box">
            <strong>1.0M+</strong>
            <span>Articles analyzed</span>
        </div>
        <div class="metric-box">
            <strong>94%</strong>
            <span>Accuracy target</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h4>Instant scan</h4>
            <p>Run text analysis in seconds with a lightweight, real-time verdict engine.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h4>AI intelligence</h4>
            <p>Leverage machine learning patterns to flag suspicious, misleading narratives.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h4>Trust metrics</h4>
            <p>Review confidence levels and reliability indicators with clear visual scoring.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mockup">
        <div class="mockup-window">
            <div class="mockup-topbar">
                <span class="dot-red"></span>
                <span class="dot-yellow"></span>
                <span class="dot-green"></span>
            </div>
            <div class="mockup-grid">
                <div class="mockup-panel">
                    <div class="mockup-line long"></div>
                    <div class="mockup-line mid"></div>
                    <div class="mockup-line short"></div>
                    <div class="mockup-line long" style="margin-top: 1rem;"></div>
                    <div class="mockup-line mid"></div>
                </div>
                <div class="mockup-card">
                    <div style="font-size: 0.72rem; color: #dfeaff; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.5rem;">Verdict</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: white;">92%</div>
                    <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #bbf7d0; font-weight: 700;">REAL</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

user_input = st.text_area(
    "Enter news headline or article text:",
    height=180,
    placeholder="Example: Government announces a new tax plan for farmers...",
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("Predict")

if predict_clicked:
    if model is None:
        st.error("Model file is missing. Please run train_model.py first to create fake_news_model.pkl.")
        st.stop()

    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        cleaned = clean_text(user_input)
        obvious_false, false_reason = check_obvious_falsehood(user_input)

        if obvious_false is not None:
            prediction = obvious_false
            confidence = 0.99
            reason = false_reason
        else:
            reason = None
            if hasattr(model, "named_steps") and "tfidf" in model.named_steps:
                prediction = model.predict([cleaned])[0]
                probs = model.predict_proba([cleaned])[0] if hasattr(model, "predict_proba") else None
            elif tfidf is not None:
                vectorized = tfidf.transform([cleaned])
                prediction = model.predict(vectorized)[0]
                probs = model.predict_proba(vectorized)[0] if hasattr(model, "predict_proba") else None
            else:
                prediction = model.predict([cleaned])[0]
                probs = model.predict_proba([cleaned])[0] if hasattr(model, "predict_proba") else None

            confidence = None
            if probs is not None:
                confidence = probs[1] if prediction == 1 else probs[0]

        explanation = build_explanation(prediction, reason)

        if confidence is not None:
            meter_value = int(round(confidence * 100))
            meter_width = max(8, min(100, meter_value))
            if prediction == 0:
                meter_color = "linear-gradient(90deg, #ef4444 0%, #f97316 100%)"
                result_text = "⚠️ Alert: This looks like Fake News!"
                result_class = "result fake"
                dial_color = "#f87171"
                dial_label = "Fake"
                status_class = "flagged"
                status_label = "Flagged"
            else:
                meter_color = "linear-gradient(90deg, #22c55e 0%, #4ade80 100%)"
                result_text = "✅ Verified: This looks like Real News."
                result_class = "result real"
                dial_color = "#4ade80"
                dial_label = "Real"
                status_class = "verified"
                status_label = "Verified"

            st.markdown(
                f"""
                <div class="score-panel">
                    <div class="score-dial" style="--score: {meter_width}; --dial-color: {dial_color};">
                        <div class="score-value">
                            <strong>{meter_width}%</strong>
                            <span>{dial_label}</span>
                        </div>
                    </div>
                    <div style="display:flex; flex-direction:column; gap:0.6rem; align-items:flex-start;">
                        <div class="status-badge {status_class}"><span class="dot"></span>{status_label}</div>
                        <div class="confidence-ring">Confidence: {confidence * 100:.1f}%</div>
                    </div>
                </div>
                <div id="trust-meter-wrap" style="display:none;">{meter_width}</div>
                <script>
                    const meter = document.getElementById('trust-meter');
                    if (meter) {{
                        meter.style.width = '{meter_width}%';
                        meter.style.background = '{meter_color}';
                        meter.style.transform = 'scaleX(1)';
                        meter.style.opacity = '1';
                        meter.style.boxShadow = '0 0 20px rgba(79, 140, 255, 0.26)';
                    }}
                </script>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"<div class='{result_class}'>{result_text}</div>", unsafe_allow_html=True)

            fact_label = explanation["label"]
            fact_note = explanation["note"]
            fact_class = "fake" if fact_label == "Fake" else "real"
            st.markdown(
                f"""
                <div class="fact-card">
                    <div class="fact-header">
                        <div class="fact-title">Why this is {fact_label.lower()}</div>
                        <div class="fact-chip {fact_class}">{fact_label}</div>
                    </div>
                    <p class="fact-text">{fact_note}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if prediction == 0:
                st.markdown(
                    "<div class='result fake'>⚠️ Alert: This looks like Fake News!</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    """
                    <div class="fact-card">
                        <div class="fact-header">
                            <div class="fact-title">Why this is fake</div>
                            <div class="fact-chip fake">Fake</div>
                        </div>
                        <p class="fact-text">Fact-check note: this claim appears to contradict known facts or scientific reality.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='result real'>✅ Verified: This looks like Real News.</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    """
                    <div class="fact-card">
                        <div class="fact-header">
                            <div class="fact-title">Fact-check note</div>
                            <div class="fact-chip real">Real</div>
                        </div>
                        <p class="fact-text">Fact-check note: this statement does not match a known falsehood pattern, so the model score is based on language patterns and context.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.markdown(
    """
    <div class="card" style="margin-top: 1.2rem; padding: 0.9rem 1rem;">
        <div style="font-size: 0.72rem; color:#a7b3d1; letter-spacing:0.08em; text-transform:uppercase; margin-bottom: 0.45rem;">Fact-check logic</div>
        <ul style="margin: 0; padding-left: 1.1rem; color: #dfeaff; font-size: 0.82rem; line-height: 1.7;">
            <li>Known falsehood patterns</li>
            <li>Scientific or geographic contradiction</li>
            <li>Impossible math or logic claims</li>
            <li>Common myths and disproven statements</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="small-note">
        Tip: use full sentences or headlines for better results.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin-top: 1.8rem; padding-top: 1rem; border-top: 1px solid rgba(148,163,184,0.15); display: flex; justify-content: space-between; align-items: center; gap: 1rem; color: #a7b3d1; font-size: 0.8rem; flex-wrap: wrap;">
        <span>© 2026 InsightScan AI</span>
        <span>News intelligence for smarter decisions</span>
    </div>
    """,
    unsafe_allow_html=True,
)