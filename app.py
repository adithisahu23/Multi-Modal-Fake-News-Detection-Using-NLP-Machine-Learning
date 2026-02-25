"""
app.py
-------
Interactive Streamlit web UI for the Fake News Detector.
Run: streamlit run app.py
"""

import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import streamlit as st

from utils.preprocessing import (
    load_sample_data, tokens_to_string, build_tfidf_vectorizer
)
from utils.feature_engineering import extract_features

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🧠",
    layout="wide",
)

# ──────────────────────────────────────────────
# Styles
# ──────────────────────────────────────────────

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .fake-badge {
        background: #dc3545;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .real-badge {
        background: #28a745;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .feature-table td { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.markdown('<div class="main-title">🧠 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Multi-Modal NLP + Machine Learning System | '
    'Logistic Regression · Random Forest · SVM · Gradient Boosting</div>',
    unsafe_allow_html=True
)

# ──────────────────────────────────────────────
# Train model (cached)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="🤖 Training models on sample data...")
def train_model():
    from sklearn.ensemble import GradientBoostingClassifier

    df = load_sample_data()
    df["clean_text"] = df["full_text"].apply(tokens_to_string)
    feature_df = extract_features(df, text_col="full_text")

    X_text = df["clean_text"]
    X_feat = feature_df.values
    y = df["label"].values

    X_text_train, X_text_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
        X_text, X_feat, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = build_tfidf_vectorizer(max_features=5000)
    X_tfidf_train = vectorizer.fit_transform(X_text_train)
    X_tfidf_test = vectorizer.transform(X_text_test)

    scaler = StandardScaler()
    X_feat_train_scaled = scaler.fit_transform(X_feat_train)

    X_train = hstack([X_tfidf_train, csr_matrix(X_feat_train_scaled)])

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, vectorizer, scaler, feature_df.columns.tolist()


model, vectorizer, scaler, feat_names = train_model()


# ──────────────────────────────────────────────
# Predict function
# ──────────────────────────────────────────────

def predict_article(text: str):
    clean = tokens_to_string(text)
    tfidf = vectorizer.transform([clean])

    temp_df = pd.DataFrame({"full_text": [text]})
    feat = extract_features(temp_df, text_col="full_text")
    feat_scaled = scaler.transform(feat.values)

    X = hstack([tfidf, csr_matrix(feat_scaled)])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return int(pred), float(proba[1]), feat.iloc[0]


# ──────────────────────────────────────────────
# UI Layout
# ──────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍 Analyze Article", "📊 Model Info", "📖 How It Works"])

# ─── Tab 1: Analyze ───

with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Paste your article or headline")
        user_text = st.text_area(
            label="",
            height=220,
            placeholder="Paste a news article, headline, or paragraph here...\n\nExample:\n'BREAKING: Scientists ADMIT the earth is flat in leaked memo!'",
        )

        btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

    with col_right:
        st.subheader("Quick Examples")
        examples = {
            "🟢 Likely Real": "The Federal Reserve held interest rates steady on Wednesday, citing ongoing concerns about inflation and a still-resilient labor market.",
            "🔴 Likely Fake": "BREAKING EXCLUSIVE: Bill Gates ADMITS 5G towers are causing bird flu and the government is covering it up — leaked documents REVEALED!",
            "🟡 Ambiguous": "Scientists discover that drinking coffee might reduce the risk of certain diseases, according to a new study.",
        }
        for label, ex in examples.items():
            if st.button(label, use_container_width=True):
                user_text = ex
                btn = True

    if btn and user_text.strip():
        with st.spinner("Analyzing..."):
            pred, fake_prob, features = predict_article(user_text)

        real_prob = 1 - fake_prob
        label = "FAKE" if pred == 1 else "REAL"
        confidence = fake_prob if pred == 1 else real_prob

        st.divider()
        verdict_col, conf_col = st.columns(2)

        with verdict_col:
            st.subheader("Verdict")
            if pred == 1:
                st.markdown(f'<div class="fake-badge">⚠️ FAKE NEWS ({confidence*100:.1f}% confidence)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="real-badge">✅ REAL NEWS ({confidence*100:.1f}% confidence)</div>', unsafe_allow_html=True)

        with conf_col:
            st.subheader("Probability")
            st.progress(float(fake_prob), text=f"Fake probability: {fake_prob*100:.1f}%")
            st.progress(float(real_prob), text=f"Real probability: {real_prob*100:.1f}%")

        # Feature breakdown
        st.divider()
        st.subheader("📐 Feature Breakdown")
        feat_data = {
            "Feature": [
                "Text Length", "Word Count", "Avg Word Length",
                "Clickbait Score", "Hedge Words (Journalistic)", "Sentiment (Compound)",
                "CAPS Ratio", "Exclamation Marks", "Has Source Citation",
                "Unique Word Ratio"
            ],
            "Value": [
                f"{int(features['text_length'])} chars",
                f"{int(features['word_count'])} words",
                f"{features['avg_word_length']:.2f}",
                f"{int(features['clickbait_score'])} words",
                f"{int(features['hedge_score'])} words",
                f"{features['sent_compound']:.3f}",
                f"{features['caps_ratio']*100:.1f}%",
                f"{int(features['exclamation_count'])}",
                "Yes ✅" if features['has_source_citation'] else "No ❌",
                f"{features['unique_word_ratio']:.2f}",
            ],
            "Signal": [
                "neutral", "neutral",
                "⬆️ higher = more formal",
                "⬆️ higher = more suspicious",
                "⬆️ higher = more credible",
                "⬆️ extreme = suspicious",
                "⬆️ higher = sensationalist",
                "⬆️ more = clickbait",
                "citations = credibility",
                "⬇️ lower = repetitive/propaganda",
            ]
        }
        st.table(pd.DataFrame(feat_data))

    elif btn:
        st.warning("Please enter some text to analyze.")


# ─── Tab 2: Model Info ───

with tab2:
    st.subheader("🤖 Models Trained")

    model_data = {
        "Model": ["Logistic Regression", "Random Forest", "SVM (Linear)", "Gradient Boosting ⭐"],
        "Strengths": [
            "Fast, interpretable, great baseline",
            "Robust, handles noisy data",
            "Excellent for text classification, high-dimensional data",
            "Best accuracy, captures nonlinear patterns"
        ],
        "Best For": ["Speed", "Robustness", "High-D Text", "Accuracy"],
    }
    st.table(pd.DataFrame(model_data))

    st.info("ℹ️ The app uses **Gradient Boosting** as the deployed model (best performance on sample data). Train on your full dataset via `python main.py` for real-world accuracy.")

    st.subheader("📐 Feature Groups")
    st.markdown("""
    **Text Embedding (TF-IDF)**
    - 5,000–10,000 top n-gram features (unigrams + bigrams)
    - Captures vocabulary and phrasing patterns

    **Hand-Crafted Features (15 features)**
    - *Structural*: text length, word count, avg word length
    - *Sensationalism*: clickbait score, CAPS ratio, exclamation count
    - *Credibility*: hedge word score, source citation presence
    - *Linguistic*: unique word ratio, number count
    - *Sentiment*: VADER neg/neu/pos/compound scores
    """)


# ─── Tab 3: How It Works ───

with tab3:
    st.subheader("🔬 Pipeline Overview")
    st.markdown("""
    ```
    Input Article
         │
         ▼
    ┌─────────────────┐
    │  Preprocessing  │  lowercase, remove HTML/URLs, tokenize, stem
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────────────┐
    │       Feature Extraction            │
    │  ┌──────────────┐ ┌───────────────┐ │
    │  │ TF-IDF       │ │ Hand-crafted  │ │
    │  │ (n-grams)    │ │ features (15) │ │
    │  └──────────────┘ └───────────────┘ │
    └──────────────┬──────────────────────┘
                   │  hstack → combined matrix
                   ▼
    ┌──────────────────────────┐
    │  ML Classifier           │
    │  (Gradient Boosting)     │
    └──────────────┬───────────┘
                   │
                   ▼
          FAKE / REAL + Probability
    ```

    **Why this works:**
    - TF-IDF captures *what words* appear and how often
    - Hand-crafted features capture *writing style* and sensationalism signals
    - Combining both gives the model vocabulary + behavioral signals
    - Gradient Boosting handles interactions between all features
    """)

    st.subheader("📚 Recommended Datasets")
    st.markdown("""
    | Dataset | Records | Notes |
    |---|---|---|
    | [Kaggle Fake News](https://www.kaggle.com/c/fake-news) | ~20K | Classic benchmark |
    | [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) | ~12K | Multi-class labels |
    | [Fake & Real News](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) | ~44K | Well balanced |
    """)
