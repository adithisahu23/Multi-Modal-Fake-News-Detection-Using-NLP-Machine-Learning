"""
utils/feature_engineering.py
------------------------------
Hand-crafted features that complement TF-IDF embeddings.
These signal "research brain" on your resume — exactly what Google looks for.
"""

import re
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Sensationalism vocabulary
# ──────────────────────────────────────────────

CLICKBAIT_WORDS = {
    "shocking", "bombshell", "exclusive", "breaking", "leaked",
    "revealed", "secret", "urgent", "exposed", "proof", "confirms",
    "conspiracy", "banned", "suppressed", "covered", "admit",
    "never", "always", "every", "all", "none", "impossible"
}

HEDGE_WORDS = {
    "according", "reported", "officials", "sources", "said",
    "stated", "confirmed", "announced", "study", "research",
    "evidence", "data", "analysis", "investigation"
}

# Simple sentiment word lists (replaces VADER — no external deps)
POSITIVE_WORDS = {
    "good","great","excellent","best","wonderful","amazing","fantastic","positive",
    "success","achieve","benefit","improve","support","safe","healthy","strong",
    "confirm","official","expert","scientist","university","study","evidence"
}
NEGATIVE_WORDS = {
    "bad","terrible","worst","horrible","awful","dangerous","failed","collapse",
    "corrupt","lie","fake","hoax","fraud","scam","cover","secret","admit",
    "exposed","leaked","crisis","disaster","attack","ban","destroy","kill"
}


def _simple_sentiment(text: str) -> dict:
    """
    Lightweight sentiment using word counts.
    Returns dict with neg, pos, compound scores.
    """
    if not isinstance(text, str) or not text.strip():
        return {"sent_neg": 0.0, "sent_neu": 1.0, "sent_pos": 0.0, "sent_compound": 0.0}
    words = set(re.findall(r"\b\w+\b", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = max(pos + neg, 1)
    compound = (pos - neg) / (len(words) + 1)
    return {
        "sent_neg": neg / total,
        "sent_neu": max(0, 1 - (pos + neg) / total),
        "sent_pos": pos / total,
        "sent_compound": float(np.clip(compound, -1, 1)),
    }


# ──────────────────────────────────────────────
# Individual feature functions
# ──────────────────────────────────────────────

def text_length(text: str) -> int:
    """Total character count — fake news often shorter/clickbait."""
    return len(text) if isinstance(text, str) else 0


def word_count(text: str) -> int:
    return len(text.split()) if isinstance(text, str) else 0


def avg_word_length(text: str) -> float:
    """Higher avg word length → more academic/factual writing."""
    words = text.split() if isinstance(text, str) else []
    return np.mean([len(w) for w in words]) if words else 0.0


def exclamation_count(text: str) -> int:
    """Fake news loves exclamation marks."""
    return text.count("!") if isinstance(text, str) else 0


def question_count(text: str) -> int:
    return text.count("?") if isinstance(text, str) else 0


def caps_ratio(text: str) -> float:
    """Ratio of UPPERCASE letters — sensationalism signal."""
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    alpha = [c for c in text if c.isalpha()]
    return sum(1 for c in alpha if c.isupper()) / len(alpha) if alpha else 0.0


def clickbait_score(text: str) -> int:
    """Count of known sensationalist / clickbait words."""
    if not isinstance(text, str):
        return 0
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return len(words & CLICKBAIT_WORDS)


def hedge_score(text: str) -> int:
    """Count of journalistic hedge words (signal of real reporting)."""
    if not isinstance(text, str):
        return 0
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return len(words & HEDGE_WORDS)


def sentiment_scores(text: str) -> dict:
    """
    Sentiment analysis using word-list heuristics (no external deps).
    Returns: neg, neu, pos, compound scores.
    Fake news tends toward extreme compound scores (very pos or very neg).
    """
    return _simple_sentiment(text)


def unique_word_ratio(text: str) -> float:
    """
    Ratio of unique words to total words.
    Lower ratio → more repetitive language (possible propaganda signal).
    """
    words = text.lower().split() if isinstance(text, str) else []
    return len(set(words)) / len(words) if words else 0.0


def number_count(text: str) -> int:
    """Count of numeric tokens — real news cites more statistics."""
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b\d+\.?\d*\b", text))


def has_source_citation(text: str) -> int:
    """Binary: does the text cite a source? (Reuters, AP, said, reported, etc.)"""
    if not isinstance(text, str):
        return 0
    pattern = r"\b(reuters|associated press|ap|said|reported|according|spokesman|official)\b"
    return 1 if re.search(pattern, text.lower()) else 0


# ──────────────────────────────────────────────
# Main feature extraction function
# ──────────────────────────────────────────────

def extract_features(df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
    """
    Run all feature extractors on a DataFrame.
    Returns a new DataFrame with all engineered features.
    """
    texts = df[text_col].fillna("").tolist()

    records = []
    for text in texts:
        sent = sentiment_scores(text)
        record = {
            "text_length": text_length(text),
            "word_count": word_count(text),
            "avg_word_length": avg_word_length(text),
            "exclamation_count": exclamation_count(text),
            "question_count": question_count(text),
            "caps_ratio": caps_ratio(text),
            "clickbait_score": clickbait_score(text),
            "hedge_score": hedge_score(text),
            "unique_word_ratio": unique_word_ratio(text),
            "number_count": number_count(text),
            "has_source_citation": has_source_citation(text),
            **sent,
        }
        records.append(record)

    feature_df = pd.DataFrame(records)
    print(f"✅ Extracted {feature_df.shape[1]} hand-crafted features for {len(feature_df)} samples")
    return feature_df
