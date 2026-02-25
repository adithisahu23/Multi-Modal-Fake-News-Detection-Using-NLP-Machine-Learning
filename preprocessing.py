"""
utils/preprocessing.py
-----------------------
Text cleaning, tokenization, and vectorization pipeline.
"""

import re
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Pure-Python stopwords (no NLTK dependency) ──
STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
    "weren","won","wouldn",
}


def _simple_stem(word: str) -> str:
    """Very lightweight suffix-stripping (no external deps needed)."""
    for suffix in ("ing", "tion", "ness", "ment", "ed", "ly", "er", "est", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


# ──────────────────────────────────────────────
# 1. Text Cleaning
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Full cleaning pipeline:
      - Lowercase
      - Remove URLs, HTML tags, special chars
      - Remove punctuation & extra whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"<.*?>", "", text)                    # HTML tags
    text = re.sub(r"[^a-z\s]", "", text)                 # non-alpha
    text = re.sub(r"\s+", " ", text).strip()             # extra spaces
    return text


def tokenize(text: str) -> list:
    """Tokenize, remove stopwords, and stem (pure Python — no NLTK needed)."""
    tokens = re.findall(r"\b[a-z]{3,}\b", text)
    tokens = [_simple_stem(t) for t in tokens if t not in STOP_WORDS]
    return tokens


def tokens_to_string(text: str) -> str:
    """Clean → tokenize → rejoin (for TF-IDF input)."""
    return " ".join(tokenize(clean_text(text)))


# ──────────────────────────────────────────────
# 2. Vectorization
# ──────────────────────────────────────────────

def build_tfidf_vectorizer(max_features: int = 10000, ngram_range=(1, 2)):
    """
    Returns a fitted TF-IDF vectorizer config.
    Bigrams included — captures 'fake president', 'breaking news', etc.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,          # log normalization — standard best practice
        min_df=2,
        max_df=0.95,
    )


# ──────────────────────────────────────────────
# 3. Dataset Loader
# ──────────────────────────────────────────────

def load_dataset(path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Load CSV dataset.
    Expects columns: text_col (article text), label_col (0=real, 1=fake).
    Also supports title column for headline features.
    """
    df = pd.read_csv(path)

    # Merge title + text if both present
    if "title" in df.columns and text_col in df.columns:
        df["full_text"] = df["title"].fillna("") + " " + df[text_col].fillna("")
    elif text_col in df.columns:
        df["full_text"] = df[text_col].fillna("")
    else:
        raise ValueError(f"Column '{text_col}' not found in dataset.")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    df["label"] = df[label_col]
    df.dropna(subset=["full_text", "label"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"✅ Loaded {len(df)} samples | Label distribution:\n{df['label'].value_counts()}\n")
    return df


def load_sample_data() -> pd.DataFrame:
    """
    Built-in sample dataset for quick testing without external files.
    Returns a small DataFrame with 'full_text' and 'label' columns.
    """
    real_headlines = [
        "Scientists discover new species of deep-sea fish near Pacific Ocean floor",
        "Federal Reserve holds interest rates steady amid inflation concerns",
        "NASA announces successful Mars rover soil sample collection",
        "Stock markets close higher after positive jobs report",
        "World Health Organization reports decline in COVID-19 cases globally",
        "Congress passes bipartisan infrastructure bill worth $1.2 trillion",
        "Apple reports record quarterly earnings driven by iPhone sales",
        "Climate summit concludes with agreement on carbon emission targets",
        "University researchers develop more efficient solar panel technology",
        "Supreme Court rules on landmark privacy case involving digital data",
        "International space station celebrates 25 years of continuous habitation",
        "New study links Mediterranean diet to reduced heart disease risk",
        "Tech giants face antitrust scrutiny from European regulators",
        "Economists predict moderate growth for global economy next year",
        "Scientists confirm existence of water ice at lunar south pole",
    ]

    fake_headlines = [
        "BREAKING: Government secretly puts microchips in COVID vaccines to track citizens",
        "EXCLUSIVE: Aliens helped build the Egyptian pyramids, leaked documents prove",
        "SHOCKING: 5G towers are causing birds to fall from the sky by the thousands",
        "Scientists ADMIT the earth is actually flat in secret memo leaked online",
        "DEEP STATE exposed: Crisis actors used in every major US disaster since 2001",
        "Doctors don't want you to know this one weird trick to cure any disease",
        "BOMBSHELL: Moon landing was filmed in Hollywood, NASA insider finally confesses",
        "Secret government program replacing world leaders with robotic clones revealed",
        "Obama and Hillary Clinton arrested for treason, mainstream media covering it up",
        "Bill Gates funding weather control satellites to cause global famine",
        "URGENT: Drinking bleach cures cancer, Big Pharma suppressing the truth",
        "FBI whistleblower reveals election machines hacked in all 50 states",
        "Mainstream scientists prove vaccines cause autism in secret study",
        "REVEALED: The sun is actually getting colder, global warming is a hoax",
        "New world order plans global takeover by 2025, leaked at Davos meeting",
    ]

    texts = real_headlines + fake_headlines
    labels = [0] * len(real_headlines) + [1] * len(fake_headlines)

    df = pd.DataFrame({"full_text": texts, "label": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✅ Loaded sample dataset: {len(df)} records | Labels: {df['label'].value_counts().to_dict()}\n")
    return df
