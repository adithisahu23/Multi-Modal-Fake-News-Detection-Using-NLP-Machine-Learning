"""
main.py
--------
Full end-to-end fake news detection pipeline.
Run: python main.py [--data path/to/data.csv]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Local modules ──
from utils.preprocessing import (
    load_dataset, load_sample_data, tokens_to_string, build_tfidf_vectorizer
)
from utils.feature_engineering import extract_features
from models.trainer import (
    train_and_evaluate, plot_model_comparison,
    plot_confusion_matrix, error_analysis, plot_shap_importance
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

def run_pipeline(data_path: str = None):
    print("\n" + "═" * 60)
    print("   🧠 FAKE NEWS DETECTION — ML PIPELINE")
    print("═" * 60 + "\n")

    # ── 1. Load Data ──
    print("📥 Step 1: Loading data...\n")
    if data_path:
        df = load_dataset(data_path)
    else:
        print("ℹ️  No dataset path provided. Using built-in sample data.")
        print("   To use your own: python main.py --data path/to/data.csv\n")
        df = load_sample_data()

    # ── 2. Preprocessing ──
    print("🔧 Step 2: Preprocessing text...\n")
    df["clean_text"] = df["full_text"].apply(tokens_to_string)

    # ── 3. Feature Engineering ──
    print("⚙️  Step 3: Engineering features...\n")
    feature_df = extract_features(df, text_col="full_text")

    # ── 4. TF-IDF Vectorization ──
    print("\n📐 Step 4: TF-IDF Vectorization...\n")
    vectorizer = build_tfidf_vectorizer(max_features=10000)

    # Train/test split (stratified)
    X_text = df["clean_text"]
    X_feat = feature_df.values
    y = df["label"].values

    (X_text_train, X_text_test,
     X_feat_train, X_feat_test,
     y_train, y_test,
     idx_train, idx_test) = train_test_split(
        X_text, X_feat, y, np.arange(len(df)),
        test_size=0.2, random_state=42, stratify=y
    )

    # Fit TF-IDF on train only
    X_tfidf_train = vectorizer.fit_transform(X_text_train)
    X_tfidf_test = vectorizer.transform(X_text_test)

    # Scale hand-crafted features
    scaler = StandardScaler()
    X_feat_train_scaled = scaler.fit_transform(X_feat_train)
    X_feat_test_scaled = scaler.transform(X_feat_test)

    # Combine TF-IDF + hand-crafted features
    X_train = hstack([X_tfidf_train, csr_matrix(X_feat_train_scaled)])
    X_test = hstack([X_tfidf_test, csr_matrix(X_feat_test_scaled)])

    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape : {X_test.shape}\n")

    # Feature names for explainability
    tfidf_feature_names = vectorizer.get_feature_names_out().tolist()
    hand_feature_names = feature_df.columns.tolist()
    all_feature_names = tfidf_feature_names + hand_feature_names

    # ── 5. Train & Evaluate ──
    print("🤖 Step 5: Training & evaluating models...\n")
    results_df, trained_models, best_model_name, all_preds = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    print("\n📋 Final Leaderboard:")
    print(results_df.to_string())

    # ── 6. Visualizations ──
    print("\n📊 Step 6: Generating visualizations...\n")

    plot_model_comparison(results_df, save_path=f"{OUTPUT_DIR}/model_comparison.png")

    best_model = trained_models[best_model_name]
    plot_confusion_matrix(
        y_test, all_preds[best_model_name], best_model_name,
        save_path=f"{OUTPUT_DIR}/confusion_matrix.png"
    )

    plot_shap_importance(
        best_model, X_train, all_feature_names, best_model_name,
        save_path=f"{OUTPUT_DIR}/feature_importance.png"
    )

    # ── 7. Error Analysis ──
    print("\n🔍 Step 7: Error analysis...\n")
    raw_test_texts = df["full_text"].iloc[idx_test].reset_index(drop=True)
    error_df = error_analysis(
        raw_test_texts, y_test, all_preds[best_model_name], best_model_name
    )
    if not error_df.empty:
        error_df.to_csv(f"{OUTPUT_DIR}/error_analysis.csv", index=False)
        print(f"  Saved error log → {OUTPUT_DIR}/error_analysis.csv")

    # ── Done ──
    print("\n" + "═" * 60)
    print(f"✅ Pipeline complete! Best model: {best_model_name}")
    print(f"   Outputs saved in: ./{OUTPUT_DIR}/")
    print("═" * 60 + "\n")

    return trained_models, vectorizer, scaler, best_model_name


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake News Detection Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    args = parser.parse_args()
    run_pipeline(data_path=args.data)
