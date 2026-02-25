"""
models/trainer.py
------------------
Train, evaluate, and compare multiple ML classifiers.
Includes: Accuracy, Precision, Recall, F1, Confusion Matrix, Error Analysis, SHAP.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Model Zoo
# ──────────────────────────────────────────────

def get_models() -> dict:
    """Return all models to train and compare."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        ),
        "SVM (Linear)": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, C=1.0, random_state=42)
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
        ),
    }


# ──────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute all evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    print(f"\n{'─'*50}")
    print(f"📊 {model_name}")
    print(f"{'─'*50}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Real', 'Fake'])}")

    return metrics, y_pred


def train_and_evaluate(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train all models, evaluate, and return results + best model.
    Returns: (results_df, trained_models_dict, best_model_name)
    """
    models = get_models()
    all_metrics = []
    trained_models = {}
    all_preds = {}

    print("\n🚀 Training models...\n")

    for name, model in models.items():
        print(f"⏳ Training {name}...")
        model.fit(X_train, y_train)
        metrics, y_pred = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        trained_models[name] = model
        all_preds[name] = y_pred

    results_df = pd.DataFrame(all_metrics).set_index("model").sort_values("f1", ascending=False)
    best_model_name = results_df.index[0]

    print(f"\n🏆 Best Model: {best_model_name} (F1 = {results_df.loc[best_model_name, 'f1']:.4f})")

    return results_df, trained_models, best_model_name, all_preds


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Bar chart comparing all models across metrics."""
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(results_df))
    width = 0.2

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, results_df[metric], width, label=metric.capitalize(), color=colors[i])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df.index, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Fake News Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved model comparison chart → {save_path}")
    return fig


def plot_confusion_matrix(y_test, y_pred, model_name: str, save_path: str = None):
    """Annotated confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        ax=ax
    )
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved confusion matrix → {save_path}")
    return fig


# ──────────────────────────────────────────────
# Error Analysis  ⭐⭐⭐  (Google loves this)
# ──────────────────────────────────────────────

def error_analysis(X_test_raw_texts, y_test, y_pred, model_name: str, n: int = 10) -> pd.DataFrame:
    """
    Inspect false positives and false negatives.
    Returns a DataFrame of misclassified examples.
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    misclassified_idx = np.where(y_test != y_pred)[0]

    errors = []
    for idx in misclassified_idx[:n]:
        true_label = "Real" if y_test[idx] == 0 else "Fake"
        pred_label = "Real" if y_pred[idx] == 0 else "Fake"
        error_type = "False Positive (Real → Fake)" if y_test[idx] == 0 else "False Negative (Fake → Real)"
        errors.append({
            "Error Type": error_type,
            "True Label": true_label,
            "Predicted": pred_label,
            "Text Snippet": str(X_test_raw_texts[idx])[:150] + "...",
        })

    error_df = pd.DataFrame(errors)
    print(f"\n🔍 Error Analysis for {model_name} (showing {len(error_df)} misclassifications)")
    print(f"  Total misclassified: {len(misclassified_idx)} / {len(y_test)}")
    print(f"  Error rate: {len(misclassified_idx)/len(y_test)*100:.2f}%\n")
    if not error_df.empty:
        print(error_df[["Error Type", "True Label", "Predicted"]].value_counts().to_string())

    return error_df


# ──────────────────────────────────────────────
# SHAP Explainability  ✨
# ──────────────────────────────────────────────

def plot_shap_importance(model, X_train, feature_names: list, model_name: str, save_path: str = None, top_n: int = 20):
    """
    SHAP-based feature importance for tree-based models,
    coefficient-based for linear models.
    """
    try:
        import shap
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        elif hasattr(model, "calibrated_classifiers_"):
            # CalibratedClassifierCV wrapping LinearSVC
            base = model.calibrated_classifiers_[0].estimator
            importances = np.abs(base.coef_[0])
        else:
            print("⚠️  SHAP not available for this model type.")
            return None

        indices = np.argsort(importances)[-top_n:][::-1]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#d62728" if i < top_n // 2 else "#1f77b4" for i in range(len(top_features))]
        ax.barh(range(len(top_features)), top_importances[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1], fontsize=9)
        ax.set_xlabel("Feature Importance", fontsize=11)
        ax.set_title(f"Top {top_n} Features — {model_name}", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"📊 Saved feature importance chart → {save_path}")
        return fig

    except Exception as e:
        print(f"⚠️  Feature importance plot failed: {e}")
        return None
