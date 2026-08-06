"""
train.py — JobGuard v7 Training Pipeline
=========================================
Improvements from v6:
  ✅ FIX: LinearSVC now wrapped in CalibratedClassifierCV(method='sigmoid', cv=5)
          — gives real predict_proba for ALL models, so threshold.json applies
          consistently across the board. No more special-casing in app.py.

  ✅ FIX: Missingness features REMOVED from training.
          v6 train.py stacked 4 binary missingness features onto TF-IDF,
          producing a 10004-dim model. But ml_predict() only sent 10000-dim
          vectors (TF-IDF only). Re-running train.py would create a model
          that crashes at inference. Missingness features are now gone from
          both training and inference.

  ✅ FIX: save_all_models() saves ABSOLUTE paths — model switching under
          gunicorn (non-root CWD) now works without fallback to model.pkl.

  ✅ IMPROVEMENT: find_optimal_threshold() is now meaningful for ALL models
          because every model has predict_proba. Previously the threshold was
          found but LinearSVC ignored it entirely.

  ✅ IMPROVEMENT: RandomForest n_estimators raised to 200 for better recall.

Usage (GitHub Codespaces / Render Shell):
  1. Put fake_job_postings.csv in data/
  2. pip install -r requirements.txt
  3. python nltk_setup.py
  4. python train.py
"""

import os
import sys
import json
import pickle
import logging
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import preprocess
from utils.evaluation import (
    compute_metrics,
    save_metrics,
    save_all_models,
    plot_model_comparison,
    plot_roc_curves,
    plot_all_confusion_matrices,
    MODELS_DIR,
    BASE_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(BASE_DIR, "data", "fake_job_postings.csv")
VEC_PATH   = os.path.join(MODELS_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
META_PATH  = os.path.join(MODELS_DIR, "model_metadata.json")
THRESH_PATH = os.path.join(MODELS_DIR, "threshold.json")

# ── Model definitions ──────────────────────────────────────────────────
# ALL models now have predict_proba via calibration or natively:
#   LogisticRegression → predict_proba natively
#   CalibratedClassifierCV(LinearSVC) → predict_proba via sigmoid calibration
#   RandomForestClassifier → predict_proba natively
#   MultinomialNB → predict_proba natively
#
# This means threshold.json applies consistently to every model.
CLASSIFIERS = OrderedDict([
    ("Logistic Regression", LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        class_weight="balanced", random_state=42,
    )),
    ("Linear SVM", CalibratedClassifierCV(
        LinearSVC(C=1.0, class_weight="balanced",
                  max_iter=2000, random_state=42),
        method="sigmoid", cv=5,
    )),
    ("Random Forest", RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        max_depth=20, min_samples_split=5,
        random_state=42, n_jobs=-1,
    )),
    ("Naive Bayes", MultinomialNB(alpha=0.5)),
])


def find_optimal_threshold(clf, X_val, y_val) -> float:
    """
    Find the probability threshold that maximises F1 for the fraud class.
    Works for ALL models now that every model has predict_proba.
    """
    try:
        probs = clf.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, probs)
        f1_scores = np.where(
            (precision + recall) == 0, 0,
            2 * precision * recall / (precision + recall),
        )
        best_idx = int(np.argmax(f1_scores[:-1]))
        best_t   = float(thresholds[best_idx])
        logger.info(
            "  Optimal threshold: %.4f  (F1=%.4f  P=%.4f  R=%.4f)",
            best_t, f1_scores[best_idx],
            precision[best_idx], recall[best_idx],
        )
        return best_t
    except Exception as e:
        logger.warning("  Could not compute optimal threshold (%s) — using 0.5", e)
        return 0.5


def main():
    logger.info("=" * 60)
    logger.info("JobGuard v7 — Training Pipeline")
    logger.info("=" * 60)

    if not os.path.exists(DATA_PATH):
        logger.error("Dataset not found: %s", DATA_PATH)
        logger.error("Download from Kaggle: 'Fake Job Postings' / EMSCAD dataset")
        sys.exit(1)

    # ── 1. Load & validate dataset ────────────────────────────────────
    logger.info("Loading dataset: %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("Shape: %s", df.shape)

    required_cols = {"title", "description", "fraudulent"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("Missing columns: %s", missing)
        sys.exit(1)

    logger.info("Class distribution:\n%s", df["fraudulent"].value_counts())

    # ── 2. Feature engineering ────────────────────────────────────────
    TEXT_COLS = ["title", "company_profile", "description",
                 "requirements", "benefits", "location",
                 "employment_type", "required_experience",
                 "required_education", "industry", "function"]

    logger.info("Building combined text feature ...")
    combined = df[[c for c in TEXT_COLS if c in df.columns]].fillna("").apply(
        lambda row: " ".join(str(v) for v in row if str(v).strip()),
        axis=1,
    )

    logger.info("Preprocessing text (this takes ~2–3 minutes) ...")
    X_text = combined.apply(preprocess)
    y      = df["fraudulent"].astype(int)

    # ── 3. Train / validation / test split ───────────────────────────
    # 70% train | 10% validation (threshold tuning) | 20% test (final eval)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_text, y, test_size=0.20, random_state=42, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=42, stratify=y_trainval,
    )
    logger.info(
        "Split: train=%d  val=%d  test=%d",
        len(X_train), len(X_val), len(X_test),
    )

    # ── 4. TF-IDF vectorizer ──────────────────────────────────────────
    logger.info("Fitting TF-IDF vectorizer (max_features=10000, ngram=(1,2)) ...")
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        min_df=3,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)
    X_test_tfidf  = vectorizer.transform(X_test)

    with open(VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Vectorizer saved → %s", VEC_PATH)

    # ── 5. Train all classifiers ──────────────────────────────────────
    results = []
    best_name  = None
    best_f1    = -1.0
    best_model = None

    for name, clf in CLASSIFIERS.items():
        logger.info("\n── Training: %s ──", name)

        try:
            result = compute_metrics(
                name, clf,
                X_train_tfidf, X_test_tfidf,
                y_train.values,  y_test.values,
                cv_folds=5,
            )
        except Exception as e:
            logger.error("Failed to train %s: %s", name, e)
            continue

        logger.info(
            "  Accuracy=%.4f  F1(fraud)=%.4f  Recall(fraud)=%.4f  "
            "Precision(fraud)=%.4f  ROC-AUC=%s",
            result["accuracy"], result["f1_fraud"],
            result["recall_fraud"], result["precision_fraud"],
            result.get("roc_auc", "N/A"),
        )
        if result.get("cv_f1_mean") is not None:
            logger.info(
                "  CV F1(fraud): %.4f ± %.4f",
                result["cv_f1_mean"], result["cv_f1_std"],
            )

        results.append(result)

        if result["f1_fraud"] > best_f1:
            best_f1    = result["f1_fraud"]
            best_name  = name
            best_model = result["model"]

    if not results:
        logger.error("No models trained successfully.")
        sys.exit(1)

    # ── 6. Find optimal threshold on validation set ───────────────────
    logger.info("\nFinding optimal threshold using validation set ...")
    best_threshold = find_optimal_threshold(best_model, X_val_tfidf, y_val.values)
    with open(THRESH_PATH, "w") as f:
        json.dump({"threshold": best_threshold, "best_model": best_name}, f, indent=2)
    logger.info("Threshold %.4f saved → %s", best_threshold, THRESH_PATH)

    # ── 7. Save models ────────────────────────────────────────────────
    logger.info("\nSaving all models ...")
    registry = save_all_models(results)
    logger.info("Registry: %s", list(registry.keys()))

    # Best model → model.pkl (app default)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    logger.info("Best model (%s) saved → %s", best_name, MODEL_PATH)

    # ── 8. Save metrics JSON ──────────────────────────────────────────
    save_metrics(results)
    logger.info("Metrics saved → models/metrics.json")

    # ── 9. Save model metadata ─────────────────────────────────────────
    best_result = next(r for r in results if r["name"] == best_name)
    metadata = {
        "best_model":           best_name,
        "best_f1_fraud":        best_result["f1_fraud"],
        "best_accuracy":        best_result["accuracy"],
        "best_roc_auc":         best_result.get("roc_auc"),
        "best_recall_fraud":    best_result["recall_fraud"],
        "best_precision_fraud": best_result["precision_fraud"],
        "optimal_threshold":    best_threshold,
        "n_train":              len(X_train),
        "n_val":                len(X_val),
        "n_test":               len(X_test),
        "tfidf_vocab_size":     len(vectorizer.vocabulary_),
        "tfidf_max_features":   10_000,
        "tfidf_ngram_range":    [1, 2],
        "models_trained":       [r["name"] for r in results],
        "calibrated_svm":       True,
        "missingness_features": False,   # explicitly removed — inference safe
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved → %s", META_PATH)

    # ── 10. Generate plots ────────────────────────────────────────────
    logger.info("\nGenerating plots ...")
    try:
        plot_model_comparison(results)
        plot_roc_curves(results)
        plot_all_confusion_matrices(results)
        logger.info("Plots saved → static/images/")
    except Exception as e:
        logger.warning("Plot generation failed: %s", e)

    # ── 11. Summary ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Training complete.")
    logger.info("Best model : %s  (F1-fraud=%.4f)", best_name, best_f1)
    logger.info("Threshold  : %.4f", best_threshold)
    logger.info("=" * 60)
    logger.info("\nModel performance summary:")
    for r in sorted(results, key=lambda x: x["f1_fraud"], reverse=True):
        logger.info(
            "  %-22s  F1=%.4f  Recall=%.4f  AUC=%s",
            r["name"], r["f1_fraud"], r["recall_fraud"],
            f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "N/A",
        )


if __name__ == "__main__":
    main()
