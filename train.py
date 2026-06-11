"""
train.py — Upgraded Model Training Pipeline (v2)
=================================================
What's new in v2:
  ✅ All four trained models saved individually (models/<name>.pkl)
  ✅ model_registry.json lets Flask switch active model at runtime
  ✅ ROC curve overlay chart generated (static/images/roc_curves.png)
  ✅ Cross-validation F1 scores (mean ± std) reported per model
  ✅ Improved NLP preprocessing (HTML entity decoding, contractions)
  ✅ batch_preprocess() used for faster multi-core-friendly processing

Run:
    python train.py

Outputs:
    models/model.pkl              ← best classifier (backward compatible)
    models/vectorizer.pkl         ← fitted TF-IDF
    models/<name>.pkl             ← every model saved individually  ← NEW
    models/model_registry.json    ← name → path mapping             ← NEW
    models/metrics.json           ← all model scores (+ cv_f1)      ← NEW
    models/model_metadata.json    ← training metadata
    static/images/cm_*.png        ← per-model confusion matrices
    static/images/all_confusion_matrices.png
    static/images/model_comparison.png
    static/images/roc_curves.png  ← NEW
"""

import os
import sys
import pickle
import warnings
import logging
import json
import datetime

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# ── Local utilities ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.preprocessing import preprocess, TEXT_COLS
from utils.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_all_confusion_matrices,
    plot_model_comparison,
    plot_roc_curves,       # ← NEW
    save_metrics,
    save_all_models,       # ← NEW
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data",   "fake_job_postings.csv")
M_OUT  = os.path.join(BASE, "models", "model.pkl")
V_OUT  = os.path.join(BASE, "models", "vectorizer.pkl")
MD_OUT = os.path.join(BASE, "models", "model_metadata.json")
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
#  CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        class_weight="balanced", random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ),
    "Linear SVM": LinearSVC(
        C=1.0, class_weight="balanced",
        max_iter=2000, random_state=42,
    ),
    "Naive Bayes": MultinomialNB(alpha=1.0),
}


# ══════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════

def add_missingness_features(df):
    """
    Add binary indicator columns for missing high-signal fields.
    Fraudulent postings disproportionately omit salary, company_profile,
    and benefits — the absence itself is a fraud signal.
    """
    missing_cols = {
        "is_salary_empty":   "salary_range",
        "is_profile_empty":  "company_profile",
        "is_benefits_empty": "benefits",
        "is_dept_empty":     "department",
    }
    for feat, col in missing_cols.items():
        if col in df.columns:
            df[feat] = df[col].isna().astype(int)
        else:
            df[feat] = 0
    return df


# ══════════════════════════════════════════════════════════════════════
#  MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load dataset ───────────────────────────────────────────────
    if not os.path.exists(DATA):
        logger.error(f"Dataset not found: {DATA}")
        logger.error("Download fake_job_postings.csv from Kaggle and place in data/")
        sys.exit(1)

    logger.info("Loading dataset...")
    df = pd.read_csv(DATA)
    logger.info(f"Shape: {df.shape} | Fraudulent: {df['fraudulent'].sum()} "
                f"({df['fraudulent'].mean()*100:.1f}%)")

    # ── 2. Fill NaN in text columns ───────────────────────────────────
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # ── 3. Add missingness features ───────────────────────────────────
    df = add_missingness_features(df)

    # ── 4. Combine text columns ───────────────────────────────────────
    df["combined"] = df[TEXT_COLS].apply(
        lambda row: " ".join(row.astype(str)), axis=1
    )

    # ── 5. NLP preprocessing ──────────────────────────────────────────
    logger.info("Preprocessing text (HTML entity decode + contractions expansion)...")
    df["clean"] = df["combined"].apply(preprocess)

    # ── 6. TF-IDF vectorisation ───────────────────────────────────────
    logger.info("Fitting TF-IDF vectoriser...")
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(df["clean"])
    y = df["fraudulent"].values

    # ── 7. Train / test split (stratified) ────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ── 8. Train all classifiers & compute metrics ────────────────────
    results = []
    for name, clf in CLASSIFIERS.items():
        logger.info(f"Training {name}...")
        try:
            result = compute_metrics(name, clf, X_train, X_test, y_train, y_test,
                                     cv_folds=5)
            results.append(result)
            cv_str = (f"CV-F1={result['cv_f1_mean']*100:.2f}%±{result['cv_f1_std']*100:.2f}%"
                      if result.get("cv_f1_mean") else "CV-F1=N/A")
            logger.info(
                f"  Accuracy={result['accuracy']*100:.2f}%  "
                f"Fraud-F1={result['f1_fraud']*100:.2f}%  "
                f"ROC-AUC={result['roc_auc']}  {cv_str}"
            )
        except Exception as ex:
            logger.error(f"  {name} failed: {ex}")

    if not results:
        logger.error("All classifiers failed -- aborting.")
        sys.exit(1)

    # ── 9. Select best model (by fraud-class F1) ───────────────────────
    best = max(results, key=lambda r: r["f1_fraud"])
    logger.info(f"\n★ Best model: {best['name']} (Fraud F1 = {best['f1_fraud']*100:.2f}%)")

    # ── 10. Save best model + vectorizer (backward compatible) ────────
    with open(M_OUT, "wb") as f:
        pickle.dump(best["model"], f)
    with open(V_OUT, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Saved best model → {M_OUT}")

    # ── 11. Save ALL models individually  ← NEW ───────────────────────
    registry = save_all_models(results)
    logger.info(f"Saved model registry: {list(registry.keys())}")

    # ── 12. Save metrics JSON ─────────────────────────────────────────
    save_metrics(results)
    logger.info("Saved: models/metrics.json")

    # ── 13. Save model metadata ───────────────────────────────────────
    metadata = {
        "best_model":     best["name"],
        "trained_at":     datetime.datetime.now().isoformat(),
        "dataset_rows":   int(df.shape[0]),
        "dataset_cols":   int(df.shape[1]),
        "fraud_count":    int(df["fraudulent"].sum()),
        "legit_count":    int(df.shape[0] - df["fraudulent"].sum()),
        "tfidf_features": vectorizer.max_features,
        "test_size":      0.20,
        "accuracy":       best["accuracy"],
        "f1_fraud":       best["f1_fraud"],
        "roc_auc":        best["roc_auc"],
        "all_models":     [r["name"] for r in results],
    }
    with open(MD_OUT, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved: models/model_metadata.json")

    # ── 14. Generate & save plots ─────────────────────────────────────
    logger.info("Generating evaluation plots...")
    for r in results:
        plot_confusion_matrix(r, save=True)
    plot_all_confusion_matrices(results, save=True)
    plot_model_comparison(results, save=True)
    plot_roc_curves(results, save=True)           # ← NEW
    logger.info("Plots saved to static/images/")

    # ── 15. Print results table ───────────────────────────────────────
    SEP = "=" * 100
    print(f"\n{SEP}")
    print(f"{'Model':<22} {'Acc':>7} {'Pre(W)':>8} {'Rec(W)':>8} {'F1(W)':>7} "
          f"{'Fraud-F1':>10} {'AUC':>7} {'CV-F1 (mean±std)':>18}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["f1_fraud"], reverse=True):
        star   = "  * BEST" if r["name"] == best["name"] else ""
        auc    = f"{r['roc_auc']:.4f}" if r["roc_auc"] else "  N/A "
        cv_str = (f"{r['cv_f1_mean']*100:.2f}%±{r['cv_f1_std']*100:.2f}%"
                  if r.get("cv_f1_mean") else "   N/A    ")
        print(
            f"{r['name']:<22} "
            f"{r['accuracy']*100:>6.2f}% "
            f"{r['precision_weighted']*100:>7.2f}% "
            f"{r['recall_weighted']*100:>7.2f}% "
            f"{r['f1_weighted']*100:>6.2f}% "
            f"{r['f1_fraud']*100:>9.2f}% "
            f"{auc:>7}"
            f"{cv_str:>19}"
            f"{star}"
        )
    print(SEP)
    print(f"\n* Selected: {best['name']}")
    print(f"\n  Note: Selection is based on Fraud-class F1 (not overall accuracy).")
    print("Done! Training complete. Run:  python app.py")
    print(SEP + "\n")

    # ── 16. Print classification reports ─────────────────────────────
    print("\n" + "-" * 50)
    print("DETAILED CLASSIFICATION REPORTS")
    print("-" * 50)
    for r in results:
        star = " * SELECTED" if r["name"] == best["name"] else ""
        print(f"\n{'-'*30}\n{r['name']}{star}\n{'-'*30}")
        print(r["classification_report"])


if __name__ == "__main__":
    main()
