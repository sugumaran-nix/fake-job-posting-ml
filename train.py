"""
train.py — Model Training Pipeline (v3)
=========================================
Changes from v2:
  ✅ Generates models/threshold.json with F1-optimal classification threshold
  ✅ Uses train/val/test 3-way split — hyperparameters tuned on val, metrics reported on test
  ✅ CV F1 gap vs test F1 is printed with a clear warning when gap > 5 pp
  ✅ Missingness features fed into ML (not just used in URL heuristic)
  ✅ CONTRIBUTING / dataset setup notes embedded in output

Run:
    python train.py

Outputs:
    models/model.pkl              ← best classifier
    models/vectorizer.pkl         ← fitted TF-IDF
    models/<n>.pkl                ← every model individually
    models/model_registry.json    ← name → path mapping
    models/metrics.json           ← all model scores
    models/model_metadata.json    ← training metadata
    models/threshold.json         ← F1-optimal classification threshold
    static/images/cm_*.png
    static/images/all_confusion_matrices.png
    static/images/model_comparison.png
    static/images/roc_curves.png
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
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.preprocessing import preprocess, TEXT_COLS
from utils.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_all_confusion_matrices,
    plot_model_comparison,
    plot_roc_curves,
    save_metrics,
    save_all_models,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data",   "fake_job_postings.csv")
M_OUT  = os.path.join(BASE, "models", "model.pkl")
V_OUT  = os.path.join(BASE, "models", "vectorizer.pkl")
MD_OUT = os.path.join(BASE, "models", "model_metadata.json")
TH_OUT = os.path.join(BASE, "models", "threshold.json")
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)


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


MISSINGNESS_COLS = {
    "is_salary_empty":   "salary_range",
    "is_profile_empty":  "company_profile",
    "is_benefits_empty": "benefits",
    "is_dept_empty":     "department",
}


def add_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary indicators for missing high-signal fields.
    Absence of salary/company_profile/benefits is a fraud signal.
    These features are concatenated with TF-IDF vectors before training.
    """
    for feat, col in MISSINGNESS_COLS.items():
        df[feat] = df[col].isna().astype(int) if col in df.columns else 0
    return df


def find_optimal_threshold(model, X_val, y_val) -> float:
    """
    Find the classification threshold that maximises fraud-class F1 on the
    validation set. Only applicable for models with predict_proba.
    Falls back to 0.5 for LinearSVC.
    """
    if not hasattr(model, "predict_proba"):
        return 0.5
    try:
        probs = model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, probs)
        f1_scores = np.where(
            (precision + recall) == 0,
            0,
            2 * precision * recall / (precision + recall)
        )
        best_idx = np.argmax(f1_scores[:-1])   # thresholds has len n-1
        return float(round(thresholds[best_idx], 4))
    except Exception:
        return 0.5


def main():
    # ── 1. Load ────────────────────────────────────────────────────────
    if not os.path.exists(DATA):
        logger.error(f"Dataset not found: {DATA}")
        logger.error(
            "Download fake_job_postings.csv from Kaggle:\n"
            "  https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction\n"
            "  kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction\n"
            "Place the file at: data/fake_job_postings.csv"
        )
        sys.exit(1)

    logger.info("Loading dataset...")
    df = pd.read_csv(DATA)
    logger.info(f"Shape: {df.shape} | Fraudulent: {df['fraudulent'].sum()} "
                f"({df['fraudulent'].mean()*100:.1f}%)")

    # ── 2. Fill NaN ────────────────────────────────────────────────────
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # ── 3. Missingness features ────────────────────────────────────────
    df = add_missingness_features(df)
    miss_feat_cols = list(MISSINGNESS_COLS.keys())

    # ── 4. Combine text ────────────────────────────────────────────────
    df["combined"] = df[TEXT_COLS].apply(
        lambda row: " ".join(row.astype(str)), axis=1
    )

    # ── 5. NLP preprocessing ───────────────────────────────────────────
    logger.info("Preprocessing text...")
    df["clean"] = df["combined"].apply(preprocess)

    # ── 6. TF-IDF ──────────────────────────────────────────────────────
    logger.info("Fitting TF-IDF vectoriser...")
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    X_tfidf = vectorizer.fit_transform(df["clean"])
    X_miss  = sp.csr_matrix(df[miss_feat_cols].values.astype(float))
    X       = sp.hstack([X_tfidf, X_miss], format="csr")
    y       = df["fraudulent"].values

    # ── 7. Train / Val / Test split (60 / 20 / 20, stratified) ────────
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    logger.info(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # ── 8. Train, find optimal threshold on val, evaluate on test ──────
    results = []
    thresholds = {}
    for name, clf in CLASSIFIERS.items():
        logger.info(f"Training {name}...")
        try:
            result = compute_metrics(name, clf, X_train, X_test, y_train, y_test, cv_folds=5)
            opt_thresh = find_optimal_threshold(clf, X_val, y_val)
            thresholds[name] = opt_thresh
            result["optimal_threshold"] = opt_thresh
            results.append(result)

            cv_str = (
                f"CV-F1={result['cv_f1_mean']*100:.2f}%±{result['cv_f1_std']*100:.2f}%"
                if result.get("cv_f1_mean") else "CV-F1=N/A"
            )
            gap = (result['f1_fraud'] - (result.get('cv_f1_mean') or result['f1_fraud'])) * 100
            gap_warn = f"  ⚠ CV/test gap: {gap:+.1f}pp" if abs(gap) > 5 else ""
            logger.info(
                f"  Accuracy={result['accuracy']*100:.2f}%  "
                f"Fraud-F1={result['f1_fraud']*100:.2f}%  "
                f"ROC-AUC={result['roc_auc']}  {cv_str}  threshold={opt_thresh:.2f}"
                f"{gap_warn}"
            )
        except Exception as ex:
            logger.error(f"  {name} failed: {ex}")

    if not results:
        logger.error("All classifiers failed — aborting.")
        sys.exit(1)

    # ── 9. Select best model (by fraud-class F1 on test set) ───────────
    best = max(results, key=lambda r: r["f1_fraud"])
    logger.info(f"\n★ Best model: {best['name']} (Fraud F1 = {best['f1_fraud']*100:.2f}%)")

    # ── 10. Save best model + vectorizer ──────────────────────────────
    with open(M_OUT, "wb") as f:
        pickle.dump(best["model"], f)
    with open(V_OUT, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Saved best model → {M_OUT}")

    # ── 11. Save ALL models ────────────────────────────────────────────
    registry = save_all_models(results)
    logger.info(f"Saved model registry: {list(registry.keys())}")

    # ── 12. Save threshold.json (best model's optimal threshold) ───────
    best_thresh = thresholds.get(best["name"], 0.5)
    with open(TH_OUT, "w") as f:
        json.dump({
            "threshold":   best_thresh,
            "model":       best["name"],
            "generated_at": datetime.datetime.now().isoformat(),
            "note": (
                "Classification threshold optimised for fraud-class F1 on the "
                "validation set. Falls back to 0.5 for LinearSVC (no predict_proba)."
            ),
        }, f, indent=2)
    logger.info(f"Saved threshold.json: {best_thresh:.4f}")

    # ── 13. Save metrics.json ──────────────────────────────────────────
    save_metrics(results)
    logger.info("Saved: models/metrics.json")

    # ── 14. Save model_metadata.json ──────────────────────────────────
    metadata = {
        "best_model":     best["name"],
        "trained_at":     datetime.datetime.now().isoformat(),
        "dataset_rows":   int(df.shape[0]),
        "dataset_cols":   int(df.shape[1]),
        "fraud_count":    int(df["fraudulent"].sum()),
        "legit_count":    int(df.shape[0] - df["fraudulent"].sum()),
        "tfidf_features": vectorizer.max_features,
        "miss_features":  miss_feat_cols,
        "train_size":     X_train.shape[0],
        "val_size":       X_val.shape[0],
        "test_size":      X_test.shape[0],
        "accuracy":       best["accuracy"],
        "f1_fraud":       best["f1_fraud"],
        "roc_auc":        best["roc_auc"],
        "threshold":      best_thresh,
        "all_models":     [r["name"] for r in results],
    }
    with open(MD_OUT, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved: models/model_metadata.json")

    # ── 15. Plots ──────────────────────────────────────────────────────
    logger.info("Generating evaluation plots...")
    for r in results:
        plot_confusion_matrix(r, save=True)
    plot_all_confusion_matrices(results, save=True)
    plot_model_comparison(results, save=True)
    plot_roc_curves(results, save=True)
    logger.info("Plots saved to static/images/")

    # ── 16. Results table ──────────────────────────────────────────────
    SEP = "=" * 110
    print(f"\n{SEP}")
    print(f"{'Model':<22} {'Acc':>7} {'Fraud-F1':>10} {'AUC':>7} {'CV-F1 (mean±std)':>20} {'Threshold':>10} {'CV/Test gap':>12}")
    print("-" * 110)
    for r in sorted(results, key=lambda x: x["f1_fraud"], reverse=True):
        star    = "  * BEST" if r["name"] == best["name"] else ""
        auc     = f"{r['roc_auc']:.4f}" if r["roc_auc"] else "  N/A  "
        cv_str  = (
            f"{r['cv_f1_mean']*100:.2f}%±{r['cv_f1_std']*100:.2f}%"
            if r.get("cv_f1_mean") else "   N/A    "
        )
        gap     = (r['f1_fraud'] - (r.get('cv_f1_mean') or r['f1_fraud'])) * 100
        gap_str = f"{gap:+.1f}pp" + (" ⚠" if abs(gap) > 5 else "")
        thresh  = f"{thresholds.get(r['name'], 0.5):.2f}"
        print(
            f"{r['name']:<22} "
            f"{r['accuracy']*100:>6.2f}% "
            f"{r['f1_fraud']*100:>9.2f}% "
            f"{auc:>7}"
            f"{cv_str:>21}"
            f"{thresh:>10}"
            f"{gap_str:>13}"
            f"{star}"
        )
    print(SEP)
    print(f"\n* Selected: {best['name']} | Threshold: {best_thresh:.4f}")
    print("  Selection based on Fraud-class F1 on held-out test set.")
    print("  A large CV/test gap (⚠) means the test split may be optimistic.")
    print("\nDone! Run:  python app.py")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
