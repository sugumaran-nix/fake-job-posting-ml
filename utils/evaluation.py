"""
utils/evaluation.py — JobGuard v7
====================================
Bug fixes from v6:
  - FIX: load_model_by_name() now resolves relative paths against BASE_DIR.
          The pre-trained model_registry.json stores relative paths
          ("models/linear_svm.pkl"). When gunicorn is launched from a
          directory other than the project root, os.path.exists(relative)
          returned False and model switching silently fell back to model.pkl.
          Now both relative and absolute paths are handled correctly.

  - FIX: save_all_models() saves absolute paths so newly trained models
          never have the relative-path problem.

  - FIX: CV scoring now uses a cloned estimator so cross_val_score runs
          on an unfitted estimator as intended, giving a cleaner estimate.
"""

import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
    classification_report,
)
from sklearn.model_selection import cross_val_score

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR   = os.path.join(BASE_DIR, "static", "images")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────
PALETTE = {
    "blue":   "#1B9FD4",
    "dark":   "#333333",
    "grey":   "#4A4757",
    "fraud":  "#DC2626",
    "legit":  "#16A34A",
    "lav":    "#E8EBF5",
    "border": "#D6DAF0",
}
MODEL_COLOURS = ["#1B9FD4", "#16A34A", "#D97706", "#7C3AED"]


# ══════════════════════════════════════════════════════════════════════
#  METRIC COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(name: str, clf, X_train, X_test, y_train, y_test,
                    cv_folds: int = 5) -> dict:
    """
    Fit classifier, evaluate on test set, and cross-validate on training set.

    FIX: CV uses clone(clf) so it starts from an unfitted state — more
    representative of true generalisation than fitting on the same data twice.
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_score = None
    auc     = None
    try:
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X_test)
        if y_score is not None:
            auc = round(float(roc_auc_score(y_test, y_score)), 4)
    except Exception:
        pass

    # FIX: clone before CV so we measure generalisation from scratch
    cv_f1_mean = cv_f1_std = None
    try:
        cv_scores = cross_val_score(
            clone(clf), X_train, y_train,
            cv=cv_folds, scoring="f1", n_jobs=-1,
        )
        cv_f1_mean = round(float(cv_scores.mean()), 4)
        cv_f1_std  = round(float(cv_scores.std()),  4)
    except Exception:
        pass

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "name":               name,
        "model":              clf,
        "accuracy":           round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted":    round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted":        round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "precision_fraud":    round(float(precision_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0)), 4),
        "recall_fraud":       round(float(recall_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0)), 4),
        "f1_fraud":           round(float(f1_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0)), 4),
        "roc_auc":            auc,
        "cv_f1_mean":         cv_f1_mean,
        "cv_f1_std":          cv_f1_std,
        "confusion_matrix":   cm,
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Legitimate", "Fraudulent"]
        ),
        "_y_test":            y_test,
        "_y_score":           y_score,
    }


# ══════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(result: dict, save: bool = True) -> str:
    cm   = np.array(result["confusion_matrix"])
    name = result["name"]
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("white")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        linewidths=1, linecolor=PALETTE["border"],
        ax=ax, cbar=False, annot_kws={"size": 16, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=10, color=PALETTE["grey"])
    ax.set_ylabel("True Label",      fontsize=10, labelpad=10, color=PALETTE["grey"])
    ax.set_title(f"Confusion Matrix\n{name}", fontsize=12, fontweight="bold",
                 color=PALETTE["dark"], pad=14)
    ax.set_xticklabels(["Legitimate", "Fraudulent"], fontsize=9, color=PALETTE["grey"])
    ax.set_yticklabels(["Legitimate", "Fraudulent"], fontsize=9, color=PALETTE["grey"], rotation=0)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.78, labels[i][j], ha="center", va="center",
                    fontsize=8, color="#888", style="italic")
    plt.tight_layout()
    if save:
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(IMAGES_DIR, f"cm_{slug}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path
    plt.close(fig)
    return ""


def plot_roc_curves(results: list, save: bool = True) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFBFF")
    plotted = 0
    for result, colour in zip(results, MODEL_COLOURS):
        y_test  = result.get("_y_test")
        y_score = result.get("_y_score")
        auc     = result.get("roc_auc")
        if y_test is None or y_score is None or auc is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax.plot(fpr, tpr, color=colour, lw=2.5,
                label=f"{result['name']}  (AUC = {auc:.4f})")
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return ""
    ax.plot([0, 1], [0, 1], linestyle="--", color="#AAAAAA", lw=1.5,
            label="Random baseline (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate", fontsize=11, color=PALETTE["grey"])
    ax.set_ylabel("True Positive Rate",  fontsize=11, color=PALETTE["grey"])
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold",
                 color=PALETTE["dark"], pad=16)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        path = os.path.join(IMAGES_DIR, "roc_curves.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path
    plt.close(fig)
    return ""


def plot_model_comparison(results: list, save: bool = True) -> str:
    metrics   = [
        ("accuracy",        "Accuracy"),
        ("f1_weighted",     "F1 (Weighted)"),
        ("precision_fraud", "Precision (Fraud)"),
        ("recall_fraud",    "Recall (Fraud)"),
        ("f1_fraud",        "F1 (Fraud class)"),
    ]
    names    = [r["name"] for r in results]
    n_models = len(names)
    n_met    = len(metrics)
    x        = np.arange(n_met)
    width    = 0.18
    offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width
    fig, ax  = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFBFF")
    for result, colour, offset in zip(results, MODEL_COLOURS[:n_models], offsets):
        values = [result.get(key, 0) * 100 for key, _ in metrics]
        bars   = ax.bar(x + offset, values, width, label=result["name"],
                        color=colour, alpha=0.88, edgecolor="white", linewidth=1.2, zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color=PALETTE["dark"], fontweight="600")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], fontsize=10, color=PALETTE["grey"])
    ax.set_ylabel("Score (%)", fontsize=10, color=PALETTE["grey"])
    ax.set_ylim(0, 110)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold",
                 color=PALETTE["dark"], pad=16)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    if save:
        path = os.path.join(IMAGES_DIR, "model_comparison.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path
    plt.close(fig)
    return ""


def plot_all_confusion_matrices(results: list, save: bool = True) -> str:
    n    = len(results)
    cols = min(n, 2)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.2))
    fig.patch.set_facecolor("white")
    axes = np.array(axes).flatten()
    for ax, result in zip(axes, results):
        cm = np.array(result["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    linewidths=1, linecolor=PALETTE["border"],
                    ax=ax, cbar=False, annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(result["name"], fontsize=11, fontweight="bold", color=PALETTE["dark"])
        ax.set_xlabel("Predicted", fontsize=9, color=PALETTE["grey"])
        ax.set_ylabel("Actual",    fontsize=9, color=PALETTE["grey"])
        ax.set_xticklabels(["Legit", "Fraud"], fontsize=8)
        ax.set_yticklabels(["Legit", "Fraud"], fontsize=8, rotation=0)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Confusion Matrices — All Models", fontsize=14,
                 fontweight="bold", color=PALETTE["dark"], y=1.02)
    plt.tight_layout()
    if save:
        path = os.path.join(IMAGES_DIR, "all_confusion_matrices.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path
    plt.close(fig)
    return ""


# ══════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════

def save_all_models(results: list) -> dict:
    """
    Save every trained model as an absolute path in the registry.
    FIX: absolute paths prevent model-switching breakage under gunicorn
    when launched from a non-project-root working directory.
    """
    registry = {}
    for r in results:
        slug = r["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(MODELS_DIR, f"{slug}.pkl")   # always absolute
        with open(path, "wb") as f:
            pickle.dump(r["model"], f)
        registry[r["name"]] = path   # store absolute path

    reg_path = os.path.join(MODELS_DIR, "model_registry.json")
    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2)
    return registry


def load_model_registry() -> dict:
    reg_path = os.path.join(MODELS_DIR, "model_registry.json")
    if not os.path.exists(reg_path):
        return {}
    with open(reg_path) as f:
        return json.load(f)


def load_model_by_name(name: str):
    """
    FIX: Resolves relative paths in old registry files against BASE_DIR.
    Pre-trained model_registry.json stores paths like "models/linear_svm.pkl"
    (relative). This function handles both old (relative) and new (absolute).
    """
    registry = load_model_registry()
    path = registry.get(name)
    if not path:
        return None

    # Resolve relative path from project root if needed
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)

    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════
#  METRICS JSON
# ══════════════════════════════════════════════════════════════════════

def save_metrics(results: list) -> None:
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    _EXCLUDE = {"model", "_y_test", "_y_score"}
    serialisable = [
        {k: v for k, v in r.items() if k not in _EXCLUDE}
        for r in results
    ]
    with open(METRICS_PATH, "w") as f:
        json.dump(serialisable, f, indent=2)


def load_metrics() -> list:
    if not os.path.exists(METRICS_PATH):
        return []
    with open(METRICS_PATH) as f:
        return json.load(f)
