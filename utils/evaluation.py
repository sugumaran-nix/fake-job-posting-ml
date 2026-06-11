"""
utils/evaluation.py
====================
Model evaluation helpers: per-class metrics, confusion matrix plots,
ROC curves, and model-comparison bar charts.

Upgrades from v1:
  ✅ plot_roc_curves()  — multi-model ROC curve overlay saved to static/images/
  ✅ Cross-validation score stored in compute_metrics() result dict
  ✅ save_all_models() / load_model_registry() — persist every trained model
  ✅ All plots consistent with site CSS colour palette
"""

import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.model_selection import cross_val_score

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR    = os.path.join(BASE_DIR, "static", "images")
METRICS_PATH  = os.path.join(BASE_DIR, "models", "metrics.json")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Colour palette (matches site CSS variables) ────────────────────────
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
    Fit classifier and return a comprehensive metrics dict.

    NEW: stores y_test and y_score for ROC curve plotting,
         and cross-validation F1 mean ± std.

    Returns
    -------
    dict with keys:
        name, model, accuracy, precision_weighted, recall_weighted,
        f1_weighted, precision_fraud, recall_fraud, f1_fraud,
        roc_auc, cv_f1_mean, cv_f1_std,
        confusion_matrix, classification_report,
        _y_test (raw array for ROC), _y_score (raw array for ROC)
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # ── ROC-AUC score and raw scores for curve plotting ────────────────
    y_score = None
    auc = None
    try:
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X_test)
        if y_score is not None:
            auc = round(float(roc_auc_score(y_test, y_score)), 4)
    except Exception:
        pass

    # ── Cross-validation F1 on training set (fraud class) ─────────────
    cv_f1_mean = cv_f1_std = None
    try:
        cv_scores = cross_val_score(
            clf, X_train, y_train,
            cv=cv_folds, scoring="f1", n_jobs=-1
        )
        cv_f1_mean = round(float(cv_scores.mean()), 4)
        cv_f1_std  = round(float(cv_scores.std()),  4)
    except Exception:
        pass

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "name":               name,
        "model":              clf,
        # Weighted averages (overall)
        "accuracy":           round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted":    round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted":        round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        # Per-class: fraud (class = 1)
        "precision_fraud":    round(float(precision_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0)), 4),
        "recall_fraud":       round(float(recall_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0)), 4),
        "f1_fraud":           round(float(f1_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0)), 4),
        "roc_auc":            auc,
        # Cross-validation
        "cv_f1_mean":         cv_f1_mean,
        "cv_f1_std":          cv_f1_std,
        "confusion_matrix":   cm,
        "classification_report": classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"]),
        # Raw arrays for ROC curve (excluded from JSON serialisation)
        "_y_test":            y_test,
        "_y_score":           y_score,
    }


# ══════════════════════════════════════════════════════════════════════
#  CONFUSION MATRIX PLOT
# ══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(result: dict, save: bool = True) -> str:
    cm   = np.array(result["confusion_matrix"])
    name = result["name"]

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        linewidths=1, linecolor=PALETTE["border"],
        ax=ax, cbar=False,
        annot_kws={"size": 16, "weight": "bold"},
    )

    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=10, color=PALETTE["grey"])
    ax.set_ylabel("True Label",      fontsize=10, labelpad=10, color=PALETTE["grey"])
    ax.set_title(f"Confusion Matrix\n{name}", fontsize=12, fontweight="bold",
                 color=PALETTE["dark"], pad=14)
    ax.set_xticklabels(["Legitimate", "Fraudulent"], fontsize=9, color=PALETTE["grey"])
    ax.set_yticklabels(["Legitimate", "Fraudulent"], fontsize=9, color=PALETTE["grey"],
                       rotation=0)

    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.78, labels[i][j],
                    ha="center", va="center", fontsize=8,
                    color="#888", style="italic")

    plt.tight_layout()

    if save:
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(IMAGES_DIR, f"cm_{slug}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path

    plt.close(fig)
    return ""


# ══════════════════════════════════════════════════════════════════════
#  ROC CURVE PLOT  ← NEW
# ══════════════════════════════════════════════════════════════════════

def plot_roc_curves(results: list, save: bool = True) -> str:
    """
    Overlay ROC curves for all models that support probability/decision scores.

    Parameters
    ----------
    results : list[dict]  from compute_metrics()
    save    : if True, write PNG to static/images/roc_curves.png

    Returns
    -------
    str  — absolute path to saved PNG, or "" if save=False
    """
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

    # Diagonal (random classifier baseline)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#AAAAAA",
            lw=1.5, label="Random baseline (AUC = 0.50)")

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


# ══════════════════════════════════════════════════════════════════════
#  MODEL COMPARISON CHART
# ══════════════════════════════════════════════════════════════════════

def plot_model_comparison(results: list, save: bool = True) -> str:
    metrics = [
        ("accuracy",        "Accuracy"),
        ("f1_weighted",     "F1 (Weighted)"),
        ("precision_fraud", "Precision (Fraud)"),
        ("recall_fraud",    "Recall (Fraud)"),
        ("f1_fraud",        "F1 (Fraud class)"),
    ]

    names     = [r["name"] for r in results]
    n_models  = len(names)
    n_metrics = len(metrics)
    x         = np.arange(n_metrics)
    width     = 0.18
    offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFBFF")

    for idx, (result, colour, offset) in enumerate(
        zip(results, MODEL_COLOURS[:n_models], offsets)
    ):
        values = [result.get(key, 0) * 100 for key, _ in metrics]
        bars   = ax.bar(x + offset, values, width, label=result["name"],
                        color=colour, alpha=0.88, edgecolor="white", linewidth=1.2,
                        zorder=3)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=7.5, color=PALETTE["dark"], fontweight="600",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], fontsize=10,
                       color=PALETTE["grey"])
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


# ══════════════════════════════════════════════════════════════════════
#  ALL CONFUSION MATRICES IN ONE GRID
# ══════════════════════════════════════════════════════════════════════

def plot_all_confusion_matrices(results: list, save: bool = True) -> str:
    n    = len(results)
    cols = min(n, 2)
    rows = (n + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.2))
    fig.patch.set_facecolor("white")
    axes = np.array(axes).flatten()

    for ax, result in zip(axes, results):
        cm = np.array(result["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            linewidths=1, linecolor=PALETTE["border"],
            ax=ax, cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(result["name"], fontsize=11, fontweight="bold",
                     color=PALETTE["dark"])
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
#  MODEL REGISTRY  ← NEW
# ══════════════════════════════════════════════════════════════════════

def save_all_models(results: list) -> dict:
    """
    Save every trained model to models/<slug>.pkl so any can be loaded later.

    Returns
    -------
    dict  mapping model name → file path
    """
    registry = {}
    for r in results:
        slug = r["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(MODELS_DIR, f"{slug}.pkl")
        with open(path, "wb") as f:
            pickle.dump(r["model"], f)
        registry[r["name"]] = path
    # Persist registry manifest
    reg_path = os.path.join(MODELS_DIR, "model_registry.json")
    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2)
    return registry


def load_model_registry() -> dict:
    """
    Load model registry manifest.

    Returns
    -------
    dict  mapping model name → file path  (empty dict if not yet trained)
    """
    reg_path = os.path.join(MODELS_DIR, "model_registry.json")
    if not os.path.exists(reg_path):
        return {}
    with open(reg_path) as f:
        return json.load(f)


def load_model_by_name(name: str):
    """
    Load a specific model from the registry by name.

    Parameters
    ----------
    name : str  — exact model name (e.g. "Random Forest")

    Returns
    -------
    fitted sklearn estimator, or None if not found
    """
    registry = load_model_registry()
    path = registry.get(name)
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════
#  PERSIST / LOAD METRICS JSON
# ══════════════════════════════════════════════════════════════════════

def save_metrics(results: list) -> None:
    """Serialise metrics (excluding model object + raw arrays) to metrics.json."""
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    # Keys to exclude from JSON (non-serialisable or internal)
    _EXCLUDE = {"model", "_y_test", "_y_score"}
    serialisable = [
        {k: v for k, v in r.items() if k not in _EXCLUDE}
        for r in results
    ]
    with open(METRICS_PATH, "w") as f:
        json.dump(serialisable, f, indent=2)


def load_metrics() -> list:
    """Load previously saved metrics. Returns [] if file not found."""
    if not os.path.exists(METRICS_PATH):
        return []
    with open(METRICS_PATH) as f:
        return json.load(f)
