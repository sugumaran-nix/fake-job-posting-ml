"""
utils/explainer.py
==================
Explainability engine for the Fake Job Posting classifier.

Approach: Coefficient × TF-IDF token attribution
-------------------------------------------------
For any linear model (LogisticRegression, LinearSVC, SGDClassifier) the
contribution of each token in a document to the final decision score is:

    impact(token) = tfidf_weight(token) × model_coefficient(token)

  positive impact  →  pushes prediction toward FRAUD
  negative impact  →  pushes prediction toward LEGITIMATE

This gives exact, deterministic, sub-millisecond attribution with no
extra dependencies (no LIME, no SHAP).

Returns
-------
explain(text, vectorizer, model) → dict:
  top_fraud_words   : list[(word, impact_pct)]   high-fraud tokens
  top_legit_words   : list[(word, impact_pct)]   high-legit tokens
  highlighted_html  : str  — description with <mark> tags for fraud words
  fraud_patterns    : list[dict]  — named fraud pattern matches
  reasons           : list[str]  — plain-English explanation bullets
  model_name        : str
  decision_score    : float
"""

import re
import html as html_lib
from typing import Optional

import numpy as np

# ── Fraud pattern definitions (rule-based, applied to raw text) ────────
FRAUD_PATTERNS = [
    {
        "id":      "registration_fee",
        "label":   "Registration / Joining Fee",
        "icon":    "💸",
        "severity":"high",
        "regex":   r"\b(registration\s+fee|joining\s+fee|pay\s+to\s+(join|start|register)|starter\s+kit|one.?time\s+fee)\b",
        "reason":  "Legitimate employers never ask candidates to pay fees to apply or start work.",
    },
    {
        "id":      "guaranteed_income",
        "label":   "Guaranteed / Unrealistic Income",
        "icon":    "💰",
        "severity":"high",
        "regex":   r"\b(guaranteed\s+(income|salary|earn|pay|money)|earn\s+up\s+to|₹[\d,]+.{0,10}(guaranteed|month|day)|lakh.{0,10}month|crore.{0,10}year)\b",
        "reason":  "Promises of guaranteed high income with no experience are a classic fraud signal.",
    },
    {
        "id":      "no_experience",
        "label":   "No Experience / Qualification Required",
        "icon":    "🎯",
        "severity":"medium",
        "regex":   r"\b(no\s+(experience|qualification|degree|skill)\s+(needed|required|necessary)|freshers?\s+welcome|anyone\s+can\s+(apply|join)|no\s+interview)\b",
        "reason":  "High-paying jobs requiring no skills are almost always fraudulent.",
    },
    {
        "id":      "personal_documents",
        "label":   "Request for Personal Documents",
        "icon":    "🪪",
        "severity":"high",
        "regex":   r"\b(aadhaar|pan\s+card|bank\s+account\s+(details|number)|passport\s+copy|send\s+your\s+(documents|id|photo)|submit\s+(id|proof))\b",
        "reason":  "Asking for Aadhaar, PAN, or bank details before an offer letter is identity fraud.",
    },
    {
        "id":      "work_from_home_easy",
        "label":   "Easy Work-From-Home Scheme",
        "icon":    "🏠",
        "severity":"medium",
        "regex":   r"\b(work\s+from\s+home.{0,30}(easy|simple|just|only)|earn.{0,20}home|home.?based\s+(job|work|earn)|part.?time.{0,20}earn|sitting\s+at\s+home)\b",
        "reason":  "Vague work-from-home offers with high pay and no required skills are scam indicators.",
    },
    {
        "id":      "urgent_hiring",
        "label":   "Urgent / Immediate Hiring",
        "icon":    "⚡",
        "severity":"low",
        "regex":   r"\b(urgent(ly)?\s+(hiring|required|needed|vacancy)|immediate\s+(joining|vacancy|opening)|apply\s+now.{0,20}(limited|hurry|last)|only\s+\d+\s+(seats?|spots?)\s+left)\b",
        "reason":  "Artificial urgency is a pressure tactic used in fraudulent postings.",
    },
    {
        "id":      "upfront_payment",
        "label":   "Upfront Payment / Investment",
        "icon":    "🔴",
        "severity":"high",
        "regex":   r"\b(pay\s+(first|upfront|advance|deposit)|refundable\s+deposit|security\s+deposit|training\s+(fee|cost|charge)|material\s+(fee|charge|cost))\b",
        "reason":  "Any upfront payment request — even framed as refundable — is a fraud tactic.",
    },
    {
        "id":      "mlm_network",
        "label":   "MLM / Network Marketing",
        "icon":    "🔺",
        "severity":"high",
        "regex":   r"\b(network\s+marketing|multi.?level|mlm|pyramid|refer\s+and\s+earn|recruit\s+others|downline|passive\s+income\s+(from|through)\s+(refer|recruit))\b",
        "reason":  "Multi-level marketing structures disguised as jobs are a well-known fraud category.",
    },
    {
        "id":      "vague_description",
        "label":   "Vague or Generic Job Description",
        "icon":    "📋",
        "severity":"low",
        "regex":   r"\b(data\s+entry\s+work\s+from\s+home|online\s+(typing|copy.?paste)|simple\s+(online\s+)?work|earn\s+by\s+(typing|clicking|liking))\b",
        "reason":  "Extremely vague job tasks with no real skill requirements indicate a fake posting.",
    },
]

# ── Severity colour mapping ────────────────────────────────────────────
SEVERITY_COLOUR = {
    "high":   "#DC2626",   # red
    "medium": "#D97706",   # amber
    "low":    "#6B7280",   # grey
}

SEVERITY_BG = {
    "high":   "#FEF2F2",
    "medium": "#FFFBEB",
    "low":    "#F9FAFB",
}

SEVERITY_BORDER = {
    "high":   "#FECACA",
    "medium": "#FDE68A",
    "low":    "#E5E7EB",
}


# ══════════════════════════════════════════════════════════════════════
#  CORE EXPLAINABILITY FUNCTION
# ══════════════════════════════════════════════════════════════════════

def explain(
    raw_text: str,
    vectorizer,
    model,
    top_n: int = 8,
) -> dict:
    """
    Compute token-level attribution and pattern-based explanations.

    Parameters
    ----------
    raw_text   : str   — original (un-preprocessed) combined job text
    vectorizer : fitted TfidfVectorizer
    model      : fitted sklearn linear classifier
    top_n      : int   — how many top words to return per direction

    Returns
    -------
    dict — see module docstring
    """
    # ── 1. Token attribution ──────────────────────────────────────────
    Xv          = vectorizer.transform([_light_clean(raw_text)])
    feat_names  = vectorizer.get_feature_names_out()
    coefs       = _get_coefs(model)          # shape (n_features,) for binary

    cx           = Xv.tocsr()
    col_indices  = cx.indices
    tfidf_vals   = cx.data

    n_coefs = len(coefs)
    token_scores = []
    for col_idx, tfidf_val in zip(col_indices, tfidf_vals):
        word = feat_names[col_idx]
        # Safe indexing: skip if coefs array is shorter than vocab (fallback model)
        if n_coefs == 0 or col_idx >= n_coefs:
            continue
        coef   = float(coefs[col_idx])
        impact = float(tfidf_val) * coef
        token_scores.append({"word": word, "impact": impact,
                              "coef": coef, "tfidf": float(tfidf_val)})

    # Separate fraud (positive) vs legit (negative) tokens
    fraud_tokens = sorted([t for t in token_scores if t["impact"] > 0],
                           key=lambda x: x["impact"], reverse=True)
    legit_tokens = sorted([t for t in token_scores if t["impact"] < 0],
                           key=lambda x: x["impact"])

    # Normalise to percentages relative to max absolute impact
    max_abs = max((abs(t["impact"]) for t in token_scores), default=1.0)
    max_abs = max(max_abs, 1e-9)

    def to_pct(tokens, n):
        out = []
        for t in tokens[:n]:
            pct = round(abs(t["impact"]) / max_abs * 100, 1)
            out.append({"word": t["word"], "pct": pct, "impact": round(t["impact"], 4)})
        return out

    top_fraud_words = to_pct(fraud_tokens, top_n)
    top_legit_words = to_pct(legit_tokens, top_n)

    # ── 2. Pattern matching on raw text ───────────────────────────────
    fraud_patterns_found = _match_patterns(raw_text)

    # ── 3. Decision score ─────────────────────────────────────────────
    decision_score = float(_decision_score(model, Xv))

    # ── 4. Word-highlighted HTML ──────────────────────────────────────
    fraud_word_set = {t["word"] for t in fraud_tokens[:top_n]}
    legit_word_set = {t["word"] for t in legit_tokens[:top_n]}
    highlighted    = _highlight_html(raw_text, fraud_word_set, legit_word_set)

    # ── 5. Plain-English reasons ──────────────────────────────────────
    reasons = _build_reasons(fraud_tokens, legit_tokens, fraud_patterns_found, decision_score)

    # ── 6. Model name ─────────────────────────────────────────────────
    model_name = type(model).__name__
    _MODEL_DISPLAY = {
        "LogisticRegression": "Logistic Regression",
        "LinearSVC":          "Linear SVM (LinearSVC)",
        "RandomForestClassifier": "Random Forest",
        "MultinomialNB":      "Naive Bayes",
        "SGDClassifier":      "SGD Classifier",
    }
    model_display = _MODEL_DISPLAY.get(model_name, model_name)

    return {
        "top_fraud_words":    top_fraud_words,
        "top_legit_words":    top_legit_words,
        "highlighted_html":   highlighted,
        "fraud_patterns":     fraud_patterns_found,
        "reasons":            reasons,
        "model_name":         model_display,
        "decision_score":     decision_score,
        "n_fraud_tokens":     len(fraud_tokens),
        "n_legit_tokens":     len(legit_tokens),
    }


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def _light_clean(text: str) -> str:
    """Minimal cleaning for vectorizer.transform() — preserves more tokens
    than the full preprocess pipeline so highlighting maps back to raw text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _get_coefs(model) -> np.ndarray:
    """
    Extract 1-D coefficient array regardless of model type.

    Model-specific handling:
      LogisticRegression / LinearSVC : coef_ shape (1, n_features) -> (n_features,)
      MultinomialNB                  : feature_log_prob_ shape (n_classes, n_features)
                                       -> diff of class-1 vs class-0 log-probs gives
                                          a signed "fraud signal" per feature
      RandomForest / trees           : feature_importances_ (unsigned proxy)
      Unknown                        : empty array; safe indexing handles it
    """
    # Linear models (LR, LinearSVC, SGD, ...)
    if hasattr(model, "coef_"):
        c = model.coef_
        if c.ndim == 2:
            return c[0]          # binary: shape (1, n) -> (n,)
        return c

    # Naive Bayes: use log-prob difference as signed fraud signal
    # feature_log_prob_ shape: (n_classes, n_features)
    # class 1 = fraud, class 0 = legit
    # positive diff -> word more associated with fraud
    if hasattr(model, "feature_log_prob_"):
        flp = model.feature_log_prob_
        if flp.shape[0] >= 2:
            return flp[1] - flp[0]   # shape (n_features,)
        return flp[0]

    # Tree / ensemble models: feature importance is unsigned (proxy)
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_

    # Fallback: empty array; safe indexing below will skip attribution
    return np.zeros(0)


def _decision_score(model, Xv) -> float:
    """Raw decision score for the positive (fraud) class."""
    if hasattr(model, "decision_function"):
        return float(model.decision_function(Xv)[0])
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(Xv)[0][1])
    return 0.0


def _match_patterns(text: str) -> list:
    """Run all fraud pattern regexes against raw lowercase text."""
    t = text.lower()
    found = []
    for pat in FRAUD_PATTERNS:
        m = re.search(pat["regex"], t, re.IGNORECASE)
        if m:
            found.append({
                "id":       pat["id"],
                "label":    pat["label"],
                "icon":     pat["icon"],
                "severity": pat["severity"],
                "reason":   pat["reason"],
                "matched":  m.group(0).strip(),
                "colour":   SEVERITY_COLOUR[pat["severity"]],
                "bg":       SEVERITY_BG[pat["severity"]],
                "border":   SEVERITY_BORDER[pat["severity"]],
            })
    # Sort: high first, then medium, then low
    order = {"high": 0, "medium": 1, "low": 2}
    found.sort(key=lambda x: order[x["severity"]])
    return found


def _highlight_html(raw_text: str, fraud_words: set, legit_words: set,
                    max_chars: int = 800) -> str:
    """
    Return HTML of raw_text with fraud words wrapped in
    <mark class="hw-fraud"> and legit words in <mark class="hw-legit">.
    Works at the word level on the original (un-lowercased) text.
    Limits output to max_chars to keep the UI manageable.
    """
    snippet = raw_text[:max_chars]
    escaped = html_lib.escape(snippet)

    # Build a combined set with direction labels
    word_map = {}
    for w in fraud_words:
        word_map[w] = "hw-fraud"
    for w in legit_words:
        if w not in word_map:               # fraud takes priority
            word_map[w] = "hw-legit"

    if not word_map:
        return escaped + ("…" if len(raw_text) > max_chars else "")

    # Sort longest first to avoid partial-word replacement bugs
    sorted_words = sorted(word_map.keys(), key=len, reverse=True)

    # Replace whole-word matches only (case-insensitive)
    for word in sorted_words:
        css_class = word_map[word]
        pattern   = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        escaped   = pattern.sub(
            lambda m: f'<mark class="{css_class}">{m.group(0)}</mark>',
            escaped
        )

    if len(raw_text) > max_chars:
        escaped += '<span class="hw-ellipsis">…</span>'

    return escaped


def _build_reasons(fraud_tokens: list, legit_tokens: list,
                   patterns: list, decision_score: float) -> list:
    """
    Generate 3–6 plain-English bullet-point reasons for the prediction.
    """
    reasons = []

    # From patterns (most reliable, human-readable)
    for p in patterns[:3]:
        reasons.append({
            "icon":  p["icon"],
            "text":  f"{p['label']}: {p['reason']}",
            "type":  "pattern",
            "sev":   p["severity"],
        })

    # From top fraud tokens (if no pattern coverage)
    if len(reasons) < 2 and fraud_tokens:
        top_words = [t["word"] for t in fraud_tokens[:4]]
        reasons.append({
            "icon": "🔍",
            "text": f"High-fraud vocabulary detected: «{', '.join(top_words)}» — "
                    "these terms appear disproportionately in fraudulent postings.",
            "type": "token",
            "sev":  "medium",
        })

    # From top legit tokens (for legitimate predictions)
    if decision_score < 0 and legit_tokens:
        top_words = [t["word"] for t in legit_tokens[:4]]
        reasons.append({
            "icon": "✅",
            "text": f"Professional vocabulary present: «{', '.join(top_words)}» — "
                    "these terms are strongly associated with genuine job postings.",
            "type": "token",
            "sev":  "low",
        })

    # Decision score magnitude
    abs_score = abs(decision_score)
    if abs_score > 2.0:
        reasons.append({
            "icon": "📊",
            "text": f"Model decision boundary crossed with high margin "
                    f"(score {decision_score:+.2f}) — prediction is confident.",
            "type": "score",
            "sev":  "low",
        })
    elif abs_score < 0.5:
        reasons.append({
            "icon": "⚠️",
            "text": f"Model decision score is near the boundary "
                    f"(score {decision_score:+.2f}) — treat this result with caution.",
            "type": "score",
            "sev":  "medium",
        })

    return reasons[:6]
