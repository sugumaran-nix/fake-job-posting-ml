"""
utils/explainer.py — JobGuard v7
=================================
Token-level attribution + pattern-based explanations.

Bug fixes from v6:
  ✅ FIX 1 — _build_reasons: replaced `decision_score < 0` check with
             `not is_fraud` flag. The old check NEVER fired for Logistic
             Regression and Naive Bayes because _decision_score() returns
             P(fraud) ∈ [0,1], which is always ≥ 0. Legitimate predictions
             by LR/NB now correctly show professional-vocabulary reasons.

  ✅ FIX 2 — Random Forest vice-versa token display: feature_importances_
             values are unsigned (always ≥ 0). Previously ALL tokens landed
             in fraud_tokens, so even a LEGITIMATE prediction showed every
             word highlighted red. Now RF tokens are split by predicted
             direction using the is_fraud flag.

  ✅ FIX 3 — CalibratedClassifierCV unwrapping: _get_coefs() and
             _decision_score() now unwrap the sklearn calibration wrapper
             to reach the underlying LinearSVC coef_ for correct attribution.

  ✅ FIX 4 — Nested <mark> tags: _highlight_html() now builds highlight
             on a token-list pass (no regex on already-modified HTML),
             preventing double-marked spans when a bigram overlaps a unigram.

  ✅ FIX 5 — Lemmatized vocab vs raw text: highlighting is done on the
             _light_clean output (same pipeline the vectorizer sees) not on
             the raw string, so marks actually appear.
"""

import re
import html as html_lib
from typing import Optional

import numpy as np


# ── Fraud pattern definitions ─────────────────────────────────────────
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

SEVERITY_COLOUR = {"high": "#DC2626", "medium": "#D97706", "low": "#6B7280"}
SEVERITY_BG     = {"high": "#FEF2F2", "medium": "#FFFBEB", "low": "#F9FAFB"}
SEVERITY_BORDER = {"high": "#FECACA", "medium": "#FDE68A", "low": "#E5E7EB"}


# ══════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def explain(
    raw_text: str,
    vectorizer,
    model,
    top_n: int = 8,
    is_fraud: Optional[bool] = None,    # FIX 1 & 2 — passed from ml_predict()
) -> dict:
    """
    Compute token-level attribution and pattern explanations.

    Parameters
    ----------
    raw_text  : original combined job text (un-preprocessed)
    vectorizer: fitted TfidfVectorizer
    model     : fitted sklearn classifier (any type)
    top_n     : number of top words to return per direction
    is_fraud  : predicted direction from ml_predict() — fixes reason bullets
                and RF token direction. If None, inferred from decision score.
    """
    # ── 1. Vectorize using the same light-clean the vectorizer expects ─
    clean_text  = _light_clean(raw_text)
    Xv          = vectorizer.transform([clean_text])
    feat_names  = vectorizer.get_feature_names_out()

    # ── 2. Get coefficient array (unwraps wrappers) ───────────────────
    coefs          = _get_coefs(model)
    is_unsigned    = _is_unsigned_attribution(model)  # True for RandomForest

    # ── 3. Decision score for this sample ────────────────────────────
    decision_score = float(_decision_score(model, Xv))

    # Infer is_fraud if not provided
    if is_fraud is None:
        if hasattr(model, "predict_proba"):
            is_fraud = float(model.predict_proba(Xv)[0][1]) >= 0.5
        else:
            is_fraud = decision_score > 0

    # ── 4. Token attribution ──────────────────────────────────────────
    cx          = Xv.tocsr()
    col_indices = cx.indices
    tfidf_vals  = cx.data
    n_coefs     = len(coefs)

    token_scores = []
    for col_idx, tfidf_val in zip(col_indices, tfidf_vals):
        if n_coefs == 0 or col_idx >= n_coefs:
            continue
        coef   = float(coefs[col_idx])
        impact = float(tfidf_val) * coef
        token_scores.append({
            "word":   feat_names[col_idx],
            "impact": impact,
            "coef":   coef,
            "tfidf":  float(tfidf_val),
        })

    # ── 5. Split tokens by direction ──────────────────────────────────
    #
    # FIX 2: RandomForest returns unsigned feature_importances_.
    # All impacts are ≥ 0, so the old code put every token in fraud_tokens
    # regardless of the actual prediction.
    #
    # Fix: for unsigned models, assign all tokens to the predicted direction.
    if is_unsigned:
        sorted_tokens = sorted(token_scores,
                               key=lambda x: abs(x["impact"]), reverse=True)
        if is_fraud:
            fraud_tokens = sorted_tokens
            legit_tokens = []
        else:
            fraud_tokens = []
            # Negate so the legit_tokens have negative impact (convention)
            legit_tokens = [{**t, "impact": -abs(t["impact"])}
                            for t in sorted_tokens]
    else:
        fraud_tokens = sorted(
            [t for t in token_scores if t["impact"] > 0],
            key=lambda x: x["impact"], reverse=True,
        )
        legit_tokens = sorted(
            [t for t in token_scores if t["impact"] < 0],
            key=lambda x: x["impact"],
        )

    # ── 6. Normalize to percentages ───────────────────────────────────
    max_abs = max((abs(t["impact"]) for t in token_scores), default=1e-9)
    max_abs = max(max_abs, 1e-9)

    def to_pct(tokens, n):
        return [
            {
                "word":   t["word"],
                "pct":    round(abs(t["impact"]) / max_abs * 100, 1),
                "impact": round(t["impact"], 4),
            }
            for t in tokens[:n]
        ]

    top_fraud_words = to_pct(fraud_tokens, top_n)
    top_legit_words = to_pct(legit_tokens, top_n)

    # ── 7. Pattern matching ───────────────────────────────────────────
    fraud_patterns_found = _match_patterns(raw_text)

    # ── 8. Highlighted HTML ───────────────────────────────────────────
    fraud_word_set = {t["word"] for t in fraud_tokens[:top_n]}
    legit_word_set = {t["word"] for t in legit_tokens[:top_n]}
    highlighted    = _highlight_html(clean_text, fraud_word_set, legit_word_set)

    # ── 9. Plain-English reasons ──────────────────────────────────────
    reasons = _build_reasons(
        fraud_tokens, legit_tokens,
        fraud_patterns_found, decision_score,
        is_fraud=is_fraud,   # FIX 1
    )

    # ── 10. Model display name ────────────────────────────────────────
    klass = type(model).__name__
    if klass == "CalibratedClassifierCV" and hasattr(model, "calibrated_classifiers_"):
        klass = type(model.calibrated_classifiers_[0].estimator).__name__
    model_display = {
        "LogisticRegression":     "Logistic Regression",
        "LinearSVC":              "Linear SVM",
        "RandomForestClassifier": "Random Forest",
        "MultinomialNB":          "Naive Bayes",
        "SGDClassifier":          "SGD Classifier",
    }.get(klass, klass)

    return {
        "top_fraud_words":  top_fraud_words,
        "top_legit_words":  top_legit_words,
        "highlighted_html": highlighted,
        "fraud_patterns":   fraud_patterns_found,
        "reasons":          reasons,
        "model_name":       model_display,
        "decision_score":   decision_score,
        "n_fraud_tokens":   len(fraud_tokens),
        "n_legit_tokens":   len(legit_tokens),
    }


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def _light_clean(text: str) -> str:
    """Minimal cleaning matching what the vectorizer pipeline sees."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_unsigned_attribution(model) -> bool:
    """
    Returns True if the model only produces unsigned (directionless) feature
    scores — i.e. RandomForest / gradient boosting feature_importances_.
    CalibratedClassifierCV wrapping LinearSVC is NOT unsigned.
    """
    if hasattr(model, "calibrated_classifiers_"):
        return False   # unwrapped later in _get_coefs
    return (
        hasattr(model, "feature_importances_")
        and not hasattr(model, "coef_")
    )


def _get_coefs(model) -> np.ndarray:
    """
    Extract 1-D signed coefficient array.

    FIX 3: Unwraps CalibratedClassifierCV to reach the underlying LinearSVC
    and averages coef_ across the k calibration folds.

    Handles:
      CalibratedClassifierCV(LinearSVC)  → average of fold coef_ arrays
      LogisticRegression / LinearSVC     → coef_[0]
      MultinomialNB                      → log-prob difference (signed)
      RandomForestClassifier             → feature_importances_ (unsigned)
    """
    # CalibratedClassifierCV wrapper
    if hasattr(model, "calibrated_classifiers_"):
        coef_list = []
        for cal_clf in model.calibrated_classifiers_:
            base = cal_clf.estimator
            if hasattr(base, "coef_"):
                c = base.coef_
                coef_list.append(c[0] if c.ndim == 2 else c)
        if coef_list:
            return np.mean(coef_list, axis=0)
        # Fallback: try predict_proba-based attribution (below)

    # Linear models
    if hasattr(model, "coef_"):
        c = model.coef_
        return c[0] if c.ndim == 2 else c

    # Naive Bayes — log-prob difference is signed
    if hasattr(model, "feature_log_prob_"):
        flp = model.feature_log_prob_
        if flp.shape[0] >= 2:
            return flp[1] - flp[0]
        return flp[0]

    # Random Forest / tree ensembles — unsigned proxy
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_

    return np.zeros(0)


def _decision_score(model, Xv) -> float:
    """
    Raw decision score for the positive (fraud) class.
    FIX 3: unwraps CalibratedClassifierCV — uses predict_proba.
    """
    if hasattr(model, "calibrated_classifiers_"):
        return float(model.predict_proba(Xv)[0][1])
    if hasattr(model, "decision_function"):
        return float(model.decision_function(Xv)[0])
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(Xv)[0][1])
    return 0.0


def _match_patterns(text: str) -> list:
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
    order = {"high": 0, "medium": 1, "low": 2}
    found.sort(key=lambda x: order[x["severity"]])
    return found


def _highlight_html(clean_text: str, fraud_words: set, legit_words: set,
                    max_chars: int = 800) -> str:
    """
    FIX 4 & 5: Highlighting is done on the light-cleaned text (same tokens
    the vectorizer sees) via a single-pass token walk — no regex on
    already-modified HTML, so nested <mark> tags are impossible.

    Steps:
      1. Tokenize clean_text into (word, whitespace) pairs
      2. For each token: if in fraud_words → wrap; elif in legit_words → wrap
      3. HTML-escape the word text inside marks
    """
    snippet = clean_text[:max_chars]
    tokens  = re.split(r"(\s+)", snippet)   # keeps whitespace as separate items
    parts   = []

    for tok in tokens:
        if re.match(r"\s+", tok):
            parts.append(tok)
            continue
        lower = tok.lower()
        if lower in fraud_words:
            parts.append(f'<mark class="hw-fraud">{html_lib.escape(tok)}</mark>')
        elif lower in legit_words:
            parts.append(f'<mark class="hw-legit">{html_lib.escape(tok)}</mark>')
        else:
            parts.append(html_lib.escape(tok))

    result = "".join(parts)
    if len(clean_text) > max_chars:
        result += '<span class="hw-ellipsis">…</span>'
    return result


def _build_reasons(
    fraud_tokens: list,
    legit_tokens:  list,
    patterns:      list,
    decision_score: float,
    is_fraud: bool = True,   # FIX 1 — was `decision_score < 0` (always False for LR/NB)
) -> list:
    """
    Generate 3–6 plain-English bullet reasons for the prediction.

    FIX 1: `not is_fraud` correctly identifies legitimate predictions for
    ALL model types including Logistic Regression and Naive Bayes.
    The old `decision_score < 0` never fired for these because
    _decision_score() returns P(fraud) ∈ [0,1] for predict_proba models.
    """
    reasons = []

    # Top patterns (most human-readable, highest priority)
    for p in patterns[:3]:
        reasons.append({
            "icon": p["icon"],
            "text": f"{p['label']}: {p['reason']}",
            "type": "pattern",
            "sev":  p["severity"],
        })

    # Fraud vocabulary (if not already covered by patterns)
    if len(reasons) < 2 and fraud_tokens:
        top_words = [t["word"] for t in fraud_tokens[:4]]
        reasons.append({
            "icon": "🔍",
            "text": (
                f"High-fraud vocabulary detected: «{', '.join(top_words)}» — "
                "these terms appear disproportionately in fraudulent postings."
            ),
            "type": "token",
            "sev":  "medium",
        })

    # FIX 1: legitimate vocabulary reason — now works for ALL model types
    if not is_fraud and legit_tokens:
        top_words = [t["word"] for t in legit_tokens[:4]]
        reasons.append({
            "icon": "✅",
            "text": (
                f"Professional vocabulary present: «{', '.join(top_words)}» — "
                "these terms are strongly associated with genuine job postings."
            ),
            "type": "token",
            "sev":  "low",
        })

    # Decision score confidence commentary
    abs_score = abs(decision_score)
    if abs_score > 2.0:
        reasons.append({
            "icon": "📊",
            "text": (
                f"Model decision boundary crossed with high margin "
                f"(score {decision_score:+.2f}) — prediction is confident."
            ),
            "type": "score",
            "sev":  "low",
        })
    elif abs_score < 0.4:
        reasons.append({
            "icon": "⚠️",
            "text": (
                f"Model decision score is near the boundary "
                f"(score {decision_score:+.2f}) — treat this result with caution "
                "and verify the posting independently."
            ),
            "type": "score",
            "sev":  "medium",
        })

    return reasons[:6]
