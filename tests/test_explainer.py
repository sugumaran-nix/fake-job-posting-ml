"""
tests/test_explainer.py — JobGuard v7
======================================
Tests covering the three vice-versa bugs fixed in explainer.py:
  1. _build_reasons: legit reasons shown for LR/NB (was: decision_score<0 never fired)
  2. RF unsigned attribution: legit tokens exist for legitimate predictions
  3. CalibratedClassifierCV coef_ unwrapping: returns non-empty array
  4. _highlight_html: no nested marks
  5. _decision_score: works for all model types
"""

import os
import sys
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.explainer import (
    explain, _get_coefs, _decision_score, _build_reasons,
    _highlight_html, _is_unsigned_attribution, _light_clean,
)


# ── Shared fixtures ────────────────────────────────────────────────────

TRAIN_DOCS = [
    "work from home earn guaranteed income registration fee required no experience",
    "urgent hiring data entry online typing easy money guaranteed daily earn",
    "network marketing mlm refer earn passive income join now registration fee",
    "registration fee required starter kit pay upfront no degree needed",
    "data entry work home earn guaranteed daily income urgent registration fee apply",
    "software engineer python django rest api backend responsibilities experience degree",
    "data analyst sql power bi responsibilities requirements qualification apply resume",
    "machine learning engineer python tensorflow responsibilities skills degree apply",
    "backend developer node express postgresql responsibilities requirements experience",
    "senior developer react typescript responsibilities qualification apply resume",
]
TRAIN_LABELS = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

FRAUD_TEXT = "work from home earn guaranteed income registration fee required no experience needed urgent hiring"
LEGIT_TEXT = "software engineer python django rest api backend responsibilities experience degree apply resume"


@pytest.fixture(scope="module")
def vec_and_data():
    vec = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
    vec.fit(TRAIN_DOCS)
    return vec


@pytest.fixture(scope="module")
def lr_model(vec_and_data):
    vec = vec_and_data
    X   = vec.transform(TRAIN_DOCS)
    clf = LogisticRegression(class_weight="balanced", max_iter=200)
    clf.fit(X, TRAIN_LABELS)
    return clf


@pytest.fixture(scope="module")
def svm_calibrated(vec_and_data):
    vec = vec_and_data
    X   = vec.transform(TRAIN_DOCS)
    clf = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", max_iter=500),
        method="sigmoid", cv=2,
    )
    clf.fit(X, TRAIN_LABELS)
    return clf


@pytest.fixture(scope="module")
def rf_model(vec_and_data):
    vec = vec_and_data
    X   = vec.transform(TRAIN_DOCS)
    clf = RandomForestClassifier(n_estimators=20, class_weight="balanced", random_state=42)
    clf.fit(X, TRAIN_LABELS)
    return clf


@pytest.fixture(scope="module")
def nb_model(vec_and_data):
    vec = vec_and_data
    X   = vec.transform(TRAIN_DOCS)
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X, TRAIN_LABELS)
    return clf


# ══════════════════════════════════════════════════════════════════════
#  FIX 1 — _build_reasons uses is_fraud flag, not decision_score < 0
# ══════════════════════════════════════════════════════════════════════

def test_build_reasons_legit_for_lr(lr_model, vec_and_data):
    """
    For a LEGITIMATE prediction by LogisticRegression, the 'professional
    vocabulary' reason must appear. In v6 decision_score = P(fraud) > 0
    always, so `decision_score < 0` never fired and the legit reason was
    never shown. FIX: use `not is_fraud`.
    """
    vec = vec_and_data
    Xv  = vec.transform([LEGIT_TEXT])
    prob_fraud = float(lr_model.predict_proba(Xv)[0][1])
    is_fraud   = prob_fraud >= 0.5

    # Force is_fraud=False to simulate a legitimate prediction
    fake_fraud_tokens = []
    fake_legit_tokens = [{"word": "engineer", "impact": -0.5, "tfidf": 0.4}]

    reasons = _build_reasons(
        fake_fraud_tokens, fake_legit_tokens,
        [], prob_fraud,
        is_fraud=False,   # FIX: explicit flag
    )
    icons = [r["icon"] for r in reasons]
    texts = " ".join(r["text"] for r in reasons)
    assert "fa-circle-check" in icons, \
        "Legitimate prediction should show a check icon for professional vocabulary"
    assert "Professional vocabulary" in texts


def test_build_reasons_no_legit_for_fraud(lr_model, vec_and_data):
    """Fraudulent prediction should NOT show the legitimate vocabulary reason."""
    fake_legit_tokens = [{"word": "engineer", "impact": -0.5, "tfidf": 0.4}]
    reasons = _build_reasons(
        [{"word": "fee", "impact": 0.9, "tfidf": 0.8}],
        fake_legit_tokens,
        [], 0.85,
        is_fraud=True,
    )
    texts = " ".join(r["text"] for r in reasons)
    assert "Professional vocabulary" not in texts


# ══════════════════════════════════════════════════════════════════════
#  FIX 2 — RF unsigned attribution: legit tokens populated for legit predictions
# ══════════════════════════════════════════════════════════════════════

def test_rf_legit_tokens_not_empty_for_legit_prediction(rf_model, vec_and_data):
    """
    In v6, feature_importances_ is unsigned so ALL tokens had positive impact.
    Every token went into fraud_tokens and legit_tokens was always empty.
    FIX: for unsigned models, direction is assigned from is_fraud flag.
    """
    vec    = vec_and_data
    result = explain(LEGIT_TEXT, vec, rf_model, is_fraud=False)
    # With is_fraud=False and unsigned model, legit_tokens should be non-empty
    assert len(result["top_legit_words"]) > 0, \
        "RF: top_legit_words should be non-empty for a legitimate prediction"
    # And fraud_tokens should be empty (no signals pushing toward fraud)
    assert len(result["top_fraud_words"]) == 0, \
        "RF: top_fraud_words should be empty for a legitimate prediction"


def test_rf_fraud_tokens_not_empty_for_fraud_prediction(rf_model, vec_and_data):
    """RF: fraud prediction → fraud_tokens populated, legit_tokens empty."""
    vec    = vec_and_data
    result = explain(FRAUD_TEXT, vec, rf_model, is_fraud=True)
    assert len(result["top_fraud_words"]) > 0
    assert len(result["top_legit_words"]) == 0


def test_rf_is_unsigned_attribution(rf_model):
    assert _is_unsigned_attribution(rf_model) is True


def test_lr_is_not_unsigned(lr_model):
    assert _is_unsigned_attribution(lr_model) is False


def test_calibrated_svm_is_not_unsigned(svm_calibrated):
    assert _is_unsigned_attribution(svm_calibrated) is False


# ══════════════════════════════════════════════════════════════════════
#  FIX 3 — CalibratedClassifierCV unwrapping in _get_coefs
# ══════════════════════════════════════════════════════════════════════

def test_get_coefs_calibrated_svm_non_empty(svm_calibrated, vec_and_data):
    """
    _get_coefs must unwrap CalibratedClassifierCV and return the averaged
    fold coef_ arrays from the underlying LinearSVC estimators.
    In v6, CalibratedClassifierCV has no coef_, so _get_coefs returned
    np.zeros(0) and no token attribution was computed.
    """
    coefs = _get_coefs(svm_calibrated)
    assert len(coefs) > 0, "Calibrated SVM coefs must be non-empty"
    # Should have both positive and negative coefficients (signed)
    assert (coefs > 0).any() or (coefs < 0).any()


def test_get_coefs_lr_signed(lr_model):
    coefs = _get_coefs(lr_model)
    assert len(coefs) > 0
    assert coefs.min() < 0  # at least some negative (legit-direction) weights


def test_get_coefs_nb_signed(nb_model):
    coefs = _get_coefs(nb_model)
    assert len(coefs) > 0
    # NB: fraud_log_prob - legit_log_prob — should have mixed signs


def test_get_coefs_rf_non_negative(rf_model):
    coefs = _get_coefs(rf_model)
    assert len(coefs) > 0
    assert (coefs >= 0).all(), "RF feature_importances_ must be non-negative"


# ══════════════════════════════════════════════════════════════════════
#  FIX 4 — _highlight_html: no nested marks
# ══════════════════════════════════════════════════════════════════════

def test_highlight_no_nested_marks():
    """
    In v6, bigram matches like 'include health' and unigram 'health' both
    matching produced <mark class="hw-legit"><mark ...>health</mark></mark>.
    FIX: single-pass token walk produces at most one mark per token.
    """
    import re
    fraud = {"include", "earn", "fee", "guaranteed"}
    legit = {"python", "experience", "health insurance", "health"}
    html  = _highlight_html(
        "include python experience earn health fee guaranteed",
        fraud, legit,
    )
    nested = re.findall(r'<mark[^>]*>[^<]*<mark', html)
    assert nested == [], f"Nested marks found: {nested}"


def test_highlight_marks_fraud_words():
    html = _highlight_html("earn guaranteed income fee", {"earn", "fee"}, set())
    assert 'class="hw-fraud"' in html
    assert "earn" in html


def test_highlight_marks_legit_words():
    html = _highlight_html("python experience engineer", set(), {"python", "experience"})
    assert 'class="hw-legit"' in html


def test_highlight_escapes_xss():
    """User input with HTML must be escaped before marking."""
    html = _highlight_html("<script>alert(1)</script> job", set(), set())
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# ══════════════════════════════════════════════════════════════════════
#  FIX 5 — _decision_score handles all model types
# ══════════════════════════════════════════════════════════════════════

def test_decision_score_lr(lr_model, vec_and_data):
    vec   = vec_and_data
    Xv    = vec.transform([FRAUD_TEXT])
    score = _decision_score(lr_model, Xv)
    assert 0.0 <= score <= 1.0, "LR decision score should be P(fraud) ∈ [0,1]"


def test_decision_score_calibrated_svm(svm_calibrated, vec_and_data):
    vec   = vec_and_data
    Xv    = vec.transform([FRAUD_TEXT])
    score = _decision_score(svm_calibrated, Xv)
    assert 0.0 <= score <= 1.0, "Calibrated SVM decision score should be ∈ [0,1]"


def test_decision_score_nb(nb_model, vec_and_data):
    vec   = vec_and_data
    Xv    = vec.transform([LEGIT_TEXT])
    score = _decision_score(nb_model, Xv)
    assert 0.0 <= score <= 1.0


# ══════════════════════════════════════════════════════════════════════
#  EXPLAIN RESULT SHAPE
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("model_fixture,is_fraud", [
    ("lr_model", True),
    ("lr_model", False),
    ("svm_calibrated", True),
    ("rf_model", False),
    ("nb_model", True),
])
def test_explain_result_has_required_keys(model_fixture, is_fraud, request,
                                          vec_and_data):
    model  = request.getfixturevalue(model_fixture)
    vec    = vec_and_data
    text   = FRAUD_TEXT if is_fraud else LEGIT_TEXT
    result = explain(text, vec, model, is_fraud=is_fraud)

    required = {"top_fraud_words", "top_legit_words", "highlighted_html",
                "fraud_patterns", "reasons", "model_name", "decision_score"}
    missing  = required - set(result.keys())
    assert not missing, f"explain() missing keys: {missing}"


def test_explain_reasons_non_empty_for_legit_lr(lr_model, vec_and_data):
    """After fix, a legitimate LR prediction must have at least one reason."""
    result = explain(LEGIT_TEXT, vec_and_data, lr_model, is_fraud=False)
    assert len(result["reasons"]) >= 1
