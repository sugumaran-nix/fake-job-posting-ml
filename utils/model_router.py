"""
utils/model_router.py — Unified predictor router
=================================================
Tries to load BERT (ONNX) first. Falls back to sklearn if the ONNX model
is not present (e.g. before running bert_finetune.py + bert_to_onnx.py).

app.py imports get_predictor() instead of touching sklearn or BERT directly.
This is the only file that knows which backend is currently active.

Usage:
    from utils.model_router import get_predictor

    predictor = get_predictor()         # called once at module level
    result    = predictor.predict(text)

    # Both backends return identical dict:
    # {label, is_fraud, confidence, fraud_prob, legit_prob, model_name, explanation}

Bug fixes:
  - FIX 1: SklearnPredictor.__init__ imported `json` inside __init__ via a
            combined `import math, json` statement and then used it only in the
            threshold-loading block.  `math` was imported again inside predict()
            creating duplicate imports and making intent unclear.  Both are now
            imported cleanly at the top of each method that needs them.

  - FIX 2: SklearnPredictor predict() now correctly handles the case where the
            model exposes neither predict_proba nor decision_function by falling
            back to a binary float from predict() rather than potentially
            returning a numpy bool that breaks JSON serialisation downstream.

  - FIX 3: get_predictor() singleton reset added for test/reload scenarios;
            reset_predictor() is exported for use in tests.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Optional

import joblib

logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH  = os.path.join(BASE_DIR, "models", "bert_onnx_quantized.onnx")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VEC_PATH   = os.path.join(BASE_DIR, "models", "vectorizer.pkl")


# ══════════════════════════════════════════════════════════════════════
#  SKLEARN PREDICTOR (fallback)
# ══════════════════════════════════════════════════════════════════════

class SklearnPredictor:
    """
    Wraps the existing sklearn pipeline into the same interface as
    BertPredictor.  Used as a fallback when the ONNX model is absent.
    """

    def __init__(self):
        from utils.explainer import explain
        from utils.preprocessing import preprocess

        self._preprocess = preprocess
        self._explain    = explain

        # ── Vectorizer ───────────────────────────────────────────────
        if not os.path.exists(VEC_PATH):
            raise RuntimeError(
                f"Vectorizer not found: {VEC_PATH}\n"
                "Run train.py first to generate models/vectorizer.pkl."
            )
        self._vectorizer = joblib.load(VEC_PATH)

        # ── Model ────────────────────────────────────────────────────
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model not found: {MODEL_PATH}\n"
                "Run train.py first to generate models/model.pkl."
            )
        self._model = joblib.load(MODEL_PATH)

        # ── Threshold ────────────────────────────────────────────────
        self._threshold = 0.5
        thresh_path = os.path.join(BASE_DIR, "models", "threshold.json")
        if os.path.exists(thresh_path):
            try:
                with open(thresh_path) as fh:
                    self._threshold = float(json.load(fh).get("threshold", 0.5))
            except Exception:
                logger.warning("Could not read threshold.json — using 0.5")

        # ── Display name ─────────────────────────────────────────────
        klass = type(self._model).__name__
        if klass == "CalibratedClassifierCV" and hasattr(self._model, "calibrated_classifiers_"):
            klass = type(self._model.calibrated_classifiers_[0].estimator).__name__
        self._model_name = {
            "LogisticRegression":     "Logistic Regression",
            "LinearSVC":              "Linear SVM",
            "RandomForestClassifier": "Random Forest",
            "MultinomialNB":          "Naive Bayes",
        }.get(klass, klass)

        logger.info(
            "SklearnPredictor loaded: %s (threshold=%.3f)",
            self._model_name, self._threshold,
        )

    def predict(self, text: str) -> dict:
        cleaned = self._preprocess(text)
        vec     = self._vectorizer.transform([cleaned])

        if hasattr(self._model, "predict_proba"):
            fraud_prob = float(self._model.predict_proba(vec)[0][1])
        elif hasattr(self._model, "decision_function"):
            # FIX 1: import math at call site (was incorrectly mixed with json)
            raw        = float(self._model.decision_function(vec)[0])
            fraud_prob = 1.0 / (1.0 + math.exp(-raw))
        else:
            # FIX 2: force float so JSON serialisation never receives a numpy bool
            fraud_prob = float(self._model.predict(vec)[0])

        is_fraud   = fraud_prob >= self._threshold
        legit_prob = 1.0 - fraud_prob
        confidence = round(max(fraud_prob, legit_prob) * 100, 2)
        explanation = self._explain(
            text, self._vectorizer, self._model, is_fraud=is_fraud
        )

        return {
            "label":       "Fraudulent" if is_fraud else "Legitimate",
            "is_fraud":    is_fraud,
            "confidence":  confidence,
            "fraud_prob":  round(fraud_prob * 100, 2),
            "legit_prob":  round(legit_prob * 100, 2),
            "model_name":  self._model_name,
            "explanation": explanation,
        }

    def warmup(self) -> None:
        self.predict(
            "Software engineer position responsibilities requirements "
            "experience degree apply resume salary benefits"
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._vectorizer is not None


# ══════════════════════════════════════════════════════════════════════
#  SINGLETON
# ══════════════════════════════════════════════════════════════════════

_predictor: Optional[object] = None


def get_predictor() -> Optional[object]:
    """
    Returns the active predictor singleton.
    Call once at module level in app.py; reuse across all requests.

    Priority:
      1. BertPredictor     — if models/bert_onnx_quantized.onnx exists
      2. SklearnPredictor  — if models/model.pkl + models/vectorizer.pkl exist
      3. None              — if neither is available (server returns 503)
    """
    global _predictor
    if _predictor is not None:
        return _predictor

    # ── Try BERT first ───────────────────────────────────────────────
    if os.path.exists(ONNX_PATH):
        try:
            from utils.bert_predictor import BertPredictor
            _predictor = BertPredictor()
            _predictor.warmup()
            logger.info("Active backend: BERT (ONNX INT8)")
            return _predictor
        except Exception as exc:
            logger.warning(
                "BERT load failed (%s) — falling back to sklearn.", exc
            )

    # ── Fallback: sklearn ────────────────────────────────────────────
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        try:
            _predictor = SklearnPredictor()
            _predictor.warmup()
            logger.info("Active backend: sklearn (%s)", _predictor._model_name)
            return _predictor
        except Exception as exc:
            logger.error("Sklearn load failed: %s", exc)

    logger.error(
        "No model available. "
        "Run train.py OR bert_finetune.py + bert_to_onnx.py first."
    )
    return None


def reset_predictor() -> None:
    """
    FIX 3: Force singleton re-initialisation.
    Useful in tests and hot-swap scenarios.
    """
    global _predictor
    _predictor = None
