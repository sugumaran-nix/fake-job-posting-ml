"""
utils/model_router.py — Unified predictor router
=================================================
Tries to load BERT (ONNX) first. Falls back to sklearn if ONNX
model is not present (e.g. before running bert_finetune.py).

app.py imports get_predictor() instead of touching sklearn or BERT directly.
This is the only file that knows which backend is active.

Usage:
    from utils.model_router import get_predictor

    predictor = get_predictor()        # called once at module level
    result    = predictor.predict(text)

    # Both backends return identical dict:
    # {label, is_fraud, confidence, fraud_prob, legit_prob, model_name, explanation}
"""

import os
import logging
import joblib

logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH  = os.path.join(BASE_DIR, "models", "bert_onnx_quantized.onnx")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VEC_PATH   = os.path.join(BASE_DIR, "models", "vectorizer.pkl")


class SklearnPredictor:
    """
    Wraps existing sklearn ml_predict() into the same interface as BertPredictor.
    Used as fallback when ONNX model is not available.
    """

    def __init__(self):
        import math, json
        from utils.explainer import explain
        from utils.preprocessing import preprocess

        self._preprocess = preprocess
        self._explain    = explain
        self._math       = math

        # Load vectorizer
        if not os.path.exists(VEC_PATH):
            raise RuntimeError(f"Vectorizer not found: {VEC_PATH}. Run train.py first.")
        self._vectorizer = joblib.load(VEC_PATH)

        # Load model
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model not found: {MODEL_PATH}. Run train.py first.")
        self._model = joblib.load(MODEL_PATH)

        # Threshold
        self._threshold = 0.5
        thresh_path = os.path.join(BASE_DIR, "models", "threshold.json")
        if os.path.exists(thresh_path):
            try:
                with open(thresh_path) as f:
                    self._threshold = float(json.load(f).get("threshold", 0.5))
            except Exception:
                pass

        # Model display name
        klass = type(self._model).__name__
        if klass == "CalibratedClassifierCV" and hasattr(self._model, "calibrated_classifiers_"):
            klass = type(self._model.calibrated_classifiers_[0].estimator).__name__
        self._model_name = {
            "LogisticRegression":     "Logistic Regression",
            "LinearSVC":              "Linear SVM",
            "RandomForestClassifier": "Random Forest",
            "MultinomialNB":          "Naive Bayes",
        }.get(klass, klass)

        logger.info("SklearnPredictor loaded: %s (threshold=%.3f)",
                    self._model_name, self._threshold)

    def predict(self, text: str) -> dict:
        import math
        cleaned = self._preprocess(text)
        vec     = self._vectorizer.transform([cleaned])

        if hasattr(self._model, "predict_proba"):
            fraud_prob = float(self._model.predict_proba(vec)[0][1])
        elif hasattr(self._model, "decision_function"):
            s          = float(self._model.decision_function(vec)[0])
            fraud_prob = 1.0 / (1.0 + math.exp(-s))
        else:
            fraud_prob = float(self._model.predict(vec)[0])

        is_fraud   = fraud_prob >= self._threshold
        legit_prob = 1.0 - fraud_prob
        confidence = round(max(fraud_prob, legit_prob) * 100, 2)
        explanation = self._explain(text, self._vectorizer, self._model, is_fraud=is_fraud)

        return {
            "label":      "Fraudulent" if is_fraud else "Legitimate",
            "is_fraud":   is_fraud,
            "confidence": confidence,
            "fraud_prob": round(fraud_prob * 100, 2),
            "legit_prob": round(legit_prob * 100, 2),
            "model_name": self._model_name,
            "explanation": explanation,
        }

    def warmup(self):
        self.predict(
            "Software engineer position responsibilities requirements "
            "experience degree apply resume salary benefits"
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._vectorizer is not None


# ── Singleton ──────────────────────────────────────────────────────────
_predictor = None


def get_predictor():
    """
    Returns the active predictor singleton.
    Call once at module level in app.py; reuse across all requests.

    Priority:
      1. BertPredictor  — if models/bert_onnx_quantized.onnx exists
      2. SklearnPredictor — if models/model.pkl + models/vectorizer.pkl exist
      3. None             — if neither is available (server returns 503)
    """
    global _predictor
    if _predictor is not None:
        return _predictor

    # Try BERT first
    if os.path.exists(ONNX_PATH):
        try:
            from utils.bert_predictor import BertPredictor
            _predictor = BertPredictor()
            _predictor.warmup()
            logger.info("Active backend: BERT (ONNX INT8)")
            return _predictor
        except Exception as e:
            logger.warning("BERT load failed (%s) — falling back to sklearn.", e)

    # Fallback: sklearn
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        try:
            _predictor = SklearnPredictor()
            _predictor.warmup()
            logger.info("Active backend: sklearn (%s)", _predictor._model_name)
            return _predictor
        except Exception as e:
            logger.error("Sklearn load failed: %s", e)

    logger.error("No model available. Run train.py or bert_finetune.py + bert_to_onnx.py.")
    return None
