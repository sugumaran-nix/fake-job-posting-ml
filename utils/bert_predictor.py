"""
utils/bert_predictor.py — BERT ONNX inference for production
=============================================================
Loads the quantized ONNX model once at startup (gunicorn --preload).
Provides predict() which returns the same dict shape as the sklearn
ml_predict() so app.py routes need zero changes.

Memory budget (Render free tier, 512MB):
  - ONNX quantized model : ~80MB
  - ONNX Runtime session : ~30MB
  - ONNXRuntime overhead : ~50MB
  - Flask + Python         ~100MB
  - Total                : ~260MB  ← well under 512MB

Inference latency (CPU, Render):
  - First call (warm)    : ~400ms
  - Subsequent calls     : ~150–300ms
  (vs sklearn LinearSVC  : ~5ms — tradeoff for much better accuracy)
"""

import os
import json
import math
import logging
import hashlib
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH  = os.path.join(BASE_DIR, "models", "bert_onnx_quantized.onnx")
TOKEN_DIR  = os.path.join(BASE_DIR, "models", "bert_tokenizer")
META_PATH  = os.path.join(BASE_DIR, "models", "bert_onnx_meta.json")
THRESH_PATH = os.path.join(BASE_DIR, "models", "threshold.json")


class BertPredictor:
    """
    Singleton-friendly BERT inference wrapper.
    Instantiate once at module level; call predict() per request.

    Usage:
        predictor = BertPredictor()          # loads on startup
        result    = predictor.predict(text)  # per-request inference
    """

    def __init__(self):
        self._session   = None
        self._tokenizer = None
        self._max_len   = 512
        self._threshold = 0.5
        self._model_name = "DistilBERT (ONNX INT8)"
        self._load()

    def _load(self):
        """Load ONNX session + tokenizer. Raises RuntimeError if missing."""
        if not os.path.exists(ONNX_PATH):
            raise RuntimeError(
                f"ONNX model not found: {ONNX_PATH}\n"
                "Run bert_finetune.py then bert_to_onnx.py first."
            )
        if not os.path.exists(TOKEN_DIR):
            raise RuntimeError(
                f"Tokenizer not found: {TOKEN_DIR}\n"
                "Run bert_finetune.py then bert_to_onnx.py first."
            )

        # ONNX Runtime session
        try:
            import onnxruntime as rt
            opts = rt.SessionOptions()
            opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 1      # Single thread: Render free tier has 0.1 vCPU
            self._session = rt.InferenceSession(
                ONNX_PATH,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            logger.info("ONNX session loaded: %s", ONNX_PATH)
        except ImportError:
            raise RuntimeError(
                "onnxruntime not installed. "
                "Add 'onnxruntime==1.19.2' to requirements.txt."
            )

        # Tokenizer
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(TOKEN_DIR)
            logger.info("Tokenizer loaded: %s", TOKEN_DIR)
        except ImportError:
            raise RuntimeError(
                "transformers not installed. "
                "Add 'transformers==4.44.2' to requirements.txt."
            )

        # Max length from metadata
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                meta = json.load(f)
            self._max_len = meta.get("max_length", 512)

        # Threshold from threshold.json (same file sklearn uses)
        if os.path.exists(THRESH_PATH):
            try:
                with open(THRESH_PATH) as f:
                    self._threshold = float(json.load(f).get("threshold", 0.5))
                logger.info("BERT threshold: %.3f", self._threshold)
            except Exception:
                pass

        size_mb = os.path.getsize(ONNX_PATH) / 1e6
        logger.info("BertPredictor ready — model %.1fMB, threshold %.3f",
                    size_mb, self._threshold)

    def _tokenize(self, text: str) -> dict:
        """Tokenize a single string, return numpy arrays."""
        enc = self._tokenizer(
            text,
            max_length=self._max_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def predict(self, text: str) -> dict:
        """
        Run inference on raw text.

        Returns a dict with the same keys as sklearn ml_predict():
            label      : "Fraudulent" | "Legitimate"
            is_fraud   : bool
            confidence : float (0–100)
            fraud_prob : float (0–100)
            legit_prob : float (0–100)
            model_name : str
            explanation: dict (token attribution not available for BERT;
                                returns fraud_patterns + reasons only)
        """
        if not self._session:
            return {"error": "ONNX session not loaded."}

        inputs  = self._tokenize(text)
        logits  = self._session.run(["logits"], inputs)[0]    # shape: (1, 2)
        probs   = self._softmax(logits)[0]                    # shape: (2,)

        fraud_prob = float(probs[1])
        legit_prob = float(probs[0])
        is_fraud   = fraud_prob >= self._threshold
        confidence = round(max(fraud_prob, legit_prob) * 100, 2)

        return {
            "label":      "Fraudulent" if is_fraud else "Legitimate",
            "is_fraud":   is_fraud,
            "confidence": confidence,
            "fraud_prob": round(fraud_prob * 100, 2),
            "legit_prob": round(legit_prob * 100, 2),
            "model_name": self._model_name,
            "explanation": self._explain(text, fraud_prob, is_fraud),
        }

    def _explain(self, text: str, fraud_prob: float, is_fraud: bool) -> dict:
        """
        BERT doesn't expose per-token coefficients like LinearSVC.
        Return pattern-based explanations only (fraud_patterns + reasons).
        Token influence bars are hidden in the template when top_fraud_words=[].
        """
        from utils.explainer import _match_patterns, _build_reasons
        patterns = _match_patterns(text)
        reasons  = _build_reasons(
            fraud_tokens=[], legit_tokens=[],
            patterns=patterns,
            decision_score=fraud_prob,
            is_fraud=is_fraud,
        )
        # Add a BERT confidence reason
        reasons.append({
            "icon": "🤖",
            "text": (
                f"DistilBERT (fine-tuned on 17,880 job postings) assigned "
                f"{fraud_prob * 100:.1f}% probability to this being fraudulent."
            ),
            "type": "model",
            "sev":  "high" if fraud_prob > 0.8 else "medium" if fraud_prob > 0.5 else "low",
        })
        return {
            "top_fraud_words":  [],         # Not available for BERT (no coef_)
            "top_legit_words":  [],
            "highlighted_html": "",
            "fraud_patterns":   patterns,
            "reasons":          reasons[:6],
            "model_name":       self._model_name,
            "decision_score":   fraud_prob,
            "n_fraud_tokens":   0,
            "n_legit_tokens":   0,
        }

    @property
    def is_loaded(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def warmup(self):
        """
        Run one dummy inference to warm up ONNX Runtime JIT.
        Call at app startup (gunicorn --preload) so the first real
        user request isn't slow.
        """
        logger.info("Warming up ONNX Runtime...")
        dummy = (
            "Software engineer position responsibilities requirements "
            "experience degree apply resume salary benefits work office"
        )
        self.predict(dummy)
        logger.info("ONNX warmup complete.")
