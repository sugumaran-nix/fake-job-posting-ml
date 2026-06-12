"""
app.py — Fake Job Posting Prediction (v4 — Multi-Model Runtime Switching)
============================================================================
Fixes applied over original v4:
  ✅ Secret key raises RuntimeError in production if unset
  ✅ .env loaded automatically via python-dotenv
  ✅ /api/predict rate-limited (Flask-Limiter)
  ✅ Optimal classification threshold loaded from models/threshold.json
  ✅ predict_proba path uses threshold; fallback to predict() for SVM/NB
  ✅ Single-worker constraint documented clearly
  ✅ Heuristic confidence nudge documented with TODO
"""

import os
import sys
import pickle
import logging
import sqlite3
import datetime
import math
import json
import time

# ── Load .env (dev convenience; no-op if python-dotenv not installed) ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import (Flask, render_template, request,
                   redirect, url_for, jsonify, flash)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import preprocess
from utils.evaluation import (
    load_metrics,
    load_model_registry,
    load_model_by_name,
)
from utils.explainer import explain
from analyzer import analyse_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

# ── Secret key validation ──────────────────────────────────────────────
if not app.secret_key:
    if os.environ.get("FLASK_ENV") == "production":
        raise RuntimeError(
            "FLASK_SECRET_KEY must be set in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    # Dev fallback — loud warning so it's never silently used
    logger.warning(
        "FLASK_SECRET_KEY not set — using insecure dev default. "
        "Set it in .env or as an environment variable."
    )
    app.secret_key = "fjp_dev_secret_only_not_for_production"

# ── Rate limiting ──────────────────────────────────────────────────────
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per day", "60 per hour"],
        storage_uri="memory://",   # swap for redis:// in multi-worker setups
    )
    _limiter_available = True
except ImportError:
    logger.warning("Flask-Limiter not installed — API rate limiting disabled. "
                   "Run: pip install Flask-Limiter")
    _limiter_available = False

    # Stub decorator so routes don't fail at import time
    class _NoopLimiter:
        def limit(self, *a, **kw):
            def decorator(f): return f
            return decorator
    limiter = _NoopLimiter()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VEC_PATH   = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
MD_PATH    = os.path.join(BASE_DIR, "models", "model_metadata.json")
ACTIVE_PATH= os.path.join(BASE_DIR, "models", "active_model.json")
THRESH_PATH= os.path.join(BASE_DIR, "models", "threshold.json")
DB_PATH    = os.path.join(BASE_DIR, "data", "predictions.db")

MAX_TEXT_LEN = 20_000

# ══════════════════════════════════════════════════════════════════════
# CLASSIFICATION THRESHOLD
# Loaded from threshold.json (written by train.py).
# Falls back to 0.5 if the file doesn't exist yet.
# ══════════════════════════════════════════════════════════════════════
_threshold = 0.5
if os.path.exists(THRESH_PATH):
    try:
        with open(THRESH_PATH) as f:
            _threshold = float(json.load(f).get("threshold", 0.5))
        logger.info(f"Loaded classification threshold: {_threshold:.2f}")
    except Exception:
        logger.warning("Could not parse threshold.json — using 0.5")

# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING + RUNTIME SWITCHING
# ══════════════════════════════════════════════════════════════════════

# Vectorizer is loaded once — it never changes between model switches.
try:
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("Vectorizer loaded.")
except FileNotFoundError:
    logger.warning("Vectorizer not found. Run train.py first.")
    vectorizer = None

# ⚠ SINGLE-WORKER ONLY: _active_state lives in one process's memory.
# With gunicorn -w N (N > 1), only the worker that receives /select-model
# switches — other workers keep their old model.
# For multi-worker production: read active_model.json per-request,
# or store the selection in the database / Redis.
_active_state: dict = {
    "model": None,
    "name":  None,
}


def load_metadata() -> dict:
    if os.path.exists(MD_PATH):
        with open(MD_PATH) as f:
            return json.load(f)
    return {}


def _load_active_model() -> None:
    """
    Load the active model, preferring the persisted selection in
    active_model.json, falling back to model.pkl (best model from training).
    """
    if os.path.exists(ACTIVE_PATH):
        try:
            with open(ACTIVE_PATH) as f:
                saved = json.load(f)
            name = saved.get("active_model")
            if name:
                m = load_model_by_name(name)
                if m is not None:
                    _active_state["model"] = m
                    _active_state["name"]  = name
                    logger.info(f"Restored active model: {name}")
                    return
        except Exception:
            pass

    # Fall back to best model (model.pkl)
    try:
        with open(MODEL_PATH, "rb") as f:
            _active_state["model"] = pickle.load(f)
        md = load_metadata()
        _active_state["name"] = md.get("best_model", type(_active_state["model"]).__name__)
        logger.info(f"Loaded default best model: {_active_state['name']}")
    except FileNotFoundError:
        logger.warning("No model found. Run train.py first.")


_load_active_model()


def switch_model(name: str) -> bool:
    """
    Hot-swap the active model by name. Persists selection to active_model.json.
    Returns True on success, False if model not found.
    See ⚠ single-worker note above.
    """
    m = load_model_by_name(name)
    if m is None:
        return False
    _active_state["model"] = m
    _active_state["name"]  = name
    os.makedirs(os.path.dirname(ACTIVE_PATH), exist_ok=True)
    with open(ACTIVE_PATH, "w") as f:
        json.dump({
            "active_model": name,
            "switched_at": datetime.datetime.now().isoformat(),
        }, f)
    logger.info(f"Switched active model → {name}")
    return True


# ══════════════════════════════════════════════════════════════════════
# ML PREDICTION
# ══════════════════════════════════════════════════════════════════════

def ml_predict(combined: str, raw_for_explain: str = "") -> dict:
    """
    Run inference and explainability on combined text using the active model.
    Uses the optimal threshold from threshold.json (default 0.5).
    """
    model = _active_state["model"]
    if not model:
        return {"error": "Model not loaded. Run train.py first."}

    combined = combined[:MAX_TEXT_LEN]
    cleaned  = preprocess(combined)
    vec      = vectorizer.transform([cleaned])

    # ── Threshold-aware prediction ─────────────────────────────────
    # Models with predict_proba (LR, RF, NB) use the tuned threshold.
    # Models without it (LinearSVC) fall back to predict().
    if hasattr(model, "predict_proba"):
        prob_fraud = float(model.predict_proba(vec)[0][1])
        pred       = int(prob_fraud >= _threshold)
        conf       = round((prob_fraud if pred == 1 else 1 - prob_fraud) * 100, 2)
    elif hasattr(model, "decision_function"):
        s    = float(model.decision_function(vec)[0])
        pred = int(model.predict(vec)[0])
        conf = round((1 / (1 + math.exp(-abs(s)))) * 100, 2)
    else:
        pred = int(model.predict(vec)[0])
        conf = 95.0

    explain_input = (raw_for_explain or combined)[:MAX_TEXT_LEN]
    explanation   = explain(explain_input, vectorizer, model)

    return {
        "label":       "Fraudulent" if pred == 1 else "Legitimate",
        "confidence":  conf,
        "is_fraud":    pred == 1,
        "model_name":  _active_state["name"] or explanation["model_name"],
        "explanation": explanation,
    }


# ══════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════

def get_db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                job_title    TEXT,
                company      TEXT,
                location     TEXT,
                salary       TEXT,
                website      TEXT,
                description  TEXT NOT NULL,
                requirements TEXT,
                prediction   TEXT NOT NULL,
                confidence   REAL NOT NULL,
                url_risk     TEXT,
                url_score    REAL,
                model_used   TEXT,
                submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
        # Idempotent schema migrations
        for col_sql in [
            "ALTER TABLE predictions ADD COLUMN model_used TEXT",
            "ALTER TABLE predictions ADD COLUMN url_risk TEXT",
            "ALTER TABLE predictions ADD COLUMN url_score REAL",
        ]:
            try:
                c.execute(col_sql)
            except Exception:
                pass
        c.commit()


def save_pred(fd: dict, result: dict, url_risk: str = "low", url_score: float = 0):
    with get_db() as c:
        c.execute(
            "INSERT INTO predictions "
            "(job_title,company,location,salary,website,description,"
            " requirements,prediction,confidence,url_risk,url_score,model_used) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (fd.get("job_title",""), fd.get("company",""), fd.get("location",""),
             fd.get("salary",""),    fd.get("website",""), fd.get("description",""),
             fd.get("requirements",""), result["label"], result["confidence"],
             url_risk, url_score,    result.get("model_name","")))
        c.commit()


def get_history(limit: int = 100):
    with get_db() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM predictions ORDER BY submitted_at DESC LIMIT ?",
            (limit,)).fetchall()]


def get_stats() -> dict:
    with get_db() as c:
        total = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        fraud = c.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction='Fraudulent'"
        ).fetchone()[0]
    return {"total": total, "fraud": fraud, "legit": total - fraud}


# ══════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html", stats=get_stats())


@app.route("/classify")
def classify():
    return render_template("classify.html", active_model=_active_state["name"])


@app.route("/predict", methods=["POST"])
def predict_route():
    t0 = time.perf_counter()

    fd = {k: request.form.get(k, "").strip()
          for k in ["job_title", "company", "location",
                    "salary", "website", "description", "requirements"]}

    if not fd["description"]:
        flash("Job description is required.", "error")
        return render_template("classify.html", form=fd,
                               active_model=_active_state["name"])

    if len(fd["description"]) > MAX_TEXT_LEN:
        fd["description"] = fd["description"][:MAX_TEXT_LEN]

    combined = " ".join(filter(None, [
        fd["job_title"], fd["company"], fd["description"], fd["requirements"]
    ]))
    raw_for_explain = " ".join(filter(None, [
        fd["job_title"], fd["company"], fd["description"],
        fd["requirements"], fd["salary"]
    ]))

    result = ml_predict(combined, raw_for_explain)
    if "error" in result:
        flash(result["error"], "error")
        return render_template("classify.html", form=fd,
                               active_model=_active_state["name"])

    analysis = analyse_all(fd.get("website", ""), fd.get("company", ""))

    # Heuristic confidence nudge based on URL risk.
    # This has no statistical grounding — it's a UX signal only.
    # A high-risk URL reinforces a fraud prediction (+5 pp, capped 99.9)
    # or weakens a legit prediction (-15 pp, floor 55) to surface doubt.
    # TODO: Replace with a calibrated ensemble (e.g. isotonic regression).
    if analysis["overall_risk"] == "high" and result["is_fraud"]:
        result["confidence"] = min(result["confidence"] + 5, 99.9)
    elif analysis["overall_risk"] == "high" and not result["is_fraud"]:
        result["confidence"] = max(result["confidence"] - 15, 55.0)

    save_pred(fd, result, analysis["overall_risk"], analysis["url"]["score"])

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("Prediction: %s | conf=%.1f%% | %dms | model=%s",
                result["label"], result["confidence"], elapsed_ms,
                result.get("model_name", "?"))

    return render_template("result.html",
                           result=result,
                           job=fd,
                           analysis=analysis,
                           elapsed_ms=elapsed_ms)


@app.route("/history")
def history():
    return render_template("history.html",
                           records=get_history(), stats=get_stats())


@app.route("/about")
def about():
    return render_template("about.html", metadata=load_metadata())


@app.route("/models")
def models_page():
    return render_template("models.html",
                           metrics=load_metrics(),
                           metadata=load_metadata(),
                           active_model=_active_state["name"],
                           registry=load_model_registry())


@app.route("/clear_history", methods=["POST"])
def clear_history():
    with get_db() as c:
        c.execute("DELETE FROM predictions")
        c.commit()
    flash("History cleared.", "success")
    return redirect(url_for("history"))


# ── Model switching ────────────────────────────────────────────────────
@app.route("/select-model", methods=["POST"])
def select_model():
    """
    Switch the active inference model without restarting Flask.
    Form field: model_name (exact name from model registry)
    ⚠ Single-worker only — see module docstring.
    """
    name = request.form.get("model_name", "").strip()
    if not name:
        flash("No model name provided.", "error")
        return redirect(url_for("models_page"))

    if switch_model(name):
        flash(f"✅ Active model switched to: {name}", "success")
    else:
        flash(f"❌ Model '{name}' not found in registry. Run train.py first.", "error")
    return redirect(url_for("models_page"))


# ── JSON API ───────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
@limiter.limit("30 per minute")
def api_predict():
    """
    Full JSON API with explanation.

    Request body (JSON):
        title       : str (optional)
        company     : str (optional)
        description : str (REQUIRED)
        requirements: str (optional)
        website     : str (optional)
        model       : str (optional — name of model to use for this request)
    """
    data = request.get_json(force=True, silent=True) or {}

    description = str(data.get("description", "")).strip()
    if not description:
        return jsonify({"error": "description is required"}), 400

    # Per-request model override (optional, reverted after request)
    req_model_name  = str(data.get("model", "")).strip()
    original_model  = _active_state["model"]
    original_name   = _active_state["name"]
    _switched       = False

    if req_model_name and req_model_name != _active_state["name"]:
        m = load_model_by_name(req_model_name)
        if m:
            _active_state["model"] = m
            _active_state["name"]  = req_model_name
            _switched = True

    combined = " ".join(filter(None, [
        str(data.get("title", "")),
        str(data.get("company", "")),
        description,
        str(data.get("requirements", "")),
    ]))
    raw = combined + " " + str(data.get("salary", ""))

    result = ml_predict(combined, raw)

    if _switched:
        _active_state["model"] = original_model
        _active_state["name"]  = original_name

    if "error" in result:
        return jsonify(result), 503

    analysis = analyse_all(
        str(data.get("website", "")),
        str(data.get("company", "")),
    )

    exp = result.get("explanation", {})
    return jsonify({
        "prediction":     result["label"],
        "is_fraud":       result["is_fraud"],
        "confidence":     result["confidence"],
        "model_used":     result.get("model_name", ""),
        "url_risk":       analysis["overall_risk"],
        "combined_score": analysis["combined_score"],
        "explanation": {
            "top_fraud_words": exp.get("top_fraud_words", []),
            "top_legit_words": exp.get("top_legit_words", []),
            "fraud_patterns": [
                {"label":    p["label"],
                 "severity": p["severity"],
                 "reason":   p["reason"],
                 "matched":  p["matched"]}
                for p in exp.get("fraud_patterns", [])
            ],
            "reasons":        exp.get("reasons", []),
            "decision_score": exp.get("decision_score", 0),
        },
    })


@app.route("/api/models", methods=["GET"])
def api_models():
    """
    Returns JSON list of all available models with their metrics.

    Response:
        active_model : str — currently active model name
        models       : list[dict] — name, path, metrics per model
    """
    registry    = load_model_registry()
    all_metrics = {m["name"]: m for m in load_metrics()}
    models_list = []

    for name, path in registry.items():
        entry = {
            "name":      name,
            "available": os.path.exists(path),
            "is_active": name == _active_state["name"],
        }
        if name in all_metrics:
            m = all_metrics[name]
            entry.update({
                "accuracy":        m.get("accuracy"),
                "f1_fraud":        m.get("f1_fraud"),
                "recall_fraud":    m.get("recall_fraud"),
                "precision_fraud": m.get("precision_fraud"),
                "roc_auc":         m.get("roc_auc"),
                "cv_f1_mean":      m.get("cv_f1_mean"),
                "cv_f1_std":       m.get("cv_f1_std"),
            })
        models_list.append(entry)

    return jsonify({
        "active_model": _active_state["name"],
        "models":       models_list,
    })


@app.route("/health")
def health():
    return jsonify({
        "status":         "ok",
        "model_loaded":   _active_state["model"] is not None,
        "active_model":   _active_state["name"],
        "vectorizer_ok":  vectorizer is not None,
        "threshold":      _threshold,
        "timestamp":      datetime.datetime.now().isoformat(),
    })


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    init_db()
    debug = os.environ.get("FLASK_ENV") == "development"
    port  = int(os.environ.get("PORT", 5000))
    logger.info(f"Fake Job Posting Prediction — http://localhost:{port}")
    app.run(debug=debug, host="0.0.0.0", port=port)
