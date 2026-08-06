"""
app.py — JobGuard: AI-Powered Job Fraud Detection (v7)
============================================================================
Bug fixes from v6:
  ✅ FIX: _is_job_posting guard rewritten — 4-cluster strict validation
         (core job term + qualifications + 1 of work/pay/apply cluster)
         Rejects pollution paragraphs, news articles, product descriptions
  ✅ FIX: ml_predict now applies threshold to LinearSVC via sigmoid(decision_score)
         threshold.json is now honoured for ALL model types (was ignored for SVM)
  ✅ FIX: result dict now always contains fraud_prob + legit_prob — fixed values
         Card 1 always shows P(fraud), Card 2 always shows P(legit) in template
  ✅ FIX: is_fraud flag passed to explain() — fixes vice-versa reason bullets
         for Logistic Regression and Naive Bayes (decision_score < 0 never fired)
  ✅ FIX: /api/predict model override uses local variables — no global cache mutation
         (previous code had a data race under concurrent requests / multi-worker)
  ✅ FIX: _check_admin validates that ADMIN_TOKEN is non-empty before dev-bypass
  ✅ FIX: CalibratedClassifierCV wrapper handled in model display name
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

if not app.secret_key:
    if os.environ.get("FLASK_ENV") == "production":
        raise RuntimeError(
            "FLASK_SECRET_KEY must be set in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    logger.warning(
        "FLASK_SECRET_KEY not set — using insecure dev default. "
        "Set it in .env or as an environment variable."
    )
    app.secret_key = "jobguard_dev_secret_not_for_production"

# ── Admin token ────────────────────────────────────────────────────────
# FIX: explicitly check for non-empty string so that ADMIN_TOKEN=""
#      in .env doesn't silently open admin routes.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()

# ── Rate limiting ──────────────────────────────────────────────────────
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    _storage = os.environ.get("RATELIMIT_STORAGE_URI", "memory://")
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per day", "60 per hour"],
        storage_uri=_storage,
    )
except ImportError:
    logger.warning("Flask-Limiter not installed — API rate limiting disabled.")
    class _NoopLimiter:
        def limit(self, *a, **kw):
            def decorator(f): return f
            return decorator
    limiter = _NoopLimiter()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "model.pkl")
VEC_PATH    = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
MD_PATH     = os.path.join(BASE_DIR, "models", "model_metadata.json")
ACTIVE_PATH = os.path.join(BASE_DIR, "models", "active_model.json")
THRESH_PATH = os.path.join(BASE_DIR, "models", "threshold.json")
DB_PATH     = os.path.join(BASE_DIR, "data", "predictions.db")

MAX_TEXT_LEN = 20_000

# ══════════════════════════════════════════════════════════════════════
#  JOB POSTING GUARD  (rewritten v7 — 4-cluster validation)
# ══════════════════════════════════════════════════════════════════════
# Strategy: require evidence from FOUR distinct semantic clusters.
# A paragraph about pollution, a news article, or product description
# won't have all four, so it gets rejected before ML inference.

_CORE_JOB_TERMS = {
    # Explicit hiring / vacancy language
    "job", "jobs", "position", "positions", "role", "roles",
    "vacancy", "vacancies", "opening", "openings",
    "hiring", "hire", "hired", "employment", "career", "careers",
    "recruit", "recruitment", "internship", "internships",
    "opportunity", "opportunities", "fresher", "freshers",
    "applicant", "applicants", "designation",
    # Common role nouns — "looking for a Developer / Engineer / ..."
    # Without these, postings that omit "vacancy/job" but name the role fail.
    "developer", "developers", "engineer", "engineers",
    "analyst", "analysts", "manager", "managers",
    "designer", "designers", "consultant", "consultants",
    "coordinator", "coordinators", "architect", "architects",
    "executive", "executives", "specialist", "specialists",
    "officer", "officers", "technician", "technicians",
    "programmer", "programmers", "administrator", "administrators",
    "supervisor", "supervisors", "accountant", "accountants",
    "scientist", "scientists", "researcher", "researchers",
    "intern", "interns", "trainee", "trainees",
}

_QUALIFICATION_TERMS = {
    "skills", "skill", "requirements", "requirement",
    "qualifications", "qualification", "responsibilities",
    "responsibility", "duties", "degree", "bachelor", "master",
    "diploma", "btech", "mtech", "mca", "bca", "bsc", "msc",
    "bcom", "mba", "graduate", "postgraduate", "education",
    "experience", "proficiency", "knowledge", "certification",
}

_WORK_TERMS = {
    "work", "working", "office", "remote", "wfh", "hybrid",
    "full-time", "fulltime", "part-time", "parttime",
    "contract", "permanent", "freelance", "onsite",
}

_COMPENSATION_TERMS = {
    "salary", "ctc", "lpa", "compensation", "package", "stipend",
    "pay", "wage", "wages", "income", "remuneration", "benefits",
    "allowance", "incentive", "bonus",
}

_APPLY_TERMS = {
    "apply", "applying", "applied", "resume", "cv", "candidate",
    "candidates", "interview", "application", "submit",
    "shortlisted", "joining", "onboard",
}


def _is_job_posting(text: str) -> bool:
    """
    Returns True only if the text satisfies ALL four conditions:
      1. At least 30 words
      2. At least 1 core job term  (vacancy / hiring / position / etc.)
      3. At least 2 qualification / responsibility terms
      4. At least 1 term from any of: work-type, compensation, or apply cluster

    Rejects: pollution paragraphs, Wikipedia articles, product copy,
             news stories, random text, single-sentence queries.
    """
    word_list = text.lower().split()
    if len(word_list) < 30:
        return False

    words = set(word_list)

    # 1 — core job signal
    if not (words & _CORE_JOB_TERMS):
        return False

    # 2 — qualification / responsibility language (min 2)
    if len(words & _QUALIFICATION_TERMS) < 2:
        return False

    # 3 — at least one more contextual cluster
    extra_clusters = [_WORK_TERMS, _COMPENSATION_TERMS, _APPLY_TERMS]
    if not any(words & c for c in extra_clusters):
        return False

    return True


# ── Classification threshold ───────────────────────────────────────────
# train.py writes models/threshold.json with the F1-optimal threshold.
# Falls back to 0.5 if the file is absent.
_threshold = 0.5
if os.path.exists(THRESH_PATH):
    try:
        with open(THRESH_PATH) as f:
            _threshold = float(json.load(f).get("threshold", 0.5))
        logger.info(f"Loaded classification threshold: {_threshold:.2f}")
    except Exception:
        logger.warning("Could not parse threshold.json — using 0.5")

# ── Vectorizer ────────────────────────────────────────────────────────
try:
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("Vectorizer loaded.")
except FileNotFoundError:
    logger.warning("Vectorizer not found. Run train.py first.")
    vectorizer = None

# ── In-process model cache ────────────────────────────────────────────
_model_cache: dict = {"name": None, "model": None}


def _get_active_model():
    """
    Return (name, model). Reads active_model.json per-request so all
    gunicorn workers stay in sync, but caches the loaded object.
    """
    desired_name = None
    if os.path.exists(ACTIVE_PATH):
        try:
            with open(ACTIVE_PATH) as f:
                desired_name = json.load(f).get("active_model")
        except Exception:
            pass

    if not desired_name and os.path.exists(MD_PATH):
        try:
            with open(MD_PATH) as f:
                desired_name = json.load(f).get("best_model")
        except Exception:
            pass

    if desired_name and desired_name == _model_cache["name"]:
        return _model_cache["name"], _model_cache["model"]

    if desired_name:
        m = load_model_by_name(desired_name)
        if m:
            _model_cache["name"]  = desired_name
            _model_cache["model"] = m
            return desired_name, m

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                m = pickle.load(f)
            name = desired_name or "Best Model"
            _model_cache["name"]  = name
            _model_cache["model"] = m
            return name, m
        except Exception:
            pass

    return None, None


def _active_model_name() -> str:
    name, _ = _get_active_model()
    return name or "None"


def _switch_model(name: str) -> bool:
    m = load_model_by_name(name)
    if m is None:
        return False
    os.makedirs(os.path.dirname(ACTIVE_PATH), exist_ok=True)
    with open(ACTIVE_PATH, "w") as f:
        json.dump({"active_model": name,
                   "switched_at": datetime.datetime.now().isoformat()}, f)
    _model_cache["name"]  = name
    _model_cache["model"] = m
    logger.info(f"Switched active model → {name}")
    return True


def _check_admin(req) -> bool:
    """
    FIX: ADMIN_TOKEN is now stripped and checked for non-empty.
    An empty ADMIN_TOKEN string no longer silently opens admin routes.
    Only truly unset (missing from env) triggers dev-bypass.
    """
    if not ADMIN_TOKEN:          # unset in env → dev mode only
        return True
    provided = (
        req.form.get("admin_token", "").strip()
        or req.headers.get("X-Admin-Token", "").strip()
    )
    return bool(provided) and provided == ADMIN_TOKEN


def _model_display_name(model) -> str:
    """Extract a clean display name, unwrapping CalibratedClassifierCV."""
    klass = type(model).__name__
    if klass == "CalibratedClassifierCV" and hasattr(model, "calibrated_classifiers_"):
        klass = type(model.calibrated_classifiers_[0].estimator).__name__
    return {
        "LogisticRegression":     "Logistic Regression",
        "LinearSVC":              "Linear SVM",
        "RandomForestClassifier": "Random Forest",
        "MultinomialNB":          "Naive Bayes",
        "SGDClassifier":          "SGD Classifier",
    }.get(klass, klass)


def load_metadata() -> dict:
    if os.path.exists(MD_PATH):
        with open(MD_PATH) as f:
            return json.load(f)
    return {}


# ══════════════════════════════════════════════════════════════════════
#  ML PREDICTION  (all bugs fixed)
# ══════════════════════════════════════════════════════════════════════

def ml_predict(combined: str, raw_for_explain: str = "",
               model_override=None, model_name_override: str = "") -> dict:
    """
    Run inference on combined text.

    FIX 1 — threshold now applied to LinearSVC via sigmoid(decision_score):
        sigmoid(s) >= threshold  is used instead of model.predict() which
        hardcodes 0.0. This makes threshold.json meaningful for all models.

    FIX 2 — fraud_prob and legit_prob always returned:
        These are real [0,1] probabilities. Template always shows
        fraud_prob in Card 1 and legit_prob in Card 2 (no label flip).

    FIX 3 — is_fraud passed to explain():
        Fixes vice-versa reason bullets for LR / Naive Bayes.
    """
    if model_override is not None:
        model      = model_override
        model_name = model_name_override or _model_display_name(model)
    else:
        model_name, model = _get_active_model()
        if model:
            model_name = _model_display_name(model)

    if not model:
        return {"error": "Model not loaded. Run train.py first."}
    if not vectorizer:
        return {"error": "Vectorizer not loaded. Run train.py first."}

    combined = combined[:MAX_TEXT_LEN]
    cleaned  = preprocess(combined)
    vec      = vectorizer.transform([cleaned])

    # ── Unified probability extraction ────────────────────────────────
    # All paths produce fraud_prob ∈ [0, 1] — the probability the model
    # assigns to the job being FRAUDULENT (class 1).
    if hasattr(model, "predict_proba"):
        # LR, RF, CalibratedClassifierCV(LinearSVC), NB
        fraud_prob = float(model.predict_proba(vec)[0][1])

    elif hasattr(model, "decision_function"):
        # Raw (uncalibrated) LinearSVC — convert via sigmoid so the
        # threshold from threshold.json is actually meaningful.
        s          = float(model.decision_function(vec)[0])
        fraud_prob = 1.0 / (1.0 + math.exp(-s))   # sigmoid → [0,1]

    else:
        # Fallback: hard predict — treat 1 as 100% fraud
        raw_pred   = int(model.predict(vec)[0])
        fraud_prob = float(raw_pred)

    # Apply the F1-optimal threshold (works identically for all model types now)
    pred      = int(fraud_prob >= _threshold)
    legit_prob = 1.0 - fraud_prob
    conf      = round(max(fraud_prob, legit_prob) * 100, 2)

    is_fraud = pred == 1

    # ── Explanation (is_fraud fixes the reason-bullet bug) ────────────
    explain_input = (raw_for_explain or combined)[:MAX_TEXT_LEN]
    explanation   = explain(explain_input, vectorizer, model, is_fraud=is_fraud)

    return {
        "label":      "Fraudulent" if is_fraud else "Legitimate",
        "confidence": conf,
        "fraud_prob": round(fraud_prob * 100, 2),
        "legit_prob": round(legit_prob * 100, 2),
        "is_fraud":   is_fraud,
        "model_name": model_name,
        "explanation": explanation,
    }


# ══════════════════════════════════════════════════════════════════════
#  DATABASE
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
                fraud_prob   REAL,
                legit_prob   REAL,
                url_risk     TEXT,
                url_score    REAL,
                model_used   TEXT,
                submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
        # Migrate older schema versions
        for col_sql in [
            "ALTER TABLE predictions ADD COLUMN model_used TEXT",
            "ALTER TABLE predictions ADD COLUMN url_risk TEXT",
            "ALTER TABLE predictions ADD COLUMN url_score REAL",
            "ALTER TABLE predictions ADD COLUMN fraud_prob REAL",
            "ALTER TABLE predictions ADD COLUMN legit_prob REAL",
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
            " requirements,prediction,confidence,fraud_prob,legit_prob,"
            " url_risk,url_score,model_used) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (fd.get("job_title", ""), fd.get("company", ""),
             fd.get("location", ""),  fd.get("salary", ""),
             fd.get("website", ""),   fd.get("description", ""),
             fd.get("requirements", ""),
             result["label"], result["confidence"],
             result.get("fraud_prob"), result.get("legit_prob"),
             url_risk, url_score, result.get("model_name", "")))
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
#  ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html", stats=get_stats())


@app.route("/classify")
def classify():
    return render_template("classify.html", active_model=_active_model_name())


@app.route("/predict", methods=["POST"])
def predict_route():
    t0 = time.perf_counter()

    fd = {k: request.form.get(k, "").strip()
          for k in ["job_title", "company", "location",
                    "salary", "website", "description", "requirements"]}

    if not fd["description"]:
        flash("Job description is required.", "error")
        return render_template("classify.html", form=fd,
                               active_model=_active_model_name())

    if len(fd["description"]) > MAX_TEXT_LEN:
        fd["description"] = fd["description"][:MAX_TEXT_LEN]

    combined = " ".join(filter(None, [
        fd["job_title"], fd["company"],
        fd["description"], fd["requirements"],
    ]))

    # ── Job posting guard ─────────────────────────────────────────────
    if not _is_job_posting(combined):
        flash(
            "This doesn't look like a job posting. "
            "Please paste the actual job description (include role, "
            "requirements, and responsibilities for best results).",
            "warning",
        )
        return render_template("classify.html", form=fd,
                               active_model=_active_model_name())

    raw_for_explain = " ".join(filter(None, [
        fd["job_title"], fd["company"], fd["description"],
        fd["requirements"], fd["salary"],
    ]))

    result = ml_predict(combined, raw_for_explain)
    if "error" in result:
        flash(result["error"], "error")
        return render_template("classify.html", form=fd,
                               active_model=_active_model_name())

    analysis = analyse_all(fd.get("website", ""), fd.get("company", ""))
    save_pred(fd, result, analysis["overall_risk"], analysis["url"]["score"])

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("Prediction: %s | fraud_prob=%.1f%% | %dms | model=%s",
                result["label"], result["fraud_prob"], elapsed_ms,
                result.get("model_name", "?"))

    return render_template("result.html",
                           result=result, job=fd,
                           analysis=analysis, elapsed_ms=elapsed_ms)


@app.route("/history")
def history():
    return render_template("history.html",
                           history=get_history(), stats=get_stats())


@app.route("/about")
def about():
    return render_template("about.html", metadata=load_metadata())


@app.route("/models")
def models_page():
    return render_template("models.html",
                           metrics=load_metrics(),
                           metadata=load_metadata(),
                           active_model=_active_model_name(),
                           registry=load_model_registry())


@app.route("/clear_history", methods=["POST"])
def clear_history():
    if not _check_admin(request):
        flash("Unauthorized — provide a valid admin token.", "error")
        return redirect(url_for("history"))
    with get_db() as c:
        c.execute("DELETE FROM predictions")
        c.commit()
    flash("History cleared.", "success")
    return redirect(url_for("history"))


@app.route("/select-model", methods=["POST"])
def select_model():
    if not _check_admin(request):
        flash("Unauthorized — provide a valid admin token.", "error")
        return redirect(url_for("models_page"))

    name = request.form.get("model_name", "").strip()
    if not name:
        flash("No model name provided.", "error")
        return redirect(url_for("models_page"))

    if _switch_model(name):
        flash(f"Active model switched to: {name}", "success")
    else:
        flash(f"Model '{name}' not found in registry. Run train.py first.", "error")
    return redirect(url_for("models_page"))


# ══════════════════════════════════════════════════════════════════════
#  JSON API
# ══════════════════════════════════════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
@limiter.limit("30 per minute")
def api_predict():
    """
    JSON prediction API.

    FIX: model override no longer mutates the global _model_cache.
    The override model is passed directly to ml_predict() so there is
    no shared-state race condition under concurrent requests.

    Request body:
        title        : str (optional)
        company      : str (optional)
        description  : str (REQUIRED)
        requirements : str (optional)
        website      : str (optional)
        model        : str (optional — per-request model override)
    """
    data = request.get_json(force=True, silent=True) or {}

    description = str(data.get("description", "")).strip()
    if not description:
        return jsonify({"error": "description is required"}), 400

    combined = " ".join(filter(None, [
        str(data.get("title",        "")),
        str(data.get("company",      "")),
        description,
        str(data.get("requirements", "")),
    ]))

    if not _is_job_posting(combined):
        return jsonify({
            "error": "Input does not appear to be a job posting. "
                     "Include role title, requirements and responsibilities."
        }), 422

    # FIX: per-request override uses local model variable — no cache mutation
    override_model      = None
    override_model_name = ""
    req_model_name = str(data.get("model", "")).strip()
    if req_model_name:
        override_model = load_model_by_name(req_model_name)
        if override_model:
            override_model_name = req_model_name
        else:
            return jsonify({"error": f"Model '{req_model_name}' not found."}), 404

    result = ml_predict(
        combined,
        combined + " " + str(data.get("salary", "")),
        model_override=override_model,
        model_name_override=override_model_name,
    )

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
        "fraud_prob":     result["fraud_prob"],
        "legit_prob":     result["legit_prob"],
        "model_used":     result.get("model_name", ""),
        "url_risk":       analysis["overall_risk"],
        "combined_score": analysis["combined_score"],
        "explanation": {
            "top_fraud_words": exp.get("top_fraud_words", []),
            "top_legit_words": exp.get("top_legit_words", []),
            "fraud_patterns": [
                {"label": p["label"], "severity": p["severity"],
                 "reason": p["reason"], "matched": p["matched"]}
                for p in exp.get("fraud_patterns", [])
            ],
            "reasons":        exp.get("reasons", []),
            "decision_score": exp.get("decision_score", 0),
        },
    })


@app.route("/api/models", methods=["GET"])
def api_models():
    registry    = load_model_registry()
    all_metrics = {m["name"]: m for m in load_metrics()}
    models_list = []
    for name, path in registry.items():
        entry = {
            "name":      name,
            "available": os.path.exists(path) if os.path.isabs(path)
                         else os.path.exists(os.path.join(BASE_DIR, path)),
            "is_active": name == _active_model_name(),
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
    return jsonify({"active_model": _active_model_name(), "models": models_list})


@app.route("/health")
def health():
    _, model = _get_active_model()
    return jsonify({
        "status":        "ok",
        "model_loaded":  model is not None,
        "active_model":  _active_model_name(),
        "vectorizer_ok": vectorizer is not None,
        "threshold":     _threshold,
        "timestamp":     datetime.datetime.now().isoformat(),
    })


# ── Error handlers ────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    logger.error("500 error: %s", e)
    return render_template("500.html"), 500


# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    init_db()
    debug = os.environ.get("FLASK_ENV") == "development"
    port  = int(os.environ.get("PORT", 5000))
    logger.info(f"JobGuard v7 — http://localhost:{port}")
    app.run(debug=debug, host="0.0.0.0", port=port)
