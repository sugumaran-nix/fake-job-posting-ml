"""
app.py — JobGuard v8 (BERT Edition)
=====================================
Key change from v7: ml_predict() replaced by model_router.get_predictor().
The router auto-selects BERT (ONNX) if available, else falls back to sklearn.

All routes, DB logic, guard, and security are identical to v7.
"""

import os
import sys
import logging
import datetime
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

from utils.evaluation import load_metrics, load_model_registry, load_model_by_name
from utils.model_router import get_predictor
from analyzer import analyse_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

if not app.secret_key:
    if os.environ.get("FLASK_ENV") == "production":
        raise RuntimeError(
            "FLASK_SECRET_KEY must be set in production. "
            "Generate: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    logger.warning("FLASK_SECRET_KEY not set — using insecure dev default.")
    app.secret_key = "jobguard_dev_secret_not_for_production"

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_ENV") == "production",
    WTF_CSRF_TIME_LIMIT=3600,
)

# ── Security headers (Flask-Talisman) ─────────────────────────────────
try:
    from flask_talisman import Talisman
    Talisman(
        app,
        content_security_policy=False,
        force_https=os.environ.get("FLASK_ENV") == "production",
        strict_transport_security=True,
        strict_transport_security_max_age=31536000,
        strict_transport_security_include_subdomains=True,
        x_content_type_options=True,
        frame_options="DENY",
        referrer_policy="strict-origin-when-cross-origin",
        session_cookie_secure=os.environ.get("FLASK_ENV") == "production",
        session_cookie_http_only=True,
    )
    logger.info("Flask-Talisman: security headers active.")
except ImportError:
    logger.warning("Flask-Talisman not installed.")

# ── CSRF ──────────────────────────────────────────────────────────────
try:
    from flask_wtf.csrf import CSRFProtect, CSRFError
    csrf = CSRFProtect(app)
    logger.info("Flask-WTF: CSRF protection active.")

    @app.errorhandler(CSRFError)
    def csrf_error(e):
        flash("Session expired. Please try again.", "error")
        return redirect(url_for("classify")), 302

except ImportError:
    logger.warning("Flask-WTF not installed — CSRF disabled.")
    csrf = None

# ── Rate limiting ─────────────────────────────────────────────────────
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(
        get_remote_address, app=app,
        default_limits=["200 per day", "60 per hour"],
        storage_uri=os.environ.get("RATELIMIT_STORAGE_URI", "memory://"),
    )
except ImportError:
    class _NoopLimiter:
        def limit(self, *a, **kw):
            def d(f): return f
            return d
    limiter = _NoopLimiter()

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ACTIVE_PATH = os.path.join(BASE_DIR, "models", "active_model.json")
MD_PATH     = os.path.join(BASE_DIR, "models", "model_metadata.json")
DB_PATH     = os.path.join(BASE_DIR, "data", "predictions.db")
MAX_TEXT_LEN = 20_000

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()


# ══════════════════════════════════════════════════════════════════════
#  PREDICTOR — loaded once at startup, reused per request
# ══════════════════════════════════════════════════════════════════════
# model_router.get_predictor() auto-selects:
#   • BERT (ONNX INT8) — if models/bert_onnx_quantized.onnx exists
#   • sklearn (joblib)  — fallback if ONNX not present

predictor = get_predictor()

if predictor:
    logger.info("Predictor ready: %s", predictor._model_name
                if hasattr(predictor, '_model_name') else type(predictor).__name__)
else:
    logger.error("No predictor loaded. Run train.py or bert_finetune.py.")



def _active_model_name() -> str:
    if predictor:
        return getattr(predictor, '_model_name',
                       type(predictor).__name__)
    return "None"


def _switch_model(name: str) -> bool:
    """Model switching only applies to sklearn backend."""
    from utils.evaluation import load_model_by_name as _lmbn
    m = _lmbn(name)
    if m is None:
        return False
    os.makedirs(os.path.dirname(ACTIVE_PATH), exist_ok=True)
    with open(ACTIVE_PATH, "w") as f:
        json.dump({"active_model": name,
                   "switched_at": datetime.datetime.now().isoformat()}, f)
    # Update the sklearn predictor's model
    if hasattr(predictor, '_model'):
        import joblib
        predictor._model = m
        predictor._model_name = name
    logger.info("Active model → %s", name)
    return True


def _check_admin(req) -> bool:
    if not ADMIN_TOKEN:
        return True
    provided = (
        req.form.get("admin_token", "").strip()
        or req.headers.get("X-Admin-Token", "").strip()
    )
    return bool(provided) and provided == ADMIN_TOKEN


def load_metadata() -> dict:
    if os.path.exists(MD_PATH):
        with open(MD_PATH) as f:
            return json.load(f)
    return {}


# ══════════════════════════════════════════════════════════════════════
#  JOB POSTING GUARD
# ══════════════════════════════════════════════════════════════════════
_CORE_JOB_TERMS = {
    "job", "jobs", "position", "positions", "role", "roles",
    "vacancy", "vacancies", "opening", "openings",
    "hiring", "hire", "hired", "employment", "career", "careers",
    "recruit", "recruitment", "internship", "internships",
    "opportunity", "opportunities", "fresher", "freshers",
    "applicant", "applicants", "designation",
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
    word_list = text.lower().split()
    if len(word_list) < 30:
        return False
    words = set(word_list)
    if not (words & _CORE_JOB_TERMS):
        return False
    if len(words & _QUALIFICATION_TERMS) < 2:
        return False
    if not any(words & c for c in [_WORK_TERMS, _COMPENSATION_TERMS, _APPLY_TERMS]):
        return False
    return True


# ══════════════════════════════════════════════════════════════════════
#  DATABASE  — PostgreSQL (Supabase) or SQLite fallback
# ══════════════════════════════════════════════════════════════════════
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

if DATABASE_URL:
    try:
        import psycopg2, psycopg2.extras

        def get_db():
            return psycopg2.connect(DATABASE_URL,
                                    cursor_factory=psycopg2.extras.RealDictCursor)

        def _exec(sql, params=()):
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                conn.commit()

        def _fetch(sql, params=()):
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return [dict(r) for r in cur.fetchall()]

        def _fetchone(sql, params=()):
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    return dict(row) if row else None

        PH = "%s"   # PostgreSQL placeholder
        logger.info("Database: PostgreSQL (DATABASE_URL)")
    except ImportError:
        DATABASE_URL = ""

if not DATABASE_URL:
    import sqlite3

    def get_db():
        c = sqlite3.connect(DB_PATH)
        c.row_factory = sqlite3.Row
        return c

    def _exec(sql, params=()):
        sql = sql.replace("%s", "?")
        with get_db() as c:
            c.execute(sql, params)
            c.commit()

    def _fetch(sql, params=()):
        sql = sql.replace("%s", "?")
        with get_db() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def _fetchone(sql, params=()):
        sql = sql.replace("%s", "?")
        with get_db() as c:
            row = c.execute(sql, params).fetchone()
            return dict(row) if row else None

    PH = "?"
    logger.info("Database: SQLite — ephemeral on Render. Set DATABASE_URL for persistence.")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    serial = "SERIAL" if DATABASE_URL else "INTEGER"
    auto_pk = "PRIMARY KEY" if DATABASE_URL else "PRIMARY KEY AUTOINCREMENT"
    ts_type = "TIMESTAMPTZ DEFAULT NOW()" if DATABASE_URL else "DATETIME DEFAULT CURRENT_TIMESTAMP"

    _exec(f"""
        CREATE TABLE IF NOT EXISTS predictions (
            id           {serial} {auto_pk},
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
            submitted_at {ts_type}
        )
    """)
    for col in [
        "ALTER TABLE predictions ADD COLUMN%s fraud_prob REAL" % (" IF NOT EXISTS" if DATABASE_URL else ""),
        "ALTER TABLE predictions ADD COLUMN%s legit_prob REAL" % (" IF NOT EXISTS" if DATABASE_URL else ""),
        "ALTER TABLE predictions ADD COLUMN%s model_used TEXT" % (" IF NOT EXISTS" if DATABASE_URL else ""),
        "ALTER TABLE predictions ADD COLUMN%s url_risk TEXT"   % (" IF NOT EXISTS" if DATABASE_URL else ""),
        "ALTER TABLE predictions ADD COLUMN%s url_score REAL"  % (" IF NOT EXISTS" if DATABASE_URL else ""),
    ]:
        try:
            _exec(col)
        except Exception:
            pass

# ── DB init at module level — runs when gunicorn imports app.py.
# init_db() is idempotent (CREATE TABLE IF NOT EXISTS) so safe to call
# on every deploy and on every gunicorn worker fork.
try:
    init_db()
    logger.info("Database initialized.")
except Exception as _db_err:
    logger.error("DB init failed: %s", _db_err)


def save_pred(fd, result, url_risk="low", url_score=0):
    _exec(
        "INSERT INTO predictions "
        "(job_title,company,location,salary,website,description,"
        " requirements,prediction,confidence,fraud_prob,legit_prob,"
        " url_risk,url_score,model_used) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (fd.get("job_title",""), fd.get("company",""),
         fd.get("location",""),  fd.get("salary",""),
         fd.get("website",""),   fd.get("description",""),
         fd.get("requirements",""),
         result["label"], result["confidence"],
         result.get("fraud_prob"), result.get("legit_prob"),
         url_risk, url_score, result.get("model_name",""))
    )


def get_history(limit=100):
    return _fetch("SELECT * FROM predictions ORDER BY submitted_at DESC LIMIT %s", (limit,))


def get_stats():
    total = _fetchone("SELECT COUNT(*) AS n FROM predictions")["n"]
    fraud = _fetchone("SELECT COUNT(*) AS n FROM predictions WHERE prediction=%s",
                      ("Fraudulent",))["n"]
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
          for k in ["job_title","company","location","salary",
                    "website","description","requirements"]}

    if not fd["description"]:
        flash("Job description is required.", "error")
        return render_template("classify.html", form=fd,
                               active_model=_active_model_name())

    fd["description"] = fd["description"][:MAX_TEXT_LEN]
    combined = " ".join(filter(None, [
        fd["job_title"], fd["company"],
        fd["description"], fd["requirements"],
    ]))

    if not _is_job_posting(combined):
        flash(
            "This doesn't look like a job posting. Include role title, "
            "responsibilities, and requirements for best results.",
            "warning",
        )
        return render_template("classify.html", form=fd,
                               active_model=_active_model_name())

    if not predictor:
        flash("Model not loaded. Run train.py or bert_finetune.py + bert_to_onnx.py.", "error")
        return render_template("classify.html", form=fd,
                               active_model=_active_model_name())

    result = predictor.predict(combined)
    if "error" in result:
        flash(result["error"], "error")
        return render_template("classify.html", form=fd,
                               active_model=_active_model_name())

    analysis   = analyse_all(fd.get("website",""), fd.get("company",""))
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    try:
        save_pred(fd, result, analysis["overall_risk"], analysis["url"]["score"])
    except Exception as e:
        logger.error("DB write failed: %s", e)

    logger.info("predict | %s | fraud=%.1f%% | %dms | model=%s | ip=%s",
                result["label"], result["fraud_prob"], elapsed_ms,
                result.get("model_name","?"), request.remote_addr)

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
    # Show BERT metadata if available, else sklearn metrics
    bert_meta_path = os.path.join(BASE_DIR, "models", "bert_results.json")
    bert_meta = {}
    if os.path.exists(bert_meta_path):
        with open(bert_meta_path) as f:
            bert_meta = json.load(f)

    return render_template("models.html",
                           metrics=load_metrics(),
                           metadata=load_metadata(),
                           bert_meta=bert_meta,
                           active_model=_active_model_name(),
                           registry=load_model_registry())


@app.route("/clear_history", methods=["POST"])
def clear_history():
    if not _check_admin(request):
        flash("Unauthorised.", "error")
        return redirect(url_for("history"))
    _exec("DELETE FROM predictions")
    flash("History cleared.", "success")
    return redirect(url_for("history"))


@app.route("/select-model", methods=["POST"])
def select_model():
    if not _check_admin(request):
        flash("Unauthorised.", "error")
        return redirect(url_for("models_page"))
    name = request.form.get("model_name","").strip()
    if not name:
        flash("No model name provided.", "error")
        return redirect(url_for("models_page"))
    if _switch_model(name):
        flash(f"Active model → {name}", "success")
    else:
        flash(f"Model '{name}' not found.", "error")
    return redirect(url_for("models_page"))


# ── JSON API ──────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
@limiter.limit("30 per minute")
def api_predict():
    data        = request.get_json(force=True, silent=True) or {}
    description = str(data.get("description","")).strip()
    if not description:
        return jsonify({"error": "description is required"}), 400

    combined = " ".join(filter(None, [
        str(data.get("title","")), str(data.get("company","")),
        description, str(data.get("requirements","")),
    ]))

    if not _is_job_posting(combined):
        return jsonify({"error": "Input does not appear to be a job posting."}), 422

    if not predictor:
        return jsonify({"error": "Model not loaded."}), 503

    result   = predictor.predict(combined)
    if "error" in result:
        return jsonify(result), 503

    analysis = analyse_all(str(data.get("website","")), str(data.get("company","")))
    exp      = result.get("explanation", {})

    return jsonify({
        "prediction":     result["label"],
        "is_fraud":       result["is_fraud"],
        "confidence":     result["confidence"],
        "fraud_prob":     result["fraud_prob"],
        "legit_prob":     result["legit_prob"],
        "model_used":     result.get("model_name",""),
        "url_risk":       analysis["overall_risk"],
        "combined_score": analysis["combined_score"],
        "explanation": {
            "top_fraud_words": exp.get("top_fraud_words",[]),
            "top_legit_words": exp.get("top_legit_words",[]),
            "fraud_patterns":  [
                {"label":p["label"],"severity":p["severity"],
                 "reason":p["reason"],"matched":p["matched"]}
                for p in exp.get("fraud_patterns",[])
            ],
            "reasons":        exp.get("reasons",[]),
            "decision_score": exp.get("decision_score",0),
        },
    })


@app.route("/api/models", methods=["GET"])
def api_models():
    registry    = load_model_registry()
    all_metrics = {m["name"]: m for m in load_metrics()}
    models_list = []
    for name, path in registry.items():
        resolved = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
        entry = {
            "name":      name,
            "available": os.path.exists(resolved),
            "is_active": name == _active_model_name(),
        }
        if name in all_metrics:
            entry.update({k: all_metrics[name].get(k) for k in
                          ["accuracy","f1_fraud","recall_fraud",
                           "precision_fraud","roc_auc","cv_f1_mean","cv_f1_std"]})
        models_list.append(entry)

    # Include BERT if it exists
    onnx_path = os.path.join(BASE_DIR, "models", "bert_onnx_quantized.onnx")
    if os.path.exists(onnx_path):
        bert_results_path = os.path.join(BASE_DIR, "models", "bert_results.json")
        bert_entry = {
            "name": "DistilBERT (ONNX INT8)",
            "available": True,
            "is_active": "BERT" in _active_model_name(),
            "size_mb": round(os.path.getsize(onnx_path) / 1e6, 1),
        }
        if os.path.exists(bert_results_path):
            with open(bert_results_path) as f:
                br = json.load(f)
            bert_entry.update({
                "accuracy":        br.get("test_accuracy"),
                "f1_fraud":        br.get("test_f1_fraud"),
                "recall_fraud":    br.get("test_recall_fraud"),
                "precision_fraud": br.get("test_precision_fraud"),
                "roc_auc":         br.get("test_roc_auc"),
            })
        models_list.append(bert_entry)

    return jsonify({"active_model": _active_model_name(), "models": models_list})


@app.route("/health")
def health():
    onnx_path   = os.path.join(BASE_DIR, "models", "bert_onnx_quantized.onnx")
    data_path   = onnx_path + ".data"
    bert_active = os.path.exists(onnx_path)
    bert_size   = 0
    if bert_active:
        bert_size += os.path.getsize(onnx_path) / 1e6
    if os.path.exists(data_path):
        bert_size += os.path.getsize(data_path) / 1e6
    return jsonify({
        "status":        "ok",
        "predictor":     type(predictor).__name__ if predictor else "None",
        "active_model":  _active_model_name(),
        "bert_onnx":     bert_active,
        "bert_size_mb":  round(bert_size, 1) if bert_active else None,
        "db_backend":    "postgresql" if DATABASE_URL else "sqlite",
        "timestamp":     datetime.datetime.now().isoformat(),
    })


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    logger.error("500: %s", e)
    return render_template("500.html"), 500


if __name__ == "__main__":
    init_db()
    debug = os.environ.get("FLASK_ENV") == "development"
    port  = int(os.environ.get("PORT", 5000))
    logger.info("JobGuard v8 — http://localhost:%d | backend=%s",
                port, type(predictor).__name__ if predictor else "None")
    app.run(debug=debug, host="0.0.0.0", port=port)
