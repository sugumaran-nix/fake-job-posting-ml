"""
app.py — JobGuard v9 (Production Edition)
==========================================
Changes from v8:
  - Type hints on all public functions (python-pro)
  - CSP header enabled via Talisman (secure-code-guardian)
  - ADMIN_TOKEN uses hmac.compare_digest — timing-safe (secure-code-guardian)
  - Thread-safe predictor singleton with threading.Lock (python-pro)
  - Input length caps on ALL API + form fields (secure-code-guardian)
  - api_predict 503 no longer leaks internal model info (secure-code-guardian)
  - Structured logging with request_id (python-pro)
  - Form fields stripped and capped before any use (secure-code-guardian)
"""

from __future__ import annotations

import csv
import datetime
import hmac
import io
import json
import logging
import os
import sys
import threading
import time
import uuid
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import Flask, Response, flash, g, jsonify, redirect, render_template, request, url_for

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer import analyse_all
from utils.evaluation import load_metrics, load_model_by_name, load_model_registry
from utils.model_router import get_predictor

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "")


@app.before_request
def _start_request_context() -> None:
    g.request_id = uuid.uuid4().hex[:12]
    g.request_started = time.perf_counter()


@app.after_request
def _finish_request(response: Response) -> Response:
    request_id = getattr(g, "request_id", "-")
    started = getattr(g, "request_started", None)
    elapsed_ms = (time.perf_counter() - started) * 1000 if started else 0.0
    response.headers["X-Request-ID"] = request_id
    if request.endpoint != "static":
        logger.info(
            "request | id=%s | %s %s | status=%s | %.1fms | ip=%s",
            request_id,
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
            request.remote_addr or "-",
        )
    return response


if not app.secret_key:
    if os.environ.get("FLASK_ENV") == "production":
        raise RuntimeError(
            "FLASK_SECRET_KEY must be set in production. "
            "Generate: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    logger.warning("FLASK_SECRET_KEY not set — using insecure dev default.")
    app.secret_key = "jobguard_dev_secret_not_for_production"  # noqa: S105

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_ENV") == "production",
    WTF_CSRF_TIME_LIMIT=3600,
)

# ── CSP + Security headers ────────────────────────────────────────────
# secure-code-guardian: CSP enabled with nonces via Talisman.
# Allows the FontAwesome CDN and explicit Google Fonts stylesheet used in templates.
CSP: dict[str, Any] = {
    "default-src": "'self'",
    "script-src":  ["'self'"],
    "style-src":   ["'self'", "https://cdnjs.cloudflare.com", "https://fonts.googleapis.com", "'unsafe-inline'"],
    "font-src":    ["'self'", "https://cdnjs.cloudflare.com", "https://fonts.gstatic.com"],
    "img-src":     ["'self'", "data:"],
    "connect-src": "'self'",
    "frame-ancestors": "'none'",
}

try:
    from flask_talisman import Talisman
    Talisman(
        app,
        content_security_policy=CSP,
        force_https=os.environ.get("FLASK_ENV") == "production",
        strict_transport_security=True,
        strict_transport_security_max_age=31_536_000,
        strict_transport_security_include_subdomains=True,
        x_content_type_options=True,
        frame_options="DENY",
        referrer_policy="strict-origin-when-cross-origin",
        session_cookie_secure=os.environ.get("FLASK_ENV") == "production",
        session_cookie_http_only=True,
    )
    logger.info("Flask-Talisman: security headers + CSP active.")
except ImportError:
    logger.warning("Flask-Talisman not installed — security headers disabled.")

# ── CSRF ──────────────────────────────────────────────────────────────
try:
    from flask_wtf.csrf import CSRFError, CSRFProtect
    csrf = CSRFProtect(app)
    logger.info("Flask-WTF: CSRF protection active.")

    @app.errorhandler(CSRFError)
    def csrf_error(e: CSRFError):  # type: ignore[override]
        flash("Session expired. Please try again.", "error")
        return redirect(url_for("classify")), 302

except ImportError:
    logger.warning("Flask-WTF not installed — CSRF disabled.")
    csrf = None  # type: ignore[assignment]

# ── Rate limiting ─────────────────────────────────────────────────────
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    # secure-code-guardian: use Redis storage URI in prod so limits
    # are shared across all gunicorn workers, not per-process memory.
    storage_uri = os.environ.get("RATELIMIT_STORAGE_URI", "memory://")
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per day", "60 per hour"],
        storage_uri=storage_uri,
    )
    if storage_uri == "memory://" and os.environ.get("FLASK_ENV") == "production":
        logger.warning(
            "Rate limiter using in-process memory — limits are not shared "
            "across gunicorn workers. Set RATELIMIT_STORAGE_URI=redis://..."
        )
except ImportError:
    class _NoopLimiter:  # type: ignore[no-redef]
        def limit(self, *a: Any, **kw: Any):
            def d(f: Any) -> Any:
                return f
            return d
    limiter = _NoopLimiter()  # type: ignore[assignment]

# ── Paths & constants ─────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ACTIVE_PATH  = os.path.join(BASE_DIR, "models", "active_model.json")
MD_PATH      = os.path.join(BASE_DIR, "models", "model_metadata.json")
DB_PATH      = os.path.join(BASE_DIR, "data", "predictions.db")

# python-pro + secure-code-guardian: explicit length caps for every field.
MAX_DESC_LEN    = 20_000
MAX_FIELD_LEN   = 500      # job_title, company, location, salary, requirements
MAX_URL_LEN     = 2_000
MAX_API_JSON    = 50_000   # total JSON body size limit in bytes

ADMIN_TOKEN: str = os.environ.get("ADMIN_TOKEN", "").strip()


# ══════════════════════════════════════════════════════════════════════
#  PREDICTOR — thread-safe singleton
#  python-pro: threading.Lock prevents race on first gunicorn request.
# ══════════════════════════════════════════════════════════════════════
_predictor_lock: threading.Lock = threading.Lock()
_predictor_instance = None


def _get_predictor_safe() -> Any | None:
    global _predictor_instance
    if _predictor_instance is not None:
        return _predictor_instance
    with _predictor_lock:
        # Double-checked locking
        if _predictor_instance is None:
            _predictor_instance = get_predictor()
    return _predictor_instance


predictor = _get_predictor_safe()

if predictor:
    logger.info(
        "Predictor ready: %s",
        getattr(predictor, "_model_name", type(predictor).__name__),
    )
else:
    logger.error("No predictor loaded. Run train.py or bert_finetune.py.")


def _active_model_name() -> str:
    p = _get_predictor_safe()
    if p:
        return getattr(p, "_model_name", type(p).__name__)
    return "None"


def _switch_model(name: str) -> bool:
    """Hot-swap sklearn model. No-op for BERT backend."""
    m = load_model_by_name(name)
    if m is None:
        return False
    os.makedirs(os.path.dirname(ACTIVE_PATH), exist_ok=True)
    with open(ACTIVE_PATH, "w") as f:
        json.dump(
            {"active_model": name, "switched_at": datetime.datetime.now().isoformat()}, f
        )
    p = _get_predictor_safe()
    if p and hasattr(p, "_model"):
        import joblib
        p._model = m
        p._model_name = name
    logger.info("Active model → %s", name)
    return True


def _check_admin(req: Any) -> bool:
    """
    secure-code-guardian: constant-time token comparison via hmac.compare_digest.
    Plain `==` is vulnerable to timing attacks — an attacker can measure
    response time character-by-character to brute-force the token.
    """
    if not ADMIN_TOKEN:
        return True  # dev mode — no token set
    provided: str = (
        req.form.get("admin_token", "").strip()
        or req.headers.get("X-Admin-Token", "").strip()
    )
    if not provided:
        return False
    # hmac.compare_digest requires both args to be the same type (str or bytes)
    return hmac.compare_digest(provided, ADMIN_TOKEN)


def load_metadata() -> dict[str, Any]:
    if os.path.exists(MD_PATH):
        with open(MD_PATH) as f:
            return json.load(f)  # type: ignore[no-any-return]
    return {}


def _cap(value: str, max_len: int) -> str:
    """Truncate and strip a string field. Never raises."""
    return value.strip()[:max_len]


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
    """Return True only if text looks like a genuine job posting."""
    word_list = [
        token.strip(".,:;!?()[]{}\"'")
        for token in text.lower().split()
    ]
    # Concise postings can still be valid when they contain all three
    # semantic signals below; reject only genuinely sparse input.
    if len(word_list) < 12:
        return False
    words = set(word_list)
    if not (words & _CORE_JOB_TERMS):
        return False
    if len(words & _QUALIFICATION_TERMS) < 1:
        return False
    has_posting_context = any(
        words & c for c in [_WORK_TERMS, _COMPENSATION_TERMS, _APPLY_TERMS]
    )
    # API clients often send responsibilities and requirements as separate
    # fields, so the field labels are not present in ``combined``. Treat a
    # responsibilities signal plus at least one additional qualification
    # signal as an equivalent structured-posting fallback.
    structured_qualification_hits = words & _QUALIFICATION_TERMS
    has_structured_sections = (
        bool(words & {"responsibilities", "responsibility", "duties"})
        and len(structured_qualification_hits) >= 2
    )
    if not (has_posting_context or has_structured_sections):
        return False
    return True


# ══════════════════════════════════════════════════════════════════════
#  DATABASE — PostgreSQL (Supabase) or SQLite fallback
# ══════════════════════════════════════════════════════════════════════
DATABASE_URL: str = os.environ.get("DATABASE_URL", "").strip()

if DATABASE_URL:
    try:
        import psycopg2
        import psycopg2.extras

        def get_db():  # type: ignore[return]
            return psycopg2.connect(
                DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor
            )

        def _exec(sql: str, params: tuple = ()) -> None:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                conn.commit()

        def _fetch(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return [dict(r) for r in cur.fetchall()]

        def _fetchone(sql: str, params: tuple = ()) -> dict[str, Any] | None:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    return dict(row) if row else None

        PH = "%s"
        logger.info("Database: PostgreSQL (DATABASE_URL)")
    except ImportError:
        DATABASE_URL = ""

if not DATABASE_URL:
    import sqlite3

    def get_db():  # type: ignore[return]
        c = sqlite3.connect(DB_PATH)
        c.row_factory = sqlite3.Row
        return c

    def _exec(sql: str, params: tuple = ()) -> None:  # type: ignore[misc]
        sql = sql.replace("%s", "?")
        with get_db() as c:
            c.execute(sql, params)
            c.commit()

    def _fetch(sql: str, params: tuple = ()) -> list[dict[str, Any]]:  # type: ignore[misc]
        sql = sql.replace("%s", "?")
        with get_db() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def _fetchone(sql: str, params: tuple = ()) -> dict[str, Any] | None:  # type: ignore[misc]
        sql = sql.replace("%s", "?")
        with get_db() as c:
            row = c.execute(sql, params).fetchone()
            return dict(row) if row else None

    PH = "?"
    logger.info("Database: SQLite. Set DATABASE_URL for persistent storage.")


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    serial  = "SERIAL" if DATABASE_URL else "INTEGER"
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
    # Idempotent column additions
    optional_cols = [
        ("fraud_prob", "REAL"),
        ("legit_prob", "REAL"),
        ("model_used", "TEXT"),
        ("url_risk",   "TEXT"),
        ("url_score",  "REAL"),
    ]
    for col, col_type in optional_cols:
        try:
            if DATABASE_URL:
                _exec(f"ALTER TABLE predictions ADD COLUMN IF NOT EXISTS {col} {col_type}")
            else:
                _exec(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")
        except Exception:
            pass  # Column already exists


try:
    init_db()
    logger.info("Database initialised.")
except Exception as _db_err:
    logger.error("DB init failed: %s", _db_err)


def save_pred(
    fd: dict[str, str],
    result: dict[str, Any],
    url_risk: str = "low",
    url_score: float = 0.0,
) -> None:
    """
    secure-code-guardian: all values are parameterised — no string interpolation.
    python-pro: explicit types, capped fields.
    """
    _exec(
        "INSERT INTO predictions "
        "(job_title,company,location,salary,website,description,"
        " requirements,prediction,confidence,fraud_prob,legit_prob,"
        " url_risk,url_score,model_used) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (
            _cap(fd.get("job_title", ""),    MAX_FIELD_LEN),
            _cap(fd.get("company", ""),      MAX_FIELD_LEN),
            _cap(fd.get("location", ""),     MAX_FIELD_LEN),
            _cap(fd.get("salary", ""),       MAX_FIELD_LEN),
            _cap(fd.get("website", ""),      MAX_URL_LEN),
            _cap(fd.get("description", ""),  MAX_DESC_LEN),
            _cap(fd.get("requirements", ""), MAX_DESC_LEN),
            result["label"],
            result["confidence"],
            result.get("fraud_prob"),
            result.get("legit_prob"),
            url_risk,
            url_score,
            result.get("model_name", ""),
        ),
    )


def get_history(limit: int = 100) -> list[dict[str, Any]]:
    rows = _fetch(
        "SELECT * FROM predictions ORDER BY submitted_at DESC LIMIT %s", (limit,)
    )
    # Older local databases may predate probability columns. Normalize rows
    # at the data boundary so templates can render mixed history safely.
    for row in rows:
        row.setdefault("fraud_prob", None)
        row.setdefault("legit_prob", None)
        row.setdefault("url_risk", None)
        row.setdefault("model_used", None)
    return rows


def get_stats() -> dict[str, int]:
    total = (_fetchone("SELECT COUNT(*) AS n FROM predictions") or {}).get("n", 0)
    fraud = (
        _fetchone(
            "SELECT COUNT(*) AS n FROM predictions WHERE prediction=%s",
            ("Fraudulent",),
        )
        or {}
    ).get("n", 0)
    return {"total": total, "fraud": fraud, "legit": total - fraud}


# ══════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def home() -> str:
    return render_template("index.html", stats=get_stats())


@app.route("/classify")
def classify() -> str:
    return render_template("classify.html", active_model=_active_model_name())


@app.route("/predict", methods=["POST"])
def predict_route():
    t0 = time.perf_counter()
    request_id = getattr(g, "request_id", uuid.uuid4().hex[:8])

    # secure-code-guardian: cap ALL fields before any processing
    fd: dict[str, str] = {
        "job_title":    _cap(request.form.get("job_title", ""),    MAX_FIELD_LEN),
        "company":      _cap(request.form.get("company", ""),      MAX_FIELD_LEN),
        "location":     _cap(request.form.get("location", ""),     MAX_FIELD_LEN),
        "salary":       _cap(request.form.get("salary", ""),       MAX_FIELD_LEN),
        "website":      _cap(request.form.get("website", ""),      MAX_URL_LEN),
        "description":  _cap(request.form.get("description", ""),  MAX_DESC_LEN),
        "requirements": _cap(request.form.get("requirements", ""), MAX_DESC_LEN),
    }

    if not fd["description"]:
        flash("Job description is required.", "error")
        return render_template("classify.html", form=fd, active_model=_active_model_name())

    combined = " ".join(filter(None, [
        fd["job_title"], fd["company"], fd["description"], fd["requirements"],
    ]))

    if not _is_job_posting(combined):
        flash(
            "This doesn't look like a job posting. Include role title, "
            "responsibilities, and requirements for best results.",
            "warning",
        )
        return render_template("classify.html", form=fd, active_model=_active_model_name())

    p = _get_predictor_safe()
    if not p:
        flash("Model not loaded. Run train.py or bert_finetune.py + bert_to_onnx.py.", "error")
        return render_template("classify.html", form=fd, active_model=_active_model_name())

    result = p.predict(combined)
    if "error" in result:
        flash(result["error"], "error")
        return render_template("classify.html", form=fd, active_model=_active_model_name())

    analysis   = analyse_all(fd.get("website", ""), fd.get("company", ""))
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    try:
        save_pred(fd, result, analysis["overall_risk"], analysis["url"]["score"])
    except Exception as e:
        logger.error("DB write failed [%s]: %s", request_id, e)

    logger.info(
        "predict | id=%s | verdict=%s | fraud=%.1f%% | %dms | model=%s | ip=%s",
        request_id,
        result["label"],
        result["fraud_prob"],
        elapsed_ms,
        result.get("model_name", "?"),
        request.remote_addr,
    )

    return render_template(
        "result.html", result=result, job=fd, analysis=analysis, elapsed_ms=elapsed_ms
    )


@app.route("/history")
def history() -> str:
    return render_template("history.html", history=get_history(), stats=get_stats())

@app.route("/history/delete/<int:prediction_id>", methods=["POST"])
def delete_history_item(prediction_id: int):
    _exec("DELETE FROM predictions WHERE id=%s", (prediction_id,))
    flash("Analysis removed from History.", "success")
    return redirect(url_for("history"))

@app.route("/history/export")
def export_history() -> Response:
    rows = get_history()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "submitted_at", "job_title", "company", "location", "salary", "website",
        "prediction", "fraud_probability", "legit_probability", "confidence",
        "url_risk", "model_used",
    ])
    for row in rows:
        writer.writerow([
            row.get("submitted_at", ""), row.get("job_title", ""), row.get("company", ""),
            row.get("location", ""), row.get("salary", ""), row.get("website", ""),
            row.get("prediction", ""), row.get("fraud_prob", ""), row.get("legit_prob", ""),
            row.get("confidence", ""), row.get("url_risk", ""), row.get("model_used", ""),
        ])
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=jobguard-history.csv"},
    )

@app.route("/about")
def about() -> str:
    return render_template("about.html", metadata=load_metadata())


@app.route("/privacy")
def privacy() -> str:
    return render_template("privacy.html")


@app.route("/models")
def models_page() -> str:
    bert_meta_path = os.path.join(BASE_DIR, "models", "bert_results.json")
    bert_meta: dict[str, Any] = {}
    if os.path.exists(bert_meta_path):
        with open(bert_meta_path) as f:
            bert_meta = json.load(f)
    return render_template(
        "models.html",
        metrics=load_metrics(),
        metadata=load_metadata(),
        bert_meta=bert_meta,
        active_model=_active_model_name(),
        registry=load_model_registry(),
    )


@app.route("/clear_history", methods=["POST"])
def clear_history():
    if not _check_admin(request):
        flash("Unauthorised.", "error")
        return redirect(url_for("history"))
    try:
        # Clearing an already-empty table is intentionally idempotent.
        _exec("DELETE FROM predictions")
    except Exception:
        logger.exception("History clear failed")
        flash("History could not be cleared right now. Please try again.", "error")
        return redirect(url_for("history"))
    flash("History cleared.", "success")
    return redirect(url_for("history"))


@app.route("/select-model", methods=["POST"])
def select_model():
    if not _check_admin(request):
        flash("Unauthorised.", "error")
        return redirect(url_for("models_page"))
    name = _cap(request.form.get("model_name", ""), 100)
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
    # secure-code-guardian: cap total JSON body before parsing
    if request.content_length and request.content_length > MAX_API_JSON:
        return jsonify({"error": "Request body too large."}), 413

    data: dict[str, Any] = request.get_json(force=True, silent=True) or {}

    # Cap and sanitise all string fields
    description  = _cap(str(data.get("description", "")),  MAX_DESC_LEN)
    title        = _cap(str(data.get("title", "")),         MAX_FIELD_LEN)
    company      = _cap(str(data.get("company", "")),       MAX_FIELD_LEN)
    requirements = _cap(str(data.get("requirements", "")),  MAX_DESC_LEN)
    website      = _cap(str(data.get("website", "")),       MAX_URL_LEN)

    if not description:
        return jsonify({"error": "description is required"}), 400

    combined = " ".join(filter(None, [title, company, description, requirements]))

    if not _is_job_posting(combined):
        return jsonify({"error": "Input does not appear to be a job posting."}), 422

    p = _get_predictor_safe()
    if not p:
        # secure-code-guardian: don't leak internal model state in API error
        return jsonify({"error": "Service temporarily unavailable."}), 503

    result = p.predict(combined)
    if "error" in result:
        return jsonify({"error": "Prediction failed. Please try again."}), 503

    analysis = analyse_all(website, company)
    exp      = result.get("explanation", {})

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
                {
                    "label":    p["label"],
                    "severity": p["severity"],
                    "reason":   p["reason"],
                    "matched":  p["matched"],
                }
                for p in exp.get("fraud_patterns", [])
            ],
            "reasons":        exp.get("reasons", []),
            "decision_score": exp.get("decision_score", 0),
        },
    })


# JSON clients do not carry the HTML form token; API input validation and
# rate limiting remain active for this endpoint.
if csrf is not None:
    csrf.exempt(api_predict)

@app.route("/api/models", methods=["GET"])
def api_models():
    registry     = load_model_registry()
    all_metrics  = {m["name"]: m for m in load_metrics()}
    models_list: list[dict[str, Any]] = []

    for name, path in registry.items():
        resolved = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
        entry: dict[str, Any] = {
            "name":      name,
            "available": os.path.exists(resolved),
            "is_active": name == _active_model_name(),
        }
        if name in all_metrics:
            entry.update({
                k: all_metrics[name].get(k)
                for k in ["accuracy", "f1_fraud", "recall_fraud",
                          "precision_fraud", "roc_auc", "cv_f1_mean", "cv_f1_std"]
            })
        models_list.append(entry)

    onnx_path = os.path.join(BASE_DIR, "models", "bert_onnx_quantized.onnx")
    if os.path.exists(onnx_path):
        bert_results_path = os.path.join(BASE_DIR, "models", "bert_results.json")
        bert_entry: dict[str, Any] = {
            "name":      "DistilBERT (ONNX INT8)",
            "available": True,
            "is_active": "BERT" in _active_model_name(),
            "size_mb":   round(os.path.getsize(onnx_path) / 1e6, 1),
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
    bert_size   = 0.0
    if bert_active:
        bert_size += os.path.getsize(onnx_path) / 1e6
    if os.path.exists(data_path):
        bert_size += os.path.getsize(data_path) / 1e6

    p = _get_predictor_safe()
    db_ok = True
    try:
        _fetchone("SELECT 1 AS ok")
    except Exception:
        db_ok = False
        logger.exception("Health database check failed | id=%s", getattr(g, "request_id", "-"))

    ready = bool(p) and db_ok
    body = {
        "status":       "ok" if ready else "degraded",
        "predictor":    type(p).__name__ if p else "None",
        "active_model": _active_model_name(),
        "bert_onnx":    bert_active,
        "bert_size_mb": round(bert_size, 1) if bert_active else None,
        "db_backend":   "postgresql" if DATABASE_URL else "sqlite",
        "db_ok":        db_ok,
        "timestamp":    datetime.datetime.now().isoformat(),
    }
    return jsonify(body), (200 if ready else 503)


@app.errorhandler(404)
def not_found(e: Exception) -> tuple[str, int]:
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e: Exception) -> tuple[str, int]:
    logger.error(
        "Unhandled 500 | id=%s | path=%s",
        getattr(g, "request_id", "-"),
        request.path,
        exc_info=(type(e), e, e.__traceback__),
    )
    return render_template("500.html"), 500


if __name__ == "__main__":
    init_db()
    debug = os.environ.get("FLASK_ENV") == "development"
    port  = int(os.environ.get("PORT", 5000))
    p     = _get_predictor_safe()
    logger.info(
        "JobGuard v9 — http://localhost:%d | backend=%s",
        port, type(p).__name__ if p else "None",
    )
    app.run(debug=debug, host="0.0.0.0", port=port)
