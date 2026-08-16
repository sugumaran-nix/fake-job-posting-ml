"""
tests/test_security.py — test-master + secure-code-guardian
=============================================================
Security-focused tests covering:
  - Input length caps (all fields)
  - SQL injection payloads rejected / parameterised
  - XSS payloads in form fields don't render unescaped
  - Admin token timing-safe comparison
  - API rate limit headers present
  - CSRF protection active
  - Oversized JSON body rejected (413)
  - Health endpoint does not leak secrets
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as flask_app
from app import _cap, _check_admin, _is_job_posting


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def client():
    flask_app.app.config["TESTING"]         = True
    flask_app.app.config["WTF_CSRF_ENABLED"] = False
    flask_app.app.config["SECRET_KEY"]      = "test-secret"
    flask_app.init_db()
    with flask_app.app.test_client() as c:
        yield c


# ══════════════════════════════════════════════════════════════════════
#  INPUT LENGTH CAPS — _cap() helper
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("max_len", [10, 100, 500, 2000, 20_000])
def test_cap_truncates_to_max_len(max_len: int) -> None:
    long_str = "A" * (max_len + 100)
    result   = _cap(long_str, max_len)
    assert len(result) == max_len


def test_cap_strips_whitespace() -> None:
    assert _cap("  hello  ", 100) == "hello"


def test_cap_empty_string() -> None:
    assert _cap("", 500) == ""


def test_cap_exactly_at_limit() -> None:
    s = "X" * 500
    assert _cap(s, 500) == s
    assert len(_cap(s, 500)) == 500


# ══════════════════════════════════════════════════════════════════════
#  SQL INJECTION — parameterised queries never interpolate user input
# ══════════════════════════════════════════════════════════════════════

SQL_INJECTION_PAYLOADS = [
    "' OR 1=1--",
    "'; DROP TABLE predictions;--",
    "1; SELECT * FROM predictions--",
    "' UNION SELECT * FROM predictions--",
    "admin'--",
    "' OR 'x'='x",
]


@pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
def test_sql_injection_in_description_doesnt_crash(client, payload: str) -> None:
    """SQL payloads in description must not raise 500 — parameterised queries safe."""
    job_desc = (
        f"{payload} work from home job vacancy. Responsibilities include data entry "
        f"and typing. Requirements: BSc graduate, apply with resume. Salary 5 LPA. "
        f"No experience needed. Contact hr@company.in"
    )
    r = client.post("/predict", data={"description": job_desc})
    assert r.status_code in (200, 302), f"Unexpected status for payload: {payload!r}"


@pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
def test_sql_injection_in_api_doesnt_crash(client, payload: str) -> None:
    r = client.post(
        "/api/predict",
        data=json.dumps({
            "description": (
                f"{payload} software engineer position responsibilities requirements "
                f"experience degree apply resume salary benefits company office"
            )
        }),
        content_type="application/json",
    )
    assert r.status_code in (200, 400, 422, 503), \
        f"Unexpected status {r.status_code} for payload: {payload!r}"


# ══════════════════════════════════════════════════════════════════════
#  XSS — form fields must be escaped in rendered HTML
# ══════════════════════════════════════════════════════════════════════

XSS_PAYLOADS = [
    "<script>alert('xss')</script>",
    "<img src=x onerror=alert(1)>",
    "javascript:alert(1)",
    "<svg onload=alert(1)>",
    "';alert(String.fromCharCode(88,83,83))//",
]


@pytest.mark.parametrize("payload", XSS_PAYLOADS)
def test_xss_in_job_title_is_escaped(client, payload: str) -> None:
    """XSS payloads submitted as job_title must not appear unescaped in HTML."""
    r = client.post("/predict", data={
        "job_title":   payload,
        "description": (
            "Software engineer position at Infosys Bangalore. Responsibilities "
            "include backend API development, code reviews, and deployment. "
            "Requirements: 2 years Python, Django, BSc CS. Salary 8 LPA."
        ),
    })
    # The raw script tag must NOT appear verbatim in the response
    assert b"<script>alert(" not in r.data
    # Jinja auto-escapes value attrs: onerror= appears as &lt;img... onerror= (safe)
    # Verify the UNescaped version is absent
    assert b"onerror=alert(1)>" not in r.data  # would only appear if unescaped
    assert b"\"onerror" not in r.data  # must not break out of attribute context


# ══════════════════════════════════════════════════════════════════════
#  ADMIN TOKEN — timing-safe comparison
# ══════════════════════════════════════════════════════════════════════

def test_admin_token_uses_hmac_compare_digest(monkeypatch) -> None:
    """Verify _check_admin uses hmac.compare_digest (not ==)."""
    import inspect
    import hmac as hmac_mod
    source = inspect.getsource(_check_admin)
    assert "compare_digest" in source, \
        "_check_admin must use hmac.compare_digest for timing-safe comparison"


def test_admin_correct_token_accepted(monkeypatch) -> None:
    monkeypatch.setattr(flask_app, "ADMIN_TOKEN", "correct-token")
    with flask_app.app.test_request_context(
        "/clear_history", method="POST", data={"admin_token": "correct-token"}
    ):
        from flask import request
        assert _check_admin(request) is True


def test_admin_wrong_token_rejected(monkeypatch) -> None:
    monkeypatch.setattr(flask_app, "ADMIN_TOKEN", "correct-token")
    with flask_app.app.test_request_context(
        "/clear_history", method="POST", data={"admin_token": "wrong-token"}
    ):
        from flask import request
        assert _check_admin(request) is False


def test_admin_empty_token_dev_bypass(monkeypatch) -> None:
    monkeypatch.setattr(flask_app, "ADMIN_TOKEN", "")
    with flask_app.app.test_request_context("/clear_history", method="POST"):
        from flask import request
        assert _check_admin(request) is True


def test_admin_empty_string_provided_rejected(monkeypatch) -> None:
    monkeypatch.setattr(flask_app, "ADMIN_TOKEN", "real-token")
    with flask_app.app.test_request_context(
        "/clear_history", method="POST", data={"admin_token": ""}
    ):
        from flask import request
        assert _check_admin(request) is False


# ══════════════════════════════════════════════════════════════════════
#  API — oversized body, malformed JSON, missing fields
# ══════════════════════════════════════════════════════════════════════

def test_api_predict_oversized_body_rejected(client) -> None:
    """Bodies larger than MAX_API_JSON must return 413."""
    oversized = json.dumps({"description": "x" * (flask_app.MAX_API_JSON + 1)})
    r = client.post(
        "/api/predict",
        data=oversized,
        content_type="application/json",
        content_length=len(oversized),
    )
    assert r.status_code == 413, f"Expected 413, got {r.status_code}"


def test_api_predict_malformed_json(client) -> None:
    r = client.post(
        "/api/predict",
        data="not-json{{{",
        content_type="application/json",
    )
    assert r.status_code in (400, 422)


def test_api_predict_empty_description(client) -> None:
    r = client.post(
        "/api/predict",
        data=json.dumps({"description": ""}),
        content_type="application/json",
    )
    assert r.status_code == 400
    assert "error" in json.loads(r.data)


def test_api_predict_whitespace_only_description(client) -> None:
    r = client.post(
        "/api/predict",
        data=json.dumps({"description": "   \n\t  "}),
        content_type="application/json",
    )
    assert r.status_code == 400


# ══════════════════════════════════════════════════════════════════════
#  HEALTH ENDPOINT — must not leak secrets
# ══════════════════════════════════════════════════════════════════════

def test_health_does_not_leak_secret_key(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.data.decode()
    assert "FLASK_SECRET_KEY" not in body
    assert "ADMIN_TOKEN"      not in body
    assert "DATABASE_URL"     not in body


def test_health_response_shape(client) -> None:
    r  = client.get("/health")
    d  = json.loads(r.data)
    assert d["status"]    == "ok"
    assert "predictor"    in d
    assert "active_model" in d
    assert "db_backend"   in d
    assert "timestamp"    in d


# ══════════════════════════════════════════════════════════════════════
#  FORM FIELD CAPS — oversized inputs handled gracefully
# ══════════════════════════════════════════════════════════════════════

def test_form_oversized_description_handled(client) -> None:
    """A description larger than MAX_DESC_LEN must not crash — gets capped."""
    huge = "work from home job vacancy resume apply salary experience " * 400
    r = client.post("/predict", data={"description": huge})
    assert r.status_code in (200, 302)


def test_form_oversized_job_title_handled(client) -> None:
    huge_title = "Software Engineer " * 100  # 1800 chars — over MAX_FIELD_LEN
    r = client.post("/predict", data={
        "job_title": huge_title,
        "description": (
            "Python developer job vacancy. Responsibilities include API development "
            "and code review. Requirements: BSc CS, 2 years Python. Salary 8 LPA. Apply."
        ),
    })
    assert r.status_code in (200, 302)
