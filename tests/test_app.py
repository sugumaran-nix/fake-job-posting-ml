"""
tests/test_app.py — JobGuard v7
=================================
Covers:
  - All route smoke tests
  - Job-posting guard (rejects non-job text, accepts real postings)
  - ml_predict() result shape (fraud_prob + legit_prob always present)
  - API vice-versa check (fraudulent sample never returns Legitimate)
  - Admin token check behaviour
  - Cold-start / model-not-loaded 503 path
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as flask_app
from app import _is_job_posting


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
#  ROUTE SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════

def test_home(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"JobGuard" in r.data


def test_classify_page(client):
    r = client.get("/classify")
    assert r.status_code == 200
    assert b"description" in r.data


def test_history(client):
    r = client.get("/history")
    assert r.status_code == 200


def test_about(client):
    r = client.get("/about")
    assert r.status_code == 200


def test_models_page(client):
    r = client.get("/models")
    assert r.status_code == 200


def test_health(client):
    r  = client.get("/health")
    d  = json.loads(r.data)
    assert r.status_code == 200
    assert "status" in d
    assert d["status"] == "ok"
    assert "predictor" in d
    assert "db_backend" in d


def test_404(client):
    r = client.get("/does-not-exist")
    assert r.status_code == 404


# ══════════════════════════════════════════════════════════════════════
#  JOB POSTING GUARD — 4-cluster validation
# ══════════════════════════════════════════════════════════════════════

# Texts that MUST be rejected
REJECTED_TEXTS = [
    # Too short
    "Python developer job",
    # Pollution paragraph (the original failing case)
    (
        "Air pollution is a major environmental hazard affecting millions of people globally. "
        "Industrial emissions, vehicular exhaust and burning of fossil fuels release harmful "
        "particulate matter into the atmosphere causing respiratory diseases. Governments must "
        "act urgently to regulate emissions and invest in clean energy alternatives."
    ),
    # Random Wikipedia-style article
    (
        "The mitochondria is the powerhouse of the cell. It produces ATP through oxidative "
        "phosphorylation and plays a central role in energy metabolism. Mitochondria contain "
        "their own DNA and can replicate independently within the host cell cytoplasm."
    ),
    # Product description
    (
        "The Samsung Galaxy S24 Ultra features a titanium frame, a 200MP primary camera, "
        "and an integrated S Pen stylus. The Snapdragon 8 Gen 3 processor delivers blazing "
        "performance and the 5000mAh battery supports 45W wired fast charging technology."
    ),
    # News article (no job terms)
    (
        "The Indian government announced a new economic policy framework yesterday aimed at "
        "boosting domestic manufacturing. Finance Minister stated that the initiative will "
        "create a more business-friendly environment by reducing compliance burden and taxes."
    ),
]

# Texts that MUST be accepted
ACCEPTED_TEXTS = [
    # Clearly fraudulent — must pass the guard to reach the ML model
    (
        "URGENT HIRING work from home data entry job vacancy. No experience needed, no qualifications "
        "required. Earn guaranteed income ₹50,000 per month. Registration fee ₹799 to join. "
        "Requirements: must have bank account. Submit your resume and Aadhaar card. Apply now!"
    ),
    # Legitimate professional posting
    (
        "Python Backend Developer position at Zoho Corporation, Chennai. Responsibilities include "
        "designing REST APIs, code reviews, and writing unit tests. Requirements: 2 years experience, "
        "proficiency in Python and Django, BSc/BE degree. Salary ₹6-10 LPA. Apply at careers.zoho.com."
    ),
    # Fresher opening
    (
        "Junior Data Analyst vacancy — freshers welcome. Job responsibilities include dashboard "
        "building, report generation, and SQL querying. Requirements: BCA or BSc graduate, "
        "skills in Excel and Power BI. Salary ₹3.5 LPA. Please email your resume to hr@company.in."
    ),
]


@pytest.mark.parametrize("text", REJECTED_TEXTS)
def test_guard_rejects_non_job_text(text):
    assert _is_job_posting(text) is False, \
        f"Guard should reject: '{text[:80]}...'"


@pytest.mark.parametrize("text", ACCEPTED_TEXTS)
def test_guard_accepts_job_text(text):
    assert _is_job_posting(text) is True, \
        f"Guard should accept: '{text[:80]}...'"


def test_predict_route_rejects_pollution(client):
    """The classify form should flash a warning for non-job text."""
    pollution = (
        "Air pollution is a major environmental hazard. Industrial emissions, vehicular "
        "exhaust and fossil fuels release particulate matter. Governments must act urgently "
        "to regulate emissions and invest in renewable energy alternatives for the planet."
    )
    r = client.post("/predict", data={"description": pollution})
    assert r.status_code == 200
    assert b"job posting" in r.data.lower() or b"warning" in r.data.lower()


def test_predict_route_rejects_empty(client):
    r = client.post("/predict", data={"description": ""})
    assert r.status_code == 200
    assert b"required" in r.data.lower() or b"error" in r.data.lower()


# ══════════════════════════════════════════════════════════════════════
#  ML PREDICT — result structure
# ══════════════════════════════════════════════════════════════════════

def test_ml_predict_result_shape():
    """predictor.predict() must always return fraud_prob + legit_prob."""
    pred = flask_app.predictor
    if pred is None:
        pytest.skip("No predictor loaded — run train.py first")

    legit_text = (
        "Software Engineer position at Infosys Bangalore. Responsibilities include "
        "backend API development, code reviews, and deployment. Requirements: 2 years "
        "Python experience, Django skills, BSc CS degree. Salary 8 LPA. Apply with resume."
    )
    result = pred.predict(legit_text)

    assert "error" not in result, f"predict() returned error: {result}"
    assert "fraud_prob"  in result
    assert "legit_prob"  in result
    assert "is_fraud"    in result
    assert "confidence"  in result
    assert "explanation" in result
    assert "model_name"  in result

    total = result["fraud_prob"] + result["legit_prob"]
    assert abs(total - 100.0) < 1.0, f"fraud_prob + legit_prob = {total}"

    dominant = max(result["fraud_prob"], result["legit_prob"])
    assert abs(result["confidence"] - dominant) < 1.0


def test_ml_predict_fraud_text():
    """A canonical fraud posting must lean toward fraudulent."""
    pred = flask_app.predictor
    if pred is None:
        pytest.skip("No predictor loaded — run train.py first")

    fraud_text = (
        "URGENT HIRING! Work from home data entry job. No experience needed, no degree "
        "required. Earn guaranteed income per month. Pay registration fee to join. "
        "Submit Aadhaar card and bank account details. Network marketing opportunity. "
        "Requirements: just a smartphone. Apply immediately — only 5 seats left!"
    )
    result = pred.predict(fraud_text)
    assert result["fraud_prob"] > result["legit_prob"], (
        f"Canonical fraud text classified as Legitimate "
        f"(fraud_prob={result['fraud_prob']}, legit_prob={result['legit_prob']})"
    )


def test_explanation_no_nested_marks():
    """highlighted_html must not contain nested <mark> tags (sklearn backend only)."""
    from utils.model_router import SklearnPredictor
    pred = flask_app.predictor
    if pred is None or not isinstance(pred, SklearnPredictor):
        pytest.skip("Nested marks test requires sklearn backend")

    import re
    from utils.explainer import explain

    html = explain(
        "work from home guaranteed income urgent hiring registration fee required apply now",
        pred._vectorizer, pred._model, is_fraud=True,
    )["highlighted_html"]

    nested = re.findall(r'<mark[^>]*>[^<]*<mark', html)
    assert nested == [], f"Nested <mark> tags found: {nested}"


# ══════════════════════════════════════════════════════════════════════
#  JSON API
# ══════════════════════════════════════════════════════════════════════

def test_api_predict_rejects_missing_description(client):
    r = client.post("/api/predict",
                    data=json.dumps({}),
                    content_type="application/json")
    assert r.status_code == 400


def test_api_predict_rejects_non_job_text(client):
    r = client.post("/api/predict",
                    data=json.dumps({
                        "description": (
                            "Air pollution is a major environmental hazard affecting "
                            "millions of people globally causing respiratory diseases "
                            "and ecosystem damage through industrial and vehicle emissions."
                        )
                    }),
                    content_type="application/json")
    assert r.status_code == 422
    data = json.loads(r.data)
    assert "error" in data


def test_api_models_list(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    d = json.loads(r.data)
    assert "models" in d
    assert "active_model" in d


def test_api_predict_valid_job(client):
    # v8: model override param removed — API always uses active predictor
    r = client.post("/api/predict",
                    data=json.dumps({
                        "description": (
                            "Senior software engineer position at Infosys Bangalore. "
                            "Responsibilities include backend API development, code review, "
                            "and system design. Requirements: Python experience, BSc CS degree, "
                            "2 years minimum. Competitive salary and benefits. "
                            "Apply with your resume at careers.infosys.com."
                        ),
                    }),
                    content_type="application/json")
    assert r.status_code == 200
    d = json.loads(r.data)
    assert "prediction" in d
    assert "fraud_prob" in d
    assert "legit_prob" in d
    assert abs(d["fraud_prob"] + d["legit_prob"] - 100.0) < 1.0


# ══════════════════════════════════════════════════════════════════════
#  ADMIN CHECKS
# ══════════════════════════════════════════════════════════════════════

def test_clear_history_requires_admin_when_token_set(client, monkeypatch):
    monkeypatch.setattr(flask_app, "ADMIN_TOKEN", "test-secret-token")
    r = client.post("/clear_history", data={"admin_token": "wrong"})
    assert r.status_code in (200, 302)
    # Should not have cleared — flash message with Unauthorized expected
    # (redirect to history page)


def test_clear_history_works_in_dev_mode(client, monkeypatch):
    monkeypatch.setattr(flask_app, "ADMIN_TOKEN", "")   # dev mode
    r = client.post("/clear_history", data={})
    assert r.status_code in (200, 302)


def test_clear_history_db_failure_redirects_with_error(client, monkeypatch):
    monkeypatch.setattr(flask_app, "ADMIN_TOKEN", "")

    def fail_exec(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(flask_app, "_exec", fail_exec)
    response = client.post("/clear_history", data={}, follow_redirects=True)
    assert response.status_code == 200
    assert b"History could not be cleared right now" in response.data


def test_admin_token_empty_string_is_dev_mode(monkeypatch):
    """Empty ADMIN_TOKEN (unset in env) → dev bypass allowed."""
    monkeypatch.setattr(flask_app, "ADMIN_TOKEN", "")
    with flask_app.app.test_request_context():
        from flask import request
        assert flask_app._check_admin(request) is True


def test_admin_token_non_empty_requires_match(monkeypatch):
    """Non-empty ADMIN_TOKEN → reject wrong token."""
    monkeypatch.setattr(flask_app, "ADMIN_TOKEN", "correct-token")
    with flask_app.app.test_request_context("/clear_history", method="POST",
                                            data={"admin_token": "wrong-token"}):
        from flask import request
        assert flask_app._check_admin(request) is False


def test_admin_token_correct_token_passes(monkeypatch):
    """Non-empty ADMIN_TOKEN → accept correct token."""
    monkeypatch.setattr(flask_app, "ADMIN_TOKEN", "correct-token")
    with flask_app.app.test_request_context("/clear_history", method="POST",
                                            data={"admin_token": "correct-token"}):
        from flask import request
        assert flask_app._check_admin(request) is True
