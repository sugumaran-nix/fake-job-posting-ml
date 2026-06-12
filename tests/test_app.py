"""
tests/test_app.py
Flask route smoke tests. These run without a trained model — they test
routing, form validation, and JSON API structure.
Run: pytest tests/test_app.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import json


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a test Flask client with a temp DB and no real model."""
    # Point DB to a temp path so tests don't touch production data
    monkeypatch.setenv("FLASK_SECRET_KEY", "test-secret-key")
    import app as flask_app
    flask_app.DB_PATH = str(tmp_path / "test.db")
    flask_app.init_db()
    flask_app.app.config["TESTING"] = True
    flask_app.app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.app.test_client() as c:
        yield c


class TestRoutes:

    def test_home_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_classify_returns_200(self, client):
        r = client.get("/classify")
        assert r.status_code == 200

    def test_history_returns_200(self, client):
        r = client.get("/history")
        assert r.status_code == 200

    def test_about_returns_200(self, client):
        r = client.get("/about")
        assert r.status_code == 200

    def test_models_page_returns_200(self, client):
        r = client.get("/models")
        assert r.status_code == 200

    def test_health_returns_json(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert "status" in data
        assert data["status"] == "ok"

    def test_predict_requires_description(self, client):
        """Submitting without a description should not crash — flash error and redisplay form."""
        r = client.post("/predict", data={"description": ""})
        # Should redirect back or re-render classify page, not 500
        assert r.status_code in (200, 302)

    def test_api_predict_missing_description(self, client):
        r = client.post(
            "/api/predict",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert r.status_code == 400
        data = json.loads(r.data)
        assert "error" in data

    def test_api_models_returns_json(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert "active_model" in data
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_api_predict_no_model_returns_503(self, client):
        """Without a trained model, /api/predict should return 503, not 500."""
        r = client.post(
            "/api/predict",
            data=json.dumps({"description": "Test job description here."}),
            content_type="application/json",
        )
        # Either 503 (no model) or 200 (if test env somehow has a model)
        assert r.status_code in (200, 503)
