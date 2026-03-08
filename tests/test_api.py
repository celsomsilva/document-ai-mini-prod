"""
API contract tests.

Validates the public behavior of the FastAPI application
without depending on index artifacts on disk.
"""

from fastapi.testclient import TestClient
from unittest.mock import patch

from mlops_api.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    # Mock the predict() function to keep tests independent from artifacts
    with patch("mlops_api.api.predict") as mock_predict:
        mock_predict.return_value = {
            "prediction": "Deductible is $5,000.",
            "model_version": "2026-02-23T00:00:00Z",
            "rmse": 0.99,
            "citations": [
                {"doc_id": "sample-policy-001", "page": 2, "chunk_id": "x", "snippet": "Schedule: deductible...", "score": 0.99}
            ],
        }

        response = client.post(
            "/predict",
            json={"question": "What is the deductible?", "top_k": 3},
        )

        assert response.status_code == 200
        body = response.json()
        assert "prediction" in body
        assert "model_version" in body
        assert "rmse" in body
        assert "citations" in body


def test_predict_invalid_question():
    response = client.post("/predict", json={"question": "", "top_k": 3})
    assert response.status_code == 400

