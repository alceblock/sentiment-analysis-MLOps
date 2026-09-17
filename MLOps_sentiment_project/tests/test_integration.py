import pytest
from fastapi.testclient import TestClient
from model_app.model_inference import app, window, conf_matrix

client = TestClient(app)


def test_predict_positive():
    data = {
        "text": "I love this project. It's awesome!",
        "label": "positive"
    }

    response = client.post("/predict", json=data)
    body = response.json()

    # response
    assert response.status_code == 200

    assert set(body.keys()) == {"label", "confidence", "user_provided_label"}

    # label
    assert isinstance(body["label"], str)
    assert body["label"].lower() in ["positive", "neutral", "negative"]

    # user label
    assert body["user_provided_label"] == "positive"

    # confidence
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_without_label():
    data = {
        "text": "This is just okay."
    }

    response = client.post("/predict", json=data)
    body = response.json()

    assert response.status_code == 200
    assert body["user_provided_label"] is None


def test_invalid_input_missing_text():
    data = {
        "label": "positive"
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 422  # FastAPI validation error


def test_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    content = response.text

    # health Prometheus metrics
    assert "http_requests_total" in content.lower()
    assert "request_latency" in content.lower()


def test_accuracy_window_updates():
    window.clear()

    samples = [
        ("I love this", "positive"),
        ("I hate this", "negative"),
        ("Average", "neutral")
    ]

    for text, label in samples:
        client.post("/predict", json={"text": text, "label": label})

    assert len(window) == len(samples)
    assert all(x in [0, 1] for x in window)
