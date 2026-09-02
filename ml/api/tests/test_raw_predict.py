"""Tests for the raw-transaction ML prediction endpoint.

Covers:
  A.  Valid raw transaction returns HTTP 200
  B.  Response contains all expected fields
  C.  Probability in [0, 1]
  D.  Prediction follows threshold 0.50
  E.  Deterministic (repeated calls identical)
  F.  Missing required field -> 422
  G.  Negative amount -> 422
  H.  Invalid timestamp (negative) -> 422
  I.  Invalid transaction_type -> 422
  J.  Invalid device_type -> 422
  K.  isFraud supplied -> 422
  L.  TransactionID supplied -> 422
  M.  Model unavailable -> 503

Run from the project root::

    python -m pytest ml/api/tests/test_raw_predict.py -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ml.api.app import app, _predictor
import ml.features.history as _history_module

# ── Valid raw transaction fixture ─────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_history():
    """Clear history store before each test for isolation."""
    _history_module.history_store.clear()
    yield
    _history_module.history_store.clear()


def _valid_raw_transaction() -> dict:
    """Minimal valid raw transaction matching backend TransactionCreate."""
    return {
        "amount": 150.0,
        "currency": "USD",
        "merchant_name": "Test Merchant",
        "merchant_category": "5732",
        "transaction_type": "purchase",
        "location_country": "US",
        "location_city": "New York",
        "device_fingerprint": "abc123def456",
        "device_type": "mobile",
        "ip_address": "192.168.1.100",
    }


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with model loaded via lifespan."""
    with TestClient(app) as c:
        yield c


def _model_available(client: TestClient) -> bool:
    """Check if the model is available before running tests."""
    resp = client.get("/health")
    return resp.json().get("status") == "ready"


# ── A. Valid raw transaction -> 200 ──────────────────────────────────


def test_valid_raw_transaction_200(client: TestClient):
    if not _model_available(client):
        pytest.skip("Model not available")
    resp = client.post("/predict", json=_valid_raw_transaction())
    assert resp.status_code == 200, resp.text


# ── B. Response contains all expected fields ──────────────────────────


def test_response_fields(client: TestClient):
    if not _model_available(client):
        pytest.skip("Model not available")
    resp = client.post("/predict", json=_valid_raw_transaction())
    data = resp.json()
    assert "fraud_probability" in data
    assert "fraud_prediction" in data
    assert "threshold" in data
    assert "model_version" in data
    assert "explanation" in data
    # Explanation is a list of factor dicts
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0
    for factor in data["explanation"]:
        assert "feature" in factor
        assert "importance" in factor


# ── C. Probability in [0, 1] ─────────────────────────────────────────


def test_probability_range(client: TestClient):
    if not _model_available(client):
        pytest.skip("Model not available")
    resp = client.post("/predict", json=_valid_raw_transaction())
    prob = resp.json()["fraud_probability"]
    assert 0.0 <= prob <= 1.0


# ── D. Prediction follows threshold 0.50 ─────────────────────────────


def test_prediction_threshold(client: TestClient):
    if not _model_available(client):
        pytest.skip("Model not available")
    resp = client.post("/predict", json=_valid_raw_transaction())
    data = resp.json()
    threshold = data["threshold"]
    prob = data["fraud_probability"]
    pred = data["fraud_prediction"]
    assert threshold == 0.50
    if prob >= threshold:
        assert pred == 1
    else:
        assert pred == 0


# ── E. Deterministic (repeated calls) ────────────────────────────────


def test_deterministic(client: TestClient):
    if not _model_available(client):
        pytest.skip("Model not available")
    payload = _valid_raw_transaction()
    resp1 = client.post("/predict", json=payload)
    resp2 = client.post("/predict", json=payload)
    d1 = resp1.json()
    d2 = resp2.json()
    assert d1["fraud_probability"] == d2["fraud_probability"]
    assert d1["fraud_prediction"] == d2["fraud_prediction"]
    assert d1["model_version"] == d2["model_version"]
    # Explanation should be identical too
    assert d1["explanation"] == d2["explanation"]


# ── F. Missing required field -> 422 ─────────────────────────────────


def test_missing_amount(client: TestClient):
    payload = _valid_raw_transaction()
    del payload["amount"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_missing_currency(client: TestClient):
    payload = _valid_raw_transaction()
    del payload["currency"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_missing_merchant_name(client: TestClient):
    payload = _valid_raw_transaction()
    del payload["merchant_name"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


# ── G. Negative amount -> 422 ────────────────────────────────────────


def test_negative_amount(client: TestClient):
    payload = _valid_raw_transaction()
    payload["amount"] = -10.0
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_zero_amount(client: TestClient):
    payload = _valid_raw_transaction()
    payload["amount"] = 0.0
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


# ── H. Invalid timestamp (negative) -> 422 ──────────────────────────


def test_negative_timestamp(client: TestClient):
    payload = _valid_raw_transaction()
    payload["timestamp"] = -100
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


# ── I. Invalid transaction_type -> 422 ──────────────────────────────


def test_invalid_transaction_type(client: TestClient):
    payload = _valid_raw_transaction()
    payload["transaction_type"] = "invalid_type"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


# ── J. Invalid device_type -> 422 ────────────────────────────────────


def test_invalid_device_type(client: TestClient):
    payload = _valid_raw_transaction()
    payload["device_type"] = "tablet"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


# ── K. isFraud supplied -> 422 ──────────────────────────────────────


def test_is_fraud_rejected(client: TestClient):
    payload = _valid_raw_transaction()
    payload["isFraud"] = 1
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


# ── L. TransactionID supplied -> 422 ────────────────────────────────


def test_transaction_id_rejected(client: TestClient):
    payload = _valid_raw_transaction()
    payload["TransactionID"] = 12345
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


# ── M. Model unavailable -> 503 ─────────────────────────────────────


def test_model_unavailable(client: TestClient):
    """Simulate model unavailable by nulling _predictor temporarily."""
    import ml.api.app as app_module

    original = app_module._predictor
    app_module._predictor = None
    try:
        resp = client.post("/predict", json=_valid_raw_transaction())
        assert resp.status_code == 503
    finally:
        app_module._predictor = original


# ── History recording ──────────────────────────────────────────────────


def test_history_recorded(client: TestClient):
    """Prediction records transaction in history store."""
    if not _model_available(client):
        pytest.skip("Model not available")
    _history_module.history_store.clear()
    client.post("/predict", json=_valid_raw_transaction())
    assert _history_module.history_store.total_count() >= 1


# ── Additional: valid with optional fields ────────────────────────────


def test_valid_with_optional_fields(client: TestClient):
    """Transaction with all optional fields should still work."""
    if not _model_available(client):
        pytest.skip("Model not available")
    payload = _valid_raw_transaction()
    payload["timestamp"] = 86400 * 3 + 3600 * 14  # day 3, hour 14
    payload["card1"] = 12345
    payload["addr1"] = 100
    payload["addr2"] = 200
    payload["ProductCD"] = "W"
    payload["has_identity_data"] = 1
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["threshold"] == 0.50


def test_valid_with_identity_fields(client: TestClient):
    """Transaction with identity fields should work."""
    if not _model_available(client):
        pytest.skip("Model not available")
    payload = _valid_raw_transaction()
    payload["id_19"] = "value_1"
    payload["id_20"] = "value_2"
    payload["DeviceType"] = "mobile"
    payload["has_identity_data"] = 1
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


def test_valid_with_customer_history(client: TestClient):
    """Transaction with customer_history should work."""
    if not _model_available(client):
        pytest.skip("Model not available")
    payload = _valid_raw_transaction()
    payload["customer_history"] = {
        "transaction_count_30d": 15,
        "avg_amount_30d": 500.0,
        "previous_flagged_count": 0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


# ── Health endpoint ───────────────────────────────────────────────────


def test_health_ready(client: TestClient):
    if not _model_available(client):
        pytest.skip("Model not available")
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["model_version"] == "fraud-xgb-v1.0.0"
    assert data["features"] == 24
