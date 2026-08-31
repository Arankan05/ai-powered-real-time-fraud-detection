"""Step 32 — Customer History Outcome Feedback Loop tests.

Validates the complete label feedback loop:
  predict → record → outcome update → next predict → verify features.

Covers:
  1. Successful fraud outcome update
  2. Successful legitimate outcome update
  3. Target transaction not found → 404
  4. Invalid outcome / input validation → 422
  5. Customer isolation
  6. previous_suspicious_count before and after outcome update
  7. SQLite persistence after update and restart
  8. In-memory repository behavior
  9. Existing prediction still works after outcome updates
 10. No current/future transaction leakage
 11. API response/error schema
 12. End-to-end integration
 13. Backend outcome endpoint wiring

Run from project root::

    python -m pytest ml/api/tests/test_step32_outcome_feedback.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml.api.app import app
import ml.features.history as _history_module
from ml.features.engineer import (
    FEATURE_LIST,
    engineer_features_for_inference,
)
from ml.features.history import (
    InMemoryHistoryStore,
    SQLiteHistoryRepository,
    record_transaction,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_history():
    """Clear history store before each test for isolation."""
    _history_module.history_store.clear()
    yield
    _history_module.history_store.clear()


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with model loaded via lifespan."""
    with TestClient(app) as c:
        yield c


def _model_available(client: TestClient) -> bool:
    """Check if the model is available before running tests."""
    resp = client.get("/health")
    return resp.json().get("status") == "ready"


# ── Helpers ───────────────────────────────────────────────────────────


def _raw(
    *,
    amount: float = 100.0,
    timestamp: int = 86_400,
    customer_id: str = "cust_32",
    device_type: str = "desktop",
) -> dict:
    """Build a minimal valid raw transaction payload."""
    return {
        "amount": amount,
        "currency": "USD",
        "merchant_name": "Test Merchant",
        "merchant_category": "5732",
        "transaction_type": "purchase",
        "location_country": "US",
        "location_city": "New York",
        "device_fingerprint": "fp_abc123",
        "device_type": device_type,
        "ip_address": "192.168.1.1",
        "customer_id": customer_id,
        "timestamp": timestamp,
    }


def _predict_and_record(client: TestClient, **kwargs) -> dict:
    """Send a prediction request and return the response JSON."""
    payload = _raw(**kwargs)
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, f"predict failed: {resp.text}"
    return resp.json()


def _record_directly(**kwargs) -> dict:
    """Record a transaction directly in the history store (no model needed)."""
    raw = _raw(**kwargs)
    store = _history_module.history_store
    record_transaction(store, raw)
    return raw


# ═══════════════════════════════════════════════════════════════════════
# 1. Successful fraud outcome update
# ═══════════════════════════════════════════════════════════════════════


class TestSuccessfulFraudUpdate:
    def test_update_to_fraud(self, client):
        """Transaction recorded with is_fraud=0 can be updated to is_fraud=1."""
        raw = _record_directly(timestamp=1000)
        ts = raw["timestamp"]

        resp = client.post("/outcome", json={
            "customer_id": "cust_32",
            "timestamp": ts,
            "is_fraud": 1,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["updated"] is True
        assert body["is_fraud"] == 1
        assert body["customer_id"] == "cust_32"
        assert body["timestamp"] == ts

    def test_history_reflects_fraud(self, client):
        """After updating to fraud, history record has is_fraud=1."""
        raw = _record_directly(timestamp=2000)
        ts = raw["timestamp"]

        client.post("/outcome", json={
            "customer_id": "cust_32",
            "timestamp": ts,
            "is_fraud": 1,
        })

        store = _history_module.history_store
        history = store.get("cust_32")
        assert len(history) == 1
        assert history[0]["is_fraud"] == 1


# ═══════════════════════════════════════════════════════════════════════
# 2. Successful legitimate outcome update
# ═══════════════════════════════════════════════════════════════════════


class TestSuccessfulLegitimateUpdate:
    def test_update_to_legitimate(self, client):
        """Explicitly setting is_fraud=0 succeeds."""
        raw = _record_directly(timestamp=3000)
        ts = raw["timestamp"]

        resp = client.post("/outcome", json={
            "customer_id": "cust_32",
            "timestamp": ts,
            "is_fraud": 0,
        })
        assert resp.status_code == 200
        assert resp.json()["is_fraud"] == 0

    def test_toggle_fraud_to_legitimate(self, client):
        """Can change from fraud=1 back to fraud=0."""
        raw = _record_directly(timestamp=3500)
        ts = raw["timestamp"]

        # First set to fraud
        client.post("/outcome", json={
            "customer_id": "cust_32",
            "timestamp": ts,
            "is_fraud": 1,
        })
        store = _history_module.history_store
        assert store.get("cust_32")[0]["is_fraud"] == 1

        # Then revert to legitimate
        resp = client.post("/outcome", json={
            "customer_id": "cust_32",
            "timestamp": ts,
            "is_fraud": 0,
        })
        assert resp.status_code == 200
        assert store.get("cust_32")[0]["is_fraud"] == 0


# ═══════════════════════════════════════════════════════════════════════
# 3. Target transaction not found → 404
# ═══════════════════════════════════════════════════════════════════════


class TestTransactionNotFound:
    def test_nonexistent_timestamp(self, client):
        """404 when timestamp doesn't match any record."""
        resp = client.post("/outcome", json={
            "customer_id": "cust_32",
            "timestamp": 999999,
            "is_fraud": 1,
        })
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_nonexistent_customer(self, client):
        """404 when customer_id doesn't match any record."""
        _record_directly(timestamp=4000)

        resp = client.post("/outcome", json={
            "customer_id": "nonexistent_customer",
            "timestamp": 4000,
            "is_fraud": 1,
        })
        assert resp.status_code == 404

    def test_no_history_at_all(self, client):
        """404 when history store is completely empty."""
        resp = client.post("/outcome", json={
            "customer_id": "nobody",
            "timestamp": 0,
            "is_fraud": 0,
        })
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════
# 4. Invalid outcome / input validation → 422
# ═══════════════════════════════════════════════════════════════════════


class TestInputValidation:
    def test_is_fraud_out_of_range_high(self, client):
        """is_fraud > 1 is rejected."""
        resp = client.post("/outcome", json={
            "customer_id": "cust_32",
            "timestamp": 5000,
            "is_fraud": 2,
        })
        assert resp.status_code == 422

    def test_is_fraud_negative(self, client):
        """Negative is_fraud is rejected."""
        resp = client.post("/outcome", json={
            "customer_id": "cust_32",
            "timestamp": 5000,
            "is_fraud": -1,
        })
        assert resp.status_code == 422

    def test_negative_timestamp(self, client):
        """Negative timestamp is rejected."""
        resp = client.post("/outcome", json={
            "customer_id": "cust_32",
            "timestamp": -1,
            "is_fraud": 0,
        })
        assert resp.status_code == 422

    def test_empty_customer_id(self, client):
        """Empty customer_id is rejected."""
        resp = client.post("/outcome", json={
            "customer_id": "",
            "timestamp": 5000,
            "is_fraud": 0,
        })
        assert resp.status_code == 422

    def test_missing_customer_id(self, client):
        """Missing customer_id is rejected."""
        resp = client.post("/outcome", json={
            "timestamp": 5000,
            "is_fraud": 0,
        })
        assert resp.status_code == 422

    def test_missing_timestamp(self, client):
        """Missing timestamp is rejected."""
        resp = client.post("/outcome", json={
            "customer_id": "cust_32",
            "is_fraud": 0,
        })
        assert resp.status_code == 422

    def test_missing_is_fraud(self, client):
        """Missing is_fraud is rejected."""
        resp = client.post("/outcome", json={
            "customer_id": "cust_32",
            "timestamp": 5000,
        })
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════
# 5. Customer isolation
# ═══════════════════════════════════════════════════════════════════════


class TestCustomerIsolation:
    def test_one_customer_outcome_doesnt_affect_other(self, client):
        """Updating customer A's outcome does not change customer B's history."""
        _record_directly(customer_id="alice", timestamp=6000, amount=50.0)
        _record_directly(customer_id="bob", timestamp=6001, amount=200.0)

        # Update Alice to fraud
        resp = client.post("/outcome", json={
            "customer_id": "alice",
            "timestamp": 6000,
            "is_fraud": 1,
        })
        assert resp.status_code == 200

        store = _history_module.history_store
        alice_hist = store.get("alice")
        bob_hist = store.get("bob")

        assert alice_hist[0]["is_fraud"] == 1
        assert bob_hist[0]["is_fraud"] == 0  # Bob unaffected

    def test_wrong_customer_timestamp_mismatch(self, client):
        """Can't update customer A's record using customer B's ID."""
        _record_directly(customer_id="alice", timestamp=7000)

        resp = client.post("/outcome", json={
            "customer_id": "bob",
            "timestamp": 7000,
            "is_fraud": 1,
        })
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════
# 6. previous_suspicious_count before and after outcome update
# ═══════════════════════════════════════════════════════════════════════


class TestPreviousSuspiciousCount:
    def test_suspicious_count_before_update(self, client):
        """Before outcome update, previous_suspicious_count = 0 (all is_fraud=0)."""
        _record_directly(timestamp=10_000, amount=100.0)
        _record_directly(timestamp=10_100, amount=200.0)

        # Third transaction: previous_suspicious_count should be 0
        features = engineer_features_for_inference(
            _raw(timestamp=10_200, amount=150.0),
            history_store=_history_module.history_store,
        )
        assert features["previous_suspicious_count"].iloc[0] == 0

    def test_suspicious_count_after_fraud_update(self, client):
        """After updating 2 records to fraud, previous_suspicious_count = 2."""
        _record_directly(timestamp=20_000, amount=100.0)
        _record_directly(timestamp=20_100, amount=200.0)

        # Update both to fraud
        client.post("/outcome", json={
            "customer_id": "cust_32", "timestamp": 20_000, "is_fraud": 1,
        })
        client.post("/outcome", json={
            "customer_id": "cust_32", "timestamp": 20_100, "is_fraud": 1,
        })

        # Third transaction should see previous_suspicious_count = 2
        features = engineer_features_for_inference(
            _raw(timestamp=20_200, amount=150.0),
            history_store=_history_module.history_store,
        )
        assert features["previous_suspicious_count"].iloc[0] == 2

    def test_suspicious_count_partial_update(self, client):
        """Only 1 of 3 records updated to fraud \u2192 count = 1."""
        _record_directly(timestamp=30_000, amount=100.0)
        _record_directly(timestamp=30_100, amount=200.0)
        _record_directly(timestamp=30_200, amount=300.0)

        # Only first record \u2192 fraud
        client.post("/outcome", json={
            "customer_id": "cust_32", "timestamp": 30_000, "is_fraud": 1,
        })

        features = engineer_features_for_inference(
            _raw(timestamp=30_300, amount=150.0),
            history_store=_history_module.history_store,
        )
        assert features["previous_suspicious_count"].iloc[0] == 1

    def test_suspicious_count_after_revert(self, client):
        """Reverting fraud to legitimate decreases the count."""
        _record_directly(timestamp=40_000, amount=100.0)
        _record_directly(timestamp=40_100, amount=200.0)

        # Set both to fraud
        client.post("/outcome", json={
            "customer_id": "cust_32", "timestamp": 40_000, "is_fraud": 1,
        })
        client.post("/outcome", json={
            "customer_id": "cust_32", "timestamp": 40_100, "is_fraud": 1,
        })

        # Revert first to legitimate
        client.post("/outcome", json={
            "customer_id": "cust_32", "timestamp": 40_000, "is_fraud": 0,
        })

        features = engineer_features_for_inference(
            _raw(timestamp=40_200, amount=150.0),
            history_store=_history_module.history_store,
        )
        assert features["previous_suspicious_count"].iloc[0] == 1


# ═══════════════════════════════════════════════════════════════════════
# 7. SQLite persistence after update and restart
# ═══════════════════════════════════════════════════════════════════════


class TestSQLitePersistence:
    def test_outcome_survives_restart(self, tmp_path):
        """Updated outcome persists after closing and reopening SQLite."""
        db_path = tmp_path / "test_outcome.db"

        # Create store, add record, update outcome
        store1 = SQLiteHistoryRepository(db_path=db_path)
        store1.add("cust_32", {
            "timestamp": 50_000, "amount": 100.0, "is_fraud": 0,
        })
        assert store1.record_outcome("cust_32", 50_000, 1) is True
        store1.close()

        # Reopen — outcome should be preserved
        store2 = SQLiteHistoryRepository(db_path=db_path)
        history = store2.get("cust_32")
        assert len(history) == 1
        assert history[0]["is_fraud"] == 1
        store2.close()

    def test_outcome_features_after_restart(self, tmp_path):
        """Features computed from restarted store reflect updated outcome."""
        db_path = tmp_path / "test_outcome_feat.db"

        store1 = SQLiteHistoryRepository(db_path=db_path)
        store1.add("cust_32", {
            "timestamp": 60_000, "amount": 100.0, "is_fraud": 0,
        })
        store1.record_outcome("cust_32", 60_000, 1)
        store1.close()

        store2 = SQLiteHistoryRepository(db_path=db_path)
        features = engineer_features_for_inference(
            _raw(timestamp=60_100, amount=150.0),
            history_store=store2,
        )
        assert features["previous_suspicious_count"].iloc[0] == 1
        store2.close()


# ═══════════════════════════════════════════════════════════════════════
# 8. In-memory repository behavior
# ═══════════════════════════════════════════════════════════════════════


class TestInMemoryStore:
    def test_in_memory_outcome_update(self):
        """InMemoryHistoryStore.record_outcome works correctly."""
        store = InMemoryHistoryStore()
        store.add("c1", {"timestamp": 100, "amount": 50.0, "is_fraud": 0})

        assert store.record_outcome("c1", 100, 1) is True
        entries = store.get("c1")
        assert entries[0]["is_fraud"] == 1

    def test_in_memory_outcome_not_found(self):
        """InMemoryHistoryStore.record_outcome returns False for missing."""
        store = InMemoryHistoryStore()
        assert store.record_outcome("c1", 999, 1) is False

    def test_in_memory_features_after_update(self):
        """Features reflect in-memory outcome update."""
        store = InMemoryHistoryStore()
        store.add("c1", {
            "timestamp": 70_000, "amount": 100.0, "is_fraud": 0,
            "product_cd": "W",
        })
        store.record_outcome("c1", 70_000, 1)

        features = engineer_features_for_inference(
            _raw(timestamp=70_100, amount=150.0, customer_id="c1"),
            history_store=store,
        )
        assert features["previous_suspicious_count"].iloc[0] == 1


# ═══════════════════════════════════════════════════════════════════════
# 9. Existing prediction still works after outcome updates
# ═══════════════════════════════════════════════════════════════════════


class TestPredictionAfterOutcomeUpdate:
    def test_predict_still_works(self, client):
        """POST /predict works normally after outcome updates."""
        if not _model_available(client):
            pytest.skip("Model not available")
        # Predict + update
        _predict_and_record(client, timestamp=80_000)
        client.post("/outcome", json={
            "customer_id": "cust_32", "timestamp": 80_000, "is_fraud": 1,
        })

        # Next prediction should still work
        resp = client.post("/predict", json=_raw(timestamp=80_100))
        assert resp.status_code == 200
        body = resp.json()
        assert "fraud_probability" in body
        assert "fraud_prediction" in body
        assert "explanation" in body
        assert 0.0 <= body["fraud_probability"] <= 1.0

    def test_24_features_after_update(self, client):
        """Feature schema remains 24 features after outcome update."""
        _record_directly(timestamp=90_000)
        client.post("/outcome", json={
            "customer_id": "cust_32", "timestamp": 90_000, "is_fraud": 1,
        })

        features = engineer_features_for_inference(
            _raw(timestamp=90_100),
            history_store=_history_module.history_store,
        )
        assert list(features.columns) == FEATURE_LIST
        assert features.shape == (1, 24)

    def test_shap_still_works_after_update(self, client):
        """SHAP explanations work after outcome updates."""
        if not _model_available(client):
            pytest.skip("Model not available")
        _predict_and_record(client, timestamp=100_000)
        client.post("/outcome", json={
            "customer_id": "cust_32", "timestamp": 100_000, "is_fraud": 1,
        })

        resp = client.post("/predict", json=_raw(timestamp=100_100))
        assert resp.status_code == 200
        body = resp.json()
        assert body["explanation"] is not None
        assert len(body["explanation"]) > 0
        for factor in body["explanation"]:
            assert "feature" in factor
            assert "importance" in factor


# ═══════════════════════════════════════════════════════════════════════
# 10. No current/future transaction leakage
# ═══════════════════════════════════════════════════════════════════════


class TestNoLeakage:
    def test_current_excluded_from_own_history(self, client):
        """Current transaction is still excluded from its own features."""
        # Record 3 prior transactions so velocity formula is unambiguous
        _record_directly(timestamp=110_000, amount=100.0)
        _record_directly(timestamp=110_060, amount=110.0)
        _record_directly(timestamp=110_120, amount=120.0)
        # Update one to fraud
        client.post("/outcome", json={
            "customer_id": "cust_32", "timestamp": 110_000, "is_fraud": 1,
        })

        # The next transaction should see 3 prior records in history
        # and velocity reflecting prior txs within 1h
        features = engineer_features_for_inference(
            _raw(timestamp=110_200, amount=200.0),
            history_store=_history_module.history_store,
        )
        # All 3 priors are within 1h (max gap = 200s < 3600s)
        assert features["tx_velocity_1h"].iloc[0] >= 1
        # previous_suspicious_count = 1 (one fraud-updated record)
        assert features["previous_suspicious_count"].iloc[0] == 1

    def test_future_timestamp_excluded(self, client):
        """Records with timestamp >= current are excluded from history."""
        # Record at ts=120_000, update to fraud
        _record_directly(timestamp=120_000, amount=100.0)
        client.post("/outcome", json={
            "customer_id": "cust_32", "timestamp": 120_000, "is_fraud": 1,
        })

        # Transaction at ts=120_000 (same timestamp) should NOT see the
        # record in its history (strict less-than)
        features = engineer_features_for_inference(
            _raw(timestamp=120_000, amount=200.0),
            history_store=_history_module.history_store,
        )
        assert features["previous_suspicious_count"].iloc[0] == 0


# ═══════════════════════════════════════════════════════════════════════
# 11. API response schema
# ═══════════════════════════════════════════════════════════════════════


class TestAPIResponseSchema:
    def test_success_response_fields(self, client):
        """Successful outcome update returns all expected fields."""
        _record_directly(timestamp=130_000)

        resp = client.post("/outcome", json={
            "customer_id": "cust_32", "timestamp": 130_000, "is_fraud": 1,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {"updated", "customer_id", "timestamp", "is_fraud"}
        assert isinstance(body["updated"], bool)
        assert isinstance(body["customer_id"], str)
        assert isinstance(body["timestamp"], int)
        assert isinstance(body["is_fraud"], int)

    def test_404_response_schema(self, client):
        """404 response has standard FastAPI error schema."""
        resp = client.post("/outcome", json={
            "customer_id": "nobody", "timestamp": 0, "is_fraud": 0,
        })
        assert resp.status_code == 404
        body = resp.json()
        assert "detail" in body

    def test_422_response_schema(self, client):
        """422 response has standard FastAPI validation error schema."""
        resp = client.post("/outcome", json={
            "customer_id": "cust_32", "timestamp": 0, "is_fraud": 5,
        })
        assert resp.status_code == 422
        body = resp.json()
        assert "detail" in body

    def test_prediction_response_includes_timestamp(self, client):
        """Prediction response now includes timestamp field."""
        if not _model_available(client):
            pytest.skip("Model not available")
        resp = client.post("/predict", json=_raw(timestamp=140_000))
        assert resp.status_code == 200
        body = resp.json()
        assert "timestamp" in body
        assert body["timestamp"] == 140_000


# ═══════════════════════════════════════════════════════════════════════
# 12. End-to-end integration
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEndIntegration:
    def test_full_feedback_loop(self, client):
        """predict \u2192 record \u2192 outcome update \u2192 predict \u2192 verify features."""
        if not _model_available(client):
            pytest.skip("Model not available")

        # Step 1: First transaction (cold start) via API
        r1 = _predict_and_record(client, timestamp=200_000, amount=100.0)
        assert r1["timestamp"] == 200_000

        # Step 2: Second transaction (has history, but is_fraud=0)
        features_before = engineer_features_for_inference(
            _raw(timestamp=200_100, amount=200.0),
            history_store=_history_module.history_store,
        )
        assert features_before["previous_suspicious_count"].iloc[0] == 0

        # Step 3: Update first transaction to fraud
        resp = client.post("/outcome", json={
            "customer_id": "cust_32", "timestamp": 200_000, "is_fraud": 1,
        })
        assert resp.status_code == 200

        # Step 4: Third transaction sees updated fraud outcome
        features_after = engineer_features_for_inference(
            _raw(timestamp=200_200, amount=150.0),
            history_store=_history_module.history_store,
        )
        assert features_after["previous_suspicious_count"].iloc[0] == 1

        # Step 5: Prediction with SHAP still works
        resp = client.post("/predict", json=_raw(timestamp=200_300, amount=300.0))
        assert resp.status_code == 200
        body = resp.json()
        assert body["fraud_prediction"] in (0, 1)
        assert body["explanation"] is not None

    def test_multi_customer_feedback_loop(self, client):
        """Multiple customers can have independent outcome updates."""
        # Customer A: predict + update to fraud
        _record_directly(customer_id="custA", timestamp=300_000, amount=100.0)
        client.post("/outcome", json={
            "customer_id": "custA", "timestamp": 300_000, "is_fraud": 1,
        })

        # Customer B: predict, no update
        _record_directly(customer_id="custB", timestamp=300_001, amount=200.0)

        # Customer A's next tx sees suspicious_count=1
        feat_a = engineer_features_for_inference(
            _raw(customer_id="custA", timestamp=300_100, amount=150.0),
            history_store=_history_module.history_store,
        )
        assert feat_a["previous_suspicious_count"].iloc[0] == 1

        # Customer B's next tx sees suspicious_count=0
        feat_b = engineer_features_for_inference(
            _raw(customer_id="custB", timestamp=300_101, amount=250.0),
            history_store=_history_module.history_store,
        )
        assert feat_b["previous_suspicious_count"].iloc[0] == 0


# ═══════════════════════════════════════════════════════════════════════
# 13. Backend outcome endpoint wiring (unit-level with mock)
# ═══════════════════════════════════════════════════════════════════════


class TestBackendOutcomeWiring:
    @pytest.mark.asyncio
    async def test_ml_client_update_outcome_success(self):
        """MLServiceClient.update_outcome calls POST /outcome."""
        from backend.services.ml_client import MLServiceClient
        from unittest.mock import MagicMock

        client = MLServiceClient(base_url="http://localhost:9999")

        # Use MagicMock (not AsyncMock) for response so .json() is sync
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "updated": True,
            "customer_id": "c1",
            "timestamp": 100,
            "is_fraud": 1,
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client.update_outcome({
                "customer_id": "c1",
                "timestamp": 100,
                "is_fraud": 1,
            })

        assert result["updated"] is True
        assert result["is_fraud"] == 1

    @pytest.mark.asyncio
    async def test_ml_client_update_outcome_not_found(self):
        """MLServiceClient.update_outcome raises on 404."""
        from backend.services.ml_client import (
            MLServiceClient,
            MLServiceResponseError,
        )
        from unittest.mock import MagicMock

        client = MLServiceClient(base_url="http://localhost:9999")

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Not found"}
        mock_response.text = '{"detail": "Not found"}'

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(MLServiceResponseError) as exc_info:
                await client.update_outcome({
                    "customer_id": "nobody",
                    "timestamp": 0,
                    "is_fraud": 0,
                })
            assert exc_info.value.status_code == 404


# ═══════════════════════════════════════════════════════════════════════
# 14. Protocol conformance
# ═══════════════════════════════════════════════════════════════════════


class TestProtocolConformance:
    def test_inmemory_satisfies_protocol(self):
        """InMemoryHistoryStore satisfies CustomerHistoryRepository."""
        from ml.features.history import CustomerHistoryRepository

        store = InMemoryHistoryStore()
        assert isinstance(store, CustomerHistoryRepository)

    def test_sqlite_satisfies_protocol(self, tmp_path):
        """SQLiteHistoryRepository satisfies CustomerHistoryRepository."""
        from ml.features.history import CustomerHistoryRepository

        store = SQLiteHistoryRepository(db_path=tmp_path / "proto.db")
        assert isinstance(store, CustomerHistoryRepository)
        store.close()
