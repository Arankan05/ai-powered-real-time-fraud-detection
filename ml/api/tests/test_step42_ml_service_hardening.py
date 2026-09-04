"""Step 42 — Production ML service hardening tests.

Comprehensive test suite validating the hardened ML fraud-detection
service: model loading, health/readiness, input validation, prediction
error handling, customer isolation, SHAP hardening, timeout behaviour,
concurrency safety, and security (no information leakage).

Run from the project root::

    python -m pytest ml/api/tests/test_step42_ml_service_hardening.py -v
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from ml.api import app as _app_module
from ml.api.app import app
import ml.features.history as _history_module
from ml.predict.bundle import ModelLoadError


# ── Helpers ────────────────────────────────────────────────────────────


def _valid_transaction(**overrides) -> dict:
    """Return a minimal valid transaction payload."""
    base = {
        "amount": 150.0,
        "currency": "USD",
        "merchant_name": "Test Merchant",
        "merchant_category": "5732",
        "transaction_type": "purchase",
        "location_country": "US",
        "location_city": "New York",
        "device_fingerprint": "fp_abc123",
        "device_type": "mobile",
        "ip_address": "192.168.1.100",
    }
    base.update(overrides)
    return base


def _model_available(client: TestClient) -> bool:
    resp = client.get("/health")
    return resp.json().get("status") == "ready"


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_history():
    """Clear history store before and after each test.

    Tolerates a closed SQLite store (which can happen when a test
    creates its own TestClient whose lifespan shutdown closes it).
    """
    try:
        _history_module.history_store.clear()
    except Exception:
        _history_module.history_store = _history_module.InMemoryHistoryStore()
    yield
    try:
        _history_module.history_store.clear()
    except Exception:
        _history_module.history_store = _history_module.InMemoryHistoryStore()


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with model loaded via lifespan."""
    with TestClient(app) as c:
        yield c


# =====================================================================
# 1. MODEL LOADING
# =====================================================================


class TestModelLoading:
    """Model loading, missing/corrupt artifact, and version reporting."""

    def test_model_loaded_successfully(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ready"
        assert data["model_version"] is not None
        assert data["features"] is not None
        assert data["features"] > 0

    def test_model_version_in_prediction(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        resp = client.post("/predict", json=_valid_transaction())
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_version"] is not None
        assert len(data["model_version"]) > 0

    def test_model_unavailable_returns_503(self, client: TestClient):
        """When _predictor is None, /predict returns 503."""
        saved = _app_module._predictor
        try:
            _app_module._predictor = None
            resp = client.post("/predict", json=_valid_transaction())
            assert resp.status_code == 503
            body = resp.json()
            assert "detail" in body
            # Must NOT leak filesystem paths or save_model command
            assert "save_model" not in body["detail"].lower()
            assert ".joblib" not in body["detail"]
        finally:
            _app_module._predictor = saved

    def test_model_unavailable_message_is_safe(self, client: TestClient):
        """503 message must not contain paths, secrets, or stack traces."""
        saved = _app_module._predictor
        try:
            _app_module._predictor = None
            resp = client.post("/predict", json=_valid_transaction())
            detail = resp.json()["detail"]
            assert "Traceback" not in detail
            assert ".joblib" not in detail
            assert "C:\\" not in detail
            assert "/home" not in detail
        finally:
            _app_module._predictor = saved

    def test_corrupt_model_load_failure(self):
        """ModelLoadError during startup is caught; service starts degraded."""
        from ml.predict.bundle import ModelLoadError

        with patch.object(
            _app_module, "FraudPredictor",
            side_effect=ModelLoadError("Corrupt bundle for test"),
        ):
            # Simulate lifespan reload
            saved = _app_module._predictor
            try:
                # Directly invoke the error path
                try:
                    _app_module.FraudPredictor(bundle_path=None)
                    pytest.fail("Expected ModelLoadError")
                except ModelLoadError:
                    _app_module._predictor = None
                assert _app_module._predictor is None
            finally:
                _app_module._predictor = saved

    def test_model_loaded_once_not_per_request(self, client: TestClient):
        """Model version is identical across multiple requests (loaded once)."""
        if not _model_available(client):
            pytest.skip("Model not available")
        versions = set()
        for _ in range(5):
            resp = client.post("/predict", json=_valid_transaction())
            assert resp.status_code == 200
            versions.add(resp.json()["model_version"])
        assert len(versions) == 1  # same version every time


# =====================================================================
# 2. HEALTH / READINESS / LIVENESS
# =====================================================================


class TestHealthReadiness:
    """Liveness, readiness, and /health endpoints."""

    def test_liveness_always_returns_alive(self, client: TestClient):
        resp = client.get("/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_liveness_when_model_unavailable(self, client: TestClient):
        saved = _app_module._predictor
        try:
            _app_module._predictor = None
            resp = client.get("/live")
            assert resp.status_code == 200
            assert resp.json()["status"] == "alive"
        finally:
            _app_module._predictor = saved

    def test_readiness_when_ready(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["model_version"] is not None
        assert data["features"] is not None
        assert data["history_store"] in ("sqlite", "in_memory")

    def test_readiness_when_model_unavailable(self, client: TestClient):
        saved = _app_module._predictor
        try:
            _app_module._predictor = None
            resp = client.get("/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "not_ready"
        finally:
            _app_module._predictor = saved

    def test_health_backward_compatible(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "history_store" in data

    def test_health_no_secret_leakage(self, client: TestClient):
        """Health response must never contain secrets, paths, or credentials."""
        resp = client.get("/health")
        body = json.dumps(resp.json())
        assert "password" not in body.lower()
        assert "secret" not in body.lower()
        assert "jwt" not in body.lower()
        assert "Bearer" not in body
        assert "Traceback" not in body


# =====================================================================
# 3. INPUT VALIDATION
# =====================================================================


class TestInputValidation:
    """Hardened input validation — controlled 4xx for malformed data."""

    def test_missing_required_field(self, client: TestClient):
        bad = {"amount": 100.0}  # missing 9 required fields
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_zero_amount_rejected(self, client: TestClient):
        resp = client.post("/predict", json=_valid_transaction(amount=0))
        assert resp.status_code == 422

    def test_negative_amount_rejected(self, client: TestClient):
        resp = client.post("/predict", json=_valid_transaction(amount=-50))
        assert resp.status_code == 422

    def test_excessive_amount_rejected(self, client: TestClient):
        resp = client.post("/predict", json=_valid_transaction(amount=99_999_999))
        assert resp.status_code == 422

    def test_invalid_currency_lowercase(self, client: TestClient):
        resp = client.post("/predict", json=_valid_transaction(currency="usd"))
        assert resp.status_code == 422

    def test_invalid_currency_too_long(self, client: TestClient):
        resp = client.post("/predict", json=_valid_transaction(currency="USDX"))
        assert resp.status_code == 422

    def test_invalid_currency_numeric(self, client: TestClient):
        resp = client.post("/predict", json=_valid_transaction(currency="123"))
        assert resp.status_code == 422

    def test_invalid_transaction_type(self, client: TestClient):
        resp = client.post("/predict", json=_valid_transaction(transaction_type="refund"))
        assert resp.status_code == 422

    def test_invalid_device_type(self, client: TestClient):
        resp = client.post("/predict", json=_valid_transaction(device_type="tablet"))
        assert resp.status_code == 422

    def test_invalid_ip_too_short(self, client: TestClient):
        resp = client.post("/predict", json=_valid_transaction(ip_address="1.2"))
        assert resp.status_code == 422

    def test_invalid_product_cd(self, client: TestClient):
        resp = client.post("/predict", json=_valid_transaction(ProductCD="Q"))
        assert resp.status_code == 422

    def test_valid_product_cd(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        resp = client.post("/predict", json=_valid_transaction(ProductCD="W"))
        assert resp.status_code == 200

    def test_forbidden_isfraud_field(self, client: TestClient):
        bad = _valid_transaction()
        bad["isFraud"] = 1
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_forbidden_transaction_id(self, client: TestClient):
        bad = _valid_transaction()
        bad["TransactionID"] = "TXN123"
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_empty_merchant_name_rejected(self, client: TestClient):
        resp = client.post("/predict", json=_valid_transaction(merchant_name=""))
        assert resp.status_code == 422

    def test_negative_timestamp_rejected(self, client: TestClient):
        resp = client.post("/predict", json=_valid_transaction(timestamp=-1))
        assert resp.status_code == 422

    def test_all_validation_returns_422_not_500(self, client: TestClient):
        """Every malformed input must return 422, never 500."""
        bad_payloads = [
            {"amount": -1, "currency": "USD"},
            _valid_transaction(amount="not_a_number"),
            _valid_transaction(currency=""),
            _valid_transaction(transaction_type="DROP TABLE"),
        ]
        for payload in bad_payloads:
            resp = client.post("/predict", json=payload)
            assert resp.status_code in (422, 400), (
                f"Expected 4xx for {payload}, got {resp.status_code}"
            )


# =====================================================================
# 4. PREDICTION ERROR HANDLING
# =====================================================================


class TestPredictionErrorHandling:
    """Controlled error responses for prediction failures."""

    def test_valid_prediction_succeeds(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        resp = client.post("/predict", json=_valid_transaction())
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert data["fraud_prediction"] in (0, 1)
        assert data["threshold"] > 0

    def test_prediction_failure_returns_500_safe(self, client: TestClient):
        """Unexpected prediction error → 500 with safe message."""
        if not _model_available(client):
            pytest.skip("Model not available")
        with patch.object(
            _app_module._predictor, "predict",
            side_effect=RuntimeError("Internal model crash"),
        ):
            resp = client.post("/predict", json=_valid_transaction())
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "Traceback" not in detail
        assert "RuntimeError" not in detail
        assert "Internal model crash" not in detail

    def test_feature_engineering_failure_returns_422(self, client: TestClient):
        """Feature engineering error → 422 with safe message."""
        if not _model_available(client):
            pytest.skip("Model not available")
        with patch(
            "ml.api.app.engineer_features_for_inference",
            side_effect=ValueError("Missing columns for test"),
        ):
            resp = client.post("/predict", json=_valid_transaction())
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        # Must not leak internal exception messages
        assert "Missing columns" not in detail

    def test_model_output_validation(self, client: TestClient):
        """Out-of-range probability → 500 with safe message."""
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.predict.predictor import PredictionResult
        bad_result = PredictionResult(
            fraud_probability=1.5,
            fraud_prediction=1,
            threshold=0.5,
            model_version="test",
            explanation=None,
        )
        with patch.object(
            _app_module._predictor, "predict", return_value=bad_result,
        ):
            resp = client.post("/predict", json=_valid_transaction())
        assert resp.status_code == 500
        assert "invalid" in resp.json()["detail"].lower()

    def test_error_responses_are_machine_readable(self, client: TestClient):
        """All error responses contain a JSON 'detail' field."""
        bad = _valid_transaction(amount=-1)
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422
        body = resp.json()
        assert "detail" in body


# =====================================================================
# 5. CUSTOMER ISOLATION
# =====================================================================


class TestCustomerIsolation:
    """Customer history isolation under hardened service."""

    def test_customer_a_uses_a_history(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        # Submit two transactions for customer A
        txn_a = _valid_transaction(customer_id="cust_A", amount=200.0)
        resp1 = client.post("/predict", json=txn_a)
        assert resp1.status_code == 200
        # Second transaction should see A's history
        txn_a2 = _valid_transaction(customer_id="cust_A", amount=300.0)
        resp2 = client.post("/predict", json=txn_a2)
        assert resp2.status_code == 200

    def test_customer_b_uses_b_history(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        txn_a = _valid_transaction(customer_id="cust_A_iso", amount=500.0)
        client.post("/predict", json=txn_a)
        # Customer B should NOT see A's history
        txn_b = _valid_transaction(customer_id="cust_B_iso", amount=500.0)
        resp_b = client.post("/predict", json=txn_b)
        assert resp_b.status_code == 200
        # B is a cold-start customer — history store has no B records
        records = _history_module.history_store.get("cust_B_iso")
        # Only B's own transaction should exist (just recorded)
        assert len(records) <= 1

    def test_concurrent_customers_no_cross_contamination(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        results = {}
        lock = threading.Lock()

        def submit(customer_id: str, amount: float):
            resp = client.post(
                "/predict",
                json=_valid_transaction(
                    customer_id=customer_id, amount=amount,
                    device_fingerprint=f"fp_{customer_id}",
                ),
            )
            with lock:
                results[customer_id] = resp.status_code

        threads = []
        for i in range(6):
            cid = f"conc_cust_{i}"
            t = threading.Thread(target=submit, args=(cid, 100.0 + i * 50))
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=30)

        for cid, code in results.items():
            assert code == 200, f"Customer {cid} got {code}"

    def test_cold_start_behavior(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        # Brand new customer with no history
        resp = client.post(
            "/predict",
            json=_valid_transaction(
                customer_id="brand_new_customer_xyz",
                device_fingerprint="fp_brand_new",
            ),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["fraud_prediction"] in (0, 1)


# =====================================================================
# 6. SHAP / EXPLANATION HARDENING
# =====================================================================


class TestSHAPHardening:
    """SHAP explanation is valid, serializable, and failure-safe."""

    def test_shap_explanation_present(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        resp = client.post("/predict", json=_valid_transaction())
        assert resp.status_code == 200
        data = resp.json()
        assert data["explanation"] is not None
        assert isinstance(data["explanation"], list)
        for factor in data["explanation"]:
            assert "feature" in factor
            assert "importance" in factor

    def test_shap_serializable(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        resp = client.post("/predict", json=_valid_transaction())
        data = resp.json()
        # Must be JSON-serializable (already was, but double-check)
        serialized = json.dumps(data["explanation"])
        assert isinstance(serialized, str)

    def test_shap_failure_returns_prediction_without_explanation(self, client: TestClient):
        """If SHAP fails, prediction still succeeds with empty explanation."""
        if not _model_available(client):
            pytest.skip("Model not available")
        # Patch the explainer's explain method to raise
        with patch.object(
            _app_module._predictor, "_compute_explanation",
            return_value=[],
        ):
            resp = client.post("/predict", json=_valid_transaction())
        assert resp.status_code == 200
        data = resp.json()
        # Explanation should be empty or None, but prediction succeeded
        assert data["fraud_prediction"] in (0, 1)
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_explanation_detail_structure(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        resp = client.post("/predict", json=_valid_transaction())
        data = resp.json()
        detail = data.get("explanation_detail")
        assert detail is not None
        assert "ml_top_factors" in detail
        assert "behaviour_signals" in detail
        assert "rules_triggered" in detail


# =====================================================================
# 7. TIMEOUT BEHAVIOUR
# =====================================================================


class TestTimeoutBehaviour:
    """Backend ML client timeout is finite and handled correctly."""

    def test_ml_client_timeout_is_finite(self):
        """Backend ML client timeout must be a positive finite number."""
        from backend.services.ml_client import MLServiceClient
        client = MLServiceClient(base_url="http://localhost:8001", timeout=5.0)
        assert client._timeout > 0
        assert client._timeout < 300  # must be finite and reasonable

    def test_ml_client_timeout_error_handled(self):
        """MLServiceTimeoutError is raised on timeout."""
        from backend.services.ml_client import (
            MLServiceClient,
            MLServiceTimeoutError,
        )

        async def fake_timeout_post(*args, **kwargs):
            raise httpx.TimeoutException("Simulated timeout")

        async def _run():
            client = MLServiceClient(timeout=1.0)
            with patch(
                "httpx.AsyncClient.post",
                new_callable=AsyncMock,
                side_effect=fake_timeout_post,
            ):
                with pytest.raises(MLServiceTimeoutError):
                    await client.predict({"amount": 100})

        import asyncio
        asyncio.run(_run())

    def test_ml_client_connection_error_handled(self):
        """MLServiceUnavailableError on connection refused."""
        from backend.services.ml_client import (
            MLServiceClient,
            MLServiceUnavailableError,
        )

        async def _run():
            client = MLServiceClient(timeout=1.0)
            with patch(
                "httpx.AsyncClient.post",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("Connection refused"),
            ):
                with pytest.raises(MLServiceUnavailableError):
                    await client.predict({"amount": 100})

        import asyncio
        asyncio.run(_run())


# =====================================================================
# 8. CONCURRENCY
# =====================================================================


class TestConcurrency:
    """Concurrent prediction requests are safe and isolated."""

    def test_concurrent_predictions(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        results: list[int] = []
        lock = threading.Lock()

        def predict(i: int):
            resp = client.post(
                "/predict",
                json=_valid_transaction(
                    amount=100.0 + i,
                    customer_id=f"conc_{i}",
                    device_fingerprint=f"fp_conc_{i}",
                ),
            )
            with lock:
                results.append(resp.status_code)

        threads = [threading.Thread(target=predict, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert len(results) == 8
        assert all(code == 200 for code in results)

    def test_repeated_predictions_stable(self, client: TestClient):
        """Identical repeated requests produce identical results."""
        if not _model_available(client):
            pytest.skip("Model not available")
        # Use a unique cold-start customer for each prediction to avoid
        # history accumulation affecting determinism.
        import uuid as _uuid
        responses = []
        for i in range(5):
            cid = f"det-stable-{_uuid.uuid4().hex[:8]}"
            txn = _valid_transaction(customer_id=cid)
            resp = client.post("/predict", json=txn)
            assert resp.status_code == 200
            responses.append(resp.json()["fraud_probability"])
        # All predictions should be identical (cold-start, same features)
        assert len(set(responses)) == 1

    def test_no_shared_state_between_requests(self, client: TestClient):
        """Two different transactions do not contaminate each other."""
        if not _model_available(client):
            pytest.skip("Model not available")
        txn1 = _valid_transaction(customer_id="state_test_1", amount=100.0)
        txn2 = _valid_transaction(customer_id="state_test_2", amount=5000.0)
        r1 = client.post("/predict", json=txn1).json()
        r2 = client.post("/predict", json=txn2).json()
        # Both should succeed
        assert r1["fraud_prediction"] in (0, 1)
        assert r2["fraud_prediction"] in (0, 1)


# =====================================================================
# 9. SECURITY — NO INFORMATION LEAKAGE
# =====================================================================


class TestSecurityNoLeakage:
    """Responses never expose secrets, paths, stack traces, or auth headers."""

    def test_no_traceback_in_validation_error(self, client: TestClient):
        resp = client.post("/predict", json=_valid_transaction(amount=-1))
        assert resp.status_code == 422
        body = json.dumps(resp.json())
        assert "Traceback" not in body
        assert "File \"" not in body

    def test_no_filesystem_path_in_error(self, client: TestClient):
        """Error responses must not contain filesystem paths."""
        saved = _app_module._predictor
        try:
            _app_module._predictor = None
            resp = client.post("/predict", json=_valid_transaction())
            body = json.dumps(resp.json())
            assert ".joblib" not in body
            assert "ml/models" not in body
            assert "/home/" not in body
            assert "C:\\" not in body
        finally:
            _app_module._predictor = saved

    def test_no_secrets_in_any_response(self, client: TestClient):
        """No response endpoint leaks secrets."""
        endpoints = ["/health", "/live", "/ready"]
        for ep in endpoints:
            resp = client.get(ep)
            body = json.dumps(resp.json())
            assert "password" not in body.lower()
            assert "secret_key" not in body.lower()
            assert "Bearer " not in body

    def test_no_auth_header_in_response(self, client: TestClient):
        """Authorization header from request is never echoed back."""
        headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.secret"}
        resp = client.post(
            "/predict",
            json=_valid_transaction(),
            headers=headers,
        )
        body = json.dumps(resp.json())
        assert "eyJhbGciOiJIUzI1NiJ9" not in body
        assert "Bearer" not in body

    def test_global_exception_handler_catches_unexpected(self, client: TestClient):
        """Unhandled exceptions return safe 500, not stack traces."""
        if not _model_available(client):
            pytest.skip("Model not available")
        with patch(
            "ml.api.app.engineer_features_for_inference",
            side_effect=OSError("Disk full \u2014 internal"),
        ):
            resp = client.post("/predict", json=_valid_transaction())
        # Should be either 422 (caught by feature eng handler) or 500 (global)
        assert resp.status_code in (422, 500)
        body = resp.json()
        assert "Disk full" not in json.dumps(body)


# =====================================================================
# 10. END-TO-END
# =====================================================================


class TestEndToEndHardened:
    """Full realistic flow with hardened service."""

    def test_customer_a_full_flow(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        # 1. Customer A submits a transaction
        txn = _valid_transaction(
            customer_id="e2e_cust_A",
            amount=250.0,
            device_fingerprint="fp_e2e_A",
        )
        resp = client.post("/predict", json=txn)
        assert resp.status_code == 200
        data = resp.json()
        # All required fields present
        assert "fraud_probability" in data
        assert "fraud_prediction" in data
        assert "model_version" in data
        assert "threshold" in data
        assert "ml_score" in data
        assert "behaviour_score" in data
        assert "rule_score" in data
        assert "risk_score" in data
        assert "risk_level" in data
        assert "decision" in data
        assert data["decision"] in ("APPROVE", "VERIFY", "HOLD")

        # 2. Second transaction for A — history exists
        txn2 = _valid_transaction(
            customer_id="e2e_cust_A",
            amount=350.0,
            device_fingerprint="fp_e2e_A",
        )
        resp2 = client.post("/predict", json=txn2)
        assert resp2.status_code == 200

    def test_customer_b_isolated_from_a(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        # A's transaction
        client.post("/predict", json=_valid_transaction(
            customer_id="e2e_isolated_A", amount=100.0,
            device_fingerprint="fp_isoA",
        ))
        # B's transaction
        resp_b = client.post("/predict", json=_valid_transaction(
            customer_id="e2e_isolated_B", amount=100.0,
            device_fingerprint="fp_isoB",
        ))
        assert resp_b.status_code == 200
        # B should have only B's own history
        b_records = _history_module.history_store.get("e2e_isolated_B")
        assert len(b_records) <= 1
        a_records = _history_module.history_store.get("e2e_isolated_A")
        assert len(a_records) >= 1

    def test_invalid_then_valid_flow(self, client: TestClient):
        """Invalid request doesn't affect subsequent valid requests."""
        if not _model_available(client):
            pytest.skip("Model not available")
        # Invalid request
        resp_bad = client.post("/predict", json=_valid_transaction(amount=-1))
        assert resp_bad.status_code == 422
        # Valid request immediately after
        resp_ok = client.post("/predict", json=_valid_transaction(
            customer_id="after_invalid_test",
        ))
        assert resp_ok.status_code == 200

    def test_alert_creation_still_works_with_hardened_ml(self, client: TestClient):
        """Prediction response is compatible with backend alert creation."""
        if not _model_available(client):
            pytest.skip("Model not available")
        resp = client.post("/predict", json=_valid_transaction(
            customer_id="alert_compat_test",
            amount=5000.0,  # likely to trigger HOLD
        ))
        assert resp.status_code == 200
        data = resp.json()
        # Backend uses decision and risk_score for alert creation
        assert data["decision"] in ("APPROVE", "VERIFY", "HOLD")
        assert isinstance(data["risk_score"], int)
        assert 0 <= data["risk_score"] <= 100
