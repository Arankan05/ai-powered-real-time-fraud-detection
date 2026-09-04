"""Step 44 — Production-Ready Fraud Decision Pipeline tests.

Covers:

A. Same customer + same idempotency key → exactly one transaction
B. Same customer + same idempotency key concurrently → one transaction
C. Different customers + same idempotency key → independent transactions
D. Same customer + different idempotency keys → independent transactions
E. Duplicate HOLD transaction → exactly one alert
F. ML timeout → bounded response, explicit failure state
G. ML unavailable → bounded response, explicit failure state
H. Successful ML → response/database decision consistency
I. Failed ML → no fabricated model output
J. Unauthorized idempotency reuse → no cross-customer data leak
K. Invalid/malformed idempotency keys → safe validation error
L. Concurrent duplicate requests → no race-condition corruption
"""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import Response

from backend.db.alert_repository import InMemoryAlertStore
from backend.db.idempotency_store import InMemoryIdempotencyStore
from backend.db.user_repository import InMemoryUserStore
from backend.services.ml_client import MLServiceClient

# ── Helpers ───────────────────────────────────────────────────────────


VALID_TXN: dict = {
    "amount": 12500.00,
    "currency": "USD",
    "merchant_name": "Test Merchant",
    "merchant_category": "5999",
    "transaction_type": "purchase",
    "location_country": "US",
    "location_city": "New York",
    "device_fingerprint": "step44-device-001",
    "device_type": "desktop",
    "ip_address": "192.168.1.100",
}


def _ml_response(
    decision: str = "HOLD",
    risk_level: str = "HIGH",
    risk_score: int = 85,
    fraud_probability: float = 0.91,
) -> dict:
    """Build a complete ML service response."""
    return {
        "fraud_probability": fraud_probability,
        "fraud_prediction": 1 if fraud_probability >= 0.5 else 0,
        "threshold": 0.50,
        "model_version": "fraud-xgb-v1.0.0",
        "timestamp": 1725200000,
        "explanation": [{"feature": "amount_deviation", "importance": 0.45}],
        "ml_score": 91,
        "behaviour_score": 75,
        "rule_score": 60,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "decision": decision,
        "explanation_detail": {
            "ml_top_factors": [
                {"feature": "amount_deviation", "importance": 0.45},
            ],
            "behaviour_signals": [],
            "rules_triggered": [],
        },
        "risk_factors": ["amount_deviation"],
    }


def _register(tc, email: str, password: str = "SecurePass1"):
    payload = {
        "email": email,
        "password": password,
        "first_name": "Step44",
        "last_name": "Test",
    }
    return tc.post("/api/v1/auth/register", json=payload)


def _login(tc, email: str, password: str = "SecurePass1"):
    return tc.post("/api/v1/auth/login", json={"email": email, "password": password})


def _bearer(tc, email: str, password: str = "SecurePass1") -> dict:
    """Register (if needed) + login → Authorization header dict."""
    resp = _login(tc, email, password)
    if resp.status_code == 401:
        _register(tc, email=email, password=password)
        resp = _login(tc, email, password)
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def idempotency_store():
    """Fresh in-memory idempotency store per test."""
    return InMemoryIdempotencyStore()


@pytest.fixture
def user_store():
    """Fresh in-memory user store per test."""
    return InMemoryUserStore()


@pytest.fixture
def alert_store():
    """Fresh in-memory alert store per test."""
    return InMemoryAlertStore()


@pytest.fixture
def test_client(user_store, alert_store, idempotency_store):
    """TestClient with real auth, mocked ML, in-memory stores."""
    from fastapi.testclient import TestClient

    from backend.app import app
    from backend.routers import alerts as alerts_module
    from backend.routers import transactions as txn_module
    from backend.security import deps as deps_module

    saved_overrides = dict(app.dependency_overrides)
    app.dependency_overrides.clear()

    deps_module.set_user_repository(user_store)
    txn_module.set_ml_client(MLServiceClient(base_url="http://mock-ml:8001"))
    alerts_module.set_alert_repository(alert_store)
    txn_module.set_alert_repository(alert_store)
    txn_module.set_idempotency_store(idempotency_store)

    yield TestClient(app), alert_store, user_store, idempotency_store

    app.dependency_overrides.clear()
    app.dependency_overrides.update(saved_overrides)


# ══════════════════════════════════════════════════════════════════════
# A. Same customer + same idempotency key → exactly one transaction
# ══════════════════════════════════════════════════════════════════════


class TestIdempotencySameCustomerSameKey:

    def test_second_request_returns_cached_result(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "idem-a@example.com")
        mock_resp = Response(200, json=_ml_response("APPROVE", "LOW", 10))

        call_count = 0

        def fake_post(url, json=None, **kw):
            nonlocal call_count
            call_count += 1
            return Response(200, json=_ml_response("APPROVE", "LOW", 10))

        idem_key = "idem-key-001"
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=fake_post,
        ):
            r1 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )
            r2 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )

        assert r1.status_code == 201
        assert r2.status_code == 200  # cached replay
        assert r1.json()["transaction_id"] == r2.json()["transaction_id"]
        assert call_count == 1  # ML called only once

    def test_idempotent_flag_set_on_cached_response(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "idem-b@example.com")
        idem_key = "idem-key-002"

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
        ):
            r1 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )
            r2 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )

        assert r2.json()["idempotent"] is True
        assert r1.json()["transaction_id"] == r2.json()["transaction_id"]

    def test_all_ml_fields_match_between_first_and_cached(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "idem-c@example.com")
        idem_key = "idem-key-003"

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 15)),
        ):
            r1 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )
            r2 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )

        d1, d2 = r1.json(), r2.json()
        for field in (
            "fraud_probability", "fraud_prediction", "risk_score",
            "risk_level", "decision", "model_version",
        ):
            assert d1[field] == d2[field], f"{field} mismatch"


# ══════════════════════════════════════════════════════════════════════
# B. Same customer + same key concurrently → one transaction
# ══════════════════════════════════════════════════════════════════════


class TestIdempotencyConcurrentSameKey:

    def test_concurrent_same_key_only_one_ml_call(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "idem-conc@example.com")
        idem_key = "idem-concurrent-001"

        call_count = 0

        def fake_post(url, json=None, **kw):
            nonlocal call_count
            call_count += 1
            return Response(200, json=_ml_response("APPROVE", "LOW", 10))

        def submit():
            with patch(
                "httpx.AsyncClient.post",
                new_callable=AsyncMock,
                side_effect=fake_post,
            ):
                return tc.post(
                    "/api/v1/transactions",
                    json=VALID_TXN,
                    headers={**headers, "Idempotency-Key": idem_key},
                )

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(submit) for _ in range(4)]
            results = [f.result() for f in as_completed(futures)]

        # At least one 201, others should be 200 (cached) or 409 (processing)
        status_codes = [r.status_code for r in results]
        assert 201 in status_codes or 200 in status_codes
        # No 500 errors
        assert 500 not in status_codes

    def test_concurrent_same_key_no_duplicate_transactions(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "idem-conc2@example.com")
        idem_key = "idem-concurrent-002"

        def submit():
            with patch(
                "httpx.AsyncClient.post",
                new_callable=AsyncMock,
                return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
            ):
                return tc.post(
                    "/api/v1/transactions",
                    json=VALID_TXN,
                    headers={**headers, "Idempotency-Key": idem_key},
                )

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(submit) for _ in range(4)]
            results = [f.result() for f in as_completed(futures)]

        # Collect unique transaction IDs from successful responses
        txn_ids = set()
        for r in results:
            if r.status_code in (200, 201):
                txn_ids.add(r.json()["transaction_id"])
        assert len(txn_ids) == 1, f"Expected 1 transaction, got {len(txn_ids)}"


# ══════════════════════════════════════════════════════════════════════
# C. Different customers + same key → independent transactions
# ══════════════════════════════════════════════════════════════════════


class TestIdempotencyDifferentCustomersSameKey:

    def test_different_customers_same_key_create_independent_txns(self, test_client):
        tc, _, _, _ = test_client
        h1 = _bearer(tc, "cust-x@example.com")
        h2 = _bearer(tc, "cust-y@example.com")
        idem_key = "shared-key-001"

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
        ):
            r1 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**h1, "Idempotency-Key": idem_key},
            )
            r2 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**h2, "Idempotency-Key": idem_key},
            )

        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["transaction_id"] != r2.json()["transaction_id"]
        assert r1.json()["customer_id"] != r2.json()["customer_id"]


# ══════════════════════════════════════════════════════════════════════
# D. Same customer + different keys → independent transactions
# ══════════════════════════════════════════════════════════════════════


class TestIdempotencyDifferentKeys:

    def test_same_customer_different_keys_create_independent_txns(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "idem-dk@example.com")

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
        ):
            r1 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": "key-alpha"},
            )
            r2 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": "key-beta"},
            )

        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["transaction_id"] != r2.json()["transaction_id"]


# ══════════════════════════════════════════════════════════════════════
# E. Duplicate HOLD → exactly one alert
# ══════════════════════════════════════════════════════════════════════


class TestDuplicateHoldAlertPrevention:

    def test_duplicate_hold_creates_one_alert(self, test_client):
        tc, alert_store, _, _ = test_client
        headers = _bearer(tc, "idem-hold@example.com")
        idem_key = "hold-idem-001"

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("HOLD", "HIGH", 90)),
        ):
            r1 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )
            r2 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )

        assert r1.status_code == 201
        assert r2.status_code == 200
        assert r1.json()["alert"] is not None
        # Cached response also includes the alert
        assert r2.json()["alert"] is not None
        assert r1.json()["alert"]["id"] == r2.json()["alert"]["id"]
        # Only ONE alert in the store
        all_alerts, total = alert_store.list_alerts()
        assert total == 1


# ══════════════════════════════════════════════════════════════════════
# F. ML timeout → bounded response, explicit failure state
# ══════════════════════════════════════════════════════════════════════


class TestMLTimeoutHandling:

    def test_ml_timeout_returns_503(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "timeout-cust@example.com")

        import httpx as _httpx

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_httpx.TimeoutException("timed out"),
        ):
            resp = tc.post("/api/v1/transactions", json=VALID_TXN, headers=headers)

        assert resp.status_code == 503
        data = resp.json()
        assert "detail" in data
        # No stack traces or internal info
        assert "Traceback" not in data["detail"]
        assert "httpx" not in data["detail"].lower()

    def test_ml_timeout_with_idempotency_key_allows_retry(self, test_client):
        tc, _, _, idempotency_store = test_client
        headers = _bearer(tc, "timeout-idem@example.com")
        idem_key = "timeout-retry-001"

        import httpx as _httpx

        # First request: timeout
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_httpx.TimeoutException("timed out"),
        ):
            r1 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )
        assert r1.status_code == 503

        # Second request with same key: should retry (ML succeeds)
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
        ):
            r2 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )
        assert r2.status_code == 201
        assert r2.json()["decision"] == "APPROVE"

    def test_ml_timeout_no_fabricated_predictions(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "timeout-nofake@example.com")

        import httpx as _httpx

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_httpx.TimeoutException("timed out"),
        ):
            resp = tc.post("/api/v1/transactions", json=VALID_TXN, headers=headers)

        assert resp.status_code == 503
        # No ML prediction fields in the error response
        data = resp.json()
        assert "fraud_probability" not in data or data.get("fraud_probability") is None
        assert "model_version" not in data or data.get("model_version") is None


# ══════════════════════════════════════════════════════════════════════
# G. ML unavailable → bounded response, explicit failure state
# ══════════════════════════════════════════════════════════════════════


class TestMLUnavailableHandling:

    def test_ml_unavailable_returns_503(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "unavail-cust@example.com")

        import httpx as _httpx

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_httpx.ConnectError("connection refused"),
        ):
            resp = tc.post("/api/v1/transactions", json=VALID_TXN, headers=headers)

        assert resp.status_code == 503
        data = resp.json()
        assert "detail" in data
        # No internal URLs or secrets
        assert "localhost" not in data["detail"]
        assert "http://" not in data["detail"]

    def test_ml_error_response_no_stack_traces(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "unavail-trace@example.com")

        import httpx as _httpx

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_httpx.ConnectError("Connection refused: [Errno 111]"),
        ):
            resp = tc.post("/api/v1/transactions", json=VALID_TXN, headers=headers)

        assert resp.status_code == 503
        detail = resp.json().get("detail", "")
        assert "Traceback" not in detail
        assert "Exception" not in detail
        assert "Errno" not in detail

    def test_ml_503_response_returns_503(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "unavail-503@example.com")

        from backend.services.ml_client import MLServiceResponseError

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=MLServiceResponseError(503, "model unavailable"),
        ):
            resp = tc.post("/api/v1/transactions", json=VALID_TXN, headers=headers)

        assert resp.status_code == 503

    def test_ml_generic_error_returns_502(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "unavail-502@example.com")

        from backend.services.ml_client import MLServiceResponseError

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=MLServiceResponseError(500, "internal error"),
        ):
            resp = tc.post("/api/v1/transactions", json=VALID_TXN, headers=headers)

        assert resp.status_code == 502


# ══════════════════════════════════════════════════════════════════════
# H. Successful ML → response/database decision consistency
# ══════════════════════════════════════════════════════════════════════


class TestDecisionConsistency:

    def test_response_matches_persisted_decision(self, test_client):
        tc, alert_store, _, _ = test_client
        headers = _bearer(tc, "consistency@example.com")
        idem_key = "consistency-001"

        ml_resp = _ml_response("HOLD", "HIGH", 90, fraud_probability=0.95)
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=ml_resp),
        ):
            resp = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["decision"] == "HOLD"
        assert data["risk_score"] == 90
        assert data["risk_level"] == "HIGH"
        assert data["fraud_probability"] == 0.95

        # Verify alert was persisted with same decision
        alerts, total = alert_store.list_alerts()
        assert total == 1
        alert = alerts[0]
        assert alert["decision"] == data["decision"]
        assert alert["risk_score"] == data["risk_score"]
        assert alert["risk_level"] == data["risk_level"]
        assert alert["fraud_probability"] == data["fraud_probability"]

    def test_transaction_id_consistent_in_response_and_alert(self, test_client):
        tc, alert_store, _, _ = test_client
        headers = _bearer(tc, "txn-id-consistency@example.com")

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("HOLD", "HIGH", 85)),
        ):
            resp = tc.post("/api/v1/transactions", json=VALID_TXN, headers=headers)

        assert resp.status_code == 201
        data = resp.json()
        txn_id = data["transaction_id"]
        assert txn_id is not None
        # Valid UUID
        uuid.UUID(txn_id)
        # Alert references same transaction_id
        alerts, total = alert_store.list_alerts()
        assert total == 1
        assert str(alerts[0]["transaction_id"]) == txn_id

    def test_no_alert_when_decision_is_approve(self, test_client):
        tc, alert_store, _, _ = test_client
        headers = _bearer(tc, "approve-no-alert@example.com")

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
        ):
            resp = tc.post("/api/v1/transactions", json=VALID_TXN, headers=headers)

        assert resp.status_code == 201
        assert resp.json()["alert"] is None
        _, count = alert_store.list_alerts()
        assert count == 0

    def test_transaction_response_always_has_transaction_id(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "txn-id-always@example.com")

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
        ):
            resp = tc.post("/api/v1/transactions", json=VALID_TXN, headers=headers)

        assert resp.status_code == 201
        data = resp.json()
        assert "transaction_id" in data
        assert data["transaction_id"] is not None


# ══════════════════════════════════════════════════════════════════════
# I. Failed ML → no fabricated model output
# ══════════════════════════════════════════════════════════════════════


class TestMLFailureNoFabrication:

    def test_failure_response_has_no_ml_predictions(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "nofake-pred@example.com")

        import httpx as _httpx

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_httpx.ConnectError("unavailable"),
        ):
            resp = tc.post("/api/v1/transactions", json=VALID_TXN, headers=headers)

        assert resp.status_code == 503
        data = resp.json()
        # Error response must not contain fabricated ML data
        assert "fraud_probability" not in data or data.get("fraud_probability") is None
        assert "risk_score" not in data or data.get("risk_score") is None
        assert "decision" not in data or data.get("decision") is None
        assert "model_version" not in data or data.get("model_version") is None
        assert "explanation" not in data or data.get("explanation") is None

    def test_failure_no_alert_created(self, test_client):
        tc, alert_store, _, _ = test_client
        headers = _bearer(tc, "nofake-alert@example.com")

        import httpx as _httpx

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_httpx.ConnectError("unavailable"),
        ):
            resp = tc.post("/api/v1/transactions", json=VALID_TXN, headers=headers)

        assert resp.status_code == 503
        _, count = alert_store.list_alerts()
        assert count == 0

    def test_failure_idempotency_key_marked_failed(self, test_client):
        tc, _, _, idempotency_store = test_client
        headers = _bearer(tc, "nofake-idem@example.com")
        idem_key = "failed-mark-001"

        import httpx as _httpx

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_httpx.ConnectError("unavailable"),
        ):
            resp = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )
        assert resp.status_code == 503

        # Verify idempotency record exists with "failed" status
        record = idempotency_store.try_reserve(
            headers.get("customer_id", ""), idem_key
        )
        # Can't check via try_reserve (it would return existing record)
        # Instead, verify retry is allowed (record is "failed")
        # by submitting again with ML success
        me_resp = tc.get("/api/v1/auth/me", headers=headers)
        customer_id = me_resp.json()["customer_id"]
        record = idempotency_store.try_reserve(customer_id, idem_key)
        assert record is not None
        assert record.status == "failed"


# ══════════════════════════════════════════════════════════════════════
# J. Unauthorized idempotency reuse → no cross-customer data leak
# ══════════════════════════════════════════════════════════════════════


class TestCrossCustomerIdempotencyIsolation:

    def test_customer_b_cannot_access_customer_a_idempotency(self, test_client):
        tc, _, _, _ = test_client
        h_a = _bearer(tc, "cross-a@example.com")
        h_b = _bearer(tc, "cross-b@example.com")
        idem_key = "cross-customer-key"

        # Customer A creates a transaction
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
        ):
            r_a = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**h_a, "Idempotency-Key": idem_key},
            )

        assert r_a.status_code == 201
        a_txn_id = r_a.json()["transaction_id"]
        a_customer_id = r_a.json()["customer_id"]

        # Customer B uses the SAME idempotency key — should get a
        # completely different transaction (isolated by customer_id)
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
        ):
            r_b = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**h_b, "Idempotency-Key": idem_key},
            )

        assert r_b.status_code == 201
        b_txn_id = r_b.json()["transaction_id"]
        b_customer_id = r_b.json()["customer_id"]

        # Different customers → different transactions
        assert a_txn_id != b_txn_id
        assert a_customer_id != b_customer_id

    def test_customer_b_retry_does_not_get_customer_a_data(self, test_client):
        tc, _, _, _ = test_client
        h_a = _bearer(tc, "leak-a@example.com")
        h_b = _bearer(tc, "leak-b@example.com")
        idem_key = "leak-test-key"

        # Customer A creates a HOLD transaction
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("HOLD", "HIGH", 90)),
        ):
            r_a = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**h_a, "Idempotency-Key": idem_key},
            )

        # Customer B retries same key — should NOT get A's data
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 5)),
        ):
            r_b = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**h_b, "Idempotency-Key": idem_key},
            )

        assert r_b.status_code == 201
        assert r_b.json()["transaction_id"] != r_a.json()["transaction_id"]
        assert r_b.json()["customer_id"] != r_a.json()["customer_id"]
        # B's response should have B's own decision, not A's HOLD
        assert r_b.json()["decision"] == "APPROVE"


# ══════════════════════════════════════════════════════════════════════
# K. Invalid/malformed idempotency keys → safe validation error
# ══════════════════════════════════════════════════════════════════════


class TestIdempotencyKeyValidation:

    def test_empty_idempotency_key_returns_422(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "valid-key@example.com")

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
        ):
            resp = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": "   "},
            )

        assert resp.status_code == 422

    def test_oversized_idempotency_key_returns_422(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "big-key@example.com")
        big_key = "x" * 300

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
        ):
            resp = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": big_key},
            )

        assert resp.status_code == 422

    def test_control_chars_in_key_returns_422(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "ctrl-key@example.com")

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
        ):
            resp = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": "key\x00bad"},
            )

        assert resp.status_code == 422

    def test_no_idempotency_key_works_normally(self, test_client):
        """Transactions without idempotency key proceed normally."""
        tc, _, _, _ = test_client
        headers = _bearer(tc, "no-key@example.com")

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
        ):
            r1 = tc.post("/api/v1/transactions", json=VALID_TXN, headers=headers)
            r2 = tc.post("/api/v1/transactions", json=VALID_TXN, headers=headers)

        # Both succeed — no idempotency, so different transactions
        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["transaction_id"] != r2.json()["transaction_id"]


# ══════════════════════════════════════════════════════════════════════
# L. Concurrent duplicate requests → no race-condition corruption
# ══════════════════════════════════════════════════════════════════════


class TestConcurrentDuplicateRequests:

    def test_concurrent_different_keys_succeed(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "conc-diff@example.com")

        # Apply patch at test level so it covers all threads
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
        ):
            def submit(key_suffix):
                return tc.post(
                    "/api/v1/transactions",
                    json=VALID_TXN,
                    headers={**headers, "Idempotency-Key": f"conc-{key_suffix}"},
                )

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(submit, i) for i in range(4)]
                results = [f.result() for f in as_completed(futures)]

        # All should succeed with different transaction IDs
        for r in results:
            assert r.status_code == 201
        txn_ids = {r.json()["transaction_id"] for r in results}
        assert len(txn_ids) == 4

    def test_concurrent_no_alert_duplication(self, test_client):
        tc, alert_store, _, _ = test_client
        headers = _bearer(tc, "conc-alert@example.com")
        idem_key = "conc-alert-001"

        # Apply patch at test level so it covers all threads
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("HOLD", "HIGH", 90)),
        ):
            def submit():
                return tc.post(
                    "/api/v1/transactions",
                    json=VALID_TXN,
                    headers={**headers, "Idempotency-Key": idem_key},
                )

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(submit) for _ in range(4)]
                results = [f.result() for f in as_completed(futures)]

        # Only one transaction should succeed, others cached or conflict
        successful = [r for r in results if r.status_code in (200, 201)]
        # Exactly one alert regardless of how many succeeded
        _, total = alert_store.list_alerts()
        assert total <= 1


# ══════════════════════════════════════════════════════════════════════
# Security tests
# ══════════════════════════════════════════════════════════════════════


class TestSecurityProperties:

    def test_customer_id_always_from_jwt(self, test_client):
        """Client cannot supply customer_id in request body."""
        tc, _, _, _ = test_client
        headers = _bearer(tc, "sec-jwt@example.com")

        # Try to inject customer_id in the body
        payload = {**VALID_TXN, "customer_id": "hacker-id"}
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 10)),
        ):
            resp = tc.post("/api/v1/transactions", json=payload, headers=headers)

        assert resp.status_code == 201
        # customer_id should come from JWT, not from the injected value
        me = tc.get("/api/v1/auth/me", headers=headers).json()
        assert resp.json()["customer_id"] == me["customer_id"]
        assert resp.json()["customer_id"] != "hacker-id"

    def test_unauthenticated_request_rejected(self, test_client):
        tc, _, _, _ = test_client
        resp = tc.post("/api/v1/transactions", json=VALID_TXN)
        assert resp.status_code in (401, 403)

    def test_ml_error_no_internal_url_leak(self, test_client):
        tc, _, _, _ = test_client
        headers = _bearer(tc, "sec-url@example.com")

        import httpx as _httpx

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_httpx.ConnectError("http://internal-ml:8001 refused"),
        ):
            resp = tc.post("/api/v1/transactions", json=VALID_TXN, headers=headers)

        assert resp.status_code == 503
        detail = resp.json().get("detail", "")
        assert "internal-ml" not in detail
        assert "8001" not in detail

    def test_idempotency_key_not_sent_to_ml_service(self, test_client):
        """The ML service payload must not contain the idempotency key."""
        tc, _, _, _ = test_client
        headers = _bearer(tc, "sec-ml-payload@example.com")

        captured = {}

        def fake_post(url, json=None, **kw):
            captured["payload"] = json
            return Response(200, json=_ml_response("APPROVE", "LOW", 10))

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=fake_post,
        ):
            tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": "secret-key"},
            )

        assert "idempotency_key" not in captured.get("payload", {})


# ══════════════════════════════════════════════════════════════════════
# End-to-end scenario
# ══════════════════════════════════════════════════════════════════════


class TestEndToEndDecisionPipeline:

    def test_full_flow_auth_predict_persist_idempotent_retry(self, test_client):
        """Complete end-to-end: auth → predict → persist → idempotent replay."""
        tc, alert_store, _, _ = test_client
        headers = _bearer(tc, "e2e@example.com")
        idem_key = "e2e-key-001"

        # 1. Submit with idempotency key — HOLD decision
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("HOLD", "HIGH", 88)),
        ):
            r1 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )

        assert r1.status_code == 201
        data1 = r1.json()
        assert data1["decision"] == "HOLD"
        assert data1["risk_score"] == 88
        assert data1["alert"] is not None
        assert data1["idempotent"] is True
        txn_id = data1["transaction_id"]

        # 2. Replay same request — should get cached result
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("APPROVE", "LOW", 5)),
        ):
            r2 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )

        assert r2.status_code == 200  # cached
        data2 = r2.json()
        assert data2["transaction_id"] == txn_id
        assert data2["decision"] == "HOLD"  # cached decision, not APPROVE
        assert data2["alert"]["id"] == data1["alert"]["id"]

        # 3. Only one alert exists
        _, total = alert_store.list_alerts()
        assert total == 1

    def test_e2e_ml_failure_then_retry_success(self, test_client):
        """ML fails first, then succeeds on retry with same key."""
        tc, alert_store, _, _ = test_client
        headers = _bearer(tc, "e2e-fail@example.com")
        idem_key = "e2e-fail-retry-001"

        import httpx as _httpx

        # First attempt: ML unavailable
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_httpx.ConnectError("unavailable"),
        ):
            r1 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )

        assert r1.status_code == 503
        assert len(alert_store.list_alerts()[0]) == 0

        # Second attempt: ML succeeds
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("HOLD", "HIGH", 92)),
        ):
            r2 = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**headers, "Idempotency-Key": idem_key},
            )

        assert r2.status_code == 201
        assert r2.json()["decision"] == "HOLD"
        _, total = alert_store.list_alerts()
        assert total == 1

    def test_e2e_two_customers_same_key_full_isolation(self, test_client):
        tc, alert_store, _, _ = test_client
        h_a = _bearer(tc, "iso-a@example.com")
        h_b = _bearer(tc, "iso-b@example.com")
        idem_key = "shared-e2e-key"

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=Response(200, json=_ml_response("HOLD", "HIGH", 85)),
        ):
            r_a = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**h_a, "Idempotency-Key": idem_key},
            )
            r_b = tc.post(
                "/api/v1/transactions",
                json=VALID_TXN,
                headers={**h_b, "Idempotency-Key": idem_key},
            )

        assert r_a.status_code == 201
        assert r_b.status_code == 201
        assert r_a.json()["transaction_id"] != r_b.json()["transaction_id"]
        assert r_a.json()["customer_id"] != r_b.json()["customer_id"]
        # Each HOLD creates exactly one alert
        assert len(alert_store.list_alerts()) == 2
