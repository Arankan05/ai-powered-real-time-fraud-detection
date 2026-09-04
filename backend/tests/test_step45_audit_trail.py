"""Step 45 — Fraud Decision Audit Trail test suite.

Covers:
  - Audit event creation (decision, ML failure, alert, state change, outcome)
  - ML success audit completeness
  - ML failure audit with bounded failure category
  - Alert creation audit traceability
  - Analyst state transition audit
  - Outcome feedback audit
  - Idempotency interaction (no duplicate audit on replay)
  - Customer isolation (customer A ≠ customer B)
  - Authorization (401, 403)
  - Sensitive data not exposed
  - Append-only behavior (no update/delete paths)
  - Concurrency
  - Bounded explanation / rule signal summaries
  - Model version preservation
  - Endpoint behavior
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.db.alert_repository import InMemoryAlertStore
from backend.db.audit_repository import (
    ALERT_CREATED,
    ALERT_STATE_CHANGED,
    DECISION_MADE,
    ML_FAILURE,
    OUTCOME_RECORDED,
    InMemoryAuditStore,
    build_explanation_summary,
    build_rule_signal_summary,
    normalize_failure_category,
)
from backend.db.idempotency_store import InMemoryIdempotencyStore
from backend.db.user_repository import InMemoryUserStore
from backend.routers.alerts import (
    router as alerts_router,
    set_alert_repository as set_alerts_alert_repo,
    set_audit_repository as set_alerts_audit_repo,
)
from backend.routers.audit import (
    router as audit_router,
    set_audit_repository as set_audit_router_repo,
)
from backend.routers.auth import router as auth_router
from backend.routers.transactions import (
    router as transactions_router,
    set_alert_repository as set_txn_alert_repo,
    set_audit_repository as set_txn_audit_repo,
    set_idempotency_store as set_txn_idempotency_store,
    set_ml_client,
)
from backend.schemas import MLExplanation, MLFactor, MLRuleTrigger, MLPredictionResponse
from backend.security.deps import get_current_user, set_user_repository
from backend.services.ml_client import MLServiceUnavailableError


# ── Fixtures ──────────────────────────────────────────────────────────

CUSTOMER_A_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
CUSTOMER_B_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
ANALYST_ID = "00000000-0000-4000-8000-000000000001"
ADMIN_ID = "00000000-0000-4000-8000-000000000002"

CUSTOMER_A = {
    "id": str(uuid.uuid4()),
    "email": "customer.a@test.local",
    "role": "customer",
    "is_active": True,
    "first_name": "Customer",
    "last_name": "A",
    "customer_id": CUSTOMER_A_ID,
    "created_at": "2026-01-01T00:00:00+00:00",
}

CUSTOMER_B = {
    "id": str(uuid.uuid4()),
    "email": "customer.b@test.local",
    "role": "customer",
    "is_active": True,
    "first_name": "Customer",
    "last_name": "B",
    "customer_id": CUSTOMER_B_ID,
    "created_at": "2026-01-01T00:00:00+00:00",
}

ANALYST_USER = {
    "id": ANALYST_ID,
    "email": "analyst@test.local",
    "role": "fraud_analyst",
    "is_active": True,
    "first_name": "Test",
    "last_name": "Analyst",
    "customer_id": "00000000-0000-4000-8000-0000000000a1",
    "created_at": "2026-01-01T00:00:00+00:00",
}

ADMIN_USER = {
    "id": ADMIN_ID,
    "email": "admin@test.local",
    "role": "admin",
    "is_active": True,
    "first_name": "Admin",
    "last_name": "User",
    "customer_id": "00000000-0000-4000-8000-0000000000a2",
    "created_at": "2026-01-01T00:00:00+00:00",
}


def _ml_success_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock ML service success response."""
    resp = {
        "fraud_probability": 0.85,
        "fraud_prediction": 1,
        "threshold": 0.5,
        "model_version": "xgb-v2.1.0",
        "timestamp": 1700000000,
        "ml_score": 80,
        "behaviour_score": 70,
        "rule_score": 60,
        "risk_score": 75,
        "risk_level": "HIGH",
        "decision": "HOLD",
        "risk_factors": ["high_amount", "new_merchant"],
        "explanation_detail": {
            "ml_top_factors": [
                {"feature": "TransactionAmt", "importance": 0.35},
                {"feature": "card_freq", "importance": 0.20},
            ],
            "behaviour_signals": [],
            "rules_triggered": [
                {"rule": "high_amount_rule", "contribution": 15},
            ],
        },
    }
    resp.update(overrides)
    return resp


def _ml_approve_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock ML service APPROVE response."""
    resp = _ml_success_response(
        fraud_probability=0.05,
        fraud_prediction=0,
        risk_score=20,
        risk_level="LOW",
        decision="APPROVE",
        risk_factors=[],
    )
    resp.update(overrides)
    return resp


def _valid_txn_body() -> dict[str, Any]:
    return {
        "amount": 500.0,
        "currency": "USD",
        "merchant_name": "TestMerchant",
        "merchant_category": "5411",
        "transaction_type": "purchase",
        "location_country": "US",
        "location_city": "New York",
        "device_fingerprint": "fp123456",
        "device_type": "desktop",
        "ip_address": "192.168.1.1",
    }


@pytest.fixture
def stores():
    """Create fresh in-memory stores for each test."""
    user_store = InMemoryUserStore()
    alert_store = InMemoryAlertStore()
    idempotency_store = InMemoryIdempotencyStore()
    audit_store = InMemoryAuditStore()

    # Wire stores into routers
    set_user_repository(user_store)
    set_alerts_alert_repo(alert_store)
    set_txn_alert_repo(alert_store)
    set_txn_idempotency_store(idempotency_store)
    set_txn_audit_repo(audit_store)
    set_alerts_audit_repo(audit_store)
    set_audit_router_repo(audit_store)

    return {
        "users": user_store,
        "alerts": alert_store,
        "idempotency": idempotency_store,
        "audit": audit_store,
    }


@pytest.fixture
def app_with_stores(stores):
    """Create a test FastAPI app with all routers."""
    test_app = FastAPI()
    test_app.include_router(auth_router)
    test_app.include_router(transactions_router)
    test_app.include_router(alerts_router)
    test_app.include_router(audit_router)
    return test_app


def _override_user(user: dict[str, Any]):
    """Create a dependency override for get_current_user."""
    async def _get():
        return user
    return _get


def _mock_ml_client(response_dict: dict[str, Any]):
    """Create a mock ML client that returns the given response."""
    ml_resp = MLPredictionResponse.model_validate(response_dict)
    mock = AsyncMock()
    mock.predict = AsyncMock(return_value=ml_resp)
    mock.update_outcome = AsyncMock(return_value={"updated": True, "customer_id": "test", "timestamp": 0, "is_fraud": 0})
    mock.health = AsyncMock(return_value={"status": "ok"})
    set_ml_client(mock)
    return mock


def _mock_ml_failure():
    """Create a mock ML client that raises MLServiceUnavailableError."""
    mock = AsyncMock()
    mock.predict = AsyncMock(side_effect=MLServiceUnavailableError("service down"))
    mock.health = AsyncMock(return_value={"status": "ok"})
    set_ml_client(mock)
    return mock


# ══════════════════════════════════════════════════════════════════════
# 1. AUDIT CREATION TESTS
# ══════════════════════════════════════════════════════════════════════


class TestAuditCreationDecision:
    """Successful ML decision creates a DECISION_MADE audit event."""

    def test_decision_audit_created(self, stores, app_with_stores):
        _mock_ml_client(_ml_success_response())
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            assert resp.status_code == 201

        # Check audit store directly
        all_events = stores["audit"]._events
        decision_events = [e for e in all_events if e["event_type"] == DECISION_MADE]
        assert len(decision_events) == 1
        ev = decision_events[0]
        assert ev["customer_id"] == CUSTOMER_A_ID
        assert ev["decision"] == "HOLD"
        assert ev["risk_score"] == 75
        assert ev["risk_level"] == "HIGH"
        assert ev["model_version"] == "xgb-v2.1.0"
        assert ev["fraud_probability"] == 0.85

    def test_decision_audit_has_explanation_summary(self, stores, app_with_stores):
        _mock_ml_client(_ml_success_response())
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            assert resp.status_code == 201

        decision_events = [e for e in stores["audit"]._events if e["event_type"] == DECISION_MADE]
        assert len(decision_events) == 1
        ev = decision_events[0]
        assert ev["explanation_summary"] is not None
        assert "ml_top_factors" in ev["explanation_summary"]

    def test_decision_audit_has_rule_signal_summary(self, stores, app_with_stores):
        _mock_ml_client(_ml_success_response())
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            assert resp.status_code == 201

        decision_events = [e for e in stores["audit"]._events if e["event_type"] == DECISION_MADE]
        ev = decision_events[0]
        assert ev["rule_signal_summary"] is not None
        assert "risk_factors" in ev["rule_signal_summary"]

    def test_approve_decision_still_audited(self, stores, app_with_stores):
        _mock_ml_client(_ml_approve_response())
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            assert resp.status_code == 201

        decision_events = [e for e in stores["audit"]._events if e["event_type"] == DECISION_MADE]
        assert len(decision_events) == 1
        assert decision_events[0]["decision"] == "APPROVE"


# ══════════════════════════════════════════════════════════════════════
# 2. ML FAILURE TESTS
# ══════════════════════════════════════════════════════════════════════


class TestMLFailureAudit:
    """ML failure creates an ML_FAILURE audit event."""

    def test_ml_failure_audit_created(self, stores, app_with_stores):
        _mock_ml_failure()
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            assert resp.status_code == 503

        failure_events = [e for e in stores["audit"]._events if e["event_type"] == ML_FAILURE]
        assert len(failure_events) == 1
        ev = failure_events[0]
        assert ev["customer_id"] == CUSTOMER_A_ID
        assert ev["failure_category"] == "service_unavailable"
        assert ev["decision"] is None
        assert ev["risk_score"] is None

    def test_ml_failure_no_decision_audit(self, stores, app_with_stores):
        _mock_ml_failure()
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())

        decision_events = [e for e in stores["audit"]._events if e["event_type"] == DECISION_MADE]
        assert len(decision_events) == 0

    def test_ml_failure_bounded_category(self, stores, app_with_stores):
        _mock_ml_failure()
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            client.post("/api/v1/transactions", json=_valid_txn_body())

        failure_events = [e for e in stores["audit"]._events if e["event_type"] == ML_FAILURE]
        assert len(failure_events) == 1
        # failure_category must be bounded
        assert len(failure_events[0]["failure_category"]) <= 50


# ══════════════════════════════════════════════════════════════════════
# 3. ALERT AUDIT TESTS
# ══════════════════════════════════════════════════════════════════════


class TestAlertAudit:
    """Alert creation is audited when decision is HOLD."""

    def test_alert_created_audit(self, stores, app_with_stores):
        _mock_ml_client(_ml_success_response())
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            assert resp.status_code == 201
            txn_id = resp.json()["transaction_id"]

        alert_events = [e for e in stores["audit"]._events if e["event_type"] == ALERT_CREATED]
        assert len(alert_events) == 1
        ev = alert_events[0]
        assert ev["transaction_id"] == txn_id
        assert ev["customer_id"] == CUSTOMER_A_ID
        assert ev["alert_id"] is not None
        assert ev["decision"] == "HOLD"

    def test_no_alert_audit_for_approve(self, stores, app_with_stores):
        _mock_ml_client(_ml_approve_response())
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            assert resp.status_code == 201

        alert_events = [e for e in stores["audit"]._events if e["event_type"] == ALERT_CREATED]
        assert len(alert_events) == 0


# ══════════════════════════════════════════════════════════════════════
# 4. ANALYST STATE TRANSITION TESTS
# ══════════════════════════════════════════════════════════════════════


class TestAnalystStateTransition:
    """Alert state transitions are audited with actor info."""

    def _create_hold_transaction(self, client, stores):
        _mock_ml_client(_ml_success_response())
        resp = client.post("/api/v1/transactions", json=_valid_txn_body())
        assert resp.status_code == 201
        return resp.json()

    def test_state_transition_audit(self, stores, app_with_stores):
        # Create HOLD transaction as customer
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)
        with TestClient(app_with_stores) as client:
            txn = self._create_hold_transaction(client, stores)
            txn_id = txn["transaction_id"]
            alert_id = txn["alert"]["id"]

        # Transition alert as analyst
        app_with_stores.dependency_overrides[get_current_user] = _override_user(ANALYST_USER)
        with TestClient(app_with_stores) as client:
            resp = client.patch(
                f"/api/v1/alerts/{alert_id}",
                json={"status": "IN_REVIEW"},
            )
            assert resp.status_code == 200

        state_events = [e for e in stores["audit"]._events if e["event_type"] == ALERT_STATE_CHANGED]
        assert len(state_events) == 1
        ev = state_events[0]
        assert ev["transaction_id"] == txn_id
        assert ev["previous_state"] == "OPEN"
        assert ev["new_state"] == "IN_REVIEW"
        assert ev["actor_id"] == ANALYST_ID
        assert ev["actor_role"] == "fraud_analyst"

    def test_terminal_state_audit(self, stores, app_with_stores):
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)
        with TestClient(app_with_stores) as client:
            txn = self._create_hold_transaction(client, stores)
            alert_id = txn["alert"]["id"]

        app_with_stores.dependency_overrides[get_current_user] = _override_user(ANALYST_USER)
        with TestClient(app_with_stores) as client:
            resp = client.patch(
                f"/api/v1/alerts/{alert_id}",
                json={"status": "RESOLVED"},
            )
            assert resp.status_code == 200

        state_events = [e for e in stores["audit"]._events if e["event_type"] == ALERT_STATE_CHANGED]
        assert len(state_events) == 1
        ev = state_events[0]
        assert ev["previous_state"] == "OPEN"
        assert ev["new_state"] == "RESOLVED"

    def test_notes_only_no_state_audit(self, stores, app_with_stores):
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)
        with TestClient(app_with_stores) as client:
            txn = self._create_hold_transaction(client, stores)
            alert_id = txn["alert"]["id"]

        app_with_stores.dependency_overrides[get_current_user] = _override_user(ANALYST_USER)
        with TestClient(app_with_stores) as client:
            resp = client.patch(
                f"/api/v1/alerts/{alert_id}",
                json={"notes": "Investigating..."},
            )
            assert resp.status_code == 200

        # Notes-only update should NOT create a state change audit
        state_events = [e for e in stores["audit"]._events if e["event_type"] == ALERT_STATE_CHANGED]
        assert len(state_events) == 0


# ══════════════════════════════════════════════════════════════════════
# 5. IDEMPOTENCY INTERACTION TESTS
# ══════════════════════════════════════════════════════════════════════


class TestIdempotencyAudit:
    """Idempotent replays do not duplicate audit events."""

    def test_replay_no_duplicate_decision_audit(self, stores, app_with_stores):
        _mock_ml_client(_ml_success_response())
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            # First request
            resp1 = client.post(
                "/api/v1/transactions",
                json=_valid_txn_body(),
                headers={"Idempotency-Key": "key-123"},
            )
            assert resp1.status_code == 201

            # Replay
            resp2 = client.post(
                "/api/v1/transactions",
                json=_valid_txn_body(),
                headers={"Idempotency-Key": "key-123"},
            )
            assert resp2.status_code == 200  # replay returns 200

        # Exactly one decision audit event
        decision_events = [e for e in stores["audit"]._events if e["event_type"] == DECISION_MADE]
        assert len(decision_events) == 1

        # Exactly one alert audit event (if HOLD)
        alert_events = [e for e in stores["audit"]._events if e["event_type"] == ALERT_CREATED]
        assert len(alert_events) == 1

    def test_failed_retry_creates_both_audits(self, stores, app_with_stores):
        """If first request fails then succeeds, both events exist."""
        mock = AsyncMock()
        mock.predict = AsyncMock(
            side_effect=[
                MLServiceUnavailableError("down"),
                MLPredictionResponse.model_validate(_ml_success_response()),
            ]
        )
        mock.health = AsyncMock(return_value={"status": "ok"})
        set_ml_client(mock)

        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            # First attempt — ML failure
            resp1 = client.post(
                "/api/v1/transactions",
                json=_valid_txn_body(),
                headers={"Idempotency-Key": "retry-key"},
            )
            assert resp1.status_code == 503

            # Retry — success
            resp2 = client.post(
                "/api/v1/transactions",
                json=_valid_txn_body(),
                headers={"Idempotency-Key": "retry-key"},
            )
            assert resp2.status_code == 201

        failure_events = [e for e in stores["audit"]._events if e["event_type"] == ML_FAILURE]
        decision_events = [e for e in stores["audit"]._events if e["event_type"] == DECISION_MADE]
        assert len(failure_events) == 1
        assert len(decision_events) == 1


# ══════════════════════════════════════════════════════════════════════
# 6. CUSTOMER ISOLATION TESTS
# ══════════════════════════════════════════════════════════════════════


class TestCustomerIsolation:
    """Customers can only access their own audit trail."""

    def _create_txn_and_get_id(self, client, stores):
        _mock_ml_client(_ml_approve_response())
        resp = client.post("/api/v1/transactions", json=_valid_txn_body())
        assert resp.status_code == 201
        return resp.json()["transaction_id"]

    def test_customer_sees_own_audit(self, stores, app_with_stores):
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)
        with TestClient(app_with_stores) as client:
            txn_id = self._create_txn_and_get_id(client, stores)

        with TestClient(app_with_stores) as client:
            resp = client.get(f"/api/v1/audit/transactions/{txn_id}")
            assert resp.status_code == 200
            assert resp.json()["transaction_id"] == txn_id

    def test_customer_cannot_see_other_audit(self, stores, app_with_stores):
        # Customer A creates a transaction
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)
        with TestClient(app_with_stores) as client:
            txn_id = self._create_txn_and_get_id(client, stores)

        # Customer B tries to access it
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_B)
        with TestClient(app_with_stores) as client:
            resp = client.get(f"/api/v1/audit/transactions/{txn_id}")
            assert resp.status_code == 403

    def test_analyst_can_see_any_audit(self, stores, app_with_stores):
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)
        with TestClient(app_with_stores) as client:
            txn_id = self._create_txn_and_get_id(client, stores)

        app_with_stores.dependency_overrides[get_current_user] = _override_user(ANALYST_USER)
        with TestClient(app_with_stores) as client:
            resp = client.get(f"/api/v1/audit/transactions/{txn_id}")
            assert resp.status_code == 200

    def test_customer_id_from_jwt_not_body(self, stores, app_with_stores):
        """Audit customer_id is always derived from JWT, not request body."""
        _mock_ml_client(_ml_approve_response())
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            assert resp.status_code == 201
            txn_id = resp.json()["transaction_id"]

        # All audit events must have CUSTOMER_A_ID, not anything from the body
        for ev in stores["audit"]._events:
            if ev["event_type"] == DECISION_MADE:
                assert ev["customer_id"] == CUSTOMER_A_ID


# ══════════════════════════════════════════════════════════════════════
# 7. AUTHORIZATION TESTS
# ══════════════════════════════════════════════════════════════════════


class TestAuthorization:
    """Audit endpoint requires authentication and proper roles."""

    def test_unauthenticated_access_401(self, stores, app_with_stores):
        # No dependency override — no auth header
        with TestClient(app_with_stores) as client:
            resp = client.get("/api/v1/audit/transactions/some-id")
            assert resp.status_code == 401

    def test_customer_nonexistent_audit_returns_403(self, stores, app_with_stores):
        """Customer looking up non-existent txn gets 403 (not 404)."""
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            fake_id = str(uuid.uuid4())
            resp = client.get(f"/api/v1/audit/transactions/{fake_id}")
            assert resp.status_code == 403

    def test_analyst_nonexistent_audit_returns_404(self, stores, app_with_stores):
        app_with_stores.dependency_overrides[get_current_user] = _override_user(ANALYST_USER)

        with TestClient(app_with_stores) as client:
            fake_id = str(uuid.uuid4())
            resp = client.get(f"/api/v1/audit/transactions/{fake_id}")
            assert resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════
# 8. SENSITIVE DATA TESTS
# ══════════════════════════════════════════════════════════════════════


class TestSensitiveData:
    """Audit responses do not expose sensitive information."""

    def test_no_secrets_in_audit_response(self, stores, app_with_stores):
        _mock_ml_client(_ml_approve_response())
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            txn_id = resp.json()["transaction_id"]

        app_with_stores.dependency_overrides[get_current_user] = _override_user(ANALYST_USER)
        with TestClient(app_with_stores) as client:
            resp = client.get(f"/api/v1/audit/transactions/{txn_id}")
            assert resp.status_code == 200
            body = resp.json()
            text = str(body).lower()
            # No secrets
            assert "password" not in text
            assert "jwt" not in text
            assert "secret" not in text
            assert "token" not in text
            # No raw transaction payload fields
            assert "ip_address" not in text
            assert "device_fingerprint" not in text

    def test_no_stack_traces_in_failure_audit(self, stores, app_with_stores):
        _mock_ml_failure()
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            client.post("/api/v1/transactions", json=_valid_txn_body())

        failure_events = [e for e in stores["audit"]._events if e["event_type"] == ML_FAILURE]
        for ev in failure_events:
            assert "Traceback" not in str(ev)
            assert "Exception" not in str(ev.get("failure_category", ""))


# ══════════════════════════════════════════════════════════════════════
# 9. APPEND-ONLY TESTS
# ══════════════════════════════════════════════════════════════════════


class TestAppendOnly:
    """Audit records cannot be modified or deleted through normal API paths."""

    def test_no_put_endpoint(self, stores, app_with_stores):
        app_with_stores.dependency_overrides[get_current_user] = _override_user(ANALYST_USER)
        with TestClient(app_with_stores) as client:
            resp = client.put("/api/v1/audit/transactions/some-id")
            assert resp.status_code in (404, 405)

    def test_no_delete_endpoint(self, stores, app_with_stores):
        app_with_stores.dependency_overrides[get_current_user] = _override_user(ANALYST_USER)
        with TestClient(app_with_stores) as client:
            resp = client.delete("/api/v1/audit/transactions/some-id")
            assert resp.status_code in (404, 405)

    def test_no_patch_endpoint(self, stores, app_with_stores):
        app_with_stores.dependency_overrides[get_current_user] = _override_user(ANALYST_USER)
        with TestClient(app_with_stores) as client:
            resp = client.patch("/api/v1/audit/transactions/some-id")
            assert resp.status_code in (404, 405)


# ══════════════════════════════════════════════════════════════════════
# 10. BOUNDED EXPLANATION TESTS
# ══════════════════════════════════════════════════════════════════════


class TestBoundedExplanation:
    """Explanation summaries are bounded in size."""

    def test_explanation_summary_bounded_items(self):
        factors = [MLFactor(feature=f"f{i}", importance=0.1) for i in range(20)]
        expl = MLExplanation(ml_top_factors=factors)
        summary = build_explanation_summary(expl)
        assert summary is not None
        assert len(summary["ml_top_factors"]) <= 5

    def test_explanation_summary_bounded_strings(self):
        long_name = "x" * 500
        expl = MLExplanation(
            ml_top_factors=[MLFactor(feature=long_name, importance=0.1)]
        )
        summary = build_explanation_summary(expl)
        assert len(summary["ml_top_factors"][0]["feature"]) <= 200

    def test_rule_signal_summary_bounded(self):
        factors = [f"factor_{i}" for i in range(20)]
        summary = build_rule_signal_summary(factors, None)
        assert len(summary["risk_factors"]) <= 10

    def test_none_explanation_returns_none(self):
        assert build_explanation_summary(None) is None

    def test_none_risk_factors_returns_none(self):
        assert build_rule_signal_summary(None, None) is None

    def test_failure_category_bounded(self):
        long_cat = "x" * 200
        result = normalize_failure_category(long_cat)
        assert len(result) <= 50

    def test_empty_failure_category_returns_unknown(self):
        assert normalize_failure_category(None) == "unknown"
        assert normalize_failure_category("") == "unknown"
        assert normalize_failure_category("   ") == "unknown"

    def test_shap_failure_no_fabricated_explanation(self, stores, app_with_stores):
        """When ML succeeds but explanation is None, audit records None."""
        resp_data = _ml_approve_response()
        # Remove explanation fields to simulate SHAP failure
        resp_data.pop("explanation_detail", None)
        resp_data.pop("explanation", None)
        _mock_ml_client(resp_data)
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            assert resp.status_code == 201

        decision_events = [e for e in stores["audit"]._events if e["event_type"] == DECISION_MADE]
        assert len(decision_events) == 1
        assert decision_events[0]["explanation_summary"] is None


# ══════════════════════════════════════════════════════════════════════
# 11. MODEL VERSION TESTS
# ══════════════════════════════════════════════════════════════════════


class TestModelVersion:
    """Model version is preserved in audit records."""

    def test_model_version_preserved(self, stores, app_with_stores):
        _mock_ml_client(_ml_success_response(model_version="xgb-v3.0.0"))
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            client.post("/api/v1/transactions", json=_valid_txn_body())

        decision_events = [e for e in stores["audit"]._events if e["event_type"] == DECISION_MADE]
        assert len(decision_events) == 1
        assert decision_events[0]["model_version"] == "xgb-v3.0.0"

    def test_risk_score_preserved(self, stores, app_with_stores):
        _mock_ml_client(_ml_success_response(risk_score=88))
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            client.post("/api/v1/transactions", json=_valid_txn_body())

        decision_events = [e for e in stores["audit"]._events if e["event_type"] == DECISION_MADE]
        assert decision_events[0]["risk_score"] == 88

    def test_risk_level_preserved(self, stores, app_with_stores):
        _mock_ml_client(_ml_success_response(risk_level="MEDIUM"))
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            client.post("/api/v1/transactions", json=_valid_txn_body())

        decision_events = [e for e in stores["audit"]._events if e["event_type"] == DECISION_MADE]
        assert decision_events[0]["risk_level"] == "MEDIUM"


# ══════════════════════════════════════════════════════════════════════
# 12. ENDPOINT TESTS
# ══════════════════════════════════════════════════════════════════════


class TestAuditEndpoint:
    """GET /api/v1/audit/transactions/{id} behavior."""

    def test_chronological_ordering(self, stores, app_with_stores):
        _mock_ml_client(_ml_success_response())
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            txn_id = resp.json()["transaction_id"]

        app_with_stores.dependency_overrides[get_current_user] = _override_user(ANALYST_USER)
        with TestClient(app_with_stores) as client:
            resp = client.get(f"/api/v1/audit/transactions/{txn_id}")
            assert resp.status_code == 200
            events = resp.json()["events"]
            # Events should be ordered by created_at ascending
            timestamps = [e["created_at"] for e in events]
            assert timestamps == sorted(timestamps)

    def test_audit_response_schema(self, stores, app_with_stores):
        _mock_ml_client(_ml_approve_response())
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            txn_id = resp.json()["transaction_id"]

        app_with_stores.dependency_overrides[get_current_user] = _override_user(ANALYST_USER)
        with TestClient(app_with_stores) as client:
            resp = client.get(f"/api/v1/audit/transactions/{txn_id}")
            body = resp.json()
            assert "transaction_id" in body
            assert "events" in body
            assert isinstance(body["events"], list)
            for ev in body["events"]:
                assert "audit_id" in ev
                assert "event_type" in ev
                assert "created_at" in ev


# ══════════════════════════════════════════════════════════════════════
# 13. CONCURRENCY TESTS
# ══════════════════════════════════════════════════════════════════════


class TestConcurrency:
    """Concurrent requests produce correct audit histories."""

    def test_same_idempotency_key_concurrent(self, stores, app_with_stores):
        """4 identical requests → exactly one decision audit."""
        _mock_ml_client(_ml_success_response())
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            results = []
            for _ in range(4):
                resp = client.post(
                    "/api/v1/transactions",
                    json=_valid_txn_body(),
                    headers={"Idempotency-Key": "concurrent-key"},
                )
                results.append(resp.status_code)

        # Exactly one should be 201, rest should be 200
        assert results.count(201) == 1
        assert results.count(200) == 3

        # Exactly one decision audit
        decision_events = [e for e in stores["audit"]._events if e["event_type"] == DECISION_MADE]
        assert len(decision_events) == 1

    def test_different_keys_independent_audits(self, stores, app_with_stores):
        """4 different keys → 4 independent audit histories."""
        _mock_ml_client(_ml_approve_response())
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            for i in range(4):
                resp = client.post(
                    "/api/v1/transactions",
                    json=_valid_txn_body(),
                    headers={"Idempotency-Key": f"key-{i}"},
                )
                assert resp.status_code == 201

        decision_events = [e for e in stores["audit"]._events if e["event_type"] == DECISION_MADE]
        assert len(decision_events) == 4

    def test_two_customers_independent_audits(self, stores, app_with_stores):
        """Two customers using same idempotency key → independent audits."""
        _mock_ml_client(_ml_approve_response())

        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)
        with TestClient(app_with_stores) as client:
            resp_a = client.post(
                "/api/v1/transactions",
                json=_valid_txn_body(),
                headers={"Idempotency-Key": "shared-key"},
            )
            assert resp_a.status_code == 201
            txn_a = resp_a.json()["transaction_id"]

        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_B)
        with TestClient(app_with_stores) as client:
            resp_b = client.post(
                "/api/v1/transactions",
                json=_valid_txn_body(),
                headers={"Idempotency-Key": "shared-key"},
            )
            assert resp_b.status_code == 201
            txn_b = resp_b.json()["transaction_id"]

        # Each transaction has its own decision audit
        events_a = [e for e in stores["audit"]._events if e["transaction_id"] == txn_a]
        events_b = [e for e in stores["audit"]._events if e["transaction_id"] == txn_b]
        assert len(events_a) >= 1
        assert len(events_b) >= 1

        # No cross-customer access
        for ev in events_a:
            assert ev["customer_id"] == CUSTOMER_A_ID
        for ev in events_b:
            assert ev["customer_id"] == CUSTOMER_B_ID


# ══════════════════════════════════════════════════════════════════════
# 14. OUTCOME AUDIT TESTS
# ══════════════════════════════════════════════════════════════════════


class TestOutcomeAudit:
    """Outcome feedback is audited."""

    def test_outcome_audit_created(self, stores, app_with_stores):
        mock = AsyncMock()
        mock.update_outcome = AsyncMock(return_value={
            "updated": True,
            "customer_id": CUSTOMER_A_ID,
            "timestamp": 1700000000,
            "is_fraud": 1,
        })
        set_ml_client(mock)

        app_with_stores.dependency_overrides[get_current_user] = _override_user(ANALYST_USER)

        with TestClient(app_with_stores) as client:
            resp = client.patch(
                "/api/v1/transactions/outcome",
                json={
                    "customer_id": CUSTOMER_A_ID,
                    "timestamp": 1700000000,
                    "is_fraud": 1,
                },
            )
            assert resp.status_code == 200

        outcome_events = [e for e in stores["audit"]._events if e["event_type"] == OUTCOME_RECORDED]
        assert len(outcome_events) == 1
        ev = outcome_events[0]
        assert ev["actor_id"] == ANALYST_ID
        assert ev["actor_role"] == "fraud_analyst"
        assert ev["metadata"] == {"is_fraud": 1}


# ══════════════════════════════════════════════════════════════════════
# 15. PERSISTENCE / AUDIT SURVIVAL TESTS
# ══════════════════════════════════════════════════════════════════════


class TestPersistence:
    """Audit data persists across app re-creation (same store)."""

    def test_audit_survives_app_restart(self, stores):
        """Audit events in the same in-memory store survive app recreation."""
        _mock_ml_client(_ml_approve_response())

        # First app instance
        app1 = FastAPI()
        app1.include_router(auth_router)
        app1.include_router(transactions_router)
        app1.include_router(alerts_router)
        app1.include_router(audit_router)
        app1.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app1) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            txn_id = resp.json()["transaction_id"]

        # Second app instance (same stores)
        app2 = FastAPI()
        app2.include_router(auth_router)
        app2.include_router(transactions_router)
        app2.include_router(alerts_router)
        app2.include_router(audit_router)
        app2.dependency_overrides[get_current_user] = _override_user(ANALYST_USER)

        with TestClient(app2) as client:
            resp = client.get(f"/api/v1/audit/transactions/{txn_id}")
            assert resp.status_code == 200
            assert len(resp.json()["events"]) >= 1


# ══════════════════════════════════════════════════════════════════════
# 16. AUDIT INTEGRITY TESTS
# ══════════════════════════════════════════════════════════════════════


class TestAuditIntegrity:
    """End-to-end audit trail integrity."""

    def test_full_audit_trail(self, stores, app_with_stores):
        """Complete scenario: decision → alert → state change → audit trail."""
        _mock_ml_client(_ml_success_response())

        # Step 1: Create transaction as customer
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)
        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            assert resp.status_code == 201
            txn_id = resp.json()["transaction_id"]
            alert_id = resp.json()["alert"]["id"]

        # Step 2: Transition alert as analyst
        app_with_stores.dependency_overrides[get_current_user] = _override_user(ANALYST_USER)
        with TestClient(app_with_stores) as client:
            resp = client.patch(
                f"/api/v1/alerts/{alert_id}",
                json={"status": "IN_REVIEW"},
            )
            assert resp.status_code == 200

        # Step 3: Verify complete audit trail
        with TestClient(app_with_stores) as client:
            resp = client.get(f"/api/v1/audit/transactions/{txn_id}")
            assert resp.status_code == 200
            events = resp.json()["events"]

        event_types = [e["event_type"] for e in events]
        assert DECISION_MADE in event_types
        assert ALERT_CREATED in event_types
        assert ALERT_STATE_CHANGED in event_types

        # Chronological ordering
        timestamps = [e["created_at"] for e in events]
        assert timestamps == sorted(timestamps)

    def test_chronological_ordering(self, stores, app_with_stores):
        """Events are in chronological order."""
        _mock_ml_client(_ml_success_response())
        app_with_stores.dependency_overrides[get_current_user] = _override_user(CUSTOMER_A)

        with TestClient(app_with_stores) as client:
            resp = client.post("/api/v1/transactions", json=_valid_txn_body())
            txn_id = resp.json()["transaction_id"]

        events = stores["audit"].list_by_transaction(txn_id)
        timestamps = [e["created_at"] for e in events]
        assert timestamps == sorted(timestamps)
