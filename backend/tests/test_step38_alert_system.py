"""Step 38 — Fraud Alert System tests.

Validates the complete alert lifecycle:
  1. Automatic OPEN alert creation when decision == HOLD (HIGH risk)
  2. No alert for APPROVE / VERIFY decisions
  3. Correct risk/transaction information stored in the alert
  4. Duplicate-alert protection
  5. GET /api/v1/alerts (list, filter, paginate)
  6. GET /api/v1/alerts/{id}
  7. PATCH /api/v1/alerts/{id} — valid transitions
     (OPEN → IN_REVIEW → RESOLVED / DISMISSED)
  8. Invalid status transitions rejected
  9. Alert not found / malformed request handling
 10. Persistence across restart (SQLite)
 11. Transaction behavior unchanged (response fields intact)
 12. ML failure behavior unchanged
 13. No sensitive information leakage
 14. No ML implementation imports in the backend
 15. Immutable risk fields cannot be modified through the API

Uses the same style as test_step37_risk_integration.py: httpx mock
transport for the ML service, TestClient for backend endpoints, and
InMemoryAlertStore / temp-file SQLiteAlertRepository for persistence.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import Response

from backend.db.alert_repository import (
    DISMISSED,
    IN_REVIEW,
    OPEN,
    RESOLVED,
    VALID_STATUSES,
    InMemoryAlertStore,
    SQLiteAlertRepository,
    is_valid_transition,
)
from backend.schemas import AlertUpdate
from backend.services.ml_client import MLServiceClient


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def valid_transaction() -> dict:
    """Valid raw transaction payload."""
    return {
        "amount": 15000.00,
        "currency": "USD",
        "merchant_name": "Offshore Trading Ltd",
        "merchant_category": "5999",
        "transaction_type": "transfer",
        "location_country": "KY",
        "location_city": "George Town",
        "device_fingerprint": "alert_test_device_001",
        "device_type": "desktop",
        "ip_address": "10.0.0.99",
    }


def _ml_response(decision: str, risk_level: str, risk_score: int) -> dict:
    """Build a complete ML response with the given decision."""
    return {
        "fraud_probability": 0.91,
        "fraud_prediction": 1,
        "threshold": 0.50,
        "model_version": "fraud-xgb-v1.0.0",
        "timestamp": 1725200000,
        "explanation": [
            {"feature": "amount_deviation", "importance": 0.45},
        ],
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
            "behaviour_signals": [
                {
                    "signal": "spending_amount_anomaly",
                    "severity": 0.85,
                    "reason": "Amount 5.2x above average",
                },
            ],
            "rules_triggered": [
                {
                    "rule": "high_amount",
                    "contribution": 15,
                    "reason": "Amount > $10000",
                },
            ],
        },
        "risk_factors": ["amount_deviation", "spending_amount_anomaly", "high_amount"],
    }


@pytest.fixture
def ml_hold_response() -> dict:
    """ML response with decision=HOLD, risk_level=HIGH, risk_score>70."""
    return _ml_response("HOLD", "HIGH", 85)


@pytest.fixture
def ml_verify_response() -> dict:
    return _ml_response("VERIFY", "MEDIUM", 50)


@pytest.fixture
def ml_approve_response() -> dict:
    return _ml_response("APPROVE", "LOW", 12)


@pytest.fixture
def test_client(auth_override):
    """TestClient with ML client + in-memory alert store configured."""
    from fastapi.testclient import TestClient
    from backend.app import app
    from backend.routers import alerts as alerts_module
    from backend.routers import transactions as txn_module

    ml_client = MLServiceClient(base_url="http://mock-ml:8001")
    txn_module.set_ml_client(ml_client)

    store = InMemoryAlertStore()
    alerts_module.set_alert_repository(store)
    txn_module.set_alert_repository(store)

    return TestClient(app), store


# ── 1. Automatic alert creation ──────────────────────────────────────


class TestAutomaticAlertCreation:
    """HOLD transactions automatically create OPEN alerts."""

    def test_hold_creates_open_alert(self, test_client, valid_transaction, ml_hold_response):
        tc, store = test_client
        mock_resp = Response(200, json=ml_hold_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)

        assert resp.status_code == 201
        data = resp.json()
        # Response embeds the alert summary
        assert data["alert"] is not None
        assert data["alert"]["status"] == "OPEN"
        assert data["alert"]["id"]
        assert data["alert"]["created_at"]

    def test_hold_alert_persisted_in_store(self, test_client, valid_transaction, ml_hold_response):
        tc, store = test_client
        mock_resp = Response(200, json=ml_hold_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)

        alert_id = resp.json()["alert"]["id"]
        stored = store.get_by_id(alert_id)
        assert stored is not None
        assert stored["status"] == "OPEN"
        assert stored["decision"] == "HOLD"
        assert stored["risk_level"] == "HIGH"
        assert stored["risk_score"] == 85

    def test_high_risk_score_above_70(self, test_client, valid_transaction):
        """risk_score > 70 with decision HOLD creates an alert."""
        tc, store = test_client
        ml_resp = _ml_response("HOLD", "HIGH", 71)
        mock_resp = Response(200, json=ml_resp)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        assert resp.json()["alert"] is not None
        assert store.get_by_id(resp.json()["alert"]["id"])["risk_score"] == 71

    def test_no_alert_for_approve(self, test_client, valid_transaction, ml_approve_response):
        tc, store = test_client
        mock_resp = Response(200, json=ml_approve_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)

        assert resp.status_code == 201
        assert resp.json()["alert"] is None
        _, total = store.list_alerts()
        assert total == 0

    def test_no_alert_for_verify(self, test_client, valid_transaction, ml_verify_response):
        tc, store = test_client
        mock_resp = Response(200, json=ml_verify_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)

        assert resp.status_code == 201
        assert resp.json()["alert"] is None
        _, total = store.list_alerts()
        assert total == 0

    def test_alert_contains_correct_risk_info(
        self, test_client, valid_transaction, ml_hold_response
    ):
        """Alert stores the complete risk result from the ML service."""
        tc, store = test_client
        mock_resp = Response(200, json=ml_hold_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)

        alert_id = resp.json()["alert"]["id"]
        stored = store.get_by_id(alert_id)
        assert stored["fraud_probability"] == 0.91
        assert stored["model_version"] == "fraud-xgb-v1.0.0"
        assert stored["risk_factors"] == [
            "amount_deviation", "spending_amount_anomaly", "high_amount",
        ]
        # Explanation is preserved
        assert stored["explanation_json"] is not None
        assert stored["explanation_json"]["ml_top_factors"][0]["feature"] == "amount_deviation"
        assert stored["explanation_json"]["behaviour_signals"][0]["signal"] == "spending_amount_anomaly"
        assert stored["explanation_json"]["rules_triggered"][0]["rule"] == "high_amount"

    def test_alert_contains_transaction_info(
        self, test_client, valid_transaction, ml_hold_response
    ):
        """Alert stores a transaction summary for analyst context."""
        tc, store = test_client
        mock_resp = Response(200, json=ml_hold_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)

        alert_id = resp.json()["alert"]["id"]
        stored = store.get_by_id(alert_id)
        assert stored["amount"] == 15000.00
        assert stored["currency"] == "USD"
        assert stored["merchant_name"] == "Offshore Trading Ltd"
        assert stored["transaction_type"] == "transfer"

    def test_no_second_ml_call_for_alert(self, test_client, valid_transaction, ml_hold_response):
        """Alert creation must not trigger a second ML service call."""
        tc, store = test_client
        mock_resp = Response(200, json=ml_hold_response)
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            resp = tc.post("/api/v1/transactions", json=valid_transaction)

        assert resp.status_code == 201
        assert mock_post.await_count == 1

    def test_duplicate_alert_protection(self, test_client, valid_transaction, ml_hold_response):
        """Re-processing the same transaction must not create a second alert."""
        tc, store = test_client
        mock_resp = Response(200, json=ml_hold_response)

        txn_id_holder = {}

        # Simulate: same transaction_id passed to _maybe_create_alert twice
        from backend.routers.transactions import _maybe_create_alert
        from backend.schemas import TransactionCreate, MLPredictionResponse

        request = TransactionCreate(**valid_transaction)
        ml_result = MLPredictionResponse.model_validate(ml_hold_response)

        txn_id = str(uuid.uuid4())
        txn_id_holder["id"] = txn_id

        summary1 = _maybe_create_alert(
            ml_result=ml_result,
            transaction_id=txn_id,
            request=request,
            explanation=ml_result.explanation_detail,
        )
        summary2 = _maybe_create_alert(
            ml_result=ml_result,
            transaction_id=txn_id,
            request=request,
            explanation=ml_result.explanation_detail,
        )

        assert summary1 is not None
        assert summary2 is not None
        assert summary1.id == summary2.id  # same alert returned, not duplicated
        _, total = store.list_alerts()
        assert total == 1

    def test_multiple_distinct_transactions_create_distinct_alerts(
        self, test_client, valid_transaction, ml_hold_response
    ):
        """Different transactions create different alerts."""
        tc, store = test_client
        mock_resp = Response(200, json=ml_hold_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            tc.post("/api/v1/transactions", json=valid_transaction)
            tc.post("/api/v1/transactions", json=valid_transaction)

        _, total = store.list_alerts()
        assert total == 2


# ── 2. GET /api/v1/alerts ────────────────────────────────────────────


class TestGetAlerts:
    """List endpoint returns analyst-friendly alerts."""

    def _create_alert(self, store, risk_score=85, risk_level="HIGH"):
        return store.create(
            transaction_id=str(uuid.uuid4()),
            risk_score=risk_score,
            risk_level=risk_level,
            decision="HOLD",
            fraud_probability=0.9,
            model_version="fraud-xgb-v1.0.0",
            risk_factors=["high_amount"],
            explanation_json={"ml_top_factors": []},
            amount=100.0,
            currency="USD",
            merchant_name="M",
            transaction_type="purchase",
        )

    def test_list_alerts_empty(self, test_client):
        tc, store = test_client
        resp = tc.get("/api/v1/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["per_page"] == 20

    def test_list_alerts_returns_created_alert(self, test_client, valid_transaction, ml_hold_response):
        tc, store = test_client
        mock_resp = Response(200, json=ml_hold_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        alert_id = resp.json()["alert"]["id"]

        list_resp = tc.get("/api/v1/alerts")
        assert list_resp.status_code == 200
        data = list_resp.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == alert_id
        assert data["items"][0]["status"] == "OPEN"
        assert data["items"][0]["risk_score"] == 85
        assert data["items"][0]["risk_level"] == "HIGH"
        assert data["items"][0]["decision"] == "HOLD"

    def test_list_alerts_filter_by_status(self, test_client):
        tc, store = test_client
        alert1 = self._create_alert(store)
        alert2 = self._create_alert(store)
        store.update_status(alert2["id"], new_status="IN_REVIEW")

        resp = tc.get("/api/v1/alerts", params={"status": "OPEN"})
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == alert1["id"]

        resp = tc.get("/api/v1/alerts", params={"status": "IN_REVIEW"})
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == alert2["id"]

    def test_list_alerts_filter_by_risk_level(self, test_client):
        tc, store = test_client
        self._create_alert(store, risk_level="HIGH")
        self._create_alert(store, risk_level="MEDIUM")

        resp = tc.get("/api/v1/alerts", params={"risk_level": "HIGH"})
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["risk_level"] == "HIGH"

    def test_list_alerts_invalid_status_filter(self, test_client):
        tc, store = test_client
        resp = tc.get("/api/v1/alerts", params={"status": "BOGUS"})
        assert resp.status_code == 422

    def test_list_alerts_invalid_risk_level_filter(self, test_client):
        tc, store = test_client
        resp = tc.get("/api/v1/alerts", params={"risk_level": "EXTREME"})
        assert resp.status_code == 422

    def test_list_alerts_pagination(self, test_client):
        tc, store = test_client
        for _ in range(5):
            self._create_alert(store)

        resp = tc.get("/api/v1/alerts", params={"page": 1, "per_page": 2})
        data = resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        assert data["page"] == 1
        assert data["per_page"] == 2

        resp = tc.get("/api/v1/alerts", params={"page": 3, "per_page": 2})
        data = resp.json()
        assert len(data["items"]) == 1  # 5 items, 2 per page → page 3 has 1

    def test_list_alerts_invalid_pagination(self, test_client):
        tc, store = test_client
        resp = tc.get("/api/v1/alerts", params={"page": 0})
        assert resp.status_code == 422
        resp = tc.get("/api/v1/alerts", params={"per_page": 101})
        assert resp.status_code == 422

    def test_get_alert_by_id(self, test_client, valid_transaction, ml_hold_response):
        tc, store = test_client
        mock_resp = Response(200, json=ml_hold_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        alert_id = resp.json()["alert"]["id"]

        detail = tc.get(f"/api/v1/alerts/{alert_id}")
        assert detail.status_code == 200
        data = detail.json()
        assert data["id"] == alert_id
        assert data["decision"] == "HOLD"
        assert data["fraud_probability"] == 0.91
        assert data["model_version"] == "fraud-xgb-v1.0.0"
        assert data["risk_factors"] == ["amount_deviation", "spending_amount_anomaly", "high_amount"]
        assert data["explanation"]["rules_triggered"][0]["rule"] == "high_amount"
        assert data["transaction_summary"]["amount"] == 15000.00
        assert data["transaction_summary"]["timestamp"] == 1725200000

    def test_get_alert_not_found(self, test_client):
        tc, store = test_client
        resp = tc.get(f"/api/v1/alerts/{uuid.uuid4()}")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ── 3. PATCH /api/v1/alerts/{id} — transitions ───────────────────────


class TestAlertStatusTransitions:
    """Valid and invalid status transitions."""

    def _create_and_get_id(self, test_client, valid_transaction, ml_hold_response):
        tc, store = test_client
        mock_resp = Response(200, json=ml_hold_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        return tc, resp.json()["alert"]["id"]

    def test_open_to_in_review(self, test_client, valid_transaction, ml_hold_response):
        tc, alert_id = self._create_and_get_id(test_client, valid_transaction, ml_hold_response)
        resp = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "IN_REVIEW"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "IN_REVIEW"
        assert resp.json()["resolved_at"] is None

    def test_open_to_resolved_directly(self, test_client, valid_transaction, ml_hold_response):
        """API contract permits OPEN → RESOLVED."""
        tc, alert_id = self._create_and_get_id(test_client, valid_transaction, ml_hold_response)
        resp = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "RESOLVED"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "RESOLVED"
        assert resp.json()["resolved_at"] is not None

    def test_open_to_dismissed_directly(self, test_client, valid_transaction, ml_hold_response):
        """API contract permits OPEN → DISMISSED."""
        tc, alert_id = self._create_and_get_id(test_client, valid_transaction, ml_hold_response)
        resp = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "DISMISSED"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "DISMISSED"
        assert resp.json()["resolved_at"] is not None

    def test_full_lifecycle_open_in_review_resolved(
        self, test_client, valid_transaction, ml_hold_response
    ):
        """OPEN → IN_REVIEW → RESOLVED complete flow."""
        tc, alert_id = self._create_and_get_id(test_client, valid_transaction, ml_hold_response)

        r1 = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "IN_REVIEW"})
        assert r1.status_code == 200
        assert r1.json()["status"] == "IN_REVIEW"

        r2 = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "RESOLVED"})
        assert r2.status_code == 200
        assert r2.json()["status"] == "RESOLVED"
        assert r2.json()["resolved_at"] is not None

    def test_full_lifecycle_open_in_review_dismissed(
        self, test_client, valid_transaction, ml_hold_response
    ):
        """OPEN → IN_REVIEW → DISMISSED complete flow."""
        tc, alert_id = self._create_and_get_id(test_client, valid_transaction, ml_hold_response)

        r1 = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "IN_REVIEW"})
        assert r1.status_code == 200

        r2 = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "DISMISSED"})
        assert r2.status_code == 200
        assert r2.json()["status"] == "DISMISSED"
        assert r2.json()["resolved_at"] is not None

    def test_invalid_resolved_to_in_review(self, test_client, valid_transaction, ml_hold_response):
        """RESOLVED is terminal — no further transitions."""
        tc, alert_id = self._create_and_get_id(test_client, valid_transaction, ml_hold_response)
        tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "RESOLVED"})

        resp = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "IN_REVIEW"})
        assert resp.status_code == 400
        assert "invalid status transition" in resp.json()["detail"].lower()

    def test_invalid_dismissed_to_open(self, test_client, valid_transaction, ml_hold_response):
        """DISMISSED is terminal — cannot reopen."""
        tc, alert_id = self._create_and_get_id(test_client, valid_transaction, ml_hold_response)
        tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "DISMISSED"})

        resp = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "OPEN"})
        assert resp.status_code == 400

    def test_invalid_resolved_to_dismissed(self, test_client, valid_transaction, ml_hold_response):
        """RESOLVED cannot transition to DISMISSED."""
        tc, alert_id = self._create_and_get_id(test_client, valid_transaction, ml_hold_response)
        tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "RESOLVED"})

        resp = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "DISMISSED"})
        assert resp.status_code == 400

    def test_notes_only_update(self, test_client, valid_transaction, ml_hold_response):
        """PATCH with only notes updates notes without changing status."""
        tc, alert_id = self._create_and_get_id(test_client, valid_transaction, ml_hold_response)
        resp = tc.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"notes": "Customer confirmed legitimate travel."},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "OPEN"  # unchanged
        assert resp.json()["notes"] == "Customer confirmed legitimate travel."

    def test_status_and_notes_together(self, test_client, valid_transaction, ml_hold_response):
        tc, alert_id = self._create_and_get_id(test_client, valid_transaction, ml_hold_response)
        resp = tc.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"status": "IN_REVIEW", "notes": "Investigating."},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "IN_REVIEW"
        assert resp.json()["notes"] == "Investigating."

    def test_patch_alert_not_found(self, test_client):
        tc, store = test_client
        resp = tc.patch(f"/api/v1/alerts/{uuid.uuid4()}", json={"status": "IN_REVIEW"})
        assert resp.status_code == 404

    def test_patch_invalid_status_value(self, test_client, valid_transaction, ml_hold_response):
        tc, alert_id = self._create_and_get_id(test_client, valid_transaction, ml_hold_response)
        resp = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "ESCALATED"})
        assert resp.status_code == 422

    def test_patch_empty_body(self, test_client, valid_transaction, ml_hold_response):
        """Both fields missing → 422."""
        tc, alert_id = self._create_and_get_id(test_client, valid_transaction, ml_hold_response)
        resp = tc.patch(f"/api/v1/alerts/{alert_id}", json={})
        assert resp.status_code == 422

    def test_patch_malformed_body(self, test_client, valid_transaction, ml_hold_response):
        """Wrong type for status → 422."""
        tc, alert_id = self._create_and_get_id(test_client, valid_transaction, ml_hold_response)
        resp = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": 123})
        assert resp.status_code == 422

    def test_immutable_risk_fields_cannot_be_modified(
        self, test_client, valid_transaction, ml_hold_response
    ):
        """PATCH cannot change risk_score, risk_level, decision, etc."""
        tc, alert_id = self._create_and_get_id(test_client, valid_transaction, ml_hold_response)
        # Attempt to tamper with risk fields
        resp = tc.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"status": "IN_REVIEW", "risk_score": 1, "risk_level": "LOW",
                  "decision": "APPROVE", "fraud_probability": 0.01},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Only status changed; risk fields are immutable through this endpoint
        assert data["risk_score"] == 85
        assert data["risk_level"] == "HIGH"
        assert data["decision"] == "HOLD"
        assert data["fraud_probability"] == 0.91


# ── 4. Repository unit tests ─────────────────────────────────────────


class TestAlertRepository:
    """In-memory and SQLite repository behaviour."""

    def test_valid_transitions_table(self):
        assert is_valid_transition(OPEN, IN_REVIEW)
        assert is_valid_transition(OPEN, RESOLVED)
        assert is_valid_transition(OPEN, DISMISSED)
        assert is_valid_transition(IN_REVIEW, RESOLVED)
        assert is_valid_transition(IN_REVIEW, DISMISSED)

    def test_invalid_transitions_table(self):
        assert not is_valid_transition(RESOLVED, IN_REVIEW)
        assert not is_valid_transition(RESOLVED, OPEN)
        assert not is_valid_transition(DISMISSED, OPEN)
        assert not is_valid_transition(DISMISSED, IN_REVIEW)
        assert not is_valid_transition(RESOLVED, DISMISSED)
        assert not is_valid_transition(DISMISSED, RESOLVED)

    def test_valid_statuses(self):
        assert VALID_STATUSES == {"OPEN", "IN_REVIEW", "RESOLVED", "DISMISSED"}

    def test_inmemory_create_defaults(self):
        store = InMemoryAlertStore()
        alert = store.create(
            transaction_id="t1",
            risk_score=80,
            risk_level="HIGH",
            decision="HOLD",
        )
        assert alert["status"] == "OPEN"
        assert alert["analyst_id"] is None
        assert alert["notes"] is None
        assert alert["created_at"] is not None
        assert alert["updated_at"] is not None
        assert alert["resolved_at"] is None

    def test_inmemory_get_by_transaction_id(self):
        store = InMemoryAlertStore()
        store.create(transaction_id="t1", risk_score=80, risk_level="HIGH", decision="HOLD")
        store.create(transaction_id="t2", risk_score=90, risk_level="HIGH", decision="HOLD")
        found = store.get_by_transaction_id("t2")
        assert found is not None
        assert found["risk_score"] == 90
        assert store.get_by_transaction_id("missing") is None

    def test_inmemory_update_status_invalid_returns_none(self):
        store = InMemoryAlertStore()
        alert = store.create(transaction_id="t1", risk_score=80, risk_level="HIGH", decision="HOLD")
        store.update_status(alert["id"], new_status="RESOLVED")
        result = store.update_status(alert["id"], new_status="IN_REVIEW")
        assert result is None  # terminal → rejected

    def test_inmemory_update_status_not_found(self):
        store = InMemoryAlertStore()
        assert store.update_status("missing", new_status="IN_REVIEW") is None


# ── 5. Persistence across restart (SQLite) ───────────────────────────


class TestSQLitePersistence:
    """Alerts survive repository re-instantiation (restart simulation)."""

    def test_alerts_persist_across_restart(self, tmp_path):
        db = tmp_path / "alerts.db"
        repo1 = SQLiteAlertRepository(db_path=db)
        alert = repo1.create(
            transaction_id="t-restart",
            risk_score=88,
            risk_level="HIGH",
            decision="HOLD",
            fraud_probability=0.93,
            model_version="fraud-xgb-v1.0.0",
            risk_factors=["high_amount", "velocity_limit"],
            explanation_json={"ml_top_factors": [{"feature": "amount_deviation", "importance": 0.4}]},
            amount=12000.0,
            currency="EUR",
            merchant_name="Test Merchant",
            transaction_type="transfer",
        )
        alert_id = alert["id"]
        repo1.update_status(alert_id, new_status="IN_REVIEW")
        repo1.close()

        # Simulate restart: new repository instance on the same file
        repo2 = SQLiteAlertRepository(db_path=db)
        restored = repo2.get_by_id(alert_id)

        assert restored is not None
        assert restored["risk_score"] == 88
        assert restored["risk_level"] == "HIGH"
        assert restored["decision"] == "HOLD"
        assert restored["status"] == "IN_REVIEW"  # status persisted
        assert restored["fraud_probability"] == 0.93
        assert restored["model_version"] == "fraud-xgb-v1.0.0"
        assert restored["risk_factors"] == ["high_amount", "velocity_limit"]  # JSON restored
        assert restored["explanation_json"]["ml_top_factors"][0]["feature"] == "amount_deviation"
        assert restored["amount"] == 12000.0
        assert restored["currency"] == "EUR"
        repo2.close()

    def test_status_transitions_persist(self, tmp_path):
        """Full lifecycle persists: OPEN → IN_REVIEW → RESOLVED."""
        db = tmp_path / "alerts.db"
        repo1 = SQLiteAlertRepository(db_path=db)
        alert = repo1.create(
            transaction_id="t-lifecycle",
            risk_score=90,
            risk_level="HIGH",
            decision="HOLD",
        )
        repo1.update_status(alert["id"], new_status="IN_REVIEW")
        repo1.update_status(alert["id"], new_status="RESOLVED")
        repo1.close()

        repo2 = SQLiteAlertRepository(db_path=db)
        restored = repo2.get_by_id(alert["id"])
        assert restored["status"] == "RESOLVED"
        assert restored["resolved_at"] is not None
        repo2.close()

    def test_sqlite_list_with_filter(self, tmp_path):
        db = tmp_path / "alerts.db"
        repo = SQLiteAlertRepository(db_path=db)
        a1 = repo.create(transaction_id="t1", risk_score=80, risk_level="HIGH", decision="HOLD")
        a2 = repo.create(transaction_id="t2", risk_score=75, risk_level="MEDIUM", decision="HOLD")
        repo.update_status(a2["id"], new_status="IN_REVIEW")

        open_alerts, total_open = repo.list_alerts(status="OPEN")
        assert total_open == 1
        assert open_alerts[0]["id"] == a1["id"]

        high_alerts, total_high = repo.list_alerts(risk_level="HIGH")
        assert total_high == 1

        all_alerts, total_all = repo.list_alerts()
        assert total_all == 2
        repo.close()

    def test_sqlite_update_invalid_transition(self, tmp_path):
        db = tmp_path / "alerts.db"
        repo = SQLiteAlertRepository(db_path=db)
        alert = repo.create(
            transaction_id="t1", risk_score=80, risk_level="HIGH", decision="HOLD"
        )
        repo.update_status(alert["id"], new_status="RESOLVED")
        result = repo.update_status(alert["id"], new_status="IN_REVIEW")
        assert result is None  # terminal
        repo.close()

    def test_sqlite_duplicate_transaction_reference(self, tmp_path):
        """get_by_transaction_id finds existing alerts (duplicate check)."""
        db = tmp_path / "alerts.db"
        repo = SQLiteAlertRepository(db_path=db)
        alert = repo.create(
            transaction_id="t-dup", risk_score=80, risk_level="HIGH", decision="HOLD"
        )
        found = repo.get_by_transaction_id("t-dup")
        assert found is not None
        assert found["id"] == alert["id"]
        assert repo.get_by_transaction_id("no-such") is None
        repo.close()


# ── 6. Transaction behavior unchanged ────────────────────────────────


class TestTransactionBehaviorUnchanged:
    """The existing transaction endpoint behavior must remain intact."""

    def test_transaction_response_fields_unchanged(
        self, test_client, valid_transaction, ml_verify_response
    ):
        """VERIFY transactions keep returning the complete risk result."""
        tc, store = test_client
        mock_resp = Response(200, json=ml_verify_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)

        assert resp.status_code == 201
        data = resp.json()
        # All original fields are still present
        assert data["amount"] == 15000.00
        assert data["fraud_probability"] == 0.91
        assert data["fraud_prediction"] == 1
        assert data["ml_score"] == 91
        assert data["behaviour_score"] == 75
        assert data["rule_score"] == 60
        assert data["risk_score"] == 50
        assert data["risk_level"] == "MEDIUM"
        assert data["decision"] == "VERIFY"
        assert data["model_version"] == "fraud-xgb-v1.0.0"
        assert data["timestamp"] == 1725200000
        assert data["explanation"] is not None
        assert data["risk_factors"] is not None
        assert data["alert"] is None  # no alert for VERIFY

    def test_ml_unavailable_still_503(self, test_client, valid_transaction):
        """ML failure behavior unchanged — no alert, 503."""
        import httpx as _httpx
        tc, store = test_client
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_httpx.ConnectError("refused"),
        ):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 503
        _, total = store.list_alerts()
        assert total == 0

    def test_ml_timeout_still_503(self, test_client, valid_transaction):
        import httpx as _httpx
        tc, store = test_client
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_httpx.ReadTimeout("timed out"),
        ):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 503
        _, total = store.list_alerts()
        assert total == 0

    def test_ml_500_still_502(self, test_client, valid_transaction):
        tc, store = test_client
        mock_resp = Response(500, json={"detail": "Internal error"})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 502
        _, total = store.list_alerts()
        assert total == 0

    def test_invalid_transaction_still_422(self, test_client):
        tc, store = test_client
        resp = tc.post("/api/v1/transactions", json={"amount": -1})
        assert resp.status_code == 422
        _, total = store.list_alerts()
        assert total == 0

    def test_alert_creation_failure_does_not_block_transaction(
        self, test_client, valid_transaction, ml_hold_response
    ):
        """A failing alert store must not break the transaction response."""
        from fastapi.testclient import TestClient
        from backend.app import app
        from backend.routers import alerts as alerts_module
        from backend.routers import transactions as txn_module

        tc, _ = test_client

        class FailingStore:
            def create(self, **kwargs):
                raise RuntimeError("disk full")

            def get_by_transaction_id(self, transaction_id):
                return None

        original = txn_module._alert_repo
        try:
            txn_module.set_alert_repository(FailingStore())
            mock_resp = Response(200, json=ml_hold_response)
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
                resp = tc.post("/api/v1/transactions", json=valid_transaction)
            # Transaction still succeeds; alert is simply not created
            assert resp.status_code == 201
            assert resp.json()["alert"] is None
        finally:
            txn_module.set_alert_repository(original)


# ── 7. Security and leakage ──────────────────────────────────────────


class TestSecurityAndLeakage:
    """No sensitive data or internal details are exposed."""

    def test_no_ml_urls_in_error_responses(self, test_client, valid_transaction):
        import httpx as _httpx
        tc, store = test_client
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=_httpx.ConnectError("refused"),
        ):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        body = resp.json()["detail"]
        assert "http" not in body.lower()
        assert "8001" not in body

    def test_no_stack_traces_in_error_responses(self, test_client, valid_transaction, ml_hold_response):
        tc, store = test_client
        resp = tc.patch(
            f"/api/v1/alerts/{uuid.uuid4()}",
            json={"status": "IN_REVIEW"},
        )
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert "Traceback" not in detail
        assert ".py" not in detail

    def test_no_secrets_in_alert_response(
        self, test_client, valid_transaction, ml_hold_response
    ):
        tc, store = test_client
        mock_resp = Response(200, json=ml_hold_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        alert_id = resp.json()["alert"]["id"]

        detail = tc.get(f"/api/v1/alerts/{alert_id}")
        data = detail.json()
        for key in ("password", "secret", "token", "api_key", "credentials"):
            assert key not in str(data).lower()

    def test_backend_does_not_import_ml_modules(self):
        """No backend module imports the ML implementation."""
        import backend.routers.transactions as txn_mod
        import backend.routers.alerts as alerts_mod
        import backend.app as app_mod
        import inspect

        for module in (txn_mod, alerts_mod, app_mod):
            source = inspect.getsource(module)
            assert "from ml." not in source
            assert "import ml." not in source

    def test_alert_endpoints_do_not_accept_direct_creation(self, test_client):
        """POST to /api/v1/alerts is not a valid route (405)."""
        tc, store = test_client
        resp = tc.post("/api/v1/alerts", json={"risk_score": 99, "decision": "HOLD"})
        assert resp.status_code == 405

    def test_patch_cannot_change_transaction_reference(
        self, test_client, valid_transaction, ml_hold_response
    ):
        """transaction_id is immutable through the PATCH endpoint."""
        tc, alert_id = None, None
        tc, store = test_client
        mock_resp = Response(200, json=ml_hold_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        alert_id = resp.json()["alert"]["id"]
        original_txn = resp.json()["alert"]["id"]

        resp = tc.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"status": "IN_REVIEW", "transaction_id": "fake-txn"},
        )
        data = resp.json()
        # transaction_id in the response comes from the store, not the request
        assert data["transaction_id"] != "fake-txn"


# ── 8. End-to-end mocked flow ────────────────────────────────────────


class TestEndToEndMockedFlow:
    """Full alert lifecycle through the API with a mocked ML service."""

    def test_full_alert_lifecycle(self, test_client, valid_transaction, ml_hold_response):
        """POST txn → OPEN alert → list → IN_REVIEW → RESOLVED → verify."""
        tc, store = test_client

        # 1. Submit HOLD transaction
        mock_resp = Response(200, json=ml_hold_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        alert_summary = resp.json()["alert"]
        assert alert_summary["status"] == "OPEN"

        # 2. List alerts — alert is there
        list_resp = tc.get("/api/v1/alerts")
        assert list_resp.status_code == 200
        assert list_resp.json()["total"] == 1
        assert list_resp.json()["items"][0]["id"] == alert_summary["id"]

        # 3. Get alert detail
        detail = tc.get(f"/api/v1/alerts/{alert_summary['id']}")
        assert detail.status_code == 200
        assert detail.json()["status"] == "OPEN"
        assert detail.json()["risk_score"] == 85

        # 4. Transition OPEN → IN_REVIEW
        r1 = tc.patch(f"/api/v1/alerts/{alert_summary['id']}", json={"status": "IN_REVIEW"})
        assert r1.status_code == 200
        assert r1.json()["status"] == "IN_REVIEW"

        # 5. Transition IN_REVIEW → RESOLVED
        r2 = tc.patch(f"/api/v1/alerts/{alert_summary['id']}", json={"status": "RESOLVED"})
        assert r2.status_code == 200
        assert r2.json()["status"] == "RESOLVED"
        assert r2.json()["resolved_at"] is not None

        # 6. Verify final persisted state
        final = tc.get(f"/api/v1/alerts/{alert_summary['id']}")
        assert final.status_code == 200
        assert final.json()["status"] == "RESOLVED"
        assert final.json()["resolved_at"] is not None

        # 7. Further transition rejected
        r3 = tc.patch(f"/api/v1/alerts/{alert_summary['id']}", json={"status": "IN_REVIEW"})
        assert r3.status_code == 400

    def test_full_alert_lifecycle_dismissed(self, test_client, valid_transaction, ml_hold_response):
        """POST txn → OPEN → IN_REVIEW → DISMISSED."""
        tc, store = test_client
        mock_resp = Response(200, json=ml_hold_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        alert_id = resp.json()["alert"]["id"]

        r1 = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "IN_REVIEW"})
        assert r1.status_code == 200
        r2 = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "DISMISSED"})
        assert r2.status_code == 200
        assert r2.json()["status"] == "DISMISSED"
        assert r2.json()["resolved_at"] is not None


# ── 9. Live E2E (requires running services) ──────────────────────────


class TestLiveEndToEnd:
    """Real E2E with a live ML service on localhost:8001.

    Automatically skipped if the ML service is not running.
    """

    @pytest.fixture(autouse=True)
    def _check_ml_service(self):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            sock.connect(("localhost", 8001))
            sock.close()
        except (ConnectionRefusedError, OSError):
            pytest.skip("ML service not running on localhost:8001")

    def test_live_full_lifecycle(self, tmp_path, auth_override):
        """Live: POST → HOLD → OPEN alert → IN_REVIEW → RESOLVED."""
        from fastapi.testclient import TestClient
        from backend.app import app
        from backend.config import get_settings
        from backend.db.alert_repository import SQLiteAlertRepository
        from backend.routers import alerts as alerts_module
        from backend.routers import transactions as txn_module

        settings = get_settings()
        ml_client = MLServiceClient(
            base_url=settings.ML_SERVICE_URL,
            timeout=float(settings.ML_REQUEST_TIMEOUT_SECONDS),
        )
        repo = SQLiteAlertRepository(db_path=tmp_path / "live_alerts.db")
        txn_module.set_ml_client(ml_client)
        txn_module.set_alert_repository(repo)
        alerts_module.set_alert_repository(repo)
        tc = TestClient(app)

        try:
            # HOLD-triggering profile: new device + gambling merchant (7995)
            # + purchase + US + mobile produces risk_score > 70 with the
            # trained model (verified against fraud-xgb-v1.0.0).
            transaction = {
                "amount": 10500.00,
                "currency": "USD",
                "merchant_name": "Wire Casino",
                "merchant_category": "7995",
                "transaction_type": "purchase",
                "location_country": "US",
                "location_city": "Miami",
                "device_fingerprint": f"live-e2e-{uuid.uuid4()}",
                "device_type": "mobile",
                "ip_address": "198.51.100.7",
            }
            resp = tc.post("/api/v1/transactions", json=transaction)
            assert resp.status_code == 201
            data = resp.json()
            assert data["decision"] in ("HOLD", "VERIFY", "APPROVE")

            if data["decision"] == "HOLD":
                # Alert must exist
                assert data["alert"] is not None
                assert data["alert"]["status"] == "OPEN"
                alert_id = data["alert"]["id"]

                # GET alerts
                lst = tc.get("/api/v1/alerts")
                assert lst.status_code == 200
                assert lst.json()["total"] >= 1

                # GET alert detail — timestamp preserved from ML result
                detail = tc.get(f"/api/v1/alerts/{alert_id}")
                assert detail.status_code == 200
                assert detail.json()["transaction_summary"]["timestamp"] == data["timestamp"]

                # PATCH transitions
                r1 = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "IN_REVIEW"})
                assert r1.status_code == 200
                r2 = tc.patch(f"/api/v1/alerts/{alert_id}", json={"status": "RESOLVED"})
                assert r2.status_code == 200
                assert r2.json()["resolved_at"] is not None
        finally:
            repo.close()


class TestAlertTransactionSummaryLinking:
    """Verifies TransactionSummary population from linked transactions."""

    def test_alert_with_linked_transaction_populates_summary(self, auth_override):
        from fastapi.testclient import TestClient
        from backend.app import app
        from backend.db.alert_repository import InMemoryAlertStore
        from backend.db.transaction_repository import InMemoryTransactionStore
        from backend.routers import alerts as alerts_module

        alert_repo = InMemoryAlertStore()
        txn_repo = InMemoryTransactionStore()

        alerts_module.set_alert_repository(alert_repo)
        alerts_module.set_transaction_repository(txn_repo)

        tx_id = str(uuid.uuid4())
        txn_repo.create(
            transaction_id=tx_id,
            customer_id=str(uuid.uuid4()),
            amount=450.75,
            currency="EUR",
            merchant_name="SuperMart Berlin",
            transaction_type="payment",
            decision="HOLD",
            status="COMPLETED",
            risk_score=85,
            risk_level="HIGH",
        )

        alert_id = str(uuid.uuid4())
        # Alert stored without amount/merchant_name (11-column PostgreSQL style)
        alert_repo.create(
            id=alert_id,
            transaction_id=tx_id,
            customer_id=str(uuid.uuid4()),
            risk_score=85,
            risk_level="HIGH",
            decision="HOLD",
        )

        tc = TestClient(app)
        resp = tc.get(f"/api/v1/alerts/{alert_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["transaction_summary"] is not None
        assert data["transaction_summary"]["amount"] == 450.75
        assert data["transaction_summary"]["currency"] == "EUR"
        assert data["transaction_summary"]["merchant_name"] == "SuperMart Berlin"
        assert data["transaction_summary"]["transaction_type"] == "payment"

    def test_legacy_alert_without_linked_transaction_remains_safe(self, auth_override):
        from fastapi.testclient import TestClient
        from backend.app import app
        from backend.db.alert_repository import InMemoryAlertStore
        from backend.db.transaction_repository import InMemoryTransactionStore
        from backend.routers import alerts as alerts_module

        alert_repo = InMemoryAlertStore()
        txn_repo = InMemoryTransactionStore()

        alerts_module.set_alert_repository(alert_repo)
        alerts_module.set_transaction_repository(txn_repo)

        alert_id = str(uuid.uuid4())
        # Alert with non-existent transaction_id and no amount
        alert_repo.create(
            id=alert_id,
            transaction_id=str(uuid.uuid4()),  # Not in txn_repo
            customer_id=str(uuid.uuid4()),
            risk_score=75,
            risk_level="HIGH",
            decision="HOLD",
        )

        tc = TestClient(app)
        resp = tc.get(f"/api/v1/alerts/{alert_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["transaction_summary"] is None

    def test_legacy_alert_with_direct_amount_preserves_behavior(self, auth_override):
        from fastapi.testclient import TestClient
        from backend.app import app
        from backend.db.alert_repository import InMemoryAlertStore
        from backend.routers import alerts as alerts_module

        alert_repo = InMemoryAlertStore()
        alerts_module.set_alert_repository(alert_repo)
        alerts_module.set_transaction_repository(None)

        alert_id = str(uuid.uuid4())
        alert_repo.create(
            id=alert_id,
            transaction_id=str(uuid.uuid4()),
            customer_id=str(uuid.uuid4()),
            risk_score=90,
            risk_level="HIGH",
            decision="HOLD",
            amount=999.99,
            currency="USD",
            merchant_name="Legacy Merchant",
            transaction_type="transfer",
        )

        tc = TestClient(app)
        resp = tc.get(f"/api/v1/alerts/{alert_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["transaction_summary"] is not None
        assert data["transaction_summary"]["amount"] == 999.99
        assert data["transaction_summary"]["currency"] == "USD"
        assert data["transaction_summary"]["merchant_name"] == "Legacy Merchant"
        assert data["transaction_summary"]["transaction_type"] == "transfer"
