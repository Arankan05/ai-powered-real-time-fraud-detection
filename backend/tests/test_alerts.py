"""Tests for alert endpoints and audit logging.

Covers:
* Alert listing with filters and pagination
* Alert detail retrieval
* Alert status update (PATCH) with state transitions
* RBAC enforcement (analyst/admin vs customer)
* Audit log creation for transactions and alerts
* Transaction/alert consistency
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.config import settings
from app.core.security import create_access_token, hash_password
from app.db.session import SessionLocal
from app.models.alert import Alert
from app.models.audit_log import AuditLog
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction
from app.models.user import User
from app.services.ml import MLServiceClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pg_available() -> bool:
    try:
        eng = create_engine(settings.postgres.database_url)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


requires_pg = pytest.mark.skipif(not _pg_available(), reason="PostgreSQL is not running")


def _unique_email() -> str:
    return f"s6-{uuid.uuid4().hex[:12]}@test.com"


# Standard ML response fixtures
_ML_HOLD = {
    "ml_score": 80, "behaviour_score": 90, "rule_score": 40,
    "risk_score": 82, "risk_level": "HIGH", "decision": "HOLD",
    "explanation": {
        "ml_top_factors": [
            {"feature": "amount_deviation", "importance": 0.45},
        ],
        "behaviour_signals": [
            {"signal": "spending_amount_anomaly", "severity": 0.9},
        ],
        "rules_triggered": [
            {"rule": "high_amount", "contribution": 15},
        ],
    },
    "risk_factors": ["amount_deviation", "spending_amount_anomaly", "high_amount"],
    "model_version": None,
}

_ML_APPROVE = {
    "ml_score": 10, "behaviour_score": 15, "rule_score": 5,
    "risk_score": 12, "risk_level": "LOW", "decision": "APPROVE",
    "explanation": {
        "ml_top_factors": [{"feature": "amount", "importance": 0.1}],
        "behaviour_signals": [{"signal": "normal_spending", "severity": 0.1}],
        "rules_triggered": [],
    },
    "risk_factors": ["amount"],
    "model_version": None,
}


def _make_ml_mock(response: dict) -> MagicMock:
    mock = MagicMock(spec=MLServiceClient)
    mock.predict.return_value = response
    return mock


_TXN_REQUEST = {
    "amount": 500.00,
    "currency": "USD",
    "merchant_name": "Test Merchant S6",
    "merchant_category": "5732",
    "transaction_type": "purchase",
    "location_country": "US",
    "location_city": "New York",
    "device_fingerprint": "s6-test-fp",
    "device_type": "mobile",
    "ip_address": "192.168.1.200",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cleanup_s6_data(db: Session) -> None:
    """Remove test data created by Step 6 tests."""
    users = db.query(User).filter(User.email.like("s6-%@test.com")).all()
    customer_ids = [u.customer_id for u in users if u.customer_id]

    if customer_ids:
        # Delete alerts
        txns = db.query(Transaction).filter(Transaction.customer_id.in_(customer_ids)).all()
        for txn in txns:
            alerts = db.query(Alert).filter(Alert.transaction_id == txn.id).all()
            for a in alerts:
                db.delete(a)
            db.delete(txn)

    # Delete audit logs for test users
    user_ids = [u.id for u in users]
    if user_ids:
        audit_logs = db.query(AuditLog).filter(AuditLog.actor_id.in_(user_ids)).all()
        for al in audit_logs:
            db.delete(al)

    for user in users:
        db.delete(user)
    db.commit()

    if customer_ids:
        customers = db.query(Customer).filter(Customer.id.in_(customer_ids)).all()
        for c in customers:
            db.delete(c)
        db.commit()

    merchants = db.query(Merchant).filter(Merchant.name.like("Test Merchant S6%")).all()
    for m in merchants:
        db.delete(m)
    db.commit()


@pytest.fixture()
def s6_db() -> Generator[Session, None, None]:
    """DB session with cleanup for Step 6 tests."""
    db = SessionLocal()
    try:
        _cleanup_s6_data(db)
        yield db
    finally:
        _cleanup_s6_data(db)
        db.close()


def _register_and_login(
    client: TestClient, *, first_name: str = "Test", last_name: str = "User",
) -> dict:
    email = _unique_email()
    reg = client.post("/api/v1/auth/register", json={
        "email": email, "password": "SecurePass1",
        "first_name": first_name, "last_name": last_name,
    })
    assert reg.status_code == 201
    login = client.post("/api/v1/auth/login", json={
        "email": email, "password": "SecurePass1",
    })
    assert login.status_code == 200
    return {**login.json(), "email": email, "user_id": reg.json()["id"],
            "customer_id": reg.json()["customer_id"]}


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _create_analyst_user(db: Session, role: str = "fraud_analyst") -> tuple[User, str]:
    user = User(
        email=_unique_email(),
        password_hash=hash_password("SecurePass1"),
        first_name="Test", last_name="Analyst",
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.id, user.role)
    return user, token


def _create_hold_transaction(
    client: TestClient, token: str,
) -> dict:
    """Create a transaction that results in a HOLD decision."""
    mock_client = _make_ml_mock(_ML_HOLD)
    with patch("app.services.transaction.MLServiceClient") as MockCls:
        MockCls.return_value = mock_client
        resp = client.post(
            "/api/v1/transactions",
            json=_TXN_REQUEST,
            headers=_auth_header(token),
        )
    assert resp.status_code == 201
    return resp.json()


def _create_approve_transaction(
    client: TestClient, token: str,
) -> dict:
    """Create a transaction that results in an APPROVE decision."""
    mock_client = _make_ml_mock(_ML_APPROVE)
    with patch("app.services.transaction.MLServiceClient") as MockCls:
        MockCls.return_value = mock_client
        resp = client.post(
            "/api/v1/transactions",
            json=_TXN_REQUEST,
            headers=_auth_header(token),
        )
    assert resp.status_code == 201
    return resp.json()


# ===================================================================
# 1. ALERT LISTING
# ===================================================================


@requires_pg
class TestAlertList:
    """GET /api/v1/alerts"""

    @patch("app.services.transaction.MLServiceClient")
    def test_analyst_list_alerts(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Authorized analyst can list alerts."""
        info = _register_and_login(client)
        _create_hold_transaction(client, info["access_token"])

        analyst, token = _create_analyst_user(s6_db)
        resp = client.get("/api/v1/alerts", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "per_page" in data
        assert data["total"] >= 1

    @patch("app.services.transaction.MLServiceClient")
    def test_admin_list_alerts(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Authorized admin can list alerts."""
        info = _register_and_login(client)
        _create_hold_transaction(client, info["access_token"])

        admin, token = _create_analyst_user(s6_db, role="admin")
        resp = client.get("/api/v1/alerts", headers=_auth_header(token))
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    def test_customer_forbidden_403(self, client: TestClient, s6_db: Session) -> None:
        """Customer cannot access alert list."""
        info = _register_and_login(client)
        resp = client.get("/api/v1/alerts", headers=_auth_header(info["access_token"]))
        assert resp.status_code == 403

    def test_no_auth_401(self, client: TestClient) -> None:
        resp = client.get("/api/v1/alerts")
        assert resp.status_code == 401

    @patch("app.services.transaction.MLServiceClient")
    def test_alert_list_item_shape(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Alert list items have the documented shape."""
        info = _register_and_login(client)
        _create_hold_transaction(client, info["access_token"])

        analyst, token = _create_analyst_user(s6_db)
        resp = client.get("/api/v1/alerts", headers=_auth_header(token))
        data = resp.json()
        assert data["items"], "Expected at least one alert"
        item = data["items"][0]
        expected_keys = {
            "id", "transaction_id", "risk_score", "risk_level", "decision",
            "status", "analyst_id", "notes", "created_at", "resolved_at",
            "transaction_summary",
        }
        assert set(item.keys()) == expected_keys

    @patch("app.services.transaction.MLServiceClient")
    def test_alert_list_transaction_summary(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Transaction summary includes amount, currency, merchant_name, etc."""
        info = _register_and_login(client)
        _create_hold_transaction(client, info["access_token"])

        analyst, token = _create_analyst_user(s6_db)
        resp = client.get("/api/v1/alerts", headers=_auth_header(token))
        item = resp.json()["items"][0]
        summary = item["transaction_summary"]
        assert "amount" in summary
        assert "currency" in summary
        assert "merchant_name" in summary
        assert "transaction_type" in summary
        assert "customer_email" in summary
        assert "timestamp" in summary

    @patch("app.services.transaction.MLServiceClient")
    def test_alert_list_filter_status(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Filter by status works."""
        info = _register_and_login(client)
        _create_hold_transaction(client, info["access_token"])

        analyst, token = _create_analyst_user(s6_db)
        resp = client.get("/api/v1/alerts?status=OPEN", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        for item in data["items"]:
            assert item["status"] == "OPEN"

    @patch("app.services.transaction.MLServiceClient")
    def test_alert_list_filter_risk_level(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Filter by risk_level works."""
        info = _register_and_login(client)
        _create_hold_transaction(client, info["access_token"])

        analyst, token = _create_analyst_user(s6_db)
        resp = client.get("/api/v1/alerts?risk_level=HIGH", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        for item in data["items"]:
            assert item["risk_level"] == "HIGH"

    def test_invalid_status_filter_422(self, client: TestClient, s6_db: Session) -> None:
        """Invalid status query param returns 422."""
        analyst, token = _create_analyst_user(s6_db)
        resp = client.get("/api/v1/alerts?status=INVALID", headers=_auth_header(token))
        assert resp.status_code == 422


# ===================================================================
# 2. ALERT DETAIL
# ===================================================================


@requires_pg
class TestAlertDetail:
    """GET /api/v1/alerts/{id}"""

    @patch("app.services.transaction.MLServiceClient")
    def test_analyst_get_alert_detail(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Authorized analyst can get alert detail."""
        info = _register_and_login(client)
        txn_data = _create_hold_transaction(client, info["access_token"])
        alert_id = txn_data["alert"]["id"]

        analyst, token = _create_analyst_user(s6_db)
        resp = client.get(f"/api/v1/alerts/{alert_id}", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == alert_id
        assert data["status"] == "OPEN"
        assert data["risk_level"] == "HIGH"
        assert data["decision"] == "HOLD"

    @patch("app.services.transaction.MLServiceClient")
    def test_alert_detail_has_explanation(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Alert detail includes explanation and risk_factors."""
        info = _register_and_login(client)
        txn_data = _create_hold_transaction(client, info["access_token"])
        alert_id = txn_data["alert"]["id"]

        analyst, token = _create_analyst_user(s6_db)
        resp = client.get(f"/api/v1/alerts/{alert_id}", headers=_auth_header(token))
        data = resp.json()
        assert "explanation" in data
        assert data["explanation"] is not None
        assert "ml_top_factors" in data["explanation"]
        assert "behaviour_signals" in data["explanation"]
        assert "rules_triggered" in data["explanation"]
        assert "risk_factors" in data
        assert len(data["risk_factors"]) > 0

    @patch("app.services.transaction.MLServiceClient")
    def test_alert_detail_has_transaction(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Alert detail includes full transaction detail."""
        info = _register_and_login(client)
        txn_data = _create_hold_transaction(client, info["access_token"])
        alert_id = txn_data["alert"]["id"]

        analyst, token = _create_analyst_user(s6_db)
        resp = client.get(f"/api/v1/alerts/{alert_id}", headers=_auth_header(token))
        data = resp.json()
        assert "transaction" in data
        txn = data["transaction"]
        assert "id" in txn
        assert "customer_id" in txn
        assert "amount" in txn
        assert "ml_score" in txn
        assert "behaviour_score" in txn
        assert "rule_score" in txn

    def test_alert_not_found_404(self, client: TestClient, s6_db: Session) -> None:
        analyst, token = _create_analyst_user(s6_db)
        resp = client.get(f"/api/v1/alerts/{uuid.uuid4()}", headers=_auth_header(token))
        assert resp.status_code == 404

    def test_customer_forbidden_403(self, client: TestClient, s6_db: Session) -> None:
        info = _register_and_login(client)
        resp = client.get(f"/api/v1/alerts/{uuid.uuid4()}", headers=_auth_header(info["access_token"]))
        assert resp.status_code == 403


# ===================================================================
# 3. ALERT UPDATE (PATCH)
# ===================================================================


@requires_pg
class TestAlertUpdate:
    """PATCH /api/v1/alerts/{id}"""

    @patch("app.services.transaction.MLServiceClient")
    def test_update_status_to_in_review(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Analyst can transition OPEN → IN_REVIEW."""
        info = _register_and_login(client)
        txn_data = _create_hold_transaction(client, info["access_token"])
        alert_id = txn_data["alert"]["id"]

        analyst, token = _create_analyst_user(s6_db)
        resp = client.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"status": "IN_REVIEW"},
            headers=_auth_header(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "IN_REVIEW"
        assert data["analyst_id"] is not None  # Auto-assigned
        assert data["resolved_at"] is None  # Not terminal

    @patch("app.services.transaction.MLServiceClient")
    def test_update_status_to_resolved(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """OPEN → RESOLVED sets resolved_at."""
        info = _register_and_login(client)
        txn_data = _create_hold_transaction(client, info["access_token"])
        alert_id = txn_data["alert"]["id"]

        analyst, token = _create_analyst_user(s6_db)
        resp = client.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"status": "RESOLVED"},
            headers=_auth_header(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "RESOLVED"
        assert data["resolved_at"] is not None

    @patch("app.services.transaction.MLServiceClient")
    def test_update_status_to_dismissed(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """OPEN → DISMISSED sets resolved_at."""
        info = _register_and_login(client)
        txn_data = _create_hold_transaction(client, info["access_token"])
        alert_id = txn_data["alert"]["id"]

        analyst, token = _create_analyst_user(s6_db)
        resp = client.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"status": "DISMISSED"},
            headers=_auth_header(token),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "DISMISSED"
        assert resp.json()["resolved_at"] is not None

    @patch("app.services.transaction.MLServiceClient")
    def test_update_notes_only(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Can update notes without changing status."""
        info = _register_and_login(client)
        txn_data = _create_hold_transaction(client, info["access_token"])
        alert_id = txn_data["alert"]["id"]

        analyst, token = _create_analyst_user(s6_db)
        resp = client.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"notes": "Investigation started"},
            headers=_auth_header(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["notes"] == "Investigation started"
        assert data["status"] == "OPEN"  # Status unchanged

    @patch("app.services.transaction.MLServiceClient")
    def test_invalid_transition_rejected_400(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """RESOLVED → OPEN is not allowed."""
        info = _register_and_login(client)
        txn_data = _create_hold_transaction(client, info["access_token"])
        alert_id = txn_data["alert"]["id"]

        analyst, token = _create_analyst_user(s6_db)
        # First resolve
        client.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"status": "RESOLVED"},
            headers=_auth_header(token),
        )
        # Then try to reopen
        resp = client.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"status": "OPEN"},
            headers=_auth_header(token),
        )
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "INVALID_STATUS_TRANSITION"

    @patch("app.services.transaction.MLServiceClient")
    def test_in_review_to_resolved(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """IN_REVIEW → RESOLVED is allowed."""
        info = _register_and_login(client)
        txn_data = _create_hold_transaction(client, info["access_token"])
        alert_id = txn_data["alert"]["id"]

        analyst, token = _create_analyst_user(s6_db)
        # First IN_REVIEW
        client.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"status": "IN_REVIEW"},
            headers=_auth_header(token),
        )
        # Then RESOLVED
        resp = client.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"status": "RESOLVED"},
            headers=_auth_header(token),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "RESOLVED"

    @patch("app.services.transaction.MLServiceClient")
    def test_empty_update_422(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """PATCH with neither status nor notes returns 422."""
        info = _register_and_login(client)
        txn_data = _create_hold_transaction(client, info["access_token"])
        alert_id = txn_data["alert"]["id"]

        analyst, token = _create_analyst_user(s6_db)
        resp = client.patch(
            f"/api/v1/alerts/{alert_id}",
            json={},
            headers=_auth_header(token),
        )
        assert resp.status_code == 422

    def test_update_not_found_404(self, client: TestClient, s6_db: Session) -> None:
        analyst, token = _create_analyst_user(s6_db)
        resp = client.patch(
            f"/api/v1/alerts/{uuid.uuid4()}",
            json={"status": "IN_REVIEW"},
            headers=_auth_header(token),
        )
        assert resp.status_code == 404

    def test_customer_cannot_update_403(self, client: TestClient, s6_db: Session) -> None:
        info = _register_and_login(client)
        resp = client.patch(
            f"/api/v1/alerts/{uuid.uuid4()}",
            json={"status": "IN_REVIEW"},
            headers=_auth_header(info["access_token"]),
        )
        assert resp.status_code == 403


# ===================================================================
# 4. AUDIT LOGGING
# ===================================================================


@requires_pg
class TestAuditLogging:
    """Verify audit log records are created for important actions."""

    @patch("app.services.transaction.MLServiceClient")
    def test_transaction_created_audit_log(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Creating a transaction generates a 'transaction_created' audit log."""
        info = _register_and_login(client)
        txn_data = _create_approve_transaction(client, info["access_token"])
        txn_id = txn_data["id"]

        # Query audit log
        logs = s6_db.query(AuditLog).filter(
            AuditLog.action == "transaction_created",
            AuditLog.resource_id == str(txn_id),
        ).all()
        assert len(logs) == 1
        log = logs[0]
        assert log.actor_id == uuid.UUID(info["user_id"])
        assert log.resource_type == "transaction"
        assert log.details_json is not None
        assert log.details_json["decision"] == "APPROVE"

    @patch("app.services.transaction.MLServiceClient")
    def test_alert_created_audit_log(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Creating a HOLD transaction generates an 'alert_created' audit log."""
        info = _register_and_login(client)
        txn_data = _create_hold_transaction(client, info["access_token"])
        alert_id = txn_data["alert"]["id"]

        logs = s6_db.query(AuditLog).filter(
            AuditLog.action == "alert_created",
            AuditLog.resource_id == str(alert_id),
        ).all()
        assert len(logs) == 1
        log = logs[0]
        assert log.actor_id == uuid.UUID(info["user_id"])
        assert log.resource_type == "alert"

    @patch("app.services.transaction.MLServiceClient")
    def test_no_alert_audit_for_approve(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """APPROVE transactions do NOT generate alert_created audit log."""
        info = _register_and_login(client)
        txn_data = _create_approve_transaction(client, info["access_token"])

        alert_logs = s6_db.query(AuditLog).filter(
            AuditLog.action == "alert_created",
            AuditLog.actor_id == uuid.UUID(info["user_id"]),
        ).all()
        assert len(alert_logs) == 0

    @patch("app.services.transaction.MLServiceClient")
    def test_alert_updated_audit_log(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Updating alert status generates an 'alert_updated' audit log."""
        info = _register_and_login(client)
        txn_data = _create_hold_transaction(client, info["access_token"])
        alert_id = txn_data["alert"]["id"]

        analyst, token = _create_analyst_user(s6_db)
        client.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"status": "IN_REVIEW"},
            headers=_auth_header(token),
        )

        logs = s6_db.query(AuditLog).filter(
            AuditLog.action == "alert_updated",
            AuditLog.resource_id == str(alert_id),
        ).all()
        assert len(logs) == 1
        assert logs[0].actor_id == analyst.id

    @patch("app.services.transaction.MLServiceClient")
    def test_no_duplicate_audit_events(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Single transaction creation produces exactly one audit record."""
        info = _register_and_login(client)
        txn_data = _create_approve_transaction(client, info["access_token"])
        txn_id = txn_data["id"]

        logs = s6_db.query(AuditLog).filter(
            AuditLog.action == "transaction_created",
            AuditLog.resource_id == str(txn_id),
        ).all()
        assert len(logs) == 1

    @patch("app.services.transaction.MLServiceClient")
    def test_audit_does_not_expose_secrets(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Audit details do not contain sensitive values."""
        info = _register_and_login(client)
        txn_data = _create_approve_transaction(client, info["access_token"])
        txn_id = txn_data["id"]

        log = s6_db.query(AuditLog).filter(
            AuditLog.resource_id == str(txn_id),
        ).first()
        assert log is not None
        details_str = str(log.details_json)
        # Should not contain password, token, or secret
        assert "SecurePass1" not in details_str
        assert "password" not in details_str.lower()
        assert info["access_token"] not in details_str


# ===================================================================
# 5. TRANSACTION + ALERT CONSISTENCY
# ===================================================================


@requires_pg
class TestTransactionAlertConsistency:
    """Verify transaction/alert relationship integrity."""

    @patch("app.services.transaction.MLServiceClient")
    def test_hold_creates_alert_with_correct_data(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """HOLD decision creates alert with matching risk data."""
        info = _register_and_login(client)
        txn_data = _create_hold_transaction(client, info["access_token"])

        # Verify alert in DB
        alert = s6_db.get(Alert, uuid.UUID(txn_data["alert"]["id"]))
        assert alert is not None
        assert alert.risk_score == 82
        assert alert.risk_level == "HIGH"
        assert alert.decision == "HOLD"
        assert alert.status == "OPEN"
        assert alert.analyst_id is None  # Unassigned initially

    @patch("app.services.transaction.MLServiceClient")
    def test_approve_no_alert(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """APPROVE decision does NOT create an alert."""
        info = _register_and_login(client)
        txn_data = _create_approve_transaction(client, info["access_token"])

        alerts = s6_db.query(Alert).filter(
            Alert.transaction_id == uuid.UUID(txn_data["id"])
        ).all()
        assert len(alerts) == 0

    @patch("app.services.transaction.MLServiceClient")
    def test_one_alert_per_transaction(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """A transaction has at most one alert."""
        info = _register_and_login(client)
        txn_data = _create_hold_transaction(client, info["access_token"])

        alerts = s6_db.query(Alert).filter(
            Alert.transaction_id == uuid.UUID(txn_data["id"])
        ).all()
        assert len(alerts) == 1


# ===================================================================
# 6. SECURITY HARDENING
# ===================================================================


@requires_pg
class TestSecurityHardening:
    """Verify security requirements."""

    def test_alerts_no_auth_401(self, client: TestClient) -> None:
        """All alert endpoints require authentication."""
        assert client.get("/api/v1/alerts").status_code == 401
        assert client.get(f"/api/v1/alerts/{uuid.uuid4()}").status_code == 401
        assert client.patch(f"/api/v1/alerts/{uuid.uuid4()}", json={"status": "IN_REVIEW"}).status_code == 401

    def test_customer_cannot_access_alerts(self, client: TestClient, s6_db: Session) -> None:
        """Customer role is denied from all alert endpoints."""
        info = _register_and_login(client)
        h = _auth_header(info["access_token"])
        assert client.get("/api/v1/alerts", headers=h).status_code == 403
        assert client.get(f"/api/v1/alerts/{uuid.uuid4()}", headers=h).status_code == 403
        assert client.patch(f"/api/v1/alerts/{uuid.uuid4()}", json={"status": "IN_REVIEW"}, headers=h).status_code == 403

    @patch("app.services.transaction.MLServiceClient")
    def test_inactive_user_rejected(self, MockML: MagicMock, client: TestClient, s6_db: Session) -> None:
        """Inactive users cannot access protected endpoints."""
        info = _register_and_login(client)
        user_id = uuid.UUID(info["user_id"])

        # Deactivate user
        user = s6_db.get(User, user_id)
        user.is_active = False
        s6_db.commit()

        resp = client.get("/api/v1/customers/me", headers=_auth_header(info["access_token"]))
        assert resp.status_code == 403

    def test_internal_errors_not_leaked(self, client: TestClient) -> None:
        """Error responses follow the documented format."""
        # Accessing a nonexistent route returns proper error
        resp = client.get("/api/v1/nonexistent")
        # Should be 404 from the router, not 500
        assert resp.status_code in {404, 405}
