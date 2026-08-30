"""Tests for customer, transaction, and fraud check APIs.

Uses ``unittest.mock.patch`` to mock the ML service for transaction tests.
Tests cover:
* Customer profile access + ownership enforcement
* Transaction creation + ML integration
* Transaction retrieval + listing
* Fraud check endpoint
* Alert creation on HOLD decisions
* ML failure handling
* RBAC enforcement
* Response shape validation
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.config import settings
from app.core.security import create_access_token, hash_password
from app.db.session import SessionLocal
from app.models.alert import Alert
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
    return f"s5-{uuid.uuid4().hex[:12]}@test.com"


# Standard ML response fixtures
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

_ML_VERIFY = {
    "ml_score": 40, "behaviour_score": 55, "rule_score": 20,
    "risk_score": 50, "risk_level": "MEDIUM", "decision": "VERIFY",
    "explanation": {
        "ml_top_factors": [{"feature": "amount_deviation", "importance": 0.35}],
        "behaviour_signals": [{"signal": "spending_amount_anomaly", "severity": 0.6}],
        "rules_triggered": [{"rule": "new_device_high_amount", "contribution": 15}],
    },
    "risk_factors": ["amount_deviation", "spending_amount_anomaly", "new_device_high_amount"],
    "model_version": None,
}

_ML_HOLD = {
    "ml_score": 80, "behaviour_score": 90, "rule_score": 40,
    "risk_score": 82, "risk_level": "HIGH", "decision": "HOLD",
    "explanation": {
        "ml_top_factors": [
            {"feature": "amount_deviation", "importance": 0.45},
            {"feature": "location_is_new", "importance": 0.30},
        ],
        "behaviour_signals": [
            {"signal": "spending_amount_anomaly", "severity": 0.9},
        ],
        "rules_triggered": [
            {"rule": "high_amount", "contribution": 15},
            {"rule": "impossible_travel", "contribution": 25},
        ],
    },
    "risk_factors": ["amount_deviation", "location_is_new", "spending_amount_anomaly", "high_amount", "impossible_travel"],
    "model_version": None,
}


def _make_ml_mock(response: dict) -> MagicMock:
    """Create a mock ML client that returns a fixed response."""
    mock = MagicMock(spec=MLServiceClient)
    mock.predict.return_value = response
    return mock


def _make_ml_unavailable_mock() -> MagicMock:
    """Create a mock ML client that raises MLServiceUnavailableError."""
    from app.services.ml import MLServiceUnavailableError

    mock = MagicMock(spec=MLServiceClient)
    mock.predict.side_effect = MLServiceUnavailableError("ML service unavailable")
    return mock


def _make_ml_invalid_mock() -> MagicMock:
    """Create a mock ML client that raises MLInvalidResponseError."""
    from app.services.ml import MLInvalidResponseError

    mock = MagicMock(spec=MLServiceClient)
    mock.predict.side_effect = MLInvalidResponseError("Invalid response")
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cleanup_s5_data(db: Session) -> None:
    """Remove test data created by Step 5 tests."""
    # Delete alerts for test transactions
    users = db.query(User).filter(User.email.like("s5-%@test.com")).all()
    customer_ids = [u.customer_id for u in users if u.customer_id]

    # Get all transactions for test customers
    if customer_ids:
        txns = db.query(Transaction).filter(Transaction.customer_id.in_(customer_ids)).all()
        for txn in txns:
            # Delete alerts for these transactions
            alerts = db.query(Alert).filter(Alert.transaction_id == txn.id).all()
            for a in alerts:
                db.delete(a)
            db.delete(txn)

    # Delete test users
    for user in users:
        db.delete(user)
    db.commit()

    # Delete test customers
    if customer_ids:
        customers = db.query(Customer).filter(Customer.id.in_(customer_ids)).all()
        for c in customers:
            db.delete(c)
        db.commit()

    # Delete test merchants
    merchants = db.query(Merchant).filter(Merchant.name.like("Test Merchant%")).all()
    for m in merchants:
        db.delete(m)
    db.commit()


@pytest.fixture()
def s5_db() -> Generator[Session, None, None]:
    """DB session with cleanup for Step 5 tests."""
    db = SessionLocal()
    try:
        _cleanup_s5_data(db)
        yield db
    finally:
        _cleanup_s5_data(db)
        db.close()


def _register_and_login(
    client: TestClient, *, first_name: str = "Test", last_name: str = "User",
) -> dict:
    """Register a test user and return tokens."""
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
    """Create a non-customer user directly and return (user, token)."""
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


_TXN_REQUEST = {
    "amount": 150.00,
    "currency": "USD",
    "merchant_name": "Test Merchant S5",
    "merchant_category": "5732",
    "transaction_type": "purchase",
    "location_country": "US",
    "location_city": "New York",
    "device_fingerprint": "test-fp-123",
    "device_type": "mobile",
    "ip_address": "192.168.1.100",
}


# ===================================================================
# 1. CUSTOMER ENDPOINTS
# ===================================================================


@requires_pg
class TestCustomerMe:
    """GET /api/v1/customers/me"""

    def test_customer_me_success(self, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        resp = client.get("/api/v1/customers/me", headers=_auth_header(info["access_token"]))
        assert resp.status_code == 200
        data = resp.json()
        assert data["first_name"] == "Test"
        assert data["last_name"] == "User"
        assert "id" in data
        assert "created_at" in data
        assert "is_active" in data
        assert data["is_active"] is True

    def test_customer_me_no_auth_401(self, client: TestClient) -> None:
        resp = client.get("/api/v1/customers/me")
        assert resp.status_code == 401

    def test_customer_me_analyst_forbidden_403(self, client: TestClient, s5_db: Session) -> None:
        user, token = _create_analyst_user(s5_db, "fraud_analyst")
        resp = client.get("/api/v1/customers/me", headers=_auth_header(token))
        assert resp.status_code == 403


@requires_pg
class TestCustomerById:
    """GET /api/v1/customers/{id}"""

    def test_customer_own_profile(self, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        resp = client.get(
            f"/api/v1/customers/{info['customer_id']}",
            headers=_auth_header(info["access_token"]),
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == info["customer_id"]

    def test_customer_other_forbidden(self, client: TestClient, s5_db: Session) -> None:
        """Customer cannot access another customer's profile."""
        info1 = _register_and_login(client, first_name="A")
        info2 = _register_and_login(client, first_name="B")
        resp = client.get(
            f"/api/v1/customers/{info2['customer_id']}",
            headers=_auth_header(info1["access_token"]),
        )
        assert resp.status_code == 403

    def test_customer_analyst_sees_any(self, client: TestClient, s5_db: Session) -> None:
        """Analyst can access any customer profile."""
        info = _register_and_login(client)
        analyst, token = _create_analyst_user(s5_db, "fraud_analyst")
        resp = client.get(
            f"/api/v1/customers/{info['customer_id']}",
            headers=_auth_header(token),
        )
        assert resp.status_code == 200

    def test_customer_not_found_404(self, client: TestClient, s5_db: Session) -> None:
        analyst, token = _create_analyst_user(s5_db, "admin")
        resp = client.get(
            f"/api/v1/customers/{uuid.uuid4()}",
            headers=_auth_header(token),
        )
        assert resp.status_code == 404


# ===================================================================
# 2. TRANSACTION CREATION
# ===================================================================


@requires_pg
class TestTransactionCreate:
    """POST /api/v1/transactions"""

    @patch("app.services.transaction.MLServiceClient")
    def test_create_approve(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_mock(_ML_APPROVE)

        with patch("app.services.transaction.TransactionService.__init__", lambda self, db, ml_client=None: None):
            pass  # We need a different approach

        # Patch at the service level
        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            resp = client.post(
                "/api/v1/transactions",
                json=_TXN_REQUEST,
                headers=_auth_header(info["access_token"]),
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "COMPLETED"
        assert data["decision"] == "APPROVE"
        assert data["risk_level"] == "LOW"
        assert data["ml_score"] == 10
        assert data["behaviour_score"] == 15
        assert data["rule_score"] == 5
        assert data["risk_score"] == 12
        assert data["model_version"] is None
        assert data["alert"] is None
        assert data["amount"] == 150.0
        assert data["currency"] == "USD"
        assert data["customer_id"] == info["customer_id"]

    @patch("app.services.transaction.MLServiceClient")
    def test_create_hold_creates_alert(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_mock(_ML_HOLD)

        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            resp = client.post(
                "/api/v1/transactions",
                json=_TXN_REQUEST,
                headers=_auth_header(info["access_token"]),
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["decision"] == "HOLD"
        assert data["risk_level"] == "HIGH"
        assert data["alert"] is not None
        assert data["alert"]["status"] == "OPEN"
        assert "id" in data["alert"]
        assert "created_at" in data["alert"]

    @patch("app.services.transaction.MLServiceClient")
    def test_create_verify_no_alert(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_mock(_ML_VERIFY)

        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            resp = client.post(
                "/api/v1/transactions",
                json=_TXN_REQUEST,
                headers=_auth_header(info["access_token"]),
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["decision"] == "VERIFY"
        assert data["alert"] is None

    @patch("app.services.transaction.MLServiceClient")
    def test_create_response_shape(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_mock(_ML_APPROVE)

        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            resp = client.post(
                "/api/v1/transactions",
                json=_TXN_REQUEST,
                headers=_auth_header(info["access_token"]),
            )
        data = resp.json()
        expected_keys = {
            "id", "customer_id", "merchant_id", "amount", "currency",
            "merchant_name", "merchant_category", "transaction_type",
            "location_country", "location_city", "device_fingerprint",
            "device_type", "ip_address", "timestamp", "status",
            "ml_score", "behaviour_score", "rule_score", "risk_score",
            "risk_level", "decision", "explanation", "risk_factors",
            "model_version", "alert",
        }
        assert set(data.keys()) == expected_keys

    @patch("app.services.transaction.MLServiceClient")
    def test_create_explanation_structure(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_mock(_ML_HOLD)

        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            resp = client.post(
                "/api/v1/transactions",
                json=_TXN_REQUEST,
                headers=_auth_header(info["access_token"]),
            )
        data = resp.json()
        expl = data["explanation"]
        assert "ml_top_factors" in expl
        assert "behaviour_signals" in expl
        assert "rules_triggered" in expl
        assert len(expl["rules_triggered"]) == 2

    def test_create_no_auth_401(self, client: TestClient) -> None:
        resp = client.post("/api/v1/transactions", json=_TXN_REQUEST)
        assert resp.status_code == 401

    @patch("app.services.transaction.MLServiceClient")
    def test_create_analyst_forbidden_403(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        analyst, token = _create_analyst_user(s5_db, "fraud_analyst")
        resp = client.post(
            "/api/v1/transactions",
            json=_TXN_REQUEST,
            headers=_auth_header(token),
        )
        assert resp.status_code == 403

    def test_create_invalid_amount_422(self, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        bad = {**_TXN_REQUEST, "amount": -10}
        resp = client.post(
            "/api/v1/transactions",
            json=bad,
            headers=_auth_header(info["access_token"]),
        )
        assert resp.status_code == 422

    def test_create_invalid_type_422(self, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        bad = {**_TXN_REQUEST, "transaction_type": "invalid"}
        resp = client.post(
            "/api/v1/transactions",
            json=bad,
            headers=_auth_header(info["access_token"]),
        )
        assert resp.status_code == 422

    @patch("app.services.transaction.MLServiceClient")
    def test_ml_unavailable_503(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_unavailable_mock()

        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            resp = client.post(
                "/api/v1/transactions",
                json=_TXN_REQUEST,
                headers=_auth_header(info["access_token"]),
            )
        assert resp.status_code == 503
        data = resp.json()
        assert data["error_code"] == "ML_SERVICE_UNAVAILABLE"

    @patch("app.services.transaction.MLServiceClient")
    def test_ml_invalid_response_500(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_invalid_mock()

        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            resp = client.post(
                "/api/v1/transactions",
                json=_TXN_REQUEST,
                headers=_auth_header(info["access_token"]),
            )
        assert resp.status_code == 500

    @patch("app.services.transaction.MLServiceClient")
    def test_transaction_persisted_in_db(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_mock(_ML_APPROVE)

        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            resp = client.post(
                "/api/v1/transactions",
                json=_TXN_REQUEST,
                headers=_auth_header(info["access_token"]),
            )
        txn_id = resp.json()["id"]

        # Verify in DB
        txn = s5_db.get(Transaction, uuid.UUID(txn_id))
        assert txn is not None
        assert txn.status == "COMPLETED"
        assert txn.decision == "APPROVE"
        assert txn.ml_score == 10
        assert txn.risk_score == 12
        assert txn.model_version is None

    @patch("app.services.transaction.MLServiceClient")
    def test_hold_alert_persisted_in_db(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_mock(_ML_HOLD)

        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            resp = client.post(
                "/api/v1/transactions",
                json=_TXN_REQUEST,
                headers=_auth_header(info["access_token"]),
            )
        txn_id = uuid.UUID(resp.json()["id"])

        # Verify alert in DB
        alert = s5_db.query(Alert).filter(Alert.transaction_id == txn_id).one()
        assert alert.status == "OPEN"
        assert alert.risk_score == 82
        assert alert.decision == "HOLD"

    @patch("app.services.transaction.MLServiceClient")
    def test_no_alert_for_approve(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_mock(_ML_APPROVE)

        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            resp = client.post(
                "/api/v1/transactions",
                json=_TXN_REQUEST,
                headers=_auth_header(info["access_token"]),
            )
        txn_id = uuid.UUID(resp.json()["id"])

        alert_count = s5_db.query(Alert).filter(Alert.transaction_id == txn_id).count()
        assert alert_count == 0


# ===================================================================
# 3. TRANSACTION RETRIEVAL
# ===================================================================


@requires_pg
class TestTransactionGet:
    """GET /api/v1/transactions/{id}"""

    @patch("app.services.transaction.MLServiceClient")
    def test_get_own_transaction(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_mock(_ML_APPROVE)

        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            create_resp = client.post(
                "/api/v1/transactions",
                json=_TXN_REQUEST,
                headers=_auth_header(info["access_token"]),
            )
        txn_id = create_resp.json()["id"]

        resp = client.get(
            f"/api/v1/transactions/{txn_id}",
            headers=_auth_header(info["access_token"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == txn_id
        assert data["decision"] == "APPROVE"

    @patch("app.services.transaction.MLServiceClient")
    def test_get_other_customer_forbidden(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        """Customer A cannot see Customer B's transaction."""
        info_a = _register_and_login(client, first_name="A")
        mock_client = _make_ml_mock(_ML_APPROVE)
        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            create_resp = client.post(
                "/api/v1/transactions",
                json=_TXN_REQUEST,
                headers=_auth_header(info_a["access_token"]),
            )
        txn_id = create_resp.json()["id"]

        info_b = _register_and_login(client, first_name="B")
        resp = client.get(
            f"/api/v1/transactions/{txn_id}",
            headers=_auth_header(info_b["access_token"]),
        )
        assert resp.status_code == 403

    def test_get_not_found_404(self, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        resp = client.get(
            f"/api/v1/transactions/{uuid.uuid4()}",
            headers=_auth_header(info["access_token"]),
        )
        assert resp.status_code == 404

    @patch("app.services.transaction.MLServiceClient")
    def test_analyst_sees_any_transaction(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_mock(_ML_APPROVE)
        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            create_resp = client.post(
                "/api/v1/transactions",
                json=_TXN_REQUEST,
                headers=_auth_header(info["access_token"]),
            )
        txn_id = create_resp.json()["id"]

        analyst, token = _create_analyst_user(s5_db, "fraud_analyst")
        resp = client.get(
            f"/api/v1/transactions/{txn_id}",
            headers=_auth_header(token),
        )
        assert resp.status_code == 200


# ===================================================================
# 4. TRANSACTION LISTING
# ===================================================================


@requires_pg
class TestTransactionList:
    """GET /api/v1/transactions"""

    @patch("app.services.transaction.MLServiceClient")
    def test_list_own_transactions(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_mock(_ML_APPROVE)
        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            client.post("/api/v1/transactions", json=_TXN_REQUEST, headers=_auth_header(info["access_token"]))

        resp = client.get("/api/v1/transactions", headers=_auth_header(info["access_token"]))
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "per_page" in data
        assert data["total"] >= 1

    def test_list_no_auth_401(self, client: TestClient) -> None:
        resp = client.get("/api/v1/transactions")
        assert resp.status_code == 401

    @patch("app.services.transaction.MLServiceClient")
    def test_list_response_shape(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_mock(_ML_APPROVE)
        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            client.post("/api/v1/transactions", json=_TXN_REQUEST, headers=_auth_header(info["access_token"]))

        resp = client.get("/api/v1/transactions", headers=_auth_header(info["access_token"]))
        data = resp.json()
        if data["items"]:
            item = data["items"][0]
            expected = {"id", "customer_id", "merchant_name", "amount", "currency",
                        "transaction_type", "timestamp", "status", "risk_score",
                        "risk_level", "decision"}
            assert set(item.keys()) == expected


# ===================================================================
# 5. FRAUD CHECK ENDPOINT
# ===================================================================


@requires_pg
class TestFraudCheck:
    """POST /api/v1/fraud/check"""

    @patch("app.services.transaction.MLServiceClient")
    def test_fraud_check_success(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        analyst, token = _create_analyst_user(s5_db, "fraud_analyst")
        # Need a customer_id for the fraud check
        info = _register_and_login(client)
        customer_id = info["customer_id"]

        mock_client = _make_ml_mock(_ML_VERIFY)
        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            resp = client.post(
                "/api/v1/fraud/check",
                json={**_TXN_REQUEST, "customer_id": customer_id},
                headers=_auth_header(token),
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "VERIFY"
        assert data["risk_level"] == "MEDIUM"
        assert "explanation" in data
        assert "risk_factors" in data
        assert data["model_version"] is None

    def test_fraud_check_customer_forbidden(self, client: TestClient, s5_db: Session) -> None:
        """Customer role cannot access fraud check."""
        info = _register_and_login(client)
        resp = client.post(
            "/api/v1/fraud/check",
            json={**_TXN_REQUEST, "customer_id": str(info["customer_id"])},
            headers=_auth_header(info["access_token"]),
        )
        assert resp.status_code == 403

    def test_fraud_check_no_auth_401(self, client: TestClient) -> None:
        resp = client.post("/api/v1/fraud/check", json={**_TXN_REQUEST, "customer_id": str(uuid.uuid4())})
        assert resp.status_code == 401

    @patch("app.services.transaction.MLServiceClient")
    def test_fraud_check_customer_not_found_404(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        analyst, token = _create_analyst_user(s5_db, "admin")
        mock_client = _make_ml_mock(_ML_APPROVE)
        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            resp = client.post(
                "/api/v1/fraud/check",
                json={**_TXN_REQUEST, "customer_id": str(uuid.uuid4())},
                headers=_auth_header(token),
            )
        assert resp.status_code == 404

    @patch("app.services.transaction.MLServiceClient")
    def test_fraud_check_ml_unavailable_503(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        analyst, token = _create_analyst_user(s5_db, "fraud_analyst")
        info = _register_and_login(client)
        mock_client = _make_ml_unavailable_mock()
        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            resp = client.post(
                "/api/v1/fraud/check",
                json={**_TXN_REQUEST, "customer_id": str(info["customer_id"])},
                headers=_auth_header(token),
            )
        assert resp.status_code == 503

    @patch("app.services.transaction.MLServiceClient")
    def test_fraud_check_no_persistence(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        """Fraud check does NOT create a transaction record."""
        analyst, token = _create_analyst_user(s5_db, "fraud_analyst")
        info = _register_and_login(client)
        mock_client = _make_ml_mock(_ML_APPROVE)

        # Count transactions before
        before_count = s5_db.query(Transaction).filter(
            Transaction.customer_id == uuid.UUID(info["customer_id"])
        ).count()

        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            resp = client.post(
                "/api/v1/fraud/check",
                json={**_TXN_REQUEST, "customer_id": str(info["customer_id"])},
                headers=_auth_header(token),
            )
        assert resp.status_code == 200

        after_count = s5_db.query(Transaction).filter(
            Transaction.customer_id == uuid.UUID(info["customer_id"])
        ).count()
        assert after_count == before_count


# ===================================================================
# 6. CUSTOMER TRANSACTIONS
# ===================================================================


@requires_pg
class TestCustomerTransactions:
    """GET /api/v1/customers/{id}/transactions"""

    @patch("app.services.transaction.MLServiceClient")
    def test_customer_own_transactions(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info = _register_and_login(client)
        mock_client = _make_ml_mock(_ML_APPROVE)
        with patch("app.services.transaction.MLServiceClient") as MockCls:
            MockCls.return_value = mock_client
            client.post("/api/v1/transactions", json=_TXN_REQUEST, headers=_auth_header(info["access_token"]))

        resp = client.get(
            f"/api/v1/customers/{info['customer_id']}/transactions",
            headers=_auth_header(info["access_token"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert data["total"] >= 1

    @patch("app.services.transaction.MLServiceClient")
    def test_other_customer_forbidden(self, MockML: MagicMock, client: TestClient, s5_db: Session) -> None:
        info_a = _register_and_login(client, first_name="A")
        info_b = _register_and_login(client, first_name="B")

        resp = client.get(
            f"/api/v1/customers/{info_b['customer_id']}/transactions",
            headers=_auth_header(info_a["access_token"]),
        )
        assert resp.status_code == 403
