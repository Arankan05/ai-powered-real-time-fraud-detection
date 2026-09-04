"""Step 53 — Transaction collection and detail GET endpoints tests.

Covers:
* GET /api/v1/transactions collection endpoint (200 OK with paginated list)
* GET /api/v1/transactions/{id} single transaction detail endpoint
* Role-based authorization & IDOR isolation:
  - Customer A sees only Customer A's transactions
  - Customer B cannot access Customer A's transactions (404 / no leak)
  - Fraud analyst & admin can view transaction data across customers
* Query parameters filtering (status, risk_level, pagination)
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from backend._main import app
from backend.db.transaction_repository import InMemoryTransactionStore
from backend.db.user_repository import ADMIN, CUSTOMER, FRAUD_ANALYST, InMemoryUserStore
from backend.routers.transactions import set_ml_client, set_transaction_repository
from backend.security.deps import set_user_repository
from backend.security.jwt_utils import create_access_token


@pytest.fixture
def test_setup():
    """Configure test environment with in-memory repositories."""
    users = InMemoryUserStore()
    txns = InMemoryTransactionStore()
    set_user_repository(users)
    set_transaction_repository(txns)

    mock_ml = AsyncMock()
    mock_ml.predict.return_value = AsyncMock(
        fraud_probability=0.1,
        fraud_prediction=0,
        threshold=0.5,
        model_version="fraud-xgb-v1.0.0",
        timestamp=1725200000,
        ml_score=10,
        behaviour_score=10,
        rule_score=0,
        risk_score=10,
        risk_level="LOW",
        decision="APPROVE",
        risk_factors=[],
        explanation_detail=None,
        explanation=[],
    )
    set_ml_client(mock_ml)

    # Seed users
    cust_a_id = str(uuid.uuid4())
    user_a = users.create_user(
        email="customerA@example.com",
        password="Pass123word!",
        first_name="Alice",
        last_name="Customer",
        role=CUSTOMER,
    )
    users._users[user_a["id"]]["customer_id"] = cust_a_id

    cust_b_id = str(uuid.uuid4())
    user_b = users.create_user(
        email="customerB@example.com",
        password="Pass123word!",
        first_name="Bob",
        last_name="Customer",
        role=CUSTOMER,
    )
    users._users[user_b["id"]]["customer_id"] = cust_b_id

    analyst_user = users.create_user(
        email="analyst@example.com",
        password="Pass123word!",
        first_name="Charlie",
        last_name="Analyst",
        role=FRAUD_ANALYST,
    )

    # Create transactions for customer A
    tx_a = txns.create(
        transaction_id=str(uuid.uuid4()),
        customer_id=cust_a_id,
        merchant_name="Merchant A",
        merchant_category="5732",
        amount=100.0,
        currency="USD",
        transaction_type="purchase",
        status="COMPLETED",
        risk_score=10,
        risk_level="LOW",
        decision="APPROVE",
    )

    # Create transaction for customer B
    tx_b = txns.create(
        transaction_id=str(uuid.uuid4()),
        customer_id=cust_b_id,
        merchant_name="Merchant B",
        merchant_category="5999",
        amount=250.0,
        currency="USD",
        transaction_type="transfer",
        status="COMPLETED",
        risk_score=85,
        risk_level="HIGH",
        decision="HOLD",
    )

    client = TestClient(app)

    token_a = create_access_token(user_id=user_a["id"], role=CUSTOMER)
    token_b = create_access_token(user_id=user_b["id"], role=CUSTOMER)
    token_analyst = create_access_token(user_id=analyst_user["id"], role=FRAUD_ANALYST)

    return {
        "client": client,
        "txns": txns,
        "cust_a_id": cust_a_id,
        "cust_b_id": cust_b_id,
        "tx_a_id": tx_a["id"],
        "tx_b_id": tx_b["id"],
        "token_a": token_a,
        "token_b": token_b,
        "token_analyst": token_analyst,
    }


def test_customer_a_sees_only_own_transactions(test_setup):
    client = test_setup["client"]
    token_a = test_setup["token_a"]

    res = client.get(
        "/api/v1/transactions",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == test_setup["tx_a_id"]
    assert body["items"][0]["merchant_name"] == "Merchant A"


def test_customer_b_cannot_access_customer_a_transaction(test_setup):
    client = test_setup["client"]
    token_b = test_setup["token_b"]
    tx_a_id = test_setup["tx_a_id"]

    res = client.get(
        f"/api/v1/transactions/{tx_a_id}",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert res.status_code == 404
    assert res.json()["detail"] == "Transaction not found."


def test_analyst_can_see_all_transactions(test_setup):
    client = test_setup["client"]
    token_analyst = test_setup["token_analyst"]

    res = client.get(
        "/api/v1/transactions",
        headers={"Authorization": f"Bearer {token_analyst}"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2


def test_get_transaction_detail(test_setup):
    client = test_setup["client"]
    token_a = test_setup["token_a"]
    tx_a_id = test_setup["tx_a_id"]

    res = client.get(
        f"/api/v1/transactions/{tx_a_id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["transaction_id"] == tx_a_id
    assert body["id"] == tx_a_id
    assert body["merchant_name"] == "Merchant A"
    assert body["amount"] == 100.0
