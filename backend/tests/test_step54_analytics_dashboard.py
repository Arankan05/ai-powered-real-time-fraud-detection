"""Step 54 unit tests — GET /api/v1/analytics/dashboard endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend._main import app
from backend.db.alert_repository import SQLiteAlertRepository
from backend.db.transaction_repository import InMemoryTransactionStore
from backend.db.user_repository import SQLiteUserRepository
from backend.routers.analytics import set_alert_repository, set_transaction_repository
from backend.security.deps import set_user_repository
from backend.security.jwt_utils import create_access_token


@pytest.fixture(autouse=True)
def setup_user_repo():
    user_repo = SQLiteUserRepository(":memory:")
    # Seed analyst user
    analyst = user_repo.create_user(
        email="analyst@example.com",
        password="Password123!",
        role="fraud_analyst",
    )
    # Seed customer user
    customer = user_repo.create_user(
        email="customer@example.com",
        password="Password123!",
        role="customer",
    )
    set_user_repository(user_repo)
    return {"analyst_id": analyst["id"], "customer_id": customer["id"]}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def analyst_token(setup_user_repo):
    return create_access_token(
        user_id=setup_user_repo["analyst_id"], role="fraud_analyst"
    )


@pytest.fixture
def customer_token(setup_user_repo):
    return create_access_token(
        user_id=setup_user_repo["customer_id"], role="customer"
    )


def test_analytics_dashboard_unauthorized(client):
    res = client.get("/api/v1/analytics/dashboard")
    assert res.status_code == 401


def test_analytics_dashboard_customer_forbidden(client, customer_token):
    res = client.get(
        "/api/v1/analytics/dashboard",
        headers={"Authorization": f"Bearer {customer_token}"},
    )
    assert res.status_code == 403


def test_analytics_dashboard_success(client, analyst_token):
    tx_store = InMemoryTransactionStore()
    alert_store = SQLiteAlertRepository(":memory:")
    set_transaction_repository(tx_store)
    set_alert_repository(alert_store)

    res = client.get(
        "/api/v1/analytics/dashboard",
        headers={"Authorization": f"Bearer {analyst_token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "total_transactions" in data
    assert "flagged_transactions" in data
    assert "alerts_open" in data
    assert "alerts_resolved" in data
    assert "risk_distribution" in data
    assert "top_risk_factors" in data
    assert "transactions_over_time" in data


def test_analytics_dashboard_invalid_date_range(client, analyst_token):
    res = client.get(
        "/api/v1/analytics/dashboard?from_date=2026-09-05T00:00:00Z&to_date=2026-08-01T00:00:00Z",
        headers={"Authorization": f"Bearer {analyst_token}"},
    )
    assert res.status_code == 422
