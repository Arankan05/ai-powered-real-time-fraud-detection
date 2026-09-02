"""Shared pytest fixtures for backend tests."""

import pytest


# Fixed analyst identity injected via dependency override for legacy
# (pre-Step-39) tests.  These tests exercise the transaction / ML /
# alert logic — not authentication — so they receive an authenticated
# analyst through FastAPI's dependency_overrides instead of real JWTs.
# Step 39 tests use the real register/login flow.
TEST_ANALYST_USER = {
    "id": "00000000-0000-4000-8000-000000000001",
    "email": "test.analyst@test.local",
    "role": "fraud_analyst",
    "is_active": True,
    "first_name": "Test",
    "last_name": "Analyst",
    "customer_id": "00000000-0000-4000-8000-0000000000a1",
    "created_at": "2026-01-01T00:00:00+00:00",
}


@pytest.fixture
def auth_override():
    """Override JWT auth with a fixed fraud-analyst identity.

    Installs a FastAPI dependency override for ``get_current_user``
    (also inherited by ``require_roles`` dependencies) and removes it
    on teardown so later tests are unaffected.
    """
    from backend.app import app
    from backend.security.deps import get_current_user

    app.dependency_overrides[get_current_user] = lambda: dict(TEST_ANALYST_USER)
    yield TEST_ANALYST_USER
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def valid_transaction() -> dict:
    """A valid raw transaction payload."""
    return {
        "amount": 1500.00,
        "currency": "USD",
        "merchant_name": "Acme Electronics",
        "merchant_category": "5732",
        "transaction_type": "purchase",
        "location_country": "US",
        "location_city": "New York",
        "device_fingerprint": "abc123def456",
        "device_type": "mobile",
        "ip_address": "192.168.1.100",
    }
