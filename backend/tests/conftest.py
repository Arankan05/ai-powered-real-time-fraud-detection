"""Shared test fixtures for the backend test suite."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import create_app
from app.db.session import SessionLocal


# ---------------------------------------------------------------------------
# Cached PostgreSQL availability check
# ---------------------------------------------------------------------------
# Each test module defines its own _pg_available() and evaluates it at import
# time (for pytest.mark.skipif).  Without caching every module opens a real
# TCP connection which is slow when the server is unreachable.  We monkey-
# patch sqlalchemy.create_engine in pytest_configure so the *first* connection
# attempt (one TCP round-trip) is cached and replayed for every subsequent
# module, keeping collection fast.

_pg_check_result: bool | None = None


def _cached_create_engine(url, *args, **kwargs):
    """Wrapper around create_engine that caches the connection check."""
    global _pg_check_result
    from sqlalchemy import create_engine as _real_create_engine

    eng = _real_create_engine(url, *args, **kwargs)
    if _pg_check_result is not None:
        # Already checked — return an engine whose connect() will
        # succeed or fail instantly based on the cached result.
        if not _pg_check_result:
            # Return an engine that fails fast on connect.
            _original_connect = eng.connect

            def _fast_fail_connect():
                from sqlalchemy.exc import OperationalError
                raise OperationalError("cached: PostgreSQL not reachable", None, None)

            eng.connect = _fast_fail_connect  # type: ignore[assignment]
        return eng

    # First call — do the real check and cache the result.
    try:
        with eng.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))
        _pg_check_result = True
    except Exception:
        _pg_check_result = False
    return eng


def pytest_configure(config):
    """Patch sqlalchemy.create_engine for fast _pg_available() checks."""
    import sqlalchemy
    sqlalchemy.create_engine = _cached_create_engine


# ---------------------------------------------------------------------------
# Application & database fixtures (app.main / SQLAlchemy path)
# ---------------------------------------------------------------------------

@pytest.fixture()
def client() -> TestClient:
    """Create a FastAPI test client backed by a fresh app instance."""
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    """Provide a real database session for tests that need direct DB access."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Auth override & shared payloads (backend.app / repository path)
# ---------------------------------------------------------------------------

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
