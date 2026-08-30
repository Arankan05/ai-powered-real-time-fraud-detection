"""Tests for the health-check endpoint.

Tests that require a running PostgreSQL instance are automatically skipped
when the database is not available.
"""

import pytest
from sqlalchemy import create_engine, text

from app.config import settings
from fastapi.testclient import TestClient


def _pg_available() -> bool:
    """Return True when a real PostgreSQL database is reachable."""
    try:
        eng = create_engine(settings.postgres.database_url)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


requires_pg = pytest.mark.skipif(
    not _pg_available(),
    reason="PostgreSQL is not running",
)


def test_health_response_shape(client: TestClient) -> None:
    """Health endpoint always returns the expected response structure."""
    response = client.get("/api/v1/health")
    data = response.json()

    assert "status" in data
    assert "version" in data
    assert data["version"] == "0.1.0"
    assert "services" in data
    assert "database" in data["services"]
    assert "ml_service" in data["services"]


@requires_pg
def test_health_200_when_db_connected(client: TestClient) -> None:
    """When PostgreSQL is running, health returns 200 and connected."""
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["services"]["database"]["status"] == "connected"
    assert data["services"]["ml_service"]["status"] == "not_configured"


@pytest.mark.skipif(
    _pg_available(),
    reason="PostgreSQL is running — testing degraded path instead",
)
def test_health_503_when_db_unavailable(client: TestClient) -> None:
    """When PostgreSQL is NOT running, health returns 503 degraded."""
    response = client.get("/api/v1/health")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["services"]["database"]["status"] == "disconnected"
