"""Tests for the health-check endpoint."""

from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_response_body(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    data = response.json()

    assert data["status"] == "healthy"
    assert "version" in data
    assert isinstance(data["version"], str)


def test_health_response_has_no_extra_fields(client: TestClient) -> None:
    """Ensure the response shape is predictable."""
    response = client.get("/api/v1/health")
    data = response.json()

    assert set(data.keys()) == {"status", "version"}
