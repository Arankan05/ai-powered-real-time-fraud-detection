"""Shared test fixtures for the backend test suite."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def client() -> TestClient:
    """Create a FastAPI test client backed by a fresh app instance."""
    app = create_app()
    return TestClient(app)
