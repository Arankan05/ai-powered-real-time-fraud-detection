"""Shared test fixtures for the backend test suite."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import create_app
from app.db.session import SessionLocal


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
