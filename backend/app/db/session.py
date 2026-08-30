"""Database engine, session factory, and FastAPI dependency injection.

Uses **synchronous** SQLAlchemy as specified in ``docs/architecture.md``
("Synchronous ORM — SQLAlchemy over PostgreSQL wire").
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

engine = create_engine(
    settings.postgres.database_url,
    connect_args=settings.postgres.connect_args,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session per request.

    The session is automatically closed when the request completes,
    ensuring clean connection lifecycle.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
