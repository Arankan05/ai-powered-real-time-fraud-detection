"""SQLAlchemy declarative base for all ORM models.

Every model in the application must inherit from ``Base``.  Alembic uses
``Base.metadata`` to detect schema changes and generate migrations.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base class for all database models."""
    pass
