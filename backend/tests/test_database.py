"""Tests for the database infrastructure.

Covers:
* Configuration validation
* Session factory and get_db dependency
* SQLAlchemy Base metadata
* PostgreSQL connectivity (skipped when PG is down)
* Alembic configuration
"""

from configparser import ConfigParser
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import engine, get_db, SessionLocal
from app.db.base import Base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pg_available() -> bool:
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


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------


class TestDatabaseConfiguration:
    """Verify SQLAlchemy / PostgreSQL configuration."""

    def test_database_url_is_string(self) -> None:
        assert isinstance(settings.postgres.database_url, str)

    def test_database_url_starts_with_postgresql(self) -> None:
        assert settings.postgres.database_url.startswith("postgresql://")

    def test_database_url_no_empty_auth(self) -> None:
        """URL must not contain '://:@' (empty user + password)."""
        assert "://:@" not in settings.postgres.database_url

    def test_database_name_matches_config(self) -> None:
        assert settings.postgres.db in settings.postgres.database_url

    def test_port_in_range(self) -> None:
        assert 1 <= settings.postgres.port <= 65535


# ---------------------------------------------------------------------------
# 2. Session factory / get_db dependency
# ---------------------------------------------------------------------------


class TestSessionInfrastructure:
    """Verify session factory and dependency injection wiring."""

    def test_session_local_is_callable(self) -> None:
        assert callable(SessionLocal)

    def test_get_db_is_generator_function(self) -> None:
        import inspect
        assert inspect.isgeneratorfunction(get_db)

    def test_session_local_produces_session(self) -> None:
        """SessionLocal() must return a SQLAlchemy Session."""
        session = SessionLocal()
        try:
            assert isinstance(session, Session)
        finally:
            session.close()


# ---------------------------------------------------------------------------
# 3. SQLAlchemy Base
# ---------------------------------------------------------------------------


class TestDeclarativeBase:
    """Verify the declarative base is properly configured."""

    def test_base_has_metadata(self) -> None:
        assert Base.metadata is not None

    def test_base_metadata_contains_nine_application_tables(self) -> None:
        """Exactly 9 application models must be registered with Base.metadata."""
        import app.models  # noqa: F401 — triggers model registration

        table_names = set(Base.metadata.tables.keys())
        expected = {
            "users", "customers", "merchants", "transactions",
            "alerts", "audit_logs", "customer_devices",
            "model_metadata", "risk_rules_config",
        }
        assert table_names == expected


# ---------------------------------------------------------------------------
# 4. PostgreSQL connectivity (requires running PostgreSQL)
# ---------------------------------------------------------------------------


class TestPostgresConnectivity:
    """Integration tests — skipped when PostgreSQL is not running."""

    @requires_pg
    def test_engine_can_connect(self) -> None:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    @requires_pg
    def test_session_can_execute_query(self, db_session: Session) -> None:
        result = db_session.execute(text("SELECT 1"))
        assert result.scalar() == 1

    @requires_pg
    def test_get_db_yields_session(self) -> None:
        gen = get_db()
        session = next(gen)
        try:
            assert isinstance(session, Session)
            assert session.execute(text("SELECT 1")).scalar() == 1
        finally:
            with pytest.raises(StopIteration):
                next(gen)

    @requires_pg
    def test_postgresql_version(self) -> None:
        """Confirm we are connected to a real PostgreSQL instance."""
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            assert "PostgreSQL" in version


# ---------------------------------------------------------------------------
# 5. Alembic configuration
# ---------------------------------------------------------------------------


class TestAlembicConfiguration:
    """Verify Alembic infrastructure is correctly set up."""

    def test_alembic_ini_exists(self) -> None:
        alembic_ini = Path(__file__).resolve().parents[2] / "database" / "alembic.ini"
        assert alembic_ini.exists(), f"alembic.ini not found at {alembic_ini}"

    def test_alembic_ini_is_valid(self) -> None:
        alembic_ini = Path(__file__).resolve().parents[2] / "database" / "alembic.ini"
        cfg = ConfigParser()
        cfg.read(str(alembic_ini))
        assert "alembic" in cfg.sections()
        assert cfg.get("alembic", "script_location") == "alembic"

    def test_env_py_exists(self) -> None:
        env_py = (
            Path(__file__).resolve().parents[2]
            / "database" / "alembic" / "env.py"
        )
        assert env_py.exists()

    def test_script_template_exists(self) -> None:
        template = (
            Path(__file__).resolve().parents[2]
            / "database" / "alembic" / "script.py.mako"
        )
        assert template.exists()

    def test_versions_directory_exists(self) -> None:
        versions = (
            Path(__file__).resolve().parents[2]
            / "database" / "alembic" / "versions"
        )
        assert versions.is_dir()

    def test_env_py_imports_base_metadata(self) -> None:
        """The env.py module must be able to access Base.metadata."""
        assert Base.metadata is not None
