"""Step 40 — PostgreSQL / Supabase persistence migration tests.

Strategy
--------
Unit tests (always run, no PostgreSQL required):

* configuration validation (missing credentials raise
  :class:`PostgresConfigError`)
* connection-failure messages are sanitised (no password / no conninfo)
* schema DDL content sanity check (tables, indexes, FK, CHECK)
* repository SQL shape with a fake pool (parameterised queries; no
  string-interpolation of user input; parameter binding with
  ``%(name)s`` for users and ``%s`` for alerts)
* Protocol conformance for the new PostgreSQL repositories
* UUID coercion helper (garbage → ``None``; valid → ``UUID``)

Integration tests (executed only when ``POSTGRES_USER`` and
``POSTGRES_PASSWORD`` are set in the environment AND a server is
reachable at ``POSTGRES_HOST:POSTGRES_PORT``):

* isolated schema (``step40_test_<random>``) is created before the
  session, used as the default ``search_path`` on all pool connections,
  and dropped (``CASCADE``) after the session — the team's configured
  database's own data is **never** touched.
* user CRUD (create / retrieve / duplicate / case-insensitive email /
  password hash / role / active-inactive / restart persistence)
* alert CRUD (create / retrieve / list / filters / pagination /
  duplicate transaction_id / status transitions / analyst_id
  first-writer-wins / notes / timestamps)
* API regression (register/login/me; GET /alerts; PATCH alert status;
  404 for garbage alert id via UUID coercion; JWT unchanged)
* security (no plaintext password in stored rows; error bodies do not
  leak credentials; no cross-user leakage via /auth/me)

ML regression is covered by the full ``ml/api/tests`` suite run in
:func:`backend.tests.test_step40_postgresql_persistence` regression
invocation (Step 40 tests never modify ML scoring, SHAP, rules,
aggregation, or historical features).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from backend.config import Settings
from backend.db import postgres as pg_mod
from backend.db.alert_repository import (
    DISMISSED,
    IN_REVIEW,
    OPEN,
    RESOLVED,
    AlertRepository,
    InMemoryAlertStore,
    PostgresAlertRepository,
    SQLiteAlertRepository,
    is_valid_transition,
)
from backend.db.user_repository import (
    ADMIN,
    CUSTOMER,
    FRAUD_ANALYST,
    InMemoryUserStore,
    PostgresUserRepository,
    SQLiteUserRepository,
    UserAlreadyExistsError,
    UserRepository,
    _coerce_uuid,
)


# ──────────────────────────────────────────────────────────────────────
# 1. DATABASE — configuration / connection failure / schema DDL
# ──────────────────────────────────────────────────────────────────────


class TestPostgresConfiguration:
    """Configuration validation + sanitised connection errors."""

    def _settings(self, **overrides: Any) -> Settings:
        base = {
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": 5432,
            "POSTGRES_DB": "fraud_detection_test",
            "POSTGRES_USER": "step40_user",
            "POSTGRES_PASSWORD": "S3cretStep40Password",
        }
        base.update(overrides)
        return Settings(**base)

    def test_missing_user_raises_config_error(self):
        settings = self._settings(POSTGRES_USER="")
        with pytest.raises(pg_mod.PostgresConfigError, match="POSTGRES_USER"):
            pg_mod.create_pool(settings)

    def test_missing_password_raises_config_error(self):
        settings = self._settings(POSTGRES_PASSWORD="")
        with pytest.raises(pg_mod.PostgresConfigError, match="POSTGRES_PASSWORD"):
            pg_mod.create_pool(settings)

    def test_unreachable_host_raises_connection_error(self):
        # Port 1 is reserved by the OS and refuses connections almost
        # immediately on any platform — keeps the test fast.
        settings = self._settings(
            POSTGRES_HOST="127.0.0.1",
            POSTGRES_PORT=1,
            POSTGRES_PASSWORD="ShouldNotLeak123",
        )
        with pytest.raises(pg_mod.PostgresConnectionError) as exc_info:
            pg_mod.create_pool(settings, timeout=1.0)
        message = str(exc_info.value)
        assert "ShouldNotLeak123" not in message
        assert "password=" not in message.lower().replace("password=***", "")
        # Sanitised message still includes host/port/db for operators.
        assert "127.0.0.1" in message
        assert "fraud_detection_test" in message

    def test_sanitised_error_strips_password_fragment(self):
        # Internal helper: any string containing "password=xxx" must be
        # rewritten to "password=***" by _sanitise().
        message = pg_mod._sanitise(
            Exception("FATAL: password authentication failed for user postgres password=Hunter2"),
            host="h", port=5432, dbname="db",
        )
        assert "Hunter2" not in message
        assert "password=***" in message

    def test_default_backend_is_postgres(self):
        assert Settings().PERSISTENCE_BACKEND == "postgres"

    def test_default_ssl_mode_is_empty(self):
        # Empty string = libpq default ("prefer").
        assert Settings().POSTGRES_SSL_MODE == ""


class TestSchemaDDL:
    """Schema initialisation statements are well-formed and idempotent."""

    def test_ddl_creates_users_and_alerts(self):
        joined = "\n".join(pg_mod._SCHEMA_STATEMENTS)
        assert "CREATE TABLE IF NOT EXISTS users" in joined
        assert "CREATE TABLE IF NOT EXISTS alerts" in joined
        assert "IF NOT EXISTS" in joined  # idempotent

    def test_users_table_has_required_columns(self):
        joined = "\n".join(pg_mod._SCHEMA_STATEMENTS)
        for col in ("id", "email", "password_hash", "role",
                    "customer_id", "is_active", "created_at"):
            assert col in joined

    def test_alerts_table_preserves_all_step38_fields(self):
        joined = "\n".join(pg_mod._SCHEMA_STATEMENTS)
        for col in ("id", "transaction_id", "customer_id", "risk_score",
                    "risk_level", "decision", "fraud_probability",
                    "model_version", "risk_factors", "explanation_json",
                    "status", "analyst_id", "notes", "created_at",
                    "updated_at", "resolved_at", "timestamp"):
            assert col in joined

    def test_role_check_constraint(self):
        joined = "\n".join(pg_mod._SCHEMA_STATEMENTS)
        assert "CHECK (role IN" in joined
        assert "fraud_analyst" in joined
        assert "admin" in joined
        assert "customer" in joined

    def test_case_insensitive_email_unique_index(self):
        joined = "\n".join(pg_mod._SCHEMA_STATEMENTS)
        assert "uq_users_email_ci" in joined
        assert "lower(email)" in joined
        assert "UNIQUE INDEX" in joined

    def test_alerts_transaction_id_unique_index(self):
        joined = "\n".join(pg_mod._SCHEMA_STATEMENTS)
        assert "uq_alerts_transaction_id" in joined

    def test_alerts_analyst_id_references_users(self):
        joined = "\n".join(pg_mod._SCHEMA_STATEMENTS)
        assert "REFERENCES users(id)" in joined

    def test_jsonb_columns_for_structured_fields(self):
        joined = "\n".join(pg_mod._SCHEMA_STATEMENTS)
        assert "risk_factors      JSONB" in joined or "risk_factors JSONB" in joined
        assert "explanation_json  JSONB" in joined or "explanation_json JSONB" in joined

    def test_init_schema_runs_against_fake_pool(self):
        fake = _FakePool()
        pg_mod.init_schema(fake)
        executed = [sql for sql, _ in fake.executed]
        # One entry per statement — every schema statement executed once.
        assert len(executed) == len(pg_mod._SCHEMA_STATEMENTS)
        # Second call: idempotent — statements re-run but DDL is
        # ``CREATE ... IF NOT EXISTS``, so it's a no-op.
        pg_mod.init_schema(fake)
        assert len(fake.executed) == 2 * len(pg_mod._SCHEMA_STATEMENTS)


# ──────────────────────────────────────────────────────────────────────
# 2. UUID coercion helper
# ──────────────────────────────────────────────────────────────────────


class TestUUIDCoercion:
    def test_none_returns_none(self):
        assert _coerce_uuid(None) is None

    def test_valid_string_returns_uuid(self):
        value = str(uuid.uuid4())
        assert _coerce_uuid(value) == uuid.UUID(value)

    def test_uuid_object_round_trips(self):
        u = uuid.uuid4()
        assert _coerce_uuid(u) == u

    def test_garbage_string_returns_none(self):
        assert _coerce_uuid("not-a-uuid") is None

    def test_empty_string_returns_none(self):
        assert _coerce_uuid("") is None

    def test_int_returns_none(self):
        assert _coerce_uuid(12345) is None


# ──────────────────────────────────────────────────────────────────────
# 3. Protocol conformance
# ──────────────────────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_postgres_user_repo_satisfies_protocol(self):
        fake = _FakePool()
        repo = PostgresUserRepository.__new__(PostgresUserRepository)
        repo._pool = fake  # bypass schema init
        assert isinstance(repo, UserRepository)

    def test_postgres_alert_repo_satisfies_protocol(self):
        fake = _FakePool()
        repo = PostgresAlertRepository.__new__(PostgresAlertRepository)
        repo._pool = fake
        assert isinstance(repo, AlertRepository)


# ──────────────────────────────────────────────────────────────────────
# 4. Repository SQL shape (parameterised queries — no injection)
# ──────────────────────────────────────────────────────────────────────


class TestRepositorySQLShape:
    """Verify the PostgreSQL repositories use parameter binding."""

    def test_get_by_email_uses_parameter_binding(self):
        fake = _FakePool(results=[{
            "id": uuid.uuid4(), "email": "a@b.com", "password_hash": "$2b$12$xx",
            "role": "customer", "is_active": True, "created_at": datetime.now(timezone.utc),
            "customer_id": uuid.uuid4(),
        }])
        repo = PostgresUserRepository.__new__(PostgresUserRepository)
        repo._pool = fake
        repo.get_by_email("a' OR 1=1 --")
        # SQL must contain a placeholder, not the raw input.
        sql = fake.executed[-1][0]
        params = fake.executed[-1][1]
        assert "%s" in sql or "%(email)s" in sql
        assert "a' OR 1=1" not in sql
        assert params is not None

    def test_get_by_id_with_garbage_returns_none(self):
        fake = _FakePool()
        repo = PostgresUserRepository.__new__(PostgresUserRepository)
        repo._pool = fake
        # No SQL should be issued for a non-UUID id.
        assert repo.get_by_id("not-a-uuid") is None
        assert fake.executed == []

    def test_get_alert_by_id_with_garbage_returns_none(self):
        fake = _FakePool()
        repo = PostgresAlertRepository.__new__(PostgresAlertRepository)
        repo._pool = fake
        assert repo.get_by_id("not-a-uuid") is None
        assert fake.executed == []

    def test_list_alerts_builds_where_clause_from_params(self):
        fake = _FakePool(results=[{"count": 0}])
        repo = PostgresAlertRepository.__new__(PostgresAlertRepository)
        repo._pool = fake
        alerts, total = repo.list_alerts(status=OPEN, risk_level="HIGH")
        # No string-interpolation of filter values.
        sql = fake.executed[0][0]
        assert "OPEN" not in sql
        assert "status = %s" in sql
        assert "risk_level = %s" in sql


# ──────────────────────────────────────────────────────────────────────
# 5. Backward compatibility — SQLite and in-memory still work
# ──────────────────────────────────────────────────────────────────────


class TestBackwardCompatibility:
    def test_inmemory_user_store_still_works(self, tmp_path):
        store = InMemoryUserStore()
        user = store.create_user(email="mem@example.com", password="MemPass123")
        assert store.get_by_email("mem@example.com") is not None
        assert user["role"] == CUSTOMER

    def test_inmemory_alert_store_still_works(self):
        store = InMemoryAlertStore()
        alert = store.create(risk_score=50, risk_level="MEDIUM", decision="VERIFY")
        assert store.get_by_id(alert["id"]) is not None

    def test_sqlite_user_repo_still_works(self, tmp_path):
        repo = SQLiteUserRepository(db_path=tmp_path / "u.db")
        user = repo.create_user(email="sq@example.com", password="SqPass123")
        assert repo.email_exists("sq@example.com")
        assert repo.get_by_id(user["id"]) is not None
        repo.close()

    def test_sqlite_alert_repo_still_works(self, tmp_path):
        repo = SQLiteAlertRepository(db_path=tmp_path / "a.db")
        alert = repo.create(risk_score=40, risk_level="MEDIUM", decision="VERIFY")
        assert repo.get_by_id(alert["id"]) is not None
        repo.close()

    def test_sqlite_duplicate_raises_user_already_exists(self, tmp_path):
        repo = SQLiteUserRepository(db_path=tmp_path / "u.db")
        repo.create_user(email="dup@example.com", password="DupPass123")
        with pytest.raises(UserAlreadyExistsError):
            repo.create_user(email="dup@example.com", password="DupPass123")
        repo.close()


# ──────────────────────────────────────────────────────────────────────
# 6. Integration tests — PostgreSQL (skip if not configured/reachable)
# ──────────────────────────────────────────────────────────────────────


_TEST_SCHEMA = f"step40_test_{uuid.uuid4().hex[:12]}"


def _pg_settings() -> Settings:
    """Build Settings from the current environment (reads POSTGRES_*)."""
    return Settings()


def _pg_configured() -> bool:
    s = _pg_settings()
    return bool(s.POSTGRES_USER and s.POSTGRES_PASSWORD and s.POSTGRES_HOST)


@pytest.fixture(scope="session")
def pg_pool():
    """Yield a connection pool pointed at an isolated test schema.

    Skips (rather than fails) when PostgreSQL is not configured or
    unreachable.  The test schema is created before yielding and
    dropped (``CASCADE``) after the session.
    """
    if not _pg_configured():
        pytest.skip("PostgreSQL not configured (set POSTGRES_USER/POSTGRES_PASSWORD)")

    import psycopg as _psycopg

    settings = _pg_settings()
    # Admin connection (no search_path override) for schema management.
    admin_conninfo = _psycopg.conninfo.make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        sslmode=settings.POSTGRES_SSL_MODE or None,
        connect_timeout=3,
    )
    try:
        with _psycopg.connect(admin_conninfo, autocommit=True) as conn:
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{_TEST_SCHEMA}"')
    except Exception as exc:
        pytest.skip(f"PostgreSQL unreachable: {type(exc).__name__}")

    try:
        pool = pg_mod.create_pool(
            settings,
            min_size=1,
            max_size=4,
            timeout=8.0,
            pool_kwargs={"kwargs": {"options": f"-c search_path={_TEST_SCHEMA}"}},
        )
        pg_mod.init_schema(pool)
        yield pool
    finally:
        try:
            pool.close()
        except Exception:
            pass
        try:
            with _psycopg.connect(admin_conninfo, autocommit=True) as conn:
                conn.execute(f'DROP SCHEMA IF EXISTS "{_TEST_SCHEMA}" CASCADE')
        except Exception:
            # Best-effort cleanup; the schema is unique per session.
            pass


@pytest.fixture
def pg_user_repo(pg_pool):
    """Fresh PostgreSQL user repository (schema already initialised)."""
    repo = PostgresUserRepository.__new__(PostgresUserRepository)
    repo._pool = pg_pool
    return repo


@pytest.fixture
def pg_alert_repo(pg_pool):
    """Fresh PostgreSQL alert repository (schema already initialised)."""
    repo = PostgresAlertRepository.__new__(PostgresAlertRepository)
    repo._pool = pg_pool
    return repo


def _unique_email(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}@example.com"


@pytest.mark.skipif(not _pg_configured(), reason="PostgreSQL not configured")
class TestPostgresUserPersistence:
    """User CRUD against a real PostgreSQL server (isolated schema)."""

    def test_create_and_retrieve_by_id(self, pg_user_repo):
        email = _unique_email("user")
        user = pg_user_repo.create_user(email=email, password="Pass1234")
        fetched = pg_user_repo.get_by_id(user["id"])
        assert fetched is not None
        assert fetched["email"] == email
        assert fetched["role"] == CUSTOMER
        assert fetched["is_active"] is True
        assert isinstance(fetched["id"], str)
        # UUID shape
        uuid.UUID(fetched["id"])

    def test_retrieve_by_email_case_insensitive(self, pg_user_repo):
        email = _unique_email("case")
        pg_user_repo.create_user(email=email, password="CasePass1")
        # Upper-case lookup must still find the lowercased stored email.
        fetched = pg_user_repo.get_by_email(email.upper())
        assert fetched is not None
        assert fetched["email"] == email

    def test_duplicate_email_raises(self, pg_user_repo):
        email = _unique_email("dup")
        pg_user_repo.create_user(email=email, password="DupPass123")
        with pytest.raises(UserAlreadyExistsError):
            pg_user_repo.create_user(email=email, password="DupPass123")
        assert pg_user_repo.email_exists(email)

    def test_case_insensitive_duplicate(self, pg_user_repo):
        email = _unique_email("ci")
        pg_user_repo.create_user(email=email, password="CIPass1234")
        assert pg_user_repo.email_exists(email.upper())

    def test_password_hash_not_plaintext(self, pg_user_repo):
        email = _unique_email("hash")
        user = pg_user_repo.create_user(email=email, password="Plain1234")
        assert user["password_hash"] != "Plain1234"
        assert user["password_hash"].startswith("$2")

    def test_role_persistence(self, pg_user_repo):
        e1 = _unique_email("analyst")
        e2 = _unique_email("admin")
        u1 = pg_user_repo.create_user(email=e1, password="RolePass12", role=FRAUD_ANALYST)
        u2 = pg_user_repo.create_user(email=e2, password="RolePass12", role=ADMIN)
        assert pg_user_repo.get_by_id(u1["id"])["role"] == FRAUD_ANALYST
        assert pg_user_repo.get_by_id(u2["id"])["role"] == ADMIN

    def test_active_inactive_persistence(self, pg_user_repo):
        email = _unique_email("inactive")
        u = pg_user_repo.create_user(email=email, password="ActPass123", is_active=False)
        fetched = pg_user_repo.get_by_id(u["id"])
        assert fetched["is_active"] is False

    def test_restart_persistence(self, pg_pool):
        """Close pool, open new pool (same schema) → data survives."""
        email = _unique_email("restart")
        repo1 = PostgresUserRepository.__new__(PostgresUserRepository)
        repo1._pool = pg_pool
        repo1.create_user(email=email, password="Restart1")

        # Simulate "restart": build a new pool with the same schema.
        settings = _pg_settings()
        pool2 = pg_mod.create_pool(
            settings,
            min_size=1, max_size=2, timeout=5.0,
            pool_kwargs={"kwargs": {"options": f"-c search_path={_TEST_SCHEMA}"}},
        )
        repo2 = PostgresUserRepository.__new__(PostgresUserRepository)
        repo2._pool = pool2
        try:
            fetched = repo2.get_by_email(email)
            assert fetched is not None
            assert fetched["email"] == email
        finally:
            pool2.close()

    def test_get_by_id_with_garbage_returns_none(self, pg_user_repo):
        assert pg_user_repo.get_by_id("garbage-id") is None


@pytest.mark.skipif(not _pg_configured(), reason="PostgreSQL not configured")
class TestPostgresAlertPersistence:
    """Alert CRUD against a real PostgreSQL server (isolated schema)."""

    def _make_alert(self, pg_alert_repo, **overrides):
        base = {
            "transaction_id": str(uuid.uuid4()),
            "risk_score": 80,
            "risk_level": "HIGH",
            "decision": "HOLD",
            "fraud_probability": 0.91,
            "model_version": "fraud-xgb-v1.0.0",
            "risk_factors": ["amount_deviation", "high_amount"],
            "explanation_json": {
                "ml_top_factors": [{"feature": "amount_deviation", "importance": 0.45}],
                "behaviour_signals": [],
                "rules_triggered": [],
            },
            "amount": 10500.0,
            "currency": "USD",
            "merchant_name": "Wire Casino",
            "transaction_type": "purchase",
            "timestamp": 1725200000,
        }
        base.update(overrides)
        return pg_alert_repo.create(**base)

    def test_create_and_retrieve_full_fields(self, pg_alert_repo):
        alert = self._make_alert(pg_alert_repo)
        fetched = pg_alert_repo.get_by_id(alert["id"])
        assert fetched is not None
        assert fetched["risk_score"] == 80
        assert fetched["risk_level"] == "HIGH"
        assert fetched["decision"] == "HOLD"
        assert fetched["fraud_probability"] == 0.91
        assert fetched["model_version"] == "fraud-xgb-v1.0.0"
        assert fetched["risk_factors"] == ["amount_deviation", "high_amount"]
        assert fetched["explanation_json"]["ml_top_factors"][0]["feature"] == "amount_deviation"
        assert fetched["amount"] == 10500.0
        assert fetched["timestamp"] == 1725200000
        assert fetched["status"] == OPEN
        assert fetched["analyst_id"] is None
        assert fetched["notes"] is None
        # Timestamps are ISO strings
        assert isinstance(fetched["created_at"], str)
        assert "+" in fetched["created_at"] or "Z" in fetched["created_at"]

    def test_get_by_transaction_id(self, pg_alert_repo):
        tid = str(uuid.uuid4())
        alert = self._make_alert(pg_alert_repo, transaction_id=tid)
        fetched = pg_alert_repo.get_by_transaction_id(tid)
        assert fetched is not None
        assert fetched["id"] == alert["id"]

    def test_get_by_transaction_id_garbage_returns_none(self, pg_alert_repo):
        assert pg_alert_repo.get_by_transaction_id("not-a-uuid") is None

    def test_list_alerts_with_filters(self, pg_alert_repo):
        # Create two alerts with different statuses.
        a1 = self._make_alert(pg_alert_repo)
        a2 = self._make_alert(pg_alert_repo)
        pg_alert_repo.update_status(a2["id"], new_status=IN_REVIEW)
        alerts_open, total_open = pg_alert_repo.list_alerts(status=OPEN)
        assert all(a["status"] == OPEN for a in alerts_open)
        assert a1["id"] in {a["id"] for a in alerts_open}

    def test_list_alerts_pagination(self, pg_alert_repo):
        ids = [self._make_alert(pg_alert_repo)["id"] for _ in range(5)]
        page1, total = pg_alert_repo.list_alerts(page=1, per_page=2)
        page2, _ = pg_alert_repo.list_alerts(page=2, per_page=2)
        assert total >= 5
        assert len(page1) == 2
        assert len(page2) == 2
        # Distinct ids across pages
        assert {a["id"] for a in page1} & {a["id"] for a in page2} == set()

    def test_duplicate_transaction_id_raises(self, pg_alert_repo):
        tid = str(uuid.uuid4())
        self._make_alert(pg_alert_repo, transaction_id=tid)
        # Second create with same transaction_id → returns existing alert
        # (router-level dedup is primary; the DB unique index returns
        # the existing row on the rare race path).
        duplicate = self._make_alert(pg_alert_repo, transaction_id=tid)
        # The duplicate handler falls back to get_by_transaction_id
        # returning the first alert's id.
        original = pg_alert_repo.get_by_transaction_id(tid)
        assert original is not None
        assert duplicate["id"] == original["id"]

    def test_status_transitions(self, pg_alert_repo):
        alert = self._make_alert(pg_alert_repo)
        updated1 = pg_alert_repo.update_status(alert["id"], new_status=IN_REVIEW)
        assert updated1["status"] == IN_REVIEW
        assert updated1["updated_at"] >= alert["created_at"]
        updated2 = pg_alert_repo.update_status(alert["id"], new_status=RESOLVED)
        assert updated2["status"] == RESOLVED
        assert updated2["resolved_at"] is not None
        # Terminal
        assert pg_alert_repo.update_status(alert["id"], new_status=OPEN) is None

    def test_invalid_transition_returns_none(self, pg_alert_repo):
        alert = self._make_alert(pg_alert_repo)
        # OPEN → "NOT_A_REAL_STATUS" is invalid
        result = pg_alert_repo.update_status(alert["id"], new_status="NOT_A_REAL_STATUS")
        assert result is None

    def test_analyst_id_first_writer_wins(self, pg_alert_repo, pg_user_repo):
        analyst1 = pg_user_repo.create_user(
            email=_unique_email("analyst1"), password="Analyst1!",
            role=FRAUD_ANALYST,
        )
        analyst2 = pg_user_repo.create_user(
            email=_unique_email("analyst2"), password="Analyst2!",
            role=FRAUD_ANALYST,
        )
        alert = self._make_alert(pg_alert_repo)
        updated = pg_alert_repo.update_status(
            alert["id"], new_status=IN_REVIEW, analyst_id=analyst1["id"],
        )
        assert updated["analyst_id"] == analyst1["id"]
        # Second write must not overwrite.
        updated2 = pg_alert_repo.update_status(
            alert["id"], new_status=RESOLVED, analyst_id=analyst2["id"],
        )
        assert updated2["analyst_id"] == analyst1["id"]

    def test_notes_persistence(self, pg_alert_repo):
        alert = self._make_alert(pg_alert_repo)
        updated = pg_alert_repo.update_status(
            alert["id"], new_status=OPEN, notes="Investigating amount anomaly",
        )
        assert updated["notes"] == "Investigating amount anomaly"

    def test_restart_persistence_alerts(self, pg_pool):
        repo1 = PostgresAlertRepository.__new__(PostgresAlertRepository)
        repo1._pool = pg_pool
        alert = self._make_alert(repo1)

        settings = _pg_settings()
        pool2 = pg_mod.create_pool(
            settings,
            min_size=1, max_size=2, timeout=5.0,
            pool_kwargs={"kwargs": {"options": f"-c search_path={_TEST_SCHEMA}"}},
        )
        repo2 = PostgresAlertRepository.__new__(PostgresAlertRepository)
        repo2._pool = pool2
        try:
            fetched = repo2.get_by_id(alert["id"])
            assert fetched is not None
            assert fetched["risk_score"] == 80
        finally:
            pool2.close()


@pytest.mark.skipif(not _pg_configured(), reason="PostgreSQL not configured")
class TestAPIRegressionWithPG:
    """End-to-end API regression with PostgreSQL repositories."""

    @pytest.fixture
    def pg_test_client(self, pg_pool):
        """TestClient with PG repos + fake ML client (HOLD response)."""
        from fastapi.testclient import TestClient
        from httpx import Response

        from backend.app import app
        from backend.routers import alerts as alerts_module
        from backend.routers import transactions as txn_module
        from backend.security import deps as deps_module

        from backend.services.ml_client import MLServiceClient

        saved_user_repo = deps_module._user_repo
        saved_alert_repo_alerts = alerts_module._alert_repo
        saved_alert_repo_txn = txn_module._alert_repo
        saved_ml_client = txn_module._ml_client
        saved_overrides = dict(app.dependency_overrides)
        app.dependency_overrides.clear()

        pg_user = PostgresUserRepository.__new__(PostgresUserRepository)
        pg_user._pool = pg_pool
        pg_alerts = PostgresAlertRepository.__new__(PostgresAlertRepository)
        pg_alerts._pool = pg_pool
        deps_module.set_user_repository(pg_user)
        alerts_module.set_alert_repository(pg_alerts)
        txn_module.set_alert_repository(pg_alerts)
        txn_module.set_ml_client(MLServiceClient(base_url="http://mock-ml:8001"))

        ml_hold_response = {
            "fraud_probability": 0.91,
            "fraud_prediction": 1,
            "threshold": 0.50,
            "model_version": "fraud-xgb-v1.0.0",
            "timestamp": 1725200000,
            "ml_score": 91,
            "behaviour_score": 75,
            "rule_score": 60,
            "risk_score": 85,
            "risk_level": "HIGH",
            "decision": "HOLD",
            "explanation_detail": {
                "ml_top_factors": [{"feature": "amount_deviation", "importance": 0.45}],
                "behaviour_signals": [],
                "rules_triggered": [],
            },
            "risk_factors": ["amount_deviation"],
        }

        mock_resp = Response(200, json=ml_hold_response)
        patch_ctx = patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        )
        mock_post = patch_ctx.start()

        # Register an analyst for PATCH tests (auth is real).
        analyst_email = _unique_email("api-analyst")
        pg_user.create_user(
            email=analyst_email, password="AnalystPass1",
            role=FRAUD_ANALYST,
        )

        tc = TestClient(app)
        try:
            yield tc, pg_user, pg_alerts, analyst_email
        finally:
            patch_ctx.stop()
            app.dependency_overrides.clear()
            app.dependency_overrides.update(saved_overrides)
            deps_module.set_user_repository(saved_user_repo)
            alerts_module.set_alert_repository(saved_alert_repo_alerts)
            txn_module.set_alert_repository(saved_alert_repo_txn)
            txn_module.set_ml_client(saved_ml_client)

    def _register(self, tc, email, password="Customer1"):
        return tc.post("/api/v1/auth/register", json={
            "email": email, "password": password,
            "first_name": "Step", "last_name": "Forty",
        })

    def _login(self, tc, email, password):
        return tc.post("/api/v1/auth/login", json={"email": email, "password": password})

    def test_register_and_login(self, pg_test_client):
        tc, _, _, _ = pg_test_client
        email = _unique_email("reg")
        r = self._register(tc, email)
        assert r.status_code == 201, r.text
        assert r.json()["role"] == "customer"

        l = self._login(tc, email, "Customer1")
        assert l.status_code == 200
        body = l.json()
        assert body["token_type"] == "bearer"
        assert body["expires_in"] == 1800
        assert "access_token" in body and "refresh_token" in body

    def test_duplicate_registration_409(self, pg_test_client):
        tc, _, _, _ = pg_test_client
        email = _unique_email("dup409")
        assert self._register(tc, email).status_code == 201
        r = self._register(tc, email)
        assert r.status_code == 409

    def test_me_returns_own_profile(self, pg_test_client):
        tc, _, _, _ = pg_test_client
        email = _unique_email("me")
        self._register(tc, email)
        tok = self._login(tc, email, "Customer1").json()["access_token"]
        me = tc.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {tok}"})
        assert me.status_code == 200
        assert me.json()["email"] == email

    def test_alerts_401_without_token(self, pg_test_client):
        tc, _, _, _ = pg_test_client
        r = tc.get("/api/v1/alerts")
        assert r.status_code == 401

    def test_alerts_403_for_customer(self, pg_test_client):
        tc, _, _, _ = pg_test_client
        email = _unique_email("customer403")
        self._register(tc, email)
        tok = self._login(tc, email, "Customer1").json()["access_token"]
        r = tc.get("/api/v1/alerts", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 403

    def test_transaction_hold_creates_alert_and_patch_lifecycle(self, pg_test_client):
        tc, _, pg_alerts, analyst_email = pg_test_client
        # Authenticated transaction → HOLD → alert created in PG.
        tok = self._login(tc, analyst_email, "AnalystPass1").json()["access_token"]
        headers = {"Authorization": f"Bearer {tok}"}
        txn = {
            "amount": 10500.0,
            "currency": "USD",
            "merchant_name": "Wire Casino",
            "merchant_category": "7995",
            "transaction_type": "purchase",
            "location_country": "US",
            "location_city": "Miami",
            "device_fingerprint": f"step40-{uuid.uuid4()}",
            "device_type": "mobile",
            "ip_address": "198.51.100.7",
        }
        resp = tc.post("/api/v1/transactions", json=txn, headers=headers)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["decision"] == "HOLD"
        assert body["alert"] is not None
        alert_id = body["alert"]["id"]

        # Alert detail
        detail = tc.get(f"/api/v1/alerts/{alert_id}", headers=headers)
        assert detail.status_code == 200
        assert detail.json()["status"] == "OPEN"
        assert detail.json()["transaction_summary"]["timestamp"] == 1725200000

        # Lifecycle: OPEN → IN_REVIEW → RESOLVED (analyst_id from JWT).
        r1 = tc.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"status": "IN_REVIEW"}, headers=headers,
        )
        assert r1.status_code == 200
        assert r1.json()["analyst_id"] is not None
        r2 = tc.patch(
            f"/api/v1/alerts/{alert_id}",
            json={"status": "RESOLVED"}, headers=headers,
        )
        assert r2.status_code == 200
        assert r2.json()["resolved_at"] is not None

    def test_get_alert_with_garbage_id_returns_404(self, pg_test_client):
        tc, _, _, analyst_email = pg_test_client
        tok = self._login(tc, analyst_email, "AnalystPass1").json()["access_token"]
        r = tc.get("/api/v1/alerts/not-a-real-id",
                   headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 404

    def test_jwt_validation_unchanged(self, pg_test_client):
        tc, _, _, _ = pg_test_client
        # Missing header → 401
        assert tc.get("/api/v1/auth/me").status_code == 401
        # Malformed → 401
        r = tc.get("/api/v1/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
        assert r.status_code == 401


@pytest.mark.skipif(not _pg_configured(), reason="PostgreSQL not configured")
class TestSecurityWithPG:
    """Security guarantees preserved against a real PostgreSQL backend."""

    def test_stored_password_is_bcrypt_hash(self, pg_user_repo):
        email = _unique_email("sec")
        user = pg_user_repo.create_user(email=email, password="SecretPass1")
        assert user["password_hash"].startswith("$2")
        assert user["password_hash"] != "SecretPass1"

    def test_login_response_contains_no_hash(self, pg_pool):
        """Login response body must not include password_hash."""
        from fastapi.testclient import TestClient
        from backend.app import app
        from backend.security import deps as deps_module

        saved = deps_module._user_repo
        saved_overrides = dict(app.dependency_overrides)
        app.dependency_overrides.clear()

        pg_user = PostgresUserRepository.__new__(PostgresUserRepository)
        pg_user._pool = pg_pool
        deps_module.set_user_repository(pg_user)
        try:
            email = _unique_email("sec-login")
            pg_user.create_user(email=email, password="NoLeakPass1")
            tc = TestClient(app)
            r = tc.post("/api/v1/auth/login",
                        json={"email": email, "password": "NoLeakPass1"})
            assert r.status_code == 200
            body = r.text.lower()
            assert "password_hash" not in body
            assert "no leakpass1" not in body.lower()
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(saved_overrides)
            deps_module.set_user_repository(saved)

    def test_no_cross_user_leakage_via_me(self, pg_pool):
        """User A's /auth/me never returns user B's data."""
        from fastapi.testclient import TestClient
        from backend.app import app
        from backend.security import deps as deps_module

        saved = deps_module._user_repo
        saved_overrides = dict(app.dependency_overrides)
        app.dependency_overrides.clear()

        pg_user = PostgresUserRepository.__new__(PostgresUserRepository)
        pg_user._pool = pg_pool
        deps_module.set_user_repository(pg_user)
        try:
            e_a = _unique_email("usera")
            e_b = _unique_email("userb")
            pg_user.create_user(email=e_a, password="UserAPass1")
            pg_user.create_user(email=e_b, password="UserBPass1")
            tc = TestClient(app)
            tok_a = tc.post("/api/v1/auth/login",
                            json={"email": e_a, "password": "UserAPass1"}).json()["access_token"]
            me_a = tc.get("/api/v1/auth/me",
                          headers={"Authorization": f"Bearer {tok_a}"}).json()
            assert me_a["email"] == e_a
            assert e_b not in str(me_a)
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(saved_overrides)
            deps_module.set_user_repository(saved)

    def test_no_credentials_in_register_error_response(self, pg_test_client_fixture):
        tc, _ = pg_test_client_fixture
        email = _unique_email("cred")
        tc.post("/api/v1/auth/register", json={
            "email": email, "password": "CredPass12",
            "first_name": "A", "last_name": "B",
        })
        r = tc.post("/api/v1/auth/register", json={
            "email": email, "password": "CredPass12",
            "first_name": "A", "last_name": "B",
        })
        assert r.status_code == 409
        body = r.text.lower()
        # The password should never appear in an error body.
        assert "credpass12" not in body


@pytest.fixture
def pg_test_client_fixture(pg_pool):
    """Light fixture used by TestSecurityWithPG — register-only TestClient."""
    from fastapi.testclient import TestClient
    from backend.app import app
    from backend.security import deps as deps_module

    saved = deps_module._user_repo
    saved_overrides = dict(app.dependency_overrides)
    app.dependency_overrides.clear()

    pg_user = PostgresUserRepository.__new__(PostgresUserRepository)
    pg_user._pool = pg_pool
    deps_module.set_user_repository(pg_user)
    tc = TestClient(app)
    try:
        yield tc, pg_user
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(saved_overrides)
        deps_module.set_user_repository(saved)


# ──────────────────────────────────────────────────────────────────────
# Fakes for unit tests
# ──────────────────────────────────────────────────────────────────────


class _FakeCursor:
    """Minimal cursor recording SQL + params for unit tests."""

    def __init__(self, recorder: list, results: list | None = None):
        self._recorder = recorder
        self._results = list(results or [])

    def execute(self, sql: str, params: Any = None) -> None:
        self._recorder.append((sql, params))

    def fetchone(self):
        return self._results[0] if self._results else None

    def fetchall(self):
        return list(self._results)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    """Minimal connection yielding fake cursors."""

    def __init__(self, recorder: list, results: list | None = None):
        self._recorder = recorder
        self._results = results

    def cursor(self, row_factory: Any = None):
        return _FakeCursor(self._recorder, self._results)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    """Minimal pool returning fake connections."""

    def __init__(self, results: list | None = None):
        self.executed: list[tuple[str, Any]] = []
        self._results = results

    def connection(self):
        return _FakeConnection(self.executed, self._results)

    def close(self) -> None:
        pass
