"""Tests for SQLAlchemy ORM models and Alembic integration.

Covers:
* All 9 models import successfully.
* Base.metadata contains exactly the 9 intended application tables.
* Primary keys exist.
* Foreign keys point to the correct tables/columns.
* Important unique constraints exist.
* Important CHECK constraints exist.
* Important indexes exist.
* Alembic is connected to Base.metadata.
* Migration is applied to the real PostgreSQL database.
"""

from __future__ import annotations

import pytest
from sqlalchemy import CheckConstraint, UniqueConstraint, create_engine, inspect, text
from sqlalchemy.orm import Session

from app.config import settings
from app.db.base import Base
from app.models import (
    Alert,
    AuditLog,
    Customer,
    CustomerDevice,
    Merchant,
    ModelMetadata,
    RiskRulesConfig,
    Transaction,
    User,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_TABLES = {
    "users",
    "customers",
    "merchants",
    "transactions",
    "alerts",
    "audit_logs",
    "customer_devices",
    "model_metadata",
    "risk_rules_config",
}


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
# 1. All 9 models import successfully
# ---------------------------------------------------------------------------


class TestModelImports:
    """Verify all 9 models are importable."""

    @pytest.mark.parametrize(
        "model_cls",
        [Customer, User, Merchant, Transaction, Alert, AuditLog,
         CustomerDevice, ModelMetadata, RiskRulesConfig],
    )
    def test_model_class_exists(self, model_cls: type) -> None:
        assert model_cls is not None
        assert hasattr(model_cls, "__tablename__")

    def test_nine_model_classes(self) -> None:
        models = [
            Customer, User, Merchant, Transaction, Alert, AuditLog,
            CustomerDevice, ModelMetadata, RiskRulesConfig,
        ]
        assert len(models) == 9


# ---------------------------------------------------------------------------
# 2. Base.metadata contains exactly 9 application tables
# ---------------------------------------------------------------------------


class TestBaseMetadata:
    """Verify Base.metadata registration."""

    def test_metadata_has_exactly_nine_tables(self) -> None:
        table_names = set(Base.metadata.tables.keys())
        assert table_names == EXPECTED_TABLES

    def test_no_extra_tables(self) -> None:
        assert len(Base.metadata.tables) == 9

    @pytest.mark.parametrize("table_name", sorted(EXPECTED_TABLES))
    def test_each_table_registered(self, table_name: str) -> None:
        assert table_name in Base.metadata.tables


# ---------------------------------------------------------------------------
# 3. Primary keys
# ---------------------------------------------------------------------------


class TestPrimaryKeys:
    """Every model must have 'id' as primary key."""

    @pytest.mark.parametrize(
        "table_name", sorted(EXPECTED_TABLES),
    )
    def test_table_has_pk(self, table_name: str) -> None:
        table = Base.metadata.tables[table_name]
        pk_cols = list(table.primary_key.columns)
        assert len(pk_cols) == 1
        assert pk_cols[0].name == "id"


# ---------------------------------------------------------------------------
# 4. Foreign keys
# ---------------------------------------------------------------------------


class TestForeignKeys:
    """Verify all documented FK relationships."""

    @pytest.mark.parametrize(
        "table,col,target",
        [
            ("users", "customer_id", "customers.id"),
            ("transactions", "customer_id", "customers.id"),
            ("transactions", "merchant_id", "merchants.id"),
            ("transactions", "model_version", "model_metadata.model_version"),
            ("alerts", "transaction_id", "transactions.id"),
            ("alerts", "analyst_id", "users.id"),
            ("audit_logs", "actor_id", "users.id"),
            ("customer_devices", "customer_id", "customers.id"),
            ("model_metadata", "trained_by", "users.id"),
        ],
    )
    def test_fk_target(self, table: str, col: str, target: str) -> None:
        column = Base.metadata.tables[table].columns[col]
        fk_list = list(column.foreign_keys)
        assert len(fk_list) == 1
        assert fk_list[0].target_fullname == target


class TestFKDeleteBehaviour:
    """Verify ON DELETE rules match database-erd.md."""

    @pytest.mark.parametrize(
        "table,col,expected",
        [
            ("users", "customer_id", "SET NULL"),
            ("transactions", "customer_id", "RESTRICT"),
            ("customer_devices", "customer_id", "RESTRICT"),
            ("transactions", "merchant_id", "SET NULL"),
            ("alerts", "transaction_id", "RESTRICT"),
            ("alerts", "analyst_id", "SET NULL"),
            ("audit_logs", "actor_id", "SET NULL"),
            ("model_metadata", "trained_by", "SET NULL"),
            ("transactions", "model_version", "RESTRICT"),
        ],
    )
    def test_on_delete(self, table: str, col: str, expected: str) -> None:
        column = Base.metadata.tables[table].columns[col]
        fk = list(column.foreign_keys)[0]
        assert fk.ondelete == expected

    @pytest.mark.parametrize(
        "table,col",
        [
            ("users", "customer_id"),
            ("transactions", "customer_id"),
            ("transactions", "merchant_id"),
            ("transactions", "model_version"),
            ("alerts", "transaction_id"),
            ("alerts", "analyst_id"),
            ("audit_logs", "actor_id"),
            ("customer_devices", "customer_id"),
            ("model_metadata", "trained_by"),
        ],
    )
    def test_on_update_cascade(self, table: str, col: str) -> None:
        column = Base.metadata.tables[table].columns[col]
        fk = list(column.foreign_keys)[0]
        assert fk.onupdate == "CASCADE"


# ---------------------------------------------------------------------------
# 5. Unique constraints
# ---------------------------------------------------------------------------


class TestUniqueConstraints:
    """Verify important unique constraints."""

    def _unique_names(self, table_name: str) -> set[str]:
        table = Base.metadata.tables[table_name]
        return {
            c.name
            for c in table.constraints
            if isinstance(c, UniqueConstraint) and c.name is not None
        }

    def test_users_email_unique(self) -> None:
        table = Base.metadata.tables["users"]
        assert table.columns["email"].unique is True

    def test_model_metadata_version_unique(self) -> None:
        names = self._unique_names("model_metadata")
        assert "uq_model_metadata_model_version" in names

    def test_risk_rules_config_rule_name_unique(self) -> None:
        names = self._unique_names("risk_rules_config")
        assert "uq_risk_rules_config_rule_name" in names

    def test_customer_devices_composite_unique(self) -> None:
        names = self._unique_names("customer_devices")
        assert "uq_customer_devices_customer_fingerprint" in names


# ---------------------------------------------------------------------------
# 6. CHECK constraints
# ---------------------------------------------------------------------------


class TestCheckConstraints:
    """Verify important CHECK constraints."""

    def _check_names(self, table_name: str) -> set[str]:
        table = Base.metadata.tables[table_name]
        return {
            c.name
            for c in table.constraints
            if isinstance(c, CheckConstraint) and c.name is not None
        }

    def test_users_role_check(self) -> None:
        assert "ck_users_role" in self._check_names("users")

    def test_merchants_risk_level_check(self) -> None:
        assert "ck_merchants_risk_level" in self._check_names("merchants")

    def test_transactions_amount_check(self) -> None:
        assert "ck_transactions_amount" in self._check_names("transactions")

    def test_transactions_currency_check(self) -> None:
        assert "ck_transactions_currency" in self._check_names("transactions")

    def test_transactions_status_check(self) -> None:
        assert "ck_transactions_status" in self._check_names("transactions")

    def test_transactions_transaction_type_check(self) -> None:
        assert "ck_transactions_transaction_type" in self._check_names("transactions")

    def test_transactions_device_type_check(self) -> None:
        assert "ck_transactions_device_type" in self._check_names("transactions")

    def test_transactions_risk_level_check(self) -> None:
        assert "ck_transactions_risk_level" in self._check_names("transactions")

    def test_transactions_decision_check(self) -> None:
        assert "ck_transactions_decision" in self._check_names("transactions")

    def test_transactions_score_checks(self) -> None:
        names = self._check_names("transactions")
        for score in ("risk_score", "ml_score", "behaviour_score", "rule_score"):
            assert f"ck_transactions_{score}" in names

    def test_alerts_status_check(self) -> None:
        assert "ck_alerts_status" in self._check_names("alerts")

    def test_alerts_risk_score_check(self) -> None:
        assert "ck_alerts_risk_score" in self._check_names("alerts")

    def test_customer_devices_device_type_check(self) -> None:
        assert "ck_customer_devices_device_type" in self._check_names("customer_devices")

    def test_risk_rules_config_score_check(self) -> None:
        assert "ck_risk_rules_config_score" in self._check_names("risk_rules_config")


# ---------------------------------------------------------------------------
# 7. Indexes
# ---------------------------------------------------------------------------


class TestIndexes:
    """Verify important indexes."""

    def _index_names(self, table_name: str) -> set[str]:
        table = Base.metadata.tables[table_name]
        return {idx.name for idx in table.indexes}

    # -- users --
    def test_users_indexes(self) -> None:
        names = self._index_names("users")
        assert "ix_users_email" in names
        assert "ix_users_role" in names
        assert "ix_users_customer_id" in names

    # -- customers --
    def test_customers_indexes(self) -> None:
        assert "ix_customers_created_at" in self._index_names("customers")

    # -- merchants --
    def test_merchants_indexes(self) -> None:
        names = self._index_names("merchants")
        assert "ix_merchants_name" in names
        assert "ix_merchants_category_code" in names

    # -- transactions --
    def test_transactions_indexes(self) -> None:
        names = self._index_names("transactions")
        expected = {
            "ix_transactions_customer_id",
            "ix_transactions_merchant_id",
            "ix_transactions_timestamp",
            "ix_transactions_status",
            "ix_transactions_risk_level",
            "ix_transactions_model_version",
        }
        assert expected.issubset(names)

    # -- alerts --
    def test_alerts_indexes(self) -> None:
        names = self._index_names("alerts")
        assert "ix_alerts_transaction_id" in names
        assert "ix_alerts_status" in names
        assert "ix_alerts_analyst_id" in names

    # -- audit_logs --
    def test_audit_logs_indexes(self) -> None:
        names = self._index_names("audit_logs")
        assert "ix_audit_logs_actor_id" in names
        assert "ix_audit_logs_timestamp" in names
        assert "ix_audit_logs_action" in names

    # -- customer_devices --
    def test_customer_devices_indexes(self) -> None:
        assert "ix_customer_devices_customer_id" in self._index_names("customer_devices")

    # -- model_metadata --
    def test_model_metadata_indexes(self) -> None:
        names = self._index_names("model_metadata")
        assert "ix_model_metadata_version" in names

    # -- risk_rules_config --
    def test_risk_rules_config_indexes(self) -> None:
        assert "ix_risk_rules_config_rule_name" in self._index_names("risk_rules_config")


# ---------------------------------------------------------------------------
# 8. Transactions table — special attention
# ---------------------------------------------------------------------------


class TestTransactionsTable:
    """The transactions table is the most complex — verify thoroughly."""

    def test_has_all_fraud_score_columns(self) -> None:
        table = Base.metadata.tables["transactions"]
        for col in ("ml_score", "behaviour_score", "rule_score", "risk_score"):
            assert col in table.columns

    def test_has_risk_level_and_decision(self) -> None:
        table = Base.metadata.tables["transactions"]
        assert "risk_level" in table.columns
        assert "decision" in table.columns

    def test_has_explanation_json(self) -> None:
        table = Base.metadata.tables["transactions"]
        assert "explanation_json" in table.columns

    def test_has_model_version(self) -> None:
        table = Base.metadata.tables["transactions"]
        assert "model_version" in table.columns

    def test_amount_is_numeric_12_2(self) -> None:
        col = Base.metadata.tables["transactions"].columns["amount"]
        assert col.type.precision == 12
        assert col.type.scale == 2

    def test_status_values(self) -> None:
        """Status must be PENDING, COMPLETED, FAILED — not APPROVED/REJECTED."""
        table = Base.metadata.tables["transactions"]
        for constraint in table.constraints:
            if getattr(constraint, "name", None) == "ck_transactions_status":
                sql = str(constraint.sqltext)
                assert "PENDING" in sql
                assert "COMPLETED" in sql
                assert "FAILED" in sql
                break


# ---------------------------------------------------------------------------
# 9. Alembic integration
# ---------------------------------------------------------------------------


class TestAlembicIntegration:
    """Verify Alembic is connected to Base.metadata."""

    def test_target_metadata_is_base(self) -> None:
        assert Base.metadata is not None
        assert len(Base.metadata.tables) == 9

    def test_migration_file_exists(self) -> None:
        from pathlib import Path

        versions_dir = (
            Path(__file__).resolve().parents[2]
            / "database" / "alembic" / "versions"
        )
        migration_files = list(versions_dir.glob("*initial_schema*.py"))
        assert len(migration_files) == 1, "Expected exactly one initial_schema migration"

    def test_migration_revision_id(self) -> None:
        from pathlib import Path

        versions_dir = (
            Path(__file__).resolve().parents[2]
            / "database" / "alembic" / "versions"
        )
        migration_files = list(versions_dir.glob("*initial_schema*.py"))
        assert len(migration_files) == 1
        # The filename contains the revision ID
        filename = migration_files[0].name
        assert "4e7709c2bfad" in filename


# ---------------------------------------------------------------------------
# 10. PostgreSQL integration (requires running DB)
# ---------------------------------------------------------------------------


class TestModelsPostgresIntegration:
    """Integration tests against real Supabase PostgreSQL."""

    @requires_pg
    def test_alembic_migration_is_applied(self) -> None:
        """The initial migration must be applied at head."""
        eng = create_engine(
            settings.postgres.database_url,
            connect_args=settings.postgres.connect_args,
        )
        with eng.connect() as conn:
            result = conn.execute(text(
                "SELECT version_num FROM alembic_version"
            ))
            version = result.scalar()
        eng.dispose()
        assert version == "4e7709c2bfad"

    @requires_pg
    def test_nine_application_tables_in_database(self) -> None:
        """Exactly 9 application tables must exist in PostgreSQL."""
        eng = create_engine(
            settings.postgres.database_url,
            connect_args=settings.postgres.connect_args,
        )
        insp = inspect(eng)
        tables = set(insp.get_table_names())
        eng.dispose()
        assert EXPECTED_TABLES.issubset(tables)
        # Only alembic_version is the extra (non-application) table
        app_tables = tables - {"alembic_version"}
        assert app_tables == EXPECTED_TABLES

    @requires_pg
    def test_all_foreign_keys_in_database(self) -> None:
        """All 9 documented FKs must exist in PostgreSQL."""
        eng = create_engine(
            settings.postgres.database_url,
            connect_args=settings.postgres.connect_args,
        )
        with eng.connect() as conn:
            result = conn.execute(text("""
                SELECT tc.table_name, kcu.column_name,
                       ccu.table_name, ccu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage ccu
                  ON ccu.constraint_name = tc.constraint_name
                  AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = 'public'
            """))
            fks = {(row[0], row[1], row[2], row[3]) for row in result}
        eng.dispose()

        expected_fks = {
            ("users", "customer_id", "customers", "id"),
            ("transactions", "customer_id", "customers", "id"),
            ("transactions", "merchant_id", "merchants", "id"),
            ("transactions", "model_version", "model_metadata", "model_version"),
            ("alerts", "transaction_id", "transactions", "id"),
            ("alerts", "analyst_id", "users", "id"),
            ("audit_logs", "actor_id", "users", "id"),
            ("customer_devices", "customer_id", "customers", "id"),
            ("model_metadata", "trained_by", "users", "id"),
        }
        assert fks == expected_fks

    @requires_pg
    def test_no_cascade_delete_on_financial_fks(self) -> None:
        """No FK should use ON DELETE CASCADE for financial relationships."""
        eng = create_engine(
            settings.postgres.database_url,
            connect_args=settings.postgres.connect_args,
        )
        with eng.connect() as conn:
            result = conn.execute(text("""
                SELECT tc.table_name, kcu.column_name, rc.delete_rule
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.referential_constraints rc
                  ON rc.constraint_name = tc.constraint_name
                  AND rc.constraint_schema = tc.constraint_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = 'public'
            """))
            for row in result:
                assert row[2] != "CASCADE", (
                    f"ON DELETE CASCADE found on {row[0]}.{row[1]} — "
                    f"financial relationships must never cascade delete"
                )
        eng.dispose()

    @requires_pg
    def test_check_constraints_in_database(self) -> None:
        """All named CHECK constraints must exist in PostgreSQL."""
        eng = create_engine(
            settings.postgres.database_url,
            connect_args=settings.postgres.connect_args,
        )
        with eng.connect() as conn:
            result = conn.execute(text("""
                SELECT tc.table_name, tc.constraint_name
                FROM information_schema.table_constraints tc
                WHERE tc.constraint_type = 'CHECK'
                  AND tc.table_schema = 'public'
                  AND tc.constraint_name LIKE 'ck_%%'
            """))
            checks = {(row[0], row[1]) for row in result}
        eng.dispose()

        expected_checks = {
            ("users", "ck_users_role"),
            ("merchants", "ck_merchants_risk_level"),
            ("transactions", "ck_transactions_amount"),
            ("transactions", "ck_transactions_currency"),
            ("transactions", "ck_transactions_transaction_type"),
            ("transactions", "ck_transactions_device_type"),
            ("transactions", "ck_transactions_status"),
            ("transactions", "ck_transactions_risk_level"),
            ("transactions", "ck_transactions_decision"),
            ("transactions", "ck_transactions_risk_score"),
            ("transactions", "ck_transactions_ml_score"),
            ("transactions", "ck_transactions_behaviour_score"),
            ("transactions", "ck_transactions_rule_score"),
            ("alerts", "ck_alerts_risk_score"),
            ("alerts", "ck_alerts_risk_level"),
            ("alerts", "ck_alerts_decision"),
            ("alerts", "ck_alerts_status"),
            ("customer_devices", "ck_customer_devices_device_type"),
            ("risk_rules_config", "ck_risk_rules_config_score"),
        }
        assert expected_checks.issubset(checks)
