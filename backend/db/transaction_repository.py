"""Transaction repository — PostgreSQL persistence for customer transactions.

Persists every customer transaction together with its fraud-decision
result into the ``transactions`` table defined by the authoritative
Alembic schema (``database/alembic/versions/20260830-4e7709c2bfad-initial_schema.py``).

The legacy Step-40 persistence layer (users / alerts) predates the Alembic
migration and never had a transactions table — submitted transactions were
only written to the audit trail.  This repository closes that gap with a
single atomic INSERT executed after the ML decision is available, so the
row is written once with its final risk fields (the psycopg equivalent of
the create-then-update flow in ``backend/app/repositories/transaction``).

``model_version`` carries a foreign key to ``model_metadata.model_version``;
:meth:`PostgresTransactionRepository.ensure_model_metadata` keeps that FK
target satisfiable for the active model (idempotent upsert, called once at
application startup).
"""

from __future__ import annotations

import logging
from typing import Any

from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

from backend.db.user_repository import _coerce_uuid

logger = logging.getLogger(__name__)

# The production fraud model family (see ml/models/xgboost_model.py and the
# model manifest written by training).  The ML /health identity exposes the
# model name/version but not the framework; the deployed model is XGBoost.
_MODEL_FRAMEWORK = "xgboost"

# Directory managed by the ML ModelRegistry (ML_MODEL_DIR default) that
# holds the manifest and artifacts for each model version.
_MODEL_DIR = "ml/models"


class PostgresTransactionRepository:
    """PostgreSQL-backed transaction persistence (Alembic schema)."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def create(self, **kwargs: Any) -> dict[str, Any]:
        """Insert a transaction row with its fraud-decision result.

        Expected keyword arguments (all supplied by the transaction
        router): ``transaction_id``, ``customer_id``, ``amount``,
        ``currency``, ``transaction_type``, ``location_country``,
        ``location_city``, ``device_fingerprint``, ``device_type``,
        ``ip_address``, ``ml_score``, ``behaviour_score``, ``rule_score``,
        ``risk_score``, ``risk_level``, ``decision``,
        ``explanation_json``, ``model_version``, ``status``.

        ``timestamp`` / ``created_at`` are left to the column server
        defaults (``now()``), matching the SQLAlchemy repository
        convention.

        Raises whatever psycopg raises (FK / CHECK violations, pool
        exhaustion) — persistence failures are hard errors, never silent.
        """
        row = {
            "id": _coerce_uuid(kwargs.get("transaction_id")),
            "customer_id": _coerce_uuid(kwargs.get("customer_id")),
            "merchant_id": None,
            "amount": kwargs.get("amount"),
            "currency": kwargs.get("currency"),
            "transaction_type": kwargs.get("transaction_type"),
            "location_country": kwargs.get("location_country"),
            "location_city": kwargs.get("location_city"),
            "device_fingerprint": kwargs.get("device_fingerprint"),
            "device_type": kwargs.get("device_type"),
            "ip_address": kwargs.get("ip_address"),
            "status": kwargs.get("status") or "COMPLETED",
            "risk_score": kwargs.get("risk_score"),
            "risk_level": kwargs.get("risk_level"),
            "decision": kwargs.get("decision"),
            "ml_score": kwargs.get("ml_score"),
            "behaviour_score": kwargs.get("behaviour_score"),
            "rule_score": kwargs.get("rule_score"),
            "explanation_json": (
                Json(kwargs["explanation_json"])
                if kwargs.get("explanation_json") else None
            ),
            "model_version": kwargs.get("model_version"),
        }
        if row["id"] is None or row["customer_id"] is None:
            raise ValueError(
                "transaction_id and customer_id are required and must be valid UUIDs"
            )

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """\
                    INSERT INTO transactions (
                        id, customer_id, merchant_id, amount, currency,
                        transaction_type, location_country, location_city,
                        device_fingerprint, device_type, ip_address,
                        status, risk_score, risk_level, decision,
                        ml_score, behaviour_score, rule_score,
                        explanation_json, model_version
                    ) VALUES (
                        %(id)s, %(customer_id)s, %(merchant_id)s, %(amount)s,
                        %(currency)s, %(transaction_type)s,
                        %(location_country)s, %(location_city)s,
                        %(device_fingerprint)s, %(device_type)s,
                        %(ip_address)s, %(status)s, %(risk_score)s,
                        %(risk_level)s, %(decision)s, %(ml_score)s,
                        %(behaviour_score)s, %(rule_score)s,
                        %(explanation_json)s, %(model_version)s
                    )""",
                    row,
                )
        return {"id": str(kwargs["transaction_id"])}

    def ensure_model_metadata(
        self, *, model_name: str, model_version: str
    ) -> None:
        """Ensure a ``model_metadata`` row exists for the active model.

        ``transactions.model_version`` references
        ``model_metadata.model_version`` (Alembic initial schema); the
        table is the registry of deployed models and is otherwise
        populated by the deployment / promotion flow.  This idempotent
        upsert keeps the FK satisfiable for the model the ML service
        actually reports as active.
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """\
                    INSERT INTO model_metadata (
                        model_name, model_version, framework, artifact_path
                    ) VALUES (
                        %(name)s, %(version)s, %(framework)s, %(path)s
                    )
                    ON CONFLICT (model_version) DO NOTHING""",
                    {
                        "name": model_name,
                        "version": model_version,
                        "framework": _MODEL_FRAMEWORK,
                        "path": f"{_MODEL_DIR}/{model_version}",
                    },
                )
