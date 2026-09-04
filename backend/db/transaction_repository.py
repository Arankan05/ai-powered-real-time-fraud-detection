"""Transaction repository — PostgreSQL persistence for customer transactions.

Persists every customer transaction together with its fraud-decision
result into the ``transactions`` table defined by the authoritative
Alembic schema (``database/alembic/versions/20260830-4e7709c2bfad-initial_schema.py``).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from psycopg.rows import dict_row
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


@runtime_checkable
class TransactionRepository(Protocol):
    """Abstract contract for transaction persistence."""

    def create(self, **kwargs: Any) -> dict[str, Any]:
        """Insert a transaction row with its fraud-decision result."""
        ...

    def get_by_id(self, transaction_id: str) -> dict[str, Any] | None:
        """Return a single transaction by ID, or None."""
        ...

    def list_transactions(
        self,
        *,
        customer_id: str | None = None,
        status: str | None = None,
        risk_level: str | None = None,
        from_date: str | datetime | None = None,
        to_date: str | datetime | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return (transactions, total_count) with optional filtering."""
        ...

    def ensure_model_metadata(
        self, *, model_name: str, model_version: str
    ) -> None:
        """Ensure model metadata exists."""
        ...


class PostgresTransactionRepository:
    """PostgreSQL-backed transaction persistence (Alembic schema)."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def _row_to_dict(self, row: dict[str, Any]) -> dict[str, Any]:
        res = dict(row)
        if res.get("id"):
            res["id"] = str(res["id"])
            res["transaction_id"] = res["id"]
        if res.get("customer_id"):
            res["customer_id"] = str(res["customer_id"])
        if res.get("merchant_id"):
            res["merchant_id"] = str(res["merchant_id"])
        if res.get("amount") is not None:
            res["amount"] = float(res["amount"])
        if isinstance(res.get("timestamp"), datetime):
            res["timestamp"] = res["timestamp"].isoformat()
        if isinstance(res.get("created_at"), datetime):
            res["created_at"] = res["created_at"].isoformat()
        if not res.get("merchant_name"):
            res["merchant_name"] = "N/A"
        if not res.get("merchant_category"):
            res["merchant_category"] = "N/A"
        if res.get("explanation_json") is not None:
            expl = res["explanation_json"]
            if isinstance(expl, str):
                try:
                    expl = json.loads(expl)
                except Exception:
                    pass
            res["explanation"] = expl
        return res

    def create(self, **kwargs: Any) -> dict[str, Any]:
        """Insert a transaction row with its fraud-decision result."""
        merchant_name = kwargs.get("merchant_name")
        merchant_category = kwargs.get("merchant_category")
        merchant_id = None

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
            with conn.cursor(row_factory=dict_row) as cur:
                if merchant_name:
                    cur.execute("SELECT id FROM merchants WHERE name = %s", (merchant_name,))
                    m_row = cur.fetchone()
                    if m_row:
                        merchant_id = m_row["id"]
                    else:
                        m_uuid = uuid.uuid4()
                        cur.execute(
                            "INSERT INTO merchants (id, name, category_code) VALUES (%s, %s, %s)",
                            (m_uuid, merchant_name, merchant_category),
                        )
                        merchant_id = m_uuid
                row["merchant_id"] = merchant_id

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

    def get_by_id(self, transaction_id: str) -> dict[str, Any] | None:
        tid = _coerce_uuid(transaction_id)
        if tid is None:
            return None
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT t.*, m.name as merchant_name, m.category_code as merchant_category
                    FROM transactions t
                    LEFT JOIN merchants m ON t.merchant_id = m.id
                    WHERE t.id = %s
                    """,
                    (tid,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def list_transactions(
        self,
        *,
        customer_id: str | None = None,
        status: str | None = None,
        risk_level: str | None = None,
        from_date: str | datetime | None = None,
        to_date: str | datetime | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        clauses: list[str] = []
        params: list[Any] = []

        if customer_id:
            cid = _coerce_uuid(customer_id)
            if cid:
                clauses.append("t.customer_id = %s")
                params.append(cid)
            else:
                return [], 0
        if status:
            clauses.append("t.status = %s")
            params.append(status)
        if risk_level:
            clauses.append("t.risk_level = %s")
            params.append(risk_level)
        if from_date:
            clauses.append("t.timestamp >= %s")
            params.append(from_date)
        if to_date:
            clauses.append("t.timestamp <= %s")
            params.append(to_date)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(f"SELECT COUNT(*) FROM transactions t {where}", params)
                total_row = cur.fetchone()
                total = int(total_row["count"] if isinstance(total_row, dict) else total_row[0])

                offset = (page - 1) * per_page
                cur.execute(
                    f"""
                    SELECT t.*, m.name as merchant_name, m.category_code as merchant_category
                    FROM transactions t
                    LEFT JOIN merchants m ON t.merchant_id = m.id
                    {where}
                    ORDER BY t.timestamp DESC
                    LIMIT %s OFFSET %s
                    """,
                    [*params, per_page, offset],
                )
                rows = cur.fetchall()

        return [self._row_to_dict(r) for r in rows], total

    def ensure_model_metadata(
        self, *, model_name: str, model_version: str
    ) -> None:
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


class InMemoryTransactionStore:
    """In-memory store for fallback/tests/SQLite mode."""

    def __init__(self) -> None:
        self._txns: dict[str, dict[str, Any]] = {}

    def create(self, **kwargs: Any) -> dict[str, Any]:
        tid = str(kwargs.get("transaction_id") or uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        expl = kwargs.get("explanation_json")
        row = {
            "id": tid,
            "transaction_id": tid,
            "customer_id": str(kwargs.get("customer_id")) if kwargs.get("customer_id") else None,
            "merchant_id": None,
            "merchant_name": kwargs.get("merchant_name") or "N/A",
            "merchant_category": kwargs.get("merchant_category") or "N/A",
            "amount": float(kwargs.get("amount", 0)),
            "currency": kwargs.get("currency", "USD"),
            "transaction_type": kwargs.get("transaction_type", "purchase"),
            "location_country": kwargs.get("location_country", "Unknown"),
            "location_city": kwargs.get("location_city", "Unknown"),
            "device_fingerprint": kwargs.get("device_fingerprint", "Unknown"),
            "device_type": kwargs.get("device_type", "desktop"),
            "ip_address": kwargs.get("ip_address", "0.0.0.0"),
            "status": kwargs.get("status") or "COMPLETED",
            "risk_score": kwargs.get("risk_score"),
            "risk_level": kwargs.get("risk_level"),
            "decision": kwargs.get("decision"),
            "ml_score": kwargs.get("ml_score"),
            "behaviour_score": kwargs.get("behaviour_score"),
            "rule_score": kwargs.get("rule_score"),
            "explanation_json": expl,
            "explanation": expl,
            "model_version": kwargs.get("model_version"),
            "timestamp": now,
            "created_at": now,
        }
        self._txns[tid] = row
        return {"id": tid}

    def get_by_id(self, transaction_id: str) -> dict[str, Any] | None:
        return self._txns.get(str(transaction_id))

    def list_transactions(
        self,
        *,
        customer_id: str | None = None,
        status: str | None = None,
        risk_level: str | None = None,
        from_date: Any = None,
        to_date: Any = None,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        filtered = list(self._txns.values())
        if customer_id is not None:
            filtered = [t for t in filtered if t.get("customer_id") == str(customer_id)]
        if status is not None:
            filtered = [t for t in filtered if t.get("status") == status]
        if risk_level is not None:
            filtered = [t for t in filtered if t.get("risk_level") == risk_level]

        filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        total = len(filtered)
        start = (page - 1) * per_page
        end = start + per_page
        return filtered[start:end], total

    def ensure_model_metadata(self, *, model_name: str, model_version: str) -> None:
        pass
