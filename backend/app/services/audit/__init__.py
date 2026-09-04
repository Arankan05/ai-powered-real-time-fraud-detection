"""Centralized audit logging service.

Creates ``AuditLog`` records for state-changing operations.
Audit logs are **append-only** — rows are never updated or deleted.
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog

logger = logging.getLogger(__name__)


class AuditService:
    """Creates audit log entries for important business/security actions."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def log(
        self,
        *,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        actor_id: UUID | None = None,
        details: dict | None = None,
        ip_address: str | None = None,
    ) -> AuditLog:
        """Append a single audit log record.

        This method must **not** break the primary business operation.
        If logging fails the error is logged but not re-raised, unless
        the caller is inside the same DB transaction (in which case the
        caller decides whether to propagate).
        """
        entry = AuditLog(
            actor_id=actor_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details_json=details,
            ip_address=ip_address,
        )
        self._db.add(entry)
        self._db.flush()
        return entry
