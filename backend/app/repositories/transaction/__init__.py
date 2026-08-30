"""Data-access layer for customer, transaction, alert, and audit operations."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from app.models.alert import Alert
from app.models.audit_log import AuditLog
from app.models.customer import Customer
from app.models.customer_device import CustomerDevice
from app.models.merchant import Merchant
from app.models.transaction import Transaction
from app.models.user import User


class CustomerRepository:
    """Database operations for customers."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def get_by_id(self, customer_id: UUID) -> Customer | None:
        return self._db.get(Customer, customer_id)


class MerchantRepository:
    """Database operations for merchants."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def get_or_create(self, name: str, category_code: str | None) -> Merchant:
        """Find a merchant by name+category or create a new one."""
        stmt = select(Merchant).where(Merchant.name == name)
        if category_code:
            stmt = stmt.where(Merchant.category_code == category_code)
        else:
            stmt = stmt.where(Merchant.category_code.is_(None))

        merchant = self._db.execute(stmt).scalar_one_or_none()
        if merchant is None:
            merchant = Merchant(name=name, category_code=category_code)
            self._db.add(merchant)
            self._db.flush()
        return merchant


class TransactionRepository:
    """Database operations for transactions."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, *, customer_id: UUID, merchant_id: UUID | None,
               amount: Decimal, currency: str, transaction_type: str,
               location_country: str | None, location_city: str | None,
               device_fingerprint: str | None, device_type: str | None,
               ip_address: str | None) -> Transaction:
        """Insert a new transaction with status=PENDING."""
        txn = Transaction(
            customer_id=customer_id,
            merchant_id=merchant_id,
            amount=amount,
            currency=currency,
            transaction_type=transaction_type,
            location_country=location_country,
            location_city=location_city,
            device_fingerprint=device_fingerprint,
            device_type=device_type,
            ip_address=ip_address,
            status="PENDING",
        )
        self._db.add(txn)
        self._db.flush()
        return txn

    def update_with_fraud_results(
        self,
        txn: Transaction,
        *,
        ml_score: int,
        behaviour_score: int,
        rule_score: int,
        risk_score: int,
        risk_level: str,
        decision: str,
        explanation_json: dict,
        model_version: str | None,
        status: str,
    ) -> Transaction:
        """Persist fraud analysis results and update transaction status."""
        txn.ml_score = ml_score
        txn.behaviour_score = behaviour_score
        txn.rule_score = rule_score
        txn.risk_score = risk_score
        txn.risk_level = risk_level
        txn.decision = decision
        txn.explanation_json = explanation_json
        txn.model_version = model_version
        txn.status = status
        self._db.flush()
        return txn

    def get_by_id(self, transaction_id: UUID) -> Transaction | None:
        return self._db.get(Transaction, transaction_id)

    def list_transactions(
        self,
        *,
        customer_id: UUID | None = None,
        status: str | None = None,
        risk_level: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[Transaction], int]:
        """Return (items, total_count) with pagination and filters."""
        stmt = select(Transaction)
        count_stmt = select(func.count()).select_from(Transaction)

        if customer_id is not None:
            stmt = stmt.where(Transaction.customer_id == customer_id)
            count_stmt = count_stmt.where(Transaction.customer_id == customer_id)
        if status is not None:
            stmt = stmt.where(Transaction.status == status)
            count_stmt = count_stmt.where(Transaction.status == status)
        if risk_level is not None:
            stmt = stmt.where(Transaction.risk_level == risk_level)
            count_stmt = count_stmt.where(Transaction.risk_level == risk_level)
        if from_date is not None:
            stmt = stmt.where(Transaction.timestamp >= from_date)
            count_stmt = count_stmt.where(Transaction.timestamp >= from_date)
        if to_date is not None:
            stmt = stmt.where(Transaction.timestamp <= to_date)
            count_stmt = count_stmt.where(Transaction.timestamp <= to_date)

        total = self._db.execute(count_stmt).scalar() or 0

        stmt = stmt.order_by(Transaction.timestamp.desc())
        stmt = stmt.offset((page - 1) * per_page).limit(per_page)

        items = list(self._db.execute(stmt).scalars().all())
        return items, total

    def get_customer_history_stats(self, customer_id: UUID) -> dict:
        """Compute basic customer history stats for the ML request payload."""
        from datetime import timedelta, timezone

        now = datetime.now(timezone.utc)
        thirty_days_ago = now - timedelta(days=30)

        # Count + aggregates for last 30 days
        stmt = select(
            func.count(Transaction.id),
            func.avg(Transaction.amount),
            func.stddev(Transaction.amount),
        ).where(
            Transaction.customer_id == customer_id,
            Transaction.timestamp >= thirty_days_ago,
        )
        row = self._db.execute(stmt).one()
        tx_count = row[0] or 0
        avg_amt = float(row[1]) if row[1] else 0.0
        std_amt = float(row[2]) if row[2] else 0.0

        # Last transaction country and timestamp
        last_txn_stmt = (
            select(Transaction.location_country, Transaction.timestamp)
            .where(Transaction.customer_id == customer_id)
            .order_by(Transaction.timestamp.desc())
            .limit(1)
        )
        last_row = self._db.execute(last_txn_stmt).first()
        last_country = last_row[0] if last_row else None
        last_ts = last_row[1].isoformat() if last_row and last_row[1] else None

        # Known device fingerprints
        devices_stmt = select(CustomerDevice.device_fingerprint).where(
            CustomerDevice.customer_id == customer_id,
        )
        known_fps = list(self._db.execute(devices_stmt).scalars().all())

        # Known merchant IDs
        merchants_stmt = (
            select(Transaction.merchant_id)
            .where(
                Transaction.customer_id == customer_id,
                Transaction.merchant_id.isnot(None),
            )
            .distinct()
        )
        known_merchants = [
            str(m) for m in self._db.execute(merchants_stmt).scalars().all()
        ]

        # Previous flagged count
        flagged_stmt = select(func.count(Transaction.id)).where(
            Transaction.customer_id == customer_id,
            Transaction.decision.in_(["VERIFY", "HOLD"]),
        )
        prev_flagged = self._db.execute(flagged_stmt).scalar() or 0

        return {
            "transaction_count_30d": tx_count,
            "avg_amount_30d": avg_amt,
            "std_amount_30d": std_amt,
            "last_transaction_country": last_country,
            "last_transaction_timestamp": last_ts,
            "known_device_fingerprints": known_fps,
            "known_merchant_ids": known_merchants,
            "previous_flagged_count": prev_flagged,
        }


class AlertRepository:
    """Database operations for alerts."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        *,
        transaction_id: UUID,
        risk_score: int,
        risk_level: str,
        decision: str,
        explanation_json: dict | None,
    ) -> Alert:
        """Create a HOLD alert for a high-risk transaction."""
        alert = Alert(
            transaction_id=transaction_id,
            risk_score=risk_score,
            risk_level=risk_level,
            decision=decision,
            explanation_json=explanation_json,
            status="OPEN",
        )
        self._db.add(alert)
        self._db.flush()
        return alert

    def get_by_transaction(self, transaction_id: UUID) -> Alert | None:
        stmt = select(Alert).where(Alert.transaction_id == transaction_id)
        return self._db.execute(stmt).scalar_one_or_none()

    def get_by_id(self, alert_id: UUID) -> Alert | None:
        """Fetch a single alert with its related transaction."""
        stmt = (
            select(Alert)
            .options(joinedload(Alert.transaction))
            .where(Alert.id == alert_id)
        )
        return self._db.execute(stmt).unique().scalar_one_or_none()

    def list_alerts(
        self,
        *,
        status: str | None = None,
        risk_level: str | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[Alert], int]:
        """Return (items, total_count) with pagination and filters."""
        stmt = select(Alert).options(joinedload(Alert.transaction))
        count_stmt = select(func.count()).select_from(Alert)

        if status is not None:
            stmt = stmt.where(Alert.status == status)
            count_stmt = count_stmt.where(Alert.status == status)
        if risk_level is not None:
            stmt = stmt.where(Alert.risk_level == risk_level)
            count_stmt = count_stmt.where(Alert.risk_level == risk_level)

        total = self._db.execute(count_stmt).scalar() or 0

        stmt = stmt.order_by(Alert.created_at.desc())
        stmt = stmt.offset((page - 1) * per_page).limit(per_page)

        items = list(self._db.execute(stmt).unique().scalars().all())
        return items, total

    def update(
        self,
        alert: Alert,
        *,
        status: str | None = None,
        notes: str | None = None,
        analyst_id: UUID | None = None,
        resolved_at: datetime | None = None,
    ) -> Alert:
        """Update alert status and/or notes."""
        if status is not None:
            alert.status = status
        if notes is not None:
            alert.notes = notes
        if analyst_id is not None and alert.analyst_id is None:
            alert.analyst_id = analyst_id
        if resolved_at is not None:
            alert.resolved_at = resolved_at
        self._db.flush()
        return alert


class AuditLogRepository:
    """Database operations for audit logs (append-only)."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        *,
        actor_id: UUID | None,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        details_json: dict | None = None,
        ip_address: str | None = None,
    ) -> AuditLog:
        entry = AuditLog(
            actor_id=actor_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details_json=details_json,
            ip_address=ip_address,
        )
        self._db.add(entry)
        self._db.flush()
        return entry
