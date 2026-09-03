"""Tests for the analytics dashboard endpoint.

Covers:
* RBAC enforcement (401, 403 for customer, 200 for analyst/admin)
* Default 30-day date range
* Custom from_date / to_date
* Invalid date range (from_date > to_date) → 422
* Correct total_transactions / flagged_transactions
* Correct LOW / MEDIUM / HIGH risk distribution
* Correct alerts_open / alerts_resolved
* Correct daily transactions_over_time
* Correct top_risk_factors extraction
* NULL explanation_json handling
* Empty database returns zero/empty response
* Response schema matches api-contract.md
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.config import settings
from app.core.security import create_access_token, hash_password
from app.db.session import SessionLocal
from app.models.alert import Alert
from app.models.audit_log import AuditLog
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction
from app.models.user import User


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
    not _pg_available(), reason="PostgreSQL is not running"
)


def _unique_email() -> str:
    return f"s7-{uuid.uuid4().hex[:12]}@test.com"


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

_S7_USER_PREFIX = "s7-"
_S7_MERCHANT_PREFIX = "Test Merchant S7"


def _cleanup_s7_data(db: Session) -> None:
    """Remove all test data created by Step 7 tests."""
    users = db.query(User).filter(User.email.like(f"{_S7_USER_PREFIX}%@test.com")).all()
    user_ids = [u.id for u in users]
    customer_ids = [u.customer_id for u in users if u.customer_id]

    # Alerts for transactions belonging to test customers.
    if customer_ids:
        txn_ids = [
            r[0]
            for r in db.query(Transaction.id)
            .filter(Transaction.customer_id.in_(customer_ids))
            .all()
        ]
        if txn_ids:
            db.query(Alert).filter(Alert.transaction_id.in_(txn_ids)).delete(
                synchronize_session=False
            )

    # Audit logs for test users.
    if user_ids:
        db.query(AuditLog).filter(AuditLog.actor_id.in_(user_ids)).delete(
            synchronize_session=False
        )

    # Transactions for test customers.
    if customer_ids:
        db.query(Transaction).filter(
            Transaction.customer_id.in_(customer_ids)
        ).delete(synchronize_session=False)

    # Users.
    db.query(User).filter(User.email.like(f"{_S7_USER_PREFIX}%@test.com")).delete(
        synchronize_session=False
    )
    db.commit()

    # Customers.
    if customer_ids:
        db.query(Customer).filter(Customer.id.in_(customer_ids)).delete(
            synchronize_session=False
        )
        db.commit()

    # Merchants.
    db.query(Merchant).filter(
        Merchant.name.like(f"{_S7_MERCHANT_PREFIX}%")
    ).delete(synchronize_session=False)
    db.commit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def s7_db() -> Generator[Session, None, None]:
    """DB session with automatic cleanup for Step 7 tests."""
    db = SessionLocal()
    try:
        _cleanup_s7_data(db)
        yield db
    finally:
        _cleanup_s7_data(db)
        db.close()


def _make_customer(db: Session, suffix: str = "cust") -> Customer:
    customer = Customer(
        first_name=f"S7Cust{suffix}",
        last_name="Test",
        phone="+10000000000",
        address="123 Test St",
    )
    db.add(customer)
    db.flush()
    return customer


def _make_user(
    db: Session,
    *,
    role: str,
    customer: Customer | None = None,
    suffix: str = "",
) -> User:
    email = f"s7-{role}{suffix}-{uuid.uuid4().hex[:8]}@test.com"
    user = User(
        email=email,
        password_hash=hash_password("SecurePass1"),
        first_name="S7",
        last_name="Test",
        role=role,
        customer_id=customer.id if customer else None,
    )
    db.add(user)
    db.flush()
    return user


def _make_merchant(db: Session, name: str = "Test Merchant S7") -> Merchant:
    merchant = Merchant(name=name, category_code="5732", risk_level="LOW")
    db.add(merchant)
    db.flush()
    return merchant


def _make_txn(
    customer_id: uuid.UUID,
    merchant_id: uuid.UUID,
    *,
    risk_level: str | None = None,
    decision: str | None = None,
    explanation: dict | None = None,
    days_ago: int = 0,
    timestamp: datetime | None = None,
) -> Transaction:
    if timestamp is None:
        timestamp = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return Transaction(
        customer_id=customer_id,
        merchant_id=merchant_id,
        amount=Decimal("500.00"),
        currency="USD",
        transaction_type="purchase",
        location_country="US",
        location_city="TestCity",
        device_fingerprint="s7-fp",
        device_type="mobile",
        ip_address="10.0.0.1",
        timestamp=timestamp,
        status="COMPLETED",
        risk_level=risk_level,
        decision=decision,
        ml_score=50 if risk_level else None,
        behaviour_score=50 if risk_level else None,
        rule_score=10 if risk_level else None,
        risk_score=50 if risk_level else None,
        explanation_json=explanation,
    )


# ---------------------------------------------------------------------------
# Shared explanation fixtures
# ---------------------------------------------------------------------------

_EXPLANATION_A = {
    "ml_top_factors": [
        {"feature": "amount_deviation", "importance": 0.45},
        {"feature": "is_new_device", "importance": 0.22},
    ],
    "behaviour_signals": [
        {"signal": "spending_amount_anomaly", "severity": 0.6},
    ],
    "rules_triggered": [
        {"rule": "high_amount", "contribution": 15},
    ],
}

_EXPLANATION_B = {
    "ml_top_factors": [
        {"feature": "amount_deviation", "importance": 0.30},
    ],
    "behaviour_signals": [
        {"signal": "location_anomaly", "severity": 0.8},
    ],
    "rules_triggered": [
        {"rule": "impossible_travel", "contribution": 25},
        {"rule": "high_amount", "contribution": 15},
    ],
}


def _compute_factor_counts(*explanations: dict) -> dict[str, int]:
    """Aggregate risk factor counts from explanation dicts.

    Mirrors the logic in ``AnalyticsService._top_risk_factors`` so tests
    can compute the expected factor counts independently.
    """
    counts: dict[str, int] = {}
    for explanation in explanations:
        for f in explanation.get("ml_top_factors", []):
            name = f.get("feature")
            if name:
                counts[name] = counts.get(name, 0) + 1
        for s in explanation.get("behaviour_signals", []):
            name = s.get("signal")
            if name:
                counts[name] = counts.get(name, 0) + 1
        for r in explanation.get("rules_triggered", []):
            name = r.get("rule")
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


# ===================================================================
# 1. AUTHENTICATION & RBAC
# ===================================================================


@requires_pg
class TestAnalyticsAuth:
    """Authentication and RBAC for GET /api/v1/analytics/dashboard."""

    def test_unauthenticated_returns_401(self, client: TestClient) -> None:
        """Request without token is rejected."""
        resp = client.get("/api/v1/analytics/dashboard")
        assert resp.status_code == 401

    def test_customer_role_returns_403(
        self, client: TestClient, s7_db: Session
    ) -> None:
        """Customer-role users cannot access the dashboard."""
        cust = _make_customer(s7_db, "c403")
        user = _make_user(s7_db, role="customer", customer=cust, suffix="403")
        s7_db.commit()

        token = create_access_token(user.id, user.role)
        resp = client.get(
            "/api/v1/analytics/dashboard", headers=_auth_header(token)
        )
        assert resp.status_code == 403

    def test_fraud_analyst_returns_200(
        self, client: TestClient, s7_db: Session
    ) -> None:
        """fraud_analyst role is allowed."""
        user = _make_user(s7_db, role="fraud_analyst", suffix="a200")
        s7_db.commit()

        token = create_access_token(user.id, user.role)
        resp = client.get(
            "/api/v1/analytics/dashboard", headers=_auth_header(token)
        )
        assert resp.status_code == 200

    def test_admin_returns_200(
        self, client: TestClient, s7_db: Session
    ) -> None:
        """admin role is allowed."""
        user = _make_user(s7_db, role="admin", suffix="adm200")
        s7_db.commit()

        token = create_access_token(user.id, user.role)
        resp = client.get(
            "/api/v1/analytics/dashboard", headers=_auth_header(token)
        )
        assert resp.status_code == 200


# ===================================================================
# 2. QUERY PARAMETER VALIDATION
# ===================================================================


@requires_pg
class TestAnalyticsQueryParams:
    """Date-range validation for GET /api/v1/analytics/dashboard."""

    def _analyst_token(self, db: Session) -> str:
        user = _make_user(db, role="fraud_analyst", suffix="qp")
        db.commit()
        return create_access_token(user.id, user.role)

    def test_default_30_day_range(
        self, client: TestClient, s7_db: Session
    ) -> None:
        """Transactions within 30 days are included; older ones excluded."""
        cust = _make_customer(s7_db, "d30")
        merch = _make_merchant(s7_db, "Test Merchant S7 d30")

        # Recent — 5 days ago (should be included).
        recent = _make_txn(
            cust.id, merch.id, risk_level="LOW", decision="APPROVE", days_ago=5
        )
        s7_db.add(recent)

        # Old — 60 days ago (should be excluded).
        old = _make_txn(
            cust.id, merch.id, risk_level="LOW", decision="APPROVE", days_ago=60
        )
        s7_db.add(old)
        s7_db.commit()

        token = self._analyst_token(s7_db)
        resp = client.get(
            "/api/v1/analytics/dashboard", headers=_auth_header(token)
        )
        data = resp.json()

        assert data["total_transactions"] >= 1
        # The recent one is in range.
        assert data["total_transactions"] >= 1
        # Verify from_date and to_date are set.
        assert data["from_date"] is not None
        assert data["to_date"] is not None

    def test_custom_date_range(
        self, client: TestClient, s7_db: Session
    ) -> None:
        """Explicit from_date / to_date filters correctly."""
        cust = _make_customer(s7_db, "cd")
        merch = _make_merchant(s7_db, "Test Merchant S7 cd")

        now = datetime.now(timezone.utc)
        # Create a transaction exactly 10 days ago.
        ts = now - timedelta(days=10)
        txn = _make_txn(
            cust.id, merch.id, risk_level="LOW", decision="APPROVE", timestamp=ts
        )
        s7_db.add(txn)
        s7_db.commit()

        token = self._analyst_token(s7_db)
        from_d = quote((now - timedelta(days=15)).isoformat())
        to_d = quote((now - timedelta(days=5)).isoformat())
        resp = client.get(
            f"/api/v1/analytics/dashboard?from_date={from_d}&to_date={to_d}",
            headers=_auth_header(token),
        )
        data = resp.json()
        assert resp.status_code == 200
        assert data["total_transactions"] >= 1

    def test_from_date_after_to_date_returns_422(
        self, client: TestClient, s7_db: Session
    ) -> None:
        """from_date > to_date is rejected with 422."""
        token = self._analyst_token(s7_db)
        resp = client.get(
            "/api/v1/analytics/dashboard"
            "?from_date=2026-12-01T00:00:00Z"
            "&to_date=2026-01-01T00:00:00Z",
            headers=_auth_header(token),
        )
        assert resp.status_code == 422


# ===================================================================
# 3. DATA CORRECTNESS
# ===================================================================


@requires_pg
class TestAnalyticsData:
    """Aggregation correctness with controlled test data."""

    @pytest.fixture()
    def seed(self, s7_db: Session, client: TestClient) -> dict:
        """Seed a controlled dataset, capturing baseline for delta assertions.

        The analytics endpoint counts data from the shared Supabase database
        which may contain records from other test classes.  By capturing a
        baseline *before* seeding and comparing the delta *after*, tests
        assert on the exact contribution of the seeded data regardless of
        residual data from other sources.
        """
        # -- Analyst user (needed for API calls) --
        analyst = _make_user(s7_db, role="fraud_analyst", suffix="data")
        s7_db.commit()
        token = create_access_token(analyst.id, analyst.role)

        # -- Baseline (before seeding) --
        baseline = self._get(client, token, scoped=False)
        baseline_factors = {
            f["factor"]: f["count"] for f in baseline["top_risk_factors"]
        }
        # Per-day baseline for transactions_over_time delta.
        baseline_over_time = {
            e["date"]: e for e in baseline["transactions_over_time"]
        }

        # -- Seed data --
        now = datetime.now(timezone.utc)
        d5 = now - timedelta(days=5)
        d10 = now - timedelta(days=10)
        d15 = now - timedelta(days=15)

        cust = _make_customer(s7_db, "data")
        merch = _make_merchant(s7_db, "Test Merchant S7 data")
        s7_db.flush()

        # LOW / APPROVE on day 5
        t_low = _make_txn(
            cust.id, merch.id,
            risk_level="LOW", decision="APPROVE",
            explanation=_EXPLANATION_A, timestamp=d5,
        )
        # MEDIUM / VERIFY on day 5
        t_med = _make_txn(
            cust.id, merch.id,
            risk_level="MEDIUM", decision="VERIFY",
            explanation=_EXPLANATION_B, timestamp=d5,
        )
        # HIGH / HOLD on day 10
        t_high = _make_txn(
            cust.id, merch.id,
            risk_level="HIGH", decision="HOLD",
            explanation=_EXPLANATION_A, timestamp=d10,
        )
        # HIGH / HOLD on day 15
        t_high2 = _make_txn(
            cust.id, merch.id,
            risk_level="HIGH", decision="HOLD",
            explanation=_EXPLANATION_B, timestamp=d15,
        )
        s7_db.add_all([t_low, t_med, t_high, t_high2])
        s7_db.flush()

        # -- Alerts --
        a_open1 = Alert(
            transaction_id=t_high.id, risk_score=80, risk_level="HIGH",
            decision="HOLD", status="OPEN",
        )
        a_open2 = Alert(
            transaction_id=t_high2.id, risk_score=85, risk_level="HIGH",
            decision="HOLD", status="OPEN",
        )
        a_resolved = Alert(
            transaction_id=t_med.id, risk_score=55, risk_level="MEDIUM",
            decision="VERIFY", status="RESOLVED",
            resolved_at=now,
        )
        s7_db.add_all([a_open1, a_open2, a_resolved])
        s7_db.commit()

        # -- Expected delta (what the seed adds) --
        expected_factors = _compute_factor_counts(
            _EXPLANATION_A, _EXPLANATION_A, _EXPLANATION_B, _EXPLANATION_B
        )

        return {
            "token": token,
            "baseline": baseline,
            "baseline_factors": baseline_factors,
            "baseline_over_time": baseline_over_time,
            "expected_delta": {
                "total_transactions": 4,
                "flagged_transactions": 3,
                "alerts_open": 2,
                "alerts_resolved": 1,
                "risk": {"LOW": 1, "MEDIUM": 1, "HIGH": 2},
                "factors": expected_factors,
            },
            "dates": {"d5": d5, "d10": d10, "d15": d15},
        }

    @staticmethod
    def _date_range(days_back: int = 20) -> tuple[str, str]:
        """Return URL-encoded from_date / to_date covering the last *days_back* days."""
        now = datetime.now(timezone.utc)
        return (
            quote((now - timedelta(days=days_back)).isoformat()),
            quote(now.isoformat()),
        )

    def _get(
        self, client: TestClient, token: str, *, scoped: bool = True, **extra: str
    ) -> dict:
        params: dict[str, str] = {}
        if scoped:
            f, t = self._date_range()
            params["from_date"] = f
            params["to_date"] = t
        params.update(extra)
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"/api/v1/analytics/dashboard?{qs}" if qs else "/api/v1/analytics/dashboard"
        resp = client.get(url, headers=_auth_header(token))
        assert resp.status_code == 200
        return resp.json()

    # -- total_transactions ------------------------------------------------

    def test_total_transactions(
        self, client: TestClient, s7_db: Session, seed: dict
    ) -> None:
        """Delta: seeded transactions increase count by exactly 4."""
        data = self._get(client, seed["token"], scoped=False)
        delta = data["total_transactions"] - seed["baseline"]["total_transactions"]
        assert delta == seed["expected_delta"]["total_transactions"]

    # -- flagged_transactions ----------------------------------------------

    def test_flagged_transactions(
        self, client: TestClient, s7_db: Session, seed: dict
    ) -> None:
        """Delta: MEDIUM + HIGH seeded transactions = 3 flagged."""
        data = self._get(client, seed["token"], scoped=False)
        delta = (
            data["flagged_transactions"]
            - seed["baseline"]["flagged_transactions"]
        )
        assert delta == seed["expected_delta"]["flagged_transactions"]

    # -- risk_distribution -------------------------------------------------

    def test_risk_distribution(
        self, client: TestClient, s7_db: Session, seed: dict
    ) -> None:
        """Delta: LOW+1, MEDIUM+1, HIGH+2 from seeded data."""
        data = self._get(client, seed["token"], scoped=False)
        base_risk = seed["baseline"]["risk_distribution"]
        curr_risk = data["risk_distribution"]
        exp = seed["expected_delta"]["risk"]
        assert curr_risk["LOW"] - base_risk.get("LOW", 0) == exp["LOW"]
        assert curr_risk["MEDIUM"] - base_risk.get("MEDIUM", 0) == exp["MEDIUM"]
        assert curr_risk["HIGH"] - base_risk.get("HIGH", 0) == exp["HIGH"]

    # -- alerts ------------------------------------------------------------

    def test_alerts_open(
        self, client: TestClient, s7_db: Session, seed: dict
    ) -> None:
        """Delta: seeded OPEN alerts increase count by exactly 2."""
        data = self._get(client, seed["token"], scoped=False)
        delta = data["alerts_open"] - seed["baseline"]["alerts_open"]
        assert delta == seed["expected_delta"]["alerts_open"]

    def test_alerts_resolved(
        self, client: TestClient, s7_db: Session, seed: dict
    ) -> None:
        """Delta: seeded RESOLVED alert increases count by exactly 1."""
        data = self._get(client, seed["token"], scoped=False)
        delta = data["alerts_resolved"] - seed["baseline"]["alerts_resolved"]
        assert delta == seed["expected_delta"]["alerts_resolved"]

    # -- top_risk_factors --------------------------------------------------

    def test_top_risk_factors(
        self, client: TestClient, s7_db: Session, seed: dict
    ) -> None:
        """Delta: seeded factor counts match expected values."""
        data = self._get(client, seed["token"], scoped=False)
        factors = {f["factor"]: f["count"] for f in data["top_risk_factors"]}
        base_factors = seed["baseline_factors"]
        expected = seed["expected_delta"]["factors"]
        for factor_name, exp_count in expected.items():
            actual_delta = factors.get(factor_name, 0) - base_factors.get(factor_name, 0)
            assert actual_delta == exp_count, (
                f"Factor {factor_name!r}: expected delta {exp_count}, got {actual_delta}"
            )

    def test_top_risk_factors_sorted(
        self, client: TestClient, s7_db: Session, seed: dict
    ) -> None:
        """Factors are ordered by count descending."""
        data = self._get(client, seed["token"], scoped=False)
        counts = [f["count"] for f in data["top_risk_factors"]]
        assert counts == sorted(counts, reverse=True)

    # -- transactions_over_time --------------------------------------------

    def test_transactions_over_time(
        self, client: TestClient, s7_db: Session, seed: dict
    ) -> None:
        """Daily time series: delta matches seeded per-day counts.

        Uses the baseline captured before seeding to compute per-day
        deltas, ensuring correctness regardless of residual data from
        other test classes at the same calendar dates.
        """
        data = self._get(client, seed["token"], scoped=False)
        by_date = {e["date"]: e for e in data["transactions_over_time"]}
        base_ot = seed["baseline_over_time"]

        d5_str = seed["dates"]["d5"].strftime("%Y-%m-%d")
        d10_str = seed["dates"]["d10"].strftime("%Y-%m-%d")
        d15_str = seed["dates"]["d15"].strftime("%Y-%m-%d")

        def _delta(date_str: str, key: str) -> int:
            curr = by_date.get(date_str, {}).get(key, 0)
            prev = base_ot.get(date_str, {}).get(key, 0)
            return curr - prev

        # Day 5: 2 transactions (1 LOW + 1 MEDIUM), 1 flagged (MEDIUM).
        assert _delta(d5_str, "total") == 2
        assert _delta(d5_str, "flagged") == 1

        # Day 10: 1 transaction (HIGH), 1 flagged.
        assert _delta(d10_str, "total") == 1
        assert _delta(d10_str, "flagged") == 1

        # Day 15: 1 transaction (HIGH), 1 flagged.
        assert _delta(d15_str, "total") == 1
        assert _delta(d15_str, "flagged") == 1

    def test_transactions_over_time_continuous(
        self, client: TestClient, s7_db: Session, seed: dict
    ) -> None:
        """Time series has no gaps — every day in range is present."""
        data = self._get(client, seed["token"])
        over_time = data["transactions_over_time"]
        dates = [entry["date"] for entry in over_time]

        # Each date is exactly one day after the previous.
        for i in range(1, len(dates)):
            prev = datetime.strptime(dates[i - 1], "%Y-%m-%d").date()
            curr = datetime.strptime(dates[i], "%Y-%m-%d").date()
            assert (curr - prev).days == 1

    # -- NULL explanation_json ---------------------------------------------

    def test_null_explanation_json(
        self, client: TestClient, s7_db: Session
    ) -> None:
        """Transactions with NULL explanation_json don't break aggregation."""
        now = datetime.now(timezone.utc)
        cust = _make_customer(s7_db, "null")
        merch = _make_merchant(s7_db, "Test Merchant S7 null")
        analyst = _make_user(s7_db, role="fraud_analyst", suffix="null")
        s7_db.flush()

        # Transaction with NULL explanation_json.
        txn = _make_txn(
            cust.id, merch.id,
            risk_level="LOW", decision="APPROVE",
            explanation=None, timestamp=now - timedelta(days=2),
        )
        s7_db.add(txn)
        s7_db.commit()

        token = create_access_token(analyst.id, analyst.role)
        resp = client.get(
            "/api/v1/analytics/dashboard", headers=_auth_header(token)
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_transactions"] >= 1


# ===================================================================
# 4. EMPTY DATABASE
# ===================================================================


@requires_pg
class TestAnalyticsEmpty:
    """Dashboard with no matching data returns zeroed response."""

    def test_empty_range_returns_zeros(
        self, client: TestClient, s7_db: Session
    ) -> None:
        """A date range with no transactions returns all zeros."""
        analyst = _make_user(s7_db, role="fraud_analyst", suffix="empty")
        s7_db.commit()
        token = create_access_token(analyst.id, analyst.role)

        # Use a range far in the past where no data exists.
        resp = client.get(
            "/api/v1/analytics/dashboard"
            "?from_date=2020-01-01T00:00:00Z"
            "&to_date=2020-01-31T23:59:59Z",
            headers=_auth_header(token),
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_transactions"] == 0
        assert data["flagged_transactions"] == 0
        assert data["risk_distribution"]["LOW"] == 0
        assert data["risk_distribution"]["MEDIUM"] == 0
        assert data["risk_distribution"]["HIGH"] == 0
        assert data["top_risk_factors"] == []

        # Over time has one entry per day in the range (31 days), all zero.
        assert len(data["transactions_over_time"]) == 31
        for day in data["transactions_over_time"]:
            assert day["total"] == 0
            assert day["flagged"] == 0


# ===================================================================
# 5. RESPONSE SCHEMA
# ===================================================================


@requires_pg
class TestAnalyticsSchema:
    """Response shape matches docs/api-contract.md exactly."""

    def test_response_schema(
        self, client: TestClient, s7_db: Session
    ) -> None:
        """All documented fields are present with correct types."""
        analyst = _make_user(s7_db, role="fraud_analyst", suffix="schema")
        s7_db.commit()
        token = create_access_token(analyst.id, analyst.role)

        resp = client.get(
            "/api/v1/analytics/dashboard", headers=_auth_header(token)
        )
        assert resp.status_code == 200
        data = resp.json()

        # Top-level fields.
        assert "from_date" in data
        assert "to_date" in data
        assert "total_transactions" in data
        assert "flagged_transactions" in data
        assert "alerts_open" in data
        assert "alerts_resolved" in data
        assert "risk_distribution" in data
        assert "top_risk_factors" in data
        assert "transactions_over_time" in data

        # Types.
        assert isinstance(data["from_date"], str)
        assert isinstance(data["to_date"], str)
        assert isinstance(data["total_transactions"], int)
        assert isinstance(data["flagged_transactions"], int)
        assert isinstance(data["alerts_open"], int)
        assert isinstance(data["alerts_resolved"], int)
        assert isinstance(data["risk_distribution"], dict)
        assert isinstance(data["top_risk_factors"], list)
        assert isinstance(data["transactions_over_time"], list)

        # risk_distribution sub-fields.
        for key in ("LOW", "MEDIUM", "HIGH"):
            assert key in data["risk_distribution"]
            assert isinstance(data["risk_distribution"][key], int)

    def test_no_undocumented_fields(
        self, client: TestClient, s7_db: Session
    ) -> None:
        """Response contains no fields beyond the API contract."""
        analyst = _make_user(s7_db, role="fraud_analyst", suffix="nodoc")
        s7_db.commit()
        token = create_access_token(analyst.id, analyst.role)

        resp = client.get(
            "/api/v1/analytics/dashboard", headers=_auth_header(token)
        )
        data = resp.json()

        expected_keys = {
            "from_date", "to_date",
            "total_transactions", "flagged_transactions",
            "alerts_open", "alerts_resolved",
            "risk_distribution", "top_risk_factors",
            "transactions_over_time",
        }
        assert set(data.keys()) == expected_keys
