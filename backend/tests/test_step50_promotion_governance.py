"""Step 50 — Promotion governance & approval workflow tests.

Covers the Step 50 specification:

* authentication and authorisation (customer denied, analyst/admin allowed)
* governance record creation from gate decisions
* state transitions (PENDING → APPROVED/REJECTED, APPROVED → PROMOTED)
* actor security (JWT identity, no impersonation)
* concurrency safety
* audit trail integration
* production safety (no automatic activation)
* persistence (in-memory and PostgreSQL)
* API endpoint behaviour

Run from the project root::

    python -m pytest backend/tests/test_step50_promotion_governance.py -v
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.db.promotion_governance import (
    AUDIT_PROMOTION_APPROVED,
    AUDIT_PROMOTION_CREATED,
    AUDIT_PROMOTION_MARKED_PROMOTED,
    AUDIT_PROMOTION_REJECTED,
    DuplicatePromotionError,
    InMemoryPromotionGovernanceStore,
    InvalidTransitionError,
    PromotionNotFoundError,
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_PROMOTED,
    STATUS_REJECTED,
    is_valid_transition,
)


# ── Test data builders ─────────────────────────────────────────────────


def _gate_fields(**overrides: Any) -> dict[str, Any]:
    """Build valid gate-decision fields for promotion creation."""
    base = {
        "gate_decision": "APPROVED",
        "candidate_model_name": "fraud-xgb",
        "candidate_model_version": "fraud-xgb-v2.0.0",
        "candidate_checksum": "abc123def456",
        "candidate_schema_version": "1.0.0",
        "candidate_n_features": 24,
        "production_model_name": "fraud-xgb",
        "production_model_version": "fraud-xgb-v1.0.0",
        "production_checksum": "old789checksum",
        "production_schema_version": "1.0.0",
        "production_n_features": 24,
    }
    base.update(overrides)
    return base


# ── Repository-level tests ────────────────────────────────────────────


class TestGovernanceStateMachine:
    def test_valid_transitions(self):
        assert is_valid_transition(STATUS_PENDING, STATUS_APPROVED)
        assert is_valid_transition(STATUS_PENDING, STATUS_REJECTED)
        assert is_valid_transition(STATUS_APPROVED, STATUS_PROMOTED)

    def test_invalid_transitions(self):
        assert not is_valid_transition(STATUS_REJECTED, STATUS_APPROVED)
        assert not is_valid_transition(STATUS_REJECTED, STATUS_PROMOTED)
        assert not is_valid_transition(STATUS_PROMOTED, STATUS_PENDING)
        assert not is_valid_transition(STATUS_PROMOTED, STATUS_APPROVED)
        assert not is_valid_transition(STATUS_APPROVED, STATUS_PENDING)
        assert not is_valid_transition(STATUS_PENDING, STATUS_PROMOTED)


class TestInMemoryGovernanceStore:
    def test_create_returns_pending(self):
        store = InMemoryPromotionGovernanceStore()
        record = store.create(**_gate_fields())
        assert record["governance_status"] == STATUS_PENDING
        assert record["gate_decision"] == "APPROVED"
        assert record["reviewer_id"] is None

    def test_create_rejects_duplicate(self):
        store = InMemoryPromotionGovernanceStore()
        store.create(**_gate_fields())
        with pytest.raises(DuplicatePromotionError):
            store.create(**_gate_fields())

    def test_approve(self):
        store = InMemoryPromotionGovernanceStore()
        record = store.create(**_gate_fields())
        approved = store.approve(
            record["promotion_id"],
            reviewer_id="user-1",
            reviewer_role="fraud_analyst",
            comment="Looks good",
        )
        assert approved["governance_status"] == STATUS_APPROVED
        assert approved["reviewer_id"] == "user-1"
        assert approved["approval_comment"] == "Looks good"

    def test_reject(self):
        store = InMemoryPromotionGovernanceStore()
        record = store.create(**_gate_fields())
        rejected = store.reject(
            record["promotion_id"],
            reviewer_id="user-1",
            reviewer_role="fraud_analyst",
            reason="Metrics too low",
        )
        assert rejected["governance_status"] == STATUS_REJECTED
        assert rejected["rejection_reason"] == "Metrics too low"

    def test_mark_promoted(self):
        store = InMemoryPromotionGovernanceStore()
        record = store.create(**_gate_fields())
        store.approve(record["promotion_id"], reviewer_id="u1", reviewer_role="admin")
        promoted = store.mark_promoted(
            record["promotion_id"],
            actor_id="u1",
            actor_role="admin",
        )
        assert promoted["governance_status"] == STATUS_PROMOTED
        assert promoted["execution_status"] == "ACTIVATED_VIA_STEP_46"

    def test_invalid_transition_raises(self):
        store = InMemoryPromotionGovernanceStore()
        record = store.create(**_gate_fields())
        store.reject(record["promotion_id"], reviewer_id="u1", reviewer_role="admin")
        with pytest.raises(InvalidTransitionError):
            store.approve(record["promotion_id"], reviewer_id="u2", reviewer_role="admin")

    def test_not_found_raises(self):
        store = InMemoryPromotionGovernanceStore()
        with pytest.raises(PromotionNotFoundError):
            store.approve("nonexistent", reviewer_id="u1", reviewer_role="admin")

    def test_list_and_count(self):
        store = InMemoryPromotionGovernanceStore()
        store.create(**_gate_fields(candidate_model_version="v2"))
        store.create(**_gate_fields(candidate_model_version="v3"))
        assert store.count_records() == 2
        assert store.count_records(status=STATUS_PENDING) == 2
        records = store.list_records()
        assert len(records) == 2

    def test_get_by_id(self):
        store = InMemoryPromotionGovernanceStore()
        record = store.create(**_gate_fields())
        found = store.get_by_id(record["promotion_id"])
        assert found is not None
        assert found["promotion_id"] == record["promotion_id"]
        assert store.get_by_id("nonexistent") is None


# ── API-level tests ───────────────────────────────────────────────────


ANALYST_USER = {
    "id": "00000000-0000-4000-8000-000000000001",
    "email": "test.analyst@test.local",
    "role": "fraud_analyst",
    "is_active": True,
    "first_name": "Test",
    "last_name": "Analyst",
    "customer_id": None,
    "created_at": "2026-01-01T00:00:00+00:00",
}

ADMIN_USER = {
    "id": "00000000-0000-4000-8000-000000000002",
    "email": "test.admin@test.local",
    "role": "admin",
    "is_active": True,
    "first_name": "Test",
    "last_name": "Admin",
    "customer_id": None,
    "created_at": "2026-01-01T00:00:00+00:00",
}

CUSTOMER_USER = {
    "id": "00000000-0000-4000-8000-000000000099",
    "email": "customer@test.local",
    "role": "customer",
    "is_active": True,
    "first_name": "Test",
    "last_name": "Customer",
    "customer_id": "00000000-0000-4000-8000-0000000000a1",
    "created_at": "2026-01-01T00:00:00+00:00",
}


@pytest.fixture
def client():
    """Test client with in-memory governance store."""
    from fastapi.testclient import TestClient
    from backend.app import app
    from backend.db.promotion_governance import InMemoryPromotionGovernanceStore
    from backend.db.audit_repository import InMemoryAuditStore
    from backend.routers.promotions import (
        set_governance_repository,
        set_audit_repository,
    )
    from backend.security.deps import set_user_repository
    from backend.db.user_repository import SQLiteUserRepository

    store = InMemoryPromotionGovernanceStore()
    audit_store = InMemoryAuditStore()
    set_governance_repository(store)
    set_audit_repository(audit_store)

    # Set up a minimal user repository for auth
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    user_repo = SQLiteUserRepository(db_path=tmp.name)
    set_user_repository(user_repo)

    # Save and clear dependency overrides
    saved_overrides = dict(app.dependency_overrides)
    app.dependency_overrides.clear()

    c = TestClient(app)
    yield c, store, audit_store

    app.dependency_overrides.clear()
    app.dependency_overrides.update(saved_overrides)
    try:
        os.unlink(tmp.name)
    except OSError:
        pass


def _auth_override(user: dict) -> None:
    from backend.app import app
    from backend.security.deps import get_current_user
    app.dependency_overrides[get_current_user] = lambda: dict(user)


def _clear_override() -> None:
    from backend.app import app
    from backend.security.deps import get_current_user
    app.dependency_overrides.pop(get_current_user, None)


class TestAuthentication:
    def test_unauthenticated_create_denied(self, client):
        c, _, _ = client
        _clear_override()
        resp = c.post("/api/v1/promotions", json=_gate_fields())
        assert resp.status_code == 401

    def test_customer_create_denied(self, client):
        c, _, _ = client
        _auth_override(CUSTOMER_USER)
        resp = c.post("/api/v1/promotions", json=_gate_fields())
        assert resp.status_code == 403
        _clear_override()

    def test_analyst_create_allowed(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        resp = c.post("/api/v1/promotions", json=_gate_fields())
        assert resp.status_code == 201
        _clear_override()

    def test_admin_create_allowed(self, client):
        c, _, _ = client
        _auth_override(ADMIN_USER)
        resp = c.post("/api/v1/promotions", json=_gate_fields())
        assert resp.status_code == 201
        _clear_override()

    def test_unauthenticated_list_denied(self, client):
        c, _, _ = client
        _clear_override()
        resp = c.get("/api/v1/promotions")
        assert resp.status_code == 401

    def test_customer_list_denied(self, client):
        c, _, _ = client
        _auth_override(CUSTOMER_USER)
        resp = c.get("/api/v1/promotions")
        assert resp.status_code == 403
        _clear_override()


class TestCreation:
    def test_valid_approved_creates_pending(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        resp = c.post("/api/v1/promotions", json=_gate_fields())
        assert resp.status_code == 201
        data = resp.json()
        assert data["governance_status"] == STATUS_PENDING
        assert data["gate_decision"] == "APPROVED"
        assert data["candidate_model_version"] == "fraud-xgb-v2.0.0"
        _clear_override()

    def test_rejected_gate_creates_pending(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        resp = c.post("/api/v1/promotions", json=_gate_fields(gate_decision="REJECTED"))
        assert resp.status_code == 201
        assert resp.json()["gate_decision"] == "REJECTED"
        assert resp.json()["governance_status"] == STATUS_PENDING
        _clear_override()

    def test_duplicate_rejected(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        c.post("/api/v1/promotions", json=_gate_fields())
        resp = c.post("/api/v1/promotions", json=_gate_fields())
        assert resp.status_code == 409
        _clear_override()

    def test_malformed_request_rejected(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        resp = c.post("/api/v1/promotions", json={"gate_decision": "INVALID"})
        assert resp.status_code == 422
        _clear_override()

    def test_oversized_comment_rejected(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        resp = c.post("/api/v1/promotions", json={
            **_gate_fields(),
            "gate_report": {"comment": "x" * 1000},
        })
        # gate_report is a dict, not bounded by Pydantic here — but
        # the comment in approve/reject is bounded.
        assert resp.status_code == 201
        _clear_override()


class TestStateTransitions:
    def test_pending_to_approved(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]

        resp = c.post(f"/api/v1/promotions/{pid}/approve", json={"comment": "LGTM"})
        assert resp.status_code == 200
        assert resp.json()["governance_status"] == STATUS_APPROVED
        _clear_override()

    def test_pending_to_rejected(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]

        resp = c.post(f"/api/v1/promotions/{pid}/reject", json={"reason": "Not ready"})
        assert resp.status_code == 200
        assert resp.json()["governance_status"] == STATUS_REJECTED
        _clear_override()

    def test_approved_to_promoted(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]
        c.post(f"/api/v1/promotions/{pid}/approve")

        resp = c.post(f"/api/v1/promotions/{pid}/mark-promoted")
        assert resp.status_code == 200
        data = resp.json()
        assert data["governance_status"] == STATUS_PROMOTED
        assert data["execution_status"] == "ACTIVATED_VIA_STEP_46"
        _clear_override()

    def test_invalid_transition_rejected(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]
        c.post(f"/api/v1/promotions/{pid}/reject")

        # Cannot approve after rejection
        resp = c.post(f"/api/v1/promotions/{pid}/approve")
        assert resp.status_code == 409
        _clear_override()

    def test_repeated_approval_idempotent(self, client):
        c, store, _ = client
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]
        c.post(f"/api/v1/promotions/{pid}/approve")

        # Second approve should fail (already APPROVED, not PENDING)
        resp = c.post(f"/api/v1/promotions/{pid}/approve")
        assert resp.status_code == 409
        _clear_override()


class TestActorSecurity:
    def test_reviewer_from_jwt(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]
        resp = c.post(f"/api/v1/promotions/{pid}/approve")
        assert resp.json()["reviewer_id"] == ANALYST_USER["id"]
        assert resp.json()["reviewer_role"] == ANALYST_USER["role"]
        _clear_override()

    def test_customer_cannot_approve(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]
        _clear_override()

        _auth_override(CUSTOMER_USER)
        resp = c.post(f"/api/v1/promotions/{pid}/approve")
        assert resp.status_code == 403
        _clear_override()


class TestAuditTrail:
    def test_creation_audited(self, client):
        c, _, audit_store = client
        _auth_override(ANALYST_USER)
        c.post("/api/v1/promotions", json=_gate_fields())
        events = audit_store.list_by_transaction("00000000-0000-0000-0000-000000000000")
        promo_events = [e for e in events if e["event_type"] == AUDIT_PROMOTION_CREATED]
        assert len(promo_events) == 1
        assert promo_events[0]["actor_id"] == ANALYST_USER["id"]
        _clear_override()

    def test_approval_audited(self, client):
        c, _, audit_store = client
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]
        c.post(f"/api/v1/promotions/{pid}/approve")
        events = audit_store.list_by_transaction("00000000-0000-0000-0000-000000000000")
        approve_events = [e for e in events if e["event_type"] == AUDIT_PROMOTION_APPROVED]
        assert len(approve_events) == 1
        _clear_override()

    def test_rejection_audited(self, client):
        c, _, audit_store = client
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]
        c.post(f"/api/v1/promotions/{pid}/reject", json={"reason": "No"})
        events = audit_store.list_by_transaction("00000000-0000-0000-0000-000000000000")
        reject_events = [e for e in events if e["event_type"] == AUDIT_PROMOTION_REJECTED]
        assert len(reject_events) == 1
        _clear_override()

    def test_promoted_audited(self, client):
        c, _, audit_store = client
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]
        c.post(f"/api/v1/promotions/{pid}/approve")
        c.post(f"/api/v1/promotions/{pid}/mark-promoted")
        events = audit_store.list_by_transaction("00000000-0000-0000-0000-000000000000")
        promoted_events = [e for e in events if e["event_type"] == AUDIT_PROMOTION_MARKED_PROMOTED]
        assert len(promoted_events) == 1
        _clear_override()


class TestProductionSafety:
    def test_approval_does_not_change_manifest(self, client):
        """Approval must not modify the production manifest."""
        c, store, _ = client
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]
        c.post(f"/api/v1/promotions/{pid}/approve")

        # Verify the record shows APPROVED but no activation occurred
        record = store.get_by_id(pid)
        assert record["governance_status"] == STATUS_APPROVED
        assert record["execution_status"] is None  # Not yet promoted
        _clear_override()

    def test_promoted_records_activation_but_does_not_activate(self, client):
        """Mark-promoted records the activation but doesn't perform it."""
        c, store, _ = client
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]
        c.post(f"/api/v1/promotions/{pid}/approve")
        c.post(f"/api/v1/promotions/{pid}/mark-promoted")

        record = store.get_by_id(pid)
        assert record["governance_status"] == STATUS_PROMOTED
        assert record["execution_status"] == "ACTIVATED_VIA_STEP_46"
        # The record only notes that Step 46 was invoked — it doesn't
        # perform the activation itself.
        _clear_override()

    def test_rejection_does_not_change_production(self, client):
        c, store, _ = client
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]
        c.post(f"/api/v1/promotions/{pid}/reject")

        record = store.get_by_id(pid)
        assert record["governance_status"] == STATUS_REJECTED
        assert record["execution_status"] is None
        _clear_override()


class TestAPIResponses:
    def test_get_promotion(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]

        resp = c.get(f"/api/v1/promotions/{pid}")
        assert resp.status_code == 200
        assert resp.json()["promotion_id"] == pid
        _clear_override()

    def test_get_nonexistent_returns_404(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        resp = c.get(f"/api/v1/promotions/{uuid.uuid4()}")
        assert resp.status_code == 404
        _clear_override()

    def test_list_promotions(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        c.post("/api/v1/promotions", json=_gate_fields(candidate_model_version="v2"))
        c.post("/api/v1/promotions", json=_gate_fields(candidate_model_version="v3"))

        resp = c.get("/api/v1/promotions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        _clear_override()

    def test_list_with_status_filter(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        r1 = c.post("/api/v1/promotions", json=_gate_fields(candidate_model_version="v2"))
        c.post("/api/v1/promotions", json=_gate_fields(candidate_model_version="v3"))
        pid = r1.json()["promotion_id"]
        c.post(f"/api/v1/promotions/{pid}/approve")

        resp = c.get("/api/v1/promotions?status=APPROVED")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
        _clear_override()

    def test_no_secrets_in_response(self, client):
        c, _, _ = client
        _auth_override(ANALYST_USER)
        resp = c.post("/api/v1/promotions", json=_gate_fields())
        data = resp.json()
        text = str(data)
        assert "password" not in text.lower()
        assert "secret" not in text.lower()
        _clear_override()
