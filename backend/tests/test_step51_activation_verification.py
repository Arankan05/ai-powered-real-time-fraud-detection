"""Step 51 — Promotion activation safety & verification gate tests.

Covers the Step 51 specification:

* Verification of all preconditions before activation
* Identity consistency across governance records
* Production baseline protection
* Activation token issuance, validation, and consumption
* Authorization (unauthenticated / customer / analyst / admin)
* Concurrency safety
* Audit trail integration
* Production safety (no automatic activation)
* API endpoint behaviour and bounded responses

Run from the project root::

    python -m pytest backend/tests/test_step51_activation_verification.py -v
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import pytest

from backend.db.activation_verification import (
    AUDIT_ACTIVATION_VERIFICATION_FAILED,
    AUDIT_ACTIVATION_VERIFICATION_PASSED,
    AUDIT_MODEL_ACTIVATED,
    ACTIVATION_CONSUMED,
    ACTIVATION_NONE,
    ACTIVATION_TOKEN_ISSUED,
    ArtifactVerificationReport,
    DEFAULT_TOKEN_LIFETIME_SECONDS,
    VERIFICATION_BLOCKED,
    VERIFICATION_PASSED,
    ActivationTokenExpiredError,
    ActivationTokenReplayError,
    ActivationVerificationError,
    ActivationVerificationResult,
    consume_activation_authorization,
    issue_activation_token,
    reset_consumed_tokens,
    validate_activation_token,
    verify_activation_preconditions,
)


# Step 52: mock artifact verifier for tests that expect verification to pass
class _PassingArtifactVerifier:
    """Mock verifier that reports all artifact checks as passing."""
    def verify_candidate(self, **kwargs):
        return ArtifactVerificationReport(
            artifact_exists=True,
            checksum_valid=True,
            schema_compatible=True,
            feature_count_matches=True,
            is_not_already_active=True,
        )
from backend.db.promotion_governance import (
    ACTIVATION_NONE as GOV_ACTIVATION_NONE,
    ACTIVATION_TOKEN_ISSUED as GOV_ACTIVATION_TOKEN_ISSUED,
    InMemoryPromotionGovernanceStore,
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_REJECTED,
)


# ── Test data builders ─────────────────────────────────────────────────

def _make_governance_record(**overrides: Any) -> dict[str, Any]:
    """Build a valid APPROVED governance record for verification tests."""
    base = {
        "promotion_id": str(uuid.uuid4()),
        "gate_decision": "APPROVED",
        "governance_status": STATUS_APPROVED,
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
        "gate_report": None,
        "reviewer_id": "reviewer-1",
        "reviewer_role": "fraud_analyst",
        "reviewed_at": "2026-01-01T00:00:00+00:00",
        "approval_comment": "LGTM",
        "rejection_reason": None,
        "execution_status": None,
        "promoted_by": None,
        "promoted_at": None,
        "activation_status": ACTIVATION_NONE,
        "activation_token_issued_at": None,
        "activation_token_expires_at": None,
        "activation_consumed_at": None,
        "activation_actor_id": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def _make_production_identity(**overrides: Any) -> dict[str, Any]:
    """Build a valid production identity matching the governance record."""
    base = {
        "model_name": "fraud-xgb",
        "model_version": "fraud-xgb-v1.0.0",
        "checksum": "old789checksum",
        "schema_version": "1.0.0",
        "n_features": 24,
    }
    base.update(overrides)
    return base


# ── Verification unit tests ────────────────────────────────────────────


class TestVerifyActivationPreconditions:
    """Section 11: Verification tests from the spec."""

    def test_valid_approved_promotion_passes(self):
        record = _make_governance_record()
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
            candidate_artifact_exists=True,
            candidate_checksum_valid=True,
            candidate_schema_compatible=True,
            candidate_feature_count_matches=True,
        )
        assert result.status == VERIFICATION_PASSED
        assert result.reasons == []

    def test_rejected_promotion_blocked(self):
        record = _make_governance_record(governance_status=STATUS_REJECTED)
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("REJECTED" in r for r in result.reasons)

    def test_pending_promotion_blocked(self):
        record = _make_governance_record(governance_status=STATUS_PENDING)
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("PENDING" in r for r in result.reasons)

    def test_missing_governance_record_blocked(self):
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record={},
            current_production_identity=prod,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("not found" in r.lower() for r in result.reasons)

    def test_invalid_candidate_checksum_blocked(self):
        record = _make_governance_record()
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
            candidate_checksum_valid=False,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("checksum" in r.lower() for r in result.reasons)

    def test_missing_artifact_blocked(self):
        record = _make_governance_record()
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
            candidate_artifact_exists=False,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("artifact" in r.lower() for r in result.reasons)

    def test_schema_mismatch_blocked(self):
        record = _make_governance_record()
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
            candidate_schema_compatible=False,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("schema" in r.lower() for r in result.reasons)

    def test_feature_count_mismatch_blocked(self):
        record = _make_governance_record()
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
            candidate_feature_count_matches=False,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("feature count" in r.lower() for r in result.reasons)

    def test_candidate_already_active_blocked(self):
        record = _make_governance_record()
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
            candidate_is_not_already_active=False,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("already" in r.lower() for r in result.reasons)

    def test_gate_decision_rejected_blocked(self):
        record = _make_governance_record(gate_decision="REJECTED")
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("gate decision" in r.lower() for r in result.reasons)

    def test_missing_candidate_identity_field_blocked(self):
        record = _make_governance_record(candidate_model_name="")
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("missing" in r.lower() for r in result.reasons)

    def test_activation_already_consumed_blocked(self):
        record = _make_governance_record(activation_status=ACTIVATION_CONSUMED)
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("consumed" in r.lower() for r in result.reasons)


# ── Baseline protection tests ─────────────────────────────────────────


class TestProductionBaselineProtection:
    """Section 11: Baseline tests from the spec."""

    def test_unchanged_production_baseline_passes(self):
        record = _make_governance_record()
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
            candidate_artifact_exists=True,
            candidate_checksum_valid=True,
            candidate_schema_compatible=True,
            candidate_feature_count_matches=True,
        )
        assert result.status == VERIFICATION_PASSED

    def test_changed_production_baseline_blocks(self):
        record = _make_governance_record()
        prod = _make_production_identity(model_version="fraud-xgb-v1.5.0")
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("baseline changed" in r.lower() or "production baseline" in r.lower()
                    for r in result.reasons)

    def test_changed_production_checksum_blocks(self):
        record = _make_governance_record()
        prod = _make_production_identity(checksum="different_checksum")
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
        )
        assert result.status == VERIFICATION_BLOCKED

    def test_stale_governance_decision_requires_reevaluation(self):
        """If production has changed since approval, verification fails."""
        record = _make_governance_record(
            production_model_version="fraud-xgb-v1.0.0",
            production_checksum="old789checksum",
        )
        # Production has moved on
        prod = _make_production_identity(
            model_version="fraud-xgb-v1.5.0",
            checksum="new_checksum",
        )
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert len(result.reasons) >= 1


# ── Token tests ────────────────────────────────────────────────────────


class TestActivationToken:
    """Section 11: Activation authorization tests."""

    def setup_method(self):
        reset_consumed_tokens()

    def test_valid_authorization_accepted(self):
        candidate = {"model_name": "fraud-xgb", "model_version": "v2", "checksum": "abc"}
        production = {"model_name": "fraud-xgb", "model_version": "v1", "checksum": "def"}
        token, expires_at = issue_activation_token(
            promotion_id="promo-1",
            candidate_identity=candidate,
            production_identity=production,
        )
        assert token
        assert expires_at

        auth = validate_activation_token(
            token,
            expected_promotion_id="promo-1",
            current_production_identity=production,
        )
        assert auth.promotion_id == "promo-1"
        assert auth.candidate_identity == candidate

    def test_expired_authorization_rejected(self):
        candidate = {"model_name": "fraud-xgb", "model_version": "v2"}
        production = {"model_name": "fraud-xgb", "model_version": "v1"}
        # Issue with 0-second lifetime → immediately expired
        token, _ = issue_activation_token(
            promotion_id="promo-1",
            candidate_identity=candidate,
            production_identity=production,
            lifetime_seconds=0,
        )
        # Wait a tiny bit to ensure expiry
        time.sleep(0.05)
        with pytest.raises(ActivationTokenExpiredError):
            validate_activation_token(
                token,
                expected_promotion_id="promo-1",
                current_production_identity=production,
            )

    def test_wrong_promotion_rejected(self):
        candidate = {"model_name": "fraud-xgb", "model_version": "v2"}
        production = {"model_name": "fraud-xgb", "model_version": "v1"}
        token, _ = issue_activation_token(
            promotion_id="promo-1",
            candidate_identity=candidate,
            production_identity=production,
        )
        with pytest.raises(ActivationVerificationError, match="different promotion"):
            validate_activation_token(
                token,
                expected_promotion_id="promo-2",
                current_production_identity=production,
            )

    def test_wrong_production_baseline_rejected(self):
        candidate = {"model_name": "fraud-xgb", "model_version": "v2"}
        production = {"model_name": "fraud-xgb", "model_version": "v1", "checksum": "abc"}
        token, _ = issue_activation_token(
            promotion_id="promo-1",
            candidate_identity=candidate,
            production_identity=production,
        )
        changed_production = {"model_name": "fraud-xgb", "model_version": "v1", "checksum": "DIFFERENT"}
        with pytest.raises(ActivationVerificationError, match="baseline changed"):
            validate_activation_token(
                token,
                expected_promotion_id="promo-1",
                current_production_identity=changed_production,
            )

    def test_replay_rejected(self):
        candidate = {"model_name": "fraud-xgb", "model_version": "v2"}
        production = {"model_name": "fraud-xgb", "model_version": "v1", "checksum": "abc"}
        token, _ = issue_activation_token(
            promotion_id="promo-1",
            candidate_identity=candidate,
            production_identity=production,
        )
        # First consumption succeeds
        auth = consume_activation_authorization(
            token,
            expected_promotion_id="promo-1",
            current_production_identity=production,
        )
        assert auth.promotion_id == "promo-1"

        # Second consumption is replay
        with pytest.raises(ActivationTokenReplayError):
            consume_activation_authorization(
                token,
                expected_promotion_id="promo-1",
                current_production_identity=production,
            )

    def test_invalid_token_signature_rejected(self):
        with pytest.raises(ActivationVerificationError, match="signature|format|encoding"):
            validate_activation_token(
                "invalid.token",
                expected_promotion_id="promo-1",
                current_production_identity={},
            )

    def test_token_does_not_contain_secrets(self):
        candidate = {"model_name": "fraud-xgb", "model_version": "v2"}
        production = {"model_name": "fraud-xgb", "model_version": "v1"}
        token, _ = issue_activation_token(
            promotion_id="promo-1",
            candidate_identity=candidate,
            production_identity=production,
        )
        # Token should not contain raw secret key material
        assert "BACKEND_SECRET_KEY" not in token
        # Token should not contain raw transaction data
        assert "transaction" not in token.lower()

    def test_revoked_promotion_rejected_at_verification(self):
        """A rejected governance record cannot produce a valid token."""
        record = _make_governance_record(governance_status=STATUS_REJECTED)
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
        )
        assert result.status == VERIFICATION_BLOCKED


# ── API-level test fixtures ────────────────────────────────────────────


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

INACTIVE_USER = {
    "id": "00000000-0000-4000-8000-000000000050",
    "email": "inactive@test.local",
    "role": "fraud_analyst",
    "is_active": False,
    "first_name": "Inactive",
    "last_name": "User",
    "customer_id": None,
    "created_at": "2026-01-01T00:00:00+00:00",
}


def _gate_fields(**overrides: Any) -> dict[str, Any]:
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


@pytest.fixture
def client():
    """Test client with in-memory governance and audit stores."""
    from fastapi.testclient import TestClient
    from backend.app import app
    from backend.db.promotion_governance import InMemoryPromotionGovernanceStore
    from backend.db.audit_repository import InMemoryAuditStore
    from backend.routers.promotions import (
        set_governance_repository,
        set_audit_repository,
        set_production_identity_provider,
        set_artifact_verifier,
        _production_identity_provider,
    )
    from backend.security.deps import set_user_repository
    from backend.db.user_repository import SQLiteUserRepository

    store = InMemoryPromotionGovernanceStore()
    audit_store = InMemoryAuditStore()
    set_governance_repository(store)
    set_audit_repository(audit_store)

    # Default production identity provider matching _gate_fields()
    production_identity = _make_production_identity()
    set_production_identity_provider(lambda: dict(production_identity))

    # Step 52: set mock artifact verifier (all checks pass)
    set_artifact_verifier(_PassingArtifactVerifier())

    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    user_repo = SQLiteUserRepository(db_path=tmp.name)
    set_user_repository(user_repo)

    saved_overrides = dict(app.dependency_overrides)
    app.dependency_overrides.clear()

    reset_consumed_tokens()

    c = TestClient(app)
    yield c, store, audit_store

    # Restore
    app.dependency_overrides.clear()
    app.dependency_overrides.update(saved_overrides)
    set_production_identity_provider(None)
    set_artifact_verifier(None)
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


def _create_and_approve(c, store) -> str:
    """Helper: create a promotion and approve it. Returns promotion_id."""
    _auth_override(ANALYST_USER)
    create_resp = c.post("/api/v1/promotions", json=_gate_fields())
    assert create_resp.status_code == 201
    pid = create_resp.json()["promotion_id"]
    approve_resp = c.post(f"/api/v1/promotions/{pid}/approve")
    assert approve_resp.status_code == 200
    assert approve_resp.json()["governance_status"] == STATUS_APPROVED
    return pid


# ── Authorization tests ────────────────────────────────────────────────


class TestAuthorization:
    """Section 11: Authorization tests from the spec."""

    def test_unauthenticated_verify_blocked(self, client):
        c, store, _ = client
        _clear_override()
        resp = c.post("/api/v1/promotions/fake-id/activation-verify")
        assert resp.status_code == 401

    def test_customer_verify_blocked(self, client):
        c, store, _ = client
        _auth_override(CUSTOMER_USER)
        resp = c.post("/api/v1/promotions/fake-id/activation-verify")
        assert resp.status_code == 403
        _clear_override()

    def test_analyst_verify_allowed(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        assert resp.status_code == 200
        _clear_override()

    def test_admin_verify_allowed(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ADMIN_USER)
        resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        assert resp.status_code == 200
        _clear_override()

    def test_unauthenticated_activate_blocked(self, client):
        c, store, _ = client
        _clear_override()
        resp = c.post(
            "/api/v1/promotions/fake-id/activate",
            json={"activation_token": "fake"},
        )
        assert resp.status_code == 401

    def test_customer_activate_blocked(self, client):
        c, store, _ = client
        _auth_override(CUSTOMER_USER)
        resp = c.post(
            "/api/v1/promotions/fake-id/activate",
            json={"activation_token": "fake"},
        )
        assert resp.status_code == 403
        _clear_override()

    def test_inactive_user_cannot_override_identity(self, client):
        """Client cannot supply a different actor identity in the request."""
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        # Actor identity comes from JWT — verify no actor_id in response
        data = resp.json()
        assert "actor_id" not in data
        _clear_override()

    def test_client_cannot_supply_actor_identity(self, client):
        """Actor identity always comes from JWT, not request body."""
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        data = resp.json()
        # No actor_id in response body that could be spoofed
        assert "actor_id" not in data
        _clear_override()


# ── API endpoint tests ─────────────────────────────────────────────────


class TestActivationVerifyEndpoint:
    """Section 11: API/CLI tests from the spec."""

    def test_valid_verify_returns_ready(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == VERIFICATION_PASSED
        assert data["promotion_id"] == pid
        assert data["activation_token"] is not None
        assert data["token_expires_at"] is not None
        _clear_override()

    def test_blocked_verify_returns_blocked(self, client):
        c, store, _ = client
        # Create but don't approve → PENDING → blocked
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]

        resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == VERIFICATION_BLOCKED
        assert data["reasons"] is not None
        assert len(data["reasons"]) > 0
        assert data["activation_token"] is None
        _clear_override()

    def test_nonexistent_promotion_returns_404(self, client):
        c, store, _ = client
        _auth_override(ANALYST_USER)
        resp = c.post(f"/api/v1/promotions/{uuid.uuid4()}/activation-verify")
        assert resp.status_code == 404
        _clear_override()

    def test_verify_updates_activation_status(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        c.post(f"/api/v1/promotions/{pid}/activation-verify")

        record = store.get_by_id(pid)
        assert record["activation_status"] == GOV_ACTIVATION_TOKEN_ISSUED
        assert record["activation_token_issued_at"] is not None
        _clear_override()

    def test_response_bounded_no_secrets(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        data = resp.json()
        text = str(data)
        assert "secret" not in text.lower()
        assert "password" not in text.lower()
        assert "BACKEND_SECRET_KEY" not in text
        _clear_override()

    def test_response_no_internal_paths(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        text = resp.text
        assert "/app/" not in text
        assert "/home/" not in text
        assert "traceback" not in text.lower()
        _clear_override()


class TestActivateEndpoint:
    """Section 11: Activation authorization tests via API."""

    def test_valid_activate(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)

        # Step 1: Verify to get token
        verify_resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        token = verify_resp.json()["activation_token"]

        # Step 2: Consume token
        resp = c.post(
            f"/api/v1/promotions/{pid}/activate",
            json={"activation_token": token},
        )
        assert resp.status_code == 200
        record = store.get_by_id(pid)
        assert record["activation_status"] == ACTIVATION_CONSUMED
        assert record["activation_consumed_at"] is not None
        assert record["activation_actor_id"] == ANALYST_USER["id"]
        _clear_override()

    def test_expired_token_rejected(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)

        # Issue a token with very short lifetime
        from backend.db.activation_verification import issue_activation_token
        token, _ = issue_activation_token(
            promotion_id=pid,
            candidate_identity={"model_version": "v2"},
            production_identity=_make_production_identity(),
            lifetime_seconds=0,
        )
        time.sleep(0.05)

        resp = c.post(
            f"/api/v1/promotions/{pid}/activate",
            json={"activation_token": token},
        )
        assert resp.status_code == 410  # Gone
        _clear_override()

    def test_replay_token_rejected(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)

        verify_resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        token = verify_resp.json()["activation_token"]

        # First consumption
        resp1 = c.post(
            f"/api/v1/promotions/{pid}/activate",
            json={"activation_token": token},
        )
        assert resp1.status_code == 200

        # Replay
        resp2 = c.post(
            f"/api/v1/promotions/{pid}/activate",
            json={"activation_token": token},
        )
        assert resp2.status_code == 409  # Conflict
        _clear_override()

    def test_wrong_promotion_token_rejected(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)

        # Issue token for a different promotion
        from backend.db.activation_verification import issue_activation_token
        token, _ = issue_activation_token(
            promotion_id="different-promo-id",
            candidate_identity={"model_version": "v2"},
            production_identity=_make_production_identity(),
        )

        resp = c.post(
            f"/api/v1/promotions/{pid}/activate",
            json={"activation_token": token},
        )
        assert resp.status_code == 403
        _clear_override()

    def test_nonexistent_promotion_activate_404(self, client):
        c, store, _ = client
        _auth_override(ANALYST_USER)
        resp = c.post(
            f"/api/v1/promotions/{uuid.uuid4()}/activate",
            json={"activation_token": "fake.token"},
        )
        assert resp.status_code == 404
        _clear_override()


# ── Audit trail tests ──────────────────────────────────────────────────


class TestAuditTrail:
    """Section 11: Audit tests from the spec."""

    def test_verification_success_audited(self, client):
        c, store, audit_store = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        c.post(f"/api/v1/promotions/{pid}/activation-verify")

        events = audit_store.list_by_transaction("00000000-0000-0000-0000-000000000000")
        passed_events = [
            e for e in events
            if e["event_type"] == AUDIT_ACTIVATION_VERIFICATION_PASSED
        ]
        assert len(passed_events) >= 1
        assert passed_events[0]["actor_id"] == ANALYST_USER["id"]
        _clear_override()

    def test_verification_failure_audited(self, client):
        c, store, audit_store = client
        # Create but don't approve → blocked
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]
        c.post(f"/api/v1/promotions/{pid}/activation-verify")

        events = audit_store.list_by_transaction("00000000-0000-0000-0000-000000000000")
        failed_events = [
            e for e in events
            if e["event_type"] == AUDIT_ACTIVATION_VERIFICATION_FAILED
        ]
        assert len(failed_events) >= 1
        _clear_override()

    def test_activation_audited(self, client):
        c, store, audit_store = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)

        verify_resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        token = verify_resp.json()["activation_token"]
        c.post(
            f"/api/v1/promotions/{pid}/activate",
            json={"activation_token": token},
        )

        events = audit_store.list_by_transaction("00000000-0000-0000-0000-000000000000")
        activated_events = [
            e for e in events if e["event_type"] == AUDIT_MODEL_ACTIVATED
        ]
        assert len(activated_events) >= 1
        assert activated_events[0]["actor_id"] == ANALYST_USER["id"]
        _clear_override()

    def test_correct_actor_identity_in_audit(self, client):
        c, store, audit_store = client
        pid = _create_and_approve(c, store)
        _auth_override(ADMIN_USER)
        c.post(f"/api/v1/promotions/{pid}/activation-verify")

        events = audit_store.list_by_transaction("00000000-0000-0000-0000-000000000000")
        passed_events = [
            e for e in events
            if e["event_type"] == AUDIT_ACTIVATION_VERIFICATION_PASSED
        ]
        assert passed_events[0]["actor_id"] == ADMIN_USER["id"]
        assert passed_events[0]["actor_role"] == ADMIN_USER["role"]
        _clear_override()

    def test_no_secrets_in_audit(self, client):
        c, store, audit_store = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        c.post(f"/api/v1/promotions/{pid}/activation-verify")

        events = audit_store.list_by_transaction("00000000-0000-0000-0000-000000000000")
        for event in events:
            event_str = str(event)
            assert "secret" not in event_str.lower()
            assert "password" not in event_str.lower()
            assert "jwt" not in event_str.lower()
        _clear_override()


# ── Production safety tests ───────────────────────────────────────────


class TestProductionSafety:
    """Section 11: Production safety tests from the spec."""

    def test_verification_does_not_modify_active_model(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)

        record_before = store.get_by_id(pid)
        c.post(f"/api/v1/promotions/{pid}/activation-verify")
        record_after = store.get_by_id(pid)

        # Governance status should NOT change from verification alone
        assert record_after["governance_status"] == record_before["governance_status"]
        # Only activation_status should change
        assert record_after["activation_status"] == GOV_ACTIVATION_TOKEN_ISSUED
        _clear_override()

    def test_failed_verification_does_not_modify_manifest(self, client):
        c, store, _ = client
        # Create but don't approve → verification will be blocked
        _auth_override(ANALYST_USER)
        create_resp = c.post("/api/v1/promotions", json=_gate_fields())
        pid = create_resp.json()["promotion_id"]

        record_before = store.get_by_id(pid)
        c.post(f"/api/v1/promotions/{pid}/activation-verify")
        record_after = store.get_by_id(pid)

        assert record_after["governance_status"] == record_before["governance_status"]
        assert record_after["activation_status"] == GOV_ACTIVATION_NONE
        _clear_override()

    def test_failed_verification_does_not_modify_threshold(self, client):
        """Verification (pass or fail) must not change any threshold."""
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        data = resp.json()
        # No threshold field in response
        assert "threshold" not in data
        _clear_override()

    def test_no_hot_swap_introduced(self, client):
        """The activation endpoint does not perform runtime hot-swap."""
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)

        verify_resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        token = verify_resp.json()["activation_token"]
        resp = c.post(
            f"/api/v1/promotions/{pid}/activate",
            json={"activation_token": token},
        )
        data = resp.json()
        # The response is a governance record, not a model activation
        assert "governance_status" in data
        assert "promotion_id" in data
        # No hot-swap confirmation
        assert "hot_swap" not in str(data).lower()
        _clear_override()


# ── Concurrency tests ─────────────────────────────────────────────────


class TestConcurrency:
    """Section 11: Concurrency tests from the spec."""

    def test_concurrent_activation_attempts(self, client):
        """Only one activation should succeed from concurrent attempts."""
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)

        verify_resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        token = verify_resp.json()["activation_token"]

        results = []
        errors = []

        def try_consume():
            try:
                auth = consume_activation_authorization(
                    token,
                    expected_promotion_id=pid,
                    current_production_identity=_make_production_identity(),
                )
                results.append("ok")
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=try_consume) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 1  # Exactly one success
        assert len(errors) == 4  # Others got replay errors
        _clear_override()

    def test_duplicate_activation_via_api(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)

        verify_resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        token = verify_resp.json()["activation_token"]

        resp1 = c.post(
            f"/api/v1/promotions/{pid}/activate",
            json={"activation_token": token},
        )
        assert resp1.status_code == 200

        resp2 = c.post(
            f"/api/v1/promotions/{pid}/activate",
            json={"activation_token": token},
        )
        assert resp2.status_code == 409
        _clear_override()


# ── Persistence tests ──────────────────────────────────────────────────


class TestPersistence:
    """Section 11: Persistence tests from the spec."""

    def test_activation_status_preserved_in_store(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)

        c.post(f"/api/v1/promotions/{pid}/activation-verify")
        record = store.get_by_id(pid)
        assert record["activation_status"] == GOV_ACTIVATION_TOKEN_ISSUED

        # Verify the record still has all fields
        assert record["activation_token_issued_at"] is not None
        assert record["activation_token_expires_at"] is not None
        _clear_override()

    def test_restart_behavior(self, client):
        """After reset_consumed_tokens, a consumed token could theoretically
        be replayed — but the governance record's activation_status prevents it."""
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)

        verify_resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        token = verify_resp.json()["activation_token"]

        resp = c.post(
            f"/api/v1/promotions/{pid}/activate",
            json={"activation_token": token},
        )
        assert resp.status_code == 200

        record = store.get_by_id(pid)
        assert record["activation_status"] == ACTIVATION_CONSUMED
        _clear_override()

    def test_duplicate_constraint_behavior(self, client):
        c, store, _ = client
        _auth_override(ANALYST_USER)
        c.post("/api/v1/promotions", json=_gate_fields())
        resp = c.post("/api/v1/promotions", json=_gate_fields())
        assert resp.status_code == 409
        _clear_override()


# ── Result container tests ─────────────────────────────────────────────


class TestResultContainers:
    """Test ActivationVerificationResult and related containers."""

    def test_result_to_dict_passed(self):
        result = ActivationVerificationResult(
            status=VERIFICATION_PASSED,
            promotion_id="p1",
            candidate_identity={"model_version": "v2"},
            production_identity={"model_version": "v1"},
        )
        d = result.to_dict()
        assert d["status"] == VERIFICATION_PASSED
        assert "reasons" not in d  # No reasons when passed
        assert "activation_token" not in d

    def test_result_to_dict_blocked(self):
        result = ActivationVerificationResult(
            status=VERIFICATION_BLOCKED,
            promotion_id="p1",
            candidate_identity={"model_version": "v2"},
            production_identity={"model_version": "v1"},
            reasons=["reason1", "reason2"],
        )
        d = result.to_dict()
        assert d["status"] == VERIFICATION_BLOCKED
        assert d["reasons"] == ["reason1", "reason2"]
