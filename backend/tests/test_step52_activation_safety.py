"""Step 52 — Final activation safety hardening tests.

Covers the Step 52 specification:

* Artifact verification is fail-closed (never assumed True)
* Token consumption is persistent / restart-safe
* Atomic state transitions (compare-and-set)
* All Step 51 invariants preserved

Run from the project root::

    python -m pytest backend/tests/test_step52_activation_safety.py -v
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import pytest

from backend.db.activation_verification import (
    ACTIVATION_CONSUMED,
    ACTIVATION_NONE,
    ACTIVATION_TOKEN_ISSUED,
    ArtifactVerificationReport,
    ArtifactVerifier,
    DefaultArtifactVerifier,
    NoOpArtifactVerifier,
    VERIFICATION_BLOCKED,
    VERIFICATION_PASSED,
    ActivationTokenExpiredError,
    ActivationTokenReplayError,
    ActivationVerificationError,
    consume_activation_authorization,
    issue_activation_token,
    reset_consumed_tokens,
    validate_activation_token,
    verify_activation_preconditions,
)
from backend.db.promotion_governance import (
    InMemoryPromotionGovernanceStore,
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_REJECTED,
)


# ── Test data builders ─────────────────────────────────────────────────


def _make_governance_record(**overrides: Any) -> dict[str, Any]:
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
    base = {
        "model_name": "fraud-xgb",
        "model_version": "fraud-xgb-v1.0.0",
        "checksum": "old789checksum",
        "schema_version": "1.0.0",
        "n_features": 24,
    }
    base.update(overrides)
    return base


# ── Mock artifact verifiers ────────────────────────────────────────────


class _PassingVerifier:
    """All artifact checks pass."""
    def verify_candidate(self, **kwargs):
        return ArtifactVerificationReport(
            artifact_exists=True, checksum_valid=True,
            schema_compatible=True, feature_count_matches=True,
            is_not_already_active=True,
        )


class _FailingVerifier:
    """All artifact checks fail."""
    def verify_candidate(self, **kwargs):
        return ArtifactVerificationReport(
            artifact_exists=False, checksum_valid=False,
            schema_compatible=False, feature_count_matches=False,
        )


class _MissingArtifactVerifier:
    """Artifact missing (exists=False)."""
    def verify_candidate(self, **kwargs):
        return ArtifactVerificationReport(
            artifact_exists=False, checksum_valid=False,
            schema_compatible=False, feature_count_matches=False,
        )


class _BadChecksumVerifier:
    """Artifact exists but checksum is wrong."""
    def verify_candidate(self, **kwargs):
        return ArtifactVerificationReport(
            artifact_exists=True, checksum_valid=False,
            schema_compatible=True, feature_count_matches=True,
        )


class _SchemaMismatchVerifier:
    """Artifact exists, checksum ok, but schema incompatible."""
    def verify_candidate(self, **kwargs):
        return ArtifactVerificationReport(
            artifact_exists=True, checksum_valid=True,
            schema_compatible=False, feature_count_matches=True,
        )


# ── Artifact verification tests ────────────────────────────────────────


class TestFailClosedDefaults:
    """Section 7: Artifact verification tests."""

    def test_default_flags_block_activation(self):
        """Without explicit artifact flags, verification blocks."""
        record = _make_governance_record()
        prod = _make_production_identity()
        result = verify_activation_preconditions(
            governance_record=record,
            current_production_identity=prod,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("artifact" in r.lower() for r in result.reasons)

    def test_missing_artifact_blocks(self):
        result = verify_activation_preconditions(
            governance_record=_make_governance_record(),
            current_production_identity=_make_production_identity(),
            candidate_artifact_exists=False,
            candidate_checksum_valid=True,
            candidate_schema_compatible=True,
            candidate_feature_count_matches=True,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("artifact" in r.lower() for r in result.reasons)

    def test_invalid_checksum_blocks(self):
        result = verify_activation_preconditions(
            governance_record=_make_governance_record(),
            current_production_identity=_make_production_identity(),
            candidate_artifact_exists=True,
            candidate_checksum_valid=False,
            candidate_schema_compatible=True,
            candidate_feature_count_matches=True,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("checksum" in r.lower() for r in result.reasons)

    def test_valid_artifact_passes(self):
        result = verify_activation_preconditions(
            governance_record=_make_governance_record(),
            current_production_identity=_make_production_identity(),
            candidate_artifact_exists=True,
            candidate_checksum_valid=True,
            candidate_schema_compatible=True,
            candidate_feature_count_matches=True,
        )
        assert result.status == VERIFICATION_PASSED

    def test_schema_mismatch_blocks(self):
        result = verify_activation_preconditions(
            governance_record=_make_governance_record(),
            current_production_identity=_make_production_identity(),
            candidate_artifact_exists=True,
            candidate_checksum_valid=True,
            candidate_schema_compatible=False,
            candidate_feature_count_matches=True,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("schema" in r.lower() for r in result.reasons)

    def test_feature_count_mismatch_blocks(self):
        result = verify_activation_preconditions(
            governance_record=_make_governance_record(),
            current_production_identity=_make_production_identity(),
            candidate_artifact_exists=True,
            candidate_checksum_valid=True,
            candidate_schema_compatible=True,
            candidate_feature_count_matches=False,
        )
        assert result.status == VERIFICATION_BLOCKED
        assert any("feature count" in r.lower() for r in result.reasons)

    def test_caller_cannot_bypass_with_defaults(self):
        """Default call (no artifact flags) always blocks."""
        result = verify_activation_preconditions(
            governance_record=_make_governance_record(),
            current_production_identity=_make_production_identity(),
        )
        assert result.status == VERIFICATION_BLOCKED


class TestNoOpArtifactVerifier:
    """NoOpArtifactVerifier always fails closed."""

    def test_noop_always_fails(self):
        verifier = NoOpArtifactVerifier()
        report = verifier.verify_candidate(
            candidate_model_version="v2",
            candidate_checksum="abc",
            candidate_schema_version="1.0.0",
            candidate_n_features=24,
            current_production_identity={},
        )
        assert report.artifact_exists is False
        assert report.checksum_valid is False


class TestDefaultArtifactVerifier:
    """DefaultArtifactVerifier with no model directory."""

    def test_no_model_dir_fails_closed(self):
        verifier = DefaultArtifactVerifier(model_directory="/nonexistent/path")
        report = verifier.verify_candidate(
            candidate_model_version="v2",
            candidate_checksum="abc",
            candidate_schema_version="1.0.0",
            candidate_n_features=24,
            current_production_identity={},
        )
        assert report.artifact_exists is False


# ── Persistent token consumption tests ─────────────────────────────────


class TestPersistentConsumption:
    """Section 7: Persistence tests."""

    def setup_method(self):
        reset_consumed_tokens()

    def test_consumed_state_persists_in_governance_record(self):
        """After consumption, governance record shows CONSUMED."""
        store = InMemoryPromotionGovernanceStore()
        record = store.create(
            gate_decision="APPROVED",
            candidate_model_name="fraud-xgb",
            candidate_model_version="v2",
            candidate_checksum="abc",
            candidate_schema_version="1.0.0",
            candidate_n_features=24,
            production_model_name="fraud-xgb",
            production_model_version="v1",
            production_checksum="def",
            production_schema_version="1.0.0",
            production_n_features=24,
        )
        pid = record["promotion_id"]
        store.approve(pid, reviewer_id="u1", reviewer_role="admin")
        store.update_activation_status(pid, activation_status=ACTIVATION_TOKEN_ISSUED)

        # Simulate consumption via CAS
        result = store.try_transition_activation_status(
            pid,
            expected_status=ACTIVATION_TOKEN_ISSUED,
            new_status=ACTIVATION_CONSUMED,
            consumed_at="2026-01-01T00:00:00+00:00",
            actor_id="u1",
        )
        assert result is not None
        assert result["activation_status"] == ACTIVATION_CONSUMED

        # Simulate restart: clear in-memory consumed tokens
        reset_consumed_tokens()

        # Governance record still shows CONSUMED
        refreshed = store.get_by_id(pid)
        assert refreshed["activation_status"] == ACTIVATION_CONSUMED

    def test_consumed_token_rejected_after_restart(self):
        """After restart, governance record's CONSUMED status blocks replay."""
        store = InMemoryPromotionGovernanceStore()
        record = store.create(
            gate_decision="APPROVED",
            candidate_model_name="fraud-xgb",
            candidate_model_version="v2",
            candidate_checksum="abc",
            candidate_schema_version="1.0.0",
            candidate_n_features=24,
            production_model_name="fraud-xgb",
            production_model_version="v1",
            production_checksum="def",
            production_schema_version="1.0.0",
            production_n_features=24,
        )
        pid = record["promotion_id"]
        store.approve(pid, reviewer_id="u1", reviewer_role="admin")
        store.update_activation_status(pid, activation_status=ACTIVATION_CONSUMED)

        # Issue a token
        token, _ = issue_activation_token(
            promotion_id=pid,
            candidate_identity={"model_version": "v2"},
            production_identity=_make_production_identity(),
        )

        # Simulate restart
        reset_consumed_tokens()

        # Governance record check should reject
        refreshed = store.get_by_id(pid)
        with pytest.raises(ActivationTokenReplayError):
            consume_activation_authorization(
                token,
                expected_promotion_id=pid,
                current_production_identity=_make_production_identity(),
                governance_record=refreshed,
            )

    def test_consumption_failure_does_not_mark_consumed(self):
        """If consumption fails (e.g. wrong promotion), status stays TOKEN_ISSUED."""
        store = InMemoryPromotionGovernanceStore()
        record = store.create(
            gate_decision="APPROVED",
            candidate_model_name="fraud-xgb",
            candidate_model_version="v2",
            candidate_checksum="abc",
            candidate_schema_version="1.0.0",
            candidate_n_features=24,
            production_model_name="fraud-xgb",
            production_model_version="v1",
            production_checksum="def",
            production_schema_version="1.0.0",
            production_n_features=24,
        )
        pid = record["promotion_id"]
        store.approve(pid, reviewer_id="u1", reviewer_role="admin")
        store.update_activation_status(pid, activation_status=ACTIVATION_TOKEN_ISSUED)

        # Issue token for a DIFFERENT promotion
        token, _ = issue_activation_token(
            promotion_id="different-pid",
            candidate_identity={"model_version": "v2"},
            production_identity=_make_production_identity(),
        )

        # Try to consume against the wrong promotion
        refreshed = store.get_by_id(pid)
        with pytest.raises(ActivationVerificationError):
            consume_activation_authorization(
                token,
                expected_promotion_id=pid,
                current_production_identity=_make_production_identity(),
                governance_record=refreshed,
            )

        # Status should still be TOKEN_ISSUED
        after = store.get_by_id(pid)
        assert after["activation_status"] == ACTIVATION_TOKEN_ISSUED


# ── Atomic CAS tests ──────────────────────────────────────────────────


class TestAtomicCAS:
    """Section 7: Concurrency tests."""

    def test_cas_succeeds_once(self):
        store = InMemoryPromotionGovernanceStore()
        record = store.create(
            gate_decision="APPROVED",
            candidate_model_name="fraud-xgb",
            candidate_model_version="v2",
            candidate_checksum="abc",
            candidate_schema_version="1.0.0",
            candidate_n_features=24,
            production_model_name="fraud-xgb",
            production_model_version="v1",
            production_checksum="def",
            production_schema_version="1.0.0",
            production_n_features=24,
        )
        pid = record["promotion_id"]
        store.update_activation_status(pid, activation_status=ACTIVATION_TOKEN_ISSUED)

        r1 = store.try_transition_activation_status(
            pid, expected_status=ACTIVATION_TOKEN_ISSUED, new_status=ACTIVATION_CONSUMED,
        )
        assert r1 is not None
        assert r1["activation_status"] == ACTIVATION_CONSUMED

        r2 = store.try_transition_activation_status(
            pid, expected_status=ACTIVATION_TOKEN_ISSUED, new_status=ACTIVATION_CONSUMED,
        )
        assert r2 is None  # CAS failed

    def test_concurrent_cas_exactly_one_succeeds(self):
        store = InMemoryPromotionGovernanceStore()
        record = store.create(
            gate_decision="APPROVED",
            candidate_model_name="fraud-xgb",
            candidate_model_version="v2",
            candidate_checksum="abc",
            candidate_schema_version="1.0.0",
            candidate_n_features=24,
            production_model_name="fraud-xgb",
            production_model_version="v1",
            production_checksum="def",
            production_schema_version="1.0.0",
            production_n_features=24,
        )
        pid = record["promotion_id"]
        store.update_activation_status(pid, activation_status=ACTIVATION_TOKEN_ISSUED)

        results = []
        failures = []

        def try_cas():
            r = store.try_transition_activation_status(
                pid, expected_status=ACTIVATION_TOKEN_ISSUED, new_status=ACTIVATION_CONSUMED,
            )
            if r is not None:
                results.append("ok")
            else:
                failures.append("cas_failed")

        threads = [threading.Thread(target=try_cas) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 1
        assert len(failures) == 4

    def test_wrong_expected_status_fails(self):
        store = InMemoryPromotionGovernanceStore()
        record = store.create(
            gate_decision="APPROVED",
            candidate_model_name="fraud-xgb",
            candidate_model_version="v2",
            candidate_checksum="abc",
            candidate_schema_version="1.0.0",
            candidate_n_features=24,
            production_model_name="fraud-xgb",
            production_model_version="v1",
            production_checksum="def",
            production_schema_version="1.0.0",
            production_n_features=24,
        )
        pid = record["promotion_id"]
        # Status is NONE, not TOKEN_ISSUED
        r = store.try_transition_activation_status(
            pid, expected_status=ACTIVATION_TOKEN_ISSUED, new_status=ACTIVATION_CONSUMED,
        )
        assert r is None


# ── API-level tests ────────────────────────────────────────────────────


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
    """Test client with configurable artifact verifier."""
    from fastapi.testclient import TestClient
    from backend.app import app
    from backend.db.promotion_governance import InMemoryPromotionGovernanceStore
    from backend.db.audit_repository import InMemoryAuditStore
    from backend.routers.promotions import (
        set_governance_repository,
        set_audit_repository,
        set_production_identity_provider,
        set_artifact_verifier,
    )
    from backend.security.deps import set_user_repository
    from backend.db.user_repository import SQLiteUserRepository

    store = InMemoryPromotionGovernanceStore()
    audit_store = InMemoryAuditStore()
    set_governance_repository(store)
    set_audit_repository(audit_store)

    production_identity = _make_production_identity()
    set_production_identity_provider(lambda: dict(production_identity))
    set_artifact_verifier(_PassingVerifier())

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
    _auth_override(ANALYST_USER)
    create_resp = c.post("/api/v1/promotions", json=_gate_fields())
    assert create_resp.status_code == 201
    pid = create_resp.json()["promotion_id"]
    approve_resp = c.post(f"/api/v1/promotions/{pid}/approve")
    assert approve_resp.status_code == 200
    return pid


class TestAPIArtifactVerification:
    """Section 7: API-level artifact verification tests."""

    def test_no_verifier_blocks(self, client):
        """Without a verifier, activation is blocked."""
        c, store, _ = client
        from backend.routers.promotions import set_artifact_verifier
        set_artifact_verifier(None)  # Remove verifier

        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        data = resp.json()
        assert data["status"] == VERIFICATION_BLOCKED
        assert any("artifact" in r.lower() for r in data["reasons"])
        _clear_override()

    def test_failing_verifier_blocks(self, client):
        """A verifier that reports all-false blocks activation."""
        c, store, _ = client
        from backend.routers.promotions import set_artifact_verifier
        set_artifact_verifier(_FailingVerifier())

        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        data = resp.json()
        assert data["status"] == VERIFICATION_BLOCKED
        _clear_override()

    def test_passing_verifier_succeeds(self, client):
        """A verifier that reports all-true passes activation."""
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        data = resp.json()
        assert data["status"] == VERIFICATION_PASSED
        assert data["activation_token"] is not None
        _clear_override()


class TestAPIPersistentConsumption:
    """Section 7: API-level persistent consumption tests."""

    def test_consumed_token_rejected_via_api(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)

        verify_resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        token = verify_resp.json()["activation_token"]

        # First consumption succeeds
        resp1 = c.post(
            f"/api/v1/promotions/{pid}/activate",
            json={"activation_token": token},
        )
        assert resp1.status_code == 200

        # Second consumption rejected
        resp2 = c.post(
            f"/api/v1/promotions/{pid}/activate",
            json={"activation_token": token},
        )
        assert resp2.status_code == 409
        _clear_override()

    def test_consumed_after_simulated_restart(self, client):
        """After clearing in-memory cache, governance record blocks replay."""
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)

        verify_resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        token = verify_resp.json()["activation_token"]

        # Consume
        resp = c.post(
            f"/api/v1/promotions/{pid}/activate",
            json={"activation_token": token},
        )
        assert resp.status_code == 200

        # Simulate restart
        reset_consumed_tokens()

        # Governance record still shows CONSUMED → blocked
        record = store.get_by_id(pid)
        assert record["activation_status"] == ACTIVATION_CONSUMED

        # Try to consume again via API
        resp2 = c.post(
            f"/api/v1/promotions/{pid}/activate",
            json={"activation_token": token},
        )
        assert resp2.status_code == 409
        _clear_override()


class TestAuthorizationStep52:
    """Section 7: Authorization tests."""

    def test_unauthenticated_rejected(self, client):
        c, store, _ = client
        _clear_override()
        resp = c.post("/api/v1/promotions/fake/activation-verify")
        assert resp.status_code == 401

    def test_customer_rejected(self, client):
        c, store, _ = client
        _auth_override(CUSTOMER_USER)
        resp = c.post("/api/v1/promotions/fake/activation-verify")
        assert resp.status_code == 403
        _clear_override()

    def test_analyst_allowed(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        assert resp.status_code == 200
        _clear_override()


class TestProductionSafetyStep52:
    """Section 7: Production safety tests."""

    def test_no_automatic_activation(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        c.post(f"/api/v1/promotions/{pid}/activation-verify")
        record = store.get_by_id(pid)
        # Governance status unchanged by verification
        assert record["governance_status"] == STATUS_APPROVED
        _clear_override()

    def test_no_threshold_mutation(self, client):
        c, store, _ = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        resp = c.post(f"/api/v1/promotions/{pid}/activation-verify")
        assert "threshold" not in resp.json()
        _clear_override()


class TestAuditStep52:
    """Section 7: Audit tests."""

    def test_no_token_in_audit(self, client):
        c, store, audit_store = client
        pid = _create_and_approve(c, store)
        _auth_override(ANALYST_USER)
        c.post(f"/api/v1/promotions/{pid}/activation-verify")

        events = audit_store.list_by_transaction("00000000-0000-0000-0000-000000000000")
        for event in events:
            event_str = str(event)
            assert "secret" not in event_str.lower()
            # Token should not appear in audit
            assert "." not in event_str or "promotion" in event_str.lower()
        _clear_override()
