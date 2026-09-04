"""Promotion governance router — Step 50.

Provides authenticated endpoints for managing model promotion
governance records.  Only ``fraud_analyst`` and ``admin`` roles may
interact with promotions.

Endpoints
---------
* ``POST /api/v1/promotions`` — create a governance record from a gate
  decision (PENDING status).
* ``GET /api/v1/promotions`` — list governance records (paginated).
* ``GET /api/v1/promotions/{promotion_id}`` — get a single record.
* ``POST /api/v1/promotions/{promotion_id}/approve`` — approve.
* ``POST /api/v1/promotions/{promotion_id}/reject`` — reject.
* ``POST /api/v1/promotions/{promotion_id}/mark-promoted`` — mark as
  promoted (Step 46 activation confirmed).

Security
--------
* Authentication required for all endpoints.
* Only ``fraud_analyst`` and ``admin`` roles are authorised.
* Customers are denied access.
* Actor identity always comes from the JWT — never from the request
  payload.
* State transitions are validated server-side.
* No automatic model activation occurs.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.db.promotion_governance import (
    AUDIT_PROMOTION_APPROVED,
    AUDIT_PROMOTION_CREATED,
    AUDIT_PROMOTION_MARKED_PROMOTED,
    AUDIT_PROMOTION_REJECTED,
    DuplicatePromotionError,
    InvalidTransitionError,
    PromotionGovernanceRepository,
    PromotionNotFoundError,
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_PROMOTED,
    STATUS_REJECTED,
)
from backend.db.activation_verification import (
    ACTIVATION_CONSUMED,
    ACTIVATION_NONE,
    ACTIVATION_TOKEN_ISSUED,
    AUDIT_ACTIVATION_VERIFICATION_FAILED,
    AUDIT_ACTIVATION_VERIFICATION_PASSED,
    AUDIT_MODEL_ACTIVATED,
    ActivationTokenExpiredError,
    ActivationTokenReplayError,
    ActivationVerificationError,
    ArtifactVerifier,
    DefaultArtifactVerifier,
    VERIFICATION_BLOCKED,
    VERIFICATION_PASSED,
    consume_activation_authorization,
    issue_activation_token,
    verify_activation_preconditions,
)
from backend.schemas import (
    ActivationConsumeRequest,
    ActivationVerifyResponse,
    PromotionApproveRequest,
    PromotionCreateRequest,
    PromotionListResponse,
    PromotionRejectRequest,
    PromotionResponse,
)
from backend.security.deps import require_roles

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["promotions"])

# Promotion governance is restricted to fraud analysts and admins.
_require_analyst = require_roles("fraud_analyst", "admin")

# Module-level repository — set at app startup
_governance_repo: PromotionGovernanceRepository | None = None

# Step 45: audit repository for governance event logging
_audit_repo: Any = None


def set_governance_repository(repo: PromotionGovernanceRepository) -> None:
    """Set the governance repository (called during app startup)."""
    global _governance_repo
    _governance_repo = repo


def set_audit_repository(repo: Any) -> None:
    """Set the audit repository for governance event logging."""
    global _audit_repo
    _audit_repo = repo


def get_governance_repository() -> PromotionGovernanceRepository:
    """Return the active governance repository."""
    if _governance_repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Promotion governance not configured.",
        )
    return _governance_repo


def _record_to_response(record: dict[str, Any]) -> PromotionResponse:
    """Convert a governance record dict to a Pydantic response."""
    return PromotionResponse(
        promotion_id=record["promotion_id"],
        gate_decision=record["gate_decision"],
        governance_status=record["governance_status"],
        candidate_model_name=record["candidate_model_name"],
        candidate_model_version=record["candidate_model_version"],
        candidate_checksum=record["candidate_checksum"],
        candidate_schema_version=record["candidate_schema_version"],
        candidate_n_features=record["candidate_n_features"],
        production_model_name=record["production_model_name"],
        production_model_version=record["production_model_version"],
        production_checksum=record["production_checksum"],
        production_schema_version=record["production_schema_version"],
        production_n_features=record["production_n_features"],
        reviewer_id=record.get("reviewer_id"),
        reviewer_role=record.get("reviewer_role"),
        reviewed_at=record.get("reviewed_at"),
        approval_comment=record.get("approval_comment"),
        rejection_reason=record.get("rejection_reason"),
        execution_status=record.get("execution_status"),
        promoted_by=record.get("promoted_by"),
        promoted_at=record.get("promoted_at"),
        created_at=record["created_at"],
        updated_at=record["updated_at"],
    )


def _audit_event(
    event_type: str,
    *,
    promotion_id: str,
    actor_id: str,
    actor_role: str,
    previous_status: str | None = None,
    new_status: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record a governance event in the audit trail (best-effort)."""
    if _audit_repo is None:
        return
    try:
        _audit_repo.create(
            transaction_id="00000000-0000-0000-0000-000000000000",
            customer_id="00000000-0000-0000-0000-000000000000",
            event_type=event_type,
            actor_id=actor_id,
            actor_role=actor_role,
            previous_state=previous_status,
            new_state=new_status,
            metadata={
                "promotion_id": promotion_id,
                **(metadata or {}),
            },
        )
    except Exception as exc:
        logger.warning("Failed to record promotion audit event: %s", exc)


# ── Endpoints ─────────────────────────────────────────────────────────


@router.post(
    "/promotions",
    response_model=PromotionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_promotion(
    request: PromotionCreateRequest,
    current_user: dict[str, Any] = Depends(_require_analyst),
) -> PromotionResponse:
    """Create a governance record from a gate decision.

    The record starts in PENDING status.  The gate decision and model
    identities are taken from the request (which should come from the
    verified Step 48 gate output).
    """
    repo = get_governance_repository()

    try:
        record = repo.create(
            gate_decision=request.gate_decision,
            candidate_model_name=request.candidate_model_name,
            candidate_model_version=request.candidate_model_version,
            candidate_checksum=request.candidate_checksum,
            candidate_schema_version=request.candidate_schema_version,
            candidate_n_features=request.candidate_n_features,
            production_model_name=request.production_model_name,
            production_model_version=request.production_model_version,
            production_checksum=request.production_checksum,
            production_schema_version=request.production_schema_version,
            production_n_features=request.production_n_features,
            gate_report=request.gate_report,
        )
    except DuplicatePromotionError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A governance record already exists for this gate decision.",
        )

    # Audit: PROMOTION_CREATED
    _audit_event(
        AUDIT_PROMOTION_CREATED,
        promotion_id=record["promotion_id"],
        actor_id=current_user["id"],
        actor_role=current_user["role"],
        new_status=STATUS_PENDING,
        metadata={
            "candidate_model_version": request.candidate_model_version,
            "gate_decision": request.gate_decision,
        },
    )

    logger.info(
        "Promotion governance record created: id=%s decision=%s by=%s",
        record["promotion_id"], request.gate_decision, current_user["id"],
    )
    return _record_to_response(record)


@router.get(
    "/promotions",
    response_model=PromotionListResponse,
)
async def list_promotions(
    status_filter: str | None = Query(
        None, alias="status",
        description="Filter by governance status",
    ),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=200, description="Items per page"),
    current_user: dict[str, Any] = Depends(_require_analyst),
) -> PromotionListResponse:
    """List promotion governance records (paginated)."""
    repo = get_governance_repository()
    offset = (page - 1) * per_page

    records = repo.list_records(
        status=status_filter,
        limit=per_page,
        offset=offset,
    )
    total = repo.count_records(status=status_filter)

    return PromotionListResponse(
        items=[_record_to_response(r) for r in records],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get(
    "/promotions/{promotion_id}",
    response_model=PromotionResponse,
)
async def get_promotion(
    promotion_id: str,
    current_user: dict[str, Any] = Depends(_require_analyst),
) -> PromotionResponse:
    """Get a single promotion governance record."""
    repo = get_governance_repository()
    record = repo.get_by_id(promotion_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Promotion governance record not found.",
        )
    return _record_to_response(record)


@router.post(
    "/promotions/{promotion_id}/approve",
    response_model=PromotionResponse,
)
async def approve_promotion(
    promotion_id: str,
    request: PromotionApproveRequest | None = None,
    current_user: dict[str, Any] = Depends(_require_analyst),
) -> PromotionResponse:
    """Approve a PENDING promotion.

    Actor identity comes from the JWT.  The promotion transitions from
    PENDING to APPROVED.
    """
    repo = get_governance_repository()
    comment = request.comment if request else None

    try:
        record = repo.approve(
            promotion_id,
            reviewer_id=current_user["id"],
            reviewer_role=current_user["role"],
            comment=comment,
        )
    except PromotionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Promotion governance record not found.",
        )
    except InvalidTransitionError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Invalid state transition for this promotion.",
        )

    # Audit: PROMOTION_APPROVED
    _audit_event(
        AUDIT_PROMOTION_APPROVED,
        promotion_id=promotion_id,
        actor_id=current_user["id"],
        actor_role=current_user["role"],
        previous_status=STATUS_PENDING,
        new_status=STATUS_APPROVED,
        metadata={"comment": comment} if comment else None,
    )

    logger.info(
        "Promotion approved: id=%s by=%s",
        promotion_id, current_user["id"],
    )
    return _record_to_response(record)


@router.post(
    "/promotions/{promotion_id}/reject",
    response_model=PromotionResponse,
)
async def reject_promotion(
    promotion_id: str,
    request: PromotionRejectRequest | None = None,
    current_user: dict[str, Any] = Depends(_require_analyst),
) -> PromotionResponse:
    """Reject a PENDING promotion.

    Actor identity comes from the JWT.  The promotion transitions from
    PENDING to REJECTED.
    """
    repo = get_governance_repository()
    reason = request.reason if request else None

    try:
        record = repo.reject(
            promotion_id,
            reviewer_id=current_user["id"],
            reviewer_role=current_user["role"],
            reason=reason,
        )
    except PromotionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Promotion governance record not found.",
        )
    except InvalidTransitionError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Invalid state transition for this promotion.",
        )

    # Audit: PROMOTION_REJECTED
    _audit_event(
        AUDIT_PROMOTION_REJECTED,
        promotion_id=promotion_id,
        actor_id=current_user["id"],
        actor_role=current_user["role"],
        previous_status=STATUS_PENDING,
        new_status=STATUS_REJECTED,
        metadata={"reason": reason} if reason else None,
    )

    logger.info(
        "Promotion rejected: id=%s by=%s",
        promotion_id, current_user["id"],
    )
    return _record_to_response(record)


@router.post(
    "/promotions/{promotion_id}/mark-promoted",
    response_model=PromotionResponse,
)
async def mark_promoted(
    promotion_id: str,
    current_user: dict[str, Any] = Depends(_require_analyst),
) -> PromotionResponse:
    """Mark an APPROVED promotion as PROMOTED.

    This confirms that the operator has performed the Step 46 activation.
    The promotion transitions from APPROVED to PROMOTED.

    This endpoint does NOT activate the model — it only records that
    the activation has been performed through the Step 46 workflow.
    """
    repo = get_governance_repository()

    try:
        record = repo.mark_promoted(
            promotion_id,
            actor_id=current_user["id"],
            actor_role=current_user["role"],
        )
    except PromotionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Promotion governance record not found.",
        )
    except InvalidTransitionError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Invalid state transition for this promotion.",
        )

    # Audit: PROMOTION_MARKED_PROMOTED
    _audit_event(
        AUDIT_PROMOTION_MARKED_PROMOTED,
        promotion_id=promotion_id,
        actor_id=current_user["id"],
        actor_role=current_user["role"],
        previous_status=STATUS_APPROVED,
        new_status=STATUS_PROMOTED,
    )

    logger.info(
        "Promotion marked as promoted: id=%s by=%s",
        promotion_id, current_user["id"],
    )
    return _record_to_response(record)


# ── Steps 51/52: Activation Verification ──────────────────────────────

# Module-level current production identity provider (set at startup or by tests)
_production_identity_provider = None
_artifact_verifier: ArtifactVerifier | None = None


def set_production_identity_provider(provider: Any) -> None:
    """Set a callable that returns the current production model identity dict."""
    global _production_identity_provider
    _production_identity_provider = provider


def set_artifact_verifier(verifier: ArtifactVerifier | None) -> None:
    """Set the artifact verifier (called at startup or by tests)."""
    global _artifact_verifier
    _artifact_verifier = verifier


def _get_current_production_identity() -> dict[str, Any]:
    """Get the current production model identity."""
    if _production_identity_provider is not None:
        return _production_identity_provider()
    return {
        "model_name": "unknown",
        "model_version": "unknown",
        "checksum": "unknown",
        "schema_version": "unknown",
        "n_features": 0,
    }


def _verify_artifact(record: dict[str, Any], production: dict[str, Any]) -> Any:
    """Run artifact verification via the configured verifier.

    Returns an ArtifactVerificationReport.  If no verifier is
    configured, returns a fail-closed report (all False).
    """
    if _artifact_verifier is not None:
        return _artifact_verifier.verify_candidate(
            candidate_model_version=record.get("candidate_model_version", ""),
            candidate_checksum=record.get("candidate_checksum", ""),
            candidate_schema_version=record.get("candidate_schema_version", ""),
            candidate_n_features=record.get("candidate_n_features", 0),
            current_production_identity=production,
        )
    # No verifier configured — fail closed
    from backend.db.activation_verification import ArtifactVerificationReport
    return ArtifactVerificationReport(
        artifact_exists=False,
        checksum_valid=False,
        schema_compatible=False,
        feature_count_matches=False,
    )


@router.post(
    "/promotions/{promotion_id}/activation-verify",
    response_model=ActivationVerifyResponse,
)
async def activation_verify(
    promotion_id: str,
    current_user: dict[str, Any] = Depends(_require_analyst),
) -> ActivationVerifyResponse:
    """Verify activation preconditions and issue a short-lived token.

    This endpoint does NOT activate the model. It verifies that all
    preconditions are met and, if so, returns an activation token.
    """
    repo = get_governance_repository()
    record = repo.get_by_id(promotion_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Promotion governance record not found.",
        )

    current_production = _get_current_production_identity()

    # Step 52: actually verify artifact (fail-closed — never assume exists)
    artifact_report = _verify_artifact(record, current_production)

    result = verify_activation_preconditions(
        governance_record=record,
        current_production_identity=current_production,
        candidate_artifact_exists=artifact_report.artifact_exists,
        candidate_checksum_valid=artifact_report.checksum_valid,
        candidate_schema_compatible=artifact_report.schema_compatible,
        candidate_feature_count_matches=artifact_report.feature_count_matches,
        candidate_is_not_already_active=artifact_report.is_not_already_active,
    )

    if result.status == VERIFICATION_PASSED:
        token, expires_at = issue_activation_token(
            promotion_id=promotion_id,
            candidate_identity=result.candidate_identity,
            production_identity=current_production,
        )

        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        if hasattr(repo, "update_activation_status"):
            repo.update_activation_status(
                promotion_id,
                activation_status=ACTIVATION_TOKEN_ISSUED,
                token_issued_at=now_iso,
                token_expires_at=expires_at,
            )

        _audit_event(
            AUDIT_ACTIVATION_VERIFICATION_PASSED,
            promotion_id=promotion_id,
            actor_id=current_user["id"],
            actor_role=current_user["role"],
            metadata={
                "candidate_model_version": result.candidate_identity.get("model_version"),
            },
        )

        logger.info(
            "Activation verification passed: id=%s by=%s",
            promotion_id, current_user["id"],
        )
        return ActivationVerifyResponse(
            status=VERIFICATION_PASSED,
            promotion_id=promotion_id,
            candidate_identity=result.candidate_identity,
            production_identity=current_production,
            activation_token=token,
            token_expires_at=expires_at,
        )
    else:
        _audit_event(
            AUDIT_ACTIVATION_VERIFICATION_FAILED,
            promotion_id=promotion_id,
            actor_id=current_user["id"],
            actor_role=current_user["role"],
            metadata={"reasons": result.reasons},
        )

        logger.info(
            "Activation verification blocked: id=%s by=%s reasons=%s",
            promotion_id, current_user["id"], result.reasons,
        )
        return ActivationVerifyResponse(
            status=VERIFICATION_BLOCKED,
            promotion_id=promotion_id,
            candidate_identity=result.candidate_identity,
            production_identity=current_production,
            reasons=result.reasons,
        )


@router.post(
    "/promotions/{promotion_id}/activate",
    response_model=PromotionResponse,
)
async def activate_promotion(
    promotion_id: str,
    request: ActivationConsumeRequest,
    current_user: dict[str, Any] = Depends(_require_analyst),
) -> PromotionResponse:
    """Consume the activation token and mark the promotion as activated.

    This endpoint does NOT perform the actual Step 46 activation.
    It consumes the activation token and marks the governance record.
    """
    repo = get_governance_repository()
    record = repo.get_by_id(promotion_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Promotion governance record not found.",
        )

    current_production = _get_current_production_identity()

    try:
        auth = consume_activation_authorization(
            request.activation_token,
            expected_promotion_id=promotion_id,
            current_production_identity=current_production,
            governance_record=record,
        )
    except ActivationTokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Activation token has expired.",
        )
    except ActivationTokenReplayError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Activation token has already been consumed.",
        )
    except ActivationVerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        )

    # Step 52: atomic CAS transition TOKEN_ISSUED → CONSUMED
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    if hasattr(repo, "try_transition_activation_status"):
        cas_result = repo.try_transition_activation_status(
            promotion_id,
            expected_status=ACTIVATION_TOKEN_ISSUED,
            new_status=ACTIVATION_CONSUMED,
            consumed_at=now_iso,
            actor_id=current_user["id"],
        )
        if cas_result is None:
            # CAS failed — check why
            refreshed = repo.get_by_id(promotion_id)
            if refreshed and refreshed.get("activation_status") == ACTIVATION_CONSUMED:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Activation already consumed for this promotion.",
                )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Activation state transition failed — record may have changed.",
            )
    elif hasattr(repo, "update_activation_status"):
        repo.update_activation_status(
            promotion_id,
            activation_status=ACTIVATION_CONSUMED,
            consumed_at=now_iso,
            actor_id=current_user["id"],
        )

    _audit_event(
        AUDIT_MODEL_ACTIVATED,
        promotion_id=promotion_id,
        actor_id=current_user["id"],
        actor_role=current_user["role"],
        previous_status=STATUS_APPROVED,
        new_status=STATUS_PROMOTED,
        metadata={
            "candidate_model_version": auth.candidate_identity.get("model_version"),
        },
    )

    logger.info(
        "Activation token consumed: id=%s by=%s",
        promotion_id, current_user["id"],
    )

    updated = repo.get_by_id(promotion_id)
    return _record_to_response(updated)
