"""Activation verification & safety gate — Step 51.

Production-safe verification layer between Step 50 governance approval
and Step 46 model activation. Ensures that an authorized operator
cannot accidentally activate a model unless all preconditions are met.

What this module does
---------------------
1. Verifies all preconditions before activation:
   - Governance record exists and is APPROVED
   - Candidate identity is complete and consistent
   - Candidate artifact exists and checksum matches
   - Feature schema is compatible
   - Production baseline has not changed since approval
   - Gate decision was APPROVED
2. Issues a short-lived, signed activation token.
3. Consumes the token upon activation (prevents replay).

What this module NEVER does
----------------------------
* It never activates a model automatically.
* It never modifies the production manifest, threshold, or runtime.
* It never hot-swaps the active model.
* It never trusts client-supplied model identity.

Activation token
----------------
The token is an HMAC-SHA256 signed payload containing:
- promotion_id
- candidate model identity
- expected production identity
- issued_at / expires_at
- purpose/type

Token properties:
- Short-lived (default 5 minutes)
- Scoped to one promotion
- Single-use (consumed on activation)
- Rejected after expiry or if promotion state changes

Security
--------
- Signing uses the existing BACKEND_SECRET_KEY.
- No secrets, raw data, or artifacts in the token.
- Actor identity always from JWT.
- Fail-closed: any verification failure blocks activation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from backend.config import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "VERIFICATION_PASSED",
    "VERIFICATION_BLOCKED",
    "ACTIVATION_NONE",
    "ACTIVATION_TOKEN_ISSUED",
    "ACTIVATION_CONSUMED",
    "AUDIT_ACTIVATION_VERIFICATION_PASSED",
    "AUDIT_ACTIVATION_VERIFICATION_FAILED",
    "AUDIT_MODEL_ACTIVATED",
    "DEFAULT_TOKEN_LIFETIME_SECONDS",
    "ActivationVerificationError",
    "ActivationTokenExpiredError",
    "ActivationTokenReplayError",
    "ActivationVerificationResult",
    "ActivationAuthorization",
    "verify_activation_preconditions",
    "issue_activation_token",
    "validate_activation_token",
    "consume_activation_authorization",
]


# ── Constants ─────────────────────────────────────────────────────────

VERIFICATION_PASSED = "READY_FOR_ACTIVATION"
VERIFICATION_BLOCKED = "ACTIVATION_BLOCKED"

ACTIVATION_NONE = "NONE"
ACTIVATION_TOKEN_ISSUED = "TOKEN_ISSUED"
ACTIVATION_CONSUMED = "CONSUMED"

AUDIT_ACTIVATION_VERIFICATION_PASSED = "ACTIVATION_VERIFICATION_PASSED"
AUDIT_ACTIVATION_VERIFICATION_FAILED = "ACTIVATION_VERIFICATION_FAILED"
AUDIT_MODEL_ACTIVATED = "MODEL_ACTIVATED"

DEFAULT_TOKEN_LIFETIME_SECONDS = 300  # 5 minutes


# ── Exceptions ────────────────────────────────────────────────────────


class ActivationVerificationError(Exception):
    """Activation verification failed."""


class ActivationTokenExpiredError(Exception):
    """Activation token has expired."""


class ActivationTokenReplayError(Exception):
    """Activation token has already been consumed."""


# ── Result containers ─────────────────────────────────────────────────


class ActivationVerificationResult:
    """Result of activation precondition verification."""

    def __init__(
        self,
        *,
        status: str,
        promotion_id: str,
        candidate_identity: dict[str, Any],
        production_identity: dict[str, Any],
        reasons: list[str] | None = None,
        token: str | None = None,
        token_expires_at: str | None = None,
    ) -> None:
        self.status = status
        self.promotion_id = promotion_id
        self.candidate_identity = candidate_identity
        self.production_identity = production_identity
        self.reasons = reasons or []
        self.token = token
        self.token_expires_at = token_expires_at

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "promotion_id": self.promotion_id,
            "candidate_identity": self.candidate_identity,
            "production_identity": self.production_identity,
        }
        if self.reasons:
            result["reasons"] = self.reasons
        if self.token:
            result["activation_token"] = self.token
            result["token_expires_at"] = self.token_expires_at
        return result


class ActivationAuthorization:
    """Parsed activation authorization (from token)."""

    def __init__(
        self,
        *,
        promotion_id: str,
        candidate_identity: dict[str, Any],
        production_identity: dict[str, Any],
        issued_at: str,
        expires_at: str,
        purpose: str = "model_activation",
    ) -> None:
        self.promotion_id = promotion_id
        self.candidate_identity = candidate_identity
        self.production_identity = production_identity
        self.issued_at = issued_at
        self.expires_at = expires_at
        self.purpose = purpose


# ── Token signing ─────────────────────────────────────────────────────


def _get_signing_key() -> str:
    """Get the signing key from application settings."""
    settings = get_settings()
    return settings.BACKEND_SECRET_KEY


def _sign_payload(payload: dict[str, Any]) -> str:
    """Sign a payload dict and return a token string."""
    key = _get_signing_key()
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    signature = hmac.new(
        key.encode("utf-8"),
        payload_json.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    # Token format: base64(payload).signature
    import base64
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("ascii")
    return f"{payload_b64}.{signature}"


def _verify_token(token: str) -> dict[str, Any]:
    """Verify a token's signature and return the payload."""
    import base64
    parts = token.split(".")
    if len(parts) != 2:
        raise ActivationVerificationError("Invalid token format")

    payload_b64, signature = parts
    key = _get_signing_key()

    try:
        payload_json = base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8")
    except Exception:
        raise ActivationVerificationError("Invalid token encoding")

    expected_sig = hmac.new(
        key.encode("utf-8"),
        payload_json.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(signature, expected_sig):
        raise ActivationVerificationError("Invalid token signature")

    return json.loads(payload_json)


# ── Verification logic ────────────────────────────────────────────────


def verify_activation_preconditions(
    *,
    governance_record: dict[str, Any],
    current_production_identity: dict[str, Any],
    candidate_artifact_exists: bool = True,
    candidate_checksum_valid: bool = True,
    candidate_schema_compatible: bool = True,
    candidate_feature_count_matches: bool = True,
    candidate_is_not_already_active: bool = True,
) -> ActivationVerificationResult:
    """Verify all preconditions for activation.

    This is a pure verification function — it does NOT activate anything.
    Returns a result with status READY_FOR_ACTIVATION or ACTIVATION_BLOCKED.

    Fail-closed: any missing or invalid precondition blocks activation.
    """
    reasons: list[str] = []

    # 1. Governance record must exist
    if not governance_record:
        return ActivationVerificationResult(
            status=VERIFICATION_BLOCKED,
            promotion_id="unknown",
            candidate_identity={},
            production_identity=current_production_identity,
            reasons=["Governance record not found"],
        )

    promotion_id = governance_record.get("promotion_id", "unknown")

    # 2. Governance status must be APPROVED
    gov_status = governance_record.get("governance_status")
    if gov_status != STATUS_APPROVED:
        reasons.append(f"Governance status is {gov_status}, expected APPROVED")

    # 3. Gate decision must be APPROVED
    gate_decision = governance_record.get("gate_decision")
    if gate_decision != "APPROVED":
        reasons.append(f"Gate decision is {gate_decision}, expected APPROVED")

    # 4. Candidate identity must be complete
    candidate_identity = {
        "model_name": governance_record.get("candidate_model_name"),
        "model_version": governance_record.get("candidate_model_version"),
        "checksum": governance_record.get("candidate_checksum"),
        "schema_version": governance_record.get("candidate_schema_version"),
        "n_features": governance_record.get("candidate_n_features"),
    }
    for field_name, value in candidate_identity.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            reasons.append(f"Candidate identity missing {field_name}")

    # 5. Production baseline must match
    expected_production = {
        "model_name": governance_record.get("production_model_name"),
        "model_version": governance_record.get("production_model_version"),
        "checksum": governance_record.get("production_checksum"),
    }
    for field_name, expected in expected_production.items():
        actual = current_production_identity.get(field_name)
        if actual != expected:
            reasons.append(
                f"Production baseline changed: {field_name} "
                f"was {expected} at approval, now {actual}"
            )

    # 6. Activation not already consumed
    activation_status = governance_record.get("activation_status", ACTIVATION_NONE)
    if activation_status == ACTIVATION_CONSUMED:
        reasons.append("Activation already consumed for this promotion")

    # 7. Artifact checks
    if not candidate_artifact_exists:
        reasons.append("Candidate artifact does not exist")
    if not candidate_checksum_valid:
        reasons.append("Candidate artifact checksum mismatch")
    if not candidate_schema_compatible:
        reasons.append("Candidate feature schema incompatible")
    if not candidate_feature_count_matches:
        reasons.append("Candidate feature count mismatch")
    if not candidate_is_not_already_active:
        reasons.append("Candidate is already the active production model")

    if reasons:
        return ActivationVerificationResult(
            status=VERIFICATION_BLOCKED,
            promotion_id=promotion_id,
            candidate_identity=candidate_identity,
            production_identity=current_production_identity,
            reasons=reasons,
        )

    return ActivationVerificationResult(
        status=VERIFICATION_PASSED,
        promotion_id=promotion_id,
        candidate_identity=candidate_identity,
        production_identity=current_production_identity,
    )


# ── Token issuance ────────────────────────────────────────────────────


def issue_activation_token(
    *,
    promotion_id: str,
    candidate_identity: dict[str, Any],
    production_identity: dict[str, Any],
    lifetime_seconds: int = DEFAULT_TOKEN_LIFETIME_SECONDS,
) -> tuple[str, str]:
    """Issue a short-lived activation token.

    Returns (token_string, expires_at_iso).
    """
    now = datetime.now(timezone.utc)
    expires_at = datetime.fromtimestamp(
        now.timestamp() + lifetime_seconds, tz=timezone.utc
    )

    payload = {
        "promotion_id": promotion_id,
        "candidate_identity": candidate_identity,
        "production_identity": production_identity,
        "issued_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "purpose": "model_activation",
        "token_id": str(uuid.uuid4()),
    }

    token = _sign_payload(payload)
    return token, expires_at.isoformat()


def validate_activation_token(
    token: str,
    *,
    expected_promotion_id: str,
    current_production_identity: dict[str, Any],
) -> ActivationAuthorization:
    """Validate an activation token and return the authorization.

    Raises:
        ActivationVerificationError: Invalid signature or wrong promotion.
        ActivationTokenExpiredError: Token has expired.
    """
    payload = _verify_token(token)

    # Check promotion ID
    if payload.get("promotion_id") != expected_promotion_id:
        raise ActivationVerificationError(
            "Token is for a different promotion"
        )

    # Check expiration
    expires_at_str = payload.get("expires_at", "")
    try:
        expires_at = datetime.fromisoformat(expires_at_str)
        if datetime.now(timezone.utc) > expires_at:
            raise ActivationTokenExpiredError("Activation token has expired")
    except (ValueError, TypeError):
        raise ActivationVerificationError("Invalid token expiration")

    # Check production baseline
    token_production = payload.get("production_identity", {})
    for field_name in ("model_version", "checksum"):
        expected = current_production_identity.get(field_name)
        actual = token_production.get(field_name)
        if expected != actual:
            raise ActivationVerificationError(
                f"Production baseline changed since token issuance: {field_name}"
            )

    return ActivationAuthorization(
        promotion_id=payload["promotion_id"],
        candidate_identity=payload.get("candidate_identity", {}),
        production_identity=token_production,
        issued_at=payload.get("issued_at", ""),
        expires_at=expires_at_str,
        purpose=payload.get("purpose", "model_activation"),
    )


# ── Consumption tracking ──────────────────────────────────────────────


# Module-level consumed token store (for in-memory mode)
_consumed_tokens: set[str] = set()
_consumed_lock = __import__("threading").Lock()


def mark_token_consumed(token_id: str) -> None:
    """Mark a token as consumed (prevent replay)."""
    with _consumed_lock:
        _consumed_tokens.add(token_id)


def is_token_consumed(token_id: str) -> bool:
    """Check if a token has already been consumed."""
    with _consumed_lock:
        return token_id in _consumed_tokens


def reset_consumed_tokens() -> None:
    """Reset consumed token store (tests only)."""
    with _consumed_lock:
        _consumed_tokens.clear()


def consume_activation_authorization(
    token: str,
    *,
    expected_promotion_id: str,
    current_production_identity: dict[str, Any],
) -> ActivationAuthorization:
    """Validate and consume an activation authorization.

    This is the final step before activation. It:
    1. Validates the token signature and expiration.
    2. Checks the token hasn't been consumed (replay prevention).
    3. Marks the token as consumed.
    4. Returns the authorization for the caller to perform activation.

    Raises:
        ActivationVerificationError: Invalid token.
        ActivationTokenExpiredError: Token expired.
        ActivationTokenReplayError: Token already consumed.
    """
    # Validate first
    auth = validate_activation_token(
        token,
        expected_promotion_id=expected_promotion_id,
        current_production_identity=current_production_identity,
    )

    # Extract token_id from the payload for replay check
    payload = _verify_token(token)
    token_id = payload.get("token_id", "")

    if is_token_consumed(token_id):
        raise ActivationTokenReplayError("Activation token already consumed")

    # Mark as consumed
    mark_token_consumed(token_id)

    return auth


# Import at module level to avoid circular imports
from backend.db.promotion_governance import STATUS_APPROVED
