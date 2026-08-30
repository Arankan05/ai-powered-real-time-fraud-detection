"""ML/Fraud Intelligence Service HTTP client.

Follows the contract defined in ``docs/ml-architecture.md`` exactly.

* Sends ``POST /predict`` to the ML service for every transaction.
* Handles timeout, connection refused, invalid responses.
* Never calculates fraud scores locally.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class MLServiceError(Exception):
    """Base exception for ML service failures."""


class MLServiceUnavailableError(MLServiceError):
    """ML service is unreachable (timeout, connection refused, 503)."""


class MLInvalidResponseError(MLServiceError):
    """ML service returned an unexpected or malformed response."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MLServiceClient:
    """HTTP client for the ML/Fraud Intelligence Service."""

    def __init__(self) -> None:
        self._base_url = (
            f"http://{settings.ml_service.host}:{settings.ml_service.port}"
        )
        self._timeout = settings.ml_service.request_timeout_seconds

    def predict(
        self,
        *,
        customer_id: UUID,
        customer_history: dict[str, Any],
        transaction_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Call ``POST /predict`` and return the parsed response dict.

        Raises
        ------
        MLServiceUnavailableError
            If the ML service is unreachable, times out, or returns 503.
        MLInvalidResponseError
            If the ML service returns an unexpected status code or
            malformed response body.
        """
        payload = {
            "customer_id": str(customer_id),
            "customer_history": customer_history,
            "transaction": transaction_data,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/predict",
                    json=payload,
                )
        except httpx.TimeoutException:
            logger.warning("ML service request timed out")
            raise MLServiceUnavailableError("ML service request timed out")
        except httpx.ConnectError:
            logger.warning("ML service connection refused")
            raise MLServiceUnavailableError("ML service connection refused")
        except httpx.HTTPError as exc:
            logger.warning("ML service HTTP error: %s", exc)
            raise MLServiceUnavailableError(f"ML service error: {exc}")

        # ML service returns 503 when model is unavailable
        if response.status_code == 503:
            raise MLServiceUnavailableError("ML service model not available")

        if response.status_code != 200:
            logger.error(
                "ML service returned unexpected status %d", response.status_code
            )
            raise MLInvalidResponseError(
                f"ML service returned status {response.status_code}"
            )

        try:
            data = response.json()
        except Exception:
            raise MLInvalidResponseError("ML service returned non-JSON response")

        self._validate_response(data)
        return data

    @staticmethod
    def _validate_response(data: dict[str, Any]) -> None:
        """Validate required fields in the ML service response."""
        required = {
            "ml_score", "behaviour_score", "rule_score",
            "risk_score", "risk_level", "decision",
            "explanation", "risk_factors", "model_version",
        }
        missing = required - set(data.keys())
        if missing:
            raise MLInvalidResponseError(
                f"ML response missing fields: {missing}"
            )

        if data["risk_level"] not in {"LOW", "MEDIUM", "HIGH"}:
            raise MLInvalidResponseError(
                f"Invalid risk_level: {data['risk_level']}"
            )

        if data["decision"] not in {"APPROVE", "VERIFY", "HOLD"}:
            raise MLInvalidResponseError(
                f"Invalid decision: {data['decision']}"
            )


def build_customer_history(
    *,
    transaction_count_30d: int = 0,
    avg_amount_30d: float = 0.0,
    std_amount_30d: float = 0.0,
    last_transaction_country: str | None = None,
    last_transaction_timestamp: str | None = None,
    known_device_fingerprints: list[str] | None = None,
    known_merchant_ids: list[str] | None = None,
    previous_flagged_count: int = 0,
) -> dict[str, Any]:
    """Build the ``customer_history`` object for the ML request payload."""
    return {
        "transaction_count_30d": transaction_count_30d,
        "avg_amount_30d": avg_amount_30d,
        "std_amount_30d": std_amount_30d,
        "last_transaction_country": last_transaction_country,
        "last_transaction_timestamp": last_transaction_timestamp,
        "known_device_fingerprints": known_device_fingerprints or [],
        "known_merchant_ids": known_merchant_ids or [],
        "previous_flagged_count": previous_flagged_count,
    }


def build_transaction_payload(
    *,
    amount: float,
    currency: str,
    merchant_id: str | None,
    merchant_name: str,
    merchant_category: str | None,
    transaction_type: str,
    location_country: str | None,
    location_city: str | None,
    device_fingerprint: str | None,
    device_type: str | None,
    ip_address: str | None,
    timestamp: str,
) -> dict[str, Any]:
    """Build the ``transaction`` object for the ML request payload."""
    return {
        "amount": amount,
        "currency": currency,
        "merchant_id": merchant_id,
        "merchant_name": merchant_name,
        "merchant_category": merchant_category,
        "transaction_type": transaction_type,
        "location_country": location_country,
        "location_city": location_city,
        "device_fingerprint": device_fingerprint,
        "device_type": device_type,
        "ip_address": ip_address,
        "timestamp": timestamp,
    }
