"""HTTP client for the ML / Fraud Intelligence Service.

Provides :class:`MLServiceClient` — an async wrapper around
``httpx.AsyncClient`` that calls the ML service's ``POST /predict``
and ``GET /health`` endpoints.

Error handling strategy
-----------------------
Every failure mode (connection refused, timeout, 4xx, 5xx, malformed
response) is caught and re-raised as one of:

  * :class:`MLServiceUnavailableError` — service cannot be reached.
  * :class:`MLServiceTimeoutError` — request exceeded the timeout.
  * :class:`MLServiceResponseError` — service returned a non-2xx
    status or an unparseable body.

The calling code (transaction router) maps these to appropriate HTTP
responses (typically **503**) without exposing internal details.

Usage::

    client = MLServiceClient(settings)
    result = await client.predict(payload)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from backend.schemas import MLPredictionResponse

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────


class MLServiceError(Exception):
    """Base exception for ML service communication failures."""


class MLServiceUnavailableError(MLServiceError):
    """ML service is unreachable (connection refused, DNS failure)."""


class MLServiceTimeoutError(MLServiceError):
    """ML service did not respond within the configured timeout."""


class MLServiceResponseError(MLServiceError):
    """ML service returned a non-2xx status or invalid response body."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"ML service returned {status_code}: {detail}")


# ── Client ────────────────────────────────────────────────────────────


class MLServiceClient:
    """Async HTTP client for the ML / Fraud Intelligence Service.

    Args:
        base_url: Root URL of the ML service (e.g. ``http://localhost:8001``).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 5.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def predict(self, payload: dict[str, Any]) -> MLPredictionResponse:
        """Send a prediction request to the ML service.

        Args:
            payload: JSON-serialisable dict matching the ML service's
                     ``POST /predict`` request schema.

        Returns:
            Parsed :class:`MLPredictionResponse`.

        Raises:
            MLServiceUnavailableError: Connection refused / DNS failure.
            MLServiceTimeoutError: Request timed out.
            MLServiceResponseError: Non-2xx status or invalid body.
        """
        url = f"{self._base_url}/predict"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.ConnectError as exc:
            logger.error("ML service connection refused: %s", exc)
            raise MLServiceUnavailableError(
                f"Cannot connect to ML service at {self._base_url}"
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("ML service timeout after %ss: %s", self._timeout, exc)
            raise MLServiceTimeoutError(
                f"ML service timed out after {self._timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("ML service HTTP error: %s", exc)
            raise MLServiceUnavailableError(
                f"ML service HTTP error: {exc}"
            ) from exc

        # Handle non-2xx responses
        if response.status_code == 503:
            detail = _safe_detail(response)
            raise MLServiceResponseError(503, detail or "Model not available")
        if response.status_code == 422:
            detail = _safe_detail(response)
            raise MLServiceResponseError(422, detail or "Invalid request to ML service")
        if response.status_code >= 400:
            detail = _safe_detail(response)
            raise MLServiceResponseError(
                response.status_code,
                detail or f"Unexpected status {response.status_code}",
            )

        # Parse the successful response
        try:
            return MLPredictionResponse.model_validate(response.json())
        except (ValueError, KeyError) as exc:
            logger.error("ML service returned invalid response: %s", exc)
            raise MLServiceResponseError(
                response.status_code,
                f"Invalid ML response body: {exc}",
            ) from exc

    async def update_outcome(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Update a transaction outcome via the ML service.

        Args:
            payload: Dict with ``customer_id``, ``timestamp``, ``is_fraud``.

        Returns:
            Parsed response dict from the ML service.

        Raises:
            MLServiceUnavailableError: Connection refused / DNS failure.
            MLServiceTimeoutError: Request timed out.
            MLServiceResponseError: Non-2xx status or invalid body.
        """
        url = f"{self._base_url}/outcome"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.ConnectError as exc:
            logger.error("ML service connection refused: %s", exc)
            raise MLServiceUnavailableError(
                f"Cannot connect to ML service at {self._base_url}"
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("ML service timeout after %ss: %s", self._timeout, exc)
            raise MLServiceTimeoutError(
                f"ML service timed out after {self._timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("ML service HTTP error: %s", exc)
            raise MLServiceUnavailableError(
                f"ML service HTTP error: {exc}"
            ) from exc

        if response.status_code == 404:
            detail = _safe_detail(response)
            raise MLServiceResponseError(404, detail or "Transaction not found")
        if response.status_code >= 400:
            detail = _safe_detail(response)
            raise MLServiceResponseError(
                response.status_code,
                detail or f"Unexpected status {response.status_code}",
            )

        try:
            return response.json()
        except (ValueError, KeyError) as exc:
            logger.error("ML service returned invalid response: %s", exc)
            raise MLServiceResponseError(
                response.status_code,
                f"Invalid outcome response body: {exc}",
            ) from exc

    async def health(self) -> dict[str, Any]:
        """Check ML service health.

        Returns:
            Dict with ``status``, ``model_version``, ``features``.

        Raises:
            MLServiceUnavailableError: Service unreachable.
        """
        url = f"{self._base_url}/health"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as exc:
            raise MLServiceUnavailableError(
                f"ML health check failed: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise MLServiceResponseError(
                response.status_code,
                f"Health endpoint returned {response.status_code}",
            )

        try:
            return response.json()
        except ValueError as exc:
            raise MLServiceResponseError(
                response.status_code,
                f"Invalid health response: {exc}",
            ) from exc


# ── Helpers ───────────────────────────────────────────────────────────


def _safe_detail(response: httpx.Response) -> str:
    """Extract a safe error detail string from an ML service response."""
    try:
        body = response.json()
        if isinstance(body, dict):
            return str(body.get("detail", ""))
    except (ValueError, KeyError):
        pass
    return response.text[:200] if response.text else ""
