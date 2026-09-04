"""Custom exception classes and global error handlers.

Provides a clean error-handling foundation that:

* Returns consistent JSON error responses.
* Never exposes internal stack traces to API clients.
* Maps known exception types to appropriate HTTP status codes.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception classes
# ---------------------------------------------------------------------------


class AppException(HTTPException):
    """Base application exception with an ``error_code`` field.

    Usage::

        raise AppException(status_code=400, error_code="INVALID_INPUT", detail="...")
    """

    def __init__(
        self,
        status_code: int = 400,
        detail: str = "An error occurred",
        error_code: str = "ERROR",
    ) -> None:
        self.error_code = error_code
        super().__init__(status_code=status_code, detail=detail)


class NotFoundException(AppException):
    """Raised when a requested resource does not exist."""

    def __init__(self, detail: str = "Resource not found") -> None:
        super().__init__(status_code=404, detail=detail, error_code="NOT_FOUND")


class UnauthorizedException(AppException):
    """Raised when the request lacks valid authentication credentials."""

    def __init__(self, detail: str = "Authentication required") -> None:
        super().__init__(status_code=401, detail=detail, error_code="UNAUTHORIZED")


class ForbiddenException(AppException):
    """Raised when the authenticated user lacks permission for the action."""

    def __init__(self, detail: str = "Insufficient permissions") -> None:
        super().__init__(status_code=403, detail=detail, error_code="FORBIDDEN")


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle custom application exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": exc.error_code},
    )


async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Handle standard HTTP exceptions raised by FastAPI or Starlette."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": "HTTP_ERROR"},
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all handler — logs the real error but returns a generic message."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred",
            "error_code": "INTERNAL_ERROR",
        },
    )


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_error_handlers(app: FastAPI) -> None:
    """Attach all global error handlers to *app*."""
    app.add_exception_handler(AppException, app_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)  # type: ignore[arg-type]
