"""FastAPI application entry point.

Provides an :func:`create_app` factory for full control during testing, and a
module-level ``app`` instance consumed by Uvicorn in production.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core.errors import register_error_handlers
from app.api.v1 import api_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="AI-Powered Financial Fraud Detection",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # -- CORS --
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.backend.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Error handlers --
    register_error_handlers(application)

    # -- API routers --
    application.include_router(api_router, prefix="/api/v1")

    return application


app: FastAPI = create_app()
