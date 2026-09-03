"""Backend application settings.

Reads configuration from environment variables (or ``.env`` file).
The ML/Fraud Intelligence Service URL and timeout are the key
integration settings; the ``POSTGRES_*`` variables configure the
production persistence layer (PostgreSQL / Supabase).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── ML / Fraud Intelligence Service ───────────────────────────────
    ML_SERVICE_URL: str = "http://localhost:8001"
    ML_REQUEST_TIMEOUT_SECONDS: int = 5

    # ── Backend ────────────────────────────────────────────────────────
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    BACKEND_CORS_ORIGINS: str = "http://localhost:5173"

    # ── Persistence backend ───────────────────────────────────────────
    # "postgres" — production default (PostgreSQL / Supabase); the
    #               backend fails to start when the database is
    #               unreachable (fail-fast, no silent data loss).
    # "sqlite"   — lightweight local development without a DB server
    #               (uses ALERT_DB_PATH / USER_DB_PATH below).
    PERSISTENCE_BACKEND: str = "postgres"

    # ── PostgreSQL / Supabase connection ─────────────────────────────
    # Credentials are never logged or returned in API responses.
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "fraud_detection"
    POSTGRES_USER: str = ""
    POSTGRES_PASSWORD: str = ""
    # TLS mode (libpq ``sslmode``).  Empty string = library default
    # ("prefer"), which works with the local docker-compose PostgreSQL.
    # Use "require" for Supabase / other managed PostgreSQL services.
    POSTGRES_SSL_MODE: str = ""

    # ── Alert persistence (SQLite backend only) ───────────────────────
    ALERT_DB_PATH: str = "data/alerts.db"

    # ── Authentication (JWT) ──────────────────────────────────────────
    # SECURITY: the default secret is for LOCAL DEVELOPMENT ONLY.
    # Production deployments MUST override BACKEND_SECRET_KEY with a
    # strong random value, e.g.:
    #   python -c "import secrets; print(secrets.token_urlsafe(48))"
    BACKEND_SECRET_KEY: str = "dev-insecure-secret-change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    # Contract: login response expires_in = 1800 seconds
    BACKEND_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── User persistence (SQLite backend only) ────────────────────────
    USER_DB_PATH: str = "data/users.db"


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
