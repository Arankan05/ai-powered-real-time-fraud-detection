"""Backend application settings.

Reads configuration from environment variables (or ``.env`` file).
The ML/Fraud Intelligence Service URL and timeout are the key
integration settings for this step.
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

    # ── Alert persistence ─────────────────────────────────────────────
    ALERT_DB_PATH: str = "data/alerts.db"


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
