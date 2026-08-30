"""Application configuration loaded from environment variables.

All sensitive values (secrets, credentials) must be provided via environment
variables or a .env file.  See .env.example at the repository root for the
complete list of expected variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from sqlalchemy.engine import URL as SAUrl

# Project root contains the .env file.  config.py lives at
# <project-root>/backend/app/config.py, so two parents up is the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class BackendSettings(BaseSettings):
    """Core backend settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    secret_key: str = Field(..., description="Secret key for JWT and cryptographic operations")
    access_token_expire_minutes: int = 30
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(env_prefix="BACKEND_", env_file=str(_ENV_FILE), extra="ignore")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v: object) -> list[str]:
        """Accept a comma-separated string from env / dotenv and return a list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v  # type: ignore[return-value]


class PostgresSettings(BaseSettings):
    """PostgreSQL connection settings."""

    host: str = "localhost"
    port: int = 5432
    db: str = "fraud_detection"
    user: str = ""
    password: str = ""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=str(_ENV_FILE), extra="ignore")

    @property
    def database_url(self) -> str:
        """Build a SQLAlchemy-compatible connection URL.

        Uses ``sqlalchemy.engine.URL.create`` so that credentials containing
        URL-special characters (e.g. ``@``, ``:``, ``/``) are encoded safely.
        """
        return SAUrl.create(
            drivername="postgresql",
            username=self.user or None,
            password=self.password or None,
            host=self.host,
            port=self.port,
            database=self.db,
        ).render_as_string(hide_password=False)


class MLServiceSettings(BaseSettings):
    """ML / Fraud Intelligence Service connection settings."""

    host: str = "localhost"
    port: int = 8001
    request_timeout_seconds: int = 5

    model_config = SettingsConfigDict(env_prefix="ML_SERVICE_", env_file=str(_ENV_FILE), extra="ignore")


class RiskThresholdSettings(BaseSettings):
    """Risk score thresholds for decision mapping."""

    low: int = 30
    medium: int = 70

    model_config = SettingsConfigDict(env_prefix="RISK_THRESHOLD_", env_file=str(_ENV_FILE), extra="ignore")


class Settings:
    """Root configuration aggregating all settings sections.

    Usage::

        from app.config import settings

        settings.backend.secret_key
        settings.postgres.database_url
        settings.ml_service.host
    """

    def __init__(self) -> None:
        self.backend = BackendSettings()
        self.postgres = PostgresSettings()
        self.ml_service = MLServiceSettings()
        self.risk_thresholds = RiskThresholdSettings()


settings = Settings()
