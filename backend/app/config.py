"""Application configuration loaded from environment variables.

All sensitive values (secrets, credentials) must be provided via environment
variables or a .env file.  See .env.example at the repository root for the
complete list of expected variables.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """Core backend settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    secret_key: str = Field(..., description="Secret key for JWT and cryptographic operations")
    access_token_expire_minutes: int = 30
    cors_origins: list[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(env_prefix="BACKEND_")


class PostgresSettings(BaseSettings):
    """PostgreSQL connection settings."""

    host: str = "localhost"
    port: int = 5432
    db: str = "fraud_detection"
    user: str = ""
    password: str = ""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

    @property
    def database_url(self) -> str:
        """Build a SQLAlchemy-compatible connection URL."""
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )


class MLServiceSettings(BaseSettings):
    """ML / Fraud Intelligence Service connection settings."""

    host: str = "localhost"
    port: int = 8001
    request_timeout_seconds: int = 5

    model_config = SettingsConfigDict(env_prefix="ML_SERVICE_")


class RiskThresholdSettings(BaseSettings):
    """Risk score thresholds for decision mapping."""

    low: int = 30
    medium: int = 70

    model_config = SettingsConfigDict(env_prefix="RISK_THRESHOLD_")


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
