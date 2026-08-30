"""Alembic migration environment.

Configures the migration engine to use:

* The project's SQLAlchemy ``Base.metadata`` (from ``app.db.base``) so that
  every model inheriting from ``Base`` is automatically tracked.
* PostgreSQL connection settings from Pydantic ``PostgresSettings`` — only
  ``POSTGRES_*`` environment variables are required (no ``BACKEND_SECRET_KEY``).

Migrations live in ``database/alembic/versions/``.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ---------------------------------------------------------------------------
# Import the declarative Base so Alembic can inspect model metadata.
# The sys.path adjustment allows ``app`` to be importable from the
# ``database/`` directory.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "backend"))

from app.db.base import Base  # noqa: E402

# Alembic Config object — provides access to alembic.ini values.
config = context.config

# Python logging configuration from alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The MetaData object used for autogenerate support.
target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Override sqlalchemy.url from project settings so that credentials come
# exclusively from environment variables.
# ---------------------------------------------------------------------------
from app.config import PostgresSettings  # noqa: E402

config.set_main_option("sqlalchemy.url", PostgresSettings().database_url)


# ---------------------------------------------------------------------------
# Migration runners
# ---------------------------------------------------------------------------


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (SQL script generation only)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
