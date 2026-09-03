"""Development seeding tool for internal (non-customer) users.

Public registration always creates ``customer`` accounts, so
``fraud_analyst`` and ``admin`` users must be provisioned separately.
This tool creates them idempotently in the configured persistence
backend (PostgreSQL by default, SQLite opt-in via ``PERSISTENCE_BACKEND``).

Usage::

    python -m backend.db.seed_users

Configuration (environment variables):

======================  ==========================================
``SEED_ANALYST_EMAIL``  Email for the fraud analyst account
                        (default ``analyst@example.com``)
``SEED_ANALYST_PASS``   Password for the analyst account
                        (default: generated and printed once)
``SEED_ADMIN_EMAIL``    Email for the admin account
                        (default ``admin@example.com``)
``SEED_ADMIN_PASS``     Password for the admin account
                        (default: generated and printed once)
``PERSISTENCE_BACKEND`` ``postgres`` (default) or ``sqlite``
``POSTGRES_*``          PostgreSQL / Supabase connection vars
                        (used when ``PERSISTENCE_BACKEND=postgres``)
``USER_DB_PATH``        SQLite database path (used when
                        ``PERSISTENCE_BACKEND=sqlite``)
======================  ==========================================

Generated passwords are printed to stdout once and never stored in
plaintext.  This tool is intended for local development and demos —
production provisioning should use a managed process.
"""

from __future__ import annotations

import os
import secrets
import sys

from backend.config import get_settings
from backend.db.user_repository import (
    ADMIN,
    FRAUD_ANALYST,
    PostgresUserRepository,
    SQLiteUserRepository,
)


def _seed(repo, *, email: str, password: str, role: str) -> bool:
    """Create a user if the email is not taken.  Returns True if created."""
    if repo.email_exists(email):
        print(f"  {role:<15} {email} — already exists, skipped")
        return False
    repo.create_user(email=email, password=password, role=role)
    print(f"  {role:<15} {email} — created (password: {password})")
    return True


def _build_user_repo(settings):
    """Construct the appropriate user repository for the configured backend.

    Returns a ``(repo, pool_or_none)`` pair — the caller is responsible
    for closing both on exit.
    """
    backend_name = (settings.PERSISTENCE_BACKEND or "postgres").strip().lower()
    if backend_name == "postgres":
        from backend.db.postgres import create_pool
        pool = create_pool(settings)
        return PostgresUserRepository(pool), pool
    if backend_name == "sqlite":
        return SQLiteUserRepository(db_path=settings.USER_DB_PATH), None
    raise SystemExit(f"Unknown PERSISTENCE_BACKEND: {backend_name!r}")


def main() -> int:
    settings = get_settings()
    backend_name = (settings.PERSISTENCE_BACKEND or "postgres").strip().lower()

    print(f"Seeding users (backend={backend_name})")
    analyst_email = os.environ.get("SEED_ANALYST_EMAIL", "analyst@example.com")
    admin_email = os.environ.get("SEED_ADMIN_EMAIL", "admin@example.com")

    analyst_password = os.environ.get("SEED_ANALYST_PASS") or secrets.token_urlsafe(12)
    admin_password = os.environ.get("SEED_ADMIN_PASS") or secrets.token_urlsafe(12)

    repo, pool = _build_user_repo(settings)
    try:
        _seed(repo, email=analyst_email, password=analyst_password, role=FRAUD_ANALYST)
        _seed(repo, email=admin_email, password=admin_password, role=ADMIN)
    finally:
        # close() on either repo closes the shared pool when applicable;
        # psycopg_pool.ConnectionPool.close() is idempotent.
        if pool is not None:
            try:
                pool.close()
            except Exception:
                pass
        elif hasattr(repo, "close"):
            repo.close()

    print("Done. Login via POST /api/v1/auth/login.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
