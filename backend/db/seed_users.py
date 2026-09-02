"""Development seeding tool for internal (non-customer) users.

Public registration always creates ``customer`` accounts, so
``fraud_analyst`` and ``admin`` users must be provisioned separately.
This tool creates them idempotently in the configured user database.

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
``USER_DB_PATH``        User database path (default ``data/users.db``)
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
    SQLiteUserRepository,
)


def _seed(repo: SQLiteUserRepository, *, email: str, password: str, role: str) -> bool:
    """Create a user if the email is not taken.  Returns True if created."""
    if repo.email_exists(email):
        print(f"  {role:<15} {email} — already exists, skipped")
        return False
    repo.create_user(email=email, password=password, role=role)
    print(f"  {role:<15} {email} — created (password: {password})")
    return True


def main() -> int:
    settings = get_settings()
    print(f"Seeding users into {settings.USER_DB_PATH}")

    analyst_email = os.environ.get("SEED_ANALYST_EMAIL", "analyst@example.com")
    admin_email = os.environ.get("SEED_ADMIN_EMAIL", "admin@example.com")

    analyst_password = os.environ.get("SEED_ANALYST_PASS") or secrets.token_urlsafe(12)
    admin_password = os.environ.get("SEED_ADMIN_PASS") or secrets.token_urlsafe(12)

    repo = SQLiteUserRepository(db_path=settings.USER_DB_PATH)
    try:
        _seed(repo, email=analyst_email, password=analyst_password, role=FRAUD_ANALYST)
        _seed(repo, email=admin_email, password=admin_password, role=ADMIN)
    finally:
        repo.close()

    print("Done. Login via POST /api/v1/auth/login.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
