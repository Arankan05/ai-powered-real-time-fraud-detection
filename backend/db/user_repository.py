"""User repository — SQLite and in-memory implementations.

Provides persistent storage for user accounts backing JWT
authentication.  Passwords are hashed with bcrypt
(:mod:`backend.security.passwords`) before storage — plaintext
passwords are never persisted.

Roles
-----
Three roles are defined by the API contract:

* ``customer`` — default role assigned by public registration
* ``fraud_analyst`` — can investigate and update alerts
* ``admin`` — full access to analyst functionality

Public registration always creates ``customer`` accounts; internal
roles are provisioned through :mod:`backend.db.seed_users` or direct
repository use.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from backend.security.passwords import hash_password

# ── Role constants ────────────────────────────────────────────────────

CUSTOMER = "customer"
FRAUD_ANALYST = "fraud_analyst"
ADMIN = "admin"

VALID_ROLES = frozenset({CUSTOMER, FRAUD_ANALYST, ADMIN})


# ── Protocol ──────────────────────────────────────────────────────────


@runtime_checkable
class UserRepository(Protocol):
    """Abstract contract for user persistence."""

    def create_user(
        self,
        *,
        email: str,
        password: str,
        role: str = CUSTOMER,
        first_name: str | None = None,
        last_name: str | None = None,
        phone: str | None = None,
        date_of_birth: str | None = None,
        address: str | None = None,
        is_active: bool = True,
    ) -> dict[str, Any]:
        """Create a user (password is hashed before storage)."""
        ...

    def get_by_id(self, user_id: str) -> dict[str, Any] | None:
        """Return a user by ID, or None."""
        ...

    def get_by_email(self, email: str) -> dict[str, Any] | None:
        """Return a user by (case-insensitive) email, or None."""
        ...

    def email_exists(self, email: str) -> bool:
        """Return True if an account already exists for the email."""
        ...


def _new_user_row(
    *,
    email: str,
    password: str,
    role: str,
    first_name: str | None,
    last_name: str | None,
    phone: str | None,
    date_of_birth: str | None,
    address: str | None,
    is_active: bool,
) -> dict[str, Any]:
    """Build the common user record shared by both implementations."""
    return {
        "id": str(uuid.uuid4()),
        "email": email.strip().lower(),
        "password_hash": hash_password(password),
        "role": role if role in VALID_ROLES else CUSTOMER,
        "first_name": first_name,
        "last_name": last_name,
        "phone": phone,
        "date_of_birth": date_of_birth,
        "address": address,
        "customer_id": str(uuid.uuid4()),
        "is_active": bool(is_active),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ── In-memory implementation ─────────────────────────────────────────


class InMemoryUserStore:
    """Volatile in-memory user store (for tests and fallback)."""

    def __init__(self) -> None:
        self._users: dict[str, dict[str, Any]] = {}
        self._by_email: dict[str, str] = {}
        self._lock = threading.Lock()

    def create_user(
        self,
        *,
        email: str,
        password: str,
        role: str = CUSTOMER,
        first_name: str | None = None,
        last_name: str | None = None,
        phone: str | None = None,
        date_of_birth: str | None = None,
        address: str | None = None,
        is_active: bool = True,
    ) -> dict[str, Any]:
        user = _new_user_row(
            email=email,
            password=password,
            role=role,
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            date_of_birth=date_of_birth,
            address=address,
            is_active=is_active,
        )
        with self._lock:
            self._users[user["id"]] = user
            self._by_email[user["email"]] = user["id"]
        return dict(user)

    def get_by_id(self, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            user = self._users.get(user_id)
        return dict(user) if user else None

    def get_by_email(self, email: str) -> dict[str, Any] | None:
        with self._lock:
            user_id = self._by_email.get(email.strip().lower())
            user = self._users.get(user_id) if user_id else None
        return dict(user) if user else None

    def email_exists(self, email: str) -> bool:
        with self._lock:
            return email.strip().lower() in self._by_email


# ── SQLite implementation ─────────────────────────────────────────────


_CREATE_USERS_TABLE = """\
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'customer',
    first_name    TEXT,
    last_name     TEXT,
    phone         TEXT,
    date_of_birth TEXT,
    address       TEXT,
    customer_id   TEXT,
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL
);
"""

_CREATE_USERS_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_users_role ON users(role);"
)


class SQLiteUserRepository:
    """Persistent SQLite-backed user repository.

    Uses WAL journal mode for concurrent read/write safety, following
    the same pattern as the alert repository and
    ``ml/features/history.py``.
    """

    def __init__(self, db_path: str | Path = "data/users.db") -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connect()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_CREATE_USERS_TABLE)
        self._conn.execute(_CREATE_USERS_INDEX)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def create_user(
        self,
        *,
        email: str,
        password: str,
        role: str = CUSTOMER,
        first_name: str | None = None,
        last_name: str | None = None,
        phone: str | None = None,
        date_of_birth: str | None = None,
        address: str | None = None,
        is_active: bool = True,
    ) -> dict[str, Any]:
        user = _new_user_row(
            email=email,
            password=password,
            role=role,
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            date_of_birth=date_of_birth,
            address=address,
            is_active=is_active,
        )
        with self._lock:
            self._conn.execute(
                """\
                INSERT INTO users (
                    id, email, password_hash, role, first_name, last_name,
                    phone, date_of_birth, address, customer_id, is_active,
                    created_at
                ) VALUES (
                    :id, :email, :password_hash, :role, :first_name,
                    :last_name, :phone, :date_of_birth, :address,
                    :customer_id, :is_active, :created_at
                )""",
                {**user, "is_active": 1 if user["is_active"] else 0},
            )
            self._conn.commit()
        return user

    def get_by_id(self, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            )
            row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_email(self, email: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM users WHERE email = ? COLLATE NOCASE",
                (email.strip().lower(),),
            )
            row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def email_exists(self, email: str) -> bool:
        return self.get_by_email(email) is not None

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        if "is_active" in d:
            d["is_active"] = bool(d["is_active"])
        return d
