"""Password hashing utilities (bcrypt).

Passwords are pre-hashed with SHA-256 before bcrypt so that the full
contract-allowed length (128 characters) is honoured — bcrypt only
considers the first 72 bytes of its input.  Plaintext passwords are
never stored or logged.
"""

from __future__ import annotations

import hashlib

import bcrypt


def _prehash(password: str) -> bytes:
    """Return the fixed-length (32-byte) SHA-256 digest of a password."""
    return hashlib.sha256(password.encode("utf-8")).digest()


def hash_password(password: str) -> str:
    """Hash a plaintext password for storage."""
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """Check a plaintext password against a stored bcrypt hash.

    Always returns a boolean — malformed stored hashes are rejected
    instead of raising.
    """
    try:
        return bcrypt.checkpw(_prehash(password), password_hash.encode("ascii"))
    except (ValueError, TypeError, UnicodeDecodeError):
        return False
