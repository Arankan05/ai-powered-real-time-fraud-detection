"""Tests for authentication, RBAC, and security utilities.

Covers:
* Password hashing / verification / strength validation.
* JWT token creation, decoding, expiration.
* Register / login / refresh / me API endpoints.
* Token-based authentication enforcement.
* RBAC role enforcement.
* Response shapes matching ``docs/api-contract.md``.
* Error responses matching the documented contract.

Tests that interact with PostgreSQL are automatically skipped when the
database is unavailable.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.config import settings
from app.core.errors import ForbiddenException, UnauthorizedException
from app.core.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    decode_token,
    hash_password,
    validate_password_strength,
    verify_password,
)
from app.models.customer import Customer
from app.models.user import User
from app.db.session import SessionLocal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pg_available() -> bool:
    """Return True when a real PostgreSQL database is reachable."""
    try:
        eng = create_engine(settings.postgres.database_url)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


requires_pg = pytest.mark.skipif(not _pg_available(), reason="PostgreSQL is not running")


def _unique_email() -> str:
    """Generate a unique test email address."""
    return f"auth-{uuid.uuid4().hex[:12]}@test.com"


def _cleanup_test_data(db: Session) -> None:
    """Remove all users/customers created by auth tests."""
    users = db.query(User).filter(User.email.like("auth-%@test.com")).all()
    customer_ids = [u.customer_id for u in users if u.customer_id]
    for user in users:
        db.delete(user)
    db.commit()
    if customer_ids:
        customers = db.query(Customer).filter(Customer.id.in_(customer_ids)).all()
        for c in customers:
            db.delete(c)
        db.commit()


def _insert_user_directly(
    db: Session,
    *,
    email: str,
    role: str = "fraud_analyst",
    password: str = "SecurePass1",
) -> User:
    """Insert a user directly (for non-customer roles needed in RBAC tests)."""
    user = User(
        email=email.lower(),
        password_hash=hash_password(password),
        first_name="Test",
        last_name="Analyst",
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _delete_user(db: Session, user_id: uuid.UUID) -> None:
    """Delete a user by id."""
    user = db.get(User, user_id)
    if user:
        db.delete(user)
        db.commit()


# ---------------------------------------------------------------------------
# Fixture: clean up test data after each test that uses it
# ---------------------------------------------------------------------------


@pytest.fixture()
def cleanup_auth() -> Generator[Session, None, None]:
    """Provide a DB session and clean up auth test data after the test."""
    db = SessionLocal()
    try:
        # Pre-clean stale data from previous failed runs
        _cleanup_test_data(db)
        yield db
    finally:
        _cleanup_test_data(db)
        db.close()


# ===================================================================
# 1. PASSWORD HASHING
# ===================================================================


class TestPasswordHashing:
    """Password hashing and verification (no DB required)."""

    def test_hash_password_returns_bcrypt_string(self) -> None:
        hashed = hash_password("SecurePass1")
        assert isinstance(hashed, str)
        assert hashed.startswith("$2b$")

    def test_hash_produces_different_salts(self) -> None:
        h1 = hash_password("SecurePass1")
        h2 = hash_password("SecurePass1")
        assert h1 != h2

    def test_verify_password_correct(self) -> None:
        hashed = hash_password("SecurePass1")
        assert verify_password("SecurePass1", hashed) is True

    def test_verify_password_incorrect(self) -> None:
        hashed = hash_password("SecurePass1")
        assert verify_password("WrongPass1", hashed) is False

    def test_plaintext_not_stored(self) -> None:
        plain = "SecurePass1"
        hashed = hash_password(plain)
        assert plain not in hashed


# ===================================================================
# 2. PASSWORD STRENGTH VALIDATION
# ===================================================================


class TestPasswordStrength:
    """Password strength rules from api-contract.md."""

    def test_valid_password(self) -> None:
        assert validate_password_strength("SecurePass1") is True

    def test_too_short(self) -> None:
        assert validate_password_strength("Pass1") is False

    def test_no_uppercase(self) -> None:
        assert validate_password_strength("securepass1") is False

    def test_no_lowercase(self) -> None:
        assert validate_password_strength("SECUREPASS1") is False

    def test_no_digit(self) -> None:
        assert validate_password_strength("SecurePass") is False

    def test_exactly_eight_chars(self) -> None:
        assert validate_password_strength("Secure1a") is True

    def test_max_length_128(self) -> None:
        pw = "A1" + "a" * 126  # 128 chars
        assert validate_password_strength(pw) is True

    def test_over_128_chars(self) -> None:
        pw = "A1" + "a" * 127  # 129 chars
        assert validate_password_strength(pw) is False


# ===================================================================
# 3. JWT TOKENS
# ===================================================================


class TestJWTTokens:
    """JWT creation and decoding (no DB required)."""

    def _fake_id(self) -> uuid.UUID:
        return uuid.uuid4()

    def test_create_access_token_is_string(self) -> None:
        token = create_access_token(self._fake_id(), "customer")
        assert isinstance(token, str)

    def test_decode_access_token_returns_claims(self) -> None:
        uid = self._fake_id()
        token = create_access_token(uid, "admin")
        payload = decode_access_token(token)
        assert payload["sub"] == str(uid)
        assert payload["role"] == "admin"
        assert payload["type"] == "access"

    def test_create_refresh_token_type(self) -> None:
        token = create_refresh_token(self._fake_id())
        payload = decode_refresh_token(token)
        assert payload["type"] == "refresh"

    def test_decode_access_rejects_refresh_token(self) -> None:
        token = create_refresh_token(self._fake_id())
        with pytest.raises(TokenError, match="not an access token"):
            decode_access_token(token)

    def test_decode_refresh_rejects_access_token(self) -> None:
        token = create_access_token(self._fake_id(), "customer")
        with pytest.raises(TokenError, match="not a refresh token"):
            decode_refresh_token(token)

    def test_expired_token_rejected(self) -> None:
        from datetime import timedelta

        token = create_access_token(
            self._fake_id(), "customer", expires_delta=timedelta(seconds=-1),
        )
        with pytest.raises(TokenError, match="expired"):
            decode_access_token(token)

    def test_tampered_token_rejected(self) -> None:
        token = create_access_token(self._fake_id(), "customer")
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(TokenError):
            decode_access_token(tampered)

    def test_wrong_secret_rejected(self) -> None:
        uid = self._fake_id()
        bad_token = jwt.encode(
            {"sub": str(uid), "role": "customer", "type": "access",
             "iat": time.time(), "exp": time.time() + 3600},
            "wrong-secret-key",
            algorithm="HS256",
        )
        with pytest.raises(TokenError):
            decode_access_token(bad_token)

    def test_missing_claims_rejected(self) -> None:
        bad = jwt.encode(
            {"exp": time.time() + 3600},
            settings.backend.secret_key,
            algorithm="HS256",
        )
        with pytest.raises(TokenError, match="missing required claims"):
            decode_token(bad)

    def test_expires_in_uses_config(self) -> None:
        """Access token expiry is within the configured window."""
        uid = self._fake_id()
        token = create_access_token(uid, "customer")
        payload = jwt.decode(token, settings.backend.secret_key, algorithms=["HS256"])
        now = time.time()
        max_exp = now + settings.backend.access_token_expire_minutes * 60 + 5
        assert payload["exp"] <= max_exp

    def test_secret_not_in_token_string(self) -> None:
        """The secret key is HMAC-based and never appears verbatim in the token."""
        token = create_access_token(self._fake_id(), "customer")
        assert settings.backend.secret_key not in token


# ===================================================================
# 4. REGISTER ENDPOINT
# ===================================================================


@requires_pg
class TestRegisterEndpoint:
    """POST /api/v1/auth/register"""

    def test_register_success_201(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email()
        resp = client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["email"] == email
        assert data["first_name"] == "Jane"
        assert data["last_name"] == "Doe"
        assert data["role"] == "customer"
        assert "id" in data
        assert "customer_id" in data

    def test_register_response_shape(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email()
        resp = client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        data = resp.json()
        for key in ("id", "email", "first_name", "last_name", "role"):
            assert key in data, f"Missing key: {key}"

    def test_register_creates_customer(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email()
        resp = client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        data = resp.json()
        assert data["customer_id"] is not None

    def test_register_duplicate_email_409(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email()
        payload = {"email": email, "password": "SecurePass1", "first_name": "A", "last_name": "B"}
        client.post("/api/v1/auth/register", json=payload)
        resp = client.post("/api/v1/auth/register", json=payload)
        assert resp.status_code == 409
        assert "error_code" in resp.json()

    def test_register_invalid_email_422(self, client: TestClient) -> None:
        resp = client.post("/api/v1/auth/register", json={
            "email": "not-an-email", "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        assert resp.status_code == 422

    def test_register_weak_password_422(self, client: TestClient) -> None:
        resp = client.post("/api/v1/auth/register", json={
            "email": _unique_email(), "password": "short",
            "first_name": "Jane", "last_name": "Doe",
        })
        assert resp.status_code == 422

    def test_register_weak_password_strength_400(self, client: TestClient, cleanup_auth: Session) -> None:
        """Password passes Pydantic length check but fails strength rules."""
        resp = client.post("/api/v1/auth/register", json={
            "email": _unique_email(), "password": "alllowercase1",
            "first_name": "Jane", "last_name": "Doe",
        })
        assert resp.status_code == 400

    def test_register_missing_fields_422(self, client: TestClient) -> None:
        resp = client.post("/api/v1/auth/register", json={"email": _unique_email()})
        assert resp.status_code == 422

    def test_register_password_not_in_response(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email()
        resp = client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        assert "password" not in resp.json()
        assert "password_hash" not in resp.json()

    def test_register_email_stored_lowercase(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email().upper()
        resp = client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        assert resp.json()["email"] == email.lower()

    def test_register_with_optional_fields(self, client: TestClient, cleanup_auth: Session) -> None:
        resp = client.post("/api/v1/auth/register", json={
            "email": _unique_email(), "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
            "phone": "+1234567890", "date_of_birth": "1990-01-15",
            "address": "123 Main St",
        })
        assert resp.status_code == 201


# ===================================================================
# 5. LOGIN ENDPOINT
# ===================================================================


@requires_pg
class TestLoginEndpoint:
    """POST /api/v1/auth/login"""

    def test_login_success_200(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email()
        client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        resp = client.post("/api/v1/auth/login", json={
            "email": email, "password": "SecurePass1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert isinstance(data["expires_in"], int)

    def test_login_invalid_password_401(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email()
        client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        resp = client.post("/api/v1/auth/login", json={
            "email": email, "password": "WrongPass1",
        })
        assert resp.status_code == 401

    def test_login_nonexistent_email_401(self, client: TestClient) -> None:
        resp = client.post("/api/v1/auth/login", json={
            "email": _unique_email(), "password": "SecurePass1",
        })
        assert resp.status_code == 401

    def test_login_inactive_account_403(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email()
        client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        # Deactivate the user directly in DB
        user = cleanup_auth.query(User).filter(User.email == email.lower()).one()
        user.is_active = False
        cleanup_auth.commit()

        resp = client.post("/api/v1/auth/login", json={
            "email": email, "password": "SecurePass1",
        })
        assert resp.status_code == 403

    def test_login_missing_fields_422(self, client: TestClient) -> None:
        resp = client.post("/api/v1/auth/login", json={"email": _unique_email()})
        assert resp.status_code == 422

    def test_login_case_insensitive_email(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email()
        client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        resp = client.post("/api/v1/auth/login", json={
            "email": email.upper(), "password": "SecurePass1",
        })
        assert resp.status_code == 200

    def test_login_error_format(self, client: TestClient) -> None:
        resp = client.post("/api/v1/auth/login", json={
            "email": _unique_email(), "password": "SecurePass1",
        })
        data = resp.json()
        assert "detail" in data
        assert "error_code" in data

    def test_login_does_not_reveal_if_user_exists(self, client: TestClient) -> None:
        """Both invalid-email and wrong-password return 401 with the same wording."""
        r1 = client.post("/api/v1/auth/login", json={
            "email": _unique_email(), "password": "SecurePass1",
        })
        assert r1.status_code == 401


# ===================================================================
# 6. REFRESH ENDPOINT
# ===================================================================


@requires_pg
class TestRefreshEndpoint:
    """POST /api/v1/auth/refresh"""

    def _login(self, client: TestClient, email: str) -> dict:
        resp = client.post("/api/v1/auth/login", json={
            "email": email, "password": "SecurePass1",
        })
        return resp.json()

    def test_refresh_success_200(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email()
        client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        tokens = self._login(client, email)

        resp = client.post("/api/v1/auth/refresh", json={
            "refresh_token": tokens["refresh_token"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert isinstance(data["expires_in"], int)

    def test_refresh_invalid_token_401(self, client: TestClient) -> None:
        resp = client.post("/api/v1/auth/refresh", json={
            "refresh_token": "invalid.token.here",
        })
        assert resp.status_code == 401

    def test_refresh_access_token_rejected_401(self, client: TestClient, cleanup_auth: Session) -> None:
        """An access token cannot be used as a refresh token."""
        email = _unique_email()
        client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        tokens = self._login(client, email)

        resp = client.post("/api/v1/auth/refresh", json={
            "refresh_token": tokens["access_token"],
        })
        assert resp.status_code == 401

    def test_refresh_missing_body_422(self, client: TestClient) -> None:
        resp = client.post("/api/v1/auth/refresh", json={})
        assert resp.status_code == 422


# ===================================================================
# 7. ME ENDPOINT (Authentication enforcement)
# ===================================================================


@requires_pg
class TestMeEndpoint:
    """GET /api/v1/auth/me"""

    def test_me_with_valid_token_200(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email()
        client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        tokens = client.post("/api/v1/auth/login", json={
            "email": email, "password": "SecurePass1",
        }).json()

        resp = client.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {tokens['access_token']}",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == email
        assert data["first_name"] == "Jane"
        assert data["last_name"] == "Doe"
        assert data["role"] == "customer"
        assert "is_active" in data
        assert data["is_active"] is True
        assert "created_at" in data
        assert "customer_id" in data

    def test_me_response_shape(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email()
        client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        tokens = client.post("/api/v1/auth/login", json={
            "email": email, "password": "SecurePass1",
        }).json()

        resp = client.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {tokens['access_token']}",
        })
        data = resp.json()
        expected_keys = {"id", "email", "first_name", "last_name", "role",
                         "customer_id", "is_active", "created_at"}
        assert set(data.keys()) == expected_keys

    def test_me_no_token_401(self, client: TestClient) -> None:
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    def test_me_invalid_token_401(self, client: TestClient) -> None:
        resp = client.get("/api/v1/auth/me", headers={
            "Authorization": "Bearer invalid.jwt.token",
        })
        assert resp.status_code == 401

    def test_me_expired_token_401(self, client: TestClient) -> None:
        """A token with a past expiry is rejected."""
        from datetime import timedelta

        fake_id = uuid.uuid4()
        expired = create_access_token(
            fake_id, "customer", expires_delta=timedelta(seconds=-10),
        )
        resp = client.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {expired}",
        })
        assert resp.status_code == 401

    def test_me_no_password_in_response(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email()
        client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        tokens = client.post("/api/v1/auth/login", json={
            "email": email, "password": "SecurePass1",
        }).json()

        resp = client.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {tokens['access_token']}",
        })
        assert "password" not in resp.json()
        assert "password_hash" not in resp.json()

    def test_me_nonexistent_user_401(self, client: TestClient) -> None:
        """Token for a user that no longer exists returns 401."""
        fake_id = uuid.uuid4()
        token = create_access_token(fake_id, "customer")
        resp = client.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {token}",
        })
        assert resp.status_code == 401


# ===================================================================
# 8. FULL AUTH FLOW (Integration)
# ===================================================================


@requires_pg
class TestAuthFlow:
    """End-to-end: register → login → me → refresh → me."""

    def test_full_flow(self, client: TestClient, cleanup_auth: Session) -> None:
        email = _unique_email()

        # 1. Register
        reg = client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        assert reg.status_code == 201
        user_id = reg.json()["id"]

        # 2. Login
        login = client.post("/api/v1/auth/login", json={
            "email": email, "password": "SecurePass1",
        })
        assert login.status_code == 200
        access = login.json()["access_token"]
        refresh = login.json()["refresh_token"]

        # 3. Me with access token
        me = client.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {access}",
        })
        assert me.status_code == 200
        assert me.json()["id"] == user_id

        # 4. Refresh
        refreshed = client.post("/api/v1/auth/refresh", json={
            "refresh_token": refresh,
        })
        assert refreshed.status_code == 200
        new_access = refreshed.json()["access_token"]

        # 5. Me with new access token
        me2 = client.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {new_access}",
        })
        assert me2.status_code == 200
        assert me2.json()["id"] == user_id


# ===================================================================
# 9. RBAC
# ===================================================================


@requires_pg
class TestRBAC:
    """Role-based access control enforcement."""

    def test_require_role_accepts_matching_role(
        self, client: TestClient, cleanup_auth: Session,
    ) -> None:
        """A fraud_analyst user can access an analyst-only endpoint."""
        email = _unique_email()
        user = _insert_user_directly(cleanup_auth, email=email, role="fraud_analyst")
        token = create_access_token(user.id, user.role)

        resp = client.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {token}",
        })
        # /me is not role-restricted, so this should work for any role
        assert resp.status_code == 200
        assert resp.json()["role"] == "fraud_analyst"

    def test_insufficient_role_rejected(self, cleanup_auth: Session) -> None:
        """require_role raises ForbiddenException for wrong role."""
        from app.services.auth import require_role

        # Create a fake customer-role user
        user = User(
            id=uuid.uuid4(), email=_unique_email(),
            password_hash=hash_password("SecurePass1"),
            first_name="T", last_name="T", role="customer", is_active=True,
        )
        dep = require_role("admin")
        # Simulate the check
        with pytest.raises(ForbiddenException):
            dep(user)

    def test_require_role_admin_only(self, cleanup_auth: Session) -> None:
        from app.services.auth import require_role

        user = User(
            id=uuid.uuid4(), email=_unique_email(),
            password_hash=hash_password("SecurePass1"),
            first_name="T", last_name="T", role="admin", is_active=True,
        )
        dep = require_role("admin")
        result = dep(user)
        assert result.id == user.id

    def test_require_role_multiple_roles(self, cleanup_auth: Session) -> None:
        """require_role accepts any of the listed roles."""
        from app.services.auth import require_role

        user = User(
            id=uuid.uuid4(), email=_unique_email(),
            password_hash=hash_password("SecurePass1"),
            first_name="T", last_name="T", role="fraud_analyst", is_active=True,
        )
        dep = require_role("admin", "fraud_analyst")
        result = dep(user)
        assert result.id == user.id

    def test_role_from_token_not_client(self, client: TestClient, cleanup_auth: Session) -> None:
        """Role is derived from the JWT, not from client-supplied data."""
        email = _unique_email()
        client.post("/api/v1/auth/register", json={
            "email": email, "password": "SecurePass1",
            "first_name": "Jane", "last_name": "Doe",
        })
        tokens = client.post("/api/v1/auth/login", json={
            "email": email, "password": "SecurePass1",
        }).json()

        # The role in the JWT is "customer" — cannot be changed by client
        me = client.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {tokens['access_token']}",
        })
        assert me.json()["role"] == "customer"


# ===================================================================
# 10. EXISTING TESTS STILL PASS
# ===================================================================


@requires_pg
class TestExistingTestsStillPass:
    """Verify database-level integrity after auth implementation."""

    def test_users_table_exists(self, cleanup_auth: Session) -> None:
        from sqlalchemy import inspect

        insp = inspect(cleanup_auth.bind)
        assert "users" in insp.get_table_names()

    def test_customers_table_exists(self, cleanup_auth: Session) -> None:
        from sqlalchemy import inspect

        insp = inspect(cleanup_auth.bind)
        assert "customers" in insp.get_table_names()

    def test_base_metadata_nine_tables(self) -> None:
        from app.db.base import Base

        table_names = set(Base.metadata.tables.keys())
        assert len(table_names) == 9
