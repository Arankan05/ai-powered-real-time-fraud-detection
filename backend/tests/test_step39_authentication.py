"""Step 39 — JWT authentication & role-based authorisation tests.

Covers:

* Registration (validation, hashing, duplicates, role assignment)
* Login (credentials, enumeration safety, inactive accounts)
* JWT validation (expiry, signature, malformed, scheme, claims)
* Token refresh (valid, expired, type confusion)
* Role-based authorisation (customer / fraud_analyst / admin, 401 vs 403)
* Alert endpoint protection and analyst_id provenance
* Transaction endpoint protection and ML-flow regression
* User persistence across repository restarts
* Security / leakage checks (no secrets, hashes, stack traces, ML URLs)

Unlike the Step 33–38 suites, these tests use the REAL authentication
flow (register → login → Bearer token) — no dependency overrides.
"""

from __future__ import annotations

import ast
import importlib
import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import jwt as pyjwt
import pytest
from httpx import Response

from backend.db.alert_repository import InMemoryAlertStore
from backend.db.user_repository import (
    ADMIN,
    CUSTOMER,
    FRAUD_ANALYST,
    InMemoryUserStore,
    SQLiteUserRepository,
)
from backend.schemas import LoginRequest, RegisterRequest
from backend.security.jwt_utils import (
    create_access_token,
    create_refresh_token,
    decode_token,
)
from backend.services.ml_client import MLServiceClient

# ── Helpers ───────────────────────────────────────────────────────────


def _ml_response(decision: str = "HOLD", risk_level: str = "HIGH", risk_score: int = 85) -> dict:
    """Build a complete ML response (same shape as Step 38 tests)."""
    return {
        "fraud_probability": 0.91,
        "fraud_prediction": 1,
        "threshold": 0.50,
        "model_version": "fraud-xgb-v1.0.0",
        "timestamp": 1725200000,
        "explanation": [{"feature": "amount_deviation", "importance": 0.45}],
        "ml_score": 91,
        "behaviour_score": 75,
        "rule_score": 60,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "decision": decision,
        "explanation_detail": {
            "ml_top_factors": [{"feature": "amount_deviation", "importance": 0.45}],
            "behaviour_signals": [
                {"signal": "spending_amount_anomaly", "severity": 0.85,
                 "reason": "Amount 5.2x above average"},
            ],
            "rules_triggered": [
                {"rule": "high_amount", "contribution": 15,
                 "reason": "Amount > $10000"},
            ],
        },
        "risk_factors": ["amount_deviation", "spending_amount_anomaly", "high_amount"],
    }


VALID_TRANSACTION = {
    "amount": 15000.00,
    "currency": "USD",
    "merchant_name": "Offshore Trading Ltd",
    "merchant_category": "5999",
    "transaction_type": "transfer",
    "location_country": "KY",
    "location_city": "George Town",
    "device_fingerprint": "step39-device-001",
    "device_type": "desktop",
    "ip_address": "10.0.0.99",
}


def _register(tc, email="user@example.com", password="SecurePass1",
              first="Jane", last="Doe", **extra):
    payload = {
        "email": email,
        "password": password,
        "first_name": first,
        "last_name": last,
        **extra,
    }
    return tc.post("/api/v1/auth/register", json=payload)


def _login(tc, email, password):
    return tc.post("/api/v1/auth/login", json={"email": email, "password": password})


def _bearer(tc, email="user@example.com", password="SecurePass1"):
    """Register (if new) + login → Authorization header dict."""
    resp = _login(tc, email, password)
    if resp.status_code == 401:
        _register(tc, email=email, password=password)
        resp = _login(tc, email, password)
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def user_store():
    """Fresh in-memory user store per test."""
    return InMemoryUserStore()


@pytest.fixture
def test_client(user_store):
    """TestClient with REAL auth (no overrides), mocked ML, in-memory alerts."""
    from fastapi.testclient import TestClient
    from backend.app import app
    from backend.routers import alerts as alerts_module
    from backend.routers import transactions as txn_module
    from backend.security import deps as deps_module

    # Guarantee a clean dependency-override state (real JWT flow only)
    saved_overrides = dict(app.dependency_overrides)
    app.dependency_overrides.clear()

    deps_module.set_user_repository(user_store)
    txn_module.set_ml_client(MLServiceClient(base_url="http://mock-ml:8001"))
    store = InMemoryAlertStore()
    alerts_module.set_alert_repository(store)
    txn_module.set_alert_repository(store)

    yield TestClient(app), store, user_store

    app.dependency_overrides.clear()
    app.dependency_overrides.update(saved_overrides)


@pytest.fixture
def analyst_headers(test_client):
    """Authenticated fraud_analyst headers (user created directly in repo)."""
    tc, _, store = test_client
    store.create_user(
        email="analyst@example.com",
        password="AnalystPass1",
        role=FRAUD_ANALYST,
        first_name="Fraud",
        last_name="Analyst",
    )
    resp = _login(tc, "analyst@example.com", "AnalystPass1")
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture
def admin_headers(test_client):
    """Authenticated admin headers (user created directly in repo)."""
    tc, _, store = test_client
    store.create_user(
        email="admin@example.com",
        password="AdminPass123",
        role=ADMIN,
        first_name="System",
        last_name="Admin",
    )
    resp = _login(tc, "admin@example.com", "AdminPass123")
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture
def customer_headers(test_client):
    """Authenticated customer headers (registered through the API)."""
    tc, _, _ = test_client
    return _bearer(tc)


@pytest.fixture
def hold_alert(test_client, analyst_headers):
    """Create one OPEN alert via a mocked HOLD transaction. Returns alert dict."""
    tc, store, _ = test_client
    mock_resp = Response(200, json=_ml_response("HOLD", "HIGH", 85))
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        resp = tc.post("/api/v1/transactions", json=VALID_TRANSACTION,
                       headers=analyst_headers)
    assert resp.status_code == 201, resp.text
    assert resp.json()["alert"] is not None
    alerts, _total = store.list_alerts()
    return alerts[0]


# ── 1. Registration ───────────────────────────────────────────────────


class TestRegistration:
    """POST /api/v1/auth/register — public, always creates customers."""

    def test_successful_registration(self, test_client):
        tc, _, _ = test_client
        resp = _register(tc, "jane@example.com")
        assert resp.status_code == 201
        data = resp.json()
        assert data["email"] == "jane@example.com"
        assert data["first_name"] == "Jane"
        assert data["last_name"] == "Doe"
        assert data["role"] == "customer"
        assert data["id"]
        assert data["customer_id"]
        assert uuid.UUID(data["id"])  # valid UUID

    def test_response_has_no_password(self, test_client):
        tc, _, _ = test_client
        resp = _register(tc)
        body = json.dumps(resp.json())
        assert "password" not in body
        assert "hash" not in body

    def test_password_is_hashed(self, test_client):
        tc, _, store = test_client
        _register(tc, "hash@example.com", "SecurePass1")
        user = store.get_by_email("hash@example.com")
        assert user["password_hash"] != "SecurePass1"
        assert user["password_hash"].startswith("$2")  # bcrypt
        assert len(user["password_hash"]) >= 50

    def test_plaintext_password_not_stored(self, test_client):
        tc, _, store = test_client
        _register(tc, "plain@example.com", "Sup3rSecret")
        raw = json.dumps(store.get_by_email("plain@example.com"))
        assert "Sup3rSecret" not in raw

    def test_duplicate_registration_rejected(self, test_client):
        tc, _, _ = test_client
        assert _register(tc, "dup@example.com").status_code == 201
        resp = _register(tc, "dup@example.com")
        assert resp.status_code == 409

    def test_duplicate_case_insensitive(self, test_client):
        tc, _, _ = test_client
        assert _register(tc, "case@example.com").status_code == 201
        resp = _register(tc, "CASE@Example.COM")
        assert resp.status_code == 409

    def test_password_too_short(self, test_client):
        tc, _, _ = test_client
        assert _register(tc, "short@example.com", "Ab1").status_code == 422

    def test_password_no_uppercase(self, test_client):
        tc, _, _ = test_client
        assert _register(tc, "upper@example.com", "lowercase1").status_code == 422

    def test_password_no_lowercase(self, test_client):
        tc, _, _ = test_client
        assert _register(tc, "lower@example.com", "UPPERCASE1").status_code == 422

    def test_password_no_digit(self, test_client):
        tc, _, _ = test_client
        assert _register(tc, "digit@example.com", "NoDigitsHere").status_code == 422

    def test_invalid_email_rejected(self, test_client):
        tc, _, _ = test_client
        resp = tc.post("/api/v1/auth/register", json={
            "email": "not-an-email", "password": "SecurePass1",
            "first_name": "A", "last_name": "B",
        })
        assert resp.status_code == 422

    def test_missing_fields_rejected(self, test_client):
        tc, _, _ = test_client
        resp = tc.post("/api/v1/auth/register", json={"email": "x@example.com"})
        assert resp.status_code == 422

    def test_role_escalation_ignored(self, test_client):
        """Client-supplied role must not be honoured."""
        tc, _, store = test_client
        resp = _register(tc, "escalate@example.com", role="admin",
                         is_active=False, password_hash="injected")
        assert resp.status_code == 201
        assert resp.json()["role"] == "customer"
        user = store.get_by_email("escalate@example.com")
        assert user["role"] == CUSTOMER
        assert user["password_hash"] != "injected"


# ── 2. Login ──────────────────────────────────────────────────────────


class TestLogin:
    """POST /api/v1/auth/login — public token issuance."""

    def test_successful_login_returns_tokens(self, test_client):
        tc, _, _ = test_client
        _register(tc, "login@example.com")
        resp = _login(tc, "login@example.com", "SecurePass1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["access_token"]
        assert data["refresh_token"]
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 1800  # contract: 30 minutes

    def test_invalid_password(self, test_client):
        tc, _, _ = test_client
        _register(tc, "badpw@example.com")
        assert _login(tc, "badpw@example.com", "WrongPass1").status_code == 401

    def test_unknown_user(self, test_client):
        tc, _, _ = test_client
        resp = _login(tc, "ghost@example.com", "Whatever123")
        assert resp.status_code == 401

    def test_no_account_enumeration(self, test_client):
        """Unknown email and wrong password must return identical errors."""
        tc, _, _ = test_client
        _register(tc, "enum@example.com")
        r_unknown = _login(tc, "ghost@example.com", "Whatever123")
        r_wrong = _login(tc, "enum@example.com", "Whatever123")
        assert r_unknown.status_code == r_wrong.status_code == 401
        assert r_unknown.json() == r_wrong.json()

    def test_login_email_case_insensitive(self, test_client):
        tc, _, _ = test_client
        _register(tc, "Case@Test.example.com")
        assert _login(tc, "case@test.example.com", "SecurePass1").status_code == 200

    def test_inactive_user_login_forbidden(self, test_client):
        tc, _, store = test_client
        store.create_user(email="inactive@example.com", password="Inactive1",
                          role=CUSTOMER, is_active=False)
        assert _login(tc, "inactive@example.com", "Inactive1").status_code == 403

    def test_access_token_claims(self, test_client):
        tc, _, store = test_client
        _register(tc, "claims@example.com")
        resp = _login(tc, "claims@example.com", "SecurePass1")
        payload = decode_token(resp.json()["access_token"])
        user = store.get_by_email("claims@example.com")
        assert payload["sub"] == user["id"]
        assert payload["role"] == "customer"
        assert payload["type"] == "access"
        assert payload["exp"] > payload["iat"]
        assert payload["exp"] - payload["iat"] == 1800

    def test_refresh_token_claims(self, test_client):
        tc, _, _ = test_client
        _register(tc, "refresh@example.com")
        resp = _login(tc, "refresh@example.com", "SecurePass1")
        payload = decode_token(resp.json()["refresh_token"])
        assert payload["type"] == "refresh"
        assert payload["role"] == "customer"

    def test_access_and_refresh_differ(self, test_client):
        tc, _, _ = test_client
        _register(tc, "pair@example.com")
        data = _login(tc, "pair@example.com", "SecurePass1").json()
        assert data["access_token"] != data["refresh_token"]

    def test_login_password_not_in_response(self, test_client):
        tc, _, _ = test_client
        _register(tc, "leak@example.com", "LeakCheck1")
        body = json.dumps(_login(tc, "leak@example.com", "LeakCheck1").json())
        assert "LeakCheck1" not in body


# ── 3. JWT validation (via protected endpoints) ───────────────────────


class TestTokenValidation:
    """Invalid tokens must be rejected with 401 — never 500."""

    def _me(self, tc, headers):
        return tc.get("/api/v1/auth/me", headers=headers)

    def test_expired_token(self, test_client):
        tc, _, store = test_client
        user = store.create_user(email="expired@example.com", password="Expiring1")
        token = create_access_token(user_id=user["id"], role=CUSTOMER,
                                    expires_minutes=-5)
        resp = self._me(tc, {"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_malformed_token(self, test_client):
        tc, _, _ = test_client
        for bad in ("not.a.jwt", "abc", "Bearer", "e30.e30.e30"):
            resp = self._me(tc, {"Authorization": f"Bearer {bad}"})
            assert resp.status_code == 401, bad

    def test_missing_authorization_header(self, test_client):
        tc, _, _ = test_client
        assert tc.get("/api/v1/auth/me").status_code == 401

    def test_wrong_scheme(self, test_client):
        tc, _, _ = test_client
        resp = self._me(tc, {"Authorization": "Basic c29tZTp0aGluZw=="})
        assert resp.status_code == 401

    def test_wrong_secret_token(self, test_client):
        tc, _, store = test_client
        user = store.create_user(email="forged@example.com", password="Forged123")
        token = pyjwt.encode(
            {"sub": user["id"], "role": ADMIN, "type": "access",
             "exp": 9999999999, "iat": 1},
            "attacker-known-secret", algorithm="HS256",
        )
        resp = self._me(tc, {"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_alg_none_attack(self, test_client):
        tc, _, _ = test_client
        token = pyjwt.encode(
            {"sub": "x", "role": "admin", "type": "access",
             "exp": 9999999999, "iat": 1},
            key="", algorithm="none",
        )
        resp = self._me(tc, {"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_token_for_nonexistent_user(self, test_client):
        tc, _, _ = test_client
        token = create_access_token(user_id=str(uuid.uuid4()), role=ADMIN)
        assert self._me(tc, {"Authorization": f"Bearer {token}"}).status_code == 401

    def test_refresh_token_rejected_as_access(self, test_client):
        tc, _, store = test_client
        user = store.create_user(email="confuse@example.com", password="Confuse123")
        refresh = create_refresh_token(user_id=user["id"], role=CUSTOMER)
        resp = self._me(tc, {"Authorization": f"Bearer {refresh}"})
        assert resp.status_code == 401

    def test_token_missing_exp_rejected(self, test_client):
        """Hand-crafted token without exp must fail the required-claims check."""
        tc, _, _ = test_client
        from backend.config import get_settings
        token = pyjwt.encode(
            {"sub": "x", "role": "admin", "type": "access"},
            get_settings().BACKEND_SECRET_KEY, algorithm="HS256",
        )
        assert self._me(tc, {"Authorization": f"Bearer {token}"}).status_code == 401

    def test_role_claim_cannot_be_forged(self, test_client):
        """A customer token with a swapped role claim breaks the signature."""
        tc, _, store = test_client
        user = store.create_user(email="swap@example.com", password="Swapping1")
        token = create_access_token(user_id=user["id"], role=CUSTOMER)
        header, payload, sig = token.split(".")
        import base64
        forged = json.loads(base64.urlsafe_b64decode(payload + "=="))
        forged["role"] = "admin"
        forged_b = base64.urlsafe_b64encode(
            json.dumps(forged).encode()).rstrip(b"=").decode()
        tampered = f"{header}.{forged_b}.{sig}"
        resp = tc.get("/api/v1/alerts", headers={"Authorization": f"Bearer {tampered}"})
        assert resp.status_code == 401

    def test_inactive_user_with_valid_token_forbidden(self, test_client):
        tc, _, store = test_client
        user = store.create_user(email="disabled@example.com", password="Disabled1",
                                 role=FRAUD_ANALYST, is_active=False)
        token = create_access_token(user_id=user["id"], role=FRAUD_ANALYST)
        resp = tc.get("/api/v1/alerts", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403


# ── 4. Token refresh ──────────────────────────────────────────────────


class TestRefresh:
    """POST /api/v1/auth/refresh."""

    def test_valid_refresh(self, test_client):
        tc, _, _ = test_client
        _register(tc, "refresh2@example.com")
        data = _login(tc, "refresh2@example.com", "SecurePass1").json()
        resp = tc.post("/api/v1/auth/refresh",
                       json={"refresh_token": data["refresh_token"]})
        assert resp.status_code == 200
        new = resp.json()
        assert new["access_token"]
        assert new["refresh_token"]
        assert new["token_type"] == "bearer"
        assert new["expires_in"] == 1800

    def test_access_token_rejected_as_refresh(self, test_client):
        tc, _, _ = test_client
        _register(tc, "confuse2@example.com")
        data = _login(tc, "confuse2@example.com", "SecurePass1").json()
        resp = tc.post("/api/v1/auth/refresh",
                       json={"refresh_token": data["access_token"]})
        assert resp.status_code == 401

    def test_expired_refresh_token(self, test_client):
        tc, _, store = test_client
        user = store.create_user(email="exprefresh@example.com", password="Expiring1")
        refresh = create_refresh_token(user_id=user["id"], role=CUSTOMER,
                                       expires_days=-1)
        resp = tc.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
        assert resp.status_code == 401

    def test_garbage_refresh_token(self, test_client):
        tc, _, _ = test_client
        resp = tc.post("/api/v1/auth/refresh",
                       json={"refresh_token": "garbage.token.value"})
        assert resp.status_code == 401

    def test_refresh_for_deleted_user(self, test_client):
        tc, _, _ = test_client
        ghost = create_refresh_token(user_id=str(uuid.uuid4()), role=ADMIN)
        resp = tc.post("/api/v1/auth/refresh", json={"refresh_token": ghost})
        assert resp.status_code == 401


# ── 5. /auth/me ───────────────────────────────────────────────────────


class TestMeEndpoint:
    """GET /api/v1/auth/me."""

    def test_me_returns_profile(self, test_client, customer_headers):
        tc, _, _ = test_client
        resp = tc.get("/api/v1/auth/me", headers=customer_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "user@example.com"
        assert data["role"] == "customer"
        assert data["is_active"] is True
        assert data["created_at"]
        assert "password" not in json.dumps(data)

    def test_me_requires_token(self, test_client):
        tc, _, _ = test_client
        assert tc.get("/api/v1/auth/me").status_code == 401

    def test_me_has_www_authenticate_challenge(self, test_client):
        tc, _, _ = test_client
        resp = tc.get("/api/v1/auth/me")
        assert resp.headers.get("WWW-Authenticate") == "Bearer"


# ── 6. Role-based authorisation ───────────────────────────────────────


class TestAuthorization:
    """401 vs 403 across roles on the alert endpoints."""

    def test_unauthenticated_list_rejected(self, test_client):
        tc, _, _ = test_client
        assert tc.get("/api/v1/alerts").status_code == 401

    def test_customer_list_forbidden(self, test_client, customer_headers):
        tc, _, _ = test_client
        assert tc.get("/api/v1/alerts", headers=customer_headers).status_code == 403

    def test_analyst_list_allowed(self, test_client, analyst_headers):
        tc, _, _ = test_client
        resp = tc.get("/api/v1/alerts", headers=analyst_headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_admin_list_allowed(self, test_client, admin_headers):
        tc, _, _ = test_client
        assert tc.get("/api/v1/alerts", headers=admin_headers).status_code == 200

    def test_unauthenticated_detail_rejected(self, test_client, hold_alert):
        tc, _, _ = test_client
        assert tc.get(f"/api/v1/alerts/{hold_alert['id']}").status_code == 401

    def test_customer_detail_forbidden(self, test_client, hold_alert, customer_headers):
        tc, _, _ = test_client
        resp = tc.get(f"/api/v1/alerts/{hold_alert['id']}", headers=customer_headers)
        assert resp.status_code == 403

    def test_analyst_detail_allowed(self, test_client, hold_alert, analyst_headers):
        tc, _, _ = test_client
        resp = tc.get(f"/api/v1/alerts/{hold_alert['id']}", headers=analyst_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == hold_alert["id"]

    def test_unauthenticated_patch_rejected(self, test_client, hold_alert):
        tc, _, _ = test_client
        resp = tc.patch(f"/api/v1/alerts/{hold_alert['id']}",
                        json={"status": "IN_REVIEW"})
        assert resp.status_code == 401

    def test_customer_patch_forbidden(self, test_client, hold_alert, customer_headers):
        tc, _, _ = test_client
        resp = tc.patch(f"/api/v1/alerts/{hold_alert['id']}",
                        json={"status": "IN_REVIEW"}, headers=customer_headers)
        assert resp.status_code == 403

    def test_401_and_403_are_distinct(self, test_client, customer_headers):
        tc, _, _ = test_client
        unauth = tc.get("/api/v1/alerts").status_code
        forbidden = tc.get("/api/v1/alerts", headers=customer_headers).status_code
        assert unauth == 401 and forbidden == 403 and unauth != forbidden


# ── 7. analyst_id provenance ─────────────────────────────────────────


class TestAnalystIdProvenance:
    """analyst_id must come from the JWT identity — never the client."""

    def test_analyst_id_from_authenticated_user(self, test_client, hold_alert,
                                                analyst_headers):
        tc, _, user_store = test_client
        analyst = user_store.get_by_email("analyst@example.com")
        resp = tc.patch(f"/api/v1/alerts/{hold_alert['id']}",
                        json={"status": "IN_REVIEW"}, headers=analyst_headers)
        assert resp.status_code == 200
        assert resp.json()["analyst_id"] == analyst["id"]

    def test_client_cannot_submit_analyst_id(self, test_client, hold_alert,
                                             analyst_headers):
        tc, _, user_store = test_client
        analyst = user_store.get_by_email("analyst@example.com")
        resp = tc.patch(
            f"/api/v1/alerts/{hold_alert['id']}",
            json={"status": "IN_REVIEW", "analyst_id": "attacker-impersonation"},
            headers=analyst_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["analyst_id"] == analyst["id"]
        assert resp.json()["analyst_id"] != "attacker-impersonation"

    def test_admin_update_sets_admin_id(self, test_client, hold_alert,
                                        admin_headers):
        tc, _, user_store = test_client
        admin = user_store.get_by_email("admin@example.com")
        resp = tc.patch(f"/api/v1/alerts/{hold_alert['id']}",
                        json={"status": "DISMISSED"}, headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json()["analyst_id"] == admin["id"]

    def test_notes_only_update_sets_analyst_id(self, test_client, hold_alert,
                                               analyst_headers):
        tc, _, user_store = test_client
        analyst = user_store.get_by_email("analyst@example.com")
        resp = tc.patch(f"/api/v1/alerts/{hold_alert['id']}",
                        json={"notes": "Investigating"}, headers=analyst_headers)
        assert resp.status_code == 200
        assert resp.json()["analyst_id"] == analyst["id"]

    def test_first_analyst_id_not_overwritten(self, test_client, hold_alert,
                                              analyst_headers):
        """A second analyst updating the alert must not steal analyst_id."""
        tc, _, user_store = test_client
        second = user_store.create_user(email="analyst2@example.com",
                                        password="Analyst2Pass1", role=FRAUD_ANALYST)
        first = user_store.get_by_email("analyst@example.com")

        tc.patch(f"/api/v1/alerts/{hold_alert['id']}",
                 json={"status": "IN_REVIEW"}, headers=analyst_headers)
        hdrs2 = {"Authorization": "Bearer " + _login(
            tc, "analyst2@example.com", "Analyst2Pass1").json()["access_token"]}
        resp = tc.patch(f"/api/v1/alerts/{hold_alert['id']}",
                        json={"notes": "Second look"}, headers=hdrs2)
        assert resp.status_code == 200
        assert resp.json()["analyst_id"] == first["id"]
        assert resp.json()["analyst_id"] != second["id"]

    def test_immutable_risk_fields(self, test_client, hold_alert, analyst_headers):
        tc, _, _ = test_client
        resp = tc.patch(
            f"/api/v1/alerts/{hold_alert['id']}",
            json={"status": "IN_REVIEW", "risk_score": 1, "risk_level": "LOW",
                  "decision": "APPROVE", "fraud_probability": 0.01,
                  "transaction_id": "swapped"},
            headers=analyst_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_score"] == 85
        assert data["risk_level"] == "HIGH"
        assert data["decision"] == "HOLD"
        assert data["fraud_probability"] == 0.91
        assert data["transaction_id"] == hold_alert["transaction_id"]

    def test_alert_created_without_analyst_id(self, test_client, hold_alert):
        """Automatic alert creation must not assign an analyst yet."""
        assert hold_alert["analyst_id"] is None


# ── 8. Transaction endpoint protection + regression ──────────────────


class TestTransactionEndpointAuth:
    """POST /api/v1/transactions requires auth; ML flow unchanged."""

    def test_unauthenticated_transaction_rejected(self, test_client):
        tc, _, _ = test_client
        resp = tc.post("/api/v1/transactions", json=VALID_TRANSACTION)
        assert resp.status_code == 401

    def test_ml_not_called_when_unauthenticated(self, test_client):
        tc, _, _ = test_client
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            resp = tc.post("/api/v1/transactions", json=VALID_TRANSACTION)
        assert resp.status_code == 401
        mock_post.assert_not_called()

    def test_customer_can_submit(self, test_client, customer_headers):
        tc, _, _ = test_client
        mock_resp = Response(200, json=_ml_response("APPROVE", "LOW", 12))
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
                   return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=VALID_TRANSACTION,
                           headers=customer_headers)
        assert resp.status_code == 201
        data = resp.json()
        assert data["decision"] == "APPROVE"
        assert data["risk_score"] == 12
        assert data["model_version"] == "fraud-xgb-v1.0.0"
        assert data["timestamp"] == 1725200000
        assert data["alert"] is None  # APPROVE → no alert

    def test_analyst_can_submit(self, test_client, analyst_headers):
        tc, _, _ = test_client
        mock_resp = Response(200, json=_ml_response("VERIFY", "MEDIUM", 50))
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
                   return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=VALID_TRANSACTION,
                           headers=analyst_headers)
        assert resp.status_code == 201
        assert resp.json()["decision"] == "VERIFY"

    def test_hold_still_creates_alert(self, test_client, analyst_headers):
        """Step 38 regression: HOLD → OPEN alert (auth changes nothing)."""
        tc, store, _ = test_client
        mock_resp = Response(200, json=_ml_response("HOLD", "HIGH", 85))
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
                   return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=VALID_TRANSACTION,
                           headers=analyst_headers)
        assert resp.status_code == 201
        assert resp.json()["alert"]["status"] == "OPEN"
        alerts, total = store.list_alerts()
        assert total == 1
        assert alerts[0]["risk_score"] == 85

    def test_invalid_transaction_validation_before_auth_response(self, test_client,
                                                                 customer_headers):
        """Validation still returns 422 for authenticated users."""
        tc, _, _ = test_client
        bad = dict(VALID_TRANSACTION, amount=-5)
        resp = tc.post("/api/v1/transactions", json=bad, headers=customer_headers)
        assert resp.status_code == 422

    def test_ml_unavailable_returns_503(self, test_client, customer_headers):
        """Step 37 regression: ML outage → 503 (auth changes nothing)."""
        tc, _, _ = test_client
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
                   side_effect=httpx.ConnectError("connection refused")):
            resp = tc.post("/api/v1/transactions", json=VALID_TRANSACTION,
                           headers=customer_headers)
        assert resp.status_code == 503


# ── 9. Status transition regression (Step 38) ────────────────────────


class TestAlertWorkflowRegression:
    """The alert lifecycle still works end-to-end behind auth."""

    def test_full_lifecycle(self, test_client, hold_alert, analyst_headers):
        tc, _, _ = test_client
        aid = hold_alert["id"]

        r1 = tc.patch(f"/api/v1/alerts/{aid}", json={"status": "IN_REVIEW"},
                      headers=analyst_headers)
        assert r1.status_code == 200
        assert r1.json()["status"] == "IN_REVIEW"

        r2 = tc.patch(f"/api/v1/alerts/{aid}", json={"status": "RESOLVED"},
                      headers=analyst_headers)
        assert r2.status_code == 200
        assert r2.json()["status"] == "RESOLVED"
        assert r2.json()["resolved_at"] is not None

        r3 = tc.patch(f"/api/v1/alerts/{aid}", json={"status": "OPEN"},
                      headers=analyst_headers)
        assert r3.status_code == 400  # terminal state

    def test_filtering_still_works(self, test_client, hold_alert, analyst_headers):
        tc, _, _ = test_client
        resp = tc.get("/api/v1/alerts", params={"status": "OPEN"},
                      headers=analyst_headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_invalid_transition_rejected(self, test_client, hold_alert,
                                         analyst_headers):
        """Terminal-state transitions are rejected with 400."""
        tc, _, _ = test_client
        aid = hold_alert["id"]
        assert tc.patch(f"/api/v1/alerts/{aid}", json={"status": "RESOLVED"},
                        headers=analyst_headers).status_code == 200
        resp = tc.patch(f"/api/v1/alerts/{aid}", json={"status": "OPEN"},
                        headers=analyst_headers)
        assert resp.status_code == 400


# ── 10. Persistence ───────────────────────────────────────────────────


class TestUserPersistence:
    """Registered users survive repository restarts (SQLite)."""

    def test_user_survives_restart(self, tmp_path):
        from fastapi.testclient import TestClient
        from backend.app import app
        from backend.routers import transactions as txn_module
        from backend.security import deps as deps_module

        db_path = tmp_path / "users.db"
        saved_overrides = dict(app.dependency_overrides)
        app.dependency_overrides.clear()

        repo = SQLiteUserRepository(db_path=db_path)
        deps_module.set_user_repository(repo)
        tc = TestClient(app)
        resp = _register(tc, "persist@example.com", "PersistPass1")
        assert resp.status_code == 201
        repo.close()

        # Simulate restart: new repository instance on the same file
        repo2 = SQLiteUserRepository(db_path=db_path)
        deps_module.set_user_repository(repo2)
        tc2 = TestClient(app)

        login_resp = _login(tc2, "persist@example.com", "PersistPass1")
        assert login_resp.status_code == 200

        me_resp = tc2.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {login_resp.json()['access_token']}"})
        assert me_resp.status_code == 200
        assert me_resp.json()["email"] == "persist@example.com"

        repo2.close()
        app.dependency_overrides.clear()
        app.dependency_overrides.update(saved_overrides)
        txn_module.set_ml_client(MLServiceClient(base_url="http://mock-ml:8001"))

    def test_sqlite_stores_hash_not_plaintext(self, tmp_path):
        repo = SQLiteUserRepository(db_path=tmp_path / "u.db")
        user = repo.create_user(email="sql@example.com", password="Hashed123")
        stored = repo.get_by_email("sql@example.com")
        assert stored["password_hash"] != "Hashed123"
        assert stored["password_hash"].startswith("$2")
        assert user["id"] == stored["id"]
        repo.close()

    def test_duplicate_email_sqlite(self, tmp_path):
        repo = SQLiteUserRepository(db_path=tmp_path / "u.db")
        repo.create_user(email="dup@example.com", password="DupPass123")
        assert repo.email_exists("dup@example.com")
        assert repo.email_exists("DUP@EXAMPLE.COM")  # case-insensitive
        repo.close()

    def test_seed_users_script(self, tmp_path, monkeypatch, capsys):
        """python -m backend.db.seed_users provisions analyst + admin."""
        from backend.db import seed_users

        db_path = tmp_path / "seed.db"
        monkeypatch.setenv("USER_DB_PATH", str(db_path))
        monkeypatch.setenv("SEED_ANALYST_EMAIL", "seed.analyst@example.com")
        monkeypatch.setenv("SEED_ANALYST_PASS", "SeededAnalyst1")
        monkeypatch.setenv("SEED_ADMIN_EMAIL", "seed.admin@example.com")
        monkeypatch.setenv("SEED_ADMIN_PASS", "SeededAdmin12")

        assert seed_users.main() == 0
        repo = SQLiteUserRepository(db_path=db_path)
        analyst = repo.get_by_email("seed.analyst@example.com")
        admin = repo.get_by_email("seed.admin@example.com")
        assert analyst is not None and analyst["role"] == FRAUD_ANALYST
        assert admin is not None and admin["role"] == ADMIN

        # Idempotent: second run skips existing users
        assert seed_users.main() == 0
        repo.close()


# ── 11. Security and leakage ──────────────────────────────────────────


class TestSecurityAndLeakage:
    """No secrets, hashes, internals, or stack traces in responses."""

    def test_register_error_no_internals(self, test_client):
        tc, _, _ = test_client
        resp = tc.post("/api/v1/auth/register", json={
            "email": "bad", "password": "x", "first_name": "", "last_name": "",
        })
        assert resp.status_code == 422
        assert "Traceback" not in resp.text
        assert "sqlite" not in resp.text.lower()
        assert "8001" not in resp.text  # no ML URLs/ports

    def test_401_body_is_minimal(self, test_client):
        tc, _, _ = test_client
        resp = tc.get("/api/v1/alerts")
        assert resp.status_code == 401
        assert set(resp.json().keys()) == {"detail"}
        assert "secret" not in resp.text.lower()
        assert "8001" not in resp.text

    def test_403_body_is_minimal(self, test_client, customer_headers):
        tc, _, _ = test_client
        resp = tc.get("/api/v1/alerts", headers=customer_headers)
        assert resp.status_code == 403
        assert set(resp.json().keys()) == {"detail"}

    def test_no_jwt_secret_in_any_response(self, test_client):
        tc, _, _ = test_client
        from backend.config import get_settings
        secret = get_settings().BACKEND_SECRET_KEY
        _register(tc, "secret@example.com")
        bodies = [
            _login(tc, "secret@example.com", "SecurePass1").text,
            tc.get("/api/v1/auth/me").text,
            tc.get("/api/v1/alerts").text,
        ]
        for body in bodies:
            assert secret not in body

    def test_no_password_hash_in_alerts(self, test_client, hold_alert,
                                        analyst_headers):
        tc, _, user_store = test_client
        user = user_store.get_by_email("analyst@example.com")
        resp = tc.get("/api/v1/alerts", headers=analyst_headers)
        assert user["password_hash"] not in resp.text

    def test_auth_layer_has_no_ml_imports(self):
        """The authentication layer must not depend on the ML package."""
        auth_modules = [
            "backend.security.passwords",
            "backend.security.jwt_utils",
            "backend.security.deps",
            "backend.routers.auth",
            "backend.db.user_repository",
        ]
        for mod_name in auth_modules:
            tree = ast.parse(Path(importlib.import_module(mod_name).__file__)
                             .read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("ml."), \
                            f"{mod_name} imports ML package: {alias.name}"
                elif isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("ml."), \
                        f"{mod_name} imports from ML package: {node.module}"

    def test_no_password_logging_in_auth_layer(self):
        """Logger calls in the auth layer must not format passwords/secrets."""
        auth_modules = [
            Path("backend/routers/auth.py"),
            Path("backend/security/deps.py"),
            Path("backend/security/jwt_utils.py"),
            Path("backend/security/passwords.py"),
            Path("backend/db/user_repository.py"),
        ]
        for path in auth_modules:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if (isinstance(func, ast.Attribute)
                            and func.attr in {"info", "debug", "warning", "error",
                                              "exception", "critical"}):
                        for arg in node.args:
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                low = arg.value.lower()
                                assert "password" not in low, \
                                    f"{path}: logging call mentions password"
                                assert "secret" not in low, \
                                    f"{path}: logging call mentions secret"

    def test_get_register_not_allowed(self, test_client):
        tc, _, _ = test_client
        assert tc.get("/api/v1/auth/register").status_code == 405

    def test_openapi_does_not_expose_secret(self, test_client):
        tc, _, _ = test_client
        resp = tc.get("/openapi.json")
        from backend.config import get_settings
        assert get_settings().BACKEND_SECRET_KEY not in resp.text


# ── 12. Password hashing unit tests ──────────────────────────────────


class TestPasswordHashing:
    """bcrypt hashing utilities."""

    def test_hash_differs_from_plaintext(self):
        from backend.security.passwords import hash_password
        h = hash_password("MySecret1")
        assert h != "MySecret1"
        assert h.startswith("$2")

    def test_hash_is_salted(self):
        from backend.security.passwords import hash_password
        assert hash_password("SamePass1") != hash_password("SamePass1")

    def test_verify_roundtrip(self):
        from backend.security.passwords import hash_password, verify_password
        h = hash_password("RoundTrip1")
        assert verify_password("RoundTrip1", h)
        assert not verify_password("WrongPass1", h)

    def test_verify_malformed_hash_returns_false(self):
        from backend.security.passwords import verify_password
        assert verify_password("Anypass1", "not-a-bcrypt-hash") is False

    def test_long_password_supported(self):
        """128-char passwords (contract max) hash and verify correctly."""
        from backend.security.passwords import hash_password, verify_password
        long_pw = ("Abcdef12" * 16)[:128]
        assert len(long_pw) == 128
        h = hash_password(long_pw)
        assert verify_password(long_pw, h)


# ── 13. Schema validation unit tests ─────────────────────────────────


class TestAuthSchemas:
    """Pydantic request schemas enforce the contract rules."""

    def test_register_schema_rejects_weak_password(self):
        with pytest.raises(Exception):
            RegisterRequest(email="a@example.com", password="alllowercase1",
                            first_name="A", last_name="B")

    def test_register_schema_accepts_valid(self):
        req = RegisterRequest(email="a@example.com", password="GoodPass1",
                              first_name="A", last_name="B")
        assert req.password == "GoodPass1"

    def test_login_schema_requires_email(self):
        with pytest.raises(Exception):
            LoginRequest(email="not-an-email", password="x")

    def test_password_max_length(self):
        with pytest.raises(Exception):
            RegisterRequest(email="a@example.com", password="Ab1" + "x" * 130,
                            first_name="A", last_name="B")
