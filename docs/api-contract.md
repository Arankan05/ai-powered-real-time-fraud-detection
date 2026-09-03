# API Contract

## Overview

RESTful JSON API. **Base URL:** `/api/v1`. All endpoints require a valid JWT bearer token unless marked **Public**.

## Global Validation Rules

| Field | Rules |
|---|---|
| `email` | Valid email format, max 255 chars, case-insensitive storage |
| `password` | Min 8 chars, max 128 chars, at least 1 uppercase, 1 lowercase, 1 digit |
| `amount` | Positive decimal, max 2 decimal places, max 9999999999.99, min 0.01 |
| `currency` | ISO 4217 3-letter code (e.g., USD, EUR, GBP) |
| `transaction_type` | Enum: `purchase`, `transfer`, `withdrawal` |
| `device_fingerprint` | Non-empty string, max 255 chars |
| `device_type` | Enum: `mobile`, `desktop`, `pos` |
| `ip_address` | Valid IPv4 or IPv6, max 45 chars |
| `location_country` | Non-empty string, max 100 chars |
| `location_city` | Max 100 chars |
| `merchant_name` | Non-empty string, max 255 chars |
| `merchant_category` | Max 10 chars (MCC code or category label) |
| `date` fields | ISO 8601 format (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SSZ`) |
| `page` | Integer ≥ 1, default 1 |
| `per_page` | Integer 1–100, default 20 |
| `risk_level` | Enum: `LOW`, `MEDIUM`, `HIGH` |
| `alert_status` | Enum: `OPEN`, `IN_REVIEW`, `RESOLVED`, `DISMISSED` |
| `transaction_status` | Enum: `PENDING`, `COMPLETED`, `FAILED` (lifecycle only; fraud outcome is in `decision`/`risk_level`) |

---

## Authentication Endpoints

### `POST /api/v1/auth/register`

**Public.** Create a new user account with `customer` role.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass1",
  "first_name": "Jane",
  "last_name": "Doe",
  "phone": "+1234567890",
  "date_of_birth": "1990-01-15",
  "address": "123 Main St, City, Country"
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "first_name": "Jane",
  "last_name": "Doe",
  "role": "customer",
  "customer_id": "uuid"
}
```

**Errors:** 400 (validation), 409 (email exists), 422 (Pydantic).

---

### `POST /api/v1/auth/login`

**Public.** Authenticate and receive JWT tokens.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass1"
}
```

**Response (200):**
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Errors:** 401 (invalid credentials), 403 (inactive account), 422.

---

### `POST /api/v1/auth/refresh`

**Auth required.** Refresh an expiring access token.

**Request:**
```json
{
  "refresh_token": "string"
}
```

**Response (200):**
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Errors:** 401 (invalid/expired refresh token), 422.

---

### `GET /api/v1/auth/me`

**Auth required.** Return the currently authenticated user's profile.

**Response (200):**
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "first_name": "Jane",
  "last_name": "Doe",
  "role": "customer",
  "customer_id": "uuid",
  "is_active": true,
  "created_at": "2026-01-15T10:30:00Z"
}
```

**Errors:** 401.

---

## Transaction Endpoints

### `POST /api/v1/transactions`

**Auth required (customer).** Submit a transaction. The backend automatically runs the full fraud detection pipeline internally and persists the transaction with all fraud results.

**Flow:**
```
POST /api/v1/transactions
  → Transaction Service (validate + persist raw transaction)
  → ML/Fraud Intelligence Service (internal HTTP call)
  → Fraud analysis (ML + behaviour + rules + aggregation + explainability)
  → Persist fraud results to transaction record
  → Create alert if decision is HOLD
  → Return complete result
```

The frontend calls only this endpoint. It must not separately call `POST /api/v1/fraud/check`.

**Request:**
```json
{
  "amount": 1500.00,
  "currency": "USD",
  "merchant_name": "Acme Electronics",
  "merchant_category": "5732",
  "transaction_type": "purchase",
  "location_country": "US",
  "location_city": "New York",
  "device_fingerprint": "abc123def456",
  "device_type": "mobile",
  "ip_address": "192.168.1.100"
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "customer_id": "uuid",
  "merchant_id": "uuid",
  "amount": 1500.00,
  "currency": "USD",
  "merchant_name": "Acme Electronics",
  "merchant_category": "5732",
  "transaction_type": "purchase",
  "location_country": "US",
  "location_city": "New York",
  "device_fingerprint": "abc123def456",
  "device_type": "mobile",
  "ip_address": "192.168.1.100",
  "timestamp": "2026-08-29T14:30:00Z",
  "status": "COMPLETED",
  "ml_score": 35,
  "behaviour_score": 52,
  "rule_score": 15,
  "risk_score": 45,
  "risk_level": "MEDIUM",
  "decision": "VERIFY",
  "explanation": {
    "ml_top_factors": [
      { "feature": "amount_deviation", "importance": 0.35 },
      { "feature": "is_new_device", "importance": 0.22 }
    ],
    "behaviour_signals": [
      { "signal": "spending_amount_anomaly", "severity": 0.6 },
      { "signal": "device_anomaly", "severity": 0.5 }
    ],
    "rules_triggered": [
      { "rule": "new_device_high_amount", "contribution": 15 }
    ]
  },
  "risk_factors": [
    "amount_deviation",
    "is_new_device",
    "new_device_high_amount"
  ],
  "model_version": "fraud-xgb-v1.2.0",
  "alert": null
}
```

When the decision is HOLD (risk_level = HIGH), the `alert` field contains:
```json
{
  "alert": {
    "id": "uuid",
    "status": "OPEN",
    "created_at": "2026-08-29T14:30:00Z"
  }
}
```

**Errors:** 400 (validation), 401, 403 (non-customer), 422, 503 (ML service unavailable).

---

### `GET /api/v1/transactions`

**Auth required.** Customers see only their own. Analysts/admins see all.

**Query parameters:**

| Param | Type | Required | Default | Description |
|---|---|---|---|---|
| `page` | int | No | 1 | Page number |
| `per_page` | int | No | 20 | Items per page (max 100) |
| `status` | string | No | — | Filter by transaction status |
| `risk_level` | string | No | — | Filter by risk level |
| `from_date` | ISO 8601 | No | — | Start of date range |
| `to_date` | ISO 8601 | No | — | End of date range |

**Response (200):**
```json
{
  "items": [
    {
      "id": "uuid",
      "customer_id": "uuid",
      "merchant_name": "Acme Electronics",
      "amount": 1500.00,
      "currency": "USD",
      "transaction_type": "purchase",
      "timestamp": "2026-08-29T14:30:00Z",
      "status": "COMPLETED",
      "risk_score": 45,
      "risk_level": "MEDIUM",
      "decision": "VERIFY"
    }
  ],
  "total": 150,
  "page": 1,
  "per_page": 20
}
```

**Errors:** 401, 422.

---

### `GET /api/v1/transactions/{id}`

**Auth required.** Customers see own only. Analysts/admins see any.

**Response (200):** Same full structure as `POST /api/v1/transactions` response (including `ml_score`, `behaviour_score`, `rule_score`, `explanation`, `risk_factors`, `model_version`, `alert`).

**Errors:** 401, 403, 404.

---

## Fraud Check Endpoint

### `POST /api/v1/fraud/check`

**Auth required (fraud_analyst, admin).** Run fraud analysis on a transaction payload without creating a persisted transaction or alert. Used by analysts to test scenarios, re-evaluate data, or preview risk scoring.

This endpoint does NOT create a transaction record, does NOT create an alert, and does NOT modify any customer/device history.

The normal customer flow does NOT call this endpoint — it calls `POST /api/v1/transactions` which internally uses the same fraud analysis logic.

**Request:**
```json
{
  "customer_id": "uuid",
  "amount": 1500.00,
  "currency": "USD",
  "merchant_name": "Acme Electronics",
  "merchant_category": "5732",
  "transaction_type": "purchase",
  "location_country": "US",
  "location_city": "New York",
  "device_fingerprint": "abc123def456",
  "device_type": "mobile",
  "ip_address": "192.168.1.100"
}
```

**Response (200):**
```json
{
  "ml_score": 35,
  "behaviour_score": 52,
  "rule_score": 15,
  "risk_score": 45,
  "risk_level": "MEDIUM",
  "decision": "VERIFY",
  "explanation": {
    "ml_top_factors": [
      { "feature": "amount_deviation", "importance": 0.35 },
      { "feature": "is_new_device", "importance": 0.22 }
    ],
    "behaviour_signals": [
      { "signal": "spending_amount_anomaly", "severity": 0.6 },
      { "signal": "device_anomaly", "severity": 0.5 }
    ],
    "rules_triggered": [
      { "rule": "new_device_high_amount", "contribution": 15 }
    ]
  },
  "risk_factors": [
    "amount_deviation",
    "is_new_device",
    "new_device_high_amount"
  ],
  "model_version": "fraud-xgb-v1.2.0"
}
```

**Errors:** 400 (validation), 401, 403 (customer role denied), 404 (customer not found), 503 (ML service unavailable).

---

## Alert Endpoints

### `GET /api/v1/alerts`

**Auth required (fraud_analyst, admin).** List fraud alerts.

**Query parameters:**

| Param | Type | Required | Default | Description |
|---|---|---|---|---|
| `page` | int | No | 1 | Page number |
| `per_page` | int | No | 20 | Items per page (max 100) |
| `status` | string | No | — | Filter: OPEN, IN_REVIEW, RESOLVED, DISMISSED |
| `risk_level` | string | No | — | Filter by risk level (via JOIN to transactions) |

**Response (200):**
```json
{
  "items": [
    {
      "id": "uuid",
      "transaction_id": "uuid",
      "risk_score": 85,
      "risk_level": "HIGH",
      "decision": "HOLD",
      "status": "OPEN",
      "analyst_id": null,
      "notes": null,
      "created_at": "2026-08-29T14:30:00Z",
      "resolved_at": null,
      "transaction_summary": {
        "amount": 15000.00,
        "currency": "USD",
        "merchant_name": "Offshore Trading Ltd",
        "transaction_type": "transfer",
        "customer_email": "user@example.com",
        "timestamp": "2026-08-29T14:30:00Z"
      }
    }
  ],
  "total": 12,
  "page": 1,
  "per_page": 20
}
```

**Errors:** 401, 403, 422.

---

### `GET /api/v1/alerts/{id}`

**Auth required (fraud_analyst, admin).** Get a single alert with full details including the associated transaction.

**Response (200):**
```json
{
  "id": "uuid",
  "transaction_id": "uuid",
  "risk_score": 85,
  "risk_level": "HIGH",
  "decision": "HOLD",
  "explanation": {
    "ml_top_factors": [
      { "feature": "amount_deviation", "importance": 0.45 },
      { "feature": "location_is_new", "importance": 0.30 }
    ],
    "behaviour_signals": [
      { "signal": "spending_amount_anomaly", "severity": 0.9 },
      { "signal": "location_anomaly", "severity": 0.8 }
    ],
    "rules_triggered": [
      { "rule": "high_amount", "contribution": 15 },
      { "rule": "impossible_travel", "contribution": 25 }
    ]
  },
  "risk_factors": ["amount_deviation", "location_is_new", "high_amount", "impossible_travel"],
  "status": "OPEN",
  "analyst_id": null,
  "notes": null,
  "created_at": "2026-08-29T14:30:00Z",
  "resolved_at": null,
  "transaction": {
    "id": "uuid",
    "customer_id": "uuid",
    "amount": 15000.00,
    "currency": "USD",
    "merchant_name": "Offshore Trading Ltd",
    "transaction_type": "transfer",
    "location_country": "KY",
    "location_city": "George Town",
    "device_type": "desktop",
    "timestamp": "2026-08-29T14:30:00Z",
    "ml_score": 78,
    "behaviour_score": 88,
    "rule_score": 40
  }
}
```

**Errors:** 401, 403, 404.

---

### `PATCH /api/v1/alerts/{id}`

**Auth required (fraud_analyst, admin).** Update alert status and/or analyst notes.

**Request:**
```json
{
  "status": "IN_REVIEW",
  "notes": "Customer confirmed this was a legitimate purchase while travelling."
}
```

Both fields are optional; at least one must be provided.

Valid status transitions:
- `OPEN` → `IN_REVIEW`, `RESOLVED`, `DISMISSED`
- `IN_REVIEW` → `RESOLVED`, `DISMISSED`
- `RESOLVED` / `DISMISSED` → no further transitions

When `status` changes to `RESOLVED` or `DISMISSED`, `resolved_at` is set automatically. When an analyst first updates an alert, `analyst_id` is set to the current user if not already assigned.

**Response (200):**
```json
{
  "id": "uuid",
  "transaction_id": "uuid",
  "risk_score": 85,
  "risk_level": "HIGH",
  "decision": "HOLD",
  "status": "IN_REVIEW",
  "analyst_id": "uuid",
  "notes": "Customer confirmed this was a legitimate purchase while travelling.",
  "created_at": "2026-08-29T14:30:00Z",
  "resolved_at": null
}
```

**Errors:** 400 (invalid transition), 401, 403, 404, 422.

---

## Customer Endpoints

### `GET /api/v1/customers/me`

**Auth required (customer).** Get the current customer's own profile.

**Response (200):**
```json
{
  "id": "uuid",
  "first_name": "Jane",
  "last_name": "Doe",
  "phone": "+1234567890",
  "address": "123 Main St, City, Country",
  "date_of_birth": "1990-01-15",
  "created_at": "2026-01-15T10:30:00Z",
  "is_active": true
}
```

**Errors:** 401, 403 (non-customer).

### `GET /api/v1/customers/{id}`

**Auth required.** Customers see own only. Analysts/admins see any.

**Response (200):** Same structure as `GET /api/v1/customers/me`.

**Errors:** 401, 403, 404.

### `GET /api/v1/customers/{id}/transactions`

**Auth required.** Get transaction history for a specific customer.

**Query parameters:** Same as `GET /api/v1/transactions` (page, per_page, status, risk_level, from_date, to_date).

**Response (200):** Same paginated structure as `GET /api/v1/transactions`.

**Errors:** 401, 403, 404, 422.

---

## Analytics Endpoint

### `GET /api/v1/analytics/dashboard`

**Auth required (fraud_analyst, admin).** Aggregated metrics for the fraud analyst dashboard.

**Query parameters:**

| Param | Type | Required | Default | Description |
|---|---|---|---|---|
| `from_date` | ISO 8601 date | No | 30 days ago | Start of analysis period (UTC) |
| `to_date` | ISO 8601 date | No | now | End of analysis period (UTC) |

Format: `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SSZ`. When omitted, the default range is the last 30 days.

**Response (200):**
```json
{
  "from_date": "2026-07-30T00:00:00Z",
  "to_date": "2026-08-29T23:59:59Z",
  "total_transactions": 15420,
  "flagged_transactions": 342,
  "alerts_open": 28,
  "alerts_resolved": 156,
  "risk_distribution": { "LOW": 12500, "MEDIUM": 2400, "HIGH": 520 },
  "top_risk_factors": [
    { "factor": "amount_deviation", "count": 180 },
    { "factor": "new_device", "count": 95 },
    { "factor": "impossible_travel", "count": 42 }
  ],
  "transactions_over_time": [
    { "date": "2026-08-01", "total": 500, "flagged": 12 },
    { "date": "2026-08-02", "total": 480, "flagged": 8 }
  ]
}
```

**Errors:** 401, 403, 422 (invalid date range).

---

## Health Endpoint

### `GET /api/v1/health`

**Public.** System health check.

**Response (200):**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {
    "database": { "status": "connected" },
    "ml_service": { "status": "connected", "model_version": "fraud-xgb-v1.2.0" }
  }
}
```

When degraded:
```json
{
  "status": "degraded",
  "version": "0.1.0",
  "services": {
    "database": { "status": "connected" },
    "ml_service": { "status": "unavailable", "error": "Connection refused" }
  }
}
```

HTTP 200 when `healthy`, 503 when `degraded`.

---

## Standard Error Response

```json
{
  "detail": "Human-readable error message",
  "error_code": "ERROR_CODE_CONSTANT"
}
```

Validation errors (422):
```json
{
  "detail": "Validation failed",
  "error_code": "VALIDATION_ERROR",
  "errors": [
    { "field": "amount", "message": "Amount must be greater than 0" },
    { "field": "email", "message": "Invalid email format" }
  ]
}
```

| Code | Meaning |
|---|---|
| 200 | Success |
| 201 | Created |
| 400 | Bad request |
| 401 | Unauthenticated |
| 403 | Unauthorised |
| 404 | Not found |
| 409 | Conflict |
| 422 | Validation error |
| 503 | Service unavailable |
| 500 | Internal server error |

## Versioning

API versioned via URL path (`/api/v1`). Breaking changes require a new version.

## Status

This contract is agreed upon but **not yet implemented**. Changes require both frontend and backend team agreement.

---

## Implementation Notes — Authentication & Authorization (Step 39)

> Status of the contract above relative to the running backend on
> `feature/ml-fraud`. The contract remains the source of truth; this
> section documents what is implemented and where behaviour is
> clarified.

### Implemented endpoints

| Endpoint | Status | Auth |
|---|---|---|
| `POST /api/v1/auth/register` | Implemented | Public — always creates `customer` accounts |
| `POST /api/v1/auth/login` | Implemented | Public |
| `POST /api/v1/auth/refresh` | Implemented | Refresh token in body (see note) |
| `GET /api/v1/auth/me` | Implemented | Bearer JWT |
| `POST /api/v1/transactions` | Implemented | Bearer JWT (any active role) |
| `PATCH /api/v1/transactions/outcome` | Implemented | Bearer JWT (`fraud_analyst`, `admin`) |
| `GET /api/v1/alerts` | Implemented | Bearer JWT (`fraud_analyst`, `admin`) |
| `GET /api/v1/alerts/{id}` | Implemented | Bearer JWT (`fraud_analyst`, `admin`) |
| `PATCH /api/v1/alerts/{id}` | Implemented | Bearer JWT (`fraud_analyst`, `admin`) |

### Roles

* `customer` — default role from public registration; can submit
  transactions and access own profile (via `/auth/me`).
* `fraud_analyst` — can list, view, and update alerts; can submit
  outcome feedback.
* `admin` — all analyst permissions.

Role escalation through the public API is impossible: registration
  ignores any client-supplied role, and roles are carried inside the
  signed JWT (tampering invalidates the signature).

### JWT details

* Signed HS256 tokens with claims `sub` (user ID), `role`, `type`
  (`access` / `refresh`), `iat`, `exp`.
* Access tokens expire after 30 minutes (response `expires_in: 1800`);
  refresh tokens after 7 days.
* Access and refresh tokens are type-checked — one cannot be used in
  place of the other.
* `analyst_id` on alert updates is always taken from the authenticated
  identity (`sub` claim); clients cannot set or override it.

### Clarifications / deviations

* `POST /auth/refresh` is authenticated by the **refresh token in the
  request body** (per its request schema) rather than a Bearer header.
* Validation failures return **422** with FastAPI's standard error
  body (the contract's 400 "validation" case is subsumed by 422,
  consistent with all other implemented endpoints).
* Duplicate registration returns **409**; login failures return
  **401** with an identical body for unknown email and wrong password
  (no account enumeration); inactive accounts receive **403**.

### Customer identity enforcement (Step 41)

* `customer_id` in transaction and alert responses is derived
  server-side from the authenticated user's JWT (`sub` → user record
  → `customer_id`).  The client cannot supply or override it.
* `TransactionCreate` has no `customer_id` field — it is not part of
  the request contract.
* The backend injects the authenticated `customer_id` into the ML
  service payload so customer-specific historical features use the
  correct identity.
* Alerts created from HOLD decisions carry the authenticated
  customer's `customer_id`.
* `PATCH /transactions/outcome` remains analyst/admin only; the
  endpoint accepts `customer_id` in the request body (analysts label
  any customer's transactions).

### ML service production hardening (Step 42)

**Health / readiness endpoints** (ML service on port 8001):

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `GET /live` | Liveness probe — process alive | `{"status": "alive"}` — always 200 |
| `GET /ready` | Readiness probe — model loaded | 200 `{"status": "ready", ...}` or 503 `{"status": "not_ready"}` |
| `GET /health` | Legacy health check | `{"status": "ready"|"model_unavailable", ...}` |

**ML failure behaviour:**

| Condition | HTTP status | Detail message |
|-----------|-------------|----------------|
| Model not loaded | 503 | Safe message (no paths/secrets) |
| Invalid transaction data | 422 | Controlled validation message |
| Internal prediction failure | 500 | "Prediction processing failed." |
| Feature engineering error | 422 | Controlled message |

**Input validation hardening (Step 42):**

* `amount`: `(0, 10,000,000]` — rejects zero, negative, and excessive values.
* `currency`: `^[A-Z]{3}$` — ISO 4217 uppercase 3-letter code.
* `ProductCD`: `^[WXYZS]$` — valid product codes only.
* `id_19`, `id_20`: max 100 chars. `DeviceType`: max 50 chars.
* All error responses are JSON with `detail` field — no stack traces or internal paths.

**Timeout configuration:**

* Backend → ML service timeout: `ML_REQUEST_TIMEOUT_SECONDS` (default 5s).
* Connection errors, timeouts, and HTTP errors are mapped to
  `MLServiceUnavailableError`, `MLServiceTimeoutError`, or
  `MLServiceResponseError` and surfaced as **503** to API clients.

**Concurrency safety:**

* Model loaded once at startup; read-only during prediction.
* SHAP explainer initialisation is thread-safe (double-checked lock).
* History store uses `threading.Lock` for thread-safe concurrent access.
* Global exception handler catches unhandled errors without leaking internals.

### ML monitoring and observability (Step 43)

**Monitoring endpoint** (ML service on port 8001):

| Endpoint | Purpose | Auth |
|----------|---------|------|
| `GET /metrics` | Aggregate prediction metrics snapshot | Internal (behind backend trust boundary) |

**Metrics response** (`GET /metrics`):

```json
{
  "total_requests": 42,
  "successful_predictions": 40,
  "failed_predictions": 2,
  "error_rate": 0.0476,
  "fraud_count": 5,
  "non_fraud_count": 35,
  "slow_predictions": 0,
  "decisions": {"APPROVE": 30, "VERIFY": 7, "HOLD": 3},
  "risk_levels": {"LOW": 28, "MEDIUM": 8, "HIGH": 4},
  "errors": {"validation": 1, "feature_engineering": 1, ...},
  "model_version": "fraud-xgb-v1.0.0",
  "latency": {
    "count": 40, "mean_seconds": 0.085,
    "p50_seconds": 0.072, "p95_seconds": 0.198, "p99_seconds": 0.310
  },
  "drift": {"baseline_configured": false, "message": "No baseline configured."},
  "config": {"latency_warn_seconds": 5.0, ...}
}
```

**What is tracked:**

* Request counts (total, success, failure)
* Prediction latency (mean, p50, p95, p99, min, max)
* Error distribution by bounded category
* Fraud/non-fraud counts
* Decision and risk-level distribution
* Model version
* Drift signals (when baseline is configured)

**What is NOT tracked:**

* Raw transaction payloads, customer IDs, merchant names
* Passwords, JWT tokens, authorization headers, filesystem paths

**Drift monitoring:**

* Optional baseline via `ML_BASELINE_*` environment variables
* Mean/std comparison with configurable threshold (`ML_DRIFT_STD_MULTIPLIER`)
* Drift is purely observational — never changes predictions or decisions
* Without baseline, drift monitoring reports "not configured"

**Process-local limitation:**

Metrics are process-local (in-memory). Multi-process deployments would
report per-process metrics. Global aggregation requires an external
metrics collector (e.g., Prometheus).

### Configuration

See `.env.example`. Key variables: `BACKEND_SECRET_KEY` (JWT signing
secret — **must** be overridden with a strong random value outside
local development), `BACKEND_ACCESS_TOKEN_EXPIRE_MINUTES`,
`JWT_ALGORITHM`, `JWT_REFRESH_TOKEN_EXPIRE_DAYS`, `USER_DB_PATH`,
`ALERT_DB_PATH`.

Internal roles (`fraud_analyst`, `admin`) are provisioned for
local development with:

```
python -m backend.db.seed_users
```

(seeds `analyst@example.com` / `admin@example.com` with generated or
`SEED_*`-environment-supplied passwords; idempotent).

---

## Implementation Notes — Production Decision Pipeline (Step 44)

### Idempotent Transaction Processing

`POST /api/v1/transactions` supports an optional `Idempotency-Key`
request header to prevent duplicate transaction submissions.

**Idempotency scope:**

* Scoped to the authenticated customer (server-derived `customer_id`)
* Different customers using the same key create independent transactions
* Same customer using different keys creates independent transactions

**Behavior:**

| Scenario | Behavior |
|---|---|
| No idempotency key | Transaction proceeds normally (no deduplication) |
| New key | Transaction proceeds; result cached in idempotency store |
| Same key (completed) | Returns cached result with HTTP 200 (`idempotent: true`) |
| Same key (processing) | Returns HTTP 409 Conflict |
| Same key (previous failure) | Retries the ML call |

**Key validation:**

* Max 255 characters
* Whitespace-only keys rejected (422)
* Control characters rejected (422)

**Response additions:**

| Field | Type | Description |
|---|---|---|
| `transaction_id` | string (UUID) | Server-generated transaction identifier |
| `idempotent` | boolean | `true` when response was replayed from cache |
| `ml_failure` | boolean | `true` when ML service was unavailable |

### ML Failure Policy

When the ML service is unavailable or times out:

* HTTP 503 (or 502 for generic errors) is returned
* No fraud predictions are fabricated
* No SHAP explanations are fabricated
* No misleading model-version data is included
* Idempotency records are marked "failed" (allowing future retry)
* No alerts are created for failed transactions

### Decision Consistency

The response and persisted alert always represent the SAME decision:

* `risk_score`, `risk_level`, `decision`, `model_version` match
  between API response and alert record
* Duplicate requests via idempotency key return the identical cached
  response
* No duplicate alerts are created for the same transaction

### Retry Policy

* No automatic retry loops around the ML service
* Clients may retry with the same idempotency key after ML failure
* Idempotency records marked "failed" allow fresh ML attempts

### Known Limitations

* Idempotency records are process-local (in-memory) when using the
  SQLite persistence backend; PostgreSQL mode uses database-backed
  idempotency with race-condition protection via UNIQUE constraints
* Idempotency records have no automatic TTL (manual cleanup required)

---

## Audit Trail Endpoints (Step 45)

### GET /api/v1/audit/transactions/{transaction_id}

Retrieve the complete fraud decision audit trail for a transaction.

**Authentication:** Required (Bearer token).

**Authorization:**
- `fraud_analyst` / `admin`: full access to any transaction's audit trail.
- `customer`: may only access their own audit trail (customer_id derived from JWT).

**Response** `200 OK`:

```json
{
  "transaction_id": "uuid",
  "events": [
    {
      "audit_id": "uuid",
      "transaction_id": "uuid",
      "customer_id": "uuid",
      "event_type": "DECISION_MADE",
      "decision": "HOLD",
      "risk_score": 75,
      "risk_level": "HIGH",
      "fraud_probability": 0.85,
      "model_version": "xgb-v2.1.0",
      "explanation_summary": {...},
      "rule_signal_summary": {...},
      "failure_category": null,
      "actor_id": null,
      "actor_role": null,
      "previous_state": null,
      "new_state": null,
      "alert_id": null,
      "created_at": "2026-01-01T00:00:00+00:00"
    }
  ]
}
```

**Event types:**
| `event_type` | Description |
|---|---|
| `DECISION_MADE` | ML prediction completed successfully |
| `ML_FAILURE` | ML service was unavailable or errored |
| `ALERT_CREATED` | Fraud alert created for HOLD decision |
| `ALERT_STATE_CHANGED` | Analyst changed alert status |
| `OUTCOME_RECORDED` | Fraud outcome feedback recorded |

**Error responses:**
| Status | Condition |
|---|---|
| `401` | Missing or invalid authentication |
| `403` | Customer accessing another customer's audit trail |
| `404` | No audit events found (analyst/admin only) |

**Security properties:**
- Customer isolation enforced from JWT, not request body
- No secrets, passwords, JWTs, or raw transaction payloads in response
- Bounded explanation summaries (max 5 factors, max 200 char strings)
- Append-only: no PUT/PATCH/DELETE endpoints exist
- Idempotent replays do not duplicate DECISION_MADE audit events

### Known Limitations (Audit)

* Audit events are append-only; no mechanism to delete or modify old events
* In-memory audit store is used for SQLite mode (volatile)
* Outcome feedback uses a deterministic placeholder transaction_id
* No automatic TTL or retention policy on audit records
