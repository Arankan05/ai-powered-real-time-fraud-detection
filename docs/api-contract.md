# API Contract

## Overview

The backend exposes a RESTful JSON API. This document defines the agreed endpoints, request/response shapes, and status codes that the frontend and backend teams will implement against.

**Base URL:** `/api/v1`

All endpoints require a valid JWT bearer token unless marked as public.

## Authentication

### `POST /auth/register`

**Public.** Create a new customer account.

**Request:**
```json
{
  "email": "string",
  "password": "string",
  "first_name": "string",
  "last_name": "string",
  "phone": "string",
  "date_of_birth": "YYYY-MM-DD",
  "address": "string"
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "email": "string",
  "first_name": "string",
  "last_name": "string"
}
```

### `POST /auth/login`

**Public.** Authenticate and receive a JWT token.

**Request:**
```json
{
  "email": "string",
  "password": "string"
}
```

**Response (200):**
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "expires_in": 1800
}
```

---

## Transactions

### `POST /transactions`

**Auth required (customer).** Submit a new transaction for fraud evaluation.

**Request:**
```json
{
  "amount": 1500.00,
  "currency": "USD",
  "merchant_name": "string",
  "merchant_category": "string",
  "transaction_type": "purchase | transfer | withdrawal",
  "location_country": "string",
  "location_city": "string",
  "device_fingerprint": "string",
  "device_type": "mobile | desktop | pos",
  "ip_address": "string"
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "amount": 1500.00,
  "currency": "USD",
  "merchant_name": "string",
  "transaction_type": "purchase",
  "timestamp": "ISO-8601",
  "risk_score": 45,
  "risk_level": "MEDIUM",
  "decision": "VERIFY",
  "explanation": {
    "top_factors": [
      { "feature": "amount_deviation", "contribution": 0.35 },
      { "feature": "new_device", "contribution": 0.22 }
    ]
  },
  "status": "PROCESSED"
}
```

### `GET /transactions`

**Auth required.** List transactions. Customers see only their own. Analysts see all.

**Query parameters:**
- `page` (int, default 1)
- `per_page` (int, default 20, max 100)
- `status` (optional filter)
- `risk_level` (optional filter)
- `from_date` / `to_date` (ISO-8601 date range)

**Response (200):**
```json
{
  "items": [ /* transaction objects */ ],
  "total": 150,
  "page": 1,
  "per_page": 20
}
```

### `GET /transactions/{id}`

**Auth required.** Get a single transaction with full fraud analysis details.

**Response (200):** Full transaction object including `ml_score`, `behaviour_score`, `rule_score`, and `explanation`.

---

## Alerts

### `GET /alerts`

**Auth required (analyst/admin).** List fraud alerts.

**Query parameters:**
- `status` — OPEN, IN_REVIEW, RESOLVED, DISMISSED
- `risk_level` — HIGH (default filter)
- `page`, `per_page`

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
      "created_at": "ISO-8601",
      "transaction_summary": {
        "amount": 15000.00,
        "merchant_name": "string",
        "customer_email": "string"
      }
    }
  ],
  "total": 12,
  "page": 1,
  "per_page": 20
}
```

### `PATCH /alerts/{id}`

**Auth required (analyst/admin).** Update alert status (assign, resolve, dismiss).

**Request:**
```json
{
  "status": "IN_REVIEW | RESOLVED | DISMISSED",
  "notes": "string (optional)"
}
```

**Response (200):** Updated alert object.

---

## Customers

### `GET /customers/{id}`

**Auth required.** Get customer profile. Customers can view their own; analysts/admins can view any.

**Response (200):**
```json
{
  "id": "uuid",
  "email": "string",
  "first_name": "string",
  "last_name": "string",
  "phone": "string",
  "address": "string",
  "created_at": "ISO-8601",
  "is_active": true
}
```

### `GET /customers/{id}/transactions`

**Auth required.** Get transaction history for a specific customer (used for behavioural baseline and dashboard).

**Response (200):** Paginated list of transactions.

---

## Analytics

### `GET /analytics/dashboard`

**Auth required (analyst/admin).** Aggregated metrics for the fraud analyst dashboard.

**Response (200):**
```json
{
  "total_transactions": 15420,
  "flagged_transactions": 342,
  "alerts_open": 28,
  "alerts_resolved": 156,
  "risk_distribution": {
    "LOW": 12500,
    "MEDIUM": 2400,
    "HIGH": 520
  },
  "top_risk_factors": [
    { "factor": "amount_deviation", "count": 180 },
    { "factor": "new_device", "count": 95 }
  ],
  "transactions_over_time": [
    { "date": "2026-08-01", "total": 500, "flagged": 12 }
  ]
}
```

---

## Standard Error Response

All error responses follow this shape:

```json
{
  "detail": "Human-readable error message",
  "error_code": "ERROR_CODE_CONSTANT"
}
```

### Common HTTP Status Codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 201 | Created |
| 400 | Bad request / validation error |
| 401 | Unauthenticated |
| 403 | Unauthorised (insufficient role) |
| 404 | Resource not found |
| 422 | Unprocessable entity (Pydantic validation) |
| 500 | Internal server error |

## Versioning

The API is versioned via URL path (`/api/v1`). Breaking changes require a new version.

## Status

This contract is agreed upon but **not yet implemented**. Endpoint shapes may be refined during implementation as long as both frontend and backend teams agree on changes.
