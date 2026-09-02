# Database Design

## Overview

PostgreSQL is the single source of truth. Schema is managed exclusively via Alembic migrations.

No real banking or customer data is ever stored. All data is synthetic or derived from legitimate public datasets.

### Current persistence phase (PostgreSQL + SQLite fallback)

Step 40 implemented the PostgreSQL backend for `users` and `alerts`.
The backend defaults to `PERSISTENCE_BACKEND=postgres`; set
`PERSISTENCE_BACKEND=sqlite` in `.env` to use the legacy SQLite path
(`USER_DB_PATH`, `ALERT_DB_PATH` — see `.env.example`).  Both backends
implement the same `UserRepository` / `AlertRepository` Protocol
interfaces (`backend/db/user_repository.py`,
`backend/db/alert_repository.py`), so routers are unchanged.

PostgreSQL schema is created idempotently at startup
(`backend/db/postgres.py :: init_schema`) — no Alembic migrations yet.
Notable differences from the SQLite path:

* `users.email` uniqueness enforced via functional index
  `uq_users_email_ci ON users (lower(email))` (replaces SQLite
  `COLLATE NOCASE`).
* `alerts.risk_factors` and `alerts.explanation_json` stored as native
  JSONB (SQLite stores JSON-encoded TEXT).
* `alerts.transaction_id` has a UNIQUE index as a secondary dedup guard
  (router pre-check remains primary).
* `users.address` is stored on the user record (the `customers` table
  does not exist yet).
* `updated_at` is not tracked on `users`.
* ML historical features (`ml/features/history.py`) remain on SQLite —
  this boundary is intentional and documented in `backend/db/__init__.py`.

## Entity Relationship Diagram

```
┌──────────────┐       ┌───────────────────┐       ┌──────────────┐
│    users     │       │   transactions    │       │    alerts    │
│──────────────│       │───────────────────│       │──────────────│
│ id (PK)      │──┐    │ id (PK)           │──┐    │ id (PK)      │
│ email (UQ)   │  │    │ customer_id (FK)  │  └────│ transaction_ │
│ password_hash│  │    │ merchant_id (FK)  │       │ id (FK)      │
│ role         │  ├───▶│ amount            │       │ risk_score   │
│ first_name   │  │    │ currency          │       │ risk_level   │
│ last_name    │  │    │ transaction_type  │       │ decision     │
│ phone        │  │    │ location_country  │       │ explanation_ │
│ date_of_birth│  │    │ location_city     │       │ json         │
│ is_active    │  │    │ device_fingerprint│       │ analyst_id   │
│ customer_id  │  │    │ device_type       │       │ (FK→users)   │
│ (FK→customers│  │    │ ip_address        │       │ status       │
│  nullable)   │  │    │ timestamp         │       │ notes        │
│ created_at   │  │    │ risk_score        │       │ created_at   │
│ updated_at   │  └────│ risk_level        │       │ resolved_at  │
└──────────────┘       │ decision          │       └──────────────┘
                       │ ml_score          │
┌──────────────┐       │ behaviour_score   │       ┌──────────────┐
│  customers   │       │ rule_score        │       │   audit_logs │
│──────────────│       │ explanation_json  │       │──────────────│
│ id (PK)      │◀──┐   │ model_version(FK) │       │ id (PK)      │
│ first_name   │   │   │ created_at        │       │ actor_id     │
│ last_name    │   │   └───────────────────┘       │ (FK→users)   │
│ phone        │   │                               │ action       │
│ address      │   │   ┌───────────────────┐       │ resource_type│
│ date_of_birth│   │   │    merchants      │       │ resource_id  │
│ created_at   │   │   │───────────────────│       │ details_json │
│ updated_at   │   │   │ id (PK)           │       │ ip_address   │
│ is_active    │   │   │ name              │       │ timestamp    │
└──────────────┘   │   │ category_code     │       └──────────────┘
                   │   │ risk_level        │
                   │   │ created_at        │       ┌──────────────┐
                   │   │ updated_at        │       │model_metadata│
                   │   └───────────────────┘       │──────────────│
┌──────────────────┐                                │ id (PK)      │
│ customer_devices │                                │ model_name   │
│──────────────────│                                │ model_version│
│ id (PK)          │                                │ framework    │
│ customer_id (FK) │                                │ artifact_path│
│ device_fingerprint│                               │ training_    │
│ device_type      │                                │ metrics      │
│ first_seen       │                                │ feature_list │
│ last_seen        │                                │ created_at   │
│ is_active        │                                │ trained_by   │
└──────────────────┘                                └──────────────┘

┌──────────────────────┐
│  risk_rules_config   │
│──────────────────────│
│ id (PK)              │
│ rule_name (UQ)       │
│ description          │
│ is_enabled           │
│ score_contribution   │
│ parameters           │
│ created_at           │
│ updated_at           │
└──────────────────────┘
```

## Table Specifications

### `users`

Authentication and authorisation table. Every person who can log into the system has a record here.

| Column | Type | Nullable | Default | PK | FK | Unique | Check |
|---|---|---|---|---|---|---|---|
| `id` | UUID | NO | gen_random_uuid() | **PK** | | | |
| `email` | VARCHAR(255) | NO | | | | **UNIQUE** | |
| `password_hash` | VARCHAR(255) | NO | | | | | |
| `role` | VARCHAR(20) | NO | `'customer'` | | | | IN ('customer','fraud_analyst','admin') |
| `first_name` | VARCHAR(100) | NO | | | | | |
| `last_name` | VARCHAR(100) | NO | | | | | |
| `phone` | VARCHAR(30) | YES | NULL | | | | |
| `date_of_birth` | DATE | YES | NULL | | | | |
| `is_active` | BOOLEAN | NO | `true` | | | | |
| `customer_id` | UUID | YES | NULL | | **→ customers(id)** | | |
| `created_at` | TIMESTAMPTZ | NO | NOW() | | | | |
| `updated_at` | TIMESTAMPTZ | NO | NOW() | | | | |

**Relationships:**
- `customer_id` → `customers(id)`: Links a user to their customer profile. NULL for analyst/admin users who are not customers.
- One-to-one when `customer_id` is not NULL (a customer has exactly one user account).

**Indexes:**
- `ix_users_email` — UNIQUE on `email`
- `ix_users_role` — on `role` (filter by role)
- `ix_users_customer_id` — on `customer_id` (lookup user by customer)

---

### `customers`

Customer business/profile information. Used for identity, behavioural baselines, and transaction association.

| Column | Type | Nullable | Default | PK | FK | Unique | Check |
|---|---|---|---|---|---|---|---|
| `id` | UUID | NO | gen_random_uuid() | **PK** | | | |
| `first_name` | VARCHAR(100) | NO | | | | | |
| `last_name` | VARCHAR(100) | NO | | | | | |
| `phone` | VARCHAR(30) | YES | NULL | | | | |
| `address` | TEXT | YES | NULL | | | | |
| `date_of_birth` | DATE | YES | NULL | | | | |
| `created_at` | TIMESTAMPTZ | NO | NOW() | | | | |
| `updated_at` | TIMESTAMPTZ | NO | NOW() | | | | |
| `is_active` | BOOLEAN | NO | `true` | | | | |

**Indexes:**
- `ix_customers_created_at` — on `created_at` (time-range queries)

---

### `merchants`

Merchant identity and categorisation. Referenced by transactions.

| Column | Type | Nullable | Default | PK | FK | Unique | Check |
|---|---|---|---|---|---|---|---|
| `id` | UUID | NO | gen_random_uuid() | **PK** | | | |
| `name` | VARCHAR(255) | NO | | | | | |
| `category_code` | VARCHAR(10) | YES | NULL | | | | |
| `risk_level` | VARCHAR(10) | NO | `'LOW'` | | | | IN ('LOW','MEDIUM','HIGH') |
| `created_at` | TIMESTAMPTZ | NO | NOW() | | | | |
| `updated_at` | TIMESTAMPTZ | NO | NOW() | | | | |

**Indexes:**
- `ix_merchants_name` — on `name` (lookup by name)
- `ix_merchants_category_code` — on `category_code` (filter by category)

---

### `transactions`

Every transaction submitted through the system, including fraud analysis results.

| Column | Type | Nullable | Default | PK | FK | Unique | Check |
|---|---|---|---|---|---|---|---|
| `id` | UUID | NO | gen_random_uuid() | **PK** | | | |
| `customer_id` | UUID | NO | | | **→ customers(id)** | | |
| `merchant_id` | UUID | YES | NULL | | **→ merchants(id)** | | |
| `amount` | DECIMAL(12,2) | NO | | | | | > 0 |
| `currency` | VARCHAR(3) | NO | `'USD'` | | | | ISO 4217 |
| `transaction_type` | VARCHAR(20) | NO | | | | | IN ('purchase','transfer','withdrawal') |
| `location_country` | VARCHAR(100) | YES | NULL | | | | |
| `location_city` | VARCHAR(100) | YES | NULL | | | | |
| `device_fingerprint` | VARCHAR(255) | YES | NULL | | | | |
| `device_type` | VARCHAR(20) | YES | NULL | | | | IN ('mobile','desktop','pos') |
| `ip_address` | VARCHAR(45) | YES | NULL | | | | |
| `timestamp` | TIMESTAMPTZ | NO | NOW() | | | | |
| `status` | VARCHAR(20) | NO | `'PENDING'` | | | | IN ('PENDING','COMPLETED','FAILED') |

> **Status semantics:** `status` represents the **transaction lifecycle only** (PENDING → COMPLETED or FAILED). Fraud outcomes are captured separately in `risk_score`, `risk_level`, and `decision`. COMPLETED ≠ APPROVED; FAILED ≠ REJECTED. See `docs/database-erd.md` for the full Transaction Status vs Fraud Analysis Status specification.
| `risk_score` | INTEGER | YES | NULL | | | | 0–100 |
| `risk_level` | VARCHAR(10) | YES | NULL | | | | IN ('LOW','MEDIUM','HIGH') |
| `decision` | VARCHAR(10) | YES | NULL | | | | IN ('APPROVE','VERIFY','HOLD') |
| `ml_score` | INTEGER | YES | NULL | | | | 0–100 |
| `behaviour_score` | INTEGER | YES | NULL | | | | 0–100 |
| `rule_score` | INTEGER | YES | NULL | | | | 0–100 |
| `explanation_json` | JSONB | YES | NULL | | | | |
| `model_version` | VARCHAR(20) | YES | NULL | | **→ model_metadata(model_version)** | | |
| `created_at` | TIMESTAMPTZ | NO | NOW() | | | | |

**Indexes:**
- `ix_transactions_customer_id` — on `customer_id` (customer history lookups, behavioural baseline)
- `ix_transactions_merchant_id` — on `merchant_id` (merchant history for `merchant_is_new` feature)
- `ix_transactions_timestamp` — on `timestamp` (time-range queries, velocity checks)
- `ix_transactions_status` — on `status` (filter by state)
- `ix_transactions_risk_level` — on `risk_level` (filter by risk)
- `ix_transactions_model_version` — on `model_version` (audit: which model scored this transaction)

---

### `alerts`

Created automatically when a transaction receives a HIGH risk decision.

| Column | Type | Nullable | Default | PK | FK | Unique | Check |
|---|---|---|---|---|---|---|---|
| `id` | UUID | NO | gen_random_uuid() | **PK** | | | |
| `transaction_id` | UUID | NO | | | **→ transactions(id)** | | |
| `risk_score` | INTEGER | NO | | | | | 0–100 |
| `risk_level` | VARCHAR(10) | NO | `'HIGH'` | | | | IN ('LOW','MEDIUM','HIGH') |
| `decision` | VARCHAR(10) | NO | `'HOLD'` | | | | IN ('APPROVE','VERIFY','HOLD') |
| `explanation_json` | JSONB | YES | NULL | | | | |
| `analyst_id` | UUID | YES | NULL | | **→ users(id)** | | |
| `status` | VARCHAR(20) | NO | `'OPEN'` | | | | IN ('OPEN','IN_REVIEW','RESOLVED','DISMISSED') |
| `notes` | TEXT | YES | NULL | | | | |
| `created_at` | TIMESTAMPTZ | NO | NOW() | | | | |
| `resolved_at` | TIMESTAMPTZ | YES | NULL | | | | |

**Indexes:**
- `ix_alerts_transaction_id` — on `transaction_id`
- `ix_alerts_status` — on `status` (open alert queues)
- `ix_alerts_analyst_id` — on `analyst_id` (per-analyst workload)

---

### `audit_logs`

Immutable record of all state-changing operations.

| Column | Type | Nullable | Default | PK | FK | Unique | Check |
|---|---|---|---|---|---|---|---|
| `id` | UUID | NO | gen_random_uuid() | **PK** | | | |
| `actor_id` | UUID | YES | NULL | | **→ users(id)** | | |
| `action` | VARCHAR(50) | NO | | | | | |
| `resource_type` | VARCHAR(50) | NO | | | | | |
| `resource_id` | VARCHAR(50) | YES | NULL | | | | |
| `details_json` | JSONB | YES | NULL | | | | |
| `ip_address` | VARCHAR(45) | YES | NULL | | | | |
| `timestamp` | TIMESTAMPTZ | NO | NOW() | | | | |

**Indexes:**
- `ix_audit_logs_actor_id` — on `actor_id` (per-user audit trails)
- `ix_audit_logs_timestamp` — on `timestamp` (time-range queries)
- `ix_audit_logs_action` — on `action` (filter by action type)

---

### `customer_devices`

Known devices associated with a customer. Used by the behaviour engine for new-device detection.

| Column | Type | Nullable | Default | PK | FK | Unique | Check |
|---|---|---|---|---|---|---|---|
| `id` | UUID | NO | gen_random_uuid() | **PK** | | | |
| `customer_id` | UUID | NO | | | **→ customers(id)** | | |
| `device_fingerprint` | VARCHAR(255) | NO | | | | | |
| `device_type` | VARCHAR(20) | YES | NULL | | | | IN ('mobile','desktop','pos') |
| `first_seen` | TIMESTAMPTZ | NO | NOW() | | | | |
| `last_seen` | TIMESTAMPTZ | NO | NOW() | | | | |
| `is_active` | BOOLEAN | NO | `true` | | | | |

**Indexes:**
- `ix_customer_devices_customer_fingerprint` — UNIQUE on (`customer_id`, `device_fingerprint`)
- `ix_customer_devices_customer_id` — on `customer_id`

---

### `model_metadata`

Tracks trained ML models for reproducibility and versioning.

| Column | Type | Nullable | Default | PK | FK | Unique | Check |
|---|---|---|---|---|---|---|---|
| `id` | UUID | NO | gen_random_uuid() | **PK** | | | |
| `model_name` | VARCHAR(100) | NO | | | | | |
| `model_version` | VARCHAR(20) | NO | | | | **UNIQUE** | |
| `framework` | VARCHAR(50) | NO | | | | | |
| `artifact_path` | VARCHAR(500) | NO | | | | | |
| `training_metrics` | JSONB | YES | NULL | | | | |
| `feature_list` | JSONB | YES | NULL | | | | |
| `created_at` | TIMESTAMPTZ | NO | NOW() | | | | |
| `trained_by` | UUID | YES | NULL | | **→ users(id)** | | |

**Indexes:**
- `ix_model_metadata_version` — UNIQUE on `model_version` (supports FK from `transactions.model_version`)

---

### `risk_rules_config`

Persistent configuration for rule-based risk scoring. Rules are evaluated by the ML/Fraud Intelligence service.

| Column | Type | Nullable | Default | PK | FK | Unique | Check |
|---|---|---|---|---|---|---|---|
| `id` | UUID | NO | gen_random_uuid() | **PK** | | | |
| `rule_name` | VARCHAR(100) | NO | | | | **UNIQUE** | |
| `description` | TEXT | YES | NULL | | | | |
| `is_enabled` | BOOLEAN | NO | `true` | | | | |
| `score_contribution` | INTEGER | NO | | | | | 0–100 |
| `parameters` | JSONB | YES | NULL | | | | |
| `created_at` | TIMESTAMPTZ | NO | NOW() | | | | |
| `updated_at` | TIMESTAMPTZ | NO | NOW() | | | | |

**Seed rules (to be created via migration):**

| rule_name | description | score_contribution | parameters |
|---|---|---|---|
| `high_amount` | Amount exceeds configurable threshold | 15 | `{"threshold": 10000}` |
| `impossible_travel` | Two transactions from distant locations within impossible time | 25 | `{"max_speed_kmh": 900}` |
| `velocity_limit` | More than N transactions within T minutes | 20 | `{"max_transactions": 5, "window_minutes": 60}` |
| `new_device_high_amount` | First-seen device with amount above threshold | 15 | `{"amount_threshold": 5000}` |
| `high_risk_merchant` | Transaction with high-risk merchant category | 10 | `{"risk_categories": ["gambling","crypto"]}` |
| `previous_suspicious` | Customer has prior flagged transactions | 10 | `{"lookback_days": 90}` |

**Indexes:**
- `ix_risk_rules_config_rule_name` — UNIQUE on `rule_name`

---

## Terminology Mapping

| Internal (DB) | API Response | Description |
|---|---|---|
| `explanation_json` | `explanation` | JSONB with SHAP values, rule triggers, behavioural deviations |
| `risk_score` | `risk_score` | Aggregated 0–100 score |
| `risk_level` | `risk_level` | LOW / MEDIUM / HIGH |
| `decision` | `decision` | APPROVE / VERIFY / HOLD |
| `ml_score` | `ml_score` | ML model fraud probability (0–100) |
| `behaviour_score` | `behaviour_score` | Behavioural anomaly score (0–100) |
| `rule_score` | `rule_score` | Rule-based cumulative score (0–100) |
| `model_version` | `model_version` | Model version that produced the fraud analysis |

The API exposes `explanation` (camelCase-friendly); the database column is `explanation_json`. The backend maps between them.

## Migration Strategy

- Alembic is the only mechanism for schema changes.
- Migrations live in `database/alembic/versions/`.
- Each migration must be reversible (up and down).
- Migrations are reviewed in pull requests like application code.

## Status

**Step 41 (Customer identity isolation) is complete.** The `users` and `alerts`
tables are implemented in PostgreSQL with idempotent schema init.
Step 41 enforces server-side customer identity: the authenticated user's
`customer_id` is injected into ML payloads and alert records, preventing
client-side impersonation or cross-customer data leakage. Alembic
migrations are not yet in use — schema is managed via
`backend/db/postgres.py :: _SCHEMA_STATEMENTS`. Remaining tables
(`customers`, `transactions`, `merchants`, `audit_logs`,
`customer_devices`, `model_metadata`, `risk_rules_config`) are designed
but not yet implemented.
