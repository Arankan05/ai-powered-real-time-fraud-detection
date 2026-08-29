# Database Design

## Overview

PostgreSQL is the single source of truth for all persistent data. The schema is managed via Alembic migrations.

No real banking or customer data will ever be stored. All data will be synthetic or derived from legitimate public datasets.

## Entity Relationship Diagram (Planned)

```
┌──────────────┐       ┌───────────────────┐       ┌──────────────┐
│  customers   │       │   transactions    │       │    alerts    │
│──────────────│       │───────────────────│       │──────────────│
│ id (PK)      │──┐    │ id (PK)           │──┐    │ id (PK)      │
│ email        │  └────│ customer_id (FK)  │  └────│ transaction_ │
│ password_hash│       │ amount            │       │ id (FK)      │
│ first_name   │       │ currency          │       │ risk_score   │
│ last_name    │       │ merchant_name     │       │ risk_level   │
│ phone        │       │ merchant_category │       │ decision     │
│ date_of_birth│       │ transaction_type  │       │ explanation  │
│ address      │       │ location_country  │       │ analyst_id   │
│ created_at   │       │ location_city     │       │ (FK, null)   │
│ updated_at   │       │ device_fingerprint│       │ status       │
│ is_active    │       │ device_type       │       │ created_at   │
└──────────────┘       │ ip_address        │       │ resolved_at  │
                       │ timestamp         │       └──────────────┘
                       │ status            │
                       │ risk_score        │
                       │ risk_level        │
                       │ decision          │
                       │ ml_score          │
                       │ behaviour_score   │
                       │ rule_score        │
                       │ explanation_json  │
                       │ created_at        │
                       └───────────────────┘

┌───────────────────┐       ┌───────────────────┐
│   audit_logs      │       │   model_metadata   │
│───────────────────│       │───────────────────│
│ id (PK)           │       │ id (PK)            │
│ actor_id (FK,null)│       │ model_name         │
│ action            │       │ model_version      │
│ resource_type     │       │ framework          │
│ resource_id       │       │ training_metrics   │
│ details_json      │       │ feature_list       │
│ ip_address        │       │ created_at         │
│ timestamp         │       │ trained_by         │
└───────────────────┘       └───────────────────┘
```

## Table Descriptions

### `customers`

Stores customer profile information used for identity verification and behavioural baselines.

- `password_hash` — bcrypt-hashed password; plaintext is never stored.
- `created_at` / `updated_at` — automatic timestamps for audit.

### `transactions`

Every transaction submitted through the simulated banking app.

- `amount` — decimal, validated on the backend.
- `merchant_category` — MCC or category label used for rule evaluation.
- `device_fingerprint` — hashed device identifier for new-device detection.
- `ml_score`, `behaviour_score`, `rule_score` — individual component scores from the fraud pipeline.
- `risk_score` — aggregated 0–100 score.
- `risk_level` — LOW / MEDIUM / HIGH.
- `decision` — APPROVE / VERIFY / HOLD.
- `explanation_json` — JSONB field with explainability output (top contributing features, SHAP values).
- `status` — lifecycle state (PENDING, PROCESSED, FLAGGED, REVIEWED, etc.).

### `alerts`

Created automatically when a transaction receives a HIGH risk decision. Can be assigned to and resolved by a fraud analyst.

- `analyst_id` — nullable FK to `customers` (the reviewing analyst).
- `status` — OPEN, IN_REVIEW, RESOLVED, DISMISSED.

### `audit_logs`

Immutable record of all state-changing operations for compliance and debugging.

- `actor_id` — nullable to allow system-level actions.
- `action` — e.g., TRANSACTION_CREATED, ALERT_RESOLVED, LOGIN_SUCCESSFUL.
- `details_json` — JSONB for flexible payload.

### `model_metadata`

Tracks trained ML models for reproducibility and versioning.

- `training_metrics` — JSONB containing accuracy, precision, recall, F1, AUC-ROC, etc.
- `feature_list` — JSONB array of feature names used during training.

## Indexing Strategy (Planned)

- `transactions.customer_id` — foreign key index.
- `transactions.timestamp` — time-range queries for velocity checks.
- `transactions.status` — filtering by transaction state.
- `alerts.status` — open alert queues.
- `audit_logs.timestamp` — time-range audit queries.
- `audit_logs.actor_id` — per-user audit trails.

## Migration Strategy

- Alembic is the only mechanism for schema changes.
- Migrations live in `database/alembic/versions/`.
- Each migration must be reversible (up and down).
- Migrations are reviewed in pull requests like application code.

## Status

This design is agreed upon but **not yet implemented**. Tables will be created via Alembic migrations during the implementation phase.
