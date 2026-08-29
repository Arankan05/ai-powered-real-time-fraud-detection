# Database ERD & Relationship Specification

## Overview

This document is the **visual database relationship contract** for the fraud detection system. It complements `database-design.md` by making every relationship, cardinality, delete/update rule, and data-flow decision explicit and verifiable.

Source of truth for table schemas: `docs/database-design.md`.

## Mermaid Entity Relationship Diagram

```mermaid
erDiagram
    customers ||--o| users : "1:1 user account"
    customers ||--o{ transactions : "1:N"
    customers ||--o{ customer_devices : "1:N"
    merchants ||--o{ transactions : "1:N"
    transactions ||--o| alerts : "1:0..1"
    users ||--o{ alerts : "1:N analyst"
    users ||--o{ audit_logs : "1:N actor"
    users ||--o{ model_metadata : "1:N trained_by"
    model_metadata ||--o{ transactions : "1:N model_version"

    customers {
        UUID id PK
        VARCHAR(100) first_name
        VARCHAR(100) last_name
        VARCHAR(30) phone
        TEXT address
        DATE date_of_birth
        BOOLEAN is_active
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }

    users {
        UUID id PK
        VARCHAR(255) email UK
        VARCHAR(255) password_hash
        VARCHAR(20) role CK
        VARCHAR(100) first_name
        VARCHAR(100) last_name
        VARCHAR(30) phone
        DATE date_of_birth
        BOOLEAN is_active
        UUID customer_id FK "NULL for analyst/admin"
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }

    merchants {
        UUID id PK
        VARCHAR(255) name
        VARCHAR(10) category_code
        VARCHAR(10) risk_level CK
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }

    transactions {
        UUID id PK
        UUID customer_id FK
        UUID merchant_id FK "NULL allowed"
        DECIMAL(12,2) amount CK_GT_0
        VARCHAR(3) currency
        VARCHAR(20) transaction_type CK
        VARCHAR(100) location_country
        VARCHAR(100) location_city
        VARCHAR(255) device_fingerprint
        VARCHAR(20) device_type CK
        VARCHAR(45) ip_address
        TIMESTAMPTZ timestamp
        VARCHAR(20) status CK
        INTEGER risk_score CK_0_100
        VARCHAR(10) risk_level CK
        VARCHAR(10) decision CK
        INTEGER ml_score CK_0_100
        INTEGER behaviour_score CK_0_100
        INTEGER rule_score CK_0_100
        JSONB explanation_json
        VARCHAR(20) model_version FK "NULL allowed"
        TIMESTAMPTZ created_at
    }

    alerts {
        UUID id PK
        UUID transaction_id FK
        INTEGER risk_score CK_0_100
        VARCHAR(10) risk_level CK
        VARCHAR(10) decision CK
        JSONB explanation_json
        UUID analyst_id FK "NULL unassigned"
        VARCHAR(20) status CK
        TEXT notes "NULL"
        TIMESTAMPTZ created_at
        TIMESTAMPTZ resolved_at "NULL"
    }

    audit_logs {
        UUID id PK
        UUID actor_id FK "NULL system"
        VARCHAR(50) action
        VARCHAR(50) resource_type
        VARCHAR(50) resource_id
        JSONB details_json
        VARCHAR(45) ip_address
        TIMESTAMPTZ timestamp
    }

    customer_devices {
        UUID id PK
        UUID customer_id FK
        VARCHAR(255) device_fingerprint
        VARCHAR(20) device_type CK
        TIMESTAMPTZ first_seen
        TIMESTAMPTZ last_seen
        BOOLEAN is_active
    }

    model_metadata {
        UUID id PK
        VARCHAR(100) model_name
        VARCHAR(20) model_version UK
        VARCHAR(50) framework
        VARCHAR(500) artifact_path
        JSONB training_metrics
        JSONB feature_list
        TIMESTAMPTZ created_at
        UUID trained_by FK "NULL allowed"
    }

    risk_rules_config {
        UUID id PK
        VARCHAR(100) rule_name UK
        TEXT description
        BOOLEAN is_enabled
        INTEGER score_contribution CK_0_100
        JSONB parameters
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }
```

**Key:** PK = Primary Key, FK = Foreign Key, UK = Unique, CK = Check constraint.

---

## Complete Relationship Table

| # | Parent Table | Parent Column | Child Table | Child Column | Cardinality | Relationship Label |
|---|---|---|---|---|---|---|
| R1 | `customers` | `id` | `users` | `customer_id` | 1:1 (when not NULL) | User → Customer profile link |
| R2 | `customers` | `id` | `transactions` | `customer_id` | 1:N | Customer's transaction history |
| R3 | `customers` | `id` | `customer_devices` | `customer_id` | 1:N | Customer's known devices |
| R4 | `merchants` | `id` | `transactions` | `merchant_id` | 1:N | Merchant's transactions |
| R5 | `transactions` | `id` | `alerts` | `transaction_id` | 1:0..1 | Alert for flagged transaction |
| R6 | `users` | `id` | `alerts` | `analyst_id` | 1:N | Analyst assigned to alerts |
| R7 | `users` | `id` | `audit_logs` | `actor_id` | 1:N | Actor's audit trail |
| R8 | `users` | `id` | `model_metadata` | `trained_by` | 1:N | User who trained the model |
| R9 | *(standalone)* | — | `risk_rules_config` | — | — | No FK; read by ML service |
| R10 | `model_metadata` | `model_version` | `transactions` | `model_version` | 1:N | Model version that scored the transaction |

**No additional relationships exist.** Every FK in the database is listed above.

---

## Cardinality Detail

### R1: `customers` 1 : 0..1 `users`

- A `customers` record has **at most one** associated `users` record (via `users.customer_id`).
- A `users` record with `customer_id IS NOT NULL` has **exactly one** `customers` record.
- `users` records with `role = 'fraud_analyst'` or `role = 'admin'` have `customer_id = NULL` — they are not customers.
- Registration (`POST /api/v1/auth/register`) creates **both** a `customers` record and a `users` record with the FK linking them.

**Why 1:1 and not 1:N?** A customer should have exactly one system account. The unique index on `users.customer_id` (or application-level enforcement) prevents multiple accounts per customer.

### R2: `customers` 1 : N `transactions`

- One customer can have many transactions.
- `transactions.customer_id` is NOT NULL — every transaction belongs to a customer.

### R3: `customers` 1 : N `customer_devices`

- One customer can have many known devices.
- Unique constraint on `(customer_id, device_fingerprint)` prevents duplicate device entries per customer.

### R4: `merchants` 1 : N `transactions`

- One merchant can appear in many transactions.
- `transactions.merchant_id` is NULL-able — a merchant record may not exist for every transaction (e.g., merchant not yet registered in the system).

### R5: `transactions` 1 : 0..1 `alerts`

- A transaction has **at most one** alert.
- An alert is created **only** when `decision = 'HOLD'` (risk_level = HIGH).
- `alerts.transaction_id` is NOT NULL — every alert references a transaction.

### R6: `users` 1 : N `alerts`

- One user (with `role = 'fraud_analyst'`) can be assigned to many alerts as the investigating analyst.
- `alerts.analyst_id` is NULL when the alert is unassigned (status = `OPEN`).
- Customers are **never** used as analyst identities. Only users with `role = 'fraud_analyst'` or `role = 'admin'` can be assigned.

### R7: `users` 1 : N `audit_logs`

- One user can have many audit log entries.
- `audit_logs.actor_id` is NULL for system-generated actions (e.g., automated system events).

### R8: `users` 1 : N `model_metadata`

- One user (typically an ML developer or admin) can train many models.
- `model_metadata.trained_by` is NULL if the model was imported or trained outside the system.

### R9: `risk_rules_config` (standalone)

- No foreign keys to any other table.
- Read by the ML/Fraud Intelligence Service at runtime.
- Rule names like `previous_suspicious` and `high_risk_merchant` **reference conceptual relationships** (prior transactions, merchant risk levels) but are evaluated via application logic, not SQL foreign keys.

### R10: `model_metadata` 1 : N `transactions`

- One model version can be referenced by many transactions.
- `transactions.model_version` is NULL-able — NULL only if the transaction was never fraud-analysed (e.g., ML service was unavailable and the transaction was not persisted as COMPLETED).
- This relationship enables full audit traceability: *"Which model produced this fraud score?"*
- The FK references `model_metadata.model_version` (UNIQUE). See [Model Version Traceability](#model-version-traceability) for design rationale.

---

## ON DELETE / ON UPDATE Behaviour

**Design principle:** Financial records (transactions, alerts, audit logs) must **never** be destroyed through cascading deletes. All behaviour defaults to `NO ACTION` unless stated otherwise.

| # | FK Column | Parent Table | ON DELETE | ON UPDATE | Rationale |
|---|---|---|---|---|---|
| R1 | `users.customer_id` | `customers` | **SET NULL** | **CASCADE** | If a customer profile is removed, the user account remains but is unlinked. |
| R2 | `transactions.customer_id` | `customers` | **RESTRICT** | **CASCADE** | Transactions must never be orphaned. Prevent customer deletion if transactions exist. |
| R3 | `customer_devices.customer_id` | `customers` | **RESTRICT** | **CASCADE** | Device history is tied to the customer. Prevent customer deletion if device records exist. |
| R4 | `transactions.merchant_id` | `merchants` | **SET NULL** | **CASCADE** | If a merchant is deleted, historical transactions retain all other data but lose the merchant link. |
| R5 | `alerts.transaction_id` | `transactions` | **RESTRICT** | **CASCADE** | Alerts must never be orphaned. Prevent transaction deletion if an alert exists. |
| R6 | `alerts.analyst_id` | `users` | **SET NULL** | **CASCADE** | If an analyst account is removed, the alert becomes unassigned but is preserved. |
| R7 | `audit_logs.actor_id` | `users` | **SET NULL** | **CASCADE** | Audit logs must survive user deletion. The `actor_id` becomes NULL but the log entry remains. |
| R8 | `model_metadata.trained_by` | `users` | **SET NULL** | **CASCADE** | Model metadata is preserved even if the training user is removed. |
| R10 | `transactions.model_version` | `model_metadata` | **RESTRICT** | **CASCADE** | Model version audit trail must be preserved. Prevent deletion of model metadata referenced by transactions. |

### Important Notes on Delete Behaviour

- **No `CASCADE` on DELETE is used for any financial or audit relationship.** This is intentional.
- `RESTRICT` is used where orphaned records would be meaningless (transactions without a customer, alerts without a transaction).
- `SET NULL` is used where the relationship is optional or the record remains valuable without the link (analyst assignment, audit actor, merchant link).
- **Transactions should never be physically deleted.** If a transaction needs to be "removed" from active views, use the `status` field (e.g., set to a terminal state). Physical deletion of transaction rows is not supported by design.
- **Audit logs are immutable.** No UPDATE or DELETE operations should be performed on `audit_logs` rows.

### Architecture Gap Flagged

The existing `database-design.md` **does not specify ON DELETE or ON UPDATE behaviour for any foreign key.** The rules above are the agreed-upon decisions documented here for the first time. These must be implemented in the Alembic migration scripts.

---

## Transaction Data Categorisation

The `transactions` table stores four distinct categories of data. They must not be confused or mixed.

### Category 1: Original Transaction Information

These fields are provided by the client at submission time and are **immutable after creation**.

| Column | Source |
|---|---|
| `customer_id` | Authenticated user's customer profile |
| `merchant_id` | Resolved from `merchant_name` by backend |
| `amount` | Client request |
| `currency` | Client request |
| `transaction_type` | Client request |
| `location_country` | Client request |
| `location_city` | Client request |
| `device_fingerprint` | Client request |
| `device_type` | Client request |
| `ip_address` | Client request |
| `timestamp` | Server-generated at receipt |

### Category 2: Customer Behavioural Features (computed, not persisted)

These are computed by the ML/Fraud Intelligence Service during analysis. They are **not stored in the transactions table**. They exist only during the request/response cycle and may be reconstructed from transaction history.

| Feature | Computed From |
|---|---|
| `amount_deviation` | Current amount vs. historical mean/std from past transactions |
| `tx_velocity_1h/24h/7d` | `COUNT(*)` on transactions within time window |
| `is_new_device` | Current `device_fingerprint` vs. `customer_devices` table |
| `location_is_new` | Current country vs. distinct countries in transaction history |
| `merchant_is_new` | Current `merchant_id` vs. `DISTINCT merchant_id` in customer's past transactions |
| `avg_spend_30d` | `AVG(amount)` on transactions from last 30 days |
| `previous_suspicious_count` | `COUNT(*)` where `decision = 'HOLD'` or `risk_level = 'HIGH'` in last 90 days |

### Category 3: Fraud Analysis Outputs

These are returned by the ML/Fraud Intelligence Service and **persisted in the transactions table**.

| Column | Produced By |
|---|---|
| `ml_score` | ML Prediction model (XGBoost/scikit-learn) |
| `behaviour_score` | Behavioural Anomaly Analysis |
| `rule_score` | Rule-Based Risk Signals |
| `risk_score` | Risk Aggregator (weighted combination of the three scores) |
| `risk_level` | Decision Engine (maps risk_score to LOW/MEDIUM/HIGH) |
| `decision` | Decision Engine (maps risk_score to APPROVE/VERIFY/HOLD) |
| `explanation_json` | Explainability module (SHAP values, behaviour signals, rules triggered) |
| `model_version` | ML/Fraud Intelligence Service (identifies which model was used) |

### Category 4: Final Decision & State

| Column | Purpose |
|---|---|
| `status` | Transaction lifecycle state: `PENDING` → `COMPLETED` or `FAILED` (see [Transaction Status vs Fraud Analysis Status](#transaction-status-vs-fraud-analysis-status)) |
| `risk_level` | Denormalised from Category 3 for fast filtering |
| `decision` | Denormalised from Category 3 for fast filtering |

### Transaction Status vs Fraud Analysis Status

**Transaction status** (`transactions.status`) represents the **transaction lifecycle**, not ML/fraud-service technical health.

```
Transaction lifecycle:

    PENDING  →  COMPLETED   (fraud analysis succeeded; any decision)
             →  FAILED      (ML service unavailable or returned an error)
```

**Status values:**

| Status | Meaning |
|---|---|
| `PENDING` | Transaction received and persisted; fraud analysis not yet complete |
| `COMPLETED` | Fraud analysis completed successfully; scores, decision, and explanation are populated |
| `FAILED` | ML/Fraud Intelligence Service was unavailable or returned an error; no fraud scores persisted |

**What status is NOT:**

```
Transaction status  ≠  Fraud analysis/service status
```

- `COMPLETED` does **not** mean the transaction was approved. A transaction with `decision = 'HOLD'` is still `COMPLETED` — it simply completed with a hold decision.
- `FAILED` does **not** mean the transaction was rejected. It means the fraud analysis could not be performed.
- Fraud outcomes (APPROVE / VERIFY / HOLD) are captured in `decision`, `risk_level`, and `risk_score` — **not** in `status`.

**ML service failure handling:**

When the ML/Fraud Intelligence Service is unavailable or fails:

- Do **not** silently approve the transaction.
- Do **not** represent the ML failure as a normal transaction status (no `ERROR` status).
- Fail safely according to the backend failure policy.
- The transaction remains `PENDING` or is marked `FAILED`.
- Technical failure information is preserved through appropriate logging and error handling.
- Fraud score columns (`ml_score`, `behaviour_score`, `rule_score`, `risk_score`, `risk_level`, `decision`, `explanation_json`, `model_version`) remain **NULL**.

**Fraud-specific states** are always determined from the dedicated columns:

| Query Need | Use |
|---|---|
| Fraud-approved transactions | `decision = 'APPROVE'` |
| Transactions requiring verification | `decision = 'VERIFY'` |
| Flagged / held transactions | `decision = 'HOLD'` or `risk_level = 'HIGH'` |
| Previously suspicious (for behaviour features) | `decision = 'HOLD'` or `risk_level = 'HIGH'` within lookback window |

---

## Fraud Result Storage

All fraud results from the ML/Fraud Intelligence Service are persisted in the **`transactions` table**, except where a separate alert is also created.

| Fraud Data | Persisted In Table | Column | Persisted? |
|---|---|---|---|
| ML prediction score | `transactions` | `ml_score` | Yes |
| Behaviour anomaly score | `transactions` | `behaviour_score` | Yes |
| Rule-based score | `transactions` | `rule_score` | Yes |
| Aggregated risk score | `transactions` | `risk_score` | Yes |
| Risk level | `transactions` | `risk_level` | Yes |
| Decision (APPROVE/VERIFY/HOLD) | `transactions` | `decision` | Yes |
| Explanation (SHAP + signals + rules) | `transactions` | `explanation_json` | Yes |
| Risk factors list | `transactions` | `explanation_json` (inside JSONB) | Yes |
| Model version | `transactions` | `model_version` | **Yes** |

### Alert Duplicate Storage

When `decision = 'HOLD'`, an `alerts` record is also created with copies of key fields:

| Field | `transactions` | `alerts` | Rationale |
|---|---|---|---|
| `risk_score` | Yes | Yes | Alert captures snapshot at creation time |
| `risk_level` | Yes | Yes | Same rationale |
| `decision` | Yes | Yes | Same rationale |
| `explanation_json` | Yes | Yes | Alert preserves its own copy for investigation |

This is **intentional denormalisation**: the alert record must remain independently queryable and historically accurate even if the transaction record is updated.

### Model Version Traceability

Every fraud-evaluated transaction must retain the model version used for that evaluation. This enables future investigations to answer: *"Which model produced this fraud score?"*

**Column:** `transactions.model_version` — `VARCHAR(20) NULL`, FK → `model_metadata.model_version`.

- **NULL** only when the transaction was never fraud-analysed (ML service unavailable, transaction not completed).
- **Non-NULL** for every transaction that passed through the fraud pipeline successfully.
- The ML service returns `model_version` in every prediction response; the backend persists it alongside the fraud scores.
- Model binaries (`.joblib` files) remain on the filesystem at `artifact_path` — **not** stored in PostgreSQL.

**FK design rationale:**

- `model_metadata.model_version` is declared **UNIQUE** to support this FK.
- A single-column FK (`model_version` → `model_version`) is preferred over a composite FK to `(model_name, model_version)` because the ML service response provides only the version string (e.g., `"fraud-xgb-v1.2.0"`).
- Version strings are globally unique across all models in practice.
- **ON DELETE RESTRICT**: model metadata rows referenced by transactions cannot be deleted, preserving the audit trail.
- **ON UPDATE CASCADE**: if a version string is ever corrected, the change propagates.
- The existing composite `UNIQUE(model_name, model_version)` in `database-design.md` is **superseded** by the stronger `UNIQUE(model_version)` constraint documented here.

### DB ↔ API Field Mapping

| DB Column | API Field | Notes |
|---|---|---|
| `transactions.explanation_json` | `explanation` | Backend maps JSONB to API JSON object |
| `alerts.explanation_json` | `explanation` | Same mapping for alert responses |
| `transactions.risk_score` | `risk_score` | Direct pass-through |
| `transactions.risk_level` | `risk_level` | Direct pass-through |
| `transactions.decision` | `decision` | Direct pass-through |
| `transactions.ml_score` | `ml_score` | Direct pass-through |
| `transactions.behaviour_score` | `behaviour_score` | Direct pass-through |
| `transactions.rule_score` | `rule_score` | Direct pass-through |
| `transactions.model_version` | `model_version` | Direct pass-through (also in `/fraud/check` response) |

The only field with a **different name** between DB and API is `explanation_json` → `explanation`. All other fields share the same name.

---

## Customer Behaviour Data Sources

The ML/Fraud Intelligence Service computes behavioural features from existing tables. **No additional tables are needed.**

| Behavioural Feature | Data Source | Query / Method |
|---|---|---|
| Historical transaction behaviour | `transactions` (filtered by `customer_id`) | Aggregate queries: COUNT, AVG, STDDEV on recent transactions |
| Known devices | `customer_devices` (filtered by `customer_id`) | `SELECT device_fingerprint FROM customer_devices WHERE customer_id = ? AND is_active = true` |
| New-device detection | `customer_devices` + current transaction | Device fingerprint not found in customer's known devices → `is_new_device = true` |
| Amount deviation | `transactions` (historical amounts for this customer) | Z-score: `(current_amount - AVG(amount)) / STDDEV(amount)` over customer's last N transactions |
| Transaction velocity | `transactions` (filtered by `customer_id` + time window) | `COUNT(*) WHERE timestamp > NOW() - interval` for 1h, 24h, 7d windows |
| Location deviation | `transactions` (distinct countries for this customer) | Current country not in `SELECT DISTINCT location_country FROM transactions WHERE customer_id = ?` → new location |
| `merchant_is_new` | `transactions` (filtered by `customer_id` + `merchant_id`) | `SELECT COUNT(*) FROM transactions WHERE customer_id = ? AND merchant_id = ?` — if 0, merchant is new for this customer |

### Device Registration

When a transaction is successfully processed, the backend should:

1. Check if `(customer_id, device_fingerprint)` exists in `customer_devices`.
2. If not found: insert a new `customer_devices` row with `first_seen = NOW()`, `last_seen = NOW()`.
3. If found: update `last_seen = NOW()`.

This ensures the known-devices list stays current for future behaviour analysis.

### Why No `customer_locations` Table

Customer location history is derived from `transactions.location_country` and `transactions.location_city`. A separate locations table is not needed because:

- Locations are transaction-scoped, not customer-scoped.
- The `transactions` table already has indexes on `customer_id` and `timestamp` for efficient historical queries.
- Adding a locations table would create a second source of truth for location data.

---

## Merchant Relationship

### `merchants.id` → `transactions.merchant_id`

```
merchants (parent)
    id UUID PK
        ↓  1:N
transactions (child)
    merchant_id UUID FK → merchants(id) NULL
```

- `merchant_id` is NULL-able because a merchant record may not exist for every transaction.
- The backend resolves or creates a merchant record from `merchant_name` and `merchant_category` submitted in the API request.

### `merchant_is_new` Calculation

`merchant_is_new` is a **computed feature** (not a stored column). It is determined by the ML/Fraud Intelligence Service at analysis time:

```
SELECT COUNT(*) FROM transactions
WHERE customer_id = :customer_id
  AND merchant_id = :merchant_id;

-- If COUNT = 0 → merchant_is_new = true
-- If COUNT > 0 → merchant_is_new = false
```

This is per-customer, not global. A merchant may be "new" for one customer but well-known for another.

---

## Model Metadata

### `model_metadata` ↔ `users` Relationship

```
users (parent)
    id UUID PK
        ↓  1:N (trained_by is nullable)
model_metadata (child)
    trained_by UUID FK → users(id) NULL
```

### What Is Stored

| Column | Purpose | Example |
|---|---|---|
| `model_name` | Human-readable model identifier | `"fraud_xgb"` |
| `model_version` | Semantic version string | `"v1.2.0"` |
| `framework` | ML framework used | `"xgboost"`, `"scikit-learn"` |
| `artifact_path` | Filesystem path to serialised model | `"ml/models/fraud_xgb_v1.joblib"` |
| `training_metrics` | JSONB with precision, recall, F1, AUC-ROC | `{"precision": 0.75, "recall": 0.82, "f1": 0.78, "auc_roc": 0.88}` |
| `feature_list` | JSONB listing features used by the model | `["amount", "amount_deviation", "is_new_device", ...]` |
| `created_at` | When the model was registered | Timestamp |
| `trained_by` | User who trained/registered the model | FK to `users(id)`, NULL if imported |

### Model Binaries

Model binaries (`.joblib` files) are **not stored in PostgreSQL**. They are stored on the filesystem at the path indicated by `artifact_path`. The `ML_MODEL_PATH` environment variable configures the base directory.

### Version Selection

- Default: latest model from `model_metadata` (ordered by `created_at DESC`).
- Override: `ML_MODEL_VERSION` environment variable specifies an exact version string.

### Uniqueness

`model_version` is **UNIQUE** (supports the FK from `transactions.model_version`).

> **Note:** `database-design.md` specifies `UNIQUE(model_name, model_version)`. The ERD supersedes this with the stronger `UNIQUE(model_version)` constraint. The composite unique becomes redundant once `model_version` alone is unique.

---

## Risk Rules Configuration

### Table Purpose

`risk_rules_config` stores configurable rule-based risk signals evaluated by the ML/Fraud Intelligence Service. Rules are **not hard-coded** — they are read from this table at runtime.

### Table Structure

| Column | Type | Purpose |
|---|---|---|
| `rule_name` | VARCHAR(100) UNIQUE | Machine-readable rule identifier |
| `description` | TEXT | Human-readable explanation of what the rule detects |
| `is_enabled` | BOOLEAN | Whether the rule is active (`true`) or disabled (`false`) |
| `score_contribution` | INTEGER (0–100) | Points added to `rule_score` when the rule triggers |
| `parameters` | JSONB | Rule-specific configuration (thresholds, windows, categories) |
| `created_at` | TIMESTAMPTZ | When the rule was created |
| `updated_at` | TIMESTAMPTZ | When the rule was last modified |

### Six Seed Rules (created via Alembic migration)

| rule_name | score_contribution | parameters |
|---|---|---|
| `high_amount` | 15 | `{"threshold": 10000}` |
| `impossible_travel` | 25 | `{"max_speed_kmh": 900}` |
| `velocity_limit` | 20 | `{"max_transactions": 5, "window_minutes": 60}` |
| `new_device_high_amount` | 15 | `{"amount_threshold": 5000}` |
| `high_risk_merchant` | 10 | `{"risk_categories": ["gambling", "crypto"]}` |
| `previous_suspicious` | 10 | `{"lookback_days": 90}` |

### Relationships

`risk_rules_config` has **no foreign keys**. It is a standalone configuration table. The ML/Fraud Intelligence Service reads enabled rules at runtime:

```sql
SELECT rule_name, score_contribution, parameters
FROM risk_rules_config
WHERE is_enabled = true;
```

Rules like `previous_suspicious` reference **conceptual** relationships (a customer's prior flagged transactions) but these are evaluated via application logic against the `transactions` table, not via SQL foreign keys.

---

## Audit Logs

### `users` → `audit_logs` Relationship

```
users (parent)
    id UUID PK
        ↓  1:N (actor_id is nullable)
audit_logs (child)
    actor_id UUID FK → users(id) NULL
```

### Historical Integrity

- **ON DELETE SET NULL**: If a user account is deleted or deactivated, the audit log entries are preserved. The `actor_id` becomes NULL but the action, timestamp, and details remain.
- **No physical deletion**: Audit log rows must never be physically deleted. They are append-only.
- **No updates**: Audit log rows must never be updated after insertion.

### `actor_id` NULL Cases

`actor_id` is NULL when:

- The action was performed by the system (e.g., automated alert creation, scheduled tasks).
- The acting user has been removed from the system (via ON DELETE SET NULL).

---

## Analyst / Alert Relationship

### `alerts.analyst_id` → `users.id`

```
users (parent, role = 'fraud_analyst' or 'admin')
    id UUID PK
        ↓  1:N (analyst_id is nullable)
alerts (child)
    analyst_id UUID FK → users(id) NULL
```

### Rules

- **Only users with `role = 'fraud_analyst'` or `role = 'admin'`** should be assigned as analysts. This is enforced at the application level (the PATCH endpoint checks roles before assignment).
- **Customers are never analysts.** A user with `role = 'customer'` has `customer_id IS NOT NULL` and cannot be assigned to alerts.
- `analyst_id` is NULL when the alert is first created (status = `OPEN`).
- When an analyst first updates an alert via `PATCH /api/v1/alerts/{id}`, `analyst_id` is auto-set to the current user if not already assigned.
- `alerts.notes` (TEXT, nullable) stores free-form investigation notes added by the analyst.

---

## Constraints Summary

### Check Constraints

| Table | Column | Constraint |
|---|---|---|
| `users` | `role` | `IN ('customer', 'fraud_analyst', 'admin')` |
| `users` | `email` | UNIQUE |
| `merchants` | `risk_level` | `IN ('LOW', 'MEDIUM', 'HIGH')` |
| `transactions` | `amount` | `> 0` |
| `transactions` | `currency` | ISO 4217 (3-letter code) |
| `transactions` | `transaction_type` | `IN ('purchase', 'transfer', 'withdrawal')` |
| `transactions` | `device_type` | `IN ('mobile', 'desktop', 'pos')` |
| `transactions` | `status` | `IN ('PENDING', 'COMPLETED', 'FAILED')` — see [Transaction Status vs Fraud Analysis Status](#transaction-status-vs-fraud-analysis-status) |
| `transactions` | `risk_level` | `IN ('LOW', 'MEDIUM', 'HIGH')` |
| `transactions` | `decision` | `IN ('APPROVE', 'VERIFY', 'HOLD')` |
| `transactions` | `risk_score` | `>= 0 AND <= 100` |
| `transactions` | `ml_score` | `>= 0 AND <= 100` |
| `transactions` | `behaviour_score` | `>= 0 AND <= 100` |
| `transactions` | `rule_score` | `>= 0 AND <= 100` |
| `alerts` | `risk_level` | `IN ('LOW', 'MEDIUM', 'HIGH')` |
| `alerts` | `decision` | `IN ('APPROVE', 'VERIFY', 'HOLD')` |
| `alerts` | `status` | `IN ('OPEN', 'IN_REVIEW', 'RESOLVED', 'DISMISSED')` |
| `alerts` | `risk_score` | `>= 0 AND <= 100` |
| `customer_devices` | `device_type` | `IN ('mobile', 'desktop', 'pos')` |
| `customer_devices` | `(customer_id, device_fingerprint)` | UNIQUE composite |
| `risk_rules_config` | `rule_name` | UNIQUE |
| `risk_rules_config` | `score_contribution` | `>= 0 AND <= 100` |
| `model_metadata` | `(model_name, model_version)` | UNIQUE composite (superseded by `model_version` UNIQUE; see below) |
| `model_metadata` | `model_version` | UNIQUE (supports FK from `transactions.model_version`) |

### Financial Data Precision

| Column | Type | Rationale |
|---|---|---|
| `transactions.amount` | `DECIMAL(12,2)` | 10 integer digits + 2 decimal places. Max value: 9,999,999,999.99. Avoids floating-point rounding errors. |
| `transactions.currency` | `VARCHAR(3)` | ISO 4217 3-letter code (e.g., USD, EUR, GBP). |

No silent changes from the existing `database-design.md` contract.

---

## Index Strategy

Every index must have a practical query reason. No speculative indexes.

### Primary Key Indexes (automatic)

All nine tables have an automatic B-tree index on `id` (UUID PK).

### Unique Indexes

| Index Name | Table | Columns | Purpose |
|---|---|---|---|
| `ix_users_email` | `users` | `email` | Login lookup, unique email enforcement |
| `ix_customer_devices_customer_fingerprint` | `customer_devices` | `(customer_id, device_fingerprint)` | Prevent duplicate device records, fast device lookup |
| `ix_model_metadata_name_version` | `model_metadata` | `(model_name, model_version)` | Prevent duplicate model versions |
| `ix_risk_rules_config_rule_name` | `risk_rules_config` | `rule_name` | Prevent duplicate rule names, fast rule lookup |

### Query-Driven Indexes

| Index Name | Table | Column(s) | Query Reason |
|---|---|---|---|
| `ix_users_role` | `users` | `role` | Filter users by role (RBAC queries) |
| `ix_users_customer_id` | `users` | `customer_id` | Lookup user by customer profile |
| `ix_customers_created_at` | `customers` | `created_at` | Time-range queries on customer registrations |
| `ix_transactions_customer_id` | `transactions` | `customer_id` | Customer transaction history, behavioural baseline queries |
| `ix_transactions_merchant_id` | `transactions` | `merchant_id` | Merchant history for `merchant_is_new` feature |
| `ix_transactions_timestamp` | `transactions` | `timestamp` | Time-range queries, velocity calculations, dashboard charts |
| `ix_transactions_status` | `transactions` | `status` | Filter by transaction state (e.g., flagged transactions) |
| `ix_transactions_risk_level` | `transactions` | `risk_level` | Filter by risk level for analytics and alerting |
| `ix_transactions_model_version` | `transactions` | `model_version` | Audit queries: "which transactions were scored by model version X?" |
| `ix_alerts_transaction_id` | `alerts` | `transaction_id` | Lookup alert by transaction |
| `ix_alerts_status` | `alerts` | `status` | Open alert queue (`WHERE status = 'OPEN'`) |
| `ix_alerts_analyst_id` | `alerts` | `analyst_id` | Per-analyst workload queries |
| `ix_audit_logs_actor_id` | `audit_logs` | `actor_id` | Per-user audit trail |
| `ix_audit_logs_timestamp` | `audit_logs` | `timestamp` | Time-range audit queries |
| `ix_audit_logs_action` | `audit_logs` | `action` | Filter by action type |
| `ix_customer_devices_customer_id` | `customer_devices` | `customer_id` | List all devices for a customer |
| `ix_merchants_name` | `merchants` | `name` | Lookup merchant by name |
| `ix_merchants_category_code` | `merchants` | `category_code` | Filter merchants by category |

### Composite Index Recommendations

| Table | Columns | Rationale |
|---|---|---|
| `transactions` | `(customer_id, timestamp)` | Combined customer history + time-range. Covers velocity and behavioural baseline queries better than single-column indexes. |

This composite index should be evaluated during implementation. If `ix_transactions_customer_id` (single column) proves sufficient, the composite can be deferred.

---

## Database Normalisation Findings

### Intentional Denormalisation (acceptable)

1. **`alerts` duplicates `risk_score`, `risk_level`, `decision`, `explanation_json` from `transactions`.**
   - Rationale: Alert records must be independently queryable and historically accurate. An alert captures the fraud state at creation time. If the transaction is later re-analysed, the alert's original snapshot is preserved.
   - Risk: If the transaction's fraud data is updated without updating the alert, inconsistency occurs. Mitigation: the application should not allow updating fraud scores on a transaction that has an associated alert without also updating the alert.

2. **`users` stores `first_name`, `last_name`, `phone`, `date_of_birth` that also exist in `customers`.**
   - Rationale: Authentication requires these fields without joining to `customers`. Admin/analyst users may not have a `customers` record at all. The `customers` table holds the canonical business profile; `users` holds auth-relevant copies.
   - Risk: Name changes must be propagated to both tables. Mitigation: the registration flow writes to both atomically; profile updates should update both.

### No Normalisation Violations Found

- No duplicated foreign keys.
- No fields that should be FKs but aren't.
- No fields referencing the wrong table (all `analyst_id` and `actor_id` correctly reference `users`, not `customers`).
- No unnecessary junction tables (all relationships are 1:N or 1:1).
- No over-engineered tables (no separate locations, categories, or config tables beyond what the architecture requires).

---

## Cross-Document Validation

### Checked Items

| Item | database-design.md | architecture.md | api-contract.md | ml-architecture.md | Result |
|---|---|---|---|---|---|
| Table names (9) | All listed | All listed in Data Layer | — | `risk_rules_config`, `model_metadata` referenced | CONSISTENT |
| `users` table fields | All specified | `role` mentioned | `email`, `role`, `customer_id` in responses | — | CONSISTENT |
| `customers` table fields | All specified | "separate from users" | `GET /customers/me` fields match | — | CONSISTENT |
| `transactions.merchant_id` | FK to merchants | — | `merchant_id` in response | `merchant_id` in request | CONSISTENT |
| `transactions` score fields | `ml_score`, `behaviour_score`, `rule_score`, `risk_score` | — | All four in response | All four in ML response | CONSISTENT |
| `transactions.model_version` | **Not yet present** | — | In `/fraud/check` response | In ML response schema | **ERD ADDITION** (see below) |
| `explanation_json` naming | Column name in DB | "explanation_json → API explanation" | `explanation` in API | `explanation` in ML response | CONSISTENT |
| `risk_level` enum | `LOW`, `MEDIUM`, `HIGH` | Same in decision table | Same in validation rules | Same in response schema | CONSISTENT |
| `decision` enum | `APPROVE`, `VERIFY`, `HOLD` | Same in decision table | Same in validation rules | Same in response schema | CONSISTENT |
| `transaction status` enum | `PENDING`, `PROCESSED`, `FLAGGED`, `REVIEWED` | — | `PENDING`, `PROCESSED`, `FLAGGED`, `REVIEWED` | — | **CONTRADICTION** (see below) |
| `alert status` enum | `OPEN`, `IN_REVIEW`, `RESOLVED`, `DISMISSED` | — | Same in PATCH transitions | — | CONSISTENT |
| `alerts.analyst_id` | FK → users(id) | — | Analyst assignment in PATCH | — | CONSISTENT |
| `alerts.notes` | TEXT nullable | — | In PATCH request/response | — | CONSISTENT |
| `users.role` values | `customer`, `fraud_analyst`, `admin` | Same in RBAC section | Same in auth/register | — | CONSISTENT |
| `model_version` (metadata) | In `model_metadata` table | — | In `/fraud/check` response | In ML response schema | CONSISTENT |
| `model_version` (uniqueness) | `UNIQUE(model_name, model_version)` | — | — | — | **ERD SUPERSED** (see below) |
| `amount` precision | `DECIMAL(12,2)` | — | "max 2 decimal places, max 9999999999.99" | — | CONSISTENT |
| `transaction_type` enum | `purchase`, `transfer`, `withdrawal` | — | Same in validation rules | — | CONSISTENT |
| `device_type` enum | `mobile`, `desktop`, `pos` | — | Same in validation rules | — | CONSISTENT |
| `risk_rules_config` rules | 6 seed rules listed | — | — | 6 planned rules listed | CONSISTENT |
| Risk Aggregator ownership | — | "ML service computes all" | — | Diagram labels as "Backend" | **CONTRADICTION** (see below) |

### Inconsistencies Found

#### CONTRADICTION 1: Transaction `status` enum values

The ERD now documents the team-agreed status values as `PENDING`, `COMPLETED`, `FAILED`. Two existing documents still list different values:

- **database-design.md** (transactions.status CHECK): `IN ('PENDING', 'PROCESSED', 'FLAGGED', 'REVIEWED')`.
- **api-contract.md** (Global Validation Rules, `transaction_status`): `PENDING`, `PROCESSED`, `FLAGGED`, `REVIEWED`.

**Impact:** The ERD and the existing documents disagree on the allowed status values. If the Alembic migration uses the database-design.md values, the ERD contract will be violated at runtime.

**Severity:** HIGH.

**Resolution required:** `database-design.md` must be updated to `IN ('PENDING', 'COMPLETED', 'FAILED')`. `api-contract.md` `transaction_status` enum must be updated to match. `ml-architecture.md` Failure Handling table must be updated to remove `status = 'ERROR'` references.

#### CONTRADICTION 2: ML architecture diagram labels Risk Aggregator as "Backend"

- **ml-architecture.md** ASCII diagram (line 28-30): The Risk Aggregator box is labelled **"Backend: Risk Aggregator"**.
- **ml-architecture.md** text (line 230): *"The ML/Fraud Intelligence Service combines the three scores internally"*.
- **architecture.md** (line 99): *"The backend never calculates ML predictions itself. All fraud intelligence is computed by this service."*
- **Team decision:** The Risk Aggregator belongs **inside** the ML/Fraud Intelligence Service.

**Impact:** The diagram contradicts the text and the team decision, potentially confusing implementers about which codebase owns the aggregation logic.

**Severity:** MEDIUM.

**Resolution required:** Update the ml-architecture.md ASCII diagram to label the Risk Aggregator as part of the ML/Fraud Intelligence Service (not "Backend").

#### ERD ADDITION: `transactions.model_version` column

The ERD now documents a new column `transactions.model_version` (VARCHAR(20), NULL, FK → `model_metadata.model_version`) that does not exist in `database-design.md`.

**Required updates to `database-design.md`:**

1. Add `model_version VARCHAR(20) NULL FK → model_metadata(model_version)` to the `transactions` table specification.
2. Add CHECK constraint: `model_version` range 0–20 chars (or no CHECK — VARCHAR handles length).
3. Add index: `ix_transactions_model_version` on `model_version`.
4. Update the `model_metadata` uniqueness: change `UNIQUE(model_name, model_version)` to `UNIQUE(model_version)` (or keep both, noting that `model_version` UNIQUE supersedes).
5. Add `model_version` to the Terminology Mapping table (DB `model_version` → API `model_version`).

#### ERD SUPERSED: `model_metadata` uniqueness constraint

The ERD documents `model_version` as **UNIQUE** (single column), superseding the `UNIQUE(model_name, model_version)` composite in `database-design.md`. Both documents are valid until `database-design.md` is updated.

---

## Summary

### File Updated

`docs/database-erd.md` — updated to incorporate three team decisions: transaction status semantics, Risk Aggregator ownership, and model version traceability.

### Tables Included

All nine: `users`, `customers`, `merchants`, `transactions`, `alerts`, `audit_logs`, `customer_devices`, `model_metadata`, `risk_rules_config`. No tables added or removed.

### Relationships

9 foreign key relationships (R1–R8 + R10) + 1 standalone table (R9). The new R10 relationship (`model_metadata.model_version` → `transactions.model_version`) enables per-transaction model version traceability.

### Cardinalities

- 1:1 — `customers` ↔ `users` (when `customer_id IS NOT NULL`)
- 1:N — `customers` → `transactions`, `customers` → `customer_devices`, `users` → `alerts`, `users` → `audit_logs`, `users` → `model_metadata`, `merchants` → `transactions`, **`model_metadata` → `transactions`** (R10, new)
- 1:0..1 — `transactions` → `alerts` (only for HOLD decisions)

### Delete/Update Rules

9 FK rules documented (R1–R8 + R10). All use `RESTRICT` or `SET NULL` on DELETE. **No CASCADE on DELETE** for any financial relationship. R10 uses **RESTRICT** to preserve model version audit trail. Also fixed ON DELETE table numbering to match the Relationship Table (R1–R8, R10).

### Transaction Status Decision (final)

`transactions.status` uses lifecycle values: **`PENDING`, `COMPLETED`, `FAILED`**. Transaction status represents the transaction lifecycle, not ML/fraud-service health. Fraud outcomes are captured in `decision`, `risk_level`, and `risk_score` columns. ML service failure does **not** produce a special status — the transaction stays `PENDING` or is marked `FAILED` with fraud scores left NULL.

### ML Failure Handling Decision (final)

If the ML/Fraud Intelligence Service is unavailable or fails: do not silently approve; do not represent the ML failure as a normal transaction status; fail safely; preserve technical failure information through logging and error handling.

### Risk Aggregator Ownership (final)

The Risk Aggregator belongs **inside** the ML/Fraud Intelligence Service. The backend never duplicates the risk aggregation algorithm. The ML service computes `risk_score`, `risk_level`, and `decision` and returns them to the backend for persistence.

### Model Version Relationship (new)

`transactions.model_version` VARCHAR(20) NULL, FK → `model_metadata.model_version` (UNIQUE). ON DELETE RESTRICT, ON UPDATE CASCADE. NULL only when the transaction was never fraud-analysed. Model binaries remain on the filesystem.

### Fraud Result Storage

All fraud results (`ml_score`, `behaviour_score`, `rule_score`, `risk_score`, `risk_level`, `decision`, `explanation_json`, **`model_version`**) are persisted in `transactions`. Alerts duplicate key fraud fields for historical independence.

### Behaviour-Data Decision

All behavioural features are **computed at runtime** from `transactions` + `customer_devices`. No additional tables needed. No `customer_locations` table. The `previous_suspicious_count` feature now queries on `decision = 'HOLD'` or `risk_level = 'HIGH'` instead of `status = 'FLAGGED'`.

### Normalisation Findings

Two instances of intentional denormalisation (alerts duplicating transaction fraud data, users duplicating customer name fields). Both are justified and documented. No unintended violations found.

### Cross-Document Consistency

**2 contradictions reported** (not silently fixed):

1. **HIGH — Transaction status enum:** ERD documents `PENDING, COMPLETED, FAILED`. `database-design.md` and `api-contract.md` still list `PENDING, PROCESSED, FLAGGED, REVIEWED`. `ml-architecture.md` references `status = 'ERROR'`. Three documents require updates.
2. **MEDIUM — Risk Aggregator diagram:** ml-architecture.md ASCII diagram labels Risk Aggregator as "Backend". The text and team decision place it inside the ML service. Diagram requires correction.

**1 ERD addition reported** (requires `database-design.md` sync):

3. `transactions.model_version` column and FK are documented in the ERD but absent from `database-design.md`. Five specific updates to `database-design.md` are listed above.

**1 ERD supersession reported:**

4. `model_metadata.model_version` is now UNIQUE (single column), superseding `UNIQUE(model_name, model_version)` in `database-design.md`.

### Unresolved Decisions

**None.** All three team decisions have been fully incorporated into the ERD. Remaining work is synchronising the ERD decisions into the other architecture documents (items 1–4 above).
