# Architecture

## System Architecture

The fraud detection system follows a layered, pipeline-based architecture that separates concerns across distinct processing stages.

```
┌─────────────────────────────────────────────────────────────┐
│                        Presentation                          │
│  ┌──────────────────────┐   ┌─────────────────────────────┐ │
│  │ Simulated Banking App│   │   Fraud Analyst Dashboard   │ │
│  │   (React + Vite)     │   │     (React + Vite)          │ │
│  └──────────┬───────────┘   └──────────────┬──────────────┘ │
└─────────────┼───────────────────────────────┼───────────────┘
              │           HTTPS               │
              ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend (FastAPI)                           │
│  ┌─────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────┐ │
│  │  Auth   │  │ Transactions │  │  Alerts   │  │Customers│ │
│  │ Service │  │   Service    │  │  Service  │  │ Service │ │
│  └─────────┘  └──────┬───────┘  └──────────┘  └─────────┘ │
└───────────────────────┼─────────────────────────────────────┘
                        │ Internal HTTP
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           ML / Fraud Intelligence Service (separate)         │
│                                                               │
│  ┌──────────────────┐                                        │
│  │ Feature          │  Extract features from transaction     │
│  │ Engineering      │  + customer profile + history          │
│  └────────┬─────────┘                                        │
│           ▼                                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │ ML Prediction  │  │ Behaviour      │  │ Rule-Based    │  │
│  │ (XGBoost/      │  │ Anomaly        │  │ Risk Signals  │  │
│  │  scikit-learn) │  │ Analysis       │  │               │  │
│  └───────┬────────┘  └───────┬────────┘  └──────┬────────┘  │
│          └───────────────────┼───────────────────┘           │
│                              ▼                                │
│                   ┌────────────────────┐                     │
│                   │  Risk Aggregator   │                     │
│                   └────────┬───────────┘                     │
│                            ▼                                  │
│                   ┌────────────────────┐                     │
│                   │  Explainability    │                     │
│                   │  (SHAP / feature   │                     │
│                   │   importance)      │                     │
│                   └────────┬───────────┘                     │
│                            ▼                                  │
│                   ┌────────────────────┐                     │
│                   │  Decision Engine   │                     │
│                   │  (configurable     │                     │
│                   │   thresholds)      │                     │
│                   └────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                                │
│  ┌──────────────┐                                           │
│  │  PostgreSQL  │  Users, customers, merchants, transactions,│
│  │              │  alerts, audit logs, model metadata,       │
│  │              │  risk rules config, customer devices       │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Presentation Layer (Frontend)

- **Simulated Banking App:** Allows users to initiate transactions (transfers, payments). Communicates with the backend via REST API.
- **Fraud Analyst Dashboard:** Displays flagged transactions, risk scores, decision explanations, and historical analytics.

### API Layer (FastAPI)

- JWT authentication and role-based authorisation (customer, fraud_analyst, admin).
- Input validation via Pydantic schemas.
- Orchestrates the fraud detection pipeline by calling the ML/Fraud Intelligence Service via internal HTTP.
- Persists transactions, fraud results, and alerts to PostgreSQL.
- Maps ML service responses to API response format.
- Exposes REST endpoints for auth, transactions, fraud checks, alerts, customers, analytics, and health.

### ML / Fraud Intelligence Service (Separate Process)

The ML/Fraud Intelligence Service runs as a **separate internal HTTP service** (not in-process). It is owned by Developer C (ML/Fraud developer).

It provides:

1. **Feature Engineering** — Transforms raw transaction + customer data into model features.
2. **ML Prediction** — Supervised model outputs a fraud probability (0–100).
3. **Behavioural Anomaly Analysis** — Detects deviations from customer baseline (0–100).
4. **Rule-Based Risk Signals** — Evaluates configurable rules from `risk_rules_config` table (0–100).
5. **Risk Aggregation** — Weighted combination: `risk_score = (w_ml × ml_score) + (w_behaviour × behaviour_score) + (w_rule × rule_score)`. Clamped to [0, 100].
6. **Explainability** — SHAP values, rule triggers, and behavioural deviations.
7. **Decision Engine** — Maps risk score to APPROVE/VERIFY/HOLD using configurable thresholds.

The backend never calculates ML predictions, behaviour scores, or rule scores itself. All fraud intelligence is computed by this service and returned via HTTP.

### Backend Responsibilities (for the fraud pipeline)

- Call the ML/Fraud Intelligence Service via internal HTTP for each transaction.
- Persist the returned scores, risk level, decision, and explanation to the `transactions` table.
- Create an `alerts` record when the decision is HOLD.
- Map the ML service response to the API response format (DB `explanation_json` → API `explanation`).
- Handle ML service failures gracefully (timeout, connection error, model unavailable).

### Explainability Module

- Produces human-readable explanations for each fraud decision.
- Uses SHAP values or feature importance to identify the top contributing factors.
- Explanations are persisted alongside the transaction decision.

### Decision Engine

- Maps the aggregated risk score to a decision using configurable thresholds:

| Score | Level | Decision |
|---|---|---|
| 0–30 | LOW | APPROVE |
| 31–70 | MEDIUM | VERIFY (require additional authentication) |
| 71–100 | HIGH | HOLD transaction + create ALERT |

- Thresholds are read from environment variables at startup and can be reloaded without redeployment.

### Data Layer

- **PostgreSQL** (Step 40) stores: `users` and `alerts` (implemented).
  Remaining tables (`customers`, `merchants`, `transactions`,
  `audit_logs`, `customer_devices`, `model_metadata`,
  `risk_rules_config`) are designed but not yet implemented.
- Schema is initialised idempotently at startup via
  `backend/db/postgres.py :: init_schema` (Alembic migrations planned
  but not yet in use).
- Backend supports `PERSISTENCE_BACKEND=postgres` (default) or
  `sqlite` (legacy fallback; see `.env.example`).
- ML historical features (`ml/features/history.py`) remain on SQLite.
- No real banking data is used. Synthetic data and public datasets are used for development and ML training.

## Communication Patterns

| Interaction | Pattern | Protocol |
|---|---|---|
| Frontend → Backend | Synchronous request/response | HTTPS (REST/JSON) |
| Backend → ML/Fraud Service | Synchronous internal HTTP | HTTP (REST/JSON) on `ML_SERVICE_HOST:ML_SERVICE_PORT` |
| Backend → Database | Synchronous SQL | psycopg 3 over PostgreSQL wire (or sqlite3 for fallback / ML history) |

## Security Architecture

- JWT bearer tokens for authentication.
- Role-based access control (RBAC): `customer`, `fraud_analyst`, `admin`. Roles stored in the `users` table.
- Separate `users` table for authentication; `customers` table for business profiles.
- All sensitive configuration via environment variables.
- Passwords hashed with bcrypt.
- Audit log captures all state-changing operations with actor, timestamp, and action.
- CORS restricted to known frontend origins.

## Deployment Model

- Local development: Docker Compose with PostgreSQL. Backend, frontend, and ML/Fraud Intelligence service run as separate local processes.
- Production (future): Containerised services behind a reverse proxy with managed PostgreSQL.

## Status

**Step 44 (production decision pipeline) is complete.** The transaction
endpoint now supports idempotent processing via `Idempotency-Key`
header, explicit ML failure handling (no fabricated predictions),
and decision consistency between response and persisted alerts.
Step 43 (ML monitoring and observability), Step 42 (production
hardening), Step 41 (customer identity isolation), Step 40
(PostgreSQL migration), and all earlier steps are complete.

**Step 45 (fraud decision audit trail) is complete.** An append-only
audit trail records all important fraud decision events:

- `DECISION_MADE` — ML prediction completed successfully
- `ML_FAILURE` — ML service unavailable or errored
- `ALERT_CREATED` — fraud alert created for HOLD decision
- `ALERT_STATE_CHANGED` — analyst changed alert status
- `OUTCOME_RECORDED` — fraud outcome feedback recorded

The audit trail is stored in the `fraud_decision_audit` table
(PostgreSQL) with customer isolation, role-based access control,
bounded explanation summaries, and idempotency coordination.
The audit endpoint (`GET /api/v1/audit/transactions/{id}`) is
protected by authentication and authorization — customers may
only access their own audit trail.
