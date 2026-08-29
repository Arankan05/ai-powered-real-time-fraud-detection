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
│                    API Gateway (FastAPI)                      │
│  ┌─────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────┐ │
│  │  Auth   │  │ Transactions │  │  Alerts   │  │Customers│ │
│  │ Service │  │   Service    │  │  Service  │  │ Service │ │
│  └─────────┘  └──────┬───────┘  └──────────┘  └─────────┘ │
└───────────────────────┼─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Fraud Detection Pipeline                    │
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
│  │  PostgreSQL  │  Transactions, customers, alerts,          │
│  │              │  audit logs, risk results                   │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Presentation Layer (Frontend)

- **Simulated Banking App:** Allows users to initiate transactions (transfers, payments). Communicates with the backend via REST API.
- **Fraud Analyst Dashboard:** Displays flagged transactions, risk scores, decision explanations, and historical analytics.

### API Layer (FastAPI)

- JWT authentication and role-based authorisation (customer, analyst, admin).
- Input validation via Pydantic schemas.
- Orchestrates the fraud detection pipeline for each transaction.
- Exposes REST endpoints for transactions, alerts, customers, and analytics.

### Fraud Detection Pipeline

The pipeline runs synchronously for each transaction and consists of four independent scoring stages followed by aggregation:

1. **ML Prediction** — Supervised model outputs a fraud probability based on engineered features.
2. **Behavioural Anomaly Analysis** — Detects deviations from the customer's established behavioural baseline (spending patterns, typical locations, devices, time-of-day).
3. **Rule-Based Risk Signals** — Evaluates configurable business rules (e.g., velocity limits, high-risk merchant categories, impossible travel).
4. **Risk Aggregator** — Combines the three scores using weighted aggregation into a single 0–100 risk score.

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

- **PostgreSQL** stores all persistent data: customers, transactions, alerts, audit logs, risk results, and model metadata.
- Schema is managed exclusively through Alembic migrations.
- No real banking data is used. Synthetic data and public datasets are used for development and ML training.

## Communication Patterns

| Interaction | Pattern | Protocol |
|---|---|---|
| Frontend → Backend | Synchronous request/response | HTTPS (REST/JSON) |
| Backend → ML | In-process call or internal HTTP | Function call / HTTP |
| Backend → Database | Synchronous ORM | SQLAlchemy over PostgreSQL wire |

## Security Architecture

- JWT bearer tokens for authentication.
- Role-based access control (RBAC): customer, fraud_analyst, admin.
- All sensitive configuration via environment variables.
- Passwords hashed with bcrypt.
- Audit log captures all state-changing operations with actor, timestamp, and action.
- CORS restricted to known frontend origins.

## Deployment Model

- Local development: Docker Compose with PostgreSQL. Backend, frontend, and ML run as local processes.
- Production (future): Containerised services behind a reverse proxy with managed PostgreSQL.

## Status

This architecture is agreed upon but **not yet implemented**. All components are pending development.
