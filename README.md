# AI-Powered Real-Time Financial Fraud Detection

A competition-ready prototype that detects fraudulent financial transactions in real time using machine learning, behavioural anomaly analysis, and rule-based risk scoring.

## System Overview

```
Customer
  ↓
Simulated Banking Application (React)
  ↓
REST API (FastAPI)
  ↓
Transaction Service
  ↓
Customer Profile + Historical Transactions
  ↓
Feature Engineering
  ↓
ML Model + Behaviour Analysis + Risk Rules
  ↓
Risk Aggregator
  ↓
Explainability
  ↓
Decision Engine
  ↓
PostgreSQL
  ↓
Fraud Alerts
  ↓
Fraud Analyst Dashboard (React)
```

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite, Tailwind CSS, shadcn/ui, React Router, Axios, React Hook Form, Zod, Recharts |
| Backend | Python, FastAPI, Pydantic, SQLAlchemy, Alembic, PostgreSQL, JWT |
| ML | pandas, NumPy, scikit-learn, XGBoost, joblib |
| Testing | Pytest, Vitest, React Testing Library, Postman |
| Infrastructure | Docker, Docker Compose, environment variables |

## Repository Structure

```
/
├── frontend/      # React banking app and fraud analyst dashboard
├── backend/       # FastAPI REST API and business logic
├── ml/            # Feature engineering, model training, and inference
├── database/      # Migrations, init scripts (synthetic data only)
├── tests/         # Cross-cutting integration and E2E tests
├── scripts/       # Developer convenience scripts
├── docs/          # Architecture and contract documentation
├── .env.example   # Environment variable template (no real values)
├── .gitignore
├── docker-compose.yml
└── README.md
```

## Getting Started

> **Prerequisites:** Docker Desktop, Python 3.11+, Node.js 20+, Git

### 1. Clone the repository

```bash
git clone <repository-url>
cd ai-powered-real-time-fraud-detection
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in appropriate local development values
```

### 3. Start infrastructure

```bash
docker compose up -d
```

### 4. Backend setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
# (requirements.txt will be created during implementation phase)
```

### 5. Frontend setup

```bash
cd frontend
npm install
# (package.json will be created during implementation phase)
```

### 6. ML setup

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
# (requirements.txt will be created during implementation phase)
```

## Risk Score Model

| Score Range | Risk Level | Decision |
|---|---|---|
| 0 – 30 | LOW | APPROVE |
| 31 – 70 | MEDIUM | VERIFY |
| 71 – 100 | HIGH | HOLD + ALERT |

Thresholds are configurable via environment variables.

## Documentation

- [Architecture](docs/architecture.md)
- [Development Workflow](docs/development-workflow.md)
- [Database Design](docs/database-design.md)
- [API Contract](docs/api-contract.md)
- [ML Architecture](docs/ml-architecture.md)

## Team

3-member development team. See [development-workflow.md](docs/development-workflow.md) for role assignments and branching strategy.

## Status

**Phase:** Foundation established. Implementation has not yet begun.
