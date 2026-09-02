# Development Workflow

## Team Structure

This is a 3-member team. Parallel development is enabled by clear module boundaries and agreed contracts.

| Role | Developer | Primary Responsibility |
|---|---|---|
| Backend + Database + Security | Developer A | FastAPI API, database (Alembic), authentication, authorisation, alerts, transaction lifecycle, ML service HTTP integration, persistence, API response mapping |
| Frontend + Dashboard + UX | Developer B | React banking app, fraud analyst dashboard, all UI components, API integration |
| ML + Fraud Intelligence | Developer C | ML model, feature engineering, behaviour engine, rule engine, risk aggregation formula, risk calculation, explainability, decision engine, ML/Fraud Intelligence HTTP service |

All three developers share ownership of tests, documentation, and Docker configuration.

### Ownership Boundaries

**Developer A (Backend)** integrates the ML/Fraud Intelligence Service via HTTP but does **not** duplicate or reimplement the risk algorithm. The backend:
- Calls the ML service for each transaction
- Persists the returned scores and decisions
- Creates alerts when decision is HOLD
- Maps ML responses to the API response format

**Developer C (ML/Fraud)** owns the complete fraud intelligence pipeline including:
- ML prediction (ml_score)
- Behaviour analysis (behaviour_score)
- Rule evaluation (rule_score)
- Risk aggregation formula and weights
- Risk calculation (risk_score, risk_level)
- Decision engine (APPROVE/VERIFY/HOLD)
- Explainability signal generation
- ML service HTTP endpoint and health check

## Branching Strategy

```
main          ← stable, reviewed, release-ready
  ↑
develop       ← integration branch; all features merge here first
  ↑
feature/*     ← one branch per task
fix/*         ← bug fix branches
```

### Branch Rules

- **`main`** — Stable, reviewed code. No direct commits. Only merged from `develop` after team review.
- **`develop`** — Integration branch. No direct feature development on `develop`. Feature branches merge into `develop` via pull request.
- **`feature/<description>`** — Created from `develop`. Merged back into `develop` via pull request.
- **`fix/<description>`** — Created from `develop`. Merged back into `develop` via pull request.

### Branch Naming Convention

```
feature/backend-auth-jwt
feature/database-schema
feature/ml-model-training
feature/behaviour-engine
feature/rule-engine
feature/frontend-dashboard
feature/frontend-banking-app
fix/transaction-validation-edge-case
```

### Merge Flow

```
feature branch → develop (via PR, ≥1 reviewer)
develop → main (via PR, ≥1 reviewer, all tests passing)
```

## Development Cycle

1. **Pick a task** from the agreed backlog.
2. **Create a feature branch** from `develop`.
3. **Implement** with tests.
4. **Self-review** before requesting a team review.
5. **Open a pull request** targeting `develop` with a clear description.
6. **At least one other team member reviews** before merge.
7. **Merge into `develop`** and delete the feature branch.
8. **Periodically merge `develop` into `main`** for stable releases.

## Conflict Resolution

1. Developer identifies conflicting changes during rebase or merge.
2. Feature branch owner updates their branch from `develop` (`git merge develop` or `git rebase develop`).
3. Owner resolves conflicts in their feature branch.
4. Owner runs all tests locally to verify the resolution.
5. Reviewer verifies the conflict resolution during PR review.
6. PR continues through the normal review and merge flow.

If two feature branches from different developers conflict, both developers coordinate. The branch that was opened first has priority. The second developer rebases on top of the first.

## Code Standards

### Backend (Python)

- Format: Black (default settings).
- Lint: Ruff.
- Type hints on all public functions and Pydantic models.
- Docstrings on all public modules, classes, and functions.
- Pytest for all tests; aim for >80% coverage on business logic.

### Frontend (TypeScript/React)

- Format: Prettier (default settings).
- Lint: ESLint with TypeScript plugin.
- Strict TypeScript (`"strict": true` in tsconfig).
- Functional components only; no class components.
- Vitest + React Testing Library for component tests.

### ML / Fraud Intelligence Service (Python)

- Format and lint: same as backend (Black + Ruff).
- All models serialised with joblib.
- Training scripts must be reproducible (fixed random seeds, logged parameters).
- Model evaluation metrics must be logged and persisted to `model_metadata` table.
- ML service exposes `/predict` and `/health` HTTP endpoints.

## Environment Setup

1. Clone the repository.
2. Copy `.env.example` to `.env` and configure local values.
3. Start PostgreSQL via `docker compose up -d`.
4. Follow per-module README for language-specific setup.

## Pull Request Checklist

- [ ] Code compiles/runs without errors.
- [ ] New tests added for new logic; existing tests pass.
- [ ] No secrets, passwords, or API keys in the diff.
- [ ] Database changes use idempotent DDL (`backend/db/postgres.py :: init_schema`) until Alembic migrations are introduced.
- [ ] README or docs updated if the change affects usage.
- [ ] No mock/hard-coded data introduced as permanent fixtures.
- [ ] API response shapes match `docs/api-contract.md`.
- [ ] ML service responses match `docs/ml-architecture.md` schemas.

## Definition of Done

A task is considered complete when:

- [ ] **Implementation complete** — All acceptance criteria satisfied.
- [ ] **Tests passing** — Unit tests pass; integration tests pass where applicable.
- [ ] **Validation complete** — Input validation implemented per `docs/api-contract.md`.
- [ ] **Documentation updated** — Relevant docs updated if the change affects contracts or behaviour.
- [ ] **No secrets** — No passwords, API keys, or credentials in code or config.
- [ ] **Integration verified** — Changes work with dependent modules (backend ↔ ML, frontend ↔ backend).
- [ ] **Code reviewed** — At least one team member has approved the PR.
- [ ] **Acceptance criteria satisfied** — All task-specific requirements met.

## Communication

- Architecture decisions are documented in `docs/` before implementation.
- API contracts in `docs/api-contract.md` are agreed before frontend/backend parallel work begins.
- ML service schemas in `docs/ml-architecture.md` are agreed before backend/ML parallel work begins.
- Breaking changes to contracts require team discussion before merging.

## Status

This workflow is agreed upon. Development is in progress (Step 43
ML monitoring and observability complete; Steps 31–42 complete).
