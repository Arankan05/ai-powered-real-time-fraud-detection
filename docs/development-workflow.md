# Development Workflow

## Team Structure

This is a 3-member team. Parallel development is enabled by clear module boundaries and agreed contracts.

| Role | Primary Responsibility |
|---|---|
| Developer A | Backend API, database, authentication |
| Developer B | Frontend (banking app + dashboard) |
| Developer C | ML pipeline, fraud engine, risk engine |

All three developers share ownership of tests, documentation, and Docker configuration.

## Branching Strategy

- `main` — Stable, reviewed, working code. Direct pushes are not permitted.
- `feature/<description>` — One branch per feature or task. Merged via pull request after review.
- `fix/<description>` — Bug fix branches.

### Branch Naming Convention

```
feature/backend-auth-jwt
feature/frontend-dashboard-alerts
feature/ml-xgboost-training
fix/transaction-validation-edge-case
```

## Development Cycle

1. **Pick a task** from the agreed backlog.
2. **Create a feature branch** from `main`.
3. **Implement** with tests.
4. **Self-review** before requesting a team review.
5. **Open a pull request** with a clear description of changes.
6. **At least one other team member reviews** before merge.
7. **Merge into `main`** and delete the feature branch.

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

### ML (Python)

- Format and lint: same as backend.
- All models serialised with joblib.
- Training scripts must be reproducible (fixed random seeds, logged parameters).
- Model evaluation metrics must be logged and persisted.

## Environment Setup

1. Clone the repository.
2. Copy `.env.example` to `.env` and configure local values.
3. Start PostgreSQL via `docker compose up -d`.
4. Follow per-module README for language-specific setup.

## Pull Request Checklist

- [ ] Code compiles/runs without errors.
- [ ] New tests added for new logic; existing tests pass.
- [ ] No secrets, passwords, or API keys in the diff.
- [ ] Database changes use Alembic migrations.
- [ ] README or docs updated if the change affects usage.
- [ ] No mock/hard-coded data introduced as permanent fixtures.

## Communication

- Architecture decisions are documented in `docs/` before implementation.
- API contracts in `docs/api-contract.md` are agreed before frontend/backend parallel work begins.
- Breaking changes to contracts require team discussion before merging.

## Status

This workflow is agreed upon. Development has not yet started.
