# Tests

Cross-cutting integration and end-to-end tests that span multiple services.

## Planned Structure

- **integration/** — Tests that verify backend + ML + database interactions
- **e2e/** — End-to-end tests simulating full transaction flows
- **fixtures/** — Shared test fixtures and synthetic test data generators

## Per-Module Tests

Unit tests live inside each module:

- `backend/tests/` — Pytest tests for API endpoints and services
- `frontend/src/**/*.test.tsx` — Vitest + React Testing Library tests
- `ml/tests/` — Pytest tests for feature engineering and model behaviour

## Status

Not yet implemented. Foundation placeholder only.
