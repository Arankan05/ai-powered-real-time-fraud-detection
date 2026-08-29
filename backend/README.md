# Backend

FastAPI-based REST API that orchestrates transaction processing, fraud detection, and alert management.

## Planned Modules

- **auth/** — JWT authentication and authorisation
- **transactions/** — Transaction ingestion and validation
- **fraud/** — Orchestration of ML prediction, behaviour analysis, and rule evaluation
- **risk/** — Risk aggregation and decision engine
- **alerts/** — Fraud alert creation and management
- **customers/** — Customer profile and history access

## Key Principles

- All transaction values are validated on the backend; never trust frontend input.
- Secrets come from environment variables only.
- All endpoints require authentication and appropriate authorisation.
- Audit logging on sensitive operations.

## Status

Not yet implemented. Foundation placeholder only.
