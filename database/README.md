# Database

PostgreSQL schema management via Alembic migrations.

## Structure

- **alembic/** — Migration environment (to be initialised during implementation)
- **init/** — Docker entrypoint scripts executed on first container start

## Rules

- Real banking/customer data must never be used.
- Synthetic data and legitimate public datasets are permitted for development and testing.
- All schema changes must go through Alembic migrations; never modify the database manually.

## Status

Not yet implemented. Foundation placeholder only.
