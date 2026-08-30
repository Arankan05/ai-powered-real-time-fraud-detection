"""Data-access layer for authentication-related database operations.

All user/customer lookups and inserts for the auth flow live here.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.customer import Customer
from app.models.user import User


class UserRepository:
    """Database operations for users and customers in the auth flow."""

    def __init__(self, db: Session) -> None:
        self._db = db

    # -- Queries ----------------------------------------------------------

    def get_user_by_email(self, email: str) -> User | None:
        """Find a user by their (case-insensitive) email address."""
        stmt = select(User).where(User.email == email.lower())
        return self._db.execute(stmt).scalar_one_or_none()

    def get_user_by_id(self, user_id: UUID) -> User | None:
        """Find a user by primary key."""
        return self._db.get(User, user_id)

    # -- Mutations --------------------------------------------------------

    def create_customer_and_user(
        self,
        *,
        email: str,
        password_hash: str,
        first_name: str,
        last_name: str,
        phone: str | None = None,
        date_of_birth=None,
        address: str | None = None,
    ) -> User:
        """Create a customer profile and linked user account atomically.

        The customer is inserted first, then the user references it via
        ``customer_id``.  Both inserts happen in a single transaction.
        """
        customer = Customer(
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            date_of_birth=date_of_birth,
            address=address,
        )
        self._db.add(customer)
        self._db.flush()  # get customer.id

        user = User(
            email=email.lower(),
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            date_of_birth=date_of_birth,
            role="customer",
            customer_id=customer.id,
        )
        self._db.add(user)
        self._db.commit()
        self._db.refresh(user)
        return user
