"""Customer service — profile retrieval with authorization."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.core.errors import ForbiddenException, NotFoundException
from app.models.user import User
from app.repositories.transaction import CustomerRepository
from app.schemas.customer import CustomerResponse


class CustomerService:
    """Customer profile operations with ownership enforcement."""

    def __init__(self, db: Session) -> None:
        self._repo = CustomerRepository(db)

    def get_me(self, current_user: User) -> CustomerResponse:
        """Get the current customer's own profile (customer role only)."""
        if current_user.role != "customer":
            raise ForbiddenException(detail="Only customers can access this endpoint")

        if current_user.customer_id is None:
            raise NotFoundException(detail="No customer profile linked to this user")

        customer = self._repo.get_by_id(current_user.customer_id)
        if customer is None:
            raise NotFoundException(detail="Customer not found")

        return CustomerResponse.model_validate(customer)

    def get_by_id(self, customer_id: UUID, current_user: User) -> CustomerResponse:
        """Get a customer profile by ID with ownership enforcement.

        Customers see own only. Analysts/admins see any.
        """
        if current_user.role == "customer":
            if customer_id != current_user.customer_id:
                raise ForbiddenException(detail="Insufficient permissions")

        customer = self._repo.get_by_id(customer_id)
        if customer is None:
            raise NotFoundException(detail="Customer not found")

        return CustomerResponse.model_validate(customer)
