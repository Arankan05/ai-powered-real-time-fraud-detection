"""Application ORM models.

Importing this package registers all nine application tables with
``Base.metadata``, ensuring Alembic autogenerate and ``create_all``
see the full schema.
"""

from app.models.alert import Alert
from app.models.audit_log import AuditLog
from app.models.customer import Customer
from app.models.customer_device import CustomerDevice
from app.models.merchant import Merchant
from app.models.model_metadata import ModelMetadata
from app.models.risk_rules_config import RiskRulesConfig
from app.models.transaction import Transaction
from app.models.user import User

__all__ = [
    "Alert",
    "AuditLog",
    "Customer",
    "CustomerDevice",
    "Merchant",
    "ModelMetadata",
    "RiskRulesConfig",
    "Transaction",
    "User",
]
