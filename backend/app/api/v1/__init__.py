"""Aggregated API v1 router.

All v1 endpoint routers are registered here under the ``/api/v1`` prefix.
Currently the health and auth endpoints are active; additional routers
will be included as they are implemented.
"""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["Health"])
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])

# Future routers — uncomment as implemented:
# from app.api.v1.customers import router as customers_router
# from app.api.v1.transactions import router as transactions_router
# from app.api.v1.fraud import router as fraud_router
# from app.api.v1.alerts import router as alerts_router
# from app.api.v1.analytics import router as analytics_router
#
# api_router.include_router(customers_router, prefix="/customers", tags=["Customers"])
# api_router.include_router(transactions_router, prefix="/transactions", tags=["Transactions"])
# api_router.include_router(fraud_router, prefix="/fraud", tags=["Fraud Check"])
# api_router.include_router(alerts_router, prefix="/alerts", tags=["Alerts"])
# api_router.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
