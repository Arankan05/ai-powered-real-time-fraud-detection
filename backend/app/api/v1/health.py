"""Health-check endpoint.

At this foundation stage the endpoint verifies only that the backend
application is responsive.  Database and ML service health checks will be
added once those layers are implemented.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Return backend availability status.

    Returns HTTP 200 when the backend is operational.
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
    }
