"""Backend application package.

Re-exports the FastAPI ``app`` instance from the root-level entry point
(``backend/_main.py``) so that ``from backend.app import app`` works
correctly even though the ``backend/app/`` sub-package exists.
"""

from backend._main import app

__all__ = ["app"]
