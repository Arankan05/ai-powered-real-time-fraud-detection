"""Model bundle serialisation — save / load tuned XGBoost artifacts.

A *ModelBundle* packages everything the prediction pipeline needs:

  - fitted XGBoost classifier
  - fitted preprocessing artifacts (StandardScaler + LabelEncoder)
  - decision threshold (selected during tuning)
  - ordered feature names (schema contract between training and inference)
  - model version string (for audit traceability)

Serialisation uses **joblib** (as specified in ``docs/ml-architecture.md``
L178 / L273).  The entire bundle is stored as a single ``.joblib`` file.

Artifact files are excluded from version control by ``.gitignore``
(``ml/models/*.joblib``).  Each developer generates them locally via::

    python -m ml.predict.save_model

The bundle is loaded once at service startup and reused for all
predictions — no retraining or refitting per request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml.models.baseline import PreprocessingArtifacts

# ── Default paths ─────────────────────────────────────────────────────

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_DEFAULT_FILENAME = "fraud_xgb_tuned.joblib"


# ── Bundle container ──────────────────────────────────────────────────


@dataclass
class ModelBundle:
    """All artifacts required for fraud prediction.

    Attributes:
        model: Fitted XGBClassifier.
        preprocessing: Fitted StandardScaler + LabelEncoder pipeline.
        threshold: Decision threshold for binary classification.
        feature_names: Ordered list of feature column names expected
                       by the model (must match training order).
        model_version: Human-readable version string.
    """

    model: Any  # XGBClassifier (avoid top-level xgboost import)
    preprocessing: PreprocessingArtifacts
    threshold: float
    feature_names: list[str]
    model_version: str = "fraud-xgb-v1.0.0"

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


# ── Save ──────────────────────────────────────────────────────────────


def save_bundle(
    bundle: ModelBundle,
    path: str | Path | None = None,
) -> Path:
    """Serialize a ModelBundle to a joblib file.

    Args:
        bundle: The trained model bundle.
        path: Destination file path.  Defaults to
              ``ml/models/fraud_xgb_tuned.joblib``.

    Returns:
        Resolved Path of the saved file.
    """
    if path is None:
        path = _DEFAULT_MODEL_DIR / _DEFAULT_FILENAME
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": bundle.model,
        "preprocessing": bundle.preprocessing,
        "threshold": bundle.threshold,
        "feature_names": bundle.feature_names,
        "model_version": bundle.model_version,
    }
    joblib.dump(payload, path, compress=3)
    return path.resolve()


# ── Load ──────────────────────────────────────────────────────────────


def load_bundle(path: str | Path | None = None) -> ModelBundle:
    """Load a ModelBundle from a joblib file.

    Args:
        path: Path to the saved bundle.  Defaults to
              ``ml/models/fraud_xgb_tuned.joblib``.

    Returns:
        Reconstructed ModelBundle.

    Raises:
        FileNotFoundError: If the artifact file does not exist.
        KeyError: If the file is missing required keys.
    """
    if path is None:
        path = _DEFAULT_MODEL_DIR / _DEFAULT_FILENAME
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. "
            f"Run `python -m ml.predict.save_model` to generate it."
        )

    payload = joblib.load(path)

    required_keys = {"model", "preprocessing", "threshold", "feature_names"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise KeyError(
            f"Model artifact is missing required keys: {sorted(missing)}"
        )

    return ModelBundle(
        model=payload["model"],
        preprocessing=payload["preprocessing"],
        threshold=float(payload["threshold"]),
        feature_names=list(payload["feature_names"]),
        model_version=str(payload.get("model_version", "unknown")),
    )


# ── Helpers ───────────────────────────────────────────────────────────


def default_model_path() -> Path:
    """Return the default model artifact path."""
    return (_DEFAULT_MODEL_DIR / _DEFAULT_FILENAME).resolve()


def model_exists(path: str | Path | None = None) -> bool:
    """Check whether a model artifact file exists."""
    if path is None:
        path = _DEFAULT_MODEL_DIR / _DEFAULT_FILENAME
    return Path(path).exists()
