"""Reusable prediction pipeline for the tuned XGBoost fraud model.

Loads a :class:`ModelBundle` (once) and provides a stateless
:func:`predict` function that:

  1. Validates the input feature DataFrame against the model schema.
  2. Applies the fitted preprocessing pipeline (StandardScaler + LabelEncoder).
  3. Runs the XGBoost model to produce fraud probabilities.
  4. Applies the tuned decision threshold (0.50) for binary prediction.
  5. Returns both probability and prediction.

The prediction function does **not** retrain, refit, or modify the model.
It is safe to call repeatedly and concurrently.

Usage::

    from ml.predict.predictor import FraudPredictor

    predictor = FraudPredictor()          # loads model at construction
    result = predictor.predict(features)  # stateless inference
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.models.baseline import apply_preprocessing
from ml.predict.bundle import ModelBundle, load_bundle

# ── Forbidden columns ─────────────────────────────────────────────────

_FORBIDDEN_INPUT_COLS = frozenset({"isFraud", "TransactionID"})


# ── Result container ──────────────────────────────────────────────────


@dataclass(frozen=True)
class PredictionResult:
    """Output of a single fraud prediction.

    Attributes:
        fraud_probability: Continuous probability ∈ [0, 1].
        fraud_prediction: Binary label (0 = legitimate, 1 = fraud).
        threshold: Decision threshold used.
        model_version: Version string of the model used.
    """

    fraud_probability: float
    fraud_prediction: int
    threshold: float
    model_version: str


# ── Predictor ─────────────────────────────────────────────────────────


class FraudPredictor:
    """Stateless fraud predictor wrapping a loaded ModelBundle.

    Construct once (e.g. at application startup) and call
    :meth:`predict` for each transaction.

    Args:
        bundle_path: Path to the saved model bundle.
                     ``None`` uses the default location.
    """

    def __init__(self, bundle_path: str | Path | None = None) -> None:
        self._bundle: ModelBundle = load_bundle(bundle_path)
        self._feature_set: set[str] = set(self._bundle.feature_names)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def model_version(self) -> str:
        return self._bundle.model_version

    @property
    def threshold(self) -> float:
        return self._bundle.threshold

    @property
    def feature_names(self) -> list[str]:
        return list(self._bundle.feature_names)

    @property
    def is_loaded(self) -> bool:
        return self._bundle is not None

    # ── Prediction ────────────────────────────────────────────────────

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """Run fraud prediction on a single transaction.

        Args:
            features: A single-row DataFrame with the 24 engineered
                      features (columns must match
                      ``ModelBundle.feature_names``).

        Returns:
            :class:`PredictionResult` with probability and prediction.

        Raises:
            ValueError: If input validation fails (missing columns,
                        wrong shape, forbidden columns).
        """
        self._validate_input(features)

        # Ensure column order matches training
        X = features[self._bundle.feature_names].copy()

        # Apply fitted preprocessing (no fitting — transform only)
        X_transformed = apply_preprocessing(X, self._bundle.preprocessing)

        # Probability from model
        prob = float(self._bundle.model.predict_proba(X_transformed)[0, 1])

        # Binary prediction via threshold
        pred = 1 if prob >= self._bundle.threshold else 0

        return PredictionResult(
            fraud_probability=prob,
            fraud_prediction=pred,
            threshold=self._bundle.threshold,
            model_version=self._bundle.model_version,
        )

    def predict_batch(self, features: pd.DataFrame) -> list[PredictionResult]:
        """Run fraud prediction on multiple transactions.

        Args:
            features: DataFrame with N rows and 24 engineered feature
                      columns.

        Returns:
            List of :class:`PredictionResult`, one per row.
        """
        self._validate_input(features, allow_multi_row=True)
        X = features[self._bundle.feature_names].copy()
        X_transformed = apply_preprocessing(X, self._bundle.preprocessing)

        probs = self._bundle.model.predict_proba(X_transformed)[:, 1]
        results = []
        for prob in probs:
            p = float(prob)
            results.append(
                PredictionResult(
                    fraud_probability=p,
                    fraud_prediction=1 if p >= self._bundle.threshold else 0,
                    threshold=self._bundle.threshold,
                    model_version=self._bundle.model_version,
                )
            )
        return results

    # ── Validation ────────────────────────────────────────────────────

    def _validate_input(
        self, features: pd.DataFrame, *, allow_multi_row: bool = False
    ) -> None:
        """Validate input DataFrame against the model schema."""
        if not isinstance(features, pd.DataFrame):
            raise ValueError(
                f"Expected pd.DataFrame, got {type(features).__name__}"
            )

        # Forbidden columns
        forbidden_found = _FORBIDDEN_INPUT_COLS & set(features.columns)
        if forbidden_found:
            raise ValueError(
                f"Forbidden columns in input: {sorted(forbidden_found)}. "
                f"Remove isFraud and TransactionID before prediction."
            )

        # Row count
        if not allow_multi_row and len(features) != 1:
            raise ValueError(
                f"Expected exactly 1 row, got {len(features)}. "
                f"Use predict_batch() for multiple rows."
            )

        if len(features) == 0:
            raise ValueError("Input DataFrame is empty (0 rows).")

        # Column check
        missing = self._feature_set - set(features.columns)
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required feature(s): "
                f"{sorted(missing)}"
            )

        extra = set(features.columns) - self._feature_set
        if extra:
            # Allow extra columns (they'll be ignored) but warn
            pass
