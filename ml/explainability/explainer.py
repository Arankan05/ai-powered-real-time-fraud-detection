"""SHAP-based explainability for the tuned XGBoost fraud model.

Uses ``shap.TreeExplainer`` — the exact, polynomial-time algorithm for
tree-based models — to compute per-feature SHAP values that explain
why a transaction received its fraud probability score.

The explainer is constructed **once** from the already-trained model
and reused for all explanation requests.  No retraining, refitting,
or test-label access occurs.

Output format (per feature)::

    {
        "feature": "<feature_name>",
        "importance": <float SHAP value>,
    }

Features are sorted by absolute SHAP value (descending) so the top
contributing factors appear first — matching the schema defined in
``docs/ml-architecture.md`` L91–L106 (``ml_top_factors``).

Usage::

    from ml.explainability.explainer import FraudExplainer
    from ml.predict.bundle import load_bundle

    bundle = load_bundle()
    explainer = FraudExplainer(bundle)
    factors = explainer.explain(X_transformed, feature_names)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ml.predict.bundle import ModelBundle


# ── Top-N factors to return by default ────────────────────────────────

_DEFAULT_TOP_N = 10


# ── Explainer ─────────────────────────────────────────────────────────


class FraudExplainer:
    """SHAP-based explainer for the tuned XGBoost fraud model.

    Computes exact TreeSHAP feature attributions using native XGBoost C++
    contributions (pred_contribs=True) with lazy fallback to the ``shap`` package.

    Args:
        bundle: A loaded ModelBundle containing the fitted XGBoost model.
    """

    def __init__(self, bundle: ModelBundle) -> None:
        self._bundle = bundle
        self._explainer = None

    @property
    def model_version(self) -> str:
        return self._bundle.model_version

    def explain(
        self,
        X_transformed: np.ndarray,
        feature_names: list[str],
        *,
        top_n: int = _DEFAULT_TOP_N,
    ) -> list[dict[str, Any]]:
        """Compute SHAP values and return the top contributing features.

        Args:
            X_transformed: 2-D array of preprocessed feature values
                           (same array fed to ``model.predict_proba``).
            feature_names: Ordered list matching the columns of
                           *X_transformed*.
            top_n: Number of top features (by |SHAP|) to return.
                   Defaults to 10.

        Returns:
            List of dicts ``{"feature": str, "importance": float}``,
            sorted by descending ``|importance|``.
        """
        row_shap = None

        # ── 1. Native XGBoost C++ TreeSHAP (fast, exact, low-memory) ────
        try:
            import xgboost as xgb

            booster = self._bundle.model.get_booster()
            dmat = xgb.DMatrix(X_transformed, feature_names=feature_names)
            contribs = booster.predict(dmat, pred_contribs=True)
            if contribs.ndim == 1:
                row_shap = contribs[:-1]
            else:
                row_shap = contribs[0, :-1]
        except Exception:
            row_shap = None

        # ── 2. Fallback to `shap` package if native computation fails ───
        if row_shap is None:
            try:
                import shap

                if self._explainer is None:
                    self._explainer = shap.TreeExplainer(self._bundle.model)
                shap_values = self._explainer.shap_values(X_transformed)
                row_shap = shap_values if shap_values.ndim == 1 else shap_values[0]
            except Exception:
                import logging
                logging.getLogger(__name__).warning("SHAP explanation failed", exc_info=True)
                return []

        # Build (feature, value, |value|) tuples and sort by |value| desc
        entries = [
            {"feature": name, "importance": float(val)}
            for name, val in zip(feature_names, row_shap)
        ]
        entries.sort(key=lambda e: abs(e["importance"]), reverse=True)

        return entries[:top_n]

    def explain_full(
        self,
        X_transformed: np.ndarray,
        feature_names: list[str],
    ) -> list[dict[str, Any]]:
        """Return SHAP values for ALL features (no truncation)."""
        return self.explain(X_transformed, feature_names, top_n=len(feature_names))
