"""Risk Aggregation — combines ML, behaviour, and rule signals into a final risk score.

Implements the risk aggregation layer described in
``docs/ml-architecture.md`` §5:

    risk_score = (w_ml × ml_score) + (w_behaviour × behaviour_score) + (w_rule × rule_score)

The aggregator is **independent** from:
* FastAPI routes — pure Python, no HTTP dependencies.
* Database implementation — no persistence.
* Model training — no labels, no retraining, no preprocessing.
* Feature engineering — operates on already-computed scores.

Design principles
-----------------
* Deterministic: identical inputs → identical outputs.
* Explicit formula: the weighted-sum is the only computation.
* Configurable: weights and thresholds are module-level constants
  (override via environment variables).
* JSON/API-friendly structured output.
* No label access: the aggregator never touches ``isFraud``.

Usage::

    from ml.risk.aggregator import aggregate_risk

    assessment = aggregate_risk(
        fraud_probability=0.72,
        behaviour_score=60,
        rule_score=35,
    )
    # assessment.ml_score        → 72
    # assessment.risk_score      → 61
    # assessment.risk_level      → "MEDIUM"
    # assessment.decision        → "VERIFY"
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any


# ── Configurable weights (architecture §5) ───────────────────────────
#
# "Weights are configurable via environment variables
#  (ML_WEIGHT_ML, ML_WEIGHT_BEHAVIOUR, ML_WEIGHT_RULE)."
# Default weights: w_ml = 0.50, w_behaviour = 0.30, w_rule = 0.20.

DEFAULT_WEIGHT_ML: float = 0.50
DEFAULT_WEIGHT_BEHAVIOUR: float = 0.30
DEFAULT_WEIGHT_RULE: float = 0.20

_WEIGHT_ML = float(os.environ.get("ML_WEIGHT_ML", str(DEFAULT_WEIGHT_ML)))
_WEIGHT_BEHAVIOUR = float(os.environ.get("ML_WEIGHT_BEHAVIOUR", str(DEFAULT_WEIGHT_BEHAVIOUR)))
_WEIGHT_RULE = float(os.environ.get("ML_WEIGHT_RULE", str(DEFAULT_WEIGHT_RULE)))


# ── Configurable thresholds (architecture §5) ────────────────────────
#
# | Score   | Level  | Decision      |
# |---------|--------|---------------|
# | 0–30    | LOW    | APPROVE       |
# | 31–70   | MEDIUM | VERIFY        |
# | 71–100  | HIGH   | HOLD + ALERT  |
#
# Stored as (lower_bound_exclusive, level, decision) tuples,
# evaluated from highest to lowest.

RISK_THRESHOLDS: list[tuple[int, str, str]] = [
    (70, "HIGH", "HOLD"),
    (30, "MEDIUM", "VERIFY"),
]
"""Threshold table: score > threshold → (level, decision).
Falls through to (LOW, APPROVE) when no threshold matches."""

DEFAULT_RISK_LEVEL: str = "LOW"
DEFAULT_DECISION: str = "APPROVE"

# Score bounds
MIN_SCORE: int = 0
MAX_SCORE: int = 100


# ── Output dataclass ─────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class RiskAssessment:
    """Complete risk assessment from the aggregation layer.

    All integer scores are in [0, 100].
    """

    ml_score: int
    """ML fraud probability scaled to [0, 100]."""

    behaviour_score: int
    """Behavioural anomaly score [0, 100]."""

    rule_score: int
    """Rule-based risk score [0, 100]."""

    risk_score: int
    """Weighted aggregate risk score [0, 100]."""

    risk_level: str
    """Risk category: LOW, MEDIUM, or HIGH."""

    decision: str
    """Recommended action: APPROVE, VERIFY, or HOLD."""

    fraud_probability: float
    """Original ML fraud probability in [0.0, 1.0]."""

    weights: dict[str, float]
    """Weights used for the aggregation (for audit trail)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (for JSON serialisation)."""
        return dataclasses.asdict(self)


# ── Main aggregation function ────────────────────────────────────────


def aggregate_risk(
    fraud_probability: float,
    behaviour_score: int,
    rule_score: int,
    *,
    w_ml: float | None = None,
    w_behaviour: float | None = None,
    w_rule: float | None = None,
) -> RiskAssessment:
    """Combine ML probability, behaviour score, and rule score into a risk assessment.

    Formula (architecture §5)::

        risk_score = (w_ml × ml_score) + (w_behaviour × behaviour_score) + (w_rule × rule_score)

    The result is clamped to [0, 100] and mapped to a risk level and
    decision using the threshold table.

    Args:
        fraud_probability: ML model's fraud probability in [0.0, 1.0].
        behaviour_score: Behavioural anomaly score [0, 100]
                         (from :func:`ml.rules.engine.evaluate_rules`).
        rule_score: Rule-based risk score [0, 100]
                    (from :func:`ml.rules.engine.evaluate_rules`).
        w_ml: Override ML weight (default: env var or 0.50).
        w_behaviour: Override behaviour weight (default: env var or 0.30).
        w_rule: Override rule weight (default: env var or 0.20).

    Returns:
        :class:`RiskAssessment` with all computed fields.

    Raises:
        ValueError: If fraud_probability is not in [0.0, 1.0].
        ValueError: If behaviour_score or rule_score is not in [0, 100].
    """
    # ── Input validation ──────────────────────────────────────────
    if not (0.0 <= fraud_probability <= 1.0):
        raise ValueError(
            f"fraud_probability must be in [0.0, 1.0], got {fraud_probability}"
        )
    if not (0 <= behaviour_score <= 100):
        raise ValueError(
            f"behaviour_score must be in [0, 100], got {behaviour_score}"
        )
    if not (0 <= rule_score <= 100):
        raise ValueError(
            f"rule_score must be in [0, 100], got {rule_score}"
        )

    # ── Resolve weights ───────────────────────────────────────────
    actual_w_ml = w_ml if w_ml is not None else _WEIGHT_ML
    actual_w_behaviour = w_behaviour if w_behaviour is not None else _WEIGHT_BEHAVIOUR
    actual_w_rule = w_rule if w_rule is not None else _WEIGHT_RULE

    # ── Compute ml_score ──────────────────────────────────────────
    ml_score = int(round(fraud_probability * 100))
    ml_score = max(MIN_SCORE, min(ml_score, MAX_SCORE))

    # ── Clamp component scores ────────────────────────────────────
    clamped_behaviour = max(MIN_SCORE, min(behaviour_score, MAX_SCORE))
    clamped_rule = max(MIN_SCORE, min(rule_score, MAX_SCORE))

    # ── Weighted sum ──────────────────────────────────────────────
    raw = (
        actual_w_ml * ml_score
        + actual_w_behaviour * clamped_behaviour
        + actual_w_rule * clamped_rule
    )
    risk_score = int(max(MIN_SCORE, min(round(raw), MAX_SCORE)))

    # ── Map to risk level / decision ──────────────────────────────
    risk_level, decision = _classify(risk_score)

    return RiskAssessment(
        ml_score=ml_score,
        behaviour_score=clamped_behaviour,
        rule_score=clamped_rule,
        risk_score=risk_score,
        risk_level=risk_level,
        decision=decision,
        fraud_probability=fraud_probability,
        weights={
            "w_ml": actual_w_ml,
            "w_behaviour": actual_w_behaviour,
            "w_rule": actual_w_rule,
        },
    )


# ── Internal helpers ─────────────────────────────────────────────────


def _classify(risk_score: int) -> tuple[str, str]:
    """Map a risk score to (level, decision) using the threshold table.

    Thresholds are checked from highest to lowest.
    A score of exactly 70 is MEDIUM; 71+ is HIGH.
    """
    for threshold, level, decision in RISK_THRESHOLDS:
        if risk_score > threshold:
            return level, decision
    return DEFAULT_RISK_LEVEL, DEFAULT_DECISION


def get_weights() -> dict[str, float]:
    """Return the currently configured weights (for health / diagnostics)."""
    return {
        "w_ml": _WEIGHT_ML,
        "w_behaviour": _WEIGHT_BEHAVIOUR,
        "w_rule": _WEIGHT_RULE,
    }


def get_thresholds() -> list[tuple[int, str, str]]:
    """Return the threshold table (for health / diagnostics)."""
    return list(RISK_THRESHOLDS)
