"""Risk aggregation layer — combines ML, behaviour, and rule scores.

This package provides the final fraud risk score computation described
in ``docs/ml-architecture.md`` §5.

Usage::

    from ml.risk import aggregate_risk, RiskAssessment

    assessment = aggregate_risk(
        fraud_probability=0.72,
        behaviour_score=60,
        rule_score=35,
    )
"""

from ml.risk.aggregator import (
    RiskAssessment,
    aggregate_risk,
    get_thresholds,
    get_weights,
)

__all__ = [
    "RiskAssessment",
    "aggregate_risk",
    "get_thresholds",
    "get_weights",
]
