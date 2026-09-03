"""Offline fraud-model evaluation, calibration & threshold governance.

Step 47.  This package is **observational**: it measures how the
verified production model performs on held-out evaluation data,
analyses alternative decision thresholds, and produces clearly-labelled
recommendations.

It never modifies the production model, the production threshold,
risk aggregation, live transaction decisions, monitoring counters, or
the fraud-decision audit trail.
"""

from ml.evaluation.config import (
    EvaluationConfig,
    EvaluationConfigError,
)
from ml.evaluation.metrics import (
    LabelError,
    ProbabilityError,
    RankingError,
    brier_score,
    calibration_metrics,
    classification_metrics,
    confusion_counts,
    ranking_metrics,
    reliability_bins,
    validate_labels,
    validate_probabilities,
)
from ml.evaluation.promotion_policy import (
    PromotionPolicy,
    PromotionPolicyError,
)
from ml.evaluation.thresholds import (
    Recommendation,
    ThresholdPoint,
    build_recommendations,
    cost_curve,
    recommend_by_cost,
    recommend_by_f1,
    recommend_min_precision,
    recommend_min_recall,
    sweep_thresholds,
    threshold_grid,
)

__all__ = [
    # config
    "EvaluationConfig",
    "EvaluationConfigError",
    # metrics
    "LabelError",
    "ProbabilityError",
    "RankingError",
    "brier_score",
    "calibration_metrics",
    "classification_metrics",
    "confusion_counts",
    "ranking_metrics",
    "reliability_bins",
    "validate_labels",
    "validate_probabilities",
    # promotion policy (Step 48)
    "PromotionPolicy",
    "PromotionPolicyError",
    # thresholds
    "Recommendation",
    "ThresholdPoint",
    "build_recommendations",
    "cost_curve",
    "recommend_by_cost",
    "recommend_by_f1",
    "recommend_min_precision",
    "recommend_min_recall",
    "sweep_thresholds",
    "threshold_grid",
]
