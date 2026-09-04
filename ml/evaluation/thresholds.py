"""Observational threshold analysis and recommendation strategies.

Step 47: Fraud model evaluation, calibration & threshold governance.

This module is **observational only**.  It computes how the model would
perform at different probability thresholds on an *evaluation* dataset
and produces clearly-labelled recommendations.

It must never modify:

* the production model,
* the production decision threshold,
* risk aggregation,
* live transaction decisions.

Every recommendation is labelled ``EVALUATION / RECOMMENDATION ONLY``.
Deploying a recommended threshold requires an explicit, controlled
process (retrain/re-save the bundle + manifest through the trusted
training pipeline) — never this module.

Determinism
-----------
All computations are pure.  Ties between equally optimal thresholds are
broken deterministically in favour of the **highest** threshold (the
most conservative operating point — fewest flagged transactions for the
same metric value).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ml.evaluation.config import EvaluationConfig
from ml.evaluation.metrics import (
    check_paired,
    _classification_from_counts,
    confusion_counts,
)

__all__ = [
    "ThresholdPoint",
    "Recommendation",
    "threshold_grid",
    "sweep_thresholds",
    "recommend_by_f1",
    "recommend_by_cost",
    "recommend_min_recall",
    "recommend_min_precision",
    "build_recommendations",
    "cost_curve",
    "RECOMMENDATION_DISCLAIMER",
    "STRATEGY_MAX_F1",
    "STRATEGY_MIN_COST",
    "STRATEGY_MIN_RECALL",
    "STRATEGY_MIN_PRECISION",
    "TIE_BREAK_RULE",
]


# ── Constants ─────────────────────────────────────────────────────────

RECOMMENDATION_DISCLAIMER: str = (
    "EVALUATION / RECOMMENDATION ONLY — does not modify the production "
    "model, the production threshold, risk aggregation, or live "
    "transaction decisions. Deployment requires an explicit controlled "
    "process."
)

STRATEGY_MAX_F1 = "max_f1"
STRATEGY_MIN_COST = "min_cost"
STRATEGY_MIN_RECALL = "min_recall"
STRATEGY_MIN_PRECISION = "min_precision"

TIE_BREAK_RULE: str = (
    "Ties between equally optimal thresholds are broken in favour of "
    "the highest threshold (fewest flagged transactions)."
)

_STRATEGY_DESCRIPTIONS: dict[str, str] = {
    STRATEGY_MAX_F1: (
        "Threshold that maximises F1 (harmonic mean of precision and "
        "recall) on the evaluation dataset."
    ),
    STRATEGY_MIN_COST: (
        "Threshold that minimises total business cost "
        "(FN x false_negative_cost + FP x false_positive_cost) on the "
        "evaluation dataset."
    ),
    STRATEGY_MIN_RECALL: (
        "Threshold that maximises precision subject to recall >= the "
        "configured minimum recall constraint."
    ),
    STRATEGY_MIN_PRECISION: (
        "Threshold that maximises recall subject to precision >= the "
        "configured minimum precision constraint."
    ),
}


# ── Containers ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThresholdPoint:
    """Metrics observed at one probability threshold.

    Attributes:
        threshold: Probability threshold applied (>= flags fraud).
        tp / tn / fp / fn: Confusion counts.
        precision: Precision among flagged transactions.
        recall: Fraud detection rate.
        f1: Harmonic mean of precision and recall.
        false_positive_rate: FP / (FP + TN).
        false_negative_rate: FN / (FN + TP).
        flagged_count: Number of transactions flagged as fraud.
        flagged_rate: flagged_count / n_samples.
    """

    threshold: float
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    flagged_count: int
    flagged_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Recommendation:
    """A threshold recommendation — observational only.

    When ``available`` is ``False`` the ``reason`` explains why (e.g.
    costs not configured, or no threshold satisfies the constraint).
    """

    strategy: str
    strategy_description: str
    available: bool
    reason: str | None
    threshold: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    false_positive_rate: float | None = None
    false_negative_rate: float | None = None
    flagged_count: int | None = None
    flagged_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": RECOMMENDATION_DISCLAIMER,
            **asdict(self),
        }


# ── Sweep ────────────────────────────────────────────────────────────


def threshold_grid(
    start: float,
    stop: float,
    step: float,
    *,
    max_points: int = 201,
) -> list[float]:
    """Deterministic ascending grid of thresholds in ``[start, stop]``.

    The stop value is inclusive (guarded against float drift with a
    half-step epsilon).

    Raises:
        ValueError: If the range/step are invalid or the grid would
            exceed *max_points* points (bounded output).
    """
    if not (0.0 <= start <= 1.0 and 0.0 <= stop <= 1.0):
        raise ValueError("Thresholds must lie within [0, 1].")
    if stop <= start:
        raise ValueError("stop must be greater than start.")
    if not (0.0 < step <= stop - start):
        raise ValueError("step must be positive and fit within the range.")

    grid: list[float] = []
    current = start
    while current <= stop + step / 2.0:
        grid.append(round(current, 10))
        current += step

    if len(grid) > max_points:
        raise ValueError(
            f"Threshold grid exceeds the bounded limit of {max_points} points."
        )
    return grid


def sweep_thresholds(
    y_true: Any,
    y_prob: Any,
    thresholds: list[float],
) -> list[ThresholdPoint]:
    """Evaluate the model at each threshold on the evaluation data.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Raw model fraud probabilities.
        thresholds: Thresholds to evaluate (ascending recommended).

    Returns:
        One :class:`ThresholdPoint` per threshold, in input order.
    """
    labels, probs = check_paired(y_true, y_prob)

    points: list[ThresholdPoint] = []
    for t in thresholds:
        preds = (probs >= t).astype(np.int64)
        cc = confusion_counts(labels, preds)
        derived = _classification_from_counts(cc)
        points.append(
            ThresholdPoint(
                threshold=float(t),
                tp=cc["tp"],
                tn=cc["tn"],
                fp=cc["fp"],
                fn=cc["fn"],
                precision=derived["precision"],
                recall=derived["recall"],
                f1=derived["f1"],
                false_positive_rate=derived["false_positive_rate"],
                false_negative_rate=derived["false_negative_rate"],
                flagged_count=derived["n_flagged"],
                flagged_rate=derived["flagged_rate"],
            )
        )
    return points


def cost_curve(
    points: list[ThresholdPoint],
    *,
    false_negative_cost: float,
    false_positive_cost: float,
) -> list[dict[str, Any]]:
    """Total business cost at each swept threshold.

    ``total_cost = fn * false_negative_cost + fp * false_positive_cost``
    — used for evaluation/recommendation only.
    """
    return [
        {
            "threshold": pt.threshold,
            "false_negatives": pt.fn,
            "false_positives": pt.fp,
            "total_cost": (
                pt.fn * false_negative_cost + pt.fp * false_positive_cost
            ),
        }
        for pt in points
    ]


# ── Recommendation strategies ─────────────────────────────────────────


def recommend_by_f1(points: list[ThresholdPoint]) -> Recommendation:
    """Recommend the threshold that maximises F1.

    Ties are broken by the highest threshold (see :data:`TIE_BREAK_RULE`).
    """
    if not points:
        return _unavailable(STRATEGY_MAX_F1, "No threshold sweep points provided.")

    best = points[0]
    for pt in points[1:]:
        if pt.f1 > best.f1 or (pt.f1 == best.f1 and pt.threshold > best.threshold):
            best = pt
    return _from_point(STRATEGY_MAX_F1, best)


def recommend_by_cost(
    points: list[ThresholdPoint],
    *,
    false_negative_cost: float | None,
    false_positive_cost: float | None,
) -> Recommendation:
    """Recommend the threshold minimising configured business cost.

    If either cost is not configured, the recommendation is reported
    as unavailable with an explicit reason — arbitrary default costs
    are never assumed.
    """
    if false_negative_cost is None or false_positive_cost is None:
        return _unavailable(
            STRATEGY_MIN_COST,
            "Business costs are not configured (EVAL_FN_COST / "
            "EVAL_FP_COST); cost-based threshold analysis is "
            "unavailable.",
        )
    if false_negative_cost < 0 or false_positive_cost < 0:
        return _unavailable(
            STRATEGY_MIN_COST, "Configured business costs must be non-negative."
        )
    if not points:
        return _unavailable(STRATEGY_MIN_COST, "No threshold sweep points provided.")

    def total_cost(pt: ThresholdPoint) -> float:
        return pt.fn * false_negative_cost + pt.fp * false_positive_cost

    best = points[0]
    best_cost = total_cost(best)
    for pt in points[1:]:
        c = total_cost(pt)
        if c < best_cost or (c == best_cost and pt.threshold > best.threshold):
            best = pt
            best_cost = c

    return _from_point(
        STRATEGY_MIN_COST,
        best,
        reason=(
            f"Total cost = FN x {false_negative_cost} + FP x "
            f"{false_positive_cost} = {best_cost} at the recommended "
            "threshold."
        ),
    )


def recommend_min_recall(
    points: list[ThresholdPoint],
    *,
    min_recall: float | None,
) -> Recommendation:
    """Maximise precision subject to ``recall >= min_recall``.

    Returns an unavailable recommendation when the constraint is not
    configured or when no swept threshold satisfies it.
    """
    if min_recall is None:
        return _unavailable(
            STRATEGY_MIN_RECALL,
            "Minimum recall constraint is not configured (EVAL_MIN_RECALL).",
        )
    if not points:
        return _unavailable(STRATEGY_MIN_RECALL, "No threshold sweep points provided.")

    feasible = [pt for pt in points if pt.recall >= min_recall]
    if not feasible:
        return _unavailable(
            STRATEGY_MIN_RECALL,
            f"No threshold in the sweep achieves recall >= {min_recall}.",
        )

    best = feasible[0]
    for pt in feasible[1:]:
        if (
            pt.precision > best.precision
            or (pt.precision == best.precision and pt.threshold > best.threshold)
        ):
            best = pt
    return _from_point(
        STRATEGY_MIN_RECALL,
        best,
        reason=f"Satisfies recall >= {min_recall} with maximum precision.",
    )


def recommend_min_precision(
    points: list[ThresholdPoint],
    *,
    min_precision: float | None,
) -> Recommendation:
    """Maximise recall subject to ``precision >= min_precision``.

    Returns an unavailable recommendation when the constraint is not
    configured or when no swept threshold satisfies it.
    """
    if min_precision is None:
        return _unavailable(
            STRATEGY_MIN_PRECISION,
            "Minimum precision constraint is not configured (EVAL_MIN_PRECISION).",
        )
    if not points:
        return _unavailable(STRATEGY_MIN_PRECISION, "No threshold sweep points provided.")

    feasible = [pt for pt in points if pt.precision >= min_precision]
    if not feasible:
        return _unavailable(
            STRATEGY_MIN_PRECISION,
            f"No threshold in the sweep achieves precision >= {min_precision}.",
        )

    best = feasible[0]
    for pt in feasible[1:]:
        if pt.recall > best.recall or (
            pt.recall == best.recall and pt.threshold > best.threshold
        ):
            best = pt
    return _from_point(
        STRATEGY_MIN_PRECISION,
        best,
        reason=f"Satisfies precision >= {min_precision} with maximum recall.",
    )


def build_recommendations(
    points: list[ThresholdPoint],
    config: EvaluationConfig,
) -> list[Recommendation]:
    """Run all configured strategies and return their recommendations."""
    return [
        recommend_by_f1(points),
        recommend_by_cost(
            points,
            false_negative_cost=config.false_negative_cost,
            false_positive_cost=config.false_positive_cost,
        ),
        recommend_min_recall(points, min_recall=config.min_recall),
        recommend_min_precision(points, min_precision=config.min_precision),
    ]


# ── Internal helpers ──────────────────────────────────────────────────


def _from_point(
    strategy: str,
    pt: ThresholdPoint,
    *,
    reason: str | None = None,
) -> Recommendation:
    """Build an available recommendation from a sweep point."""
    return Recommendation(
        strategy=strategy,
        strategy_description=_STRATEGY_DESCRIPTIONS[strategy],
        available=True,
        reason=reason,
        threshold=pt.threshold,
        precision=pt.precision,
        recall=pt.recall,
        f1=pt.f1,
        false_positive_rate=pt.false_positive_rate,
        false_negative_rate=pt.false_negative_rate,
        flagged_count=pt.flagged_count,
        flagged_rate=pt.flagged_rate,
    )


def _unavailable(strategy: str, reason: str) -> Recommendation:
    """Build an unavailable recommendation with an explicit reason."""
    return Recommendation(
        strategy=strategy,
        strategy_description=_STRATEGY_DESCRIPTIONS[strategy],
        available=False,
        reason=reason,
    )
