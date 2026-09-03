"""Core evaluation metrics — classification, ranking, and calibration.

Step 47: Fraud model evaluation, calibration & threshold governance.

All functions are **pure** and deterministic: identical inputs always
produce identical outputs.  They operate on labels and probabilities
only — never on raw transactions — so evaluation outputs contain no
customer-level information.

Label handling
--------------
Required labels must be present, numeric, and binary (0/1).  Missing
(``NaN``), non-numeric, or non-binary labels raise :class:`LabelError`
— metrics are **never** silently computed against invalid labels.

Probability handling
--------------------
Probabilities must be finite and within ``[0, 1]``.  Violations raise
:class:`ProbabilityError`.

Degenerate labels (all-positive / all-negative)
-----------------------------------------------
Confusion-matrix–style classification metrics remain well defined and
are computed with documented zero-division conventions (a ratio with a
zero denominator is reported as ``0.0``).  Ranking metrics (ROC-AUC,
PR-AUC) require both classes and raise :class:`RankingError` with a
clear message instead of returning a meaningless number.

Imbalance guidance
------------------
Fraud data is heavily imbalanced, so ``accuracy`` alone is misleading.
:class:`classification_metrics` always reports precision, recall, F1,
false-positive rate, false-negative rate, and fraud prevalence
alongside accuracy; use PR-AUC (:func:`ranking_metrics`) rather than
accuracy to summarise ranking quality.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

__all__ = [
    "EvaluationError",
    "LabelError",
    "ProbabilityError",
    "RankingError",
    "validate_labels",
    "validate_probabilities",
    "check_paired",
    "confusion_counts",
    "classification_metrics",
    "ranking_metrics",
    "brier_score",
    "reliability_bins",
    "calibration_metrics",
]


# ── Exceptions ─────────────────────────────────────────────────────────


class EvaluationError(Exception):
    """Base class for evaluation errors."""


class LabelError(EvaluationError):
    """Labels are missing, non-numeric, non-binary, or empty."""


class ProbabilityError(EvaluationError):
    """Probabilities are missing, non-finite, out of bounds, or misaligned."""


class RankingError(EvaluationError):
    """Ranking metrics cannot be computed for the given labels."""


# ── Validation ─────────────────────────────────────────────────────────


def validate_labels(y: Any) -> np.ndarray:
    """Validate and normalise a label vector to ``int64`` binary values.

    Args:
        y: Labels as any array-like (list, Series, ndarray).

    Returns:
        ``int64`` ndarray containing only 0/1.

    Raises:
        LabelError: If *y* is empty, non-numeric, contains missing
            values, or contains values other than 0/1.
    """
    arr = np.asarray(y)
    if arr.size == 0:
        raise LabelError("Labels are empty — evaluation requires labels.")

    try:
        arr = arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise LabelError(
            "Labels contain non-numeric values — evaluation requires "
            "binary (0/1) labels."
        ) from exc

    if not np.all(np.isfinite(arr)):
        raise LabelError(
            "Labels contain missing or non-finite values (NaN/inf) — "
            "evaluation requires complete labels."
        )

    unique = np.unique(arr)
    if not np.all(np.isin(unique, (0.0, 1.0))):
        raise LabelError(
            "Labels must be binary (0/1) — found values: "
            f"{sorted(float(v) for v in unique)}"
        )

    return arr.astype(np.int64)


def validate_probabilities(p: Any) -> np.ndarray:
    """Validate and normalise a probability vector to ``float64``.

    Args:
        p: Predicted fraud probabilities (array-like).

    Returns:
        ``float64`` ndarray with values in ``[0, 1]``.

    Raises:
        ProbabilityError: If *p* is empty, non-numeric, non-finite,
            or outside ``[0, 1]``.
    """
    arr = np.asarray(p)
    if arr.size == 0:
        raise ProbabilityError(
            "Predicted probabilities are empty — evaluation requires "
            "model probabilities."
        )

    try:
        arr = arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ProbabilityError(
            "Predicted probabilities contain non-numeric values."
        ) from exc

    if not np.all(np.isfinite(arr)):
        raise ProbabilityError(
            "Predicted probabilities contain missing or non-finite "
            "values (NaN/inf)."
        )

    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ProbabilityError(
            "Predicted probabilities must be within [0, 1] — found "
            f"min={float(arr.min()):.6g}, max={float(arr.max()):.6g}"
        )

    return arr


def check_paired(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate labels and probabilities and check length alignment."""
    labels = validate_labels(y_true)
    probs = validate_probabilities(y_prob)
    if len(labels) != len(probs):
        raise ProbabilityError(
            f"Labels ({len(labels)}) and probabilities ({len(probs)}) "
            "must have the same length."
        )
    return labels, probs


# ── Confusion counts ──────────────────────────────────────────────────


def _make_confusion(tp: int, tn: int, fp: int, fn: int) -> dict[str, int]:
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def confusion_counts(y_true: Any, y_pred: Any) -> dict[str, int]:
    """Compute confusion counts for binary fraud labels.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels (0/1) — already thresholded.

    Returns:
        Dict with keys ``tp``, ``tn``, ``fp``, ``fn`` (fraud is the
        positive class, label 1).

    Raises:
        LabelError: If either vector has invalid labels.
        ProbabilityError: If the vectors differ in length.
    """
    truth = validate_labels(y_true)
    pred = validate_labels(y_pred)
    if len(truth) != len(pred):
        raise ProbabilityError(
            f"y_true ({len(truth)}) and y_pred ({len(pred)}) must have "
            "the same length."
        )

    tp = int(np.sum((truth == 1) & (pred == 1)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    return _make_confusion(tp, tn, fp, fn)


# ── Classification metrics ────────────────────────────────────────────


def classification_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Full classification metrics for fraud detection.

    Returns a dict containing:

    * ``tp`` / ``tn`` / ``fp`` / ``fn``
    * ``confusion_matrix`` — ``[[tn, fp], [fn, tp]]`` (sklearn layout)
    * ``n_samples`` / ``n_fraud`` / ``n_legitimate`` / ``n_flagged``
    * ``accuracy`` — (TP+TN)/N (misleading under imbalance; reported
      for completeness only)
    * ``precision`` — precision **among flagged transactions**
      (TP/(TP+FP); ``0.0`` when nothing is flagged)
    * ``recall`` — fraud detection rate (TP/(TP+FN); ``0.0`` when
      there is no fraud)
    * ``f1`` — harmonic mean of precision and recall (``0.0`` when
      both are zero)
    * ``false_positive_rate`` — FP/(FP+TN); ``0.0`` when there are no
      legitimate transactions
    * ``false_negative_rate`` — FN/(FN+TP); ``0.0`` when there is no
      fraud
    * ``fraud_prevalence`` — (TP+FN)/N
    * ``flagged_rate`` — (TP+FP)/N

    All zero-division cases are reported as ``0.0`` by documented
    convention so the function stays deterministic for degenerate
    label sets.
    """
    cc = confusion_counts(y_true, y_pred)
    return _classification_from_counts(cc)


def _classification_from_counts(cc: dict[str, int]) -> dict[str, Any]:
    """Compute derived metrics from validated confusion counts."""
    tp, tn, fp, fn = cc["tp"], cc["tn"], cc["fp"], cc["fn"]
    n = tp + tn + fp + fn

    n_fraud = tp + fn
    n_legit = tn + fp
    n_flagged = tp + fp

    precision = tp / n_flagged if n_flagged > 0 else 0.0
    recall = tp / n_fraud if n_fraud > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "n_samples": n,
        "n_fraud": n_fraud,
        "n_legitimate": n_legit,
        "n_flagged": n_flagged,
        "accuracy": (tp + tn) / n if n > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / n_legit if n_legit > 0 else 0.0,
        "false_negative_rate": fn / n_fraud if n_fraud > 0 else 0.0,
        "fraud_detection_rate": recall,
        "fraud_prevalence": n_fraud / n if n > 0 else 0.0,
        "precision_among_flagged": precision,
        "flagged_rate": n_flagged / n if n > 0 else 0.0,
    }


# ── Ranking metrics ───────────────────────────────────────────────────


def ranking_metrics(y_true: Any, y_prob: Any) -> dict[str, Any]:
    """Threshold-independent ranking quality (ROC-AUC and PR-AUC).

    Returns:
        Dict with ``roc_auc`` (area under the ROC curve) and
        ``pr_auc`` (average precision — area under the precision-recall
        curve; the appropriate ranking metric for imbalanced data).

    Raises:
        LabelError / ProbabilityError: For invalid inputs.
        RankingError: If the labels contain only one class — both
            metrics are undefined in that case and this fails clearly
            instead of returning a misleading value.
    """
    labels, probs = check_paired(y_true, y_prob)

    if len(np.unique(labels)) < 2:
        raise RankingError(
            "Ranking metrics (ROC-AUC, PR-AUC) require both fraud and "
            "legitimate samples; the evaluation labels contain a "
            "single class."
        )

    roc_auc = float(roc_auc_score(labels, probs))
    pr_auc = float(average_precision_score(labels, probs))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "pr_auc_definition": "average_precision",
        "n_samples": int(len(labels)),
    }


# ── Calibration ───────────────────────────────────────────────────────


def brier_score(y_true: Any, y_prob: Any) -> float:
    """Brier score — mean squared error of predicted probabilities.

    ``0.0`` is perfect; lower is better.  Computed on **raw** model
    probabilities (no recalibration).
    """
    labels, probs = check_paired(y_true, y_prob)
    return float(brier_score_loss(labels, probs))


def reliability_bins(
    y_true: Any,
    y_prob: Any,
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Reliability-diagram bins over ``[0, 1]`` (uniform width).

    Each returned bin dict has: ``bin_index``, ``bin_lower``,
    ``bin_upper``, ``count``, ``mean_predicted_probability`` and
    ``fraction_positive``.  Empty bins are omitted.  Bin assignment
    matches ``sklearn.calibration.calibration_curve`` with
    ``strategy="uniform"`` exactly: a probability that falls on an
    interior bin edge belongs to the bin below it, and ``1.0``
    belongs to the final bin.  Counts are included for
    interpretability.

    Raises:
        ValueError: If ``n_bins`` is out of the supported range.
    """
    if not (2 <= n_bins <= 50):
        raise ValueError(f"n_bins must be between 2 and 50, got {n_bins}")

    labels, probs = check_paired(y_true, y_prob)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # sklearn's uniform-strategy convention (calibration_curve):
    # bin id = number of interior edges strictly below the value, so
    # a probability on an interior edge belongs to the bin below it
    # and 1.0 belongs to the final bin.
    bin_ids = np.searchsorted(edges[1:-1], probs)
    bins: list[dict[str, Any]] = []

    for i in range(n_bins):
        mask = bin_ids == i
        count = int(np.sum(mask))
        if count == 0:
            continue

        bins.append(
            {
                "bin_index": i,
                "bin_lower": float(edges[i]),
                "bin_upper": float(edges[i + 1]),
                "count": count,
                "mean_predicted_probability": float(probs[mask].mean()),
                "fraction_positive": float(labels[mask].mean()),
            }
        )

    return bins


def calibration_metrics(
    y_true: Any,
    y_prob: Any,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Calibration assessment for **raw** model probabilities.

    Reports the Brier score and reliability bins.  This is an
    evaluation-only observation: no recalibration model (Platt /
    isotonic) is fitted, applied, or returned.  A calibrated
    probability would be a separate offline artifact requiring
    explicit approval — it must never silently replace the live
    model probability.
    """
    if not (2 <= n_bins <= 50):
        raise ValueError(f"n_bins must be between 2 and 50, got {n_bins}")

    labels, probs = check_paired(y_true, y_prob)

    return {
        "brier_score": brier_score(labels, probs),
        "n_bins": n_bins,
        "reliability_bins": reliability_bins(labels, probs, n_bins=n_bins),
        "probability_type": "raw_model_probability",
        "recalibration_applied": False,
        "note": (
            "Computed on RAW model probabilities. No recalibration is "
            "applied; calibrated probabilities are not produced by this "
            "evaluation."
        ),
    }
