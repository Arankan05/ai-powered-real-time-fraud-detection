"""Evaluation-only configuration (EVAL_* environment variables).

Step 47: Fraud model evaluation, calibration & threshold governance.

These parameters configure **offline evaluation only**.  They are
deliberately namespaced ``EVAL_*`` so they can never be confused with
live production configuration such as the model decision threshold
(bundled in the model artifact / manifest) or the risk-aggregation
thresholds (``ML_WEIGHT_*`` in :mod:`ml.risk.aggregator`).

Evaluation configuration **never** overrides the live fraud threshold.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "EvaluationConfig",
    "EvaluationConfigError",
    "DEFAULT_THRESHOLD_START",
    "DEFAULT_THRESHOLD_STOP",
    "DEFAULT_THRESHOLD_STEP",
    "DEFAULT_CALIBRATION_BINS",
    "MAX_THRESHOLD_POINTS",
]


class EvaluationConfigError(ValueError):
    """Evaluation configuration is invalid."""


# ── Defaults ───────────────────────────────────────────────────────────

DEFAULT_THRESHOLD_START: float = 0.05
DEFAULT_THRESHOLD_STOP: float = 0.95
DEFAULT_THRESHOLD_STEP: float = 0.05
DEFAULT_CALIBRATION_BINS: int = 10

# Hard upper bound on the number of sweep points so evaluation output
# stays bounded regardless of configuration.
MAX_THRESHOLD_POINTS: int = 201


# ── Configuration container ───────────────────────────────────────────


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for offline model evaluation.

    Attributes:
        threshold_start: First probability threshold in the sweep.
        threshold_stop: Last probability threshold in the sweep
            (inclusive).
        threshold_step: Step between sweep thresholds.
        min_recall: Optional minimum-recall constraint for the
            recommendation strategy (``None`` = not configured).
        min_precision: Optional minimum-precision constraint for the
            recommendation strategy (``None`` = not configured).
        false_negative_cost: Optional business cost of one missed fraud
            (``None`` = cost analysis unavailable).
        false_positive_cost: Optional business cost of one falsely
            flagged legitimate transaction (``None`` = cost analysis
            unavailable).
        calibration_bins: Number of reliability-diagram bins.

    All values are evaluation-only.  Changing them has **no** effect on
    production inference, the production threshold, or risk
    aggregation.
    """

    threshold_start: float = DEFAULT_THRESHOLD_START
    threshold_stop: float = DEFAULT_THRESHOLD_STOP
    threshold_step: float = DEFAULT_THRESHOLD_STEP
    min_recall: float | None = None
    min_precision: float | None = None
    false_negative_cost: float | None = None
    false_positive_cost: float | None = None
    calibration_bins: int = DEFAULT_CALIBRATION_BINS

    # ── Validation ────────────────────────────────────────────────

    def validate(self) -> None:
        """Raise :class:`EvaluationConfigError` on invalid settings."""
        if not (0.0 <= self.threshold_start <= 1.0):
            raise EvaluationConfigError(
                f"threshold_start must be within [0, 1], got {self.threshold_start}"
            )
        if not (0.0 <= self.threshold_stop <= 1.0):
            raise EvaluationConfigError(
                f"threshold_stop must be within [0, 1], got {self.threshold_stop}"
            )
        if self.threshold_stop <= self.threshold_start:
            raise EvaluationConfigError(
                "threshold_stop must be greater than threshold_start "
                f"(start={self.threshold_start}, stop={self.threshold_stop})"
            )
        if not (0.0 < self.threshold_step <= self.threshold_stop - self.threshold_start):
            raise EvaluationConfigError(
                "threshold_step must be positive and no larger than the "
                f"sweep range (got step={self.threshold_step}, "
                f"range={self.threshold_stop - self.threshold_start})"
            )
        if len(self.threshold_grid()) > MAX_THRESHOLD_POINTS:
            raise EvaluationConfigError(
                "threshold sweep exceeds the bounded limit of "
                f"{MAX_THRESHOLD_POINTS} points"
            )
        for name, value in (
            ("min_recall", self.min_recall),
            ("min_precision", self.min_precision),
        ):
            if value is not None and not (0.0 <= value <= 1.0):
                raise EvaluationConfigError(
                    f"{name} must be within [0, 1] when configured, got {value}"
                )
        for name, value in (
            ("false_negative_cost", self.false_negative_cost),
            ("false_positive_cost", self.false_positive_cost),
        ):
            if value is not None and value < 0.0:
                raise EvaluationConfigError(
                    f"{name} must be non-negative when configured, got {value}"
                )
        if not (2 <= self.calibration_bins <= 50):
            raise EvaluationConfigError(
                "calibration_bins must be between 2 and 50, "
                f"got {self.calibration_bins}"
            )

    # ── Derived values ────────────────────────────────────────────

    def threshold_grid(self) -> list[float]:
        """Deterministic ascending grid of sweep thresholds.

        The stop value is inclusive (subject to float rounding, which is
        guarded with a half-step epsilon).
        """
        grid: list[float] = []
        current = self.threshold_start
        while current <= self.threshold_stop + self.threshold_step / 2.0:
            grid.append(round(current, 10))
            current += self.threshold_step
        return grid

    def costs_configured(self) -> bool:
        """Whether both business costs are configured."""
        return (
            self.false_negative_cost is not None
            and self.false_positive_cost is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation for reproducibility metadata."""
        return asdict(self)

    # ── Construction ───────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> EvaluationConfig:
        """Build a configuration from ``EVAL_*`` environment variables.

        Unset/empty variables fall back to defaults (constraints and
        costs stay unconfigured).  Invalid values raise
        :class:`EvaluationConfigError` naming the offending variable.
        """
        return cls(
            threshold_start=_env_float(
                "EVAL_THRESHOLD_START", DEFAULT_THRESHOLD_START
            ),
            threshold_stop=_env_float(
                "EVAL_THRESHOLD_STOP", DEFAULT_THRESHOLD_STOP
            ),
            threshold_step=_env_float(
                "EVAL_THRESHOLD_STEP", DEFAULT_THRESHOLD_STEP
            ),
            min_recall=_env_optional_float("EVAL_MIN_RECALL"),
            min_precision=_env_optional_float("EVAL_MIN_PRECISION"),
            false_negative_cost=_env_optional_float("EVAL_FN_COST"),
            false_positive_cost=_env_optional_float("EVAL_FP_COST"),
            calibration_bins=_env_int(
                "EVAL_CALIBRATION_BINS", DEFAULT_CALIBRATION_BINS
            ),
        )


# ── Environment parsing helpers ────────────────────────────────────────


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise EvaluationConfigError(
            f"{name} must be a number, got {raw!r}"
        ) from None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise EvaluationConfigError(
            f"{name} must be an integer, got {raw!r}"
        ) from None


def _env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise EvaluationConfigError(
            f"{name} must be a number, got {raw!r}"
        ) from None
