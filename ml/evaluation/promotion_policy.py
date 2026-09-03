"""Promotion-gate policy configuration (PROMO_* environment variables).

Step 48: Automated model validation & promotion gate.

The promotion policy defines **when a candidate model may be approved**
by the offline promotion gate.  It distinguishes two kinds of gates:

* **Absolute minimum requirements** — the candidate must reach a fixed
  quality floor on its own (e.g. ``PR-AUC >= 0.10``).
* **Relative regression limits** — the candidate may not degrade more
  than a configured fraction *relative to the current production model*
  (e.g. ``F1 >= production F1 x (1 - 0.10)``), and the Brier score may
  not increase more than a configured fraction
  (``Brier <= production Brier x (1 + 0.10)``).

All values are **gate-local configuration**.  They never modify the
production model, the production decision threshold, risk aggregation,
or live transaction decisions — they only decide whether the promotion
gate answers ``APPROVED`` or ``REJECTED``.

Degradation limits are fractions in ``[0, 1]``: ``0.10`` means "the
candidate may be at most 10 % worse than production for this metric".
A limit of ``0.0`` forbids any degradation.

Boundary semantics
------------------
Gates are inclusive at the boundary: a candidate exactly at a required
minimum, exactly at a degradation limit, or exactly at a Brier ceiling
**passes**.  The gate applies a tolerance of ``1e-9`` when comparing so
that boundary cases are robust to floating-point rounding.

Environment variables
---------------------
Each field maps to a ``PROMO_*`` variable.  Unset/empty falls back to
the documented default; an explicit ``none`` (or ``off``) disables that
gate; a numeric value configures it.  Values outside the valid range
raise :class:`PromotionPolicyError` (the gate then fails closed).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "PromotionPolicy",
    "PromotionPolicyError",
    "DEFAULT_MIN_PR_AUC",
    "DEFAULT_MIN_ROC_AUC",
    "DEFAULT_MIN_RECALL",
    "DEFAULT_MIN_PRECISION",
    "DEFAULT_MIN_F1",
    "DEFAULT_MAX_BRIER",
    "DEFAULT_MAX_PR_AUC_DEGRADATION",
    "DEFAULT_MAX_ROC_AUC_DEGRADATION",
    "DEFAULT_MAX_RECALL_DEGRADATION",
    "DEFAULT_MAX_PRECISION_DEGRADATION",
    "DEFAULT_MAX_F1_DEGRADATION",
    "DEFAULT_MAX_BRIER_INCREASE",
]


class PromotionPolicyError(ValueError):
    """Promotion policy configuration is invalid."""


# ── Defaults ───────────────────────────────────────────────────────────
#
# Defaults are deliberately conservative but satisfiable by a model of
# similar quality to the current production model, so that promoting an
# unchanged/identical model is APPROVED while a materially worse one is
# REJECTED.  Every value is explicitly configurable.

DEFAULT_MIN_PR_AUC: float = 0.10
DEFAULT_MIN_ROC_AUC: float = 0.75
DEFAULT_MIN_RECALL: float = 0.70
DEFAULT_MIN_PRECISION: float = 0.05
DEFAULT_MIN_F1: float = 0.10
DEFAULT_MAX_BRIER: float = 0.25

DEFAULT_MAX_PR_AUC_DEGRADATION: float = 0.05
DEFAULT_MAX_ROC_AUC_DEGRADATION: float = 0.02
DEFAULT_MAX_RECALL_DEGRADATION: float = 0.10
DEFAULT_MAX_PRECISION_DEGRADATION: float = 0.10
DEFAULT_MAX_F1_DEGRADATION: float = 0.10
DEFAULT_MAX_BRIER_INCREASE: float = 0.10


# ── Policy container ──────────────────────────────────────────────────


@dataclass(frozen=True)
class PromotionPolicy:
    """Configurable gates for the offline promotion gate.

    Absolute minimum requirements (``None`` = gate disabled):

    * ``min_pr_auc`` / ``min_roc_auc`` — ranking quality floors.
    * ``min_recall`` / ``min_precision`` / ``min_f1`` — classification
      floors at the model's own bundled production threshold.
    * ``max_brier`` — maximum calibration error.

    Relative regression limits vs production (``None`` = gate
    disabled; fractions in ``[0, 1]``):

    * ``max_<metric>_degradation`` — the candidate metric may not fall
      more than this fraction below the production metric
      (higher-is-better metrics).
    * ``max_brier_increase`` — the candidate Brier score may not rise
      more than this fraction above the production Brier score
      (lower-is-better metric).

    Changing these values has **no** effect on production inference,
    the production threshold, or the active model.
    """

    min_pr_auc: float | None = DEFAULT_MIN_PR_AUC
    min_roc_auc: float | None = DEFAULT_MIN_ROC_AUC
    min_recall: float | None = DEFAULT_MIN_RECALL
    min_precision: float | None = DEFAULT_MIN_PRECISION
    min_f1: float | None = DEFAULT_MIN_F1
    max_brier: float | None = DEFAULT_MAX_BRIER

    max_pr_auc_degradation: float | None = DEFAULT_MAX_PR_AUC_DEGRADATION
    max_roc_auc_degradation: float | None = DEFAULT_MAX_ROC_AUC_DEGRADATION
    max_recall_degradation: float | None = DEFAULT_MAX_RECALL_DEGRADATION
    max_precision_degradation: float | None = DEFAULT_MAX_PRECISION_DEGRADATION
    max_f1_degradation: float | None = DEFAULT_MAX_F1_DEGRADATION
    max_brier_increase: float | None = DEFAULT_MAX_BRIER_INCREASE

    # ── Validation ────────────────────────────────────────────────

    def validate(self) -> None:
        """Raise :class:`PromotionPolicyError` on invalid settings."""
        for name, value in (
            ("min_pr_auc", self.min_pr_auc),
            ("min_roc_auc", self.min_roc_auc),
            ("min_recall", self.min_recall),
            ("min_precision", self.min_precision),
            ("min_f1", self.min_f1),
            ("max_brier", self.max_brier),
        ):
            if value is not None and not (0.0 <= value <= 1.0):
                raise PromotionPolicyError(
                    f"{name} must be within [0, 1] when configured, got {value}"
                )
        for name, value in (
            ("max_pr_auc_degradation", self.max_pr_auc_degradation),
            ("max_roc_auc_degradation", self.max_roc_auc_degradation),
            ("max_recall_degradation", self.max_recall_degradation),
            ("max_precision_degradation", self.max_precision_degradation),
            ("max_f1_degradation", self.max_f1_degradation),
            ("max_brier_increase", self.max_brier_increase),
        ):
            if value is not None and not (0.0 <= value <= 1.0):
                raise PromotionPolicyError(
                    f"{name} must be within [0, 1] when configured "
                    f"(fraction of the production value), got {value}"
                )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation for the promotion report."""
        return asdict(self)

    # ── Construction ───────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> PromotionPolicy:
        """Build a policy from ``PROMO_*`` environment variables.

        Unset/empty variables fall back to the documented defaults.
        An explicit ``none`` (or ``off``) disables that gate entirely.
        Invalid values raise :class:`PromotionPolicyError` naming the
        offending variable — the gate then fails closed.
        """
        return cls(
            min_pr_auc=_policy_env("PROMO_MIN_PR_AUC", DEFAULT_MIN_PR_AUC),
            min_roc_auc=_policy_env("PROMO_MIN_ROC_AUC", DEFAULT_MIN_ROC_AUC),
            min_recall=_policy_env("PROMO_MIN_RECALL", DEFAULT_MIN_RECALL),
            min_precision=_policy_env("PROMO_MIN_PRECISION", DEFAULT_MIN_PRECISION),
            min_f1=_policy_env("PROMO_MIN_F1", DEFAULT_MIN_F1),
            max_brier=_policy_env("PROMO_MAX_BRIER", DEFAULT_MAX_BRIER),
            max_pr_auc_degradation=_policy_env(
                "PROMO_MAX_PR_AUC_DEGRADATION", DEFAULT_MAX_PR_AUC_DEGRADATION
            ),
            max_roc_auc_degradation=_policy_env(
                "PROMO_MAX_ROC_AUC_DEGRADATION", DEFAULT_MAX_ROC_AUC_DEGRADATION
            ),
            max_recall_degradation=_policy_env(
                "PROMO_MAX_RECALL_DEGRADATION", DEFAULT_MAX_RECALL_DEGRADATION
            ),
            max_precision_degradation=_policy_env(
                "PROMO_MAX_PRECISION_DEGRADATION", DEFAULT_MAX_PRECISION_DEGRADATION
            ),
            max_f1_degradation=_policy_env(
                "PROMO_MAX_F1_DEGRADATION", DEFAULT_MAX_F1_DEGRADATION
            ),
            max_brier_increase=_policy_env(
                "PROMO_MAX_BRIER_INCREASE", DEFAULT_MAX_BRIER_INCREASE
            ),
        )


# ── Environment parsing helper ────────────────────────────────────────


def _policy_env(name: str, default: float | None) -> float | None:
    """Parse one PROMO_* variable.

    Empty → default; ``none``/``off`` → disabled (``None``); otherwise a
    float.  Non-numeric garbage raises a policy error naming the
    variable.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    if raw.lower() in ("none", "off"):
        return None
    try:
        return float(raw)
    except ValueError:
        raise PromotionPolicyError(
            f"{name} must be a number (or 'none'/'off' to disable the "
            f"gate), got {raw!r}"
        ) from None
