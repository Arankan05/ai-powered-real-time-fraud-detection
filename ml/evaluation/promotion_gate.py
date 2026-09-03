"""Automated model validation & promotion gate (Step 48).

Offline, production-safe candidate-model validation built on the Step 46
model-governance pipeline and the Step 47 evaluation framework.

What this gate does
-------------------
1. Validates the candidate model through the **full Step 46 governance
   sequence** (manifest → SHA-256 checksum → bundle load → interface
   validation → feature-schema/count compatibility) using a *scratch*
   :class:`~ml.predict.registry.ModelRegistry` instance.  The candidate is
   never activated as, or swapped into, the production service.
2. Activates the **current production model** through the same verified
   governance path (the configured trusted model directory — never a
   caller-claimed identity).
3. Evaluates both models on the **same held-out evaluation dataset**
   through the Step 47 evaluation framework (``build_report``).
4. Compares the candidate against a configurable promotion policy
   (:class:`~ml.evaluation.promotion_policy.PromotionPolicy`) with
   absolute minimum requirements and relative regression limits, and
   answers exactly ``APPROVED`` or ``REJECTED``.

What this gate never does
-------------------------
* It never modifies the active production manifest, artifact, threshold,
  or runtime model (no activation, no hot-swap, no rollback).
* It never changes the production fraud decision threshold.
* ``APPROVED`` means "passed every configured policy gate on the
  evaluation dataset" — promotion itself remains an **explicit,
  controlled operator action** through the existing Step 46 governance
  workflow.

Fail-closed behaviour
---------------------
The gate answers ``REJECTED`` whenever validation is incomplete: a
missing candidate artifact, an invalid checksum, a malformed manifest,
a failed evaluation, metrics that cannot be calculated, an unavailable
production baseline, or an invalid policy/evaluation configuration.  It
never answers ``APPROVED`` unless every configured gate was actually
evaluated *and* passed.

Usage::

    python -m ml.evaluation.promotion_gate --candidate-model-dir <dir> \\
        [--output PATH]

Exit codes: ``0`` approved, ``1`` rejected (including fail-closed
validation failures), ``2`` unexpected internal error.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ml.evaluation.config import EvaluationConfig, EvaluationConfigError
from ml.evaluation.metrics import LabelError, ProbabilityError
from ml.evaluation.promotion_policy import PromotionPolicy, PromotionPolicyError
from ml.evaluation.runner import (
    DATASET_IDENTIFIER,
    EvaluationError,
    EvaluationReport,
    build_report,
    load_holdout_test_set,
)
from ml.predict.integrity import _MANIFEST_FILENAME, default_model_directory
from ml.predict.registry import ActivationError, ModelRegistry

logger = logging.getLogger(__name__)

__all__ = [
    "PromotionDecision",
    "GateOutcome",
    "GATE_REPORT_SCOPE",
    "GATE_DISCLAIMER",
    "DECISION_APPROVED",
    "DECISION_REJECTED",
    "REPORT_SCHEMA_VERSION",
    "evaluate_promotion",
    "run_promotion_gate",
    "main",
]


# ── Constants ─────────────────────────────────────────────────────────


class PromotionGateError(Exception):
    """The promotion gate could not complete safely (fail-closed)."""


GATE_REPORT_SCOPE: str = "offline_model_promotion_gate"
GATE_DISCLAIMER: str = (
    "OFFLINE PROMOTION GATE — EVALUATION ONLY. APPROVED means the "
    "candidate passed every configured policy gate on the evaluation "
    "dataset; it does NOT activate the candidate, modify the production "
    "manifest, change the production threshold, or hot-swap the runtime "
    "model. Promotion is an explicit operator action through the "
    "existing Step 46 governance workflow."
)
DECISION_APPROVED: str = "APPROVED"
DECISION_REJECTED: str = "REJECTED"
REPORT_SCHEMA_VERSION: str = "1.0.0"

# Inclusive boundary tolerance: a candidate exactly at a required
# minimum, degradation limit, or Brier ceiling passes.
_BOUNDARY_EPSILON: float = 1e-9

# Absolute minimum requirements: (policy field, metric, comparison).
_ABSOLUTE_GATE_DEFS: tuple[tuple[str, str, str], ...] = (
    ("min_pr_auc", "pr_auc", ">="),
    ("min_roc_auc", "roc_auc", ">="),
    ("min_recall", "recall", ">="),
    ("min_precision", "precision", ">="),
    ("min_f1", "f1", ">="),
    ("max_brier", "brier_score", "<="),
)

# Relative regression limits vs production:
# (policy field, metric, sense) where sense is "floor" (higher-is-better)
# or "ceiling" (lower-is-better, Brier).
_RELATIVE_GATE_DEFS: tuple[tuple[str, str, str], ...] = (
    ("max_pr_auc_degradation", "pr_auc", "floor"),
    ("max_roc_auc_degradation", "roc_auc", "floor"),
    ("max_recall_degradation", "recall", "floor"),
    ("max_precision_degradation", "precision", "floor"),
    ("max_f1_degradation", "f1", "floor"),
    ("max_brier_increase", "brier_score", "ceiling"),
)


# ── Result containers ─────────────────────────────────────────────────


@dataclass(frozen=True)
class GateOutcome:
    """Result of one configured policy gate.

    Attributes:
        gate: Policy field name (e.g. ``min_pr_auc``).
        kind: ``"absolute_requirement"`` or ``"relative_regression_limit"``.
        metric: Metric the gate applies to.
        comparison: ``">="`` or ``"<="``.
        candidate_value: Candidate metric value (``None`` if unavailable).
        production_value: Production metric value (relative gates only).
        required_value: The bound the candidate had to satisfy.
        passed: Whether the gate passed (never ``None``).
        detail: Human-readable one-line explanation (bounded).
    """

    gate: str
    kind: str
    metric: str
    comparison: str
    candidate_value: float | None
    production_value: float | None
    required_value: float | None
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation."""
        return {
            "gate": self.gate,
            "kind": self.kind,
            "metric": self.metric,
            "comparison": self.comparison,
            "candidate_value": self.candidate_value,
            "production_value": self.production_value,
            "required_value": self.required_value,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PromotionDecision:
    """Structured result of one promotion-gate run.

    Contains model identities (from verified governance), evaluation
    metadata, bounded metric summaries for both models, every policy
    gate with actual/required values and PASS/FAIL status, the overall
    decision, and clear rejection reasons.

    It contains **aggregate metrics only** — never raw transactions,
    customer IDs, raw labels, prediction arrays, filesystem paths, or
    secrets.
    """

    decision: str
    report_scope: str
    disclaimer: str
    gate_timestamp: str
    failure_stage: str | None
    candidate_identity: dict[str, Any] | None
    production_identity: dict[str, Any] | None
    candidate_is_production: bool | None
    evaluation_metadata: dict[str, Any] | None
    candidate_metrics: dict[str, Any] | None
    production_metrics: dict[str, Any] | None
    policy_configuration: dict[str, Any]
    policy_gates: list[dict[str, Any]]
    rejection_reasons: list[str]
    promotion_instruction: dict[str, Any] | None
    reproducibility: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-safe representation."""
        return {
            "report_scope": self.report_scope,
            "decision": self.decision,
            "disclaimer": self.disclaimer,
            "gate_timestamp": self.gate_timestamp,
            "failure_stage": self.failure_stage,
            "candidate_identity": self.candidate_identity,
            "production_identity": self.production_identity,
            "candidate_is_production": self.candidate_is_production,
            "evaluation_metadata": self.evaluation_metadata,
            "candidate_metrics": self.candidate_metrics,
            "production_metrics": self.production_metrics,
            "policy_configuration": self.policy_configuration,
            "policy_gates": self.policy_gates,
            "rejection_reasons": self.rejection_reasons,
            "promotion_instruction": self.promotion_instruction,
            "reproducibility": self.reproducibility,
        }


# ── Helpers ───────────────────────────────────────────────────────────


def _bounded_error(exc: BaseException) -> str:
    """Bounded, path-free error description (never a stack trace)."""
    if isinstance(
        exc, (EvaluationError, LabelError, ProbabilityError, ActivationError)
    ):
        return str(exc)
    return type(exc).__name__


def _extract_metric_summary(report: EvaluationReport) -> dict[str, Any]:
    """Bounded per-model metric summary extracted from a Step 47 report."""
    cm = report.classification_metrics
    rk = report.ranking_metrics
    cal = report.calibration_metrics
    summary: dict[str, Any] = {
        "threshold": float(cm["threshold"]),
        "precision": float(cm["precision"]),
        "recall": float(cm["recall"]),
        "f1": float(cm["f1"]),
        "roc_auc": float(rk["roc_auc"]) if rk is not None else None,
        "pr_auc": float(rk["pr_auc"]) if rk is not None else None,
        "brier_score": float(cal["brier_score"]),
        "n_samples": int(cm["n_samples"]),
        "n_fraud": int(cm["n_fraud"]),
        "fraud_prevalence": float(cm["fraud_prevalence"]),
    }
    if rk is None:
        summary["ranking_unavailable_reason"] = report.ranking_unavailable_reason
    return summary


def _evaluate_gates(
    candidate_metrics: dict[str, Any],
    production_metrics: dict[str, Any],
    policy: PromotionPolicy,
) -> list[GateOutcome]:
    """Evaluate every configured gate; unavailable metrics fail closed."""
    outcomes: list[GateOutcome] = []

    # ── Absolute minimum requirements ──────────────────────────────
    for gate_name, metric, comparison in _ABSOLUTE_GATE_DEFS:
        required = getattr(policy, gate_name)
        if required is None:
            continue
        actual = candidate_metrics.get(metric)
        if actual is None:
            outcomes.append(
                GateOutcome(
                    gate=gate_name,
                    kind="absolute_requirement",
                    metric=metric,
                    comparison=comparison,
                    candidate_value=None,
                    production_value=None,
                    required_value=float(required),
                    passed=False,
                    detail=(
                        f"candidate {metric} is unavailable — "
                        "gate fails closed"
                    ),
                )
            )
            continue
        if comparison == ">=":
            passed = bool(actual >= required - _BOUNDARY_EPSILON)
        else:
            passed = bool(actual <= required + _BOUNDARY_EPSILON)
        detail = (
            f"candidate {metric}={actual:.6f} "
            f"vs required {comparison} {required:.6f}"
        )
        outcomes.append(
            GateOutcome(
                gate=gate_name,
                kind="absolute_requirement",
                metric=metric,
                comparison=comparison,
                candidate_value=float(actual),
                production_value=None,
                required_value=float(required),
                passed=passed,
                detail=detail,
            )
        )

    # ── Relative regression limits vs production ──────────────────
    for gate_name, metric, sense in _RELATIVE_GATE_DEFS:
        limit = getattr(policy, gate_name)
        if limit is None:
            continue
        cand = candidate_metrics.get(metric)
        prod = production_metrics.get(metric)
        if cand is None or prod is None:
            outcomes.append(
                GateOutcome(
                    gate=gate_name,
                    kind="relative_regression_limit",
                    metric=metric,
                    comparison=">=" if sense == "floor" else "<=",
                    candidate_value=cand,
                    production_value=prod,
                    required_value=None,
                    passed=False,
                    detail=(
                        f"candidate or production {metric} is "
                        "unavailable — gate fails closed"
                    ),
                )
            )
            continue
        if sense == "floor":
            required = prod * (1.0 - limit)
            passed = bool(cand >= required - _BOUNDARY_EPSILON)
            detail = (
                f"candidate {metric}={cand:.6f} vs required >= "
                f"{required:.6f} (production {prod:.6f}, max "
                f"degradation {limit:.2%})"
            )
        else:  # ceiling — Brier score (lower is better)
            required = prod * (1.0 + limit)
            passed = bool(cand <= required + _BOUNDARY_EPSILON)
            detail = (
                f"candidate {metric}={cand:.6f} vs required <= "
                f"{required:.6f} (production {prod:.6f}, max "
                f"increase {limit:.2%})"
            )
        outcomes.append(
            GateOutcome(
                gate=gate_name,
                kind="relative_regression_limit",
                metric=metric,
                comparison=">=" if sense == "floor" else "<=",
                candidate_value=float(cand),
                production_value=float(prod),
                required_value=float(required),
                passed=passed,
                detail=detail,
            )
        )

    return outcomes


def _rejection_reasons(gates: list[GateOutcome]) -> list[str]:
    return [
        f"Gate '{outcome.gate}' failed: {outcome.detail}"
        for outcome in gates
        if not outcome.passed
    ]


def _rejected_decision(
    stage: str,
    reasons: list[str],
    *,
    policy: PromotionPolicy | None = None,
    candidate_identity: dict[str, Any] | None = None,
    production_identity: dict[str, Any] | None = None,
) -> PromotionDecision:
    """Build a fail-closed REJECTED decision (validation incomplete)."""
    return PromotionDecision(
        decision=DECISION_REJECTED,
        report_scope=GATE_REPORT_SCOPE,
        disclaimer=GATE_DISCLAIMER,
        gate_timestamp=datetime.now(timezone.utc).isoformat(),
        failure_stage=stage,
        candidate_identity=candidate_identity,
        production_identity=production_identity,
        candidate_is_production=None,
        evaluation_metadata=None,
        candidate_metrics=None,
        production_metrics=None,
        policy_configuration=(
            policy.to_dict()
            if policy is not None
            else {"unavailable": "policy could not be constructed"}
        ),
        policy_gates=[],
        rejection_reasons=list(reasons),
        promotion_instruction=None,
        reproducibility={
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "failure_stage": stage,
            "fail_closed": True,
            "note": (
                "Validation did not complete; the gate never approves "
                "an incompletely validated candidate."
            ),
        },
    )


def _build_promotion_instruction(candidate_identity: dict[str, Any]) -> dict[str, Any]:
    """Safe, promotion-ready instructions for the Step 46 workflow."""
    return {
        "action_required": (
            "Promotion is NOT performed by this gate. Complete it "
            "through the existing Step 46 governance workflow "
            "(explicit, operator-controlled)."
        ),
        "candidate_model_version": candidate_identity.get("model_version"),
        "candidate_artifact_checksum": candidate_identity.get("artifact_checksum"),
        "operator_steps": [
            "1. Review and retain this promotion gate report for audit.",
            "2. Place the verified candidate artifact and model_manifest.json "
            "in the production model directory through the trusted release "
            "pipeline.",
            "3. Restart (or reload) the ML service — Step 46 activation "
            "re-verifies the checksum, interface, and feature compatibility "
            "before switching.",
            "4. Confirm the new model identity via /health, /ready, and "
            "/metrics.",
        ],
        "note": (
            "This gate did not modify the active production manifest, "
            "artifact, threshold, or runtime model."
        ),
    }


# ── Core evaluation (reuses the Step 47 framework) ────────────────────


def evaluate_promotion(
    *,
    production_bundle,
    production_identity,
    candidate_bundle,
    candidate_identity,
    X: pd.DataFrame,
    y: Any,
    policy: PromotionPolicy,
    evaluation_config: EvaluationConfig | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    policy_source: str = "explicit argument",
) -> PromotionDecision:
    """Compare a verified candidate against the production baseline.

    Both models are evaluated on the *same* dataset through the Step 47
    ``build_report`` framework (each at its own bundled production
    threshold — observed, never modified).  The result is a structured,
    bounded :class:`PromotionDecision`.

    Fail-closed: invalid policy, failed evaluation, or unavailable
    metrics yield ``REJECTED`` — never an approval with incomplete
    validation.
    """
    if evaluation_config is None:
        evaluation_config = EvaluationConfig()

    # ── 1. Policy validation (fail closed) ────────────────────────
    try:
        policy.validate()
    except PromotionPolicyError as exc:
        return _rejected_decision(
            "policy_configuration",
            [f"Invalid promotion policy: {exc}"],
            policy=policy,
            candidate_identity=candidate_identity.to_dict(),
            production_identity=production_identity.to_dict(),
        )

    # ── 2. Production baseline evaluation (Step 47 framework) ─────
    try:
        production_report = build_report(
            bundle=production_bundle,
            identity=production_identity,
            X=X,
            y=y,
            config=evaluation_config,
            dataset_metadata=dataset_metadata,
        )
    except Exception as exc:  # fail closed, bounded reason
        logger.error("Production baseline evaluation failed: %s", _bounded_error(exc))
        return _rejected_decision(
            "production_evaluation",
            [f"Production baseline evaluation failed: {_bounded_error(exc)}"],
            policy=policy,
            candidate_identity=candidate_identity.to_dict(),
            production_identity=production_identity.to_dict(),
        )

    # ── 3. Candidate evaluation on the SAME dataset ──────────────
    try:
        candidate_report = build_report(
            bundle=candidate_bundle,
            identity=candidate_identity,
            X=X,
            y=y,
            config=evaluation_config,
            dataset_metadata=dataset_metadata,
        )
    except Exception as exc:  # fail closed, bounded reason
        logger.error("Candidate evaluation failed: %s", _bounded_error(exc))
        return _rejected_decision(
            "candidate_evaluation",
            [f"Candidate evaluation failed: {_bounded_error(exc)}"],
            policy=policy,
            candidate_identity=candidate_identity.to_dict(),
            production_identity=production_identity.to_dict(),
        )

    # ── 4. Bounded metric summaries ──────────────────────────────
    production_metrics = _extract_metric_summary(production_report)
    candidate_metrics = _extract_metric_summary(candidate_report)

    # ── 5. Policy gates ──────────────────────────────────────────
    gates = _evaluate_gates(candidate_metrics, production_metrics, policy)
    rejection_reasons = _rejection_reasons(gates)

    # ── 6. Production-threshold safety guard (paranoia) ──────────
    threshold_unchanged = (
        production_bundle.threshold == production_report.production_threshold
        and candidate_bundle.threshold == candidate_report.production_threshold
    )
    if not threshold_unchanged:  # pragma: no cover — defensive tripwire
        return _rejected_decision(
            "production_safety",
            [
                "A model threshold changed during evaluation — failing "
                "closed.",
                *rejection_reasons,
            ],
            policy=policy,
            candidate_identity=candidate_identity.to_dict(),
            production_identity=production_identity.to_dict(),
        )

    # ── 7. Overall decision ───────────────────────────────────────
    approved = not rejection_reasons
    candidate_is_production = (
        candidate_identity.artifact_checksum == production_identity.artifact_checksum
    )

    evaluation_metadata: dict[str, Any] = {
        "dataset_identifier": (
            (dataset_metadata or {}).get("dataset_identifier", DATASET_IDENTIFIER)
        ),
        "n_test_samples": production_metrics["n_samples"],
        "fraud_prevalence": production_metrics["fraud_prevalence"],
        "split_strategy": (dataset_metadata or {}).get("split_strategy"),
        "evaluation_config": evaluation_config.to_dict(),
        "candidate_threshold": candidate_metrics["threshold"],
        "production_threshold": production_metrics["threshold"],
        "threshold_semantics": (
            "each model is evaluated at its own bundled production "
            "threshold (observed — never modified)"
        ),
        "same_dataset_for_both_models": True,
    }

    reproducibility: dict[str, Any] = {
        "policy_source": policy_source,
        "policy_configuration": policy.to_dict(),
        "evaluation_config": evaluation_config.to_dict(),
        "metric_configuration": (
            "precision / recall / F1 at each model's bundled threshold; "
            "ROC-AUC and PR-AUC (average precision, threshold-free); "
            "Brier score on raw probabilities"
        ),
        "dataset_identifier": evaluation_metadata["dataset_identifier"],
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "deterministic": True,
        "random_seed": "not applicable — the gate contains no stochastic steps",
        "production_threshold_unchanged": True,
        "production_safety": (
            "The gate is read-only for production: activation, manifest "
            "updates, and model switching happen only through the "
            "Step 46 governance workflow."
        ),
    }

    return PromotionDecision(
        decision=DECISION_APPROVED if approved else DECISION_REJECTED,
        report_scope=GATE_REPORT_SCOPE,
        disclaimer=GATE_DISCLAIMER,
        gate_timestamp=datetime.now(timezone.utc).isoformat(),
        failure_stage=None,
        candidate_identity=candidate_identity.to_dict(),
        production_identity=production_identity.to_dict(),
        candidate_is_production=candidate_is_production,
        evaluation_metadata=evaluation_metadata,
        candidate_metrics=candidate_metrics,
        production_metrics=production_metrics,
        policy_configuration=policy.to_dict(),
        policy_gates=[gate.to_dict() for gate in gates],
        rejection_reasons=rejection_reasons,
        promotion_instruction=(
            _build_promotion_instruction(candidate_identity.to_dict())
            if approved
            else None
        ),
        reproducibility=reproducibility,
    )


# ── Full gate run (candidate validation + baseline + evaluation) ──────


def _resolve_production_directory(
    production_model_directory: str | Path | None,
) -> Path:
    """Resolve the trusted production model directory.

    Priority: explicit argument → ``ML_MODEL_DIR`` environment variable
    (the documented Step 46 configuration override) → the default
    ``ml/models/`` directory.  The production *identity* is always
    verified from the manifest — never claimed by the caller.
    """
    if production_model_directory is not None:
        return Path(production_model_directory)
    env_dir = os.environ.get("ML_MODEL_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    return default_model_directory()


def run_promotion_gate(
    candidate_model_directory: str | Path,
    policy: PromotionPolicy | None = None,
    *,
    evaluation_config: EvaluationConfig | None = None,
    production_model_directory: str | Path | None = None,
) -> PromotionDecision:
    """Run the complete promotion gate.

    1. Resolves and validates the promotion policy.
    2. Validates the candidate through the full Step 46 governance
       sequence using a scratch registry (no activation anywhere).
    3. Activates the production baseline through the same verified
       governance path.
    4. Loads the held-out evaluation dataset from the approved source.
    5. Evaluates both models and applies every configured gate.

    Returns a structured :class:`PromotionDecision` — ``APPROVED`` only
    when every configured gate passed on a fully validated candidate.
    Any validation failure fails closed with a bounded reason.

    The active production model, its manifest, and the production
    threshold are **never** modified.
    """
    policy_source = (
        "explicit argument" if policy is not None else "environment (PROMO_*)"
    )

    # ── 1. Policy configuration ─────────────────────────────────
    if policy is None:
        try:
            policy = PromotionPolicy.from_env()
        except PromotionPolicyError as exc:
            return _rejected_decision(
                "policy_configuration",
                [f"Invalid promotion policy: {exc}"],
            )
    try:
        policy.validate()
    except PromotionPolicyError as exc:
        return _rejected_decision(
            "policy_configuration",
            [f"Invalid promotion policy: {exc}"],
            policy=policy,
        )

    if evaluation_config is None:
        try:
            evaluation_config = EvaluationConfig.from_env()
        except EvaluationConfigError as exc:
            return _rejected_decision(
                "evaluation_configuration",
                [f"Invalid evaluation configuration: {exc}"],
                policy=policy,
            )
    try:
        evaluation_config.validate()
    except EvaluationConfigError as exc:
        return _rejected_decision(
            "evaluation_configuration",
            [f"Invalid evaluation configuration: {exc}"],
            policy=policy,
        )

    # ── 2. Candidate validation (scratch registry — Step 46 sequence)
    # The scratch registry is an isolated, offline object; activating
    # in it never touches the production service or any manifest file.
    try:
        candidate_registry = ModelRegistry(model_directory=candidate_model_directory)
        candidate_identity = candidate_registry.activate_from_manifest()
        candidate_bundle = candidate_registry.bundle
    except Exception as exc:
        reason = _bounded_error(exc)
        logger.error("Candidate failed governance validation: %s", reason)
        return _rejected_decision(
            "candidate_validation",
            [f"Candidate model failed governance validation: {reason}"],
            policy=policy,
        )
    if candidate_bundle is None:  # pragma: no cover — activate raises first
        return _rejected_decision(
            "candidate_validation",
            ["Candidate model failed governance validation: no bundle produced."],
            policy=policy,
        )

    # ── 3. Production baseline (verified governance — never claimed) ──
    production_directory = _resolve_production_directory(production_model_directory)
    try:
        production_registry = ModelRegistry(model_directory=production_directory)
        production_identity = production_registry.activate_from_manifest()
        production_bundle = production_registry.bundle
    except Exception as exc:
        reason = _bounded_error(exc)
        logger.error("Production baseline could not be activated: %s", reason)
        return _rejected_decision(
            "production_baseline",
            [f"Production baseline could not be activated: {reason}"],
            policy=policy,
            candidate_identity=candidate_identity.to_dict(),
        )
    if production_bundle is None:  # pragma: no cover — activate raises first
        return _rejected_decision(
            "production_baseline",
            ["Production baseline could not be activated: no bundle produced."],
            policy=policy,
            candidate_identity=candidate_identity.to_dict(),
        )

    # Snapshot the production manifest bytes for the read-only guard.
    production_manifest_path = Path(production_directory) / _MANIFEST_FILENAME
    try:
        manifest_before = production_manifest_path.read_bytes()
    except OSError:
        manifest_before = None

    # ── 4. Evaluation dataset (approved source) ───────────────────
    try:
        X, y, dataset_metadata = load_holdout_test_set()
    except Exception as exc:
        logger.error("Evaluation dataset could not be loaded: %s", _bounded_error(exc))
        return _rejected_decision(
            "evaluation",
            [f"Evaluation dataset could not be loaded: {_bounded_error(exc)}"],
            policy=policy,
            candidate_identity=candidate_identity.to_dict(),
            production_identity=production_identity.to_dict(),
        )

    # ── 5. Evaluate and decide ───────────────────────────────────
    decision = evaluate_promotion(
        production_bundle=production_bundle,
        production_identity=production_identity,
        candidate_bundle=candidate_bundle,
        candidate_identity=candidate_identity,
        X=X,
        y=y,
        policy=policy,
        evaluation_config=evaluation_config,
        dataset_metadata=dataset_metadata,
        policy_source=policy_source,
    )

    # ── 6. Production read-only guard (fail closed if tripped) ───
    if manifest_before is not None:
        try:
            manifest_after = production_manifest_path.read_bytes()
        except OSError:
            manifest_after = None
        if manifest_after != manifest_before:  # pragma: no cover — tripwire
            return _rejected_decision(
                "production_safety",
                [
                    "The production manifest changed during the gate "
                    "run — failing closed."
                ],
                policy=policy,
                candidate_identity=candidate_identity.to_dict(),
                production_identity=production_identity.to_dict(),
            )

    reproducibility = dict(decision.reproducibility)
    reproducibility["production_manifest_unchanged"] = manifest_before is not None
    return replace(decision, reproducibility=reproducibility)


# ── CLI ───────────────────────────────────────────────────────────────


def _fmt(value: Any) -> str:
    """Format a metric value for the console summary."""
    if value is None:
        return "unavailable"
    return f"{value:.6f}"


def _print_summary(decision: PromotionDecision) -> None:
    """Print a concise, bounded human-readable summary."""
    print("=" * 70)
    print("FRAUD MODEL PROMOTION GATE — OFFLINE VALIDATION")
    print("=" * 70)
    print(f"  Decision:  {decision.decision}")
    if decision.failure_stage:
        print(
            f"  Stage:     {decision.failure_stage} "
            "(validation incomplete — failed closed)"
        )
    print(f"  Timestamp: {decision.gate_timestamp}")
    print()

    for role, ident in (
        ("Candidate", decision.candidate_identity),
        ("Production", decision.production_identity),
    ):
        if ident:
            print(
                f"  {role:<11} {ident['model_name']} {ident['model_version']}  "
                f"checksum {ident['artifact_checksum'][:12]}..."
            )
        else:
            print(f"  {role:<11} unavailable (validation failed)")
    if decision.candidate_is_production is not None:
        print(f"  Candidate is the production artifact: {decision.candidate_is_production}")

    if decision.evaluation_metadata:
        em = decision.evaluation_metadata
        print(
            f"  Dataset:   {em['dataset_identifier']} "
            f"({em['n_test_samples']:,} samples, "
            f"prevalence {em['fraud_prevalence']:.4%})"
        )
        print(
            f"  Thresholds: candidate {em['candidate_threshold']:.2f} / "
            f"production {em['production_threshold']:.2f} (observed, not modified)"
        )

    if decision.candidate_metrics and decision.production_metrics:
        print()
        print("  Metric comparison (candidate / production)")
        for metric in ("precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"):
            print(
                f"    {metric:<12} "
                f"{_fmt(decision.candidate_metrics.get(metric))} / "
                f"{_fmt(decision.production_metrics.get(metric))}"
            )

    if decision.policy_gates:
        print()
        print(f"  Policy gates ({len(decision.policy_gates)})")
        for gate in decision.policy_gates:
            status = "PASS" if gate["passed"] else "FAIL"
            print(f"    [{status}] {gate['gate']}: {gate['detail']}")

    if decision.rejection_reasons:
        print()
        print("  Rejection reasons:")
        for reason in decision.rejection_reasons:
            print(f"    - {reason}")

    if decision.promotion_instruction:
        instruction = decision.promotion_instruction
        print()
        print("  Promotion (explicit operator action — Step 46 governance):")
        print(f"    {instruction['action_required']}")
        for step in instruction["operator_steps"]:
            print(f"    {step}")

    print()
    print("-" * 70)
    print("  DISCLAIMER:", decision.disclaimer)
    print("=" * 70)


def main() -> None:
    """CLI entry point: run the promotion gate.

    Exit codes: ``0`` approved; ``1`` rejected (including fail-closed
    validation failures); ``2`` unexpected internal error.  No stack
    traces or internal paths are printed in normal output.
    """
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.promotion_gate",
        description=(
            "Offline promotion gate: validates a candidate fraud model "
            "against the current production model and answers APPROVED "
            "or REJECTED. Never activates a model automatically."
        ),
    )
    parser.add_argument(
        "--candidate-model-dir",
        required=True,
        help=(
            "Directory containing the candidate model_manifest.json and "
            "its artifact (produced by the trusted training pipeline)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the bounded JSON promotion report.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        decision = run_promotion_gate(args.candidate_model_dir)
    except Exception as exc:  # bounded, no stack trace
        logger.error("Promotion gate failed unexpectedly: %s", type(exc).__name__)
        print(f"Promotion gate error: {type(exc).__name__}", file=sys.stderr)
        sys.exit(2)

    _print_summary(decision)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(decision.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nReport written to: {out_path}")

    sys.exit(0 if decision.decision == DECISION_APPROVED else 1)


if __name__ == "__main__":
    main()
