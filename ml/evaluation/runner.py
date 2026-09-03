"""Offline evaluation runner — the Step 47 operator entry point.

Runs a full offline evaluation of the **verified** production model
(activated through the Step 46 governance pipeline) on the held-out
temporal test split of the IEEE-CIS dataset and produces a
JSON-serializable :class:`EvaluationReport`.

Production safety
-----------------
* Evaluation is **offline** — it constructs its own
  :class:`~ml.predict.registry.ModelRegistry` and never touches the
  running ML service, its predictor, or its monitoring counters.
* The production threshold is *observed and reported* (for
  classification metrics at the current operating point) but never
  modified.
* No evaluation API endpoint exists; recommendations are labelled
  ``EVALUATION / RECOMMENDATION ONLY``.
* No files are written unless ``--output`` is passed explicitly by
  the operator.
* The dataset is loaded only from the approved default location via
  the existing :mod:`ml.data.loader` / :mod:`ml.features.engineer`
  pipeline — no caller-supplied paths.

Reproducibility
---------------
The report records the model identity (from the verified manifest),
dataset metadata, evaluation configuration, threshold tie-break rule,
and dataset-source documentation.  The underlying dataset is not
committed to the repository; see ``docs/ml-architecture.md`` for the
authorized operator reproduction workflow.

Usage::

    python -m ml.evaluation.runner [--output PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.evaluation.config import EvaluationConfig
from ml.evaluation.metrics import (
    RankingError,
    calibration_metrics,
    check_paired,
    classification_metrics,
    ranking_metrics,
)
from ml.evaluation.thresholds import (
    TIE_BREAK_RULE,
    build_recommendations,
    cost_curve,
    sweep_thresholds,
)
from ml.models.baseline import apply_preprocessing

logger = logging.getLogger(__name__)

__all__ = [
    "EvaluationReport",
    "EvaluationError",
    "REPORT_SCOPE",
    "REPORT_DISCLAIMER",
    "DATASET_IDENTIFIER",
    "score_with_bundle",
    "build_report",
    "load_holdout_test_set",
    "run_offline_evaluation",
    "main",
]


class EvaluationError(Exception):
    """Offline evaluation could not be completed safely."""


REPORT_SCOPE: str = "offline_model_evaluation"
REPORT_DISCLAIMER: str = (
    "OFFLINE MODEL EVALUATION — EVALUATION / RECOMMENDATION ONLY. "
    "Results and recommendations DO NOT automatically change "
    "production decisions, the production threshold, risk aggregation, "
    "or the active model."
)
DATASET_IDENTIFIER: str = "ieee-cis-fraud/holdout-test-split"


# ── Report container ──────────────────────────────────────────────────


@dataclass(frozen=True)
class EvaluationReport:
    """Structured result of one offline model evaluation.

    All nested values are JSON-serializable primitives.  The report
    contains aggregate metrics only — never raw transactions,
    customer information, card data, secrets, or filesystem paths.
    """

    report_scope: str
    disclaimer: str
    evaluation_timestamp: str
    model_identity: dict[str, Any]
    production_threshold: float
    dataset_metadata: dict[str, Any]
    classification_metrics: dict[str, Any]
    ranking_metrics: dict[str, Any] | None
    ranking_unavailable_reason: str | None
    calibration_metrics: dict[str, Any]
    threshold_analysis: list[dict[str, Any]]
    cost_analysis: dict[str, Any]
    recommendations: list[dict[str, Any]]
    reproducibility: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-safe representation."""
        return {
            "report_scope": self.report_scope,
            "disclaimer": self.disclaimer,
            "evaluation_timestamp": self.evaluation_timestamp,
            "model_identity": self.model_identity,
            "production_threshold": self.production_threshold,
            "dataset_metadata": self.dataset_metadata,
            "classification_metrics": self.classification_metrics,
            "ranking_metrics": self.ranking_metrics,
            "ranking_unavailable_reason": self.ranking_unavailable_reason,
            "calibration_metrics": self.calibration_metrics,
            "threshold_analysis": self.threshold_analysis,
            "cost_analysis": self.cost_analysis,
            "recommendations": self.recommendations,
            "reproducibility": self.reproducibility,
        }


# ── Scoring (reuses fitted preprocessing — transform only) ───────────


def score_with_bundle(bundle, X: pd.DataFrame) -> np.ndarray:
    """Score features with a verified model bundle.

    Applies the bundle's **fitted** preprocessing (transform only —
    never re-fitted on evaluation data) and returns raw fraud
    probabilities from ``predict_proba``.

    Only the bundle's declared feature columns are used; any extra
    columns in *X* (including label or ID columns that must never be
    features) are ignored.

    Raises:
        EvaluationError: If required feature columns are missing.
    """
    missing = set(bundle.feature_names) - set(X.columns)
    if missing:
        raise EvaluationError(
            f"Evaluation features are missing {len(missing)} required "
            "model feature column(s)."
        )

    X_selected = X[list(bundle.feature_names)]
    X_transformed = apply_preprocessing(X_selected, bundle.preprocessing)
    probabilities = bundle.model.predict_proba(X_transformed)[:, 1]
    return np.asarray(probabilities, dtype=np.float64)


# ── Report assembly ──────────────────────────────────────────────────


def build_report(
    *,
    bundle,
    identity,
    X: pd.DataFrame,
    y: Any,
    config: EvaluationConfig,
    dataset_metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Assemble a full evaluation report.

    Args:
        bundle: Verified :class:`~ml.predict.bundle.ModelBundle` used
            for scoring.  It is only read — never modified.
        identity: The :class:`~ml.predict.registry.ModelIdentity` of
            the verified active model.  Model identity is taken from
            governance — the caller cannot claim an arbitrary version.
        X: Evaluation feature matrix (extra columns ignored).
        y: Ground-truth labels aligned with *X*.
        config: Evaluation-only configuration.
        dataset_metadata: Optional loader-provided metadata (dataset
            identifier, split details) merged into the report.

    Raises:
        EvaluationError: If bundle and identity disagree (traceability
            integrity check) or the inputs are invalid.
        LabelError / ProbabilityError: For invalid labels/probabilities
            (propagated from metric validation — never silenced).
    """
    # ── Traceability integrity: bundle must match the claimed identity ──
    if bundle.model_version != identity.model_version:
        raise EvaluationError(
            "Model identity mismatch: the bundle reports "
            f"{bundle.model_version!r} but the verified identity is "
            f"{identity.model_version!r}."
        )

    config.validate()

    # ── Score with the verified bundle (read-only) ─────────────────────
    y_prob = score_with_bundle(bundle, X)
    labels, _ = check_paired(y, y_prob)
    n = int(len(labels))
    n_fraud = int(np.sum(labels == 1))
    n_legitimate = n - n_fraud

    # ── Classification at the CURRENT production threshold ──────────────
    y_pred = (y_prob >= bundle.threshold).astype(np.int64)
    classification = dict(classification_metrics(labels, y_pred))
    classification["threshold"] = float(bundle.threshold)
    classification["threshold_source"] = (
        "production model bundle (observed — NOT modified by evaluation)"
    )

    # ── Ranking (explicit failure when only one class is present) ───────
    ranking: dict[str, Any] | None
    ranking_reason: str | None = None
    try:
        ranking = ranking_metrics(labels, y_prob)
    except RankingError as exc:
        ranking = None
        ranking_reason = str(exc)

    # ── Calibration (raw probabilities — no recalibration) ──────────────
    calibration = calibration_metrics(
        labels, y_prob, n_bins=config.calibration_bins
    )

    # ── Threshold sweep (observational) ─────────────────────────────────
    points = sweep_thresholds(labels, y_prob, config.threshold_grid())
    threshold_analysis = [pt.to_dict() for pt in points]

    # ── Cost analysis (only when costs are configured) ──────────────────
    if config.costs_configured():
        cost_analysis: dict[str, Any] = {
            "available": True,
            "false_negative_cost": config.false_negative_cost,
            "false_positive_cost": config.false_positive_cost,
            "total_cost_definition": (
                "total_cost = FN * false_negative_cost + FP * false_positive_cost"
            ),
            "cost_curve": cost_curve(
                points,
                false_negative_cost=config.false_negative_cost,
                false_positive_cost=config.false_positive_cost,
            ),
        }
    else:
        cost_analysis = {
            "available": False,
            "reason": (
                "Business costs are not configured (EVAL_FN_COST / "
                "EVAL_FP_COST); cost-based threshold analysis is "
                "unavailable."
            ),
        }

    # ── Recommendations (EVALUATION / RECOMMENDATION ONLY) ───────────────
    recommendations = [
        rec.to_dict() for rec in build_recommendations(points, config)
    ]

    # ── Dataset metadata ────────────────────────────────────────────────
    computed_dataset_meta: dict[str, Any] = {
        "dataset_identifier": DATASET_IDENTIFIER,
        "n_samples": n,
        "n_fraud": n_fraud,
        "n_legitimate": n_legitimate,
        "fraud_prevalence": (n_fraud / n) if n > 0 else 0.0,
        "label_column": "isFraud",
    }
    if dataset_metadata:
        computed_dataset_meta.update(dataset_metadata)

    # ── Reproducibility metadata ────────────────────────────────────────
    reproducibility: dict[str, Any] = {
        "evaluation_config": config.to_dict(),
        "split_strategy": (
            "time_based 80/20 split on TransactionDT "
            "(ml.split.splitter.time_based_split) — leakage-safe"
        ),
        "dataset_source": (
            "IEEE-CIS Fraud Detection dataset at ml/datasets/raw/ "
            "(local, not committed; see docs/ml-architecture.md for the "
            "authorized reproduction workflow)"
        ),
        "tie_break_rule": TIE_BREAK_RULE,
        "deterministic": True,
        "random_seed": (
            "not applicable — evaluation contains no stochastic steps"
        ),
        "monitoring_separation": (
            "Offline evaluation metrics are separate from live Step 43 "
            "monitoring counters and never overwrite them."
        ),
    }

    return EvaluationReport(
        report_scope=REPORT_SCOPE,
        disclaimer=REPORT_DISCLAIMER,
        evaluation_timestamp=datetime.now(timezone.utc).isoformat(),
        model_identity=identity.to_dict(),
        production_threshold=float(bundle.threshold),
        dataset_metadata=computed_dataset_meta,
        classification_metrics=classification,
        ranking_metrics=ranking,
        ranking_unavailable_reason=ranking_reason,
        calibration_metrics=calibration,
        threshold_analysis=threshold_analysis,
        cost_analysis=cost_analysis,
        recommendations=recommendations,
        reproducibility=reproducibility,
    )


# ── Dataset loading (approved source, leakage-safe split) ────────────


def load_holdout_test_set() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Engineer features and return the held-out temporal test split.

    Reuses the existing feature engineering and time-based split —
    no preprocessing or splitting logic is duplicated.  The test set is
    strictly **after** every training transaction in ``TransactionDT``,
    and historical features are computed strictly prior to each
    transaction (see ``ml/features/historical.py``).

    Returns:
        ``(X_test, y_test, metadata)`` where *metadata* records split
        details for reproducibility.
    """
    from ml.data.loader import load_transaction_dataset
    from ml.features.engineer import engineer_features
    from ml.split.splitter import time_based_split

    fe = engineer_features()
    features: pd.DataFrame = fe["features"]
    target: pd.Series = fe["target"]
    txn_ids: pd.Series = fe["transaction_ids"]

    txn = load_transaction_dataset()
    timestamps = txn.set_index("TransactionID")["TransactionDT"]
    timestamps = timestamps.loc[txn_ids.values].reset_index(drop=True)
    timestamps.index = features.index

    split = time_based_split(
        features=features,
        target=target,
        timestamps=timestamps,
        transaction_ids=txn_ids,
    )

    metadata: dict[str, Any] = {
        "dataset_identifier": DATASET_IDENTIFIER,
        "split_strategy": "time_based_80_20",
        "split_timestamp": int(split.split_timestamp),
        "test_fraction": float(split.test_fraction),
        "n_test_samples": int(len(split.X_test)),
    }
    return split.X_test, split.y_test, metadata


# ── Full offline evaluation ──────────────────────────────────────────


def run_offline_evaluation(
    config: EvaluationConfig | None = None,
    *,
    model_directory: str | Path | None = None,
) -> EvaluationReport:
    """Run the complete offline evaluation.

    1. Activates the model through the Step 46 governance pipeline
       (manifest → checksum → bundle → interface/schema validation).
    2. Loads the held-out temporal test split from the approved
       dataset source.
    3. Scores, computes metrics, sweeps thresholds, and produces
       labelled recommendations.

    The active production model and threshold are **never** modified.
    """
    from ml.predict.registry import ModelRegistry

    if config is None:
        config = EvaluationConfig.from_env()
    config.validate()

    registry = ModelRegistry(model_directory=model_directory)
    identity = registry.activate_from_manifest()
    bundle = registry.bundle
    if bundle is None:  # pragma: no cover — activate raises on failure
        raise EvaluationError("Model registry did not produce a bundle.")

    logger.info(
        "Offline evaluation starting: model=%s checksum=%s",
        identity.model_version,
        identity.checksum_short,
    )

    X_test, y_test, dataset_metadata = load_holdout_test_set()

    report = build_report(
        bundle=bundle,
        identity=identity,
        X=X_test,
        y=y_test,
        config=config,
        dataset_metadata=dataset_metadata,
    )

    # Governance integrity: the threshold must be untouched by evaluation.
    if bundle.threshold != report.production_threshold:  # pragma: no cover
        raise EvaluationError(
            "Production threshold changed during evaluation — aborting."
        )

    logger.info("Offline evaluation complete (evaluation-only).")
    return report


# ── CLI ──────────────────────────────────────────────────────────────


def _print_summary(report: EvaluationReport) -> None:
    """Print a concise human-readable summary."""
    ident = report.model_identity
    cm = report.classification_metrics
    ds = report.dataset_metadata

    print("=" * 70)
    print("OFFLINE FRAUD MODEL EVALUATION")
    print("=" * 70)
    print(f"  Scope:      {report.report_scope}")
    print(f"  Timestamp:  {report.evaluation_timestamp}")
    print()
    print(f"  Model:         {ident['model_name']} {ident['model_version']}")
    print(f"  Checksum:     {ident['artifact_checksum'][:16]}...")
    print(f"  Schema:       {ident['feature_schema_version']} "
          f"({ident['n_features']} features)")
    print(f"  Dataset:      {ds['dataset_identifier']}")
    print(f"  Samples:      {ds['n_samples']:,} "
          f"(fraud {ds['n_fraud']:,} / legit {ds['n_legitimate']:,}, "
          f"prevalence {ds['fraud_prevalence']:.4%})")
    print()
    print(f"  Production threshold (observed, NOT modified): "
          f"{report.production_threshold:.2f}")
    print()
    print("  Classification @ production threshold")
    print(f"    TP={cm['tp']:,}  TN={cm['tn']:,}  FP={cm['fp']:,}  FN={cm['fn']:,}")
    print(f"    Precision={cm['precision']:.4f}  Recall={cm['recall']:.4f}  "
          f"F1={cm['f1']:.4f}")
    print(f"    FPR={cm['false_positive_rate']:.4f}  "
          f"FNR={cm['false_negative_rate']:.4f}  "
          f"Flagged={cm['n_flagged']:,} ({cm['flagged_rate']:.4%})")

    if report.ranking_metrics is not None:
        print()
        print("  Ranking")
        print(f"    ROC-AUC={report.ranking_metrics['roc_auc']:.4f}  "
              f"PR-AUC={report.ranking_metrics['pr_auc']:.4f}")
    else:
        print()
        print(f"  Ranking unavailable: {report.ranking_unavailable_reason}")

    cal = report.calibration_metrics
    print()
    print("  Calibration (raw probabilities)")
    print(f"    Brier score={cal['brier_score']:.6f}  "
          f"bins={cal['n_bins']}  recalibration={cal['recalibration_applied']}")

    if report.cost_analysis.get("available"):
        curve = report.cost_analysis["cost_curve"]
        best = min(curve, key=lambda c: c["total_cost"])
        print()
        print("  Cost analysis")
        print(f"    Min total cost {best['total_cost']:,.0f} at threshold "
              f"{best['threshold']:.2f}")
    else:
        print()
        print(f"  Cost analysis unavailable: {report.cost_analysis['reason']}")

    print()
    print("  Recommendations (EVALUATION / RECOMMENDATION ONLY)")
    for rec in report.recommendations:
        if rec["available"]:
            print(
                f"    [{rec['strategy']}] threshold={rec['threshold']:.2f}  "
                f"P={rec['precision']:.4f}  R={rec['recall']:.4f}  "
                f"F1={rec['f1']:.4f}  flagged={rec['flagged_count']:,}"
            )
        else:
            print(f"    [{rec['strategy']}] unavailable — {rec['reason']}")

    print()
    print("-" * 70)
    print("  DISCLAIMER:", report.disclaimer)
    print("=" * 70)


def main() -> None:
    """CLI entry point: run the offline evaluation."""
    parser = argparse.ArgumentParser(
        prog="python -m ml.evaluation.runner",
        description=(
            "Offline evaluation of the verified fraud model "
            "(evaluation/recommendation only — production is not modified)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the JSON evaluation report.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    report = run_offline_evaluation()
    _print_summary(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
