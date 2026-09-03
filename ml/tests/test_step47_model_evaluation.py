"""Step 47 — Fraud model evaluation, calibration & threshold governance tests.

Covers (spec §14 A–AD):
confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, fraud
prevalence, FPR/FNR, threshold sweep, recommendations (F1 / cost /
min-recall / min-precision), missing labels, invalid probabilities,
probability bounds, calibration & Brier score, calibration under
imbalance, model-version traceability, checksum traceability,
reproducibility metadata, deterministic repeated evaluation, leakage
protection, production-threshold protection, live prediction
behaviour unchanged, no sensitive data in output, no arbitrary model
selection, no production model mutation, and degenerate label sets.

Run from the project root::

    python -m pytest ml/tests/test_step47_model_evaluation.py -v
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

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
from ml.evaluation.thresholds import (
    RECOMMENDATION_DISCLAIMER,
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
from ml.evaluation.runner import (
    EvaluationError,
    REPORT_DISCLAIMER,
    REPORT_SCOPE,
    build_report,
    score_with_bundle,
)

# ── Shared synthetic data (fully hand-computable) ─────────────────────
#
# Y_TRUE / Y_PROB at threshold 0.5 → preds [1,1,0,0,0,0,1,1]
#   TP=3  TN=3  FP=1  FN=1
#   precision = recall = F1 = 0.75, accuracy = 0.75
#   FPR = 1/4, FNR = 1/4, prevalence = 4/8
# ROC-AUC = 15/16 (only the (0.4, 0.5) pair is mis-ordered)
# Brier  = 0.96 / 8 = 0.12

Y_TRUE = np.array([1, 1, 1, 0, 0, 0, 1, 0])
Y_PROB = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.1, 0.6, 0.5])
PREDS_05 = (Y_PROB >= 0.5).astype(int)


# ── Stub bundle / identity (deterministic, no artifacts needed) ───────


class _StubScaler:
    def transform(self, values):
        return np.asarray(values, dtype=np.float64)


class _StubPreprocessing:
    def __init__(self) -> None:
        self.label_encoders: dict = {}
        self.numeric_cols = ["score"]
        self.scaler = _StubScaler()


class _StubModel:
    """Identity model — the first feature column IS the probability."""

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.column_stack([1.0 - X[:, 0], X[:, 0]])


class StubBundle:
    """Minimal read-only ModelBundle stand-in."""

    def __init__(self, threshold: float = 0.5, version: str = "stub-v1") -> None:
        self.model = _StubModel()
        self.preprocessing = _StubPreprocessing()
        self.threshold = threshold
        self.feature_names = ["score"]
        self.model_version = version

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


class StubIdentity:
    """Minimal ModelIdentity stand-in (governance-shaped)."""

    def __init__(self, version: str = "stub-v1") -> None:
        self.model_name = "stub-model"
        self.model_version = version
        self.artifact_checksum = "a" * 64
        self.feature_schema_version = "1.0.0"
        self.n_features = 1
        self.status = "active"

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "artifact_checksum": self.artifact_checksum,
            "feature_schema_version": self.feature_schema_version,
            "n_features": self.n_features,
            "status": self.status,
        }


@pytest.fixture
def stub_bundle() -> StubBundle:
    return StubBundle()


@pytest.fixture
def stub_identity() -> StubIdentity:
    return StubIdentity()


@pytest.fixture
def stub_features() -> pd.DataFrame:
    return pd.DataFrame({"score": Y_PROB})


@pytest.fixture(autouse=True)
def _clean_eval_env(monkeypatch):
    """Isolate tests from any EVAL_* variables in the environment."""
    for name in (
        "EVAL_THRESHOLD_START",
        "EVAL_THRESHOLD_STOP",
        "EVAL_THRESHOLD_STEP",
        "EVAL_MIN_RECALL",
        "EVAL_MIN_PRECISION",
        "EVAL_FN_COST",
        "EVAL_FP_COST",
        "EVAL_CALIBRATION_BINS",
    ):
        monkeypatch.delenv(name, raising=False)


def _default_config(**overrides) -> EvaluationConfig:
    config = EvaluationConfig(**overrides)
    config.validate()
    return config


# ═══════════════════════════════════════════════════════════════════
# A. Confusion matrix
# ═══════════════════════════════════════════════════════════════════


class TestConfusionMatrix:
    def test_hand_computed_counts(self):
        cc = confusion_counts(Y_TRUE, PREDS_05)
        assert cc == {"tp": 3, "tn": 3, "fp": 1, "fn": 1}

    def test_matrix_layout(self):
        cm = classification_metrics(Y_TRUE, PREDS_05)
        assert cm["confusion_matrix"] == [[3, 1], [1, 3]]

    def test_matches_sklearn(self):
        cm = classification_metrics(Y_TRUE, PREDS_05)
        assert np.array_equal(
            np.array(cm["confusion_matrix"]),
            confusion_matrix(Y_TRUE, PREDS_05),
        )

    def test_perfect_predictions(self):
        cc = confusion_counts(Y_TRUE, Y_TRUE)
        assert cc == {"tp": 4, "tn": 4, "fp": 0, "fn": 0}

    def test_all_wrong_predictions(self):
        cc = confusion_counts(Y_TRUE, 1 - Y_TRUE)
        assert cc == {"tp": 0, "tn": 0, "fp": 4, "fn": 4}

    def test_predictions_must_be_binary(self):
        with pytest.raises(LabelError):
            confusion_counts(Y_TRUE, np.array([2, 0, 1, 0, 1, 0, 1, 0]))

    def test_predictions_must_align(self):
        with pytest.raises(ProbabilityError):
            confusion_counts(Y_TRUE, np.array([1, 0, 1]))


# ═══════════════════════════════════════════════════════════════════
# B / C / D. Precision, recall, F1
# ═══════════════════════════════════════════════════════════════════


class TestPrecisionRecallF1:
    def test_hand_computed_values(self):
        cm = classification_metrics(Y_TRUE, PREDS_05)
        assert cm["precision"] == pytest.approx(0.75)
        assert cm["recall"] == pytest.approx(0.75)
        assert cm["f1"] == pytest.approx(0.75)
        assert cm["accuracy"] == pytest.approx(0.75)

    def test_matches_sklearn(self):
        cm = classification_metrics(Y_TRUE, PREDS_05)
        assert cm["precision"] == pytest.approx(
            precision_score(Y_TRUE, PREDS_05, zero_division=0)
        )
        assert cm["recall"] == pytest.approx(
            recall_score(Y_TRUE, PREDS_05, zero_division=0)
        )

    def test_f1_harmonic_mean(self):
        # Construct P=1.0, R=0.5 → F1 = 2/3.
        y = np.array([1, 1, 0])
        preds = np.array([1, 0, 0])
        cm = classification_metrics(y, preds)
        assert cm["precision"] == pytest.approx(1.0)
        assert cm["recall"] == pytest.approx(0.5)
        assert cm["f1"] == pytest.approx(2.0 / 3.0)

    def test_nothing_flagged_precision_zero(self):
        y = np.array([1, 0, 1])
        preds = np.array([0, 0, 0])
        cm = classification_metrics(y, preds)
        assert cm["precision"] == 0.0  # documented zero-division convention
        assert cm["recall"] == 0.0
        assert cm["f1"] == 0.0

    def test_no_fraud_recall_zero(self):
        y = np.array([0, 0, 0])
        preds = np.array([0, 1, 0])
        cm = classification_metrics(y, preds)
        assert cm["recall"] == 0.0
        assert cm["precision"] == 0.0
        assert cm["f1"] == 0.0

    def test_aliases(self):
        cm = classification_metrics(Y_TRUE, PREDS_05)
        assert cm["fraud_detection_rate"] == cm["recall"]
        assert cm["precision_among_flagged"] == cm["precision"]


# ═══════════════════════════════════════════════════════════════════
# E / F. ROC-AUC and PR-AUC
# ═══════════════════════════════════════════════════════════════════


class TestRankingMetrics:
    def test_roc_auc_hand_computed(self):
        rm = ranking_metrics(Y_TRUE, Y_PROB)
        assert rm["roc_auc"] == pytest.approx(15.0 / 16.0)

    def test_roc_auc_matches_sklearn(self):
        rm = ranking_metrics(Y_TRUE, Y_PROB)
        assert rm["roc_auc"] == pytest.approx(roc_auc_score(Y_TRUE, Y_PROB))

    def test_pr_auc_matches_sklearn(self):
        rm = ranking_metrics(Y_TRUE, Y_PROB)
        assert rm["pr_auc"] == pytest.approx(
            average_precision_score(Y_TRUE, Y_PROB)
        )

    def test_pr_auc_definition_documented(self):
        rm = ranking_metrics(Y_TRUE, Y_PROB)
        assert rm["pr_auc_definition"] == "average_precision"

    def test_perfect_ranking(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        rm = ranking_metrics(y, p)
        assert rm["roc_auc"] == pytest.approx(1.0)
        assert rm["pr_auc"] == pytest.approx(1.0)

    def test_constant_probabilities(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.5, 0.5, 0.5, 0.5])
        rm = ranking_metrics(y, p)
        assert rm["roc_auc"] == pytest.approx(0.5)

    def test_single_class_raises_clear_error(self):
        with pytest.raises(RankingError, match="single class"):
            ranking_metrics(np.ones(10), np.linspace(0, 1, 10))

    def test_length_mismatch_raises(self):
        with pytest.raises(ProbabilityError, match="same length"):
            ranking_metrics(Y_TRUE, Y_PROB[:-1])


# ═══════════════════════════════════════════════════════════════════
# G / H / I. Fraud prevalence, FPR, FNR
# ═══════════════════════════════════════════════════════════════════


class TestPrevalenceAndRates:
    def test_fraud_prevalence(self):
        cm = classification_metrics(Y_TRUE, PREDS_05)
        assert cm["fraud_prevalence"] == pytest.approx(0.5)
        assert cm["n_fraud"] == 4
        assert cm["n_legitimate"] == 4

    def test_false_positive_rate(self):
        cm = classification_metrics(Y_TRUE, PREDS_05)
        assert cm["false_positive_rate"] == pytest.approx(1.0 / 4.0)

    def test_false_negative_rate(self):
        cm = classification_metrics(Y_TRUE, PREDS_05)
        assert cm["false_negative_rate"] == pytest.approx(1.0 / 4.0)

    def test_fpr_zero_when_no_legitimate(self):
        y = np.array([1, 1])
        preds = np.array([1, 0])
        cm = classification_metrics(y, preds)
        assert cm["false_positive_rate"] == 0.0  # documented convention

    def test_fnr_zero_when_no_fraud(self):
        y = np.array([0, 0])
        preds = np.array([1, 0])
        cm = classification_metrics(y, preds)
        assert cm["false_negative_rate"] == 0.0  # documented convention

    def test_flagged_rate(self):
        cm = classification_metrics(Y_TRUE, PREDS_05)
        assert cm["n_flagged"] == 4
        assert cm["flagged_rate"] == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════
# J. Threshold sweep
# ═══════════════════════════════════════════════════════════════════


class TestThresholdSweep:
    GRID = threshold_grid(0.05, 0.95, 0.05)

    def test_grid_bounds_and_count(self):
        assert self.GRID[0] == pytest.approx(0.05)
        assert self.GRID[-1] == pytest.approx(0.95)  # stop inclusive
        assert len(self.GRID) == 19

    def test_grid_rejects_invalid_range(self):
        with pytest.raises(ValueError):
            threshold_grid(0.5, 0.5, 0.1)
        with pytest.raises(ValueError):
            threshold_grid(0.9, 0.1, 0.05)

    def test_grid_rejects_bad_step(self):
        with pytest.raises(ValueError):
            threshold_grid(0.1, 0.2, 0.5)  # step larger than range
        with pytest.raises(ValueError):
            threshold_grid(0.1, 0.9, 0.0)

    def test_grid_rejects_unbounded_points(self):
        with pytest.raises(ValueError, match="bounded"):
            threshold_grid(0.0, 1.0, 0.001, max_points=100)

    def test_sweep_point_count_and_order(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        assert len(points) == len(self.GRID)
        thresholds = [p.threshold for p in points]
        assert thresholds == sorted(thresholds)

    def test_each_point_matches_direct_computation(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        for pt in points:
            preds = (Y_PROB >= pt.threshold).astype(int)
            direct = classification_metrics(Y_TRUE, preds)
            assert pt.tp == direct["tp"]
            assert pt.tn == direct["tn"]
            assert pt.fp == direct["fp"]
            assert pt.fn == direct["fn"]
            assert pt.precision == pytest.approx(direct["precision"])
            assert pt.recall == pytest.approx(direct["recall"])
            assert pt.f1 == pytest.approx(direct["f1"])
            assert pt.false_positive_rate == pytest.approx(
                direct["false_positive_rate"]
            )
            assert pt.false_negative_rate == pytest.approx(
                direct["false_negative_rate"]
            )
            assert pt.flagged_count == direct["n_flagged"]
            assert pt.flagged_rate == pytest.approx(direct["flagged_rate"])

    def test_flagged_count_monotonically_non_increasing(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        counts = [p.flagged_count for p in points]
        assert counts == sorted(counts, reverse=True)

    def test_tp_monotonically_non_increasing(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, 500)
        p = rng.random(500)
        points = sweep_thresholds(y, p, self.GRID)
        tps = [pt.tp for pt in points]
        assert tps == sorted(tps, reverse=True)

    def test_sweep_validates_labels(self):
        bad = Y_TRUE.astype(float)
        bad[0] = np.nan
        with pytest.raises(LabelError):
            sweep_thresholds(bad, Y_PROB, self.GRID)

    def test_sweep_validates_probabilities(self):
        bad = Y_PROB.copy()
        bad[0] = 1.5
        with pytest.raises(ProbabilityError):
            sweep_thresholds(Y_TRUE, bad, self.GRID)

    def test_sweep_empty_thresholds(self):
        assert sweep_thresholds(Y_TRUE, Y_PROB, []) == []


# ═══════════════════════════════════════════════════════════════════
# K. Threshold recommendation by F1
# ═══════════════════════════════════════════════════════════════════


class TestRecommendationByF1:
    GRID = threshold_grid(0.05, 0.95, 0.05)

    def test_selects_max_f1(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        rec = recommend_by_f1(points)
        # At t=0.4: TP=4, FN=0, FP=1 → P=0.8, R=1.0, F1≈0.8889 (best).
        assert rec.available is True
        assert rec.threshold == pytest.approx(0.4)
        assert rec.f1 == pytest.approx(0.8 / 0.9)

    def test_matches_bruteforce(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        rec = recommend_by_f1(points)
        best_f1 = max(pt.f1 for pt in points)
        assert rec.f1 == pytest.approx(best_f1)
        assert any(
            pt.threshold == rec.threshold and pt.f1 == best_f1 for pt in points
        )

    def test_tie_break_prefers_highest_threshold(self):
        a = ThresholdPoint(0.10, 1, 1, 0, 0, 0.5, 0.5, 0.5, 0.0, 0.0, 1, 0.5)
        b = ThresholdPoint(0.40, 1, 1, 0, 0, 0.5, 0.5, 0.5, 0.0, 0.0, 1, 0.5)
        rec = recommend_by_f1([a, b])
        assert rec.threshold == pytest.approx(0.4)

    def test_empty_points_unavailable(self):
        rec = recommend_by_f1([])
        assert rec.available is False
        assert "No threshold" in rec.reason

    def test_disclaimer_labelled(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        rec = recommend_by_f1(points)
        assert RECOMMENDATION_DISCLAIMER in rec.to_dict()["classification"]

    def test_report_contains_all_strategy_fields(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        rec = recommend_by_f1(points).to_dict()
        for key in (
            "strategy",
            "strategy_description",
            "available",
            "threshold",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "false_negative_rate",
            "flagged_count",
            "flagged_rate",
            "classification",
        ):
            assert key in rec


# ═══════════════════════════════════════════════════════════════════
# L. Threshold recommendation by business cost
# ═══════════════════════════════════════════════════════════════════


class TestRecommendationByCost:
    GRID = threshold_grid(0.05, 0.95, 0.05)

    def test_minimises_total_cost(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        rec = recommend_by_cost(
            points, false_negative_cost=10.0, false_positive_cost=1.0
        )
        best_cost = min(pt.fn * 10.0 + pt.fp * 1.0 for pt in points)
        optimal = [
            pt
            for pt in points
            if pt.fn * 10.0 + pt.fp * 1.0 == pytest.approx(best_cost)
        ]
        assert rec.available is True
        # Ties resolve to the highest threshold (documented rule).
        assert rec.threshold == pytest.approx(
            max(pt.threshold for pt in optimal)
        )

    def test_matches_bruteforce_cost(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        fn_cost, fp_cost = 7.5, 2.25
        rec = recommend_by_cost(
            points, false_negative_cost=fn_cost, false_positive_cost=fp_cost
        )
        best_cost = min(
            pt.fn * fn_cost + pt.fp * fp_cost for pt in points
        )
        chosen = next(pt for pt in points if pt.threshold == rec.threshold)
        assert chosen.fn * fn_cost + chosen.fp * fp_cost == pytest.approx(best_cost)

    def test_unconfigured_costs_reported_unavailable(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        rec = recommend_by_cost(
            points, false_negative_cost=None, false_positive_cost=None
        )
        assert rec.available is False
        assert "not configured" in rec.reason
        assert rec.threshold is None

    def test_half_configured_costs_unavailable(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        rec = recommend_by_cost(
            points, false_negative_cost=10.0, false_positive_cost=None
        )
        assert rec.available is False

    def test_negative_costs_rejected(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        rec = recommend_by_cost(
            points, false_negative_cost=-1.0, false_positive_cost=1.0
        )
        assert rec.available is False
        assert "non-negative" in rec.reason

    def test_reason_states_cost_model(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        rec = recommend_by_cost(
            points, false_negative_cost=10.0, false_positive_cost=2.0
        )
        assert "FN x 10.0" in rec.reason
        assert "FP x 2.0" in rec.reason

    def test_cost_curve_values(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        curve = cost_curve(
            points, false_negative_cost=10.0, false_positive_cost=1.0
        )
        assert len(curve) == len(points)
        for entry, pt in zip(curve, points):
            assert entry["threshold"] == pt.threshold
            assert entry["false_negatives"] == pt.fn
            assert entry["false_positives"] == pt.fp
            assert entry["total_cost"] == pytest.approx(pt.fn * 10.0 + pt.fp * 1.0)

    def test_zero_costs_allowed(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        rec = recommend_by_cost(
            points, false_negative_cost=0.0, false_positive_cost=0.0
        )
        assert rec.available is True


# ═══════════════════════════════════════════════════════════════════
# M / N. Constraint-based strategies
# ═══════════════════════════════════════════════════════════════════


class TestMinRecallStrategy:
    GRID = threshold_grid(0.05, 0.95, 0.05)

    def test_satisfies_constraint_with_max_precision(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        rec = recommend_min_recall(points, min_recall=0.7)
        assert rec.available is True
        # At t=0.6: TP=3, FN=1, FP=0 → R=0.75 ≥ 0.7 with P=1.0.
        assert rec.threshold == pytest.approx(0.6)
        assert rec.recall >= 0.7
        assert rec.precision == pytest.approx(1.0)

    def test_unsatisfiable_constraint(self):
        # Restricted grid: the maximum achievable recall is 0.5
        # (at t=0.7: TP=2, FN=2), so 0.99 is unreachable.
        grid = threshold_grid(0.7, 0.95, 0.05)
        points = sweep_thresholds(Y_TRUE, Y_PROB, grid)
        rec = recommend_min_recall(points, min_recall=0.99)
        assert rec.available is False
        assert "No threshold" in rec.reason

    def test_not_configured(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        rec = recommend_min_recall(points, min_recall=None)
        assert rec.available is False
        assert "not configured" in rec.reason

    def test_all_feasible_points_satisfy_constraint(self):
        rng = np.random.default_rng(11)
        y = rng.integers(0, 2, 300)
        p = rng.random(300)
        points = sweep_thresholds(y, p, self.GRID)
        feasible = [pt for pt in points if pt.recall >= 0.4]
        rec = recommend_min_recall(points, min_recall=0.4)
        if feasible:
            assert rec.available is True
            assert rec.recall >= 0.4
            assert rec.precision == pytest.approx(
                max(pt.precision for pt in feasible)
            )


class TestMinPrecisionStrategy:
    GRID = threshold_grid(0.05, 0.95, 0.05)

    def test_satisfies_constraint_with_max_recall(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        rec = recommend_min_precision(points, min_precision=0.5)
        assert rec.available is True
        # At t=0.4: TP=4, FN=0, FP=1 → P=0.8 ≥ 0.5 with R=1.0.
        assert rec.threshold == pytest.approx(0.4)
        assert rec.precision >= 0.5
        assert rec.recall == pytest.approx(1.0)

    def test_unsatisfiable_constraint(self):
        # Restricted grid: the maximum achievable precision is 0.8
        # (at t=0.4: TP=4, FP=1), so 0.99 is unreachable.
        grid = threshold_grid(0.1, 0.5, 0.1)
        points = sweep_thresholds(Y_TRUE, Y_PROB, grid)
        rec = recommend_min_precision(points, min_precision=0.99)
        assert rec.available is False
        assert "No threshold" in rec.reason

    def test_not_configured(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        rec = recommend_min_precision(points, min_precision=None)
        assert rec.available is False
        assert "not configured" in rec.reason

    def test_all_feasible_points_satisfy_constraint(self):
        rng = np.random.default_rng(13)
        y = rng.integers(0, 2, 300)
        p = rng.random(300)
        points = sweep_thresholds(y, p, self.GRID)
        feasible = [pt for pt in points if pt.precision >= 0.3]
        rec = recommend_min_precision(points, min_precision=0.3)
        if feasible:
            assert rec.available is True
            assert rec.precision >= 0.3
            assert rec.recall == pytest.approx(
                max(pt.recall for pt in feasible)
            )


class TestBuildRecommendations:
    GRID = threshold_grid(0.05, 0.95, 0.05)

    def test_all_four_strategies_present(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        recs = build_recommendations(points, _default_config())
        strategies = {r.strategy for r in recs}
        assert strategies == {"max_f1", "min_cost", "min_recall", "min_precision"}

    def test_unconfigured_strategies_unavailable(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        recs = build_recommendations(points, _default_config())
        by_strategy = {r.strategy: r for r in recs}
        assert by_strategy["min_cost"].available is False
        assert by_strategy["min_recall"].available is False
        assert by_strategy["min_precision"].available is False
        assert by_strategy["max_f1"].available is True

    def test_configured_strategies_available(self):
        points = sweep_thresholds(Y_TRUE, Y_PROB, self.GRID)
        config = _default_config(
            false_negative_cost=10.0,
            false_positive_cost=1.0,
            min_recall=0.7,
            min_precision=0.5,
        )
        recs = build_recommendations(points, config)
        assert all(r.available for r in recs)

# ═══════════════════════════════════════════════════════════════════
# O. Missing labels rejected safely
# ═══════════════════════════════════════════════════════════════════


class TestLabelValidation:
    def test_nan_labels_rejected(self):
        y = Y_TRUE.astype(float)
        y[2] = np.nan
        with pytest.raises(LabelError, match="missing"):
            classification_metrics(y, PREDS_05)

    def test_none_labels_rejected(self):
        with pytest.raises(LabelError):
            classification_metrics([None, 1, 0], [0, 1, 0])

    def test_non_binary_labels_rejected(self):
        with pytest.raises(LabelError, match="binary"):
            classification_metrics([0, 1, 2], [0, 1, 0])

    def test_empty_labels_rejected(self):
        with pytest.raises(LabelError, match="empty"):
            classification_metrics([], [])

    def test_string_labels_rejected(self):
        with pytest.raises(LabelError):
            classification_metrics(["a", "b"], [0, 1])

    def test_float_binary_labels_accepted(self):
        result = validate_labels(np.array([0.0, 1.0, 1.0]))
        assert result.tolist() == [0, 1, 1]

    def test_labels_never_silently_imputed(self):
        y = Y_TRUE.astype(float)
        y[0] = np.nan
        with pytest.raises(LabelError):
            ranking_metrics(y, Y_PROB)
        with pytest.raises(LabelError):
            confusion_counts(y, PREDS_05)
        with pytest.raises(LabelError):
            brier_score(y, Y_PROB)


# ═══════════════════════════════════════════════════════════════════
# P / Q. Invalid probabilities & probability bounds
# ═══════════════════════════════════════════════════════════════════


class TestProbabilityValidation:
    def test_above_bound_rejected(self):
        with pytest.raises(ProbabilityError, match=r"\[0, 1\]"):
            ranking_metrics(Y_TRUE, np.append(Y_PROB[:-1], 1.5))

    def test_below_bound_rejected(self):
        with pytest.raises(ProbabilityError, match=r"\[0, 1\]"):
            ranking_metrics(Y_TRUE, np.append(Y_PROB[:-1], -0.01))

    def test_nan_probabilities_rejected(self):
        p = Y_PROB.astype(float)
        p[1] = np.nan
        with pytest.raises(ProbabilityError, match="missing"):
            ranking_metrics(Y_TRUE, p)

    def test_inf_probabilities_rejected(self):
        with pytest.raises(ProbabilityError, match="finite"):
            ranking_metrics(Y_TRUE, np.array([0.5, np.inf]))

    def test_empty_probabilities_rejected(self):
        with pytest.raises(ProbabilityError, match="empty"):
            ranking_metrics(Y_TRUE, [])
        # Empty labels are rejected first, with an equally clear error.
        with pytest.raises(LabelError, match="empty"):
            ranking_metrics([], [])

    def test_boundary_probabilities_accepted(self):
        result = validate_probabilities(np.array([0.0, 1.0, 0.5]))
        assert result.tolist() == [0.0, 1.0, 0.5]

    def test_length_mismatch_rejected(self):
        with pytest.raises(ProbabilityError, match="same length"):
            classification_metrics(Y_TRUE, PREDS_05[:-1])

    def test_none_probabilities_rejected(self):
        with pytest.raises(ProbabilityError):
            ranking_metrics(Y_TRUE, [None, 0.5, 1.0, 0.2, 0.3, 0.4, 0.5, 0.6])


# ═══════════════════════════════════════════════════════════════════
# R / S. Calibration and Brier score
# ═══════════════════════════════════════════════════════════════════


class TestCalibration:
    def test_brier_hand_computed(self):
        # Sum of squared errors = 0.96 over 8 samples.
        assert brier_score(Y_TRUE, Y_PROB) == pytest.approx(0.12)

    def test_brier_matches_sklearn(self):
        assert brier_score(Y_TRUE, Y_PROB) == pytest.approx(
            brier_score_loss(Y_TRUE, Y_PROB)
        )

    def test_brier_perfect(self):
        p = Y_TRUE.astype(float)
        assert brier_score(Y_TRUE, p) == pytest.approx(0.0)

    def test_brier_worst(self):
        p = (1 - Y_TRUE).astype(float)
        assert brier_score(Y_TRUE, p) == pytest.approx(1.0)

    def test_reliability_bins_match_sklearn(self):
        bins = reliability_bins(Y_TRUE, Y_PROB, n_bins=10)
        prob_true, prob_pred = calibration_curve(
            Y_TRUE, Y_PROB, n_bins=10, strategy="uniform"
        )
        assert len(bins) == len(prob_true)
        for entry, expected_pred, expected_true in zip(
            bins, prob_pred, prob_true
        ):
            assert entry["mean_predicted_probability"] == pytest.approx(
                float(expected_pred)
            )
            assert entry["fraction_positive"] == pytest.approx(
                float(expected_true)
            )

    def test_reliability_bins_counts_sum(self):
        bins = reliability_bins(Y_TRUE, Y_PROB, n_bins=10)
        assert sum(b["count"] for b in bins) == len(Y_TRUE)

    def test_reliability_bin_semantics(self):
        bins = reliability_bins(Y_TRUE, Y_PROB, n_bins=10)
        by_lower = {round(b["bin_lower"], 2): b for b in bins}
        # sklearn convention: a probability on an interior edge
        # belongs to the bin below it — the 0.8–0.9 bin contains only
        # prob 0.9 with label 1.
        assert by_lower[0.8]["count"] == 1
        assert by_lower[0.8]["mean_predicted_probability"] == pytest.approx(0.9)
        assert by_lower[0.8]["fraction_positive"] == pytest.approx(1.0)
        # The 0.4–0.5 bin contains only prob 0.5 with label 0.
        assert by_lower[0.4]["count"] == 1
        assert by_lower[0.4]["fraction_positive"] == pytest.approx(0.0)

    def test_calibration_with_imbalanced_data(self):
        # 970 legitimate @ p=0.25, 30 fraud @ p=0.75 — heavily imbalanced.
        y = np.array([0] * 970 + [1] * 30)
        p = np.array([0.25] * 970 + [0.75] * 30)
        cal = calibration_metrics(y, p, n_bins=10)
        bins = cal["reliability_bins"]
        assert sum(b["count"] for b in bins) == 1000
        for b in bins:
            assert 0.0 <= b["fraction_positive"] <= 1.0
        # Fraud-heavy bin is perfectly separated in this synthetic case.
        fraud_bin = next(
            b for b in bins if abs(b["mean_predicted_probability"] - 0.75) < 1e-9
        )
        legit_bin = next(
            b for b in bins if abs(b["mean_predicted_probability"] - 0.25) < 1e-9
        )
        assert fraud_bin["fraction_positive"] == pytest.approx(1.0)
        assert legit_bin["fraction_positive"] == pytest.approx(0.0)
        assert cal["brier_score"] == pytest.approx(0.25 ** 2)

    def test_calibration_monotone_on_separable_data(self):
        rng = np.random.default_rng(42)
        n_legit, n_fraud = 800, 200
        y = np.array([0] * n_legit + [1] * n_fraud)
        p = np.concatenate(
            [rng.uniform(0.0, 0.4, n_legit), rng.uniform(0.6, 1.0, n_fraud)]
        )
        cal = calibration_metrics(y, p, n_bins=10)
        bins = cal["reliability_bins"]
        assert len(bins) >= 2
        # Fraction-positive must grow with the predicted probability.
        fractions = [b["fraction_positive"] for b in bins]
        assert fractions == sorted(fractions)

    def test_recalibration_flags(self):
        cal = calibration_metrics(Y_TRUE, Y_PROB, n_bins=5)
        assert cal["recalibration_applied"] is False
        assert cal["probability_type"] == "raw_model_probability"
        assert "RAW" in cal["note"]

    def test_bins_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            reliability_bins(Y_TRUE, Y_PROB, n_bins=1)
        with pytest.raises(ValueError):
            calibration_metrics(Y_TRUE, Y_PROB, n_bins=51)


# ═══════════════════════════════════════════════════════════════════
# T / U. Model-version & checksum traceability
# ═══════════════════════════════════════════════════════════════════


class TestModelTraceability:
    def test_report_identity_from_governance(
        self, stub_bundle, stub_identity, stub_features
    ):
        report = build_report(
            bundle=stub_bundle,
            identity=stub_identity,
            X=stub_features,
            y=Y_TRUE,
            config=_default_config(),
        )
        assert report.model_identity == stub_identity.to_dict()

    def test_checksum_preserved_in_report(
        self, stub_bundle, stub_identity, stub_features
    ):
        report = build_report(
            bundle=stub_bundle,
            identity=stub_identity,
            X=stub_features,
            y=Y_TRUE,
            config=_default_config(),
        )
        assert (
            report.model_identity["artifact_checksum"]
            == stub_identity.artifact_checksum
        )
        assert len(report.model_identity["artifact_checksum"]) == 64

    def test_identity_mismatch_rejected(self, stub_bundle, stub_features):
        wrong = StubIdentity(version="attacker-claimed-v9")
        with pytest.raises(EvaluationError, match="identity mismatch"):
            build_report(
                bundle=stub_bundle,
                identity=wrong,
                X=stub_features,
                y=Y_TRUE,
                config=_default_config(),
            )

    def test_schema_version_preserved(
        self, stub_bundle, stub_identity, stub_features
    ):
        report = build_report(
            bundle=stub_bundle,
            identity=stub_identity,
            X=stub_features,
            y=Y_TRUE,
            config=_default_config(),
        )
        assert (
            report.model_identity["feature_schema_version"]
            == stub_identity.feature_schema_version
        )

    def test_metrics_functions_accept_no_model_claim(self):
        # Metric functions compute from data only — a caller cannot
        # attach a fabricated model version to them.
        import inspect

        for func in (classification_metrics, ranking_metrics, calibration_metrics):
            params = inspect.signature(func).parameters
            assert "model_version" not in params
            assert "model" not in params
            assert "artifact" not in params


# ═══════════════════════════════════════════════════════════════════
# V / W. Reproducibility & determinism
# ═══════════════════════════════════════════════════════════════════


class TestReproducibility:
    def _build(self, stub_bundle, stub_identity, stub_features, config):
        return build_report(
            bundle=stub_bundle,
            identity=stub_identity,
            X=stub_features,
            y=Y_TRUE,
            config=config,
            dataset_metadata={"split_timestamp": 12345, "test_fraction": 0.2},
        )

    def test_report_contains_reproducibility_metadata(
        self, stub_bundle, stub_identity, stub_features
    ):
        report = self._build(
            stub_bundle, stub_identity, stub_features, _default_config()
        )
        repro = report.reproducibility
        assert repro["evaluation_config"] == _default_config().to_dict()
        assert "dataset_source" in repro
        assert "tie_break_rule" in repro
        assert repro["deterministic"] is True

    def test_dataset_metadata_counts(
        self, stub_bundle, stub_identity, stub_features
    ):
        report = self._build(
            stub_bundle, stub_identity, stub_features, _default_config()
        )
        ds = report.dataset_metadata
        assert ds["n_samples"] == len(Y_TRUE)
        assert ds["n_fraud"] == int(Y_TRUE.sum())
        assert ds["n_legitimate"] == len(Y_TRUE) - int(Y_TRUE.sum())
        assert ds["fraud_prevalence"] == pytest.approx(0.5)
        assert ds["split_timestamp"] == 12345
        assert ds["test_fraction"] == pytest.approx(0.2)

    def test_deterministic_repeated_evaluation(
        self, stub_bundle, stub_identity, stub_features
    ):
        config = _default_config(
            false_negative_cost=10.0,
            false_positive_cost=1.0,
            min_recall=0.7,
            min_precision=0.5,
        )
        first = self._build(stub_bundle, stub_identity, stub_features, config)
        second = self._build(stub_bundle, stub_identity, stub_features, config)
        d1 = first.to_dict()
        d2 = second.to_dict()
        # Timestamps legitimately differ; everything else must be equal.
        d1.pop("evaluation_timestamp")
        d2.pop("evaluation_timestamp")
        assert d1 == d2

    def test_config_recorded_in_report(
        self, stub_bundle, stub_identity, stub_features
    ):
        config = _default_config(
            threshold_start=0.1, threshold_stop=0.9, threshold_step=0.1
        )
        report = self._build(stub_bundle, stub_identity, stub_features, config)
        assert report.reproducibility["evaluation_config"]["threshold_start"] == 0.1
        assert report.reproducibility["evaluation_config"]["threshold_step"] == 0.1
        # Sweep honours the configured grid.
        thresholds = [pt["threshold"] for pt in report.threshold_analysis]
        assert thresholds[0] == pytest.approx(0.1)
        assert thresholds[-1] == pytest.approx(0.9)


# ═══════════════════════════════════════════════════════════════════
# X. Leakage protection
# ═══════════════════════════════════════════════════════════════════


class TestLeakageProtection:
    def test_scoring_ignores_forbidden_columns(self, stub_bundle):
        # Labels / IDs must never be usable as features: the scorer
        # selects only the bundle's declared feature columns.
        clean = pd.DataFrame({"score": Y_PROB})
        polluted = pd.DataFrame(
            {
                "score": Y_PROB,
                "isFraud": Y_TRUE,  # target leakage attempt
                "TransactionID": np.arange(8),
            }
        )
        assert np.array_equal(
            score_with_bundle(stub_bundle, clean),
            score_with_bundle(stub_bundle, polluted),
        )

    def test_fit_preprocessing_never_called_during_evaluation(
        self, stub_bundle, stub_identity, stub_features, monkeypatch
    ):
        import ml.models.baseline as baseline

        def _forbidden(*args, **kwargs):
            raise AssertionError(
                "Evaluation must never fit preprocessing on evaluation data."
            )

        monkeypatch.setattr(baseline, "fit_preprocessing", _forbidden)
        report = build_report(
            bundle=stub_bundle,
            identity=stub_identity,
            X=stub_features,
            y=Y_TRUE,
            config=_default_config(),
        )
        assert report.report_scope == REPORT_SCOPE

    def test_evaluation_features_never_contain_target(self, stub_bundle):
        # score_with_bundle itself refuses to require forbidden columns.
        assert "isFraud" not in stub_bundle.feature_names
        assert "TransactionID" not in stub_bundle.feature_names

    def test_temporal_split_test_strictly_after_train(self):
        from ml.split.splitter import time_based_split, validate_split

        rng = np.random.default_rng(5)
        n = 400
        timestamps = pd.Series(np.sort(rng.integers(1_000, 100_000, n)))
        features = pd.DataFrame({"f1": rng.random(n), "f2": rng.random(n)})
        target = pd.Series(rng.integers(0, 2, n))
        ids = pd.Series(np.arange(n))

        split = time_based_split(
            features=features,
            target=target,
            timestamps=timestamps,
            transaction_ids=ids,
            test_fraction=0.2,
        )
        assert int(split.train_timestamps.max()) < int(split.test_timestamps.min())
        assert "isFraud" not in split.X_test.columns
        checks = validate_split(split)
        failed_names = {c["check"] for c in checks if c["status"] != "PASS"}
        # Temporal ordering, no ID overlap, and column consistency must hold.
        assert "temporal_ordering" not in failed_names
        assert "no_id_overlap" not in failed_names
        assert "feature_columns_match" not in failed_names

    def test_previous_suspicious_count_excludes_current_label(self):
        # The only feature built from the target must use strictly
        # prior transactions (verified at the source of truth).
        from ml.features.historical import compute_previous_suspicious_count

        df = pd.DataFrame(
            {
                "card1": [1, 1, 1, 2],
                "isFraud": [1, 0, 1, 1],
                "TransactionDT": [10, 20, 30, 40],
            }
        )
        result = compute_previous_suspicious_count(df)
        assert result.tolist() == [0, 1, 1, 0]


# ═══════════════════════════════════════════════════════════════════
# Y / Z / AC. Production safety
# ═══════════════════════════════════════════════════════════════════


class TestProductionSafety:
    def test_threshold_not_modified(
        self, stub_bundle, stub_identity, stub_features
    ):
        before = stub_bundle.threshold
        build_report(
            bundle=stub_bundle,
            identity=stub_identity,
            X=stub_features,
            y=Y_TRUE,
            config=_default_config(),
        )
        assert stub_bundle.threshold == before

    def test_report_records_production_threshold(
        self, stub_identity, stub_features
    ):
        report = build_report(
            bundle=StubBundle(threshold=0.42),
            identity=stub_identity,
            X=stub_features,
            y=Y_TRUE,
            config=_default_config(),
        )
        assert report.production_threshold == pytest.approx(0.42)
        assert report.classification_metrics["threshold"] == pytest.approx(0.42)
        assert "NOT modified" in report.classification_metrics["threshold_source"]

    def test_report_disclaimer_present(
        self, stub_bundle, stub_identity, stub_features
    ):
        report = build_report(
            bundle=stub_bundle,
            identity=stub_identity,
            X=stub_features,
            y=Y_TRUE,
            config=_default_config(),
        )
        assert "EVALUATION / RECOMMENDATION ONLY" in report.disclaimer
        assert "DO NOT automatically change production" in report.disclaimer

    def test_risk_aggregation_unchanged(
        self, stub_bundle, stub_identity, stub_features
    ):
        build_report(
            bundle=stub_bundle,
            identity=stub_identity,
            X=stub_features,
            y=Y_TRUE,
            config=_default_config(),
        )
        from ml.risk import aggregator

        assert aggregator.RISK_THRESHOLDS == [
            (70, "HIGH", "HOLD"),
            (30, "MEDIUM", "VERIFY"),
        ]
        assert aggregator.DEFAULT_WEIGHT_ML == pytest.approx(0.50)
        assert aggregator.DEFAULT_WEIGHT_BEHAVIOUR == pytest.approx(0.30)
        assert aggregator.DEFAULT_WEIGHT_RULE == pytest.approx(0.20)

    def test_evaluation_module_has_no_production_side_effect_imports(self):
        import ast
        import inspect

        import ml.evaluation.config as cfg
        import ml.evaluation.metrics as met
        import ml.evaluation.runner as run
        import ml.evaluation.thresholds as thr

        forbidden = ("ml.risk", "ml.api", "ml.monitoring", "backend")
        for module in (cfg, met, thr, run):
            tree = ast.parse(inspect.getsource(module))
            imported: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            for name in imported:
                assert not name.startswith(forbidden), (
                    module.__name__,
                    name,
                )

    def test_build_report_performs_no_file_io(
        self, stub_bundle, stub_identity, stub_features, monkeypatch
    ):
        def _guarded_open(*args, **kwargs):
            raise AssertionError(
                "build_report must not perform file I/O (offline, read-only)."
            )

        monkeypatch.setattr("builtins.open", _guarded_open)
        report = build_report(
            bundle=stub_bundle,
            identity=stub_identity,
            X=stub_features,
            y=Y_TRUE,
            config=_default_config(),
        )
        assert report.report_scope == REPORT_SCOPE

    def test_monitoring_counters_unchanged_by_evaluation(
        self, stub_bundle, stub_identity, stub_features
    ):
        from ml.monitoring.metrics import metrics

        before = metrics.snapshot()
        build_report(
            bundle=stub_bundle,
            identity=stub_identity,
            X=stub_features,
            y=Y_TRUE,
            config=_default_config(),
        )
        after = metrics.snapshot()
        assert before == after


# ═══════════════════════════════════════════════════════════════════
# Evaluation configuration (EVAL_* namespace)
# ═══════════════════════════════════════════════════════════════════

from ml.evaluation.config import (
    DEFAULT_CALIBRATION_BINS,
    DEFAULT_THRESHOLD_START,
    DEFAULT_THRESHOLD_STEP,
    DEFAULT_THRESHOLD_STOP,
    MAX_THRESHOLD_POINTS,
)


class TestEvaluationConfig:
    def test_from_env_defaults(self):
        config = EvaluationConfig.from_env()
        config.validate()
        assert config.threshold_start == DEFAULT_THRESHOLD_START
        assert config.threshold_stop == DEFAULT_THRESHOLD_STOP
        assert config.threshold_step == DEFAULT_THRESHOLD_STEP
        assert config.min_recall is None
        assert config.min_precision is None
        assert config.false_negative_cost is None
        assert config.false_positive_cost is None
        assert config.calibration_bins == DEFAULT_CALIBRATION_BINS

    def test_from_env_reads_all_variables(self, monkeypatch):
        monkeypatch.setenv("EVAL_THRESHOLD_START", "0.1")
        monkeypatch.setenv("EVAL_THRESHOLD_STOP", "0.9")
        monkeypatch.setenv("EVAL_THRESHOLD_STEP", "0.2")
        monkeypatch.setenv("EVAL_MIN_RECALL", "0.8")
        monkeypatch.setenv("EVAL_MIN_PRECISION", "0.6")
        monkeypatch.setenv("EVAL_FN_COST", "150.0")
        monkeypatch.setenv("EVAL_FP_COST", "2.5")
        monkeypatch.setenv("EVAL_CALIBRATION_BINS", "20")
        config = EvaluationConfig.from_env()
        config.validate()
        assert config.threshold_start == pytest.approx(0.1)
        assert config.threshold_stop == pytest.approx(0.9)
        assert config.threshold_step == pytest.approx(0.2)
        assert config.min_recall == pytest.approx(0.8)
        assert config.min_precision == pytest.approx(0.6)
        assert config.false_negative_cost == pytest.approx(150.0)
        assert config.false_positive_cost == pytest.approx(2.5)
        assert config.calibration_bins == 20

    def test_from_env_empty_values_fall_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("EVAL_THRESHOLD_START", "   ")
        monkeypatch.setenv("EVAL_MIN_RECALL", "")
        config = EvaluationConfig.from_env()
        config.validate()
        assert config.threshold_start == DEFAULT_THRESHOLD_START
        assert config.min_recall is None

    def test_from_env_invalid_number_names_variable(self, monkeypatch):
        monkeypatch.setenv("EVAL_THRESHOLD_START", "abc")
        with pytest.raises(EvaluationConfigError, match="EVAL_THRESHOLD_START"):
            EvaluationConfig.from_env()

    def test_from_env_invalid_bins_names_variable(self, monkeypatch):
        monkeypatch.setenv("EVAL_CALIBRATION_BINS", "ten")
        with pytest.raises(EvaluationConfigError, match="EVAL_CALIBRATION_BINS"):
            EvaluationConfig.from_env()

    def test_validation_rejects_inverted_range(self):
        with pytest.raises(EvaluationConfigError):
            _default_config(threshold_start=0.9, threshold_stop=0.1)

    def test_validation_rejects_out_of_bounds(self):
        with pytest.raises(EvaluationConfigError):
            _default_config(threshold_start=-0.1)
        with pytest.raises(EvaluationConfigError):
            _default_config(threshold_stop=1.5)

    def test_validation_rejects_non_positive_step(self):
        with pytest.raises(EvaluationConfigError):
            _default_config(threshold_step=0.0)
        with pytest.raises(EvaluationConfigError):
            _default_config(threshold_step=-0.05)

    def test_validation_rejects_step_larger_than_range(self):
        with pytest.raises(EvaluationConfigError):
            _default_config(
                threshold_start=0.1, threshold_stop=0.5, threshold_step=0.6
            )

    def test_validation_rejects_unbounded_sweep(self):
        config = EvaluationConfig(
            threshold_start=0.0, threshold_stop=1.0, threshold_step=0.001
        )
        with pytest.raises(EvaluationConfigError, match="bounded"):
            config.validate()

    def test_validation_rejects_bad_calibration_bins(self):
        with pytest.raises(EvaluationConfigError):
            _default_config(calibration_bins=1)
        with pytest.raises(EvaluationConfigError):
            _default_config(calibration_bins=51)

    def test_validation_rejects_bad_constraints(self):
        with pytest.raises(EvaluationConfigError):
            _default_config(min_recall=1.2)
        with pytest.raises(EvaluationConfigError):
            _default_config(min_precision=-0.1)

    def test_validation_rejects_negative_costs(self):
        with pytest.raises(EvaluationConfigError):
            _default_config(false_negative_cost=-1.0)
        with pytest.raises(EvaluationConfigError):
            _default_config(false_positive_cost=-0.5)

    def test_namespace_separated_from_production_variables(self, monkeypatch):
        # Production-style variables must never leak into evaluation config.
        monkeypatch.setenv("ML_MODEL_DIR", "C:/somewhere/else")
        monkeypatch.setenv("ML_THRESHOLD", "0.99")
        monkeypatch.setenv("ML_HISTORY_DB_PATH", "C:/nowhere/db.sqlite")
        config = EvaluationConfig.from_env()
        config.validate()
        assert config.to_dict() == EvaluationConfig().to_dict()


# ═══════════════════════════════════════════════════════════════════
# AA. No sensitive data in evaluation output
# ═══════════════════════════════════════════════════════════════════


class TestSensitiveDataSafety:
    def _report(self, stub_bundle, stub_identity, stub_features):
        return build_report(
            bundle=stub_bundle,
            identity=stub_identity,
            X=stub_features,
            y=Y_TRUE,
            config=_default_config(),
        )

    def test_report_is_json_serializable(
        self, stub_bundle, stub_identity, stub_features
    ):
        report = self._report(stub_bundle, stub_identity, stub_features)
        text = json.dumps(report.to_dict(), sort_keys=True)
        assert '"report_scope"' in text

    def test_no_customer_or_card_level_fields(
        self, stub_bundle, stub_identity, stub_features
    ):
        text = json.dumps(
            self._report(stub_bundle, stub_identity, stub_features).to_dict()
        )
        for field in (
            "card1",
            "card2",
            "addr1",
            "addr2",
            "TransactionID",
            "transaction_id",
            "ip_address",
            "device_fingerprint",
            "merchant_name",
            "customer_id",
            "emaildomain",
        ):
            assert field not in text, field

    def test_no_secrets_or_absolute_paths(
        self, stub_bundle, stub_identity, stub_features
    ):
        text = json.dumps(
            self._report(stub_bundle, stub_identity, stub_features).to_dict()
        )
        for marker in (
            "C:\\",
            "c:\\",
            "/home/",
            "/Users/",
            "/tmp/",
            "password",
            "secret",
            "token",
            "api_key",
            "Bearer ",
        ):
            assert marker not in text, marker

    def test_output_is_bounded(self, stub_bundle, stub_identity, stub_features):
        report = self._report(stub_bundle, stub_identity, stub_features)
        assert len(report.threshold_analysis) <= MAX_THRESHOLD_POINTS
        assert len(report.calibration_metrics["reliability_bins"]) <= 50


# ═══════════════════════════════════════════════════════════════════
# AB. No client-controlled evaluation surface
# ═══════════════════════════════════════════════════════════════════


class TestNoClientControl:
    def test_no_evaluation_api_routes(self):
        from ml.api.app import app

        for route in app.routes:
            path = getattr(route, "path", "")
            assert "eval" not in path.lower(), path

    def test_runner_accepts_no_untrusted_artifact_paths(self):
        import inspect

        from ml.evaluation.runner import run_offline_evaluation

        params = inspect.signature(run_offline_evaluation).parameters
        assert "manifest_path" not in params
        assert "model_path" not in params
        assert "artifact_path" not in params
        assert "bundle" not in params
        assert "X" not in params
        assert "y" not in params

    def test_dataset_loader_takes_no_caller_paths(self):
        import inspect

        from ml.evaluation.runner import load_holdout_test_set

        params = inspect.signature(load_holdout_test_set).parameters
        assert not params  # approved dataset source only

    def test_build_report_cannot_override_production_threshold(self):
        import inspect

        params = inspect.signature(build_report).parameters
        assert "threshold" not in params

    def test_cli_has_output_flag_only(self):
        import inspect

        from ml.evaluation import runner

        source = inspect.getsource(runner)
        assert source.count("add_argument") == 1
        assert '"--output"' in source


# ═══════════════════════════════════════════════════════════════════
# AD. Edge cases and degenerate label sets
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_all_positive_labels(
        self, stub_bundle, stub_identity, stub_features
    ):
        y = np.ones(len(Y_PROB))
        report = build_report(
            bundle=stub_bundle,
            identity=stub_identity,
            X=stub_features,
            y=y,
            config=_default_config(),
        )
        cm = report.classification_metrics
        assert cm["n_samples"] == 8
        assert cm["n_fraud"] == 8
        assert cm["tn"] == 0
        assert cm["fp"] == 0
        assert report.ranking_metrics is None
        assert "single class" in report.ranking_unavailable_reason

    def test_all_negative_labels(
        self, stub_bundle, stub_identity, stub_features
    ):
        y = np.zeros(len(Y_PROB))
        report = build_report(
            bundle=stub_bundle,
            identity=stub_identity,
            X=stub_features,
            y=y,
            config=_default_config(),
        )
        cm = report.classification_metrics
        assert cm["n_fraud"] == 0
        assert cm["tp"] == 0
        assert cm["fn"] == 0
        assert report.ranking_metrics is None
        assert "single class" in report.ranking_unavailable_reason

    def test_single_sample_report(self, stub_bundle, stub_identity):
        X = pd.DataFrame({"score": [0.7]})
        report = build_report(
            bundle=stub_bundle,
            identity=stub_identity,
            X=X,
            y=[1],
            config=_default_config(),
        )
        assert report.dataset_metadata["n_samples"] == 1
        assert report.classification_metrics["recall"] == pytest.approx(1.0)
        assert report.classification_metrics["precision"] == pytest.approx(1.0)

    def test_empty_dataset_rejected(self, stub_bundle, stub_identity):
        with pytest.raises(LabelError):
            build_report(
                bundle=stub_bundle,
                identity=stub_identity,
                X=pd.DataFrame({"score": []}),
                y=[],
                config=_default_config(),
            )


# ═══════════════════════════════════════════════════════════════════
# Real-model integration (skipped when the model is not available)
# ═══════════════════════════════════════════════════════════════════


def _model_available() -> bool:
    try:
        from ml.predict.registry import ModelRegistry

        registry = ModelRegistry()
        registry.activate_from_manifest()
        return registry.is_ready
    except Exception:
        return False


MODEL_AVAILABLE = _model_available()
requires_model = pytest.mark.skipif(
    not MODEL_AVAILABLE, reason="Model not available"
)


def _synthetic_real_features(bundle, n: int = 200):
    """Deterministic feature matrix matching the real bundle schema."""
    rng = np.random.default_rng(42)
    encoders = getattr(bundle.preprocessing, "label_encoders", {}) or {}
    data = {}
    for name in bundle.feature_names:
        encoder = encoders.get(name)
        if encoder is not None and len(encoder.classes_):
            data[name] = rng.choice(encoder.classes_, size=n)
        else:
            data[name] = rng.uniform(0.0, 1.0, n)
    X = pd.DataFrame(data)
    y = pd.Series(np.arange(n) % 2)
    return X, y


@requires_model
class TestRealModelIntegration:
    def _activate(self):
        from ml.predict.registry import ModelRegistry

        registry = ModelRegistry()
        identity = registry.activate_from_manifest()
        return registry, identity, registry.bundle

    def test_identity_matches_manifest(self):
        from ml.predict.integrity import default_model_directory, load_manifest

        _, identity, _ = self._activate()
        manifest = load_manifest(default_model_directory())
        assert identity.model_version == manifest.model_version
        assert identity.artifact_checksum == manifest.artifact_checksum

    def test_report_traceability_with_real_model(self):
        from ml.predict.integrity import default_model_directory, load_manifest

        _, identity, bundle = self._activate()
        manifest = load_manifest(default_model_directory())
        X, y = _synthetic_real_features(bundle)
        report = build_report(
            bundle=bundle,
            identity=identity,
            X=X,
            y=y,
            config=_default_config(),
        )
        assert report.model_identity["model_version"] == manifest.model_version
        assert (
            report.model_identity["artifact_checksum"]
            == manifest.artifact_checksum
        )
        assert report.production_threshold == pytest.approx(bundle.threshold)
        assert report.report_scope == REPORT_SCOPE
        json.dumps(report.to_dict())  # JSON-safe with the real model

    def test_artifact_unchanged_after_evaluation(self):
        from ml.predict.integrity import (
            compute_checksum,
            default_model_directory,
            load_manifest,
        )

        directory = default_model_directory()
        manifest = load_manifest(directory)
        artifact_path = directory / manifest.artifact_filename
        before = compute_checksum(artifact_path)

        _, identity, bundle = self._activate()
        X, y = _synthetic_real_features(bundle)
        build_report(
            bundle=bundle,
            identity=identity,
            X=X,
            y=y,
            config=_default_config(),
        )

        after = compute_checksum(artifact_path)
        assert after == before == manifest.artifact_checksum

    def test_manifest_file_unchanged_after_evaluation(self):
        from ml.predict.integrity import default_model_directory, load_manifest

        directory = default_model_directory()
        manifest_path = directory / "model_manifest.json"
        before = manifest_path.read_bytes()
        manifest = load_manifest(directory)

        _, identity, bundle = self._activate()
        X, y = _synthetic_real_features(bundle)
        build_report(
            bundle=bundle,
            identity=identity,
            X=X,
            y=y,
            config=_default_config(),
        )

        assert manifest_path.read_bytes() == before
        reloaded = load_manifest(directory)
        assert reloaded.model_version == manifest.model_version
        assert reloaded.artifact_checksum == manifest.artifact_checksum


@requires_model
class TestLiveServiceUnchanged:
    PAYLOAD = {
        "amount": 250.0,
        "currency": "USD",
        "merchant_name": "Step47 Probe Merchant",
        "merchant_category": "5732",
        "transaction_type": "purchase",
        "location_country": "US",
        "location_city": "Testville",
        "device_fingerprint": "fp_step47_probe",
        "device_type": "desktop",
        "ip_address": "192.168.1.50",
        "timestamp": 1_000_000,
        "card1": 47001,
    }

    DETERMINISTIC_FIELDS = (
        "fraud_probability",
        "fraud_prediction",
        "threshold",
        "model_version",
        "ml_score",
        "behaviour_score",
        "rule_score",
        "risk_score",
        "risk_level",
        "decision",
    )

    def _client(self):
        from fastapi.testclient import TestClient

        from ml.api.app import app

        return TestClient(app)

    def _evaluate(self):
        from ml.predict.registry import ModelRegistry

        registry = ModelRegistry()
        identity = registry.activate_from_manifest()
        bundle = registry.bundle
        X, y = _synthetic_real_features(bundle)
        report = build_report(
            bundle=bundle,
            identity=identity,
            X=X,
            y=y,
            config=_default_config(),
        )
        assert report.report_scope == REPORT_SCOPE

    def test_predict_identical_before_and_after_evaluation(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ML_HISTORY_DB_PATH", str(tmp_path / "history.db"))
        with self._client() as client:
            before = client.post("/predict", json=self.PAYLOAD)
            assert before.status_code == 200
            before_json = before.json()

            self._evaluate()

            after = client.post("/predict", json=self.PAYLOAD)
            assert after.status_code == 200
            after_json = after.json()

            for field in self.DETERMINISTIC_FIELDS:
                assert after_json[field] == before_json[field], field

    def test_monitoring_counters_only_incremented_by_probes(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ML_HISTORY_DB_PATH", str(tmp_path / "history.db"))
        with self._client() as client:
            probe = client.post("/predict", json=self.PAYLOAD)
            assert probe.status_code == 200
            before = client.get("/metrics").json()

            self._evaluate()

            mid = client.get("/metrics").json()
            assert mid["total_requests"] == before["total_requests"]

            probe = client.post("/predict", json=self.PAYLOAD)
            assert probe.status_code == 200
            after = client.get("/metrics").json()
            assert after["total_requests"] == mid["total_requests"] + 1

    def test_health_and_ready_report_active_model(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ML_HISTORY_DB_PATH", str(tmp_path / "history.db"))
        from ml.predict.integrity import default_model_directory, load_manifest

        manifest = load_manifest(default_model_directory())
        with self._client() as client:
            self._evaluate()

            health = client.get("/health").json()
            ready = client.get("/ready").json()
            assert health["status"] == "ready"
            assert ready["status"] == "ready"
            for payload in (health, ready):
                assert payload["model_version"] == manifest.model_version
                assert (
                    payload["model_identity"]["artifact_checksum"]
                    == manifest.artifact_checksum
                )
