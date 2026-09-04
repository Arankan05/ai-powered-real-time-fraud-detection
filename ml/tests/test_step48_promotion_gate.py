"""Step 48 — Automated model validation & promotion gate tests.

Covers the Step 48 specification:

* promotion policy configuration (PROMO_* parsing, defaults, disabling,
  validation errors — fail closed)
* candidate-model validation through the Step 46 governance sequence
  (missing artifact, invalid checksum, malformed manifest, path
  traversal, interface validation, feature-schema / feature-count
  incompatibility)
* production-baseline activation failures
* gate decisions: valid candidate approval, rejection per metric
  (PR-AUC / ROC-AUC / recall / precision / F1 / Brier — absolute and
  relative), multiple simultaneous failures, exact boundary conditions,
  unavailable metrics failing closed
* fail-closed behaviour (invalid policy/evaluation configuration,
  evaluation failures — never APPROVED with incomplete validation)
* deterministic repeated execution
* bounded, safe output (no raw data, labels, paths, or secrets)
* production model NOT modified after approval or rejection
* CLI exit codes (0 approved / 1 rejected / 2 error) and JSON report
* real-model integration (skipped when the artifact is unavailable)

Run from the project root::

    python -m pytest ml/tests/test_step48_promotion_gate.py -v
"""

from __future__ import annotations

import ast
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.config import EvaluationConfig
from ml.evaluation.promotion_gate import (
    DECISION_APPROVED,
    DECISION_REJECTED,
    GATE_DISCLAIMER,
    GATE_REPORT_SCOPE,
    PromotionDecision,
    REPORT_SCHEMA_VERSION,
    evaluate_promotion,
    main,
    run_promotion_gate,
)
from ml.evaluation.promotion_policy import (
    DEFAULT_MAX_BRIER,
    DEFAULT_MAX_F1_DEGRADATION,
    DEFAULT_MIN_PR_AUC,
    DEFAULT_MIN_RECALL,
    DEFAULT_MIN_ROC_AUC,
    PromotionPolicy,
    PromotionPolicyError,
)
from ml.predict.bundle import ModelBundle, save_bundle
from ml.predict.integrity import build_manifest, save_manifest
from ml.predict.registry import ModelRegistry

# ── Environment isolation ──────────────────────────────────────────────

_PROMO_ENV_VARS = (
    "PROMO_MIN_PR_AUC",
    "PROMO_MIN_ROC_AUC",
    "PROMO_MIN_RECALL",
    "PROMO_MIN_PRECISION",
    "PROMO_MIN_F1",
    "PROMO_MAX_BRIER",
    "PROMO_MAX_PR_AUC_DEGRADATION",
    "PROMO_MAX_ROC_AUC_DEGRADATION",
    "PROMO_MAX_RECALL_DEGRADATION",
    "PROMO_MAX_PRECISION_DEGRADATION",
    "PROMO_MAX_F1_DEGRADATION",
    "PROMO_MAX_BRIER_INCREASE",
)

_EVAL_ENV_VARS = (
    "EVAL_THRESHOLD_START",
    "EVAL_THRESHOLD_STOP",
    "EVAL_THRESHOLD_STEP",
    "EVAL_MIN_RECALL",
    "EVAL_MIN_PRECISION",
    "EVAL_FN_COST",
    "EVAL_FP_COST",
    "EVAL_CALIBRATION_BINS",
)


@pytest.fixture(autouse=True)
def _clean_gate_env(monkeypatch):
    """Isolate tests from any PROMO_*/EVAL_*/ML_MODEL_DIR variables."""
    for name in (*_PROMO_ENV_VARS, *_EVAL_ENV_VARS, "ML_MODEL_DIR"):
        monkeypatch.delenv(name, raising=False)


# ── Deterministic test estimator / preprocessing (joblib-safe) ─────────


class _IdentityScaler:
    """Test-only scaler that passes values through unchanged."""

    def transform(self, values):
        return np.asarray(values, dtype=np.float64)


class _PassthroughPreprocessing:
    """Fitted-preprocessing stand-in compatible with apply_preprocessing."""

    def __init__(self) -> None:
        self.label_encoders: dict = {}
        self.numeric_cols = ["score"]
        self.scaler = _IdentityScaler()


class _CurveProbModel:
    """Deterministic, picklable test estimator with selectable curves.

    Maps the single feature column ``score`` to a fraud probability:

    * ``linear`` — ``intercept + slope * score`` (identity by default)
    * ``constant_half`` — always 0.5 (no ranking signal; AUC 0.5)
    * ``inverse`` — ``1 - score`` (perfectly wrong ranking; AUC 0.0)
    * ``confident`` — ``score ** 0.1`` (perfect ranking, badly
      over-confident toward fraud → high Brier score)
    """

    def __init__(
        self,
        mode: str = "linear",
        intercept: float = 0.0,
        slope: float = 1.0,
    ) -> None:
        self.mode = mode
        self.intercept = float(intercept)
        self.slope = float(slope)

    def predict_proba(self, X):
        x = np.asarray(X, dtype=np.float64)[:, 0]
        if self.mode == "linear":
            p = self.intercept + self.slope * x
        elif self.mode == "constant_half":
            p = np.full_like(x, 0.5)
        elif self.mode == "inverse":
            p = 1.0 - x
        elif self.mode == "confident":
            p = np.power(x, 0.1)
        else:  # pragma: no cover — test-helper misuse
            raise ValueError(f"unknown mode: {self.mode!r}")
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


# ── Fixtures / builders ────────────────────────────────────────────────


def _deterministic_eval_data(n: int = 200):
    """Fully deterministic evaluation set: fraud iff score >= 0.5.

    ``score`` is an even grid over (0, 1) so every derived metric is
    analytically predictable and repeatable.
    """
    scores = np.linspace(0.005, 0.995, n)
    y = pd.Series((scores >= 0.5).astype(int))
    X = pd.DataFrame({"score": scores})
    metadata = {
        "dataset_identifier": "synthetic-step48/holdout",
        "split_strategy": "deterministic_grid",
        "n_test_samples": int(n),
    }
    return X, y, metadata


def _build_model_dir(
    base_dir: Path,
    name: str,
    *,
    mode: str = "linear",
    intercept: float = 0.0,
    slope: float = 1.0,
    threshold: float = 0.5,
    model_version: str | None = None,
    feature_names: list[str] | None = None,
    model=None,
) -> Path:
    """Create a model directory (artifact + manifest) in a temp path.

    Uses the real Step 46 machinery: ``save_bundle`` →
    ``build_manifest`` (SHA-256 checksum) → ``save_manifest``.
    """
    directory = base_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    feature_names = feature_names if feature_names is not None else ["score"]
    bundle = ModelBundle(
        model=model if model is not None else _CurveProbModel(
            mode=mode, intercept=intercept, slope=slope
        ),
        preprocessing=_PassthroughPreprocessing(),
        threshold=threshold,
        feature_names=feature_names,
        model_version=model_version or f"{name}-v1.0.0",
    )
    artifact_path = save_bundle(bundle, directory / f"{name}.joblib")
    manifest = build_manifest(
        model_name=name,
        model_version=bundle.model_version,
        artifact_path=artifact_path,
        n_features=bundle.n_features,
        threshold=threshold,
    )
    save_manifest(manifest, directory)
    return directory


def _rewrite_manifest(directory: Path, **changes) -> None:
    """Rewrite a manifest field (for tamper tests)."""
    path = directory / "model_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _policy(**overrides) -> PromotionPolicy:
    """Policy with every gate disabled except the ones given."""
    base = {
        "min_pr_auc": None,
        "min_roc_auc": None,
        "min_recall": None,
        "min_precision": None,
        "min_f1": None,
        "max_brier": None,
        "max_pr_auc_degradation": None,
        "max_roc_auc_degradation": None,
        "max_recall_degradation": None,
        "max_precision_degradation": None,
        "max_f1_degradation": None,
        "max_brier_increase": None,
    }
    base.update(overrides)
    return PromotionPolicy(**base)


def _run_gate(
    monkeypatch,
    production_dir: Path,
    candidate_dir: Path,
    policy: PromotionPolicy | None = None,
    data=None,
    **kwargs,
):
    """Run the full gate against synthetic data (dataset injected)."""
    X, y, metadata = data if data is not None else _deterministic_eval_data()
    monkeypatch.setattr(
        "ml.evaluation.promotion_gate.load_holdout_test_set",
        lambda: (X, y, metadata),
    )
    return run_promotion_gate(
        candidate_dir,
        policy=policy,
        production_model_directory=production_dir,
        **kwargs,
    )


def _snapshot_tree(directory: Path) -> dict[str, bytes]:
    """Snapshot every file (relative path → bytes) under a directory."""
    snapshot: dict[str, bytes] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            snapshot[str(path.relative_to(directory))] = path.read_bytes()
    return snapshot


@pytest.fixture()
def model_setup(tmp_path):
    """Standard setup: perfect production model + good candidate."""
    production_dir = _build_model_dir(
        tmp_path,
        "prod-model",
        intercept=0.0,
        slope=1.0,
        model_version="prod-v1.0.0",
    )
    candidate_dir = _build_model_dir(
        tmp_path,
        "cand-model",
        intercept=0.02,
        slope=1.0,
        model_version="cand-v1.0.0",
    )
    return production_dir, candidate_dir


# ── Promotion policy configuration ─────────────────────────────────────


class TestPromotionPolicy:
    def test_defaults_are_sensible_and_valid(self):
        policy = PromotionPolicy()
        policy.validate()
        assert policy.min_pr_auc == DEFAULT_MIN_PR_AUC
        assert policy.min_roc_auc == DEFAULT_MIN_ROC_AUC
        assert policy.min_recall == DEFAULT_MIN_RECALL
        assert policy.max_brier == DEFAULT_MAX_BRIER
        assert policy.max_f1_degradation == DEFAULT_MAX_F1_DEGRADATION

    def test_from_env_uses_defaults_when_unset(self):
        policy = PromotionPolicy.from_env()
        assert policy == PromotionPolicy()

    def test_from_env_reads_values(self, monkeypatch):
        monkeypatch.setenv("PROMO_MIN_RECALL", "0.85")
        monkeypatch.setenv("PROMO_MAX_BRIER", "0.3")
        monkeypatch.setenv("PROMO_MAX_F1_DEGRADATION", "0.25")
        policy = PromotionPolicy.from_env()
        policy.validate()
        assert policy.min_recall == 0.85
        assert policy.max_brier == 0.3
        assert policy.max_f1_degradation == 0.25

    def test_from_env_none_disables_gate(self, monkeypatch):
        monkeypatch.setenv("PROMO_MIN_RECALL", "none")
        monkeypatch.setenv("PROMO_MAX_BRIER", "off")
        policy = PromotionPolicy.from_env()
        assert policy.min_recall is None
        assert policy.max_brier is None

    def test_from_env_rejects_garbage(self, monkeypatch):
        monkeypatch.setenv("PROMO_MIN_RECALL", "high")
        with pytest.raises(PromotionPolicyError, match="PROMO_MIN_RECALL"):
            PromotionPolicy.from_env()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("min_pr_auc", -0.1),
            ("min_pr_auc", 1.5),
            ("min_roc_auc", 1.5),
            ("min_recall", -0.01),
            ("min_precision", 2.0),
            ("min_f1", 1.01),
            ("max_brier", -0.5),
            ("max_brier", 1.5),
            ("max_pr_auc_degradation", -0.1),
            ("max_roc_auc_degradation", 1.5),
            ("max_recall_degradation", 2.0),
            ("max_precision_degradation", -1.0),
            ("max_f1_degradation", 1.5),
            ("max_brier_increase", 2.0),
        ],
    )
    def test_validate_rejects_out_of_range(self, field, value):
        policy = _policy(**{field: value})
        with pytest.raises(PromotionPolicyError, match=field):
            policy.validate()

    def test_validate_accepts_boundary_values(self):
        policy = _policy(min_recall=0.0, max_brier=1.0, max_f1_degradation=1.0)
        policy.validate()

    def test_to_dict_round_trips_all_fields(self):
        policy = PromotionPolicy()
        data = policy.to_dict()
        assert len(data) == 12
        assert data["min_pr_auc"] == policy.min_pr_auc
        assert PromotionPolicy(**data) == policy

    def test_none_disables_individual_gates(self):
        policy = _policy(min_recall=0.8)
        policy.validate()
        assert policy.min_roc_auc is None
        assert policy.max_brier_increase is None


# ── Candidate-model governance validation (Step 46 sequence) ───────────


class TestCandidateGovernanceValidation:
    def test_valid_candidate_passes_all_gates(self, monkeypatch, model_setup):
        production_dir, candidate_dir = model_setup
        decision = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=PromotionPolicy()
        )
        assert decision.decision == DECISION_APPROVED
        assert decision.failure_stage is None
        assert decision.rejection_reasons == []
        assert all(gate["passed"] for gate in decision.policy_gates)
        assert decision.candidate_metrics is not None
        assert decision.production_metrics is not None
        assert decision.promotion_instruction is not None

    def test_identical_model_candidate_is_approved(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(
            tmp_path, "prod", model_version="prod-v1.0.0"
        )
        candidate_dir = _build_model_dir(
            tmp_path, "cand", model_version="cand-identical-v1.0.0"
        )
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_APPROVED
        assert decision.candidate_is_production is False
        # Identical behaviour → zero degradation on every relative gate.
        for gate in decision.policy_gates:
            if gate["kind"] == "relative_regression_limit":
                assert gate["passed"]

    def test_candidate_copy_of_production_is_flagged_as_identical(
        self, monkeypatch, tmp_path
    ):
        production_dir = _build_model_dir(
            tmp_path, "prod", model_version="prod-v1.0.0"
        )
        candidate_dir = tmp_path / "cand-copy"
        shutil.copytree(production_dir, candidate_dir, dirs_exist_ok=True)
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_APPROVED
        assert decision.candidate_is_production is True

    def test_missing_candidate_artifact_fails_closed(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand")
        (candidate_dir / "cand.joblib").unlink()
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "candidate_validation"
        assert any(
            "does not exist" in reason for reason in decision.rejection_reasons
        )
        assert decision.candidate_identity is None

    def test_invalid_checksum_fails_closed(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand")
        artifact = candidate_dir / "cand.joblib"
        artifact.write_bytes(artifact.read_bytes() + b"\x00tamper")
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "candidate_validation"
        assert any(
            "checksum" in reason.lower() for reason in decision.rejection_reasons
        )

    def test_malformed_manifest_fails_closed(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand")
        (candidate_dir / "model_manifest.json").write_text(
            "{not valid json", encoding="utf-8"
        )
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "candidate_validation"

    def test_manifest_missing_required_fields_fails_closed(
        self, monkeypatch, tmp_path
    ):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand")
        (candidate_dir / "model_manifest.json").write_text(
            json.dumps({"model_name": "only-one-field"}),
            encoding="utf-8",
        )
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "candidate_validation"
        assert any(
            "required fields" in reason for reason in decision.rejection_reasons
        )

    def test_missing_manifest_fails_closed(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand")
        (candidate_dir / "model_manifest.json").unlink()
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "candidate_validation"

    def test_manifest_path_traversal_fails_closed(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand")
        _rewrite_manifest(candidate_dir, artifact_filename="../outside.joblib")
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "candidate_validation"
        assert any(
            "traversal" in reason.lower() for reason in decision.rejection_reasons
        )

    def test_incompatible_feature_schema_fails_closed(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand")
        _rewrite_manifest(candidate_dir, feature_schema_version="9.9.9")
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "candidate_validation"
        assert any(
            "schema version mismatch" in reason
            for reason in decision.rejection_reasons
        )

    def test_incompatible_feature_count_fails_closed(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand")
        _rewrite_manifest(candidate_dir, n_features=5)
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "candidate_validation"
        assert any(
            "Feature count mismatch" in reason
            for reason in decision.rejection_reasons
        )

    def test_invalid_model_interface_fails_closed(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(
            tmp_path, "cand", model=object()  # no predict_proba
        )
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "candidate_validation"

    def test_candidate_evaluation_feature_mismatch_fails_closed(
        self, monkeypatch, tmp_path
    ):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(
            tmp_path, "cand", feature_names=["nonexistent_feature"]
        )
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "candidate_evaluation"
        assert decision.candidate_metrics is None

    def test_candidate_identity_comes_from_verified_manifest(
        self, monkeypatch, model_setup
    ):
        production_dir, candidate_dir = model_setup
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        registry = ModelRegistry(model_directory=candidate_dir)
        identity = registry.activate_from_manifest()
        assert decision.candidate_identity == identity.to_dict()
        assert decision.candidate_identity["model_version"] == "cand-v1.0.0"


# ── Production baseline ────────────────────────────────────────────────


class TestProductionBaseline:
    def test_missing_production_manifest_fails_closed(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand")
        (production_dir / "model_manifest.json").unlink()
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "production_baseline"
        assert decision.candidate_identity is not None

    def test_corrupt_production_artifact_fails_closed(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand")
        artifact = production_dir / "prod.joblib"
        artifact.write_bytes(artifact.read_bytes() + b"\x00tamper")
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "production_baseline"

    def test_production_evaluation_feature_mismatch_fails_closed(
        self, monkeypatch, tmp_path
    ):
        production_dir = _build_model_dir(
            tmp_path, "prod", feature_names=["nonexistent_feature"]
        )
        candidate_dir = _build_model_dir(tmp_path, "cand")
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "production_evaluation"

    def test_production_identity_is_verified_not_claimed(
        self, monkeypatch, model_setup
    ):
        production_dir, candidate_dir = model_setup
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        registry = ModelRegistry(model_directory=production_dir)
        identity = registry.activate_from_manifest()
        assert decision.production_identity == identity.to_dict()

    def test_ml_model_dir_env_resolves_production(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand")
        X, y, metadata = _deterministic_eval_data()
        monkeypatch.setenv("ML_MODEL_DIR", str(production_dir))
        monkeypatch.setattr(
            "ml.evaluation.promotion_gate.load_holdout_test_set",
            lambda: (X, y, metadata),
        )
        decision = run_promotion_gate(candidate_dir)
        assert decision.decision == DECISION_APPROVED
        assert decision.production_identity["model_version"] == "prod-v1.0.0"

    def test_default_directory_used_when_no_override(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand")
        X, y, metadata = _deterministic_eval_data()
        monkeypatch.setattr(
            "ml.evaluation.promotion_gate.default_model_directory",
            lambda: production_dir,
        )
        monkeypatch.setattr(
            "ml.evaluation.promotion_gate.load_holdout_test_set",
            lambda: (X, y, metadata),
        )
        decision = run_promotion_gate(candidate_dir)
        assert decision.decision == DECISION_APPROVED
        assert decision.production_identity["model_name"] == "prod"

# ── Gate decisions: per-metric rejections ──────────────────────────────


class TestGateDecisions:
    """Candidate performance vs the configurable promotion policy."""

    def _failed_gates(self, decision: PromotionDecision) -> list[dict]:
        return [gate for gate in decision.policy_gates if not gate["passed"]]

    def _gate(self, decision: PromotionDecision, name: str) -> dict:
        matches = [g for g in decision.policy_gates if g["gate"] == name]
        assert len(matches) == 1, f"gate {name!r} not reported exactly once"
        return matches[0]

    # ── Absolute minimum requirements ──────────────────────────────

    def test_candidate_rejected_for_low_pr_auc(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", mode="constant_half")
        decision = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=_policy(min_pr_auc=0.75)
        )
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage is None  # gates failed — validation completed
        gate = self._gate(decision, "min_pr_auc")
        assert gate["kind"] == "absolute_requirement"
        assert gate["comparison"] == ">="
        assert gate["candidate_value"] == pytest.approx(0.5)
        assert gate["required_value"] == 0.75
        assert gate["production_value"] is None
        assert gate["passed"] is False
        assert len(decision.rejection_reasons) == 1
        assert "min_pr_auc" in decision.rejection_reasons[0]

    def test_candidate_rejected_for_low_roc_auc(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", mode="inverse")
        decision = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=_policy(min_roc_auc=0.75)
        )
        assert decision.decision == DECISION_REJECTED
        gate = self._gate(decision, "min_roc_auc")
        assert gate["candidate_value"] == pytest.approx(0.0)
        assert gate["required_value"] == 0.75
        assert gate["passed"] is False

    def test_candidate_rejected_for_low_recall(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", threshold=0.95)
        decision = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=_policy(min_recall=0.70)
        )
        assert decision.decision == DECISION_REJECTED
        gate = self._gate(decision, "min_recall")
        assert gate["candidate_value"] == pytest.approx(0.10)
        assert gate["required_value"] == 0.70
        assert gate["passed"] is False

    def test_candidate_rejected_for_low_precision(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", threshold=0.05)
        decision = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(min_precision=0.60),
        )
        assert decision.decision == DECISION_REJECTED
        gate = self._gate(decision, "min_precision")
        assert gate["candidate_value"] == pytest.approx(100.0 / 190.0)
        assert gate["required_value"] == 0.60
        assert gate["passed"] is False

    def test_candidate_rejected_for_low_f1(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", mode="constant_half")
        decision = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=_policy(min_f1=0.80)
        )
        assert decision.decision == DECISION_REJECTED
        gate = self._gate(decision, "min_f1")
        assert gate["candidate_value"] == pytest.approx(2.0 / 3.0)
        assert gate["required_value"] == 0.80
        assert gate["passed"] is False

    def test_candidate_rejected_for_excessive_brier(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", mode="confident")
        decision = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=_policy(max_brier=0.25)
        )
        assert decision.decision == DECISION_REJECTED
        gate = self._gate(decision, "max_brier")
        assert gate["comparison"] == "<="
        assert gate["candidate_value"] > 0.25
        assert gate["required_value"] == 0.25
        assert gate["passed"] is False

    # ── Relative regression limits vs production ───────────────────

    def test_candidate_rejected_for_pr_auc_degradation(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", mode="constant_half")
        decision = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(max_pr_auc_degradation=0.10),
        )
        assert decision.decision == DECISION_REJECTED
        gate = self._gate(decision, "max_pr_auc_degradation")
        assert gate["kind"] == "relative_regression_limit"
        assert gate["comparison"] == ">="
        assert gate["production_value"] == pytest.approx(1.0)
        assert gate["required_value"] == pytest.approx(0.9)
        assert gate["candidate_value"] == pytest.approx(0.5)
        assert gate["passed"] is False
        assert len(decision.rejection_reasons) == 1

    def test_candidate_rejected_for_roc_auc_degradation(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", mode="constant_half")
        decision = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(max_roc_auc_degradation=0.02),
        )
        assert decision.decision == DECISION_REJECTED
        gate = self._gate(decision, "max_roc_auc_degradation")
        assert gate["required_value"] == pytest.approx(0.98)
        assert gate["candidate_value"] == pytest.approx(0.5)
        assert gate["passed"] is False

    def test_candidate_rejected_for_recall_degradation(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", threshold=0.95)
        decision = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(max_recall_degradation=0.10),
        )
        assert decision.decision == DECISION_REJECTED
        gate = self._gate(decision, "max_recall_degradation")
        assert gate["production_value"] == pytest.approx(1.0)
        assert gate["required_value"] == pytest.approx(0.9)
        assert gate["candidate_value"] == pytest.approx(0.10)
        assert gate["passed"] is False

    def test_candidate_rejected_for_precision_degradation(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", threshold=0.05)
        decision = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(max_precision_degradation=0.10),
        )
        assert decision.decision == DECISION_REJECTED
        gate = self._gate(decision, "max_precision_degradation")
        assert gate["required_value"] == pytest.approx(0.9)
        assert gate["candidate_value"] == pytest.approx(100.0 / 190.0)
        assert gate["passed"] is False

    def test_candidate_rejected_for_f1_degradation(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", mode="constant_half")
        decision = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(max_f1_degradation=0.10),
        )
        assert decision.decision == DECISION_REJECTED
        gate = self._gate(decision, "max_f1_degradation")
        assert gate["required_value"] == pytest.approx(0.9)
        assert gate["candidate_value"] == pytest.approx(2.0 / 3.0)
        assert gate["passed"] is False

    def test_candidate_rejected_for_brier_increase(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", mode="confident")
        decision = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(max_brier_increase=0.10),
        )
        assert decision.decision == DECISION_REJECTED
        gate = self._gate(decision, "max_brier_increase")
        assert gate["comparison"] == "<="
        prod_brier = decision.production_metrics["brier_score"]
        assert gate["required_value"] == pytest.approx(prod_brier * 1.1)
        assert gate["candidate_value"] > gate["required_value"]
        assert gate["passed"] is False

    def test_multiple_simultaneous_failures(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", mode="inverse")
        decision = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=PromotionPolicy()
        )
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage is None
        failed = self._failed_gates(decision)
        failed_names = {gate["gate"] for gate in failed}
        # Only the very permissive absolute PR-AUC floor (0.10 vs 0.31) passes.
        assert len(failed) == 11
        assert "min_pr_auc" not in failed_names
        assert failed_names == {
            "min_roc_auc",
            "min_recall",
            "min_precision",
            "min_f1",
            "max_brier",
            "max_pr_auc_degradation",
            "max_roc_auc_degradation",
            "max_recall_degradation",
            "max_precision_degradation",
            "max_f1_degradation",
            "max_brier_increase",
        }
        assert len(decision.rejection_reasons) == 11
        for reason in decision.rejection_reasons:
            assert reason.startswith("Gate '")

    # ── Exact boundary conditions ──────────────────────────────────

    def test_absolute_minimum_boundary_is_inclusive(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", intercept=0.02)
        baseline = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=_policy()
        )
        actual = baseline.candidate_metrics["precision"]
        assert actual == pytest.approx(100.0 / 104.0)
        passing = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(min_precision=actual),
        )
        assert passing.decision == DECISION_APPROVED
        assert self._gate(passing, "min_precision")["passed"] is True
        failing = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(min_precision=actual + 1e-6),
        )
        assert failing.decision == DECISION_REJECTED
        assert self._gate(failing, "min_precision")["passed"] is False

    def test_relative_floor_boundary_is_inclusive(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", intercept=0.02)
        baseline = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=_policy()
        )
        cand_f1 = baseline.candidate_metrics["f1"]
        prod_f1 = baseline.production_metrics["f1"]
        limit = 1.0 - cand_f1 / prod_f1  # floor sits exactly at the candidate
        passing = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(max_f1_degradation=limit),
        )
        assert passing.decision == DECISION_APPROVED
        failing = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(max_f1_degradation=limit - 1e-6),
        )
        assert failing.decision == DECISION_REJECTED

    def test_brier_ceiling_boundary_is_inclusive(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", intercept=0.02)
        baseline = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=_policy()
        )
        actual = baseline.candidate_metrics["brier_score"]
        passing = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=_policy(max_brier=actual)
        )
        assert passing.decision == DECISION_APPROVED
        failing = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(max_brier=actual - 1e-6),
        )
        assert failing.decision == DECISION_REJECTED

    def test_relative_brier_ceiling_boundary_is_inclusive(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", intercept=0.02)
        baseline = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=_policy()
        )
        cand_brier = baseline.candidate_metrics["brier_score"]
        prod_brier = baseline.production_metrics["brier_score"]
        limit = cand_brier / prod_brier - 1.0  # ceiling sits exactly at candidate
        passing = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(max_brier_increase=limit),
        )
        assert passing.decision == DECISION_APPROVED
        failing = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(max_brier_increase=limit - 1e-6),
        )
        assert failing.decision == DECISION_REJECTED

    # ── Unavailable metrics / disabled gates ───────────────────────

    def _single_class_data(self):
        X, _, _ = _deterministic_eval_data()
        return (
            X,
            pd.Series(np.ones(len(X), dtype=int)),
            {
                "dataset_identifier": "synthetic-step48/single-class",
                "split_strategy": "all_positive",
                "n_test_samples": int(len(X)),
            },
        )

    def test_unavailable_absolute_metric_fails_closed(self, monkeypatch, model_setup):
        production_dir, candidate_dir = model_setup
        decision = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(min_pr_auc=0.10),
            data=self._single_class_data(),
        )
        assert decision.decision == DECISION_REJECTED
        gate = self._gate(decision, "min_pr_auc")
        assert gate["candidate_value"] is None
        assert gate["required_value"] == 0.10
        assert gate["passed"] is False
        assert "unavailable" in gate["detail"]
        assert "ranking_unavailable_reason" in decision.candidate_metrics
        assert "ranking_unavailable_reason" in decision.production_metrics

    def test_unavailable_relative_metric_fails_closed(self, monkeypatch, model_setup):
        production_dir, candidate_dir = model_setup
        decision = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(max_pr_auc_degradation=0.05),
            data=self._single_class_data(),
        )
        assert decision.decision == DECISION_REJECTED
        gate = self._gate(decision, "max_pr_auc_degradation")
        assert gate["candidate_value"] is None
        assert gate["production_value"] is None
        assert gate["required_value"] is None
        assert gate["passed"] is False

    def test_disabled_gates_are_absent_from_report(self, monkeypatch, model_setup):
        production_dir, candidate_dir = model_setup
        decision = _run_gate(monkeypatch, production_dir, candidate_dir, policy=_policy())
        assert decision.policy_gates == []
        assert decision.decision == DECISION_APPROVED  # nothing configured to fail
        assert decision.policy_configuration == _policy().to_dict()

    def test_gate_dicts_have_required_fields(self, monkeypatch, model_setup):
        production_dir, candidate_dir = model_setup
        decision = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=PromotionPolicy()
        )
        assert len(decision.policy_gates) == 12
        required = {
            "gate",
            "kind",
            "metric",
            "comparison",
            "candidate_value",
            "production_value",
            "required_value",
            "passed",
            "detail",
        }
        for gate in decision.policy_gates:
            assert set(gate) == required
            assert isinstance(gate["passed"], bool)
            assert gate["kind"] in ("absolute_requirement", "relative_regression_limit")
            assert gate["comparison"] in (">=", "<=")
            assert isinstance(gate["detail"], str) and gate["detail"]
            assert len(gate["detail"]) <= 300

    def test_decision_is_always_approved_or_rejected(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        good = _build_model_dir(tmp_path, "cand", intercept=0.02)
        bad = _build_model_dir(tmp_path, "bad", mode="inverse")
        tampered = _build_model_dir(tmp_path, "tampered")
        (tampered / "tampered.joblib").write_bytes(b"\x00corrupt")
        for candidate_dir, policy in (
            (good, PromotionPolicy()),
            (bad, PromotionPolicy()),
            (tampered, PromotionPolicy()),
            (good, None),
        ):
            decision = _run_gate(
                monkeypatch, production_dir, candidate_dir, policy=policy
            )
            assert decision.decision in (DECISION_APPROVED, DECISION_REJECTED)
            assert decision.report_scope == GATE_REPORT_SCOPE

    def test_repeated_execution_is_deterministic(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", intercept=0.02)
        first = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=PromotionPolicy()
        )
        second = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=PromotionPolicy()
        )
        assert first.decision == second.decision == DECISION_APPROVED
        assert first.candidate_metrics == second.candidate_metrics
        assert first.production_metrics == second.production_metrics
        assert first.policy_gates == second.policy_gates
        assert first.rejection_reasons == second.rejection_reasons
        dict_one = first.to_dict()
        dict_two = second.to_dict()
        dict_one.pop("gate_timestamp")
        dict_two.pop("gate_timestamp")
        assert dict_one == dict_two

    def test_policy_env_changes_decision(self, monkeypatch, model_setup):
        production_dir, candidate_dir = model_setup
        monkeypatch.setenv("PROMO_MIN_PRECISION", "0.99")
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        gate = self._gate(decision, "min_precision")
        assert gate["candidate_value"] == pytest.approx(100.0 / 104.0)
        assert gate["required_value"] == 0.99
        monkeypatch.delenv("PROMO_MIN_PRECISION")
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_APPROVED

    def test_each_model_evaluated_at_own_bundled_threshold(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod", threshold=0.4)
        candidate_dir = _build_model_dir(
            tmp_path, "cand", intercept=0.02, threshold=0.6
        )
        decision = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=_policy()
        )
        assert decision.evaluation_metadata["production_threshold"] == 0.4
        assert decision.evaluation_metadata["candidate_threshold"] == 0.6
        assert decision.production_metrics["threshold"] == 0.4
        assert decision.candidate_metrics["threshold"] == 0.6
        semantics = decision.evaluation_metadata["threshold_semantics"]
        assert "own bundled production threshold" in semantics
        assert "never modified" in semantics
        assert decision.evaluation_metadata["same_dataset_for_both_models"] is True

    def test_evaluate_promotion_low_level_api(self, monkeypatch, tmp_path):
        """The low-level comparison mirrors the full-gate result."""
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", intercept=0.02)
        X, y, _ = _deterministic_eval_data()
        prod_registry = ModelRegistry(model_directory=production_dir)
        prod_identity = prod_registry.activate_from_manifest()
        cand_registry = ModelRegistry(model_directory=candidate_dir)
        cand_identity = cand_registry.activate_from_manifest()
        decision = evaluate_promotion(
            production_bundle=prod_registry.bundle,
            production_identity=prod_identity,
            candidate_bundle=cand_registry.bundle,
            candidate_identity=cand_identity,
            X=X,
            y=y,
            policy=PromotionPolicy(),
        )
        assert decision.decision == DECISION_APPROVED
        assert decision.reproducibility["policy_source"] == "explicit argument"
        # The run-level read-only guard is added only by run_promotion_gate.
        assert "production_manifest_unchanged" not in decision.reproducibility


# ── Fail-closed behaviour ──────────────────────────────────────────────


class TestFailClosedBehavior:
    """The gate never approves an incompletely validated candidate."""

    def test_invalid_policy_object_fails_closed(self, monkeypatch, model_setup):
        production_dir, candidate_dir = model_setup
        decision = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(min_recall=-0.5),
        )
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "policy_configuration"
        assert any(
            "Invalid promotion policy" in reason
            for reason in decision.rejection_reasons
        )
        assert decision.policy_gates == []
        assert decision.promotion_instruction is None

    def test_invalid_policy_env_fails_closed(self, monkeypatch, model_setup):
        production_dir, candidate_dir = model_setup
        monkeypatch.setenv("PROMO_MIN_RECALL", "not-a-number")
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "policy_configuration"
        assert any(
            "PROMO_MIN_RECALL" in reason for reason in decision.rejection_reasons
        )

    def test_invalid_evaluation_config_object_fails_closed(
        self, monkeypatch, model_setup
    ):
        production_dir, candidate_dir = model_setup
        decision = _run_gate(
            monkeypatch,
            production_dir,
            candidate_dir,
            policy=_policy(),
            evaluation_config=EvaluationConfig(threshold_start=1.5),
        )
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "evaluation_configuration"
        assert any(
            "Invalid evaluation configuration" in reason
            for reason in decision.rejection_reasons
        )

    def test_invalid_evaluation_config_env_fails_closed(
        self, monkeypatch, model_setup
    ):
        production_dir, candidate_dir = model_setup
        monkeypatch.setenv("EVAL_THRESHOLD_START", "7.5")
        decision = _run_gate(monkeypatch, production_dir, candidate_dir, policy=_policy())
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "evaluation_configuration"

    def test_evaluation_dataset_load_failure_fails_closed(
        self, monkeypatch, model_setup
    ):
        production_dir, candidate_dir = model_setup

        def _boom():
            raise RuntimeError("secret dataset failure")

        monkeypatch.setattr(
            "ml.evaluation.promotion_gate.load_holdout_test_set", _boom
        )
        decision = run_promotion_gate(
            candidate_dir,
            policy=_policy(),
            production_model_directory=production_dir,
        )
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "evaluation"
        assert len(decision.rejection_reasons) == 1
        assert (
            decision.rejection_reasons[0]
            == "Evaluation dataset could not be loaded: RuntimeError"
        )
        assert "secret" not in decision.rejection_reasons[0]

    def test_approved_decision_implies_complete_validation(
        self, monkeypatch, model_setup
    ):
        production_dir, candidate_dir = model_setup
        decision = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=PromotionPolicy()
        )
        assert decision.decision == DECISION_APPROVED
        assert decision.failure_stage is None
        assert decision.candidate_identity is not None
        assert decision.production_identity is not None
        assert decision.candidate_metrics is not None
        assert decision.production_metrics is not None
        assert len(decision.policy_gates) == 12
        assert all(gate["passed"] for gate in decision.policy_gates)
        assert decision.rejection_reasons == []
        assert decision.promotion_instruction is not None

    def test_rejected_decisions_never_carry_promotion_instruction(
        self, monkeypatch, tmp_path
    ):
        production_dir = _build_model_dir(tmp_path, "prod")
        tampered = _build_model_dir(tmp_path, "tampered")
        (tampered / "tampered.joblib").write_bytes(b"\x00corrupt")
        inverse = _build_model_dir(tmp_path, "inverse", mode="inverse")
        decisions = [
            _run_gate(monkeypatch, production_dir, tampered),
            _run_gate(monkeypatch, production_dir, inverse),
        ]
        monkeypatch.setenv("PROMO_MIN_RECALL", "not-a-number")
        decisions.append(_run_gate(monkeypatch, production_dir, inverse))
        for decision in decisions:
            assert decision.decision == DECISION_REJECTED
            assert decision.promotion_instruction is None

    def test_disclaimer_documents_no_automatic_activation(
        self, monkeypatch, model_setup
    ):
        production_dir, candidate_dir = model_setup
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.disclaimer == GATE_DISCLAIMER
        text = decision.disclaimer.lower()
        assert "does not activate" in text
        assert "step 46" in text
        assert "threshold" in text


# ── Decision safety (bounded, JSON-safe, no sensitive data) ─────────────


_EXPECTED_TOP_LEVEL_KEYS = {
    "report_scope",
    "decision",
    "disclaimer",
    "gate_timestamp",
    "failure_stage",
    "candidate_identity",
    "production_identity",
    "candidate_is_production",
    "evaluation_metadata",
    "candidate_metrics",
    "production_metrics",
    "policy_configuration",
    "policy_gates",
    "rejection_reasons",
    "promotion_instruction",
    "reproducibility",
}

_EXPECTED_METRIC_KEYS = {
    "threshold",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "n_samples",
    "n_fraud",
    "fraud_prevalence",
}

_FORBIDDEN_KEY_SUBSTRINGS = (
    "password",
    "secret",
    "token",
    "api_key",
    "credential",
    "transaction",
    "customer",
    "prediction",
    "y_true",
)


def _walk_values(node, prefix=""):
    """Yield ``(key-path, value)`` for every leaf in a nested dict/list tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_values(value, new_prefix)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk_values(value, f"{prefix}[{index}]")
    else:
        yield prefix, node


class TestDecisionSafety:
    """The promotion result is bounded and safe to persist or share."""

    def _approved_decision(self, monkeypatch, model_setup):
        production_dir, candidate_dir = model_setup
        return _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=PromotionPolicy()
        )

    def test_top_level_keys_are_exact(self, monkeypatch, model_setup):
        decision = self._approved_decision(monkeypatch, model_setup)
        assert set(decision.to_dict()) == _EXPECTED_TOP_LEVEL_KEYS

    def test_report_is_json_serializable_and_bounded(self, monkeypatch, model_setup):
        decision = self._approved_decision(monkeypatch, model_setup)
        payload = json.dumps(decision.to_dict(), allow_nan=False, sort_keys=True)
        assert len(payload) < 32768

    def test_no_local_paths_or_secrets_in_report(self, monkeypatch, model_setup):
        decision = self._approved_decision(monkeypatch, model_setup)
        for key, value in _walk_values(decision.to_dict()):
            lowered = key.lower()
            for forbidden in _FORBIDDEN_KEY_SUBSTRINGS:
                assert forbidden not in lowered, f"forbidden key: {key}"
            if isinstance(value, str):
                assert "AppData" not in value
                assert "pytest" not in value
                assert ":\\Users" not in value

    def test_no_raw_data_arrays_in_report(self, monkeypatch, model_setup):
        decision = self._approved_decision(monkeypatch, model_setup)
        for key, value in _walk_values(decision.to_dict()):
            if isinstance(value, list):
                numeric = [
                    item
                    for item in value
                    if isinstance(item, (int, float)) and not isinstance(item, bool)
                ]
                assert not (len(value) > 16 and len(numeric) == len(value)), key

    def test_all_numbers_are_finite(self, monkeypatch, model_setup):
        decision = self._approved_decision(monkeypatch, model_setup)
        for _, value in _walk_values(decision.to_dict()):
            if isinstance(value, float):
                assert math.isfinite(value)

    def test_metric_summaries_are_bounded(self, monkeypatch, model_setup):
        decision = self._approved_decision(monkeypatch, model_setup)
        assert set(decision.candidate_metrics) == _EXPECTED_METRIC_KEYS
        assert set(decision.production_metrics) == _EXPECTED_METRIC_KEYS
        for summary in (decision.candidate_metrics, decision.production_metrics):
            assert 0.0 <= summary["threshold"] <= 1.0
            for metric in (
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "brier_score",
                "fraud_prevalence",
            ):
                value = summary[metric]
                assert value is None or 0.0 <= value <= 1.0

    def test_identities_contain_only_governance_fields(
        self, monkeypatch, model_setup
    ):
        decision = self._approved_decision(monkeypatch, model_setup)
        expected = {
            "model_name",
            "model_version",
            "artifact_checksum",
            "feature_schema_version",
            "n_features",
            "status",
        }
        assert set(decision.candidate_identity) == expected
        assert set(decision.production_identity) == expected

    def test_promotion_instruction_shape(self, monkeypatch, model_setup):
        decision = self._approved_decision(monkeypatch, model_setup)
        instruction = decision.promotion_instruction
        assert instruction is not None
        assert "action_required" in instruction
        assert "note" in instruction
        assert instruction["candidate_model_version"] == "cand-v1.0.0"
        assert (
            instruction["candidate_artifact_checksum"]
            == decision.candidate_identity["artifact_checksum"]
        )
        assert len(instruction["operator_steps"]) == 4
        joined = " ".join(str(step) for step in instruction["operator_steps"]).lower()
        assert "step 46" in joined or "governance" in joined

    def test_policy_configuration_matches_policy(self, monkeypatch, model_setup):
        policy = PromotionPolicy(min_recall=0.6, max_brier_increase=0.05)
        production_dir, candidate_dir = model_setup
        decision = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=policy
        )
        assert decision.policy_configuration == policy.to_dict()

    def test_reproducibility_metadata_is_complete(self, monkeypatch, model_setup):
        decision = self._approved_decision(monkeypatch, model_setup)
        reproducibility = decision.reproducibility
        assert reproducibility["policy_source"] == "explicit argument"
        assert reproducibility["report_schema_version"] == REPORT_SCHEMA_VERSION
        assert reproducibility["deterministic"] is True
        assert reproducibility["dataset_identifier"] == "synthetic-step48/holdout"
        assert reproducibility["production_manifest_unchanged"] is True
        assert "policy_configuration" in reproducibility
        assert "evaluation_config" in reproducibility
        assert "metric_configuration" in reproducibility
        assert "production_threshold_unchanged" in reproducibility


# ── Production model must remain unchanged ──────────────────────────────


class TestProductionUnchanged:
    """The gate is strictly read-only for the model directories."""

    def test_approval_modifies_nothing(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(
            tmp_path, "prod", model_version="prod-v1.0.0"
        )
        candidate_dir = _build_model_dir(
            tmp_path, "cand", intercept=0.02, model_version="cand-v1.0.0"
        )
        production_before = _snapshot_tree(production_dir)
        candidate_before = _snapshot_tree(candidate_dir)
        decision = _run_gate(
            monkeypatch, production_dir, candidate_dir, policy=PromotionPolicy()
        )
        assert decision.decision == DECISION_APPROVED
        assert _snapshot_tree(production_dir) == production_before
        assert _snapshot_tree(candidate_dir) == candidate_before

    def test_rejection_modifies_nothing(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", mode="inverse")
        production_before = _snapshot_tree(production_dir)
        candidate_before = _snapshot_tree(candidate_dir)
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert _snapshot_tree(production_dir) == production_before
        assert _snapshot_tree(candidate_dir) == candidate_before

    def test_validation_failure_modifies_nothing(self, monkeypatch, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand")
        (candidate_dir / "model_manifest.json").unlink()
        production_before = _snapshot_tree(production_dir)
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "candidate_validation"
        assert _snapshot_tree(production_dir) == production_before

    def test_production_identity_and_threshold_unchanged(
        self, monkeypatch, tmp_path
    ):
        production_dir = _build_model_dir(tmp_path, "prod", threshold=0.5)
        candidate_dir = _build_model_dir(tmp_path, "cand", intercept=0.02)
        registry = ModelRegistry(model_directory=production_dir)
        identity_before = registry.activate_from_manifest()
        threshold_before = registry.bundle.threshold
        decision = _run_gate(monkeypatch, production_dir, candidate_dir)
        assert decision.decision == DECISION_APPROVED
        fresh = ModelRegistry(model_directory=production_dir)
        identity_after = fresh.activate_from_manifest()
        assert identity_after == identity_before
        assert fresh.bundle.threshold == threshold_before == 0.5
        assert decision.evaluation_metadata["production_threshold"] == 0.5
        assert identity_after.status == "active"


# ── Static (AST) safety checks on the gate modules ──────────────────────


_GATE_SOURCE_FILES = (
    Path(__file__).resolve().parents[1] / "evaluation" / "promotion_gate.py",
    Path(__file__).resolve().parents[1] / "evaluation" / "promotion_policy.py",
)

_FORBIDDEN_IMPORT_ROOTS = ("ml.api", "ml.risk", "ml.monitoring", "backend")

_MUTATING_CALL_NAMES = {
    "save_manifest",
    "save_bundle",
    "rollback",
    "rmtree",
    "unlink",
    "rename",
    "remove",
}


def _module_roots(tree: ast.Module) -> set:
    roots: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module)
    return roots


def _call_names(tree: ast.Module) -> set:
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


class TestModuleSafety:
    """Static (AST) safety checks on the gate implementation."""

    def test_gate_modules_do_not_import_serving_or_backend_code(self):
        for path in _GATE_SOURCE_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for root in _module_roots(tree):
                assert not root.startswith(_FORBIDDEN_IMPORT_ROOTS), (
                    path.name,
                    root,
                )

    def test_gate_performs_no_model_mutation_calls(self):
        for path in _GATE_SOURCE_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names = _call_names(tree)
            assert not (names & _MUTATING_CALL_NAMES), (
                path.name,
                names & _MUTATING_CALL_NAMES,
            )

    def test_gate_uses_only_trusted_model_interfaces(self):
        # Models must be loaded exclusively through the Step 46 governance
        # path (registry + bundle) — never directly via joblib/pickle.
        source = _GATE_SOURCE_FILES[0].read_text(encoding="utf-8")
        assert "joblib" not in source
        assert "pickle" not in source
        assert "subprocess" not in source
        assert "eval(" not in source
        assert "exec(" not in source


# ── CLI ─────────────────────────────────────────────────────────────────


def _run_cli(monkeypatch, candidate_dir, production_dir, output=None):
    """Invoke the CLI entry point in-process; return its exit code."""
    X, y, metadata = _deterministic_eval_data()
    monkeypatch.setattr(
        "ml.evaluation.promotion_gate.load_holdout_test_set",
        lambda: (X, y, metadata),
    )
    monkeypatch.setenv("ML_MODEL_DIR", str(production_dir))
    argv = ["promotion_gate", "--candidate-model-dir", str(candidate_dir)]
    if output is not None:
        argv += ["--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        main()
    return excinfo.value.code


class TestCLI:
    def test_cli_exit_zero_on_approval(self, monkeypatch, capsys, model_setup):
        production_dir, candidate_dir = model_setup
        code = _run_cli(monkeypatch, candidate_dir, production_dir)
        assert code == 0
        out = capsys.readouterr().out
        assert "APPROVED" in out
        assert "Traceback" not in out

    def test_cli_exit_one_on_rejection(self, monkeypatch, capsys, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", mode="inverse")
        code = _run_cli(monkeypatch, candidate_dir, production_dir)
        assert code == 1
        out = capsys.readouterr().out
        assert "REJECTED" in out

    def test_cli_exit_one_on_validation_failure(self, monkeypatch, capsys, tmp_path):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand")
        (candidate_dir / "cand.joblib").unlink()
        code = _run_cli(monkeypatch, candidate_dir, production_dir)
        assert code == 1
        out = capsys.readouterr().out
        assert "REJECTED" in out
        assert "candidate_validation" in out

    def test_cli_exit_two_on_missing_argument(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["promotion_gate"])
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 2

    def test_cli_internal_error_is_bounded(self, monkeypatch, capsys, tmp_path):
        def _boom(_candidate_dir):
            raise RuntimeError("secret internal detail C:\\Users\\abi\\private.pem")

        monkeypatch.setattr("ml.evaluation.promotion_gate.run_promotion_gate", _boom)
        monkeypatch.setattr(
            sys,
            "argv",
            ["promotion_gate", "--candidate-model-dir", str(tmp_path / "cand")],
        )
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 2
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "Promotion gate error: RuntimeError" in captured.err
        assert "secret internal detail" not in combined
        assert "Traceback" not in combined

    def test_cli_writes_bounded_json_report_on_approval(
        self, monkeypatch, tmp_path, model_setup
    ):
        production_dir, candidate_dir = model_setup
        report_path = tmp_path / "reports" / "promotion.json"
        code = _run_cli(
            monkeypatch, candidate_dir, production_dir, output=report_path
        )
        assert code == 0
        assert report_path.exists()
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["decision"] == "APPROVED"
        assert set(data) == _EXPECTED_TOP_LEVEL_KEYS
        assert (
            data["promotion_instruction"]["candidate_model_version"]
            == "cand-v1.0.0"
        )
        payload = report_path.read_text(encoding="utf-8")
        assert len(payload) < 32768
        assert str(tmp_path) not in payload

    def test_cli_writes_rejection_report_without_instruction(
        self, monkeypatch, tmp_path
    ):
        production_dir = _build_model_dir(tmp_path, "prod")
        candidate_dir = _build_model_dir(tmp_path, "cand", mode="inverse")
        report_path = tmp_path / "promotion.json"
        code = _run_cli(
            monkeypatch, candidate_dir, production_dir, output=report_path
        )
        assert code == 1
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["decision"] == "REJECTED"
        assert data["promotion_instruction"] is None
        assert data["rejection_reasons"]


# ── Real-model integration (skipped when the artifact is unavailable) ───


def _model_available() -> bool:
    try:
        registry = ModelRegistry()
        registry.activate_from_manifest()
        return registry.is_ready
    except Exception:
        return False


MODEL_AVAILABLE = _model_available()
requires_model = pytest.mark.skipif(
    not MODEL_AVAILABLE, reason="Production model not available"
)


def _synthetic_real_features(bundle, n: int = 200):
    """Deterministic feature matrix matching the real bundle's schema."""
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


_PERMISSIVE_POLICY = PromotionPolicy(
    min_pr_auc=0.0,
    min_roc_auc=0.0,
    min_recall=0.0,
    min_precision=0.0,
    min_f1=0.0,
    max_brier=1.0,
    max_pr_auc_degradation=None,
    max_roc_auc_degradation=None,
    max_recall_degradation=None,
    max_precision_degradation=None,
    max_f1_degradation=None,
    max_brier_increase=None,
)


@requires_model
class TestRealModelIntegration:
    """End-to-end gate runs against the real governed production model."""

    def _copy_governed_model(self, target: Path) -> None:
        from ml.predict.integrity import default_model_directory, load_manifest

        source = default_model_directory()
        manifest = load_manifest(source)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            source / "model_manifest.json", target / "model_manifest.json"
        )
        shutil.copy2(
            source / manifest.artifact_filename, target / manifest.artifact_filename
        )

    def _patch_dataset(self, monkeypatch):
        from ml.predict.bundle import load_bundle
        from ml.predict.integrity import default_model_directory, load_manifest

        source = default_model_directory()
        manifest = load_manifest(source)
        bundle = load_bundle(source / manifest.artifact_filename)
        X, y = _synthetic_real_features(bundle)
        metadata = {
            "dataset_identifier": "synthetic-step48/real-model",
            "split_strategy": "rng-seed-42",
            "n_test_samples": int(len(X)),
        }
        monkeypatch.setattr(
            "ml.evaluation.promotion_gate.load_holdout_test_set",
            lambda: (X, y, metadata),
        )
        return manifest

    def test_identical_copy_of_production_is_approved(self, monkeypatch, tmp_path):
        manifest = self._patch_dataset(monkeypatch)
        production_dir = tmp_path / "prod-real"
        candidate_dir = tmp_path / "cand-real"
        self._copy_governed_model(production_dir)
        self._copy_governed_model(candidate_dir)
        decision = run_promotion_gate(
            candidate_dir,
            policy=_PERMISSIVE_POLICY,
            production_model_directory=production_dir,
        )
        assert decision.decision == DECISION_APPROVED
        assert decision.candidate_is_production is True
        assert decision.candidate_metrics == decision.production_metrics
        assert decision.production_identity["model_version"] == manifest.model_version

    def test_tampered_copy_of_production_is_rejected(self, monkeypatch, tmp_path):
        from ml.predict.integrity import load_manifest

        self._patch_dataset(monkeypatch)
        production_dir = tmp_path / "prod-real"
        candidate_dir = tmp_path / "cand-real"
        self._copy_governed_model(production_dir)
        self._copy_governed_model(candidate_dir)
        artifact = candidate_dir / load_manifest(candidate_dir).artifact_filename
        artifact.write_bytes(artifact.read_bytes() + b"\x00tamper")
        decision = run_promotion_gate(
            candidate_dir,
            policy=_PERMISSIVE_POLICY,
            production_model_directory=production_dir,
        )
        assert decision.decision == DECISION_REJECTED
        assert decision.failure_stage == "candidate_validation"
        assert any(
            "checksum" in reason.lower() for reason in decision.rejection_reasons
        )
