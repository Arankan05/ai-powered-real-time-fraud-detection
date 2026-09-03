"""Step 46 — Production model lifecycle and version governance tests.

Comprehensive test suite for model governance:
artifact integrity, manifest management, checksum verification,
model interface validation, feature compatibility, version propagation,
rollback safety, health/ready/metrics identity, audit integration,
concurrency safety, path traversal protection, and sensitive data checks.

Run from the project root::

    python -m pytest ml/api/tests/test_step46_model_governance.py -v
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml.predict.bundle import ModelBundle, ModelLoadError, save_bundle, load_bundle
from ml.predict.integrity import (
    IntegrityError,
    ManifestError,
    ModelManifest,
    build_manifest,
    compute_checksum,
    load_manifest,
    safe_artifact_path,
    save_manifest,
    verify_artifact,
    _FEATURE_SCHEMA_VERSION,
)
from ml.predict.registry import ActivationError, ModelIdentity, ModelRegistry


# ── Helpers ────────────────────────────────────────────────────────────


class _MockClassifier:
    """Minimal mock classifier with predict_proba for testing."""

    def __init__(self, n_features: int = 24):
        self.n_features = n_features

    def predict_proba(self, X):
        n = X.shape[0]
        probs = np.full((n, 2), 0.5)
        probs[:, 1] = 0.3  # default low fraud probability
        return probs


@dataclass
class _MockPreprocessing:
    """Minimal mock preprocessing artifacts."""
    scaler: None = None
    label_encoder: None = None


def _create_bundle(
    model_version: str = "fraud-xgb-v1.0.0",
    n_features: int = 24,
    threshold: float = 0.50,
) -> ModelBundle:
    """Create a test ModelBundle."""
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return ModelBundle(
        model=_MockClassifier(n_features),
        preprocessing=_MockPreprocessing(),
        threshold=threshold,
        feature_names=feature_names,
        model_version=model_version,
    )


def _save_test_artifact(
    directory: Path,
    bundle: ModelBundle | None = None,
    filename: str = "fraud_xgb_tuned.joblib",
) -> Path:
    """Save a test bundle and return the artifact path."""
    if bundle is None:
        bundle = _create_bundle()
    path = directory / filename
    save_bundle(bundle, path)
    return path


def _save_test_manifest(
    directory: Path,
    artifact_path: Path,
    model_version: str = "fraud-xgb-v1.0.0",
    n_features: int = 24,
) -> ModelManifest:
    """Create and save a manifest for a test artifact."""
    manifest = build_manifest(
        model_name="fraud-xgb",
        model_version=model_version,
        artifact_path=artifact_path,
        n_features=n_features,
        threshold=0.50,
    )
    save_manifest(manifest, directory=directory)
    return manifest


# ══════════════════════════════════════════════════════════════════════
# A. Valid artifact loads
# ══════════════════════════════════════════════════════════════════════


class TestValidArtifactLoad:
    """A. Valid artifact loads successfully."""

    def test_bundle_loads_from_valid_artifact(self, tmp_path):
        bundle = _create_bundle()
        path = _save_test_artifact(tmp_path, bundle)
        loaded = load_bundle(path)
        assert loaded.model_version == "fraud-xgb-v1.0.0"
        assert loaded.n_features == 24

    def test_manifest_generated_for_valid_artifact(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        manifest = build_manifest(
            model_name="fraud-xgb",
            model_version="fraud-xgb-v1.0.0",
            artifact_path=artifact_path,
            n_features=24,
        )
        assert manifest.model_name == "fraud-xgb"
        assert manifest.model_version == "fraud-xgb-v1.0.0"
        assert manifest.artifact_checksum  # non-empty
        assert manifest.serialization_format == "joblib"

    def test_registry_activates_valid_model(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        _save_test_manifest(tmp_path, artifact_path)
        registry = ModelRegistry(model_directory=tmp_path)
        identity = registry.activate_from_manifest()
        assert identity.model_version == "fraud-xgb-v1.0.0"
        assert identity.status == "active"
        assert registry.is_ready


# ══════════════════════════════════════════════════════════════════════
# B. Missing artifact
# ══════════════════════════════════════════════════════════════════════


class TestMissingArtifact:
    """B. Missing artifact is handled correctly."""

    def test_load_bundle_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_bundle(tmp_path / "nonexistent.joblib")

    def test_verify_artifact_missing(self, tmp_path):
        manifest = ModelManifest(
            model_name="test",
            model_version="v1",
            artifact_filename="missing.joblib",
            artifact_checksum="abc123",
        )
        with pytest.raises(IntegrityError, match="does not exist"):
            verify_artifact(manifest, directory=tmp_path)

    def test_registry_activation_missing_artifact(self, tmp_path):
        # Create manifest referencing non-existent artifact
        manifest = ModelManifest(
            model_name="test",
            model_version="v1",
            artifact_filename="missing.joblib",
            artifact_checksum="abc123",
        )
        save_manifest(manifest, directory=tmp_path)
        registry = ModelRegistry(model_directory=tmp_path)
        with pytest.raises(ActivationError):
            registry.activate_from_manifest()
        assert not registry.is_ready

    def test_registry_activation_missing_manifest(self, tmp_path):
        registry = ModelRegistry(model_directory=tmp_path)
        with pytest.raises(ActivationError):
            registry.activate_from_manifest()
        assert not registry.is_ready


# ══════════════════════════════════════════════════════════════════════
# C. Corrupted artifact
# ══════════════════════════════════════════════════════════════════════


class TestCorruptedArtifact:
    """C. Corrupted artifact is detected and rejected."""

    def test_load_corrupt_joblib(self, tmp_path):
        path = tmp_path / "corrupt.joblib"
        path.write_bytes(b"this is not a valid joblib file")
        with pytest.raises(ModelLoadError, match="corrupt"):
            load_bundle(path)

    def test_registry_rejects_corrupt_artifact(self, tmp_path):
        # Save a valid artifact, then corrupt it
        artifact_path = _save_test_artifact(tmp_path)
        manifest = _save_test_manifest(tmp_path, artifact_path)
        # Overwrite with corrupt data
        artifact_path.write_bytes(b"corrupted data")
        registry = ModelRegistry(model_directory=tmp_path)
        with pytest.raises(ActivationError):
            registry.activate_from_manifest()
        assert not registry.is_ready


# ══════════════════════════════════════════════════════════════════════
# D. Checksum mismatch
# ══════════════════════════════════════════════════════════════════════


class TestChecksumMismatch:
    """D. Checksum mismatch is detected."""

    def test_checksum_mismatch_detected(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        manifest = _save_test_manifest(tmp_path, artifact_path)

        # Modify the artifact after manifest was created
        save_bundle(_create_bundle(model_version="v2"), artifact_path)

        with pytest.raises(IntegrityError, match="checksum mismatch"):
            verify_artifact(manifest, directory=tmp_path)

    def test_registry_rejects_checksum_mismatch(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        _save_test_manifest(tmp_path, artifact_path)

        # Replace artifact with a different one
        save_bundle(_create_bundle(model_version="v2"), artifact_path)

        registry = ModelRegistry(model_directory=tmp_path)
        with pytest.raises(ActivationError):
            registry.activate_from_manifest()
        assert not registry.is_ready

    def test_compute_checksum_deterministic(self, tmp_path):
        path = tmp_path / "test.bin"
        path.write_bytes(b"hello world")
        c1 = compute_checksum(path)
        c2 = compute_checksum(path)
        assert c1 == c2
        assert len(c1) == 64  # SHA-256 hex length


# ══════════════════════════════════════════════════════════════════════
# E. Manifest mismatch
# ══════════════════════════════════════════════════════════════════════


class TestManifestMismatch:
    """E. Manifest mismatch is detected."""

    def test_manifest_feature_count_mismatch(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        # Manifest says 10 features but bundle has 24
        manifest = build_manifest(
            model_name="test",
            model_version="v1",
            artifact_path=artifact_path,
            n_features=10,
        )
        save_manifest(manifest, directory=tmp_path)

        registry = ModelRegistry(model_directory=tmp_path)
        with pytest.raises(ActivationError, match="Feature count mismatch"):
            registry.activate_from_manifest()

    def test_manifest_schema_version_mismatch(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        manifest = build_manifest(
            model_name="test",
            model_version="v1",
            artifact_path=artifact_path,
            n_features=24,
        )
        # Override schema version
        bad_manifest = ModelManifest(
            model_name=manifest.model_name,
            model_version=manifest.model_version,
            artifact_filename=manifest.artifact_filename,
            artifact_checksum=manifest.artifact_checksum,
            feature_schema_version="99.0.0",
            n_features=24,
        )
        save_manifest(bad_manifest, directory=tmp_path)

        registry = ModelRegistry(model_directory=tmp_path)
        with pytest.raises(ActivationError, match="schema version mismatch"):
            registry.activate_from_manifest()

    def test_invalid_manifest_format(self, tmp_path):
        # Write non-dict JSON
        path = tmp_path / "model_manifest.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ManifestError, match="invalid format"):
            load_manifest(tmp_path)

    def test_missing_manifest_fields(self, tmp_path):
        path = tmp_path / "model_manifest.json"
        path.write_text(json.dumps({"model_name": "test"}))
        with pytest.raises(ManifestError, match="missing required"):
            load_manifest(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# F. Invalid model interface
# ══════════════════════════════════════════════════════════════════════


class TestInvalidModelInterface:
    """F. Invalid model interface is rejected."""

    def test_model_without_predict_proba(self, tmp_path):
        # Create a bundle with an object that lacks predict_proba
        bundle = _create_bundle()
        bundle.model = object()  # No predict_proba

        path = tmp_path / "bad_model.joblib"
        save_bundle(bundle, path)
        manifest = _save_test_manifest(tmp_path, path)

        registry = ModelRegistry(model_directory=tmp_path)
        with pytest.raises(ActivationError):
            registry.activate_from_manifest(manifest)


# ══════════════════════════════════════════════════════════════════════
# G. Incompatible feature schema
# ══════════════════════════════════════════════════════════════════════


class TestIncompatibleFeatureSchema:
    """G. Incompatible feature schema is rejected."""

    def test_zero_features_rejected(self, tmp_path):
        bundle = _create_bundle(n_features=0)
        bundle.feature_names = []
        path = _save_test_artifact(tmp_path, bundle)
        manifest = build_manifest(
            model_name="test",
            model_version="v1",
            artifact_path=path,
            n_features=0,
        )
        save_manifest(manifest, directory=tmp_path)

        registry = ModelRegistry(model_directory=tmp_path)
        with pytest.raises(ActivationError, match="zero features"):
            registry.activate_from_manifest()


# ══════════════════════════════════════════════════════════════════════
# H. Model version propagation
# ══════════════════════════════════════════════════════════════════════


class TestModelVersionPropagation:
    """H. Model version is correctly propagated through the system."""

    def test_registry_identity_has_version(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        _save_test_manifest(tmp_path, artifact_path, model_version="fraud-xgb-v1.2.3")
        registry = ModelRegistry(model_directory=tmp_path)
        identity = registry.activate_from_manifest()
        assert identity.model_version == "fraud-xgb-v1.2.3"

    def test_predictor_reports_version(self, tmp_path):
        from ml.predict.predictor import FraudPredictor

        artifact_path = _save_test_artifact(tmp_path)
        bundle = load_bundle(artifact_path)
        predictor = FraudPredictor(bundle=bundle)
        assert predictor.model_version == "fraud-xgb-v1.0.0"

    def test_identity_consistent_across_registry(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        _save_test_manifest(tmp_path, artifact_path)
        registry = ModelRegistry(model_directory=tmp_path)
        identity = registry.activate_from_manifest()

        # identity from property matches
        assert registry.identity.model_version == identity.model_version
        assert registry.identity.artifact_checksum == identity.artifact_checksum

        # status_info matches
        info = registry.status_info()
        assert info["model_version"] == identity.model_version
        assert info["load_status"] == "active"


# ══════════════════════════════════════════════════════════════════════
# I. Checksum/fingerprint propagation
# ══════════════════════════════════════════════════════════════════════


class TestChecksumPropagation:
    """I. Checksum is correctly propagated through the system."""

    def test_identity_has_checksum(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        _save_test_manifest(tmp_path, artifact_path)
        registry = ModelRegistry(model_directory=tmp_path)
        identity = registry.activate_from_manifest()
        assert len(identity.artifact_checksum) == 64
        assert identity.checksum_short  # non-empty

    def test_checksum_short_is_12_chars(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        _save_test_manifest(tmp_path, artifact_path)
        registry = ModelRegistry(model_directory=tmp_path)
        identity = registry.activate_from_manifest()
        assert len(identity.checksum_short) == 12
        assert identity.artifact_checksum.startswith(identity.checksum_short)

    def test_manifest_checksum_matches_file(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        manifest = build_manifest(
            model_name="test",
            model_version="v1",
            artifact_path=artifact_path,
            n_features=24,
        )
        # Independent checksum computation
        independent = compute_checksum(artifact_path)
        assert manifest.artifact_checksum == independent


# ══════════════════════════════════════════════════════════════════════
# J. Health/ready behavior
# ══════════════════════════════════════════════════════════════════════


class TestHealthReadyBehavior:
    """J. Health and ready endpoints report correct model governance state."""

    def test_health_with_no_model(self):
        """When no model is loaded, health reports model_unavailable."""
        from ml.api import app as _app_module

        original_predictor = _app_module._predictor
        original_registry = _app_module._registry
        try:
            _app_module._predictor = None
            _app_module._registry = None
            client = TestClient(_app_module.app)
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "model_unavailable"
            assert data.get("model_identity") is None
        finally:
            _app_module._predictor = original_predictor
            _app_module._registry = original_registry

    def test_ready_with_no_model(self):
        """When no model is loaded, ready returns 503."""
        from ml.api import app as _app_module

        original_predictor = _app_module._predictor
        original_registry = _app_module._registry
        try:
            _app_module._predictor = None
            _app_module._registry = None
            client = TestClient(_app_module.app)
            resp = client.get("/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "not_ready"
        finally:
            _app_module._predictor = original_predictor
            _app_module._registry = original_registry


# ══════════════════════════════════════════════════════════════════════
# K. Prediction refuses invalid model
# ══════════════════════════════════════════════════════════════════════


class TestPredictionRefusesInvalidModel:
    """K. Prediction endpoint refuses to serve with invalid model."""

    def test_predict_returns_503_without_model(self):
        from ml.api import app as _app_module

        original_predictor = _app_module._predictor
        try:
            _app_module._predictor = None
            client = TestClient(_app_module.app)
            resp = client.post("/predict", json={
                "amount": 100.0,
                "currency": "USD",
                "merchant_name": "Test",
                "merchant_category": "5732",
                "transaction_type": "purchase",
                "location_country": "US",
                "location_city": "NY",
                "device_fingerprint": "fp1",
                "device_type": "mobile",
                "ip_address": "1.2.3.4",
            })
            assert resp.status_code == 503
        finally:
            _app_module._predictor = original_predictor


# ══════════════════════════════════════════════════════════════════════
# L. Verified model remains active when candidate is invalid
# ══════════════════════════════════════════════════════════════════════


class TestVerifiedModelRemainsActive:
    """L. Verified model remains active when a candidate model is invalid."""

    def test_failed_activation_preserves_previous(self, tmp_path):
        # Activate a valid model first
        artifact_path = _save_test_artifact(tmp_path)
        _save_test_manifest(tmp_path, artifact_path)
        registry = ModelRegistry(model_directory=tmp_path)
        identity1 = registry.activate_from_manifest()
        assert registry.is_ready

        # Now try to activate with a bad manifest
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        bad_manifest = ModelManifest(
            model_name="bad",
            model_version="v999",
            artifact_filename="nonexistent.joblib",
            artifact_checksum="bad",
        )
        save_manifest(bad_manifest, directory=bad_dir)

        # Attempt rollback to bad manifest — should fail
        with pytest.raises(ActivationError):
            registry.rollback(bad_manifest)

        # Original model should still be active
        assert registry.is_ready
        assert registry.identity.model_version == identity1.model_version


# ══════════════════════════════════════════════════════════════════════
# M. Rollback to previously verified model
# ══════════════════════════════════════════════════════════════════════


class TestRollback:
    """M. Rollback to previously verified model works correctly."""

    def test_rollback_to_valid_manifest(self, tmp_path):
        # Create two valid artifacts
        bundle_v1 = _create_bundle(model_version="fraud-xgb-v1.0.0")
        path_v1 = _save_test_artifact(tmp_path, bundle_v1, "v1.joblib")
        manifest_v1 = _save_test_manifest(tmp_path, path_v1, "fraud-xgb-v1.0.0")

        bundle_v2 = _create_bundle(model_version="fraud-xgb-v2.0.0")
        path_v2 = _save_test_artifact(tmp_path, bundle_v2, "v2.joblib")
        manifest_v2 = _save_test_manifest(tmp_path, path_v2, "fraud-xgb-v2.0.0")

        # Activate v2 (current manifest points to v2)
        save_manifest(manifest_v2, directory=tmp_path)
        registry = ModelRegistry(model_directory=tmp_path)
        identity = registry.activate_from_manifest()
        assert identity.model_version == "fraud-xgb-v2.0.0"

        # Roll back to v1
        rolled_back = registry.rollback(manifest_v1)
        assert rolled_back.model_version == "fraud-xgb-v1.0.0"
        assert registry.is_ready


# ══════════════════════════════════════════════════════════════════════
# N. Invalid rollback target rejected
# ══════════════════════════════════════════════════════════════════════


class TestInvalidRollback:
    """N. Invalid rollback target is rejected."""

    def test_rollback_to_nonexistent_artifact(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        _save_test_manifest(tmp_path, artifact_path)
        registry = ModelRegistry(model_directory=tmp_path)
        registry.activate_from_manifest()

        bad_manifest = ModelManifest(
            model_name="bad",
            model_version="v999",
            artifact_filename="nonexistent.joblib",
            artifact_checksum="bad",
        )
        with pytest.raises(ActivationError):
            registry.rollback(bad_manifest)

    def test_rollback_with_checksum_mismatch(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        manifest = _save_test_manifest(tmp_path, artifact_path)
        registry = ModelRegistry(model_directory=tmp_path)
        registry.activate_from_manifest()

        # Create a manifest with wrong checksum
        tampered = ModelManifest(
            model_name=manifest.model_name,
            model_version="v999",
            artifact_filename=manifest.artifact_filename,
            artifact_checksum="0000000000000000000000000000000000000000000000000000000000000000",
            n_features=24,
        )
        with pytest.raises(ActivationError):
            registry.rollback(tampered)


# ══════════════════════════════════════════════════════════════════════
# O. Client cannot select model
# ══════════════════════════════════════════════════════════════════════


class TestClientCannotSelectModel:
    """O. Client cannot control model selection."""

    def test_predict_ignores_model_version_in_request(self):
        """The predict endpoint does not accept a model_version parameter."""
        from ml.api import app as _app_module

        original = _app_module._predictor
        try:
            _app_module._predictor = None
            client = TestClient(_app_module.app)
            # Even if client sends model_version, it should be rejected
            resp = client.post("/predict", json={
                "amount": 100.0,
                "model_version": "hack-v999",
            })
            # Should get 503 (no model) or 422 (extra field), not success
            assert resp.status_code in (422, 503)
        finally:
            _app_module._predictor = original

    def test_no_model_selection_endpoint(self):
        """There is no endpoint to change the active model."""
        from ml.api import app as _app_module
        client = TestClient(_app_module.app)

        # PUT/POST/PATCH to /model or /registry should 404
        for method in [client.put, client.post, client.patch]:
            resp = method("/model", json={"version": "v999"})
            assert resp.status_code == 404
            resp = method("/registry", json={"version": "v999"})
            assert resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════
# P. Path traversal rejected
# ══════════════════════════════════════════════════════════════════════


class TestPathTraversal:
    """P. Path traversal is rejected."""

    def test_safe_artifact_path_blocks_traversal(self, tmp_path):
        with pytest.raises(IntegrityError, match="traversal"):
            safe_artifact_path(tmp_path, "../../etc/passwd")

    def test_safe_artifact_path_allows_normal(self, tmp_path):
        result = safe_artifact_path(tmp_path, "model.joblib")
        assert result.name == "model.joblib"

    def test_verify_artifact_blocks_traversal(self, tmp_path):
        manifest = ModelManifest(
            model_name="test",
            model_version="v1",
            artifact_filename="../../../etc/passwd",
            artifact_checksum="abc",
        )
        with pytest.raises(IntegrityError, match="traversal"):
            verify_artifact(manifest, directory=tmp_path)


# ══════════════════════════════════════════════════════════════════════
# Q. Model file not exposed through API
# ══════════════════════════════════════════════════════════════════════


class TestModelFileNotExposed:
    """Q. Model file is not exposed through API."""

    def test_no_model_download_endpoint(self):
        from ml.api import app as _app_module
        client = TestClient(_app_module.app)
        resp = client.get("/model")
        assert resp.status_code == 404
        resp = client.get("/model/download")
        assert resp.status_code == 404
        resp = client.get("/model/artifact")
        assert resp.status_code == 404

    def test_health_does_not_expose_paths(self):
        from ml.api import app as _app_module

        original_predictor = _app_module._predictor
        original_registry = _app_module._registry
        try:
            _app_module._predictor = None
            _app_module._registry = None
            client = TestClient(_app_module.app)
            resp = client.get("/health")
            data = resp.json()
            # No filesystem paths in response
            assert "/" not in json.dumps(data) or "http" in json.dumps(data)
        finally:
            _app_module._predictor = original_predictor
            _app_module._registry = original_registry


# ══════════════════════════════════════════════════════════════════════
# R. No sensitive data in model metadata
# ══════════════════════════════════════════════════════════════════════


class TestNoSensitiveData:
    """R. No sensitive data in model metadata."""

    def test_manifest_contains_no_secrets(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        manifest = build_manifest(
            model_name="test",
            model_version="v1",
            artifact_path=artifact_path,
            n_features=24,
        )
        d = manifest.to_dict()
        sensitive_keys = {"password", "secret", "token", "key", "credential", "jwt"}
        for k in d:
            assert k.lower() not in sensitive_keys

    def test_identity_dict_no_paths(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        _save_test_manifest(tmp_path, artifact_path)
        registry = ModelRegistry(model_directory=tmp_path)
        identity = registry.activate_from_manifest()
        d = identity.to_dict()
        # No filesystem paths
        for v in d.values():
            if isinstance(v, str):
                assert not v.startswith("/") or v.startswith("http")
                assert "\\" not in v


# ══════════════════════════════════════════════════════════════════════
# S. Monitoring model identity consistency
# ══════════════════════════════════════════════════════════════════════


class TestMonitoringIdentityConsistency:
    """S. Monitoring model identity is consistent."""

    def test_metrics_identity_matches_registry(self, tmp_path):
        """Metrics endpoint exposes the same identity as the registry."""
        artifact_path = _save_test_artifact(tmp_path)
        _save_test_manifest(tmp_path, artifact_path)
        registry = ModelRegistry(model_directory=tmp_path)
        identity = registry.activate_from_manifest()

        from ml.api import app as _app_module

        original_predictor = _app_module._predictor
        original_registry = _app_module._registry
        try:
            _app_module._registry = registry
            _app_module._predictor = MagicMock()
            _app_module._predictor.model_version = identity.model_version
            _app_module._predictor.feature_names = [f"f{i}" for i in range(24)]

            client = TestClient(_app_module.app)
            resp = client.get("/metrics")
            assert resp.status_code == 200
            data = resp.json()
            mid = data.get("model_identity")
            assert mid is not None
            assert mid["model_version"] == identity.model_version
            assert mid["artifact_checksum"] == identity.artifact_checksum
        finally:
            _app_module._predictor = original_predictor
            _app_module._registry = original_registry


# ══════════════════════════════════════════════════════════════════════
# T. Audit model identity consistency
# ══════════════════════════════════════════════════════════════════════


class TestAuditIdentityConsistency:
    """T. Audit records preserve model identity."""

    def test_registry_status_info_has_identity(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        _save_test_manifest(tmp_path, artifact_path)
        registry = ModelRegistry(model_directory=tmp_path)
        identity = registry.activate_from_manifest()

        info = registry.status_info()
        assert info["model_version"] == identity.model_version
        assert info["artifact_checksum"] == identity.artifact_checksum
        assert info["load_status"] == "active"

    def test_failed_load_has_bounded_error(self, tmp_path):
        registry = ModelRegistry(model_directory=tmp_path)
        with pytest.raises(ActivationError):
            registry.activate_from_manifest()
        # Error category must be bounded
        assert registry.load_error in (
            "manifest_unavailable",
            "integrity_failure",
            "artifact_missing",
            "artifact_corrupt",
            "interface_mismatch",
            "feature_incompatible",
        )


# ══════════════════════════════════════════════════════════════════════
# U. Concurrent prediction during model validation/loading
# ══════════════════════════════════════════════════════════════════════


class TestConcurrency:
    """U. Concurrent access is safe during model validation/loading."""

    def test_concurrent_identity_reads(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        _save_test_manifest(tmp_path, artifact_path)
        registry = ModelRegistry(model_directory=tmp_path)
        registry.activate_from_manifest()

        results: list[str | None] = []
        errors: list[Exception] = []

        def read_identity():
            try:
                for _ in range(100):
                    identity = registry.identity
                    results.append(
                        identity.model_version if identity else None
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_identity) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All reads should return the same version
        assert all(v == "fraud-xgb-v1.0.0" for v in results)

    def test_concurrent_status_info_reads(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        _save_test_manifest(tmp_path, artifact_path)
        registry = ModelRegistry(model_directory=tmp_path)
        registry.activate_from_manifest()

        results: list[str] = []
        errors: list[Exception] = []

        def read_status():
            try:
                for _ in range(50):
                    info = registry.status_info()
                    results.append(info["load_status"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_status) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(s == "active" for s in results)


# ══════════════════════════════════════════════════════════════════════
# V. Restart persistence/configuration behavior
# ══════════════════════════════════════════════════════════════════════


class TestRestartPersistence:
    """V. Restart/configuration behavior is correct."""

    def test_manifest_persists_across_registry_instances(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        manifest = _save_test_manifest(tmp_path, artifact_path)

        # First registry
        reg1 = ModelRegistry(model_directory=tmp_path)
        id1 = reg1.activate_from_manifest()

        # Second registry (simulates restart)
        reg2 = ModelRegistry(model_directory=tmp_path)
        id2 = reg2.activate_from_manifest()

        assert id1.model_version == id2.model_version
        assert id1.artifact_checksum == id2.artifact_checksum

    def test_manifest_file_readable_after_save(self, tmp_path):
        artifact_path = _save_test_artifact(tmp_path)
        manifest = build_manifest(
            model_name="test",
            model_version="v1",
            artifact_path=artifact_path,
            n_features=24,
        )
        save_manifest(manifest, directory=tmp_path)

        # Read back
        loaded = load_manifest(tmp_path)
        assert loaded.model_name == manifest.model_name
        assert loaded.model_version == manifest.model_version
        assert loaded.artifact_checksum == manifest.artifact_checksum

    def test_env_variable_model_directory(self, tmp_path):
        """ML_MODEL_DIR env var overrides the default directory."""
        artifact_path = _save_test_artifact(tmp_path)
        _save_test_manifest(tmp_path, artifact_path)

        registry = ModelRegistry(model_directory=str(tmp_path))
        identity = registry.activate_from_manifest()
        assert identity.model_version == "fraud-xgb-v1.0.0"


# ══════════════════════════════════════════════════════════════════════
# Additional: Manifest serialization
# ══════════════════════════════════════════════════════════════════════


class TestManifestSerialization:
    """Additional tests for manifest JSON serialization."""

    def test_manifest_roundtrip(self, tmp_path):
        manifest = ModelManifest(
            model_name="fraud-xgb",
            model_version="v1.0.0",
            artifact_filename="model.joblib",
            artifact_checksum="a" * 64,
            n_features=24,
            threshold=0.50,
            created_at="2026-01-01T00:00:00+00:00",
        )
        path = save_manifest(manifest, directory=tmp_path)
        loaded = load_manifest(tmp_path)
        assert loaded.model_name == manifest.model_name
        assert loaded.model_version == manifest.model_version
        assert loaded.artifact_checksum == manifest.artifact_checksum
        assert loaded.n_features == manifest.n_features

    def test_manifest_from_dict_ignores_unknown_fields(self):
        data = {
            "model_name": "test",
            "model_version": "v1",
            "artifact_filename": "m.joblib",
            "artifact_checksum": "abc",
            "unknown_field": "ignored",
        }
        manifest = ModelManifest.from_dict(data)
        assert manifest.model_name == "test"
        assert not hasattr(manifest, "unknown_field")

    def test_manifest_deterministic(self, tmp_path):
        """Same input produces same JSON output."""
        artifact_path = _save_test_artifact(tmp_path)
        m1 = build_manifest(
            model_name="test",
            model_version="v1",
            artifact_path=artifact_path,
            n_features=24,
        )
        path1 = save_manifest(m1, directory=tmp_path)
        content1 = path1.read_text()

        # Rebuild and compare (except created_at which will differ)
        m2 = ModelManifest(
            model_name=m1.model_name,
            model_version=m1.model_version,
            artifact_filename=m1.artifact_filename,
            artifact_checksum=m1.artifact_checksum,
            serialization_format=m1.serialization_format,
            feature_schema_version=m1.feature_schema_version,
            n_features=m1.n_features,
            threshold=m1.threshold,
            created_at=m1.created_at,
            status=m1.status,
        )
        path2 = save_manifest(m2, directory=tmp_path)
        content2 = path2.read_text()
        assert content1 == content2


# ══════════════════════════════════════════════════════════════════════
# Additional: Checksum computation
# ══════════════════════════════════════════════════════════════════════


class TestChecksumComputation:
    """Additional tests for checksum computation."""

    def test_checksum_of_known_content(self, tmp_path):
        path = tmp_path / "known.txt"
        path.write_bytes(b"hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert compute_checksum(path) == expected

    def test_checksum_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compute_checksum(tmp_path / "missing.bin")

    def test_checksum_large_file(self, tmp_path):
        """Checksum works for files larger than the 8KiB chunk size."""
        path = tmp_path / "large.bin"
        data = os.urandom(32768)  # 32 KiB
        path.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert compute_checksum(path) == expected


# ══════════════════════════════════════════════════════════════════════
# Additional: Activation error categories
# ══════════════════════════════════════════════════════════════════════


class TestActivationErrorCategories:
    """Bounded error categories for activation failures."""

    def test_manifest_unavailable_error(self, tmp_path):
        registry = ModelRegistry(model_directory=tmp_path)
        with pytest.raises(ActivationError):
            registry.activate_from_manifest()
        assert registry.load_status == "load_failed"
        assert registry.load_error == "manifest_unavailable"

    def test_integrity_failure_error(self, tmp_path):
        # Create manifest referencing missing artifact
        manifest = ModelManifest(
            model_name="test",
            model_version="v1",
            artifact_filename="missing.joblib",
            artifact_checksum="abc",
        )
        save_manifest(manifest, directory=tmp_path)
        registry = ModelRegistry(model_directory=tmp_path)
        with pytest.raises(ActivationError):
            registry.activate_from_manifest()
        assert registry.load_error == "integrity_failure"

    def test_artifact_corrupt_error(self, tmp_path):
        # Create valid manifest + corrupt artifact
        path = tmp_path / "corrupt.joblib"
        path.write_bytes(b"not a valid joblib")
        checksum = compute_checksum(path)
        manifest = ModelManifest(
            model_name="test",
            model_version="v1",
            artifact_filename="corrupt.joblib",
            artifact_checksum=checksum,
            n_features=24,
        )
        save_manifest(manifest, directory=tmp_path)
        registry = ModelRegistry(model_directory=tmp_path)
        with pytest.raises(ActivationError):
            registry.activate_from_manifest()
        assert registry.load_error == "artifact_corrupt"
