"""Lightweight model registry — active model identity and rollback.

Step 46: Production model lifecycle and version governance.

The :class:`ModelRegistry` is the single authoritative source for
the active model identity.  It wraps a :class:`ModelManifest` and
provides:

* :meth:`activate` — activate a model after integrity + interface checks.
* :meth:`identity` — authoritative identity dict (name, version,
  checksum, schema).
* :meth:`rollback` — switch to a previously verified manifest.
* :attr:`is_ready` — whether a validated model is currently active.

The registry does **not** expose raw filesystem paths, secrets, or
model binaries.  It is designed for startup/configuration-based
activation, not runtime hot-swap.

Concurrency
-----------
All public methods are protected by a ``threading.Lock`` so concurrent
health/ready/predict requests see a consistent identity.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml.predict.bundle import ModelBundle, ModelLoadError, load_bundle
from ml.predict.integrity import (
    IntegrityError,
    ManifestError,
    ModelManifest,
    load_manifest,
    verify_artifact,
    safe_artifact_path,
    _FEATURE_SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────


class ActivationError(Exception):
    """Model activation failed — integrity, interface, or validation error."""


# ── Identity container ───────────────────────────────────────────────


@dataclass(frozen=True)
class ModelIdentity:
    """Authoritative identity of the active model.

    Attributes:
        model_name: Descriptive model name.
        model_version: Version string.
        artifact_checksum: SHA-256 hex digest (short prefix safe to expose).
        feature_schema_version: Feature pipeline version.
        n_features: Number of features.
        status: Lifecycle status (``active``).
    """

    model_name: str
    model_version: str
    artifact_checksum: str
    feature_schema_version: str
    n_features: int
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "artifact_checksum": self.artifact_checksum,
            "feature_schema_version": self.feature_schema_version,
            "n_features": self.n_features,
            "status": self.status,
        }

    @property
    def checksum_short(self) -> str:
        """First 12 hex chars of the checksum for display."""
        return self.artifact_checksum[:12] if self.artifact_checksum else ""


# ── Registry ──────────────────────────────────────────────────────────


class ModelRegistry:
    """Authoritative source for active model identity.

    Typical lifecycle:

    1. Construct with the model directory.
    2. Call :meth:`activate_from_manifest` at startup.
    3. Use :attr:`identity` and :attr:`bundle` for predictions.
    4. Optionally call :meth:`rollback` for controlled version switch.
    """

    def __init__(self, model_directory: str | Path | None = None) -> None:
        from ml.predict.bundle import _DEFAULT_MODEL_DIR, _DEFAULT_FILENAME

        self._directory = Path(model_directory) if model_directory else _DEFAULT_MODEL_DIR
        self._lock = threading.Lock()

        # Active state — None means no validated model is loaded
        self._active_identity: ModelIdentity | None = None
        self._active_bundle: ModelBundle | None = None
        self._active_manifest: ModelManifest | None = None

        # Activation status for monitoring
        self._load_status: str = "not_loaded"
        self._load_error: str | None = None

    # ── Properties ────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """Whether a validated model is currently active."""
        with self._lock:
            return self._active_bundle is not None

    @property
    def identity(self) -> ModelIdentity | None:
        """Active model identity, or ``None`` if no model is loaded."""
        with self._lock:
            return self._active_identity

    @property
    def bundle(self) -> ModelBundle | None:
        """Active model bundle, or ``None`` if no model is loaded."""
        with self._lock:
            return self._active_bundle

    @property
    def load_status(self) -> str:
        """Current load status: ``active``, ``not_loaded``, ``load_failed``."""
        with self._lock:
            return self._load_status

    @property
    def load_error(self) -> str | None:
        """Bounded error category for the last load failure, if any."""
        with self._lock:
            return self._load_error

    @property
    def manifest(self) -> ModelManifest | None:
        """Active model manifest, or ``None``."""
        with self._lock:
            return self._active_manifest

    # ── Activation ────────────────────────────────────────────────

    def activate_from_manifest(
        self,
        manifest: ModelManifest | None = None,
    ) -> ModelIdentity:
        """Activate a model through the full validation pipeline.

        Sequence:
          1. Load manifest (if not provided).
          2. Verify artifact checksum.
          3. Load model bundle.
          4. Validate model interface.
          5. Validate feature compatibility.
          6. Mark active.

        Args:
            manifest: Explicit manifest. If ``None``, loads from disk.

        Returns:
            The active :class:`ModelIdentity`.

        Raises:
            ActivationError: If any validation stage fails.
        """
        with self._lock:
            return self._activate_locked(manifest)

    def _activate_locked(
        self,
        manifest: ModelManifest | None,
    ) -> ModelIdentity:
        """Internal activation — must be called with ``_lock`` held."""

        # ── 1. Manifest ──────────────────────────────────────────
        if manifest is None:
            try:
                manifest = load_manifest(self._directory)
            except ManifestError as exc:
                self._set_load_failed("manifest_unavailable")
                raise ActivationError(str(exc)) from exc

        # ── 2. Integrity ─────────────────────────────────────────
        try:
            verify_artifact(manifest, self._directory)
        except IntegrityError as exc:
            self._set_load_failed("integrity_failure")
            raise ActivationError(str(exc)) from exc

        # ── 3. Load bundle ───────────────────────────────────────
        artifact_path = safe_artifact_path(
            self._directory, manifest.artifact_filename
        )
        try:
            bundle = load_bundle(artifact_path)
        except FileNotFoundError as exc:
            self._set_load_failed("artifact_missing")
            raise ActivationError(str(exc)) from exc
        except ModelLoadError as exc:
            self._set_load_failed("artifact_corrupt")
            raise ActivationError(str(exc)) from exc

        # ── 4. Interface validation ──────────────────────────────
        try:
            self._validate_interface(bundle, manifest)
        except ActivationError:
            self._set_load_failed("interface_mismatch")
            raise

        # ── 5. Feature compatibility ─────────────────────────────
        try:
            self._validate_features(bundle, manifest)
        except ActivationError:
            self._set_load_failed("feature_incompatible")
            raise

        # ── 6. Activate ──────────────────────────────────────────
        identity = ModelIdentity(
            model_name=manifest.model_name,
            model_version=manifest.model_version,
            artifact_checksum=manifest.artifact_checksum,
            feature_schema_version=manifest.feature_schema_version,
            n_features=manifest.n_features,
            status="active",
        )

        self._active_manifest = manifest
        self._active_bundle = bundle
        self._active_identity = identity
        self._load_status = "active"
        self._load_error = None

        logger.info(
            "Model activated: version=%s checksum=%s",
            identity.model_version,
            identity.checksum_short,
        )
        return identity

    # ── Rollback ──────────────────────────────────────────────────

    def rollback(
        self,
        manifest: ModelManifest,
    ) -> ModelIdentity:
        """Roll back to a previously verified model version.

        The target manifest must still pass full integrity and
        interface validation.  If rollback fails, the current active
        model remains unchanged.

        Args:
            manifest: The target manifest to activate.

        Returns:
            The new active :class:`ModelIdentity`.

        Raises:
            ActivationError: If validation fails for the target.
        """
        # activate_from_manifest acquires the lock and runs full validation.
        # If it fails, the previous active model is untouched because we
        # only update _active_* fields after all checks pass.
        return self.activate_from_manifest(manifest)

    # ── Validation helpers ────────────────────────────────────────

    @staticmethod
    def _validate_interface(bundle: ModelBundle, manifest: ModelManifest) -> None:
        """Validate that the loaded model has the required interface."""
        model = bundle.model
        if not hasattr(model, "predict_proba"):
            raise ActivationError(
                "Model does not expose predict_proba method."
            )
        if not callable(getattr(model, "predict_proba", None)):
            raise ActivationError(
                "Model predict_proba is not callable."
            )

    @staticmethod
    def _validate_features(bundle: ModelBundle, manifest: ModelManifest) -> None:
        """Validate feature schema compatibility."""
        # Manifest n_features must match bundle feature count
        if manifest.n_features > 0 and manifest.n_features != bundle.n_features:
            raise ActivationError(
                f"Feature count mismatch: manifest expects "
                f"{manifest.n_features}, bundle has {bundle.n_features}."
            )

        # Bundle must have at least some features
        if bundle.n_features == 0:
            raise ActivationError("Model bundle has zero features.")

        # Feature schema version must match
        if (
            manifest.feature_schema_version
            and manifest.feature_schema_version != _FEATURE_SCHEMA_VERSION
        ):
            raise ActivationError(
                f"Feature schema version mismatch: manifest has "
                f"{manifest.feature_schema_version}, "
                f"expected {_FEATURE_SCHEMA_VERSION}."
            )

    def _set_load_failed(self, error_category: str) -> None:
        """Record a bounded load failure category."""
        self._load_status = "load_failed"
        self._load_error = error_category

    # ── Info ──────────────────────────────────────────────────────

    def status_info(self) -> dict[str, Any]:
        """Return a summary dict for health/monitoring endpoints.

        Never exposes filesystem paths, secrets, or model binaries.
        """
        with self._lock:
            info: dict[str, Any] = {
                "load_status": self._load_status,
            }
            if self._active_identity:
                info.update(self._active_identity.to_dict())
            if self._load_error:
                info["load_error"] = self._load_error
            return info
