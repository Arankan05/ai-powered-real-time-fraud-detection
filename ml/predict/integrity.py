"""Model artifact integrity verification and manifest management.

Step 46: Production model lifecycle and version governance.

Provides:

* :func:`compute_checksum` — SHA-256 hash of a model artifact file.
* :class:`ModelManifest` — immutable metadata record for a model version.
* :func:`load_manifest` / :func:`save_manifest` — JSON manifest I/O.
* :func:`verify_artifact` — verify artifact exists and checksum matches.

Trust boundary
--------------
Model artifacts use the **joblib** serialization format, which internally
uses Python pickle.  Pickle deserialization can execute arbitrary code.
Therefore:

* Artifacts **must** originate from a trusted build/training pipeline.
* Checksum verification ensures the artifact on disk has not been
  modified since it was saved — but it does **not** make an untrusted
  artifact safe.
* The manifest and checksum together provide tamper detection, not
  sandboxing.

This module never exposes filesystem paths, raw errors, or secrets
through public interfaces.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

_MANIFEST_FILENAME = "model_manifest.json"
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_FEATURE_SCHEMA_VERSION = "1.0.0"  # matches 24-feature pipeline
_SERIALIZATION_FORMAT = "joblib"


# ── Exceptions ────────────────────────────────────────────────────────


class IntegrityError(Exception):
    """Artifact integrity verification failed."""


class ManifestError(Exception):
    """Manifest is missing, unreadable, or invalid."""


# ── Checksum ──────────────────────────────────────────────────────────


def compute_checksum(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Reads the file in 8 KiB chunks to avoid loading large artifacts
    into memory at once.

    Raises:
        FileNotFoundError: If the file does not exist.
        IntegrityError: If the file cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact file not found: {path.name}")
    try:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except OSError as exc:
        raise IntegrityError(
            f"Cannot read artifact file: {type(exc).__name__}"
        ) from exc


# ── Manifest ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelManifest:
    """Immutable metadata record for a model version.

    Attributes:
        model_name: Descriptive name (e.g. ``fraud-xgb``).
        model_version: Semantic version string (e.g. ``fraud-xgb-v1.0.0``).
        artifact_filename: Basename of the artifact file.
        artifact_checksum: SHA-256 hex digest of the artifact.
        serialization_format: Serialization library (``joblib``).
        feature_schema_version: Feature pipeline version.
        n_features: Number of features expected by the model.
        threshold: Decision threshold used at training time.
        created_at: ISO 8601 timestamp of artifact creation.
        status: Lifecycle status (``active``, ``archived``).
    """

    model_name: str
    model_version: str
    artifact_filename: str
    artifact_checksum: str
    serialization_format: str = _SERIALIZATION_FORMAT
    feature_schema_version: str = _FEATURE_SCHEMA_VERSION
    n_features: int = 0
    threshold: float = 0.5
    created_at: str = ""
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelManifest:
        """Deserialize from a dict (e.g. parsed JSON)."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


def save_manifest(
    manifest: ModelManifest,
    directory: str | Path | None = None,
) -> Path:
    """Write a model manifest to a JSON file.

    Args:
        manifest: The manifest to save.
        directory: Target directory. Defaults to ``ml/models/``.

    Returns:
        Resolved path of the written manifest file.
    """
    directory = Path(directory) if directory else _DEFAULT_MODEL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _MANIFEST_FILENAME

    # Write atomically: write to temp, then rename
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)
    return path.resolve()


def load_manifest(
    directory: str | Path | None = None,
) -> ModelManifest:
    """Read a model manifest from a JSON file.

    Raises:
        ManifestError: If the manifest is missing, unreadable, or invalid.
    """
    directory = Path(directory) if directory else _DEFAULT_MODEL_DIR
    path = directory / _MANIFEST_FILENAME

    if not path.exists():
        raise ManifestError("Model manifest not found.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise ManifestError(
            f"Model manifest is unreadable: {type(exc).__name__}"
        ) from exc

    if not isinstance(data, dict):
        raise ManifestError("Model manifest has invalid format.")

    # Validate required fields
    required = {"model_name", "model_version", "artifact_filename", "artifact_checksum"}
    missing = required - set(data.keys())
    if missing:
        raise ManifestError(
            f"Model manifest is missing required fields: {sorted(missing)}"
        )

    return ModelManifest.from_dict(data)


# ── Verification ──────────────────────────────────────────────────────


def verify_artifact(
    manifest: ModelManifest,
    directory: str | Path | None = None,
) -> bool:
    """Verify that an artifact exists and its checksum matches the manifest.

    Args:
        manifest: The expected manifest.
        directory: Directory containing the artifact.

    Returns:
        ``True`` if verification passes.

    Raises:
        IntegrityError: If the artifact is missing or checksum mismatches.
    """
    directory = Path(directory) if directory else _DEFAULT_MODEL_DIR
    artifact_path = directory / manifest.artifact_filename

    # Prevent path traversal: artifact must be within the expected directory
    try:
        artifact_path.resolve().relative_to(directory.resolve())
    except ValueError:
        raise IntegrityError("Artifact path traversal detected.")

    if not artifact_path.exists():
        raise IntegrityError(
            "Artifact file referenced by manifest does not exist."
        )

    actual_checksum = compute_checksum(artifact_path)
    if actual_checksum != manifest.artifact_checksum:
        raise IntegrityError(
            "Artifact checksum mismatch — artifact may be corrupt or tampered."
        )

    return True


def build_manifest(
    *,
    model_name: str,
    model_version: str,
    artifact_path: str | Path,
    n_features: int = 0,
    threshold: float = 0.5,
) -> ModelManifest:
    """Build a new manifest for a freshly saved artifact.

    Computes the SHA-256 checksum and records the current timestamp.

    Args:
        model_name: Descriptive model name.
        model_version: Version string.
        artifact_path: Path to the saved artifact file.
        n_features: Number of model features.
        threshold: Decision threshold.

    Returns:
        A new :class:`ModelManifest`.
    """
    artifact_path = Path(artifact_path)
    checksum = compute_checksum(artifact_path)
    now = datetime.now(timezone.utc).isoformat()

    return ModelManifest(
        model_name=model_name,
        model_version=model_version,
        artifact_filename=artifact_path.name,
        artifact_checksum=checksum,
        serialization_format=_SERIALIZATION_FORMAT,
        feature_schema_version=_FEATURE_SCHEMA_VERSION,
        n_features=n_features,
        threshold=threshold,
        created_at=now,
        status="active",
    )


# ── Model directory helpers ──────────────────────────────────────────


def default_model_directory() -> Path:
    """Return the default model artifacts directory."""
    return _DEFAULT_MODEL_DIR.resolve()


def safe_artifact_path(
    directory: str | Path,
    filename: str,
) -> Path:
    """Resolve an artifact path safely, preventing directory traversal.

    Raises:
        IntegrityError: If the resolved path escapes the directory.
    """
    directory = Path(directory).resolve()
    artifact = (directory / filename).resolve()
    try:
        artifact.relative_to(directory)
    except ValueError:
        raise IntegrityError("Artifact path traversal detected.")
    return artifact
