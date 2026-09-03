"""Validation suite for model serialisation and prediction pipeline.

Runs 13 checks covering:

  - Model artifact loading
  - Round-trip serialisation integrity
  - Valid prediction
  - Invalid / missing input handling
  - Preprocessing consistency
  - Threshold logic
  - Reproducibility
  - Leakage guards
  - API response schema

Usage::

    python -m ml.predict._validate

Requires a saved model artifact (``python -m ml.predict.save_model``).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features.engineer import FEATURE_LIST
from ml.predict.bundle import (
    ModelBundle,
    default_model_path,
    load_bundle,
    model_exists,
    save_bundle,
)
from ml.predict.predictor import FraudPredictor, PredictionResult

# ── Helpers ───────────────────────────────────────────────────────────

_PASSED = 0
_FAILED = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global _PASSED, _FAILED
    tag = "PASS" if condition else "FAIL"
    if condition:
        _PASSED += 1
    else:
        _FAILED += 1
    suffix = f" — {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")


def _make_sample_features(n: int = 1) -> pd.DataFrame:
    """Create a minimal sample feature DataFrame for testing."""
    rng = np.random.RandomState(42)
    data = {
        "amount": rng.uniform(10, 5000, n),
        "amount_deviation": rng.normal(0, 1, n),
        "amount_to_avg_ratio": rng.uniform(0.5, 3.0, n),
        "location_country": rng.randint(0, 50, n).astype(float),
        "location_region": rng.randint(0, 100, n).astype(float),
        "location_is_new": rng.randint(0, 2, n),
        "location_change": rng.randint(0, 2, n),
        "device_fingerprint": np.where(
            rng.random(n) > 0.3,
            "device_" + rng.randint(1, 10, n).astype(str),
            "no_device_data",
        ),
        "is_new_device": rng.randint(0, 2, n),
        "hour_of_day_raw": rng.randint(0, 24, n),
        "hour_of_day_sin": rng.uniform(-1, 1, n),
        "hour_of_day_cos": rng.uniform(-1, 1, n),
        "day_of_week_raw": rng.randint(0, 7, n),
        "day_of_week_sin": rng.uniform(-1, 1, n),
        "day_of_week_cos": rng.uniform(-1, 1, n),
        "is_unusual_hour": rng.randint(0, 2, n),
        "tx_velocity_1h": rng.randint(0, 5, n),
        "tx_velocity_24h": rng.randint(0, 20, n),
        "tx_velocity_7d": rng.randint(0, 50, n),
        "merchant_category": rng.randint(0, 5, n),
        "merchant_is_new": rng.randint(0, 2, n),
        "avg_spend_30d": rng.uniform(50, 2000, n),
        "previous_suspicious_count": rng.randint(0, 5, n),
        "has_identity_data": rng.randint(0, 2, n),
    }
    return pd.DataFrame(data)


# ── Checks ────────────────────────────────────────────────────────────


def check_01_artifact_exists() -> None:
    """Model artifact file exists at default path."""
    path = default_model_path()
    _check(
        "artifact_exists",
        model_exists(),
        f"path={path}, exists={model_exists()}",
    )


def check_02_load_bundle() -> None:
    """Model bundle loads without errors."""
    try:
        bundle = load_bundle()
        ok = bundle is not None
        detail = f"version={bundle.model_version}, features={bundle.n_features}"
    except Exception as exc:
        ok = False
        detail = str(exc)
        bundle = None
    _check("load_bundle", ok, detail)
    return bundle


def check_03_round_trip(bundle: ModelBundle) -> None:
    """Save → load round-trip preserves all fields."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "test_bundle.joblib"
        save_bundle(bundle, tmp_path)
        loaded = load_bundle(tmp_path)

    ok = (
        loaded.model_version == bundle.model_version
        and loaded.threshold == bundle.threshold
        and loaded.feature_names == bundle.feature_names
        and loaded.n_features == bundle.n_features
    )
    _check(
        "round_trip_serialisation",
        ok,
        f"version={loaded.model_version}, threshold={loaded.threshold:.2f}, "
        f"features={loaded.n_features}",
    )


def check_04_predictor_loads() -> FraudPredictor | None:
    """FraudPredictor initialises from saved artifact."""
    try:
        p = FraudPredictor()
        ok = p.is_loaded
        detail = f"version={p.model_version}"
    except Exception as exc:
        ok = False
        detail = str(exc)
        p = None
    _check("predictor_loads", ok, detail)
    return p


def check_05_valid_prediction(predictor: FraudPredictor) -> None:
    """Valid input produces a PredictionResult."""
    features = _make_sample_features(1)
    try:
        result = predictor.predict(features)
        ok = (
            isinstance(result, PredictionResult)
            and 0.0 <= result.fraud_probability <= 1.0
            and result.fraud_prediction in (0, 1)
            and result.threshold == predictor.threshold
        )
        detail = (
            f"prob={result.fraud_probability:.4f}, "
            f"pred={result.fraud_prediction}, "
            f"threshold={result.threshold:.2f}"
        )
    except Exception as exc:
        ok = False
        detail = str(exc)
    _check("valid_prediction", ok, detail)


def check_06_reproducibility(predictor: FraudPredictor) -> None:
    """Same input produces identical results on repeated calls."""
    features = _make_sample_features(1)
    r1 = predictor.predict(features)
    r2 = predictor.predict(features)
    ok = (
        abs(r1.fraud_probability - r2.fraud_probability) < 1e-10
        and r1.fraud_prediction == r2.fraud_prediction
    )
    _check(
        "reproducibility",
        ok,
        f"run1={r1.fraud_probability:.6f}, run2={r2.fraud_probability:.6f}",
    )


def check_07_threshold_logic(predictor: FraudPredictor) -> None:
    """Prediction matches threshold application on probability."""
    features = _make_sample_features(5)
    results = predictor.predict_batch(features)
    ok = all(
        r.fraud_prediction == (1 if r.fraud_probability >= r.threshold else 0)
        for r in results
    )
    _check("threshold_logic", ok, f"checked {len(results)} predictions")


def check_08_missing_features(predictor: FraudPredictor) -> None:
    """Missing feature columns raise ValueError."""
    features = _make_sample_features(1).drop(columns=["amount"])
    try:
        predictor.predict(features)
        ok = False
        detail = "no error raised"
    except ValueError as exc:
        ok = "Missing" in str(exc) or "missing" in str(exc).lower()
        detail = str(exc)[:80]
    _check("missing_features_rejected", ok, detail)


def check_09_forbidden_columns(predictor: FraudPredictor) -> None:
    """isFraud in input raises ValueError."""
    features = _make_sample_features(1)
    features["isFraud"] = 0
    try:
        predictor.predict(features)
        ok = False
        detail = "no error raised"
    except ValueError as exc:
        ok = "Forbidden" in str(exc) or "forbidden" in str(exc).lower()
        detail = str(exc)[:80]
    _check("isfraud_rejected", ok, detail)


def check_10_empty_input(predictor: FraudPredictor) -> None:
    """Empty DataFrame raises ValueError."""
    features = _make_sample_features(1).iloc[:0]
    try:
        predictor.predict(features)
        ok = False
        detail = "no error raised"
    except ValueError:
        ok = True
        detail = "ValueError raised as expected"
    _check("empty_input_rejected", ok, detail)


def check_11_preprocessing_consistency(bundle: ModelBundle) -> None:
    """Loaded preprocessing has expected artifacts."""
    prep = bundle.preprocessing
    ok = (
        hasattr(prep, "scaler")
        and hasattr(prep, "label_encoders")
        and hasattr(prep, "numeric_cols")
        and hasattr(prep, "categorical_cols")
        and prep.scaler is not None
        and len(prep.numeric_cols) > 0
    )
    _check(
        "preprocessing_consistency",
        ok,
        f"numeric_cols={len(prep.numeric_cols)}, "
        f"categorical_cols={prep.categorical_cols}",
    )


def check_12_feature_schema(bundle: ModelBundle) -> None:
    """Bundle feature names match FEATURE_LIST from engineer.py (as a set).

    Column order in the bundle reflects the actual training DataFrame
    order (the authoritative schema for inference).  FEATURE_LIST is a
    declaration list whose order may differ.  The predictor enforces
    correct ordering via ``features[bundle.feature_names]``.
    """
    bundle_set = set(bundle.feature_names)
    spec_set = set(FEATURE_LIST)
    ok = bundle_set == spec_set and len(bundle.feature_names) == len(FEATURE_LIST)
    if not ok:
        missing = spec_set - bundle_set
        extra = bundle_set - spec_set
        detail = f"missing={missing}, extra={extra}"
    else:
        detail = f"{len(bundle.feature_names)} features match (set equality)"
    _check("feature_schema_match", ok, detail)


def check_13_batch_prediction(predictor: FraudPredictor) -> None:
    """Batch prediction returns correct number of results."""
    features = _make_sample_features(10)
    results = predictor.predict_batch(features)
    ok = len(results) == 10 and all(isinstance(r, PredictionResult) for r in results)
    _check("batch_prediction", ok, f"{len(results)} results for 10 rows")


# ── Main ──────────────────────────────────────────────────────────────

_CHECKS = [
    check_01_artifact_exists,
]


def main() -> None:
    global _PASSED, _FAILED

    print("Prediction Pipeline Validation")
    print("=" * 60)
    print()

    if not model_exists():
        print("  ERROR: Model artifact not found.")
        print(f"  Run: python -m ml.predict.save_model")
        print()
        sys.exit(1)

    # Check 1: artifact exists
    check_01_artifact_exists()

    # Check 2: load bundle
    bundle = check_02_load_bundle()

    # Check 3: round-trip
    check_03_round_trip(bundle)

    # Check 4: predictor loads
    predictor = check_04_predictor_loads()
    if predictor is None:
        print("\n  FATAL: Predictor failed to load. Aborting.")
        sys.exit(1)

    # Checks 5-13
    check_05_valid_prediction(predictor)
    check_06_reproducibility(predictor)
    check_07_threshold_logic(predictor)
    check_08_missing_features(predictor)
    check_09_forbidden_columns(predictor)
    check_10_empty_input(predictor)
    check_11_preprocessing_consistency(bundle)
    check_12_feature_schema(bundle)
    check_13_batch_prediction(predictor)

    print()
    print(f"Results: {_PASSED} passed, {_FAILED} failed")

    if _FAILED > 0:
        sys.exit(1)
    print("\nAll validations PASSED.")


if __name__ == "__main__":
    main()
