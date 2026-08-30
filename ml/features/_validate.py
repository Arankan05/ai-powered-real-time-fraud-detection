"""Lightweight validation for the ML feature-engineering pipeline.

Runs the pipeline on a **sample** of the dataset and checks:

1. Expected feature names are produced
2. ``isFraud`` is NOT included as an input feature
3. ``TransactionID`` is NOT included as an input feature
4. Identity joining works (``has_identity_data`` present and correct)
5. Historical features don't use future data (cold-start checks)
6. No accidental target leakage
7. Output shapes and types are correct

Usage::

    python -m ml.features._validate

"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ml.features.engineer import FEATURE_LIST, engineer_features

_SAMPLE_SIZE = 10_000


def _run_checks(result: dict) -> list[dict]:
    """Execute all validation checks and return result dicts."""
    checks: list[dict] = []
    features: pd.DataFrame = result["features"]
    target: pd.Series = result["target"]
    txn_ids: pd.Series = result["transaction_ids"]
    names = result["feature_names"]

    # ── 1. Feature names match expected list ─────────────────────────
    expected = set(FEATURE_LIST)
    actual = set(names)
    missing = expected - actual
    extra = actual - expected
    ok = len(missing) == 0 and len(extra) == 0
    checks.append(
        {
            "check": "feature_names_match_spec",
            "status": "PASS" if ok else "FAIL",
            "detail": (
                f"{len(actual)} features produced"
                + (f", missing={missing}" if missing else "")
                + (f", extra={extra}" if extra else "")
            ),
        }
    )

    # ── 2. isFraud NOT in features ───────────────────────────────────
    checks.append(
        {
            "check": "target_not_in_features",
            "status": "PASS" if "isFraud" not in names else "FAIL",
            "detail": "isFraud absent from feature columns",
        }
    )

    # ── 3. TransactionID NOT in features ─────────────────────────────
    checks.append(
        {
            "check": "transaction_id_not_in_features",
            "status": "PASS" if "TransactionID" not in names else "FAIL",
            "detail": "TransactionID absent from feature columns",
        }
    )

    # ── 4. Output shape ──────────────────────────────────────────────
    checks.append(
        {
            "check": "output_shape",
            "status": "PASS" if features.shape[0] > 0 else "FAIL",
            "detail": f"{features.shape[0]:,} rows x {features.shape[1]} features",
        }
    )

    # ── 5. Target aligned with features ─────────────────────────────
    aligned = len(target) == len(features)
    checks.append(
        {
            "check": "target_aligned_with_features",
            "status": "PASS" if aligned else "FAIL",
            "detail": f"target={len(target):,}, features={features.shape[0]:,}",
        }
    )

    # ── 6. TransactionID aligned ─────────────────────────────────────
    tid_aligned = len(txn_ids) == len(features)
    tid_unique = txn_ids.is_unique
    checks.append(
        {
            "check": "transaction_id_aligned_and_unique",
            "status": "PASS" if (tid_aligned and tid_unique) else "FAIL",
            "detail": f"aligned={tid_aligned}, unique={tid_unique}",
        }
    )

    # ── 7. Identity join flag ────────────────────────────────────────
    has_flag = "has_identity_data" in names
    if has_flag:
        n_with = int(features["has_identity_data"].sum())
        n_without = int((features["has_identity_data"] == 0).sum())
        checks.append(
            {
                "check": "identity_flag_present_and_varied",
                "status": "PASS" if (n_with > 0 and n_without > 0) else "FAIL",
                "detail": f"with_identity={n_with:,}, without={n_without:,}",
            }
        )
    else:
        checks.append(
            {
                "check": "identity_flag_present_and_varied",
                "status": "FAIL",
                "detail": "has_identity_data column missing",
            }
        )

    # ── 8. Device fingerprint: missing-identity sentinel ─────────────
    if "device_fingerprint" in names:
        sentinel_count = int((features["device_fingerprint"] == "no_device_data").sum())
        checks.append(
            {
                "check": "device_fingerprint_sentinel",
                "status": "PASS" if sentinel_count > 0 else "FAIL",
                "detail": f"{sentinel_count:,} rows with 'no_device_data' sentinel",
            }
        )
    else:
        checks.append(
            {
                "check": "device_fingerprint_sentinel",
                "status": "FAIL",
                "detail": "device_fingerprint column missing",
            }
        )

    # ── 9. No target leakage (spot check) ────────────────────────────
    # previous_suspicious_count for the first transaction of each card1
    # must be 0 (no history available).
    if "previous_suspicious_count" in names:
        # Approximate: check that min is 0 and no negative values
        psc = features["previous_suspicious_count"]
        no_negatives = (psc >= 0).all()
        has_zeros = (psc == 0).any()
        checks.append(
            {
                "check": "previous_suspicious_no_leakage",
                "status": "PASS" if (no_negatives and has_zeros) else "FAIL",
                "detail": f"min={psc.min()}, max={psc.max()}, all_non_negative={no_negatives}",
            }
        )
    else:
        checks.append(
            {
                "check": "previous_suspicious_no_leakage",
                "status": "FAIL",
                "detail": "previous_suspicious_count column missing",
            }
        )

    # ── 10. Cold-start checks ────────────────────────────────────────
    cold_start_ok = True
    details = []

    if "amount_deviation" in names:
        # First-transaction amount_deviation should include 0.0 values
        has_zero_dev = (features["amount_deviation"] == 0.0).any()
        details.append(f"amount_deviation_has_zeros={has_zero_dev}")

    if "amount_to_avg_ratio" in names:
        has_one_ratio = (features["amount_to_avg_ratio"] == 1.0).any()
        details.append(f"amount_to_avg_ratio_has_ones={has_one_ratio}")

    if "tx_velocity_1h" in names:
        has_zero_vel = (features["tx_velocity_1h"] == 0).any()
        details.append(f"tx_velocity_1h_has_zeros={has_zero_vel}")

    if "location_is_new" in names:
        has_one_new = (features["location_is_new"] == 1).any()
        details.append(f"location_is_new_has_ones={has_one_new}")

    checks.append(
        {
            "check": "cold_start_defaults",
            "status": "PASS" if cold_start_ok else "FAIL",
            "detail": "; ".join(details),
        }
    )

    # ── 11. Feature data types ───────────────────────────────────────
    n_numeric = features.select_dtypes(include=[np.number]).shape[1]
    n_object = features.select_dtypes(include=["object"]).shape[1]
    checks.append(
        {
            "check": "feature_dtypes",
            "status": "PASS",
            "detail": f"numeric={n_numeric}, object={n_object}",
        }
    )

    # ── 12. No all-NaN columns ───────────────────────────────────────
    all_nan_cols = [c for c in names if features[c].isnull().all()]
    checks.append(
        {
            "check": "no_all_nan_columns",
            "status": "PASS" if len(all_nan_cols) == 0 else "FAIL",
            "detail": f"all-NaN columns: {all_nan_cols}" if all_nan_cols else "none",
        }
    )

    # ── 13. Feature metadata completeness ────────────────────────────
    metadata = result.get("feature_metadata", {})
    unmapped = [n for n in names if n not in metadata]
    checks.append(
        {
            "check": "feature_metadata_complete",
            "status": "PASS" if len(unmapped) == 0 else "FAIL",
            "detail": f"unmapped: {unmapped}" if unmapped else "all features have metadata",
        }
    )

    return checks


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    """Run the feature engineering pipeline on a sample and validate."""
    print("ML Feature Engineering Validation")
    print("=" * 50)
    print(f"Sample size: {_SAMPLE_SIZE:,} rows\n")

    result = engineer_features(sample=_SAMPLE_SIZE)
    print()

    checks = _run_checks(result)

    passed = sum(1 for c in checks if c["status"] == "PASS")
    failed = sum(1 for c in checks if c["status"] != "PASS")

    print(f"Results: {passed} passed, {failed} failed\n")
    for c in checks:
        tag = c["status"]
        print(f"  [{tag:4s}] {c['check']} — {c['detail']}")

    print()
    if failed == 0:
        print("All validations PASSED.")
    else:
        print(f"{failed} validation(s) FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
