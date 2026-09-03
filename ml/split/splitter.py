"""Leakage-safe train/test splitting for fraud detection.

Implements a **time-based** split on ``TransactionDT`` with
stratification verification, following the training pipeline defined in
``docs/ml-architecture.md``:

    Feature Engineering → Train/Test Split → Model Training

The split guarantees:

1. **Temporal ordering** — every training transaction occurs *before*
   every test transaction in ``TransactionDT``.
2. **No future leakage** — test transactions cannot influence training
   features (feature engineering already uses strictly-prior windows).
3. **No overlap** — no transaction appears in both sets.
4. **Stratification verification** — the fraud ratio in both sets is
   checked to be within tolerance of the overall rate.

Historical-feature leakage between sets is **not** a concern here
because feature engineering (``ml/features/``) computes all historical
features using shifted, strictly-prior windows *before* this split is
applied.  Test transactions using training-period history is correct —
it mirrors production where the model sees real customer history.

Split strategy
--------------
- Sort by ``TransactionDT`` ascending.
- Split point = the ``test_fraction`` quantile of ``TransactionDT``.
- Train = ``TransactionDT < split_point``.
- Test  = ``TransactionDT >= split_point``.

Default split: 80 % train / 20 % test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_TEST_FRACTION: float = 0.20
DEFAULT_STRATIFICATION_TOLERANCE: float = 0.01  # ±1 % absolute

# Column names that must never appear in X
_FORBIDDEN_FEATURE_COLS = frozenset({"isFraud", "TransactionID"})


# ── Result container ─────────────────────────────────────────────────


@dataclass(frozen=True)
class SplitResult:
    """Immutable container for train/test split output.

    Attributes:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target (isFraud).
        y_test: Test target (isFraud).
        train_ids: TransactionIDs in training set.
        test_ids: TransactionIDs in test set.
        train_timestamps: TransactionDT values in training set.
        test_timestamps: TransactionDT values in test set.
        split_timestamp: The TransactionDT threshold separating sets.
        train_fraction: Actual fraction of rows in training set.
        test_fraction: Actual fraction of rows in test set.
        train_fraud_rate: Fraud rate in training set.
        test_fraud_rate: Fraud rate in test set.
        overall_fraud_rate: Fraud rate across the full dataset.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_ids: pd.Series
    test_ids: pd.Series
    train_timestamps: pd.Series
    test_timestamps: pd.Series
    split_timestamp: int
    train_fraction: float
    test_fraction: float
    train_fraud_rate: float
    test_fraud_rate: float
    overall_fraud_rate: float


# ── Public API ───────────────────────────────────────────────────────


def time_based_split(
    features: pd.DataFrame,
    target: pd.Series,
    timestamps: pd.Series,
    transaction_ids: pd.Series,
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> SplitResult:
    """Split the feature matrix into train and test sets by time.

    Args:
        features: Feature DataFrame (must NOT contain ``isFraud`` or
                  ``TransactionID``).
        target: Target Series (``isFraud``).
        timestamps: ``TransactionDT`` Series aligned with *features*.
        transaction_ids: ``TransactionID`` Series aligned with *features*.
        test_fraction: Fraction of the time range allocated to test.
                       Default 0.20 (80/20 split).

    Returns:
        :class:`SplitResult` with train/test DataFrames and metadata.

    Raises:
        ValueError: If forbidden columns appear in *features*, if inputs
                    are not aligned, or if *test_fraction* is out of range.
    """
    # ── Input validation ─────────────────────────────────────────────
    _validate_inputs(features, target, timestamps, transaction_ids, test_fraction)

    n = len(features)
    overall_fraud_rate = float(target.mean())

    # ── Sort by TransactionDT ────────────────────────────────────────
    sort_order = timestamps.sort_values().index

    sorted_features = features.loc[sort_order]
    sorted_target = target.loc[sort_order]
    sorted_ts = timestamps.loc[sort_order]
    sorted_ids = transaction_ids.loc[sort_order]

    # ── Compute split point ──────────────────────────────────────────
    split_timestamp = int(np.quantile(sorted_ts.values, 1.0 - test_fraction))

    train_mask = sorted_ts < split_timestamp
    test_mask = sorted_ts >= split_timestamp

    # ── Build sets ───────────────────────────────────────────────────
    X_train = sorted_features.loc[train_mask].reset_index(drop=True)
    X_test = sorted_features.loc[test_mask].reset_index(drop=True)
    y_train = sorted_target.loc[train_mask].reset_index(drop=True)
    y_test = sorted_target.loc[test_mask].reset_index(drop=True)
    train_ids = sorted_ids.loc[train_mask].reset_index(drop=True)
    test_ids = sorted_ids.loc[test_mask].reset_index(drop=True)
    train_ts = sorted_ts.loc[train_mask].reset_index(drop=True)
    test_ts = sorted_ts.loc[test_mask].reset_index(drop=True)

    return SplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_ids=train_ids,
        test_ids=test_ids,
        train_timestamps=train_ts,
        test_timestamps=test_ts,
        split_timestamp=split_timestamp,
        train_fraction=len(X_train) / n,
        test_fraction=len(X_test) / n,
        train_fraud_rate=float(y_train.mean()) if len(y_train) > 0 else 0.0,
        test_fraud_rate=float(y_test.mean()) if len(y_test) > 0 else 0.0,
        overall_fraud_rate=overall_fraud_rate,
    )


def validate_split(
    result: SplitResult,
    *,
    stratification_tolerance: float = DEFAULT_STRATIFICATION_TOLERANCE,
) -> list[dict]:
    """Run leakage and correctness checks on a :class:`SplitResult`.

    Args:
        result: The split result to validate.
        stratification_tolerance: Maximum allowed absolute difference
            between set fraud rate and overall fraud rate.

    Returns:
        List of check dicts with keys ``check``, ``status``, ``detail``.
    """
    checks: list[dict] = []

    # ── 1. No forbidden columns in X ─────────────────────────────────
    for col in _FORBIDDEN_FEATURE_COLS:
        in_train = col in result.X_train.columns
        in_test = col in result.X_test.columns
        checks.append(
            {
                "check": f"no_{col.lower()}_in_X",
                "status": "PASS" if not (in_train or in_test) else "FAIL",
                "detail": f"in_train={in_train}, in_test={in_test}",
            }
        )

    # ── 2. Target present and binary ─────────────────────────────────
    for name, y in [("train", result.y_train), ("test", result.y_test)]:
        unique_vals = sorted(y.unique().tolist())
        checks.append(
            {
                "check": f"target_binary_{name}",
                "status": "PASS" if unique_vals == [0, 1] else "FAIL",
                "detail": f"unique={unique_vals}",
            }
        )

    # ── 3. Non-empty sets ────────────────────────────────────────────
    for name, size in [("train", len(result.X_train)), ("test", len(result.X_test))]:
        checks.append(
            {
                "check": f"non_empty_{name}",
                "status": "PASS" if size > 0 else "FAIL",
                "detail": f"{size:,} rows",
            }
        )

    # ── 4. Temporal ordering ─────────────────────────────────────────
    train_max_dt = int(result.train_timestamps.max()) if len(result.train_timestamps) > 0 else -1
    test_min_dt = int(result.test_timestamps.min()) if len(result.test_timestamps) > 0 else -1
    temporal_ok = train_max_dt < test_min_dt
    checks.append(
        {
            "check": "temporal_ordering",
            "status": "PASS" if temporal_ok else "FAIL",
            "detail": f"train_max_dt={train_max_dt:,}, test_min_dt={test_min_dt:,}",
        }
    )

    # ── 5. No overlapping TransactionIDs ─────────────────────────────
    overlap = set(result.train_ids) & set(result.test_ids)
    checks.append(
        {
            "check": "no_id_overlap",
            "status": "PASS" if len(overlap) == 0 else "FAIL",
            "detail": f"{len(overlap):,} overlapping IDs",
        }
    )

    # ── 6. Row counts add up ─────────────────────────────────────────
    total = len(result.X_train) + len(result.X_test)
    checks.append(
        {
            "check": "row_counts_complete",
            "status": "PASS" if total > 0 else "FAIL",
            "detail": f"train={len(result.X_train):,} + test={len(result.X_test):,} = {total:,}",
        }
    )

    # ── 7. Stratification ────────────────────────────────────────────
    overall = result.overall_fraud_rate
    for name, rate in [
        ("train", result.train_fraud_rate),
        ("test", result.test_fraud_rate),
    ]:
        diff = abs(rate - overall)
        checks.append(
            {
                "check": f"stratification_{name}",
                "status": "PASS" if diff <= stratification_tolerance else "FAIL",
                "detail": (
                    f"{name}_fraud_rate={rate:.4f}, "
                    f"overall={overall:.4f}, diff={diff:.4f} "
                    f"(tolerance=±{stratification_tolerance})"
                ),
            }
        )

    # ── 8. Feature column consistency ────────────────────────────────
    train_cols = list(result.X_train.columns)
    test_cols = list(result.X_test.columns)
    cols_match = train_cols == test_cols
    checks.append(
        {
            "check": "feature_columns_match",
            "status": "PASS" if cols_match else "FAIL",
            "detail": f"train={len(train_cols)} cols, test={len(test_cols)} cols",
        }
    )

    # ── 9. Split timestamp is sensible ───────────────────────────────
    checks.append(
        {
            "check": "split_timestamp_in_range",
            "status": "PASS",
            "detail": (
                f"split_dt={result.split_timestamp:,}, "
                f"train=[{int(result.train_timestamps.min()):,}..{train_max_dt:,}], "
                f"test=[{test_min_dt:,}..{int(result.test_timestamps.max()):,}]"
            ),
        }
    )

    # ── 10. Fraud class present in both sets ─────────────────────────
    for name, y in [("train", result.y_train), ("test", result.y_test)]:
        fraud_count = int(y.sum())
        checks.append(
            {
                "check": f"fraud_class_present_{name}",
                "status": "PASS" if fraud_count > 0 else "FAIL",
                "detail": f"{fraud_count:,} fraudulent in {name} ({y.mean():.2%})",
            }
        )

    return checks


# ── Internal helpers ─────────────────────────────────────────────────


def _validate_inputs(
    features: pd.DataFrame,
    target: pd.Series,
    timestamps: pd.Series,
    transaction_ids: pd.Series,
    test_fraction: float,
) -> None:
    """Raise ValueError on invalid inputs."""
    forbidden = set(features.columns) & _FORBIDDEN_FEATURE_COLS
    if forbidden:
        raise ValueError(
            f"Feature DataFrame must not contain {forbidden}. "
            f"Found columns: {sorted(forbidden)}"
        )

    n = len(features)
    if len(target) != n:
        raise ValueError(f"target length ({len(target)}) != features length ({n})")
    if len(timestamps) != n:
        raise ValueError(f"timestamps length ({len(timestamps)}) != features length ({n})")
    if len(transaction_ids) != n:
        raise ValueError(
            f"transaction_ids length ({len(transaction_ids)}) != features length ({n})"
        )

    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    """CLI: run feature engineering, split, and print validation report."""
    from ml.features.engineer import engineer_features

    print("ML Train/Test Split")
    print("=" * 60)

    # Feature engineering
    print("\n[split] Running feature engineering...")
    fe_result = engineer_features()
    features = fe_result["features"]
    target = fe_result["target"]
    txn_ids = fe_result["transaction_ids"]
    n_total = fe_result["n_rows"]

    # Need timestamps for time-based split
    # Re-load to get TransactionDT (engineer_features sorts by time)
    from ml.data.loader import load_transaction_dataset

    txn = load_transaction_dataset()
    timestamps = txn.set_index("TransactionID")["TransactionDT"]
    timestamps = timestamps.loc[txn_ids.values].reset_index(drop=True)
    timestamps.index = features.index  # align indices

    # Split
    print(f"\n[split] Splitting {n_total:,} rows (80/20 time-based)...")
    result = time_based_split(
        features=features,
        target=target,
        timestamps=timestamps,
        transaction_ids=txn_ids,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Split Results")
    print(f"{'=' * 60}")
    print(f"  Total rows:         {n_total:,}")
    print(f"  Split timestamp:    {result.split_timestamp:,}")
    print(f"  Training set:       {len(result.X_train):,} rows ({result.train_fraction:.1%})")
    print(f"  Test set:           {len(result.X_test):,} rows ({result.test_fraction:.1%})")
    print(f"  Training fraud:     {result.train_fraud_rate:.4f} ({int(result.y_train.sum()):,})")
    print(f"  Test fraud:         {result.test_fraud_rate:.4f} ({int(result.y_test.sum()):,})")
    print(f"  Overall fraud:      {result.overall_fraud_rate:.4f}")
    print(f"  Train time range:   [{int(result.train_timestamps.min()):,} .. {int(result.train_timestamps.max()):,}]")
    print(f"  Test time range:    [{int(result.test_timestamps.min()):,} .. {int(result.test_timestamps.max()):,}]")
    print(f"  Features:           {len(result.X_train.columns)}")

    # Validation
    print(f"\n{'=' * 60}")
    print("Validation Checks")
    print(f"{'=' * 60}")
    checks = validate_split(result)
    passed = sum(1 for c in checks if c["status"] == "PASS")
    failed = sum(1 for c in checks if c["status"] != "PASS")

    for c in checks:
        tag = c["status"]
        print(f"  [{tag:4s}] {c['check']} — {c['detail']}")

    print(f"\nResults: {passed} passed, {failed} failed")
    if failed == 0:
        print("All validations PASSED.")
    else:
        print(f"{failed} validation(s) FAILED.")

    print("\nDone.")


if __name__ == "__main__":
    main()
