"""
Dataset loading and validation for the IEEE-CIS Fraud Detection dataset.

Provides safe functions to load raw transaction and identity CSV files
and validate that they meet the requirements of the ML fraud-detection
pipeline before any preprocessing or feature engineering begins.

Usage (CLI)::

    python -m ml.data.loader

Usage (import)::

    from ml.data.loader import load_datasets, validate_datasets

    txn, ident = load_datasets()
    results = validate_datasets(txn, ident)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _PROJECT_ROOT / "ml" / "datasets" / "raw"

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

# ── Required column definitions ────────────────────────────────────────
TRANSACTION_REQUIRED_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "addr1",
    "addr2",
]

IDENTITY_REQUIRED_COLUMNS = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo",
]


# ── Public API ─────────────────────────────────────────────────────────


def get_raw_data_path() -> Path:
    """Return the absolute path to the raw dataset directory.

    Returns:
        Path to ``ml/datasets/raw/`` resolved from the project root.
    """
    return _RAW_DIR


def load_transaction_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the raw transaction CSV into a DataFrame.

    Args:
        path: Override path to the transaction CSV file.
              Defaults to ``ml/datasets/raw/train_transaction.csv``.

    Returns:
        Transaction DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = path or (_RAW_DIR / TRANSACTION_FILE)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found: {file_path}\n"
            f"Expected location: {_RAW_DIR / TRANSACTION_FILE}"
        )
    return pd.read_csv(file_path)


def load_identity_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the raw identity CSV into a DataFrame.

    Args:
        path: Override path to the identity CSV file.
              Defaults to ``ml/datasets/raw/train_identity.csv``.

    Returns:
        Identity DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = path or (_RAW_DIR / IDENTITY_FILE)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Identity dataset not found: {file_path}\n"
            f"Expected location: {_RAW_DIR / IDENTITY_FILE}"
        )
    return pd.read_csv(file_path)


def load_datasets(
    txn_path: Path | None = None,
    ident_path: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both raw datasets.

    Convenience wrapper that calls :func:`load_transaction_dataset` and
    :func:`load_identity_dataset`.

    Returns:
        Tuple of (transaction_df, identity_df).
    """
    txn = load_transaction_dataset(txn_path)
    ident = load_identity_dataset(ident_path)
    return txn, ident


# ── Validation functions ──────────────────────────────────────────────


def validate_columns(
    df: pd.DataFrame,
    required: list[str],
    dataset_name: str,
) -> list[dict]:
    """Check that all required columns exist in the DataFrame.

    Returns:
        List of result dicts with keys *check*, *column*, *status*,
        and *detail*.
    """
    results: list[dict] = []
    for col in required:
        present = col in df.columns
        dtype = str(df[col].dtype) if present else "N/A"
        results.append(
            {
                "check": "column_exists",
                "column": col,
                "dataset": dataset_name,
                "status": "pass" if present else "FAIL",
                "detail": f"dtype={dtype}" if present else "missing",
            }
        )
    return results


def validate_target(df: pd.DataFrame) -> list[dict]:
    """Validate the ``isFraud`` target column.

    Checks:
    - Column exists.
    - Contains only binary values (0 and 1).

    Returns:
        List of result dicts.
    """
    results: list[dict] = []

    exists = "isFraud" in df.columns
    results.append(
        {
            "check": "target_exists",
            "status": "pass" if exists else "FAIL",
            "detail": "isFraud column present" if exists else "isFraud column missing",
        }
    )

    if exists:
        unique_vals = sorted(df["isFraud"].dropna().unique().tolist())
        is_binary = unique_vals == [0, 1]
        results.append(
            {
                "check": "target_binary",
                "status": "pass" if is_binary else "FAIL",
                "detail": f"unique values={unique_vals}",
            }
        )

    return results


def validate_transaction_ids(
    txn: pd.DataFrame,
    ident: pd.DataFrame,
) -> list[dict]:
    """Validate TransactionID in both datasets and check join feasibility.

    Checks:
    - TransactionID exists in both datasets.
    - TransactionID is unique within each dataset.
    - Identity IDs are a subset of transaction IDs (overlap).

    Returns:
        List of result dicts.
    """
    results: list[dict] = []

    # Existence
    txn_has = "TransactionID" in txn.columns
    ident_has = "TransactionID" in ident.columns
    results.append(
        {
            "check": "transaction_id_exists",
            "dataset": "transaction",
            "status": "pass" if txn_has else "FAIL",
        }
    )
    results.append(
        {
            "check": "transaction_id_exists",
            "dataset": "identity",
            "status": "pass" if ident_has else "FAIL",
        }
    )

    if not (txn_has and ident_has):
        return results

    # Uniqueness
    txn_unique = txn["TransactionID"].is_unique
    ident_unique = ident["TransactionID"].is_unique
    results.append(
        {
            "check": "transaction_id_unique",
            "dataset": "transaction",
            "status": "pass" if txn_unique else "FAIL",
            "detail": f"{txn['TransactionID'].nunique():,} unique of {len(txn):,} rows",
        }
    )
    results.append(
        {
            "check": "transaction_id_unique",
            "dataset": "identity",
            "status": "pass" if ident_unique else "FAIL",
            "detail": f"{ident['TransactionID'].nunique():,} unique of {len(ident):,} rows",
        }
    )

    # Overlap
    txn_ids = set(txn["TransactionID"])
    ident_ids = set(ident["TransactionID"])
    overlap = len(txn_ids & ident_ids)
    orphan = len(ident_ids - txn_ids)
    results.append(
        {
            "check": "identity_subset_of_transaction",
            "status": "pass" if orphan == 0 else "FAIL",
            "detail": f"{overlap:,} overlapping, {orphan} orphan identity IDs",
        }
    )

    return results


def validate_datasets(
    txn: pd.DataFrame,
    ident: pd.DataFrame,
) -> list[dict]:
    """Run the full validation suite on both datasets.

    Combines :func:`validate_columns`, :func:`validate_target`, and
    :func:`validate_transaction_ids` into a single call.

    Returns:
        List of all validation result dicts.
    """
    results: list[dict] = []
    results.extend(validate_columns(txn, TRANSACTION_REQUIRED_COLUMNS, "transaction"))
    results.extend(validate_columns(ident, IDENTITY_REQUIRED_COLUMNS, "identity"))
    results.extend(validate_target(txn))
    results.extend(validate_transaction_ids(txn, ident))
    return results


# ── CLI entry-point ───────────────────────────────────────────────────


def _print_results(results: list[dict]) -> None:
    """Pretty-print validation results to stdout."""
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] != "pass")

    print(f"Validation results: {passed} passed, {failed} failed\n")
    for r in results:
        tag = "PASS" if r["status"] == "pass" else "FAIL"
        ds = r.get("dataset", "")
        col = r.get("column", "")
        detail = r.get("detail", "")

        parts = [f"[{tag}]"]
        if ds:
            parts.append(f"({ds})")
        parts.append(r["check"])
        if col:
            parts.append(f"— {col}")
        if detail:
            parts.append(f"— {detail}")
        print("  ".join(parts))


def main() -> None:
    """CLI entry-point: load datasets and run validation."""
    print(f"Raw data directory: {_RAW_DIR}")
    print(f"Transaction file:   {_RAW_DIR / TRANSACTION_FILE}")
    print(f"Identity file:      {_RAW_DIR / IDENTITY_FILE}")
    print()

    txn, ident = load_datasets()
    print(f"Transaction dataset: {txn.shape[0]:,} rows x {txn.shape[1]} cols")
    print(f"Identity dataset:    {ident.shape[0]:,} rows x {ident.shape[1]} cols")
    print()

    results = validate_datasets(txn, ident)
    _print_results(results)

    all_passed = all(r["status"] == "pass" for r in results)
    print()
    if all_passed:
        print("All validations PASSED.")
    else:
        print("Some validations FAILED.")


if __name__ == "__main__":
    main()
