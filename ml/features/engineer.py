"""ML Feature Engineering Pipeline.

Orchestrates loading, joining, and feature computation for the IEEE-CIS
Fraud Detection dataset.  Produces a feature matrix ready for model
training, following the specification in
``docs/ml-feature-engineering-spec.md``.

Usage (CLI)::

    python -m ml.features.engineer

Usage (import)::

    from ml.features.engineer import engineer_features

    result = engineer_features()
    features_df = result["features"]
    target = result["target"]
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ml.data.loader import load_transaction_dataset, load_identity_dataset
from ml.features.historical import compute_all_historical_features
from ml.features.identity import (
    NO_DEVICE_FINGERPRINT,
    compute_is_new_device,
    construct_device_fingerprint,
)

# ── Feature list ─────────────────────────────────────────────────────
# 18 features from spec § 4 (transaction_type excluded — Unavailable).
# 3 supplementary features from spec § 9 Decisions #4, #6.

FEATURE_LIST: list[str] = [
    "amount",                         # Direct
    "amount_deviation",               # Derived
    "amount_to_avg_ratio",            # Derived
    "location_country",               # Partial — proxy: addr2
    "location_region",                # Supplementary — addr1
    "location_is_new",                # Derived
    "location_change",                # Derived
    "device_fingerprint",             # Partial — identity proxy
    "is_new_device",                  # Derived
    "hour_of_day_raw",                # Derived (raw integer)
    "hour_of_day_sin",               # Derived (cyclical)
    "hour_of_day_cos",               # Derived (cyclical)
    "day_of_week_raw",                # Derived (raw integer)
    "day_of_week_sin",               # Derived (cyclical)
    "day_of_week_cos",               # Derived (cyclical)
    "is_unusual_hour",               # Derived
    "tx_velocity_1h",                # Derived
    "tx_velocity_24h",               # Derived
    "tx_velocity_7d",                # Derived
    "merchant_category",              # Partial — proxy: ProductCD
    "merchant_is_new",               # Derived
    "avg_spend_30d",                 # Derived
    "previous_suspicious_count",      # Partial — prior isFraud
    "has_identity_data",              # Supplementary flag
]


# ── Columns required from raw datasets ───────────────────────────────

_TRANSACTION_COLS: list[str] = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "addr1",
    "addr2",
]

_IDENTITY_COLS: list[str] = [
    "TransactionID",
    "id_19",
    "id_20",
    "DeviceType",
]


# ── Data loading & join ─────────────────────────────────────────────


def _load_and_join() -> pd.DataFrame:
    """Load datasets, left-join identity onto transaction, and sort.

    Only the columns needed for feature engineering are selected,
    reducing memory from ~2 GB (full join) to ~50 MB.

    Returns:
        Joined DataFrame sorted by ``TransactionDT`` (ascending).
        Original row order is **not** preserved — callers should use
        ``TransactionID`` to re-join downstream results.
    """
    txn = load_transaction_dataset()
    ident = load_identity_dataset()

    # Select only required columns
    txn = txn[_TRANSACTION_COLS].copy()
    ident = ident[_IDENTITY_COLS].copy()

    # Left join on TransactionID
    df = txn.merge(ident, on="TransactionID", how="left")

    # Flag identity availability (spec Decision #4)
    df["has_identity_data"] = df["id_19"].notna().astype(np.int8)

    # Sort by time for historical computations
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    return df


# ── Direct / simple features ────────────────────────────────────────


def _compute_direct_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features that require only the current row.

    Returns:
        DataFrame with ``amount``, ``location_country``,
        ``location_region``, ``merchant_category``, cyclical time
        features, and ``has_identity_data``.
    """
    features = pd.DataFrame(index=df.index)

    # amount — Direct pass-through
    features["amount"] = df["TransactionAmt"]

    # location_country — Partial proxy: addr2 (integer)
    features["location_country"] = df["addr2"].fillna(-1).astype(np.int64)

    # location_region — Supplementary: addr1 (integer)
    features["location_region"] = df["addr1"].fillna(-1).astype(np.int64)

    # merchant_category — Partial proxy: ProductCD (label-encoded)
    features["merchant_category"] = df["ProductCD"].astype("category").cat.codes.astype(
        np.int8
    )

    # hour_of_day — cyclical encoding
    hour = ((df["TransactionDT"] % 86_400) // 3_600).astype(np.int32)
    features["hour_of_day_raw"] = hour
    features["hour_of_day_sin"] = np.sin(2.0 * math.pi * hour / 24.0)
    features["hour_of_day_cos"] = np.cos(2.0 * math.pi * hour / 24.0)

    # day_of_week — cyclical encoding
    day = ((df["TransactionDT"] // 86_400) % 7).astype(np.int32)
    features["day_of_week_raw"] = day
    features["day_of_week_sin"] = np.sin(2.0 * math.pi * day / 7.0)
    features["day_of_week_cos"] = np.cos(2.0 * math.pi * day / 7.0)

    # has_identity_data — supplementary flag
    features["has_identity_data"] = df["has_identity_data"]

    return features


# ── Main pipeline ────────────────────────────────────────────────────


def engineer_features(
    *,
    sample: int | None = None,
) -> dict[str, Any]:
    """Run the full feature-engineering pipeline.

    Args:
        sample: If given, randomly sample this many rows after joining
                (for quick validation runs).

    Returns:
        Dict with keys:

        ``"features"``
            DataFrame with :data:`FEATURE_LIST` columns (no target).
        ``"target"``
            Series ``isFraud`` aligned to ``features.index``.
        ``"transaction_ids"``
            Series ``TransactionID`` aligned to ``features.index``.
        ``"feature_names"``
            List of feature column names.
        ``"n_rows"``
            Number of rows in the output.
        ``"feature_metadata"``
            Dict mapping feature name → availability status
            (Direct / Derived / Partial / Unavailable / Supplementary).
    """
    print("[engineer] Loading and joining datasets...")
    df = _load_and_join()
    print(f"[engineer] Joined shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=42).sort_values(
            "TransactionDT"
        ).reset_index(drop=True)
        print(f"[engineer] Sampled {sample:,} rows")

    # Direct features
    print("[engineer] Computing direct features...")
    direct = _compute_direct_features(df)

    # Identity features
    print("[engineer] Computing identity features...")
    device_fp = construct_device_fingerprint(df)
    is_new_dev = compute_is_new_device(df, device_fp)

    identity = pd.DataFrame(
        {
            "device_fingerprint": device_fp,
            "is_new_device": is_new_dev,
        }
    )

    # Historical features
    print("[engineer] Computing historical features (this may take a while)...")
    historical = compute_all_historical_features(df)

    # Assemble output
    print("[engineer] Assembling output...")
    features = pd.concat([direct, identity, historical], axis=1)

    # Metadata
    metadata: dict[str, str] = {
        "amount": "Direct",
        "amount_deviation": "Derived",
        "amount_to_avg_ratio": "Derived",
        "location_country": "Partial",
        "location_region": "Supplementary",
        "location_is_new": "Derived",
        "location_change": "Derived",
        "device_fingerprint": "Partial",
        "is_new_device": "Derived",
        "hour_of_day_raw": "Derived",
        "hour_of_day_sin": "Derived",
        "hour_of_day_cos": "Derived",
        "day_of_week_raw": "Derived",
        "day_of_week_sin": "Derived",
        "day_of_week_cos": "Derived",
        "is_unusual_hour": "Derived",
        "tx_velocity_1h": "Derived",
        "tx_velocity_24h": "Derived",
        "tx_velocity_7d": "Derived",
        "merchant_category": "Partial",
        "merchant_is_new": "Derived",
        "avg_spend_30d": "Derived",
        "previous_suspicious_count": "Partial",
        "has_identity_data": "Supplementary",
    }

    return {
        "features": features,
        "target": df["isFraud"],
        "transaction_ids": df["TransactionID"],
        "feature_names": list(features.columns),
        "n_rows": len(features),
        "feature_metadata": metadata,
    }


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry-point: run feature engineering and print summary."""
    print("ML Feature Engineering Pipeline")
    print("=" * 50)

    result = engineer_features()

    features = result["features"]
    target = result["target"]
    n_rows = result["n_rows"]
    names = result["feature_names"]

    print(f"\nOutput: {n_rows:,} rows x {len(names)} features")
    print(f"\nFeature columns ({len(names)}):")
    for name in names:
        status = result["feature_metadata"].get(name, "unknown")
        print(f"  [{status:12s}] {name}")

    print(f"\nTarget distribution:")
    print(f"  isFraud=0: {(target == 0).sum():,} ({(target == 0).mean():.1%})")
    print(f"  isFraud=1: {(target == 1).sum():,} ({(target == 1).mean():.1%})")

    print(f"\nFeature null counts:")
    null_counts = features.isnull().sum()
    if null_counts.any():
        for col in null_counts[null_counts > 0].index:
            print(f"  {col}: {null_counts[col]:,} ({null_counts[col] / n_rows:.1%})")
    else:
        print("  None — all features fully populated")

    print(f"\nFeature dtypes:")
    dtype_counts = features.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count}")

    print("\nDone.")


if __name__ == "__main__":
    main()
