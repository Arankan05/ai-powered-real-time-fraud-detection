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


# ── Single-transaction inference ────────────────────────────────────


def _resolve_customer_id(raw: dict[str, Any]) -> str:
    """Determine the customer identifier from a raw transaction dict.

    Priority:
      1. Explicit ``customer_id`` field.
      2. ``device_fingerprint`` (always present from the backend).
      3. Fallback ``"unknown"``.
    """
    cid = raw.get("customer_id")
    if cid is not None and str(cid).strip():
        return str(cid)
    fp = raw.get("device_fingerprint")
    if fp is not None and str(fp).strip():
        return str(fp)
    return "unknown"


def engineer_features_for_inference(
    raw: dict[str, Any],
    *,
    history_store: Any | None = None,
) -> pd.DataFrame:
    """Compute the 24 engineered features for a **single** raw transaction.

    When *history_store* is provided and contains prior transactions
    for the same customer, historical features are computed from real
    history.  Otherwise cold-start defaults are used (as documented in
    :mod:`ml.features.historical`, lines 8-19).

    ``previous_suspicious_count`` uses the ``is_fraud`` field stored
    in the history records (``0`` at prediction time).  No external
    label data is accessed.

    Args:
        raw: Dict with keys matching the raw transaction payload
             sent by the backend (``TransactionCreate.model_dump()``).
             Optional: ``customer_id``, ``timestamp``, ``card1``,
             ``addr1``, ``addr2``, ``id_19``, ``id_20``, ``DeviceType``,
             ``has_identity_data``.
        history_store: Optional
             :class:`~ml.features.history.TransactionHistoryStore`
             instance for customer-history lookup.

    Returns:
        Single-row :class:`pandas.DataFrame` with columns matching
        :data:`FEATURE_LIST` in the correct order.
    """
    # -- defaults for optional raw fields --------------------------------
    def _int_or(key: str, default: int) -> int:
        v = raw.get(key)
        return default if v is None else int(v)

    def _str_or(key: str, default: str | None) -> str | None:
        v = raw.get(key)
        return default if v is None else str(v)

    timestamp = _int_or("timestamp", 0)
    addr1 = _int_or("addr1", -1)
    addr2 = _int_or("addr2", -1)
    id_19 = _str_or("id_19", None)
    id_20 = _str_or("id_20", None)
    device_type = _str_or("DeviceType", None)
    product_cd = _str_or("ProductCD", None) or _str_or("merchant_category", "W")
    has_identity = _int_or("has_identity_data", 0)
    amount = float(raw["amount"])

    # -- customer identification & card1 ---------------------------------
    customer_id = _resolve_customer_id(raw)
    # card1 must be consistent per customer for groupby.
    # Use explicit card1 if provided, else hash of customer_id.
    raw_card1 = raw.get("card1")
    card1 = int(raw_card1) if raw_card1 is not None else hash(customer_id) & 0x7FFFFFFF

    # -- history lookup --------------------------------------------------
    history = []
    if history_store is not None:
        history = history_store.get(
            customer_id, before_timestamp=timestamp
        )

    # -- build DataFrame rows (history + current) -----------------------
    def _h_int(record: dict, key: str, default: int) -> int:
        v = record.get(key)
        return default if v is None else int(v)

    def _h_float(record: dict, key: str, default: float) -> float:
        v = record.get(key)
        return default if v is None else float(v)

    def _h_str(record: dict, key: str, default: str | None) -> str | None:
        v = record.get(key)
        return default if v is None else str(v)

    rows: list[dict[str, Any]] = []
    for h in history:
        rows.append(
            {
                "TransactionID": 0,
                "isFraud": _h_int(h, "is_fraud", 0),
                "TransactionDT": _h_int(h, "timestamp", 0),
                "TransactionAmt": _h_float(h, "amount", 0.0),
                "ProductCD": _h_str(h, "product_cd", "W") or "W",
                "card1": card1,
                "addr1": _h_int(h, "addr1", -1),
                "addr2": _h_int(h, "addr2", -1),
                "id_19": _h_str(h, "id_19", None),
                "id_20": _h_str(h, "id_20", None),
                "DeviceType": _h_str(h, "device_type", None),
                "has_identity_data": np.int8(_h_int(h, "has_identity_data", 0)),
            }
        )

    # Current transaction (always last after sort)
    rows.append(
        {
            "TransactionID": 0,
            "isFraud": 0,  # placeholder — never used as a feature
            "TransactionDT": timestamp,
            "TransactionAmt": amount,
            "ProductCD": product_cd,
            "card1": card1,
            "addr1": addr1,
            "addr2": addr2,
            "id_19": id_19,
            "id_20": id_20,
            "DeviceType": device_type,
            "has_identity_data": np.int8(has_identity),
        }
    )

    df = pd.DataFrame(rows)
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)  # total rows (history + 1)

    # -- direct features -------------------------------------------------
    direct = _compute_direct_features(df)

    # -- identity features -----------------------------------------------
    device_fp = construct_device_fingerprint(df)
    is_new_dev = compute_is_new_device(df, device_fp)
    identity = pd.DataFrame(
        {"device_fingerprint": device_fp, "is_new_device": is_new_dev}
    )

    # -- historical features (real history or cold-start) ----------------
    historical = compute_all_historical_features(df)

    # -- assemble & extract current transaction (last row) ---------------
    features = pd.concat([direct, identity, historical], axis=1)
    current = features.iloc[[-1]].reset_index(drop=True)

    # -- guarantee FEATURE_LIST columns ----------------------------------
    missing = set(FEATURE_LIST) - set(current.columns)
    if missing:
        raise ValueError(
            f"Feature engineering missing columns: {sorted(missing)}"
        )

    return current[FEATURE_LIST]


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
