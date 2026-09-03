"""Identity-based feature computations.

Constructs the composite device fingerprint from identity columns and
derives the ``is_new_device`` flag.

Identity data covers only ~24.42 % of transactions.  All identity-derived
features must gracefully handle the 75.58 % missing-identity case using
defined sentinel / default values.

Decision reference: ``docs/ml-feature-engineering-spec.md`` § 9 #2, #3, #4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sentinel fingerprint value for transactions without identity data.
NO_DEVICE_FINGERPRINT = "no_device_data"


# ── Device fingerprint ───────────────────────────────────────────────


def construct_device_fingerprint(df: pd.DataFrame) -> pd.Series:
    """Build a composite device fingerprint from identity columns.

    Concatenates ``id_19``, ``id_20``, and ``DeviceType`` as strings and
    joins them with ``|`` separators.  Rows where **all three** columns
    are ``NaN`` receive the sentinel value ``no_device_data``.

    Spec reference: § 9 Decision #2 — hash of three columns.

    Returns:
        Series named ``device_fingerprint`` (object / string).
    """
    id_19 = df["id_19"].fillna("").astype(str)
    id_20 = df["id_20"].fillna("").astype(str)
    device_type = df["DeviceType"].fillna("").astype(str)

    fingerprint = id_19.str.cat(id_20, sep="|").str.cat(device_type, sep="|")

    # All-missing check: every component was originally NaN
    all_missing = df["id_19"].isna() & df["id_20"].isna() & df["DeviceType"].isna()
    fingerprint = fingerprint.where(~all_missing, NO_DEVICE_FINGERPRINT)

    return fingerprint.rename("device_fingerprint")


# ── Is new device ────────────────────────────────────────────────────


def compute_is_new_device(
    df: pd.DataFrame,
    device_fingerprints: pd.Series,
) -> pd.Series:
    """Flag whether the device fingerprint is new for this ``card1``.

    For transactions without identity data the fingerprint equals
    ``no_device_data``.  Per spec Decision #3 these receive a default
    of ``0`` (absence of device data ≠ new device).

    For transactions **with** identity data the flag is 1 if the
    fingerprint has not appeared in any prior transaction for the same
    ``card1``, 0 otherwise.

    Returns:
        Series named ``is_new_device`` (int8).
    """
    is_new = np.ones(len(df), dtype=np.int8)

    # Pre-mark no-identity rows as 0 (spec Decision #3)
    no_identity_mask = device_fingerprints == NO_DEVICE_FINGERPRINT
    is_new[no_identity_mask.values] = 0

    grouped = df.groupby("card1")
    for _, group in grouped:
        sorted_g = group.sort_values("TransactionDT")
        seen_fps: set[str] = set()

        for i, (idx_i, _) in enumerate(sorted_g.iterrows()):
            fp = device_fingerprints.loc[idx_i]
            if fp == NO_DEVICE_FINGERPRINT:
                is_new[idx_i] = 0
            elif i == 0:
                is_new[idx_i] = 1  # cold-start: everything is new
            else:
                is_new[idx_i] = 0 if fp in seen_fps else 1
            if fp != NO_DEVICE_FINGERPRINT:
                seen_fps.add(fp)

    return pd.Series(is_new, index=df.index, name="is_new_device", dtype=np.int8)
