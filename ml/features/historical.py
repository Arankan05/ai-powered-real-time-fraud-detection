"""Historical (rolling / time-window) feature computations.

All functions in this module operate on data sorted by ``TransactionDT``
within each ``card1`` group.  Every calculation uses **only prior**
transactions to prevent future-data leakage.

Cold-start defaults (first transaction per ``card1``):

=======================  =========  ================================
Feature type             Default    Rationale
=======================  =========  ================================
Rolling statistics       0.0        No deviation from non-existent baseline
Ratios                   1.0        Amount equals its own value
"Is new" flags           1          Everything is new on first tx
"Change" flags           0          No previous state
Velocity counts          0          No prior transactions
``avg_spend_30d``        own amount Single data point as its own average
``is_unusual_hour``      0          No baseline yet
``previous_suspicious``  0          No prior history
=======================  =========  ================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Time-window constants (seconds) ─────────────────────────────────
_ONE_HOUR = 3_600
_ONE_DAY = 86_400
_SEVEN_DAYS = 604_800
_THIRTY_DAYS = 2_592_000


# ── Expanding-window amount features ────────────────────────────────


def compute_amount_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute ``amount_deviation`` and ``amount_to_avg_ratio``.

    Uses an expanding (cumulative) mean / std of ``TransactionAmt``
    grouped by ``card1``, shifted by one row so the current transaction
    is **excluded** from its own statistics.

    Returns:
        DataFrame with columns ``amount_deviation``, ``amount_to_avg_ratio``.
    """
    grouped = df.groupby("card1")["TransactionAmt"]

    expanding_mean = grouped.transform(
        lambda x: x.expanding(min_periods=1).mean().shift(1)
    )
    expanding_std = grouped.transform(
        lambda x: x.expanding(min_periods=2).std().shift(1)
    )

    amounts = df["TransactionAmt"]

    # amount_deviation = z-score against prior history
    deviation = (amounts - expanding_mean) / expanding_std.replace(0, np.nan)
    deviation = deviation.fillna(0.0)

    # amount_to_avg_ratio = current amount / prior mean
    ratio = amounts / expanding_mean.replace(0, np.nan)
    ratio = ratio.fillna(1.0)

    return pd.DataFrame(
        {
            "amount_deviation": deviation.values,
            "amount_to_avg_ratio": ratio.values,
        },
        index=df.index,
    )


# ── Velocity features ───────────────────────────────────────────────


def compute_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute transaction-velocity counts in fixed time windows.

    For each transaction, counts **strictly prior** transactions for the
    same ``card1`` falling within three windows:

    * ``tx_velocity_1h``  — prior 1 hour  (3 600 s)
    * ``tx_velocity_24h`` — prior 24 hours (86 400 s)
    * ``tx_velocity_7d``  — prior 7 days  (604 800 s)

    Uses ``np.searchsorted`` for O(log n) per-group efficiency.

    Returns:
        DataFrame with three ``tx_velocity_*`` columns.
    """
    result = pd.DataFrame(
        {
            "tx_velocity_1h": np.zeros(len(df), dtype=np.int32),
            "tx_velocity_24h": np.zeros(len(df), dtype=np.int32),
            "tx_velocity_7d": np.zeros(len(df), dtype=np.int32),
        },
        index=df.index,
    )

    grouped = df.groupby("card1")
    for _, group in grouped:
        if len(group) < 2:
            continue  # cold-start: all zeros

        sorted_g = group.sort_values("TransactionDT")
        dts = sorted_g["TransactionDT"].values
        idx = sorted_g.index

        for window, col in [
            (_ONE_HOUR, "tx_velocity_1h"),
            (_ONE_DAY, "tx_velocity_24h"),
            (_SEVEN_DAYS, "tx_velocity_7d"),
        ]:
            starts = dts - window
            positions = np.searchsorted(dts, starts, side="left")
            counts = np.arange(len(dts), dtype=np.int32) - positions.astype(
                np.int32
            )
            # Exclude the current transaction itself
            counts = np.maximum(counts - 1, 0)
            result.loc[idx, col] = counts

    return result


# ── Average spend (30-day window) ───────────────────────────────────


def compute_avg_spend_30d(df: pd.DataFrame) -> pd.Series:
    """Compute rolling 30-day average spend per ``card1``.

    Window: ``[current_DT - 2_592_000, current_DT)`` — strictly prior.
    Cold-start default: the current ``TransactionAmt``.

    Returns:
        Series named ``avg_spend_30d``.
    """
    result = pd.Series(np.nan, index=df.index, dtype=np.float64, name="avg_spend_30d")

    grouped = df.groupby("card1")
    for _, group in grouped:
        if len(group) == 1:
            result.loc[group.index[0]] = group["TransactionAmt"].iloc[0]
            continue

        sorted_g = group.sort_values("TransactionDT")
        dts = sorted_g["TransactionDT"].values
        amounts = sorted_g["TransactionAmt"].values
        idx = sorted_g.index

        starts = dts - _THIRTY_DAYS
        positions = np.searchsorted(dts, starts, side="left")

        values = np.empty(len(dts), dtype=np.float64)
        for i in range(len(dts)):
            start_pos = int(positions[i])
            if start_pos >= i:
                values[i] = amounts[i]  # cold-start
            else:
                window_amounts = amounts[start_pos:i]
                values[i] = window_amounts.mean() if len(window_amounts) > 0 else amounts[i]

        result.loc[idx] = values

    return result


# ── Set-membership "is new" flags ───────────────────────────────────


def compute_is_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ``location_is_new`` and ``merchant_is_new``.

    For each transaction, checks whether the value (``addr2`` /
    ``ProductCD``) has **not** appeared in any prior transaction for the
    same ``card1``.

    Returns:
        DataFrame with ``location_is_new``, ``merchant_is_new`` (int8).
    """
    location_is_new = np.ones(len(df), dtype=np.int8)
    merchant_is_new = np.ones(len(df), dtype=np.int8)

    grouped = df.groupby("card1")
    for _, group in grouped:
        sorted_g = group.sort_values("TransactionDT")

        seen_locs: set = set()
        seen_prods: set = set()

        for i, (idx_i, row) in enumerate(sorted_g.iterrows()):
            if i > 0:
                location_is_new[idx_i] = 0 if row["addr2"] in seen_locs else 1
                merchant_is_new[idx_i] = 0 if row["ProductCD"] in seen_prods else 1
            seen_locs.add(row["addr2"])
            seen_prods.add(row["ProductCD"])

    return pd.DataFrame(
        {
            "location_is_new": location_is_new,
            "merchant_is_new": merchant_is_new,
        },
        index=df.index,
    )


# ── Location change ─────────────────────────────────────────────────


def compute_location_change(df: pd.DataFrame) -> pd.Series:
    """Compute ``location_change`` flag.

    Flag = 1 if current ``addr2`` differs from the most recent prior
    ``addr2`` for the same ``card1``; 0 otherwise (including cold-start).

    Returns:
        Series named ``location_change`` (int8).
    """
    prev_addr2 = df.groupby("card1")["addr2"].shift(1)
    changed = (df["addr2"] != prev_addr2) & prev_addr2.notna()
    return changed.astype(np.int8).rename("location_change")


# ── Unusual hour ─────────────────────────────────────────────────────

_UNUSUAL_HOUR_THRESHOLD = 0.10


def compute_is_unusual_hour(df: pd.DataFrame) -> pd.Series:
    """Compute ``is_unusual_hour`` flag.

    An hour is *unusual* if, based on all prior transactions for the
    same ``card1``, it either has never been seen or accounts for less
    than 10 % of prior transactions.

    Returns:
        Series named ``is_unusual_hour`` (int8).
    """
    hours = ((df["TransactionDT"] % 86_400) // 3_600).astype(np.int32)
    is_unusual = np.zeros(len(df), dtype=np.int8)

    grouped = df.groupby("card1")
    for _, group in grouped:
        sorted_g = group.sort_values("TransactionDT")
        idx = sorted_g.index.values
        group_hours = hours.loc[sorted_g.index].values

        hour_counts = np.zeros(24, dtype=np.float64)
        total = 0

        for i in range(len(sorted_g)):
            h = int(group_hours[i])
            if total == 0:
                is_unusual[idx[i]] = 0  # cold-start
            else:
                proportion = hour_counts[h] / total
                is_unusual[idx[i]] = (
                    1 if hour_counts[h] == 0 or proportion < _UNUSUAL_HOUR_THRESHOLD else 0
                )
            hour_counts[h] += 1
            total += 1

    return pd.Series(is_unusual, index=df.index, name="is_unusual_hour", dtype=np.int8)


# ── Previous suspicious count ──────────────────────────────────────


def compute_previous_suspicious_count(df: pd.DataFrame) -> pd.Series:
    """Compute ``previous_suspicious_count`` (cumulative prior fraud flags).

    Uses ``cumsum().shift(1)`` grouped by ``card1`` to ensure the
    current transaction's label is **never** included — preventing
    target leakage.

    Returns:
        Series named ``previous_suspicious_count`` (int64).
    """
    cumsum = df.groupby("card1")["isFraud"].cumsum()
    shifted = cumsum - df["isFraud"]
    return shifted.fillna(0).astype(np.int64).rename("previous_suspicious_count")


# ── Aggregate ────────────────────────────────────────────────────────


def compute_all_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every historical feature and return a single DataFrame.

    Args:
        df: Joined transaction + identity DataFrame.  **Must** be sorted
            by ``TransactionDT`` (ascending).

    Returns:
        DataFrame with all historical feature columns, aligned to
        ``df.index``.
    """
    parts = [
        compute_amount_rolling_features(df),
        compute_velocity_features(df),
        compute_is_new_features(df),
        pd.DataFrame({"location_change": compute_location_change(df)}),
        pd.DataFrame({"avg_spend_30d": compute_avg_spend_30d(df)}),
        pd.DataFrame({"is_unusual_hour": compute_is_unusual_hour(df)}),
        pd.DataFrame({"previous_suspicious_count": compute_previous_suspicious_count(df)}),
    ]
    return pd.concat(parts, axis=1)
