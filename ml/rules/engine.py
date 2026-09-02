"""Rule-based risk signals and behavioural anomaly analysis.

Implements the components described in ``docs/ml-architecture.md``:

* **Section 3 — Behavioural Anomaly Analysis**: statistical signals
  that detect deviations from the customer's established baseline.
* **Section 4 — Rule-Based Risk Signals**: configurable business rules
  that flag known fraud patterns.

Both components consume the already-engineered 24-feature DataFrame
and (optionally) the customer history records.  They are independent
from the trained XGBoost model — no labels, no retraining, no
preprocessing fitting.

Design principles
-----------------
* Reuse features from :data:`ml.features.engineer.FEATURE_LIST` — never
  recompute what already exists.
* Deterministic: identical inputs → identical outputs.
* JSON/API-friendly structured output.
* Cold-start safe: all rules degrade gracefully when no history
  exists (first transaction).
* Leakage-safe: never access training labels, never use future
  transactions, never count the current transaction as historical
  evidence.

Usage::

    from ml.rules.engine import evaluate_rules

    result = evaluate_rules(features_df, raw_data, history)
    # result.rule_score        → int [0, 100]
    # result.behaviour_score   → int [0, 100]
    # result.rules_triggered   → list[RuleTrigger]
    # result.behaviour_signals → list[BehaviourSignal]
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pandas as pd

# ── Configurable thresholds ──────────────────────────────────────────
# Rules and thresholds from docs/ml-architecture.md §4.
# "Rules and their score contributions are configurable and can be
#  adjusted without code changes (via configuration file or environment)."

# Rule contribution scores (architecture §4)
HIGH_AMOUNT_CONTRIBUTION = 15
IMPOSSIBLE_TRAVEL_CONTRIBUTION = 25
VELOCITY_LIMIT_CONTRIBUTION = 20
NEW_DEVICE_HIGH_AMOUNT_CONTRIBUTION = 15
HIGH_RISK_MERCHANT_CONTRIBUTION = 10
PREVIOUS_SUSPICIOUS_CONTRIBUTION = 10

# Rule thresholds
HIGH_AMOUNT_THRESHOLD: float = 10_000.0
"""Amount above which the high_amount rule triggers."""

NEW_DEVICE_HIGH_AMOUNT_THRESHOLD: float = 5_000.0
"""Amount threshold for the new_device + high_amount combo rule."""

VELOCITY_LIMIT_1H: int = 5
"""More than this many transactions within 1 hour triggers velocity_limit."""

HIGH_RISK_MERCHANT_CATEGORIES: frozenset[str] = frozenset({
    "7995",  # Gambling
    "7996",  # Amusement parks / casinos
    "6012",  # Financial institutions — merchandise/services
    "6051",  # Non-financial institutions — foreign currency / crypto
    "4829",  # Wire transfers / money orders
})
"""Merchant category codes flagged as high-risk."""

IMPOSSIBLE_TRAVEL_SECONDS: int = 7_200
"""Minimum seconds between distant-location transactions to be plausible.

Two transactions from different countries within this window are
flagged as impossible travel.  Default: 2 hours (7200 s).
"""

# Behaviour signal thresholds
SPENDING_ANOMALY_ZSCORE: float = 2.0
"""Z-score threshold for spending amount anomaly."""

VELOCITY_ANOMALY_THRESHOLD: int = 3
"""tx_velocity_1h above this triggers velocity anomaly."""


# ── Output dataclasses ───────────────────────────────────────────────


@dataclasses.dataclass
class RuleTrigger:
    """A single triggered rule."""

    rule: str
    """Rule identifier (snake_case)."""

    contribution: int
    """Score contribution to the cumulative rule score."""

    reason: str
    """Human-readable explanation of why the rule triggered."""

    value: float | int | str | None = None
    """Relevant feature/value that triggered the rule."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (for JSON serialisation)."""
        d = dataclasses.asdict(self)
        return {k: v for k, v in d.items() if v is not None or k in ("rule", "contribution", "reason")}


@dataclasses.dataclass
class BehaviourSignal:
    """A single behavioural anomaly signal."""

    signal: str
    """Signal identifier (snake_case)."""

    severity: float
    """Severity in [0.0, 1.0].  Higher = more anomalous."""

    reason: str
    """Human-readable explanation."""

    value: float | int | str | None = None
    """Relevant feature/value."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (for JSON serialisation)."""
        d = dataclasses.asdict(self)
        return {k: v for k, v in d.items() if v is not None or k in ("signal", "severity", "reason")}


@dataclasses.dataclass
class RuleResult:
    """Complete output from the rule engine evaluation."""

    rule_score: int
    """Cumulative rule-based score [0, 100], capped at 100."""

    behaviour_score: int
    """Behavioural anomaly score [0, 100], capped at 100."""

    rules_triggered: list[RuleTrigger]
    """List of triggered rules (deterministic order)."""

    behaviour_signals: list[BehaviourSignal]
    """List of behavioural anomaly signals (deterministic order)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (for JSON serialisation)."""
        return {
            "rule_score": self.rule_score,
            "behaviour_score": self.behaviour_score,
            "rules_triggered": [r.to_dict() for r in self.rules_triggered],
            "behaviour_signals": [s.to_dict() for s in self.behaviour_signals],
        }


# ── Main evaluation entry point ──────────────────────────────────────


def evaluate_rules(
    features: pd.DataFrame,
    raw: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> RuleResult:
    """Evaluate all rules and behavioural signals.

    Args:
        features: Single-row DataFrame with :data:`FEATURE_LIST` columns
                  (output of :func:`engineer_features_for_inference`).
        raw: Raw transaction dict (same payload sent to ``/predict``).
        history: Customer history records (from the history store's
                 ``get()`` method).  May be ``None`` or empty.

    Returns:
        :class:`RuleResult` with scores and triggered items.
    """
    row = features.iloc[0]
    history = history or []

    rules: list[RuleTrigger] = []
    signals: list[BehaviourSignal] = []

    # ── Behavioural anomaly signals (§3) ──────────────────────────
    _eval_spending_anomaly(row, signals)
    _eval_location_anomaly(row, signals)
    _eval_device_anomaly(row, signals)
    _eval_time_anomaly(row, signals)
    _eval_velocity_anomaly(row, signals)

    # ── Rule-based risk signals (§4) ──────────────────────────────
    _eval_high_amount(row, rules)
    _eval_impossible_travel(row, raw, history, rules)
    _eval_velocity_limit(row, rules)
    _eval_new_device_high_amount(row, rules)
    _eval_high_risk_merchant(row, raw, rules)
    _eval_previous_suspicious(row, rules)

    # ── Aggregate scores ──────────────────────────────────────────
    rule_score = min(sum(r.contribution for r in rules), 100)
    behaviour_score = min(
        int(sum(s.severity for s in signals) / max(len(signals), 1) * 100),
        100,
    )

    return RuleResult(
        rule_score=rule_score,
        behaviour_score=behaviour_score,
        rules_triggered=rules,
        behaviour_signals=signals,
    )


# ── Behavioural anomaly signal evaluators ────────────────────────────


def _eval_spending_anomaly(
    row: pd.Series, signals: list[BehaviourSignal]
) -> None:
    """Spending amount anomaly — Z-score against customer baseline.

    Uses ``amount_deviation`` (pre-computed Z-score) from features.
    """
    z = float(row.get("amount_deviation", 0.0))
    abs_z = abs(z)
    if abs_z >= SPENDING_ANOMALY_ZSCORE:
        severity = min(abs_z / 5.0, 1.0)
        signals.append(BehaviourSignal(
            signal="spending_amount_anomaly",
            severity=round(severity, 4),
            reason=f"Spending Z-score {z:.2f} exceeds threshold {SPENDING_ANOMALY_ZSCORE}",
            value=round(z, 4),
        ))


def _eval_location_anomaly(
    row: pd.Series, signals: list[BehaviourSignal]
) -> None:
    """Location anomaly — first occurrence of country in history.

    Uses ``location_is_new`` from features.
    """
    if int(row.get("location_is_new", 0)) == 1:
        signals.append(BehaviourSignal(
            signal="location_anomaly",
            severity=0.8,
            reason="Transaction from a country not seen in customer history",
            value=int(row.get("location_country", 0)),
        ))


def _eval_device_anomaly(
    row: pd.Series, signals: list[BehaviourSignal]
) -> None:
    """Device anomaly — first occurrence of device fingerprint.

    Uses ``is_new_device`` from features.
    """
    if int(row.get("is_new_device", 0)) == 1:
        signals.append(BehaviourSignal(
            signal="device_anomaly",
            severity=0.7,
            reason="Transaction from a previously unseen device",
            value=str(row.get("device_fingerprint", "unknown")),
        ))


def _eval_time_anomaly(
    row: pd.Series, signals: list[BehaviourSignal]
) -> None:
    """Time anomaly — transaction outside typical activity hours.

    Uses ``is_unusual_hour`` from features.
    """
    if int(row.get("is_unusual_hour", 0)) == 1:
        signals.append(BehaviourSignal(
            signal="time_anomaly",
            severity=0.5,
            reason="Transaction occurred during an unusual hour for this customer",
            value=int(row.get("hour_of_day_raw", 0)),
        ))


def _eval_velocity_anomaly(
    row: pd.Series, signals: list[BehaviourSignal]
) -> None:
    """Velocity anomaly — transaction count exceeds baseline.

    Uses ``tx_velocity_1h`` from features.
    """
    v1h = int(row.get("tx_velocity_1h", 0))
    if v1h >= VELOCITY_ANOMALY_THRESHOLD:
        severity = min(v1h / 10.0, 1.0)
        signals.append(BehaviourSignal(
            signal="velocity_anomaly",
            severity=round(severity, 4),
            reason=f"{v1h} transactions in the last hour (threshold: {VELOCITY_ANOMALY_THRESHOLD})",
            value=v1h,
        ))


# ── Rule-based signal evaluators ─────────────────────────────────────


def _eval_high_amount(
    row: pd.Series, rules: list[RuleTrigger]
) -> None:
    """High amount — amount exceeds configurable threshold.

    Uses ``amount`` from features.  Contribution: +15.
    """
    amount = float(row.get("amount", 0.0))
    if amount > HIGH_AMOUNT_THRESHOLD:
        rules.append(RuleTrigger(
            rule="high_amount",
            contribution=HIGH_AMOUNT_CONTRIBUTION,
            reason=f"Transaction amount {amount:.2f} exceeds threshold {HIGH_AMOUNT_THRESHOLD:.2f}",
            value=amount,
        ))


def _eval_impossible_travel(
    row: pd.Series,
    raw: dict[str, Any],
    history: list[dict[str, Any]],
    rules: list[RuleTrigger],
) -> None:
    """Impossible travel — distant locations within implausibly short time.

    Compares the current transaction's location (addr2 / location_country)
    with the most recent historical transaction.  If the country changed
    and the time gap is below :data:`IMPOSSIBLE_TRAVEL_SECONDS`, the rule
    triggers.

    Uses ``location_country`` (addr2) and history records.
    Contribution: +25.
    """
    if not history:
        return

    current_ts = int(raw.get("timestamp", 0))
    current_country = raw.get("location_country") or str(int(row.get("location_country", 0)))

    # Find the most recent prior transaction
    last = history[-1]
    last_ts = int(last.get("timestamp", 0))
    last_country = last.get("location_country")
    # History stores addr2 (integer country code); fall back to raw field
    if last_country is None:
        last_addr2 = last.get("addr2")
        last_country = str(last_addr2) if last_addr2 is not None else None

    if last_country is None or current_country is None:
        return

    # Normalise for comparison
    if str(current_country) == str(last_country):
        return  # Same country — no travel

    time_gap = abs(current_ts - last_ts)
    if time_gap < IMPOSSIBLE_TRAVEL_SECONDS and time_gap > 0:
        rules.append(RuleTrigger(
            rule="impossible_travel",
            contribution=IMPOSSIBLE_TRAVEL_CONTRIBUTION,
            reason=(
                f"Location changed from {last_country} to {current_country} "
                f"within {time_gap}s (threshold: {IMPOSSIBLE_TRAVEL_SECONDS}s)"
            ),
            value=time_gap,
        ))


def _eval_velocity_limit(
    row: pd.Series, rules: list[RuleTrigger]
) -> None:
    """Velocity limit — more than N transactions within T minutes.

    Uses ``tx_velocity_1h`` from features.  Contribution: +20.
    """
    v1h = int(row.get("tx_velocity_1h", 0))
    if v1h >= VELOCITY_LIMIT_1H:
        rules.append(RuleTrigger(
            rule="velocity_limit",
            contribution=VELOCITY_LIMIT_CONTRIBUTION,
            reason=f"{v1h} transactions in the last hour (limit: {VELOCITY_LIMIT_1H})",
            value=v1h,
        ))


def _eval_new_device_high_amount(
    row: pd.Series, rules: list[RuleTrigger]
) -> None:
    """New device + high amount — first-seen device with large amount.

    Uses ``is_new_device`` and ``amount`` from features.
    Contribution: +15.
    """
    is_new = int(row.get("is_new_device", 0))
    amount = float(row.get("amount", 0.0))
    if is_new == 1 and amount > NEW_DEVICE_HIGH_AMOUNT_THRESHOLD:
        rules.append(RuleTrigger(
            rule="new_device_high_amount",
            contribution=NEW_DEVICE_HIGH_AMOUNT_CONTRIBUTION,
            reason=(
                f"New device with amount {amount:.2f} "
                f"(threshold: {NEW_DEVICE_HIGH_AMOUNT_THRESHOLD:.2f})"
            ),
            value=amount,
        ))


def _eval_high_risk_merchant(
    row: pd.Series,
    raw: dict[str, Any],
    rules: list[RuleTrigger],
) -> None:
    """High-risk merchant category — MCC in flagged set.

    Checks the raw ``merchant_category`` field against
    :data:`HIGH_RISK_MERCHANT_CATEGORIES`.  Contribution: +10.
    """
    # Prefer raw merchant_category (string MCC code from backend)
    mcc = str(raw.get("merchant_category", ""))
    if mcc in HIGH_RISK_MERCHANT_CATEGORIES:
        rules.append(RuleTrigger(
            rule="high_risk_merchant",
            contribution=HIGH_RISK_MERCHANT_CONTRIBUTION,
            reason=f"Merchant category {mcc} is flagged as high-risk",
            value=mcc,
        ))


def _eval_previous_suspicious(
    row: pd.Series, rules: list[RuleTrigger]
) -> None:
    """Previous suspicious activity — customer has prior flagged transactions.

    Uses ``previous_suspicious_count`` from features.
    Contribution: +10.
    """
    count = int(row.get("previous_suspicious_count", 0))
    if count > 0:
        rules.append(RuleTrigger(
            rule="previous_suspicious",
            contribution=PREVIOUS_SUSPICIOUS_CONTRIBUTION,
            reason=f"Customer has {count} prior flagged transaction(s)",
            value=count,
        ))
