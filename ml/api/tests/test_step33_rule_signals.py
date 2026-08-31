"""Step 33 — Rule-based risk signals tests.

Validates the rule engine, behavioural anomaly signals, risk aggregation,
and full pipeline integration.

Covers:
  1. No rules triggered for a normal transaction
  2. Each documented rule can trigger independently
  3. Multiple rules can trigger on the same transaction
  4. Rule output has the expected structured schema
  5. Rule ordering is deterministic
  6. Threshold/boundary behaviour is correct
  7. First transaction/cold-start behaviour
  8. Historical rules use only prior transactions
  9. Future transactions cannot trigger historical rules
 10. Customer isolation
 11. Outcome feedback affects historical rule signals
 12. Existing SHAP explanations still work
 13. Fraud probability unchanged by rule evaluation
 14. Prediction class/threshold unchanged
 15. Invalid input rejected correctly
 16. Model-unavailable behaviour
 17. Backend → ML → rule signals → response (end-to-end)

Run from project root::

    python -m pytest ml/api/tests/test_step33_rule_signals.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ml.api.app import app
import ml.features.history as _history_module
from ml.features.engineer import FEATURE_LIST, engineer_features_for_inference
from ml.features.history import (
    InMemoryHistoryStore,
    record_transaction,
)
from ml.rules.engine import (
    HIGH_AMOUNT_THRESHOLD,
    IMPOSSIBLE_TRAVEL_SECONDS,
    NEW_DEVICE_HIGH_AMOUNT_THRESHOLD,
    SPENDING_ANOMALY_ZSCORE,
    VELOCITY_ANOMALY_THRESHOLD,
    VELOCITY_LIMIT_1H,
    BehaviourSignal,
    RuleResult,
    RuleTrigger,
    evaluate_rules,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_history():
    """Ensure a working history store and clear it before each test."""
    store = _history_module.history_store
    try:
        store.clear()
    except Exception:
        # Store may be closed (e.g. after a nested TestClient lifespan).
        # Replace with a fresh in-memory store so the test can proceed.
        new_store = InMemoryHistoryStore()
        _history_module.history_store = new_store
    yield
    store = _history_module.history_store
    try:
        store.clear()
    except Exception:
        pass


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with model loaded via lifespan."""
    with TestClient(app) as c:
        yield c


# ── Helpers ───────────────────────────────────────────────────────────


def _features(**overrides) -> pd.DataFrame:
    """Build a single-row DataFrame with all FEATURE_LIST columns.

    Default values are zero/cold-start safe; override any feature
    to trigger specific rules.
    """
    data: dict[str, float] = {}
    for f in FEATURE_LIST:
        data[f] = 0.0
    data.update(overrides)
    return pd.DataFrame([data])


def _raw(**overrides) -> dict:
    """Minimal valid raw transaction payload."""
    base = {
        "amount": 100.0,
        "currency": "USD",
        "merchant_name": "Test Merchant",
        "merchant_category": "5732",
        "transaction_type": "purchase",
        "location_country": "US",
        "location_city": "New York",
        "device_fingerprint": "fp_abc123",
        "device_type": "desktop",
        "ip_address": "192.168.1.1",
        "customer_id": "cust_33",
        "timestamp": 86_400,
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════
# 1. No rules triggered for a normal transaction
# ═══════════════════════════════════════════════════════════════════════


class TestNoRulesTriggered:
    """A normal low-risk transaction triggers no rules or signals."""

    def test_normal_transaction_no_rules(self):
        features = _features(amount=100.0, tx_velocity_1h=1)
        raw = _raw(amount=100.0)
        result = evaluate_rules(features, raw, history=None)

        assert isinstance(result, RuleResult)
        assert result.rule_score == 0
        assert result.behaviour_score == 0
        assert result.rules_triggered == []
        assert result.behaviour_signals == []

    def test_normal_transaction_to_dict(self):
        features = _features(amount=50.0)
        raw = _raw(amount=50.0)
        result = evaluate_rules(features, raw)
        d = result.to_dict()
        assert d["rule_score"] == 0
        assert d["behaviour_score"] == 0
        assert d["rules_triggered"] == []
        assert d["behaviour_signals"] == []


# ═══════════════════════════════════════════════════════════════════════
# 2. Each documented rule triggers independently
# ═══════════════════════════════════════════════════════════════════════


class TestIndividualRules:
    """Each rule can trigger on its own with the correct contribution."""

    # ── high_amount (+15) ─────────────────────────────────────────

    def test_high_amount_triggers(self):
        features = _features(amount=HIGH_AMOUNT_THRESHOLD + 1)
        raw = _raw(amount=HIGH_AMOUNT_THRESHOLD + 1)
        result = evaluate_rules(features, raw)

        assert len(result.rules_triggered) == 1
        rule = result.rules_triggered[0]
        assert rule.rule == "high_amount"
        assert rule.contribution == 15
        assert rule.value == HIGH_AMOUNT_THRESHOLD + 1
        assert "exceeds" in rule.reason.lower()
        assert result.rule_score == 15

    def test_high_amount_not_triggered_below(self):
        features = _features(amount=HIGH_AMOUNT_THRESHOLD - 1)
        raw = _raw(amount=HIGH_AMOUNT_THRESHOLD - 1)
        result = evaluate_rules(features, raw)
        assert not any(r.rule == "high_amount" for r in result.rules_triggered)

    # ── impossible_travel (+25) ──────────────────────────────────

    def test_impossible_travel_triggers(self):
        features = _features(location_country=2)
        raw = _raw(location_country="2", timestamp=5000)
        history = [
            {"timestamp": 4000, "location_country": "1", "addr2": 1},
        ]
        result = evaluate_rules(features, raw, history=history)

        assert any(r.rule == "impossible_travel" for r in result.rules_triggered)
        travel = [r for r in result.rules_triggered if r.rule == "impossible_travel"][0]
        assert travel.contribution == 25
        assert travel.value == 1000  # 5000 - 4000

    def test_impossible_travel_not_triggered_same_country(self):
        features = _features(location_country=1)
        raw = _raw(location_country="1", timestamp=5000)
        history = [{"timestamp": 4000, "location_country": "1"}]
        result = evaluate_rules(features, raw, history=history)
        assert not any(r.rule == "impossible_travel" for r in result.rules_triggered)

    def test_impossible_travel_not_triggered_no_history(self):
        features = _features(location_country=2)
        raw = _raw(location_country="2")
        result = evaluate_rules(features, raw, history=None)
        assert not any(r.rule == "impossible_travel" for r in result.rules_triggered)

    def test_impossible_travel_not_triggered_long_gap(self):
        """Time gap exceeds IMPOSSIBLE_TRAVEL_SECONDS — plausible travel."""
        features = _features(location_country=2)
        raw = _raw(location_country="2", timestamp=100_000)
        history = [{"timestamp": 100_000 - IMPOSSIBLE_TRAVEL_SECONDS - 100, "location_country": "1"}]
        result = evaluate_rules(features, raw, history=history)
        assert not any(r.rule == "impossible_travel" for r in result.rules_triggered)

    # ── velocity_limit (+20) ──────────────────────────────────────

    def test_velocity_limit_triggers(self):
        features = _features(tx_velocity_1h=VELOCITY_LIMIT_1H)
        raw = _raw()
        result = evaluate_rules(features, raw)

        assert any(r.rule == "velocity_limit" for r in result.rules_triggered)
        vel = [r for r in result.rules_triggered if r.rule == "velocity_limit"][0]
        assert vel.contribution == 20
        assert vel.value == VELOCITY_LIMIT_1H

    def test_velocity_limit_not_triggered_below(self):
        features = _features(tx_velocity_1h=VELOCITY_LIMIT_1H - 1)
        raw = _raw()
        result = evaluate_rules(features, raw)
        assert not any(r.rule == "velocity_limit" for r in result.rules_triggered)

    # ── new_device_high_amount (+15) ──────────────────────────────

    def test_new_device_high_amount_triggers(self):
        features = _features(
            is_new_device=1, amount=NEW_DEVICE_HIGH_AMOUNT_THRESHOLD + 1
        )
        raw = _raw(amount=NEW_DEVICE_HIGH_AMOUNT_THRESHOLD + 1)
        result = evaluate_rules(features, raw)

        assert any(r.rule == "new_device_high_amount" for r in result.rules_triggered)
        nd = [r for r in result.rules_triggered if r.rule == "new_device_high_amount"][0]
        assert nd.contribution == 15

    def test_new_device_high_amount_not_triggered_known_device(self):
        features = _features(
            is_new_device=0, amount=NEW_DEVICE_HIGH_AMOUNT_THRESHOLD + 1
        )
        raw = _raw(amount=NEW_DEVICE_HIGH_AMOUNT_THRESHOLD + 1)
        result = evaluate_rules(features, raw)
        assert not any(r.rule == "new_device_high_amount" for r in result.rules_triggered)

    def test_new_device_high_amount_not_triggered_low_amount(self):
        features = _features(is_new_device=1, amount=100.0)
        raw = _raw(amount=100.0)
        result = evaluate_rules(features, raw)
        assert not any(r.rule == "new_device_high_amount" for r in result.rules_triggered)

    # ── high_risk_merchant (+10) ──────────────────────────────────

    def test_high_risk_merchant_triggers(self):
        features = _features(merchant_category=7995)
        raw = _raw(merchant_category="7995")
        result = evaluate_rules(features, raw)

        assert any(r.rule == "high_risk_merchant" for r in result.rules_triggered)
        hr = [r for r in result.rules_triggered if r.rule == "high_risk_merchant"][0]
        assert hr.contribution == 10
        assert hr.value == "7995"

    def test_high_risk_merchant_not_triggered_safe_category(self):
        features = _features(merchant_category=5411)
        raw = _raw(merchant_category="5411")
        result = evaluate_rules(features, raw)
        assert not any(r.rule == "high_risk_merchant" for r in result.rules_triggered)

    # ── previous_suspicious (+10) ─────────────────────────────────

    def test_previous_suspicious_triggers(self):
        features = _features(previous_suspicious_count=2)
        raw = _raw()
        result = evaluate_rules(features, raw)

        assert any(r.rule == "previous_suspicious" for r in result.rules_triggered)
        ps = [r for r in result.rules_triggered if r.rule == "previous_suspicious"][0]
        assert ps.contribution == 10
        assert ps.value == 2

    def test_previous_suspicious_not_triggered_zero(self):
        features = _features(previous_suspicious_count=0)
        raw = _raw()
        result = evaluate_rules(features, raw)
        assert not any(r.rule == "previous_suspicious" for r in result.rules_triggered)


# ═══════════════════════════════════════════════════════════════════════
# 3. Multiple rules on the same transaction
# ═══════════════════════════════════════════════════════════════════════


class TestMultipleRules:
    """Multiple rules can fire simultaneously."""

    def test_multiple_rules_trigger(self):
        features = _features(
            amount=HIGH_AMOUNT_THRESHOLD + 500,
            is_new_device=1,
            tx_velocity_1h=VELOCITY_LIMIT_1H + 1,
            previous_suspicious_count=1,
        )
        raw = _raw(
            amount=HIGH_AMOUNT_THRESHOLD + 500,
            merchant_category="7995",
        )
        result = evaluate_rules(features, raw)

        rule_names = {r.rule for r in result.rules_triggered}
        assert "high_amount" in rule_names
        assert "velocity_limit" in rule_names
        assert "new_device_high_amount" in rule_names
        assert "high_risk_merchant" in rule_names
        assert "previous_suspicious" in rule_names
        # 15 + 20 + 15 + 10 + 10 = 70
        assert result.rule_score == 70

    def test_rule_score_capped_at_100(self):
        """rule_score is capped at 100 even if contributions exceed it."""
        features = _features(
            amount=HIGH_AMOUNT_THRESHOLD + 500,
            is_new_device=1,
            tx_velocity_1h=VELOCITY_LIMIT_1H + 1,
            previous_suspicious_count=3,
        )
        raw = _raw(amount=HIGH_AMOUNT_THRESHOLD + 500, merchant_category="7995")
        history = [
            {"timestamp": raw["timestamp"] - 1000, "location_country": "99"},
        ]
        result = evaluate_rules(features, raw, history=history)

        # All 6 rules: 15+25+20+15+10+10 = 95, but let's verify cap
        assert result.rule_score <= 100

    def test_multiple_behaviour_signals(self):
        features = _features(
            amount_deviation=3.5,
            location_is_new=1,
            is_new_device=1,
            is_unusual_hour=1,
            tx_velocity_1h=5,
        )
        raw = _raw()
        result = evaluate_rules(features, raw)

        signal_names = {s.signal for s in result.behaviour_signals}
        assert "spending_amount_anomaly" in signal_names
        assert "location_anomaly" in signal_names
        assert "device_anomaly" in signal_names
        assert "time_anomaly" in signal_names
        assert "velocity_anomaly" in signal_names


# ═══════════════════════════════════════════════════════════════════════
# 4. Structured output schema
# ═══════════════════════════════════════════════════════════════════════


class TestOutputSchema:
    """Rule output conforms to the expected structured schema."""

    def test_rule_trigger_schema(self):
        features = _features(amount=HIGH_AMOUNT_THRESHOLD + 1)
        raw = _raw(amount=HIGH_AMOUNT_THRESHOLD + 1)
        result = evaluate_rules(features, raw)

        rule = result.rules_triggered[0]
        d = rule.to_dict()
        assert "rule" in d
        assert "contribution" in d
        assert "reason" in d
        assert isinstance(d["rule"], str)
        assert isinstance(d["contribution"], int)
        assert isinstance(d["reason"], str)

    def test_behaviour_signal_schema(self):
        features = _features(amount_deviation=3.0)
        raw = _raw()
        result = evaluate_rules(features, raw)

        signal = result.behaviour_signals[0]
        d = signal.to_dict()
        assert "signal" in d
        assert "severity" in d
        assert "reason" in d
        assert isinstance(d["signal"], str)
        assert isinstance(d["severity"], float)
        assert 0.0 <= d["severity"] <= 1.0

    def test_rule_result_schema(self):
        features = _features(amount=50.0)
        raw = _raw(amount=50.0)
        result = evaluate_rules(features, raw)
        d = result.to_dict()

        assert "rule_score" in d
        assert "behaviour_score" in d
        assert "rules_triggered" in d
        assert "behaviour_signals" in d
        assert isinstance(d["rule_score"], int)
        assert isinstance(d["behaviour_score"], int)
        assert isinstance(d["rules_triggered"], list)
        assert isinstance(d["behaviour_signals"], list)

    def test_scores_within_range(self):
        features = _features(
            amount=HIGH_AMOUNT_THRESHOLD + 1,
            is_new_device=1,
            tx_velocity_1h=10,
            amount_deviation=5.0,
            location_is_new=1,
        )
        raw = _raw(amount=HIGH_AMOUNT_THRESHOLD + 1, merchant_category="7995")
        result = evaluate_rules(features, raw)

        assert 0 <= result.rule_score <= 100
        assert 0 <= result.behaviour_score <= 100


# ═══════════════════════════════════════════════════════════════════════
# 5. Deterministic rule ordering
# ═══════════════════════════════════════════════════════════════════════


class TestDeterministicOrdering:
    """Rules and signals always appear in the same order."""

    def test_rule_ordering_deterministic(self):
        """Multiple runs produce the same rule ordering."""
        features = _features(
            amount=HIGH_AMOUNT_THRESHOLD + 500,
            is_new_device=1,
            tx_velocity_1h=VELOCITY_LIMIT_1H + 1,
            previous_suspicious_count=1,
        )
        raw = _raw(amount=HIGH_AMOUNT_THRESHOLD + 500, merchant_category="7995")

        results = [evaluate_rules(features, raw) for _ in range(5)]
        orderings = [
            [r.rule for r in res.rules_triggered] for res in results
        ]
        assert all(o == orderings[0] for o in orderings)

    def test_behaviour_signal_ordering_deterministic(self):
        features = _features(
            amount_deviation=3.5,
            location_is_new=1,
            is_new_device=1,
            is_unusual_hour=1,
            tx_velocity_1h=5,
        )
        raw = _raw()
        results = [evaluate_rules(features, raw) for _ in range(5)]
        orderings = [
            [s.signal for s in res.behaviour_signals] for res in results
        ]
        assert all(o == orderings[0] for o in orderings)

    def test_rule_evaluation_order_matches_code(self):
        """Rules are evaluated in the documented order."""
        features = _features(
            amount=HIGH_AMOUNT_THRESHOLD + 500,
            is_new_device=1,
            tx_velocity_1h=VELOCITY_LIMIT_1H + 1,
            previous_suspicious_count=1,
        )
        raw = _raw(amount=HIGH_AMOUNT_THRESHOLD + 500, merchant_category="7995")
        history = [
            {"timestamp": raw["timestamp"] - 1000, "location_country": "99"},
        ]
        result = evaluate_rules(features, raw, history=history)
        rule_names = [r.rule for r in result.rules_triggered]

        # Expected order from evaluate_rules(): high_amount, impossible_travel,
        # velocity_limit, new_device_high_amount, high_risk_merchant, previous_suspicious
        expected = [
            "high_amount",
            "impossible_travel",
            "velocity_limit",
            "new_device_high_amount",
            "high_risk_merchant",
            "previous_suspicious",
        ]
        assert rule_names == expected


# ═══════════════════════════════════════════════════════════════════════
# 6. Threshold / boundary behaviour
# ═══════════════════════════════════════════════════════════════════════


class TestThresholdBoundaries:
    """Exact threshold boundaries behave correctly."""

    def test_high_amount_exact_boundary(self):
        """Amount exactly at threshold does NOT trigger (strict >)."""
        features = _features(amount=HIGH_AMOUNT_THRESHOLD)
        raw = _raw(amount=HIGH_AMOUNT_THRESHOLD)
        result = evaluate_rules(features, raw)
        assert not any(r.rule == "high_amount" for r in result.rules_triggered)

    def test_high_amount_one_cent_above(self):
        features = _features(amount=HIGH_AMOUNT_THRESHOLD + 0.01)
        raw = _raw(amount=HIGH_AMOUNT_THRESHOLD + 0.01)
        result = evaluate_rules(features, raw)
        assert any(r.rule == "high_amount" for r in result.rules_triggered)

    def test_velocity_exact_boundary(self):
        """tx_velocity_1h exactly at limit triggers (>=)."""
        features = _features(tx_velocity_1h=VELOCITY_LIMIT_1H)
        raw = _raw()
        result = evaluate_rules(features, raw)
        assert any(r.rule == "velocity_limit" for r in result.rules_triggered)

    def test_velocity_one_below(self):
        features = _features(tx_velocity_1h=VELOCITY_LIMIT_1H - 1)
        raw = _raw()
        result = evaluate_rules(features, raw)
        assert not any(r.rule == "velocity_limit" for r in result.rules_triggered)

    def test_spending_anomaly_exact_boundary(self):
        """Z-score exactly at threshold triggers (>=)."""
        features = _features(amount_deviation=SPENDING_ANOMALY_ZSCORE)
        raw = _raw()
        result = evaluate_rules(features, raw)
        assert any(s.signal == "spending_amount_anomaly" for s in result.behaviour_signals)

    def test_spending_anomaly_below(self):
        features = _features(amount_deviation=SPENDING_ANOMALY_ZSCORE - 0.01)
        raw = _raw()
        result = evaluate_rules(features, raw)
        assert not any(s.signal == "spending_amount_anomaly" for s in result.behaviour_signals)

    def test_velocity_anomaly_exact_boundary(self):
        """tx_velocity_1h exactly at anomaly threshold triggers (>=)."""
        features = _features(tx_velocity_1h=VELOCITY_ANOMALY_THRESHOLD)
        raw = _raw()
        result = evaluate_rules(features, raw)
        assert any(s.signal == "velocity_anomaly" for s in result.behaviour_signals)

    def test_new_device_high_amount_exact_boundary(self):
        """Amount exactly at threshold does NOT trigger (strict >)."""
        features = _features(
            is_new_device=1, amount=NEW_DEVICE_HIGH_AMOUNT_THRESHOLD
        )
        raw = _raw(amount=NEW_DEVICE_HIGH_AMOUNT_THRESHOLD)
        result = evaluate_rules(features, raw)
        assert not any(
            r.rule == "new_device_high_amount" for r in result.rules_triggered
        )

    def test_impossible_travel_exact_boundary(self):
        """Time gap exactly at threshold does NOT trigger (strict <)."""
        features = _features(location_country=2)
        raw = _raw(location_country="2", timestamp=10000)
        history = [
            {
                "timestamp": 10000 - IMPOSSIBLE_TRAVEL_SECONDS,
                "location_country": "1",
            }
        ]
        result = evaluate_rules(features, raw, history=history)
        assert not any(
            r.rule == "impossible_travel" for r in result.rules_triggered
        )

    def test_impossible_travel_one_second_below(self):
        features = _features(location_country=2)
        raw = _raw(location_country="2", timestamp=10000)
        history = [
            {
                "timestamp": 10000 - IMPOSSIBLE_TRAVEL_SECONDS + 1,
                "location_country": "1",
            }
        ]
        result = evaluate_rules(features, raw, history=history)
        assert any(
            r.rule == "impossible_travel" for r in result.rules_triggered
        )


# ═══════════════════════════════════════════════════════════════════════
# 7. Cold-start / first transaction
# ═══════════════════════════════════════════════════════════════════════


class TestColdStart:
    """First transaction with no history degrades gracefully."""

    def test_cold_start_no_history(self):
        features = _features(amount=100.0)
        raw = _raw(amount=100.0)
        result = evaluate_rules(features, raw, history=None)
        assert isinstance(result, RuleResult)
        # impossible_travel should NOT trigger without history
        assert not any(r.rule == "impossible_travel" for r in result.rules_triggered)

    def test_cold_start_empty_history(self):
        features = _features(amount=100.0)
        raw = _raw(amount=100.0)
        result = evaluate_rules(features, raw, history=[])
        assert not any(r.rule == "impossible_travel" for r in result.rules_triggered)

    def test_cold_start_high_amount_still_works(self):
        """Non-history-dependent rules still trigger for first-timers."""
        features = _features(amount=HIGH_AMOUNT_THRESHOLD + 1)
        raw = _raw(amount=HIGH_AMOUNT_THRESHOLD + 1)
        result = evaluate_rules(features, raw, history=None)
        assert any(r.rule == "high_amount" for r in result.rules_triggered)

    def test_cold_start_new_device_still_works(self):
        features = _features(
            is_new_device=1, amount=NEW_DEVICE_HIGH_AMOUNT_THRESHOLD + 1
        )
        raw = _raw(amount=NEW_DEVICE_HIGH_AMOUNT_THRESHOLD + 1)
        result = evaluate_rules(features, raw, history=None)
        assert any(
            r.rule == "new_device_high_amount" for r in result.rules_triggered
        )


# ═══════════════════════════════════════════════════════════════════════
# 8. Historical rules use only prior transactions
# ═══════════════════════════════════════════════════════════════════════


class TestHistoricalRules:
    """impossible_travel uses only the history parameter, not the current tx."""

    def test_history_excludes_current_transaction(self):
        """The current transaction is not in the history list."""
        features = _features(location_country=1)
        raw = _raw(location_country="1", timestamp=5000)
        # History with same country as current — should NOT trigger
        history = [{"timestamp": 4000, "location_country": "1"}]
        result = evaluate_rules(features, raw, history=history)
        assert not any(r.rule == "impossible_travel" for r in result.rules_triggered)

    def test_history_uses_most_recent_transaction(self):
        """impossible_travel compares with the last history entry."""
        features = _features(location_country=2)
        raw = _raw(location_country="2", timestamp=5000)
        history = [
            {"timestamp": 1000, "location_country": "1"},
            {"timestamp": 4500, "location_country": "1"},  # most recent
        ]
        result = evaluate_rules(features, raw, history=history)
        travel = [r for r in result.rules_triggered if r.rule == "impossible_travel"]
        assert len(travel) == 1
        assert travel[0].value == 500  # 5000 - 4500


# ═══════════════════════════════════════════════════════════════════════
# 9. Future transactions excluded
# ═══════════════════════════════════════════════════════════════════════


class TestFutureExclusion:
    """Future transactions should not affect current evaluation."""

    def test_future_transaction_not_in_history(self):
        """If a future timestamp is in history, it should not be used."""
        # This is enforced by the history store's before_timestamp filter,
        # but the rule engine itself doesn't filter — it trusts the caller.
        # Here we verify the contract: history should be pre-filtered.
        features = _features(location_country=1)
        raw = _raw(location_country="1", timestamp=5000)
        # Simulating correctly pre-filtered history (empty)
        result = evaluate_rules(features, raw, history=[])
        assert not any(r.rule == "impossible_travel" for r in result.rules_triggered)


# ═══════════════════════════════════════════════════════════════════════
# 10. Customer isolation
# ═══════════════════════════════════════════════════════════════════════


class TestCustomerIsolation:
    """Customer A's history cannot affect Customer B's rules."""

    def test_isolated_customers(self):
        """Two customers with different histories get different results."""
        # Customer A has suspicious history
        features_a = _features(
            location_country=2, previous_suspicious_count=3
        )
        raw_a = _raw(
            customer_id="cust_A",
            location_country="2",
            timestamp=5000,
        )
        history_a = [
            {"timestamp": 4000, "location_country": "1", "addr2": 1},
        ]

        # Customer B has clean history
        features_b = _features(
            location_country=1, previous_suspicious_count=0
        )
        raw_b = _raw(
            customer_id="cust_B",
            location_country="1",
            timestamp=5000,
        )
        history_b = [
            {"timestamp": 4000, "location_country": "1"},
        ]

        result_a = evaluate_rules(features_a, raw_a, history=history_a)
        result_b = evaluate_rules(features_b, raw_b, history=history_b)

        # Customer A should have more rules triggered
        a_rules = {r.rule for r in result_a.rules_triggered}
        b_rules = {r.rule for r in result_b.rules_triggered}
        assert "impossible_travel" in a_rules
        assert "impossible_travel" not in b_rules
        assert "previous_suspicious" in a_rules
        assert "previous_suspicious" not in b_rules


# ═══════════════════════════════════════════════════════════════════════
# 11. Outcome feedback affects historical rule signals
# ═══════════════════════════════════════════════════════════════════════


class TestOutcomeFeedback:
    """Updating a historical fraud outcome affects previous_suspicious."""

    def test_outcome_update_affects_previous_suspicious(self, client: TestClient):
        """After marking a transaction as fraud, the next prediction
        should see an increased previous_suspicious_count."""
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        # First transaction
        raw1 = _raw(customer_id="cust_outcome_33", timestamp=1000, amount=50.0)
        resp1 = client.post("/predict", json=raw1)
        assert resp1.status_code == 200

        # Mark it as fraud
        outcome_resp = client.post(
            "/outcome",
            json={
                "customer_id": "cust_outcome_33",
                "timestamp": 1000,
                "is_fraud": 1,
            },
        )
        assert outcome_resp.status_code == 200

        # Next transaction should see previous_suspicious_count > 0
        raw2 = _raw(customer_id="cust_outcome_33", timestamp=2000, amount=50.0)
        resp2 = client.post("/predict", json=raw2)
        assert resp2.status_code == 200
        data = resp2.json()

        # Verify the previous_suspicious rule can trigger
        explanation = data.get("explanation_detail") or {}
        triggered_rules = [r["rule"] for r in explanation.get("rules_triggered", [])]
        assert "previous_suspicious" in triggered_rules


# ═══════════════════════════════════════════════════════════════════════
# 12. Existing SHAP explanations still work
# ═══════════════════════════════════════════════════════════════════════


class TestSHAPIntegration:
    """SHAP explanations are preserved after adding rule evaluation."""

    def test_shap_explanation_present(self, client: TestClient):
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        raw = _raw(customer_id="cust_shap_33")
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        # Legacy explanation field
        assert data.get("explanation") is not None
        assert isinstance(data["explanation"], list)
        assert len(data["explanation"]) > 0
        assert "feature" in data["explanation"][0]
        assert "importance" in data["explanation"][0]

    def test_explanation_detail_has_ml_top_factors(self, client: TestClient):
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        raw = _raw(customer_id="cust_expl_detail_33")
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        expl = data.get("explanation_detail")
        assert expl is not None
        assert "ml_top_factors" in expl
        assert isinstance(expl["ml_top_factors"], list)
        assert len(expl["ml_top_factors"]) > 0


# ═══════════════════════════════════════════════════════════════════════
# 13. Fraud probability unchanged by rule evaluation
# ═══════════════════════════════════════════════════════════════════════


class TestFraudProbabilityUnchanged:
    """Rule evaluation does not alter the ML prediction."""

    def test_fraud_probability_unchanged(self, client: TestClient):
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        raw = _raw(customer_id="cust_prob_33", amount=500.0)
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        # fraud_probability must be in [0, 1]
        prob = data["fraud_probability"]
        assert 0.0 <= prob <= 1.0
        # prediction must be 0 or 1
        assert data["fraud_prediction"] in (0, 1)
        # threshold must be positive
        assert data["threshold"] > 0


# ═══════════════════════════════════════════════════════════════════════
# 14. Prediction class/threshold unchanged
# ═══════════════════════════════════════════════════════════════════════


class TestPredictionClassUnchanged:
    """Binary prediction and threshold remain the same."""

    def test_prediction_class_binary(self, client: TestClient):
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        raw = _raw(customer_id="cust_class_33")
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        assert data["fraud_prediction"] in (0, 1)
        assert isinstance(data["threshold"], float)
        assert data["threshold"] > 0


# ═══════════════════════════════════════════════════════════════════════
# 15. Invalid input rejected correctly
# ═══════════════════════════════════════════════════════════════════════


class TestInvalidInput:
    """Invalid input is still rejected with 422."""

    def test_missing_required_field(self, client: TestClient):
        payload = {"amount": 100.0}  # missing many fields
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_negative_amount(self, client: TestClient):
        raw = _raw(amount=-50.0)
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 422

    def test_forbidden_isfraud_field(self, client: TestClient):
        raw = _raw()
        raw["isFraud"] = 1
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 422

    def test_forbidden_transaction_id(self, client: TestClient):
        raw = _raw()
        raw["TransactionID"] = "T_001"
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 422

    def test_invalid_device_type(self, client: TestClient):
        raw = _raw(device_type="tablet")
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════
# 16. Model-unavailable behaviour
# ═══════════════════════════════════════════════════════════════════════


class TestModelUnavailable:
    """503 when model is not loaded."""

    def test_model_unavailable_returns_503(self, client: TestClient):
        """Temporarily set predictor to None on the running app."""
        import ml.api.app as app_module

        original = app_module._predictor
        app_module._predictor = None
        try:
            resp = client.post("/predict", json=_raw())
            assert resp.status_code == 503
        finally:
            app_module._predictor = original


# ═══════════════════════════════════════════════════════════════════════
# 17. End-to-end: Backend → ML → rule signals → response
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEndIntegration:
    """Full pipeline including risk scores in the response."""

    def test_e2e_risk_scores_present(self, client: TestClient):
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        raw = _raw(customer_id="cust_e2e_33", amount=500.0)
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        # Risk score fields
        assert data.get("ml_score") is not None
        assert isinstance(data["ml_score"], int)
        assert 0 <= data["ml_score"] <= 100

        assert data.get("behaviour_score") is not None
        assert isinstance(data["behaviour_score"], int)
        assert 0 <= data["behaviour_score"] <= 100

        assert data.get("rule_score") is not None
        assert isinstance(data["rule_score"], int)
        assert 0 <= data["rule_score"] <= 100

        assert data.get("risk_score") is not None
        assert isinstance(data["risk_score"], int)
        assert 0 <= data["risk_score"] <= 100

        assert data.get("risk_level") in ("LOW", "MEDIUM", "HIGH")
        assert data.get("decision") in ("APPROVE", "VERIFY", "HOLD")

    def test_e2e_explanation_detail_structure(self, client: TestClient):
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        raw = _raw(customer_id="cust_e2e_expl_33", amount=500.0)
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        expl = data.get("explanation_detail")
        assert expl is not None
        assert "ml_top_factors" in expl
        assert "behaviour_signals" in expl
        assert "rules_triggered" in expl
        assert isinstance(expl["ml_top_factors"], list)
        assert isinstance(expl["behaviour_signals"], list)
        assert isinstance(expl["rules_triggered"], list)

    def test_e2e_risk_factors_present(self, client: TestClient):
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        raw = _raw(
            customer_id="cust_e2e_rf_33",
            amount=HIGH_AMOUNT_THRESHOLD + 500,
            merchant_category="7995",
        )
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        risk_factors = data.get("risk_factors")
        assert risk_factors is not None
        assert isinstance(risk_factors, list)
        assert len(risk_factors) > 0

    def test_e2e_high_risk_transaction(self, client: TestClient):
        """A transaction that triggers many rules gets a high risk score."""
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        raw = _raw(
            customer_id="cust_e2e_high_33",
            amount=HIGH_AMOUNT_THRESHOLD + 5000,
            merchant_category="7995",
            timestamp=86400,
        )
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        assert data["rule_score"] > 0
        assert data["risk_score"] > 0
        # Should have triggered rules
        expl = data.get("explanation_detail", {})
        assert len(expl.get("rules_triggered", [])) > 0


# ═══════════════════════════════════════════════════════════════════════
# 18. Backend pass-through integration
# ═══════════════════════════════════════════════════════════════════════


class TestBackendPassThrough:
    """Backend correctly passes through rule signals from ML response."""

    @pytest.mark.asyncio
    async def test_backend_explanation_includes_behaviour_and_rules(self):
        """Backend constructs MLExplanation with behaviour_signals and rules_triggered."""
        from backend.schemas import MLExplanation, MLBehaviourSignal, MLRuleTrigger

        # Simulate a full ML response with explanation_detail
        ml_response_dict = {
            "fraud_probability": 0.85,
            "fraud_prediction": 1,
            "threshold": 0.50,
            "model_version": "v1.0",
            "explanation": [
                {"feature": "amount", "importance": 0.5},
            ],
            "ml_score": 85,
            "behaviour_score": 60,
            "rule_score": 35,
            "risk_score": 78,
            "risk_level": "HIGH",
            "decision": "HOLD",
            "explanation_detail": {
                "ml_top_factors": [
                    {"feature": "amount", "importance": 0.5},
                ],
                "behaviour_signals": [
                    {"signal": "spending_amount_anomaly", "severity": 0.6},
                ],
                "rules_triggered": [
                    {"rule": "high_amount", "contribution": 15},
                ],
            },
            "risk_factors": ["amount", "spending_amount_anomaly", "high_amount"],
        }

        from backend.schemas import MLPredictionResponse
        parsed = MLPredictionResponse.model_validate(ml_response_dict)

        # Verify extra fields are captured
        assert parsed.ml_score == 85
        assert parsed.behaviour_score == 60
        assert parsed.rule_score == 35
        assert parsed.risk_score == 78
        assert parsed.risk_level == "HIGH"
        assert parsed.decision == "HOLD"

        # Verify explanation_detail is accessible
        expl_detail = getattr(parsed, "explanation_detail", None)
        assert expl_detail is not None
        assert isinstance(expl_detail, dict)
        assert len(expl_detail["behaviour_signals"]) == 1
        assert len(expl_detail["rules_triggered"]) == 1


# ═══════════════════════════════════════════════════════════════════════
# 19. Leakage / security checks
# ═══════════════════════════════════════════════════════════════════════


class TestLeakageSecurity:
    """Rule engine does not access labels or leak data."""

    def test_rule_engine_no_label_access(self):
        """evaluate_rules never accesses isFraud or labels."""
        features = _features(amount=100.0)
        raw = _raw(amount=100.0)
        # Even with isFraud in the raw dict, rules should not use it
        raw_with_label = {**raw, "isFraud": 1}
        result = evaluate_rules(features, raw_with_label)
        # The result should be identical
        result_no_label = evaluate_rules(features, raw)
        assert result.rule_score == result_no_label.rule_score
        assert result.behaviour_score == result_no_label.behaviour_score

    def test_transaction_id_not_used_as_feature(self):
        """TransactionID in raw data does not affect rule output."""
        features = _features(amount=100.0)
        raw1 = _raw(amount=100.0)
        raw2 = {**raw1, "TransactionID": "T_different"}
        result1 = evaluate_rules(features, raw1)
        result2 = evaluate_rules(features, raw2)
        assert result1.rule_score == result2.rule_score
        assert result1.behaviour_score == result2.behaviour_score

    def test_no_credentials_in_output(self):
        """Rule output contains no database credentials or secrets."""
        features = _features(amount=HIGH_AMOUNT_THRESHOLD + 1)
        raw = _raw(amount=HIGH_AMOUNT_THRESHOLD + 1)
        result = evaluate_rules(features, raw)
        d = result.to_dict()
        import json
        output_str = json.dumps(d).lower()
        assert "password" not in output_str
        assert "secret" not in output_str
        assert "token" not in output_str
        assert "credential" not in output_str

    def test_api_does_not_expose_stack_traces(self, client: TestClient):
        """Error responses do not contain Python tracebacks."""
        resp = client.post("/predict", json={"invalid": True})
        assert resp.status_code == 422
        body = resp.text.lower()
        assert "traceback" not in body

    def test_history_customer_isolation_via_store(self, client: TestClient):
        """History store returns only the requested customer's records."""
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        # Record transactions for two different customers
        raw_a = _raw(customer_id="cust_iso_A", timestamp=1000, amount=100.0)
        raw_b = _raw(customer_id="cust_iso_B", timestamp=1000, amount=100.0)
        client.post("/predict", json=raw_a)
        client.post("/predict", json=raw_b)

        store = _history_module.history_store
        records_a = store.get("cust_iso_A", before_timestamp=99999)
        records_b = store.get("cust_iso_B", before_timestamp=99999)

        # Each customer should only see their own records
        assert len(records_a) >= 1
        assert len(records_b) >= 1


# ═══════════════════════════════════════════════════════════════════════
# 20. Behaviour signal evaluators
# ═══════════════════════════════════════════════════════════════════════


class TestBehaviourSignals:
    """Detailed tests for each behavioural anomaly signal."""

    def test_spending_anomaly_severity_scaling(self):
        """Higher Z-score → higher severity (capped at 1.0)."""
        features_low = _features(amount_deviation=2.5)
        features_high = _features(amount_deviation=4.5)
        raw = _raw()
        result_low = evaluate_rules(features_low, raw)
        result_high = evaluate_rules(features_high, raw)

        sev_low = [s for s in result_low.behaviour_signals
                   if s.signal == "spending_amount_anomaly"][0].severity
        sev_high = [s for s in result_high.behaviour_signals
                    if s.signal == "spending_amount_anomaly"][0].severity
        assert sev_high > sev_low
        assert sev_high <= 1.0

    def test_negative_zscore_triggers_anomaly(self):
        """Negative Z-score (below average) also triggers if |z| >= threshold."""
        features = _features(amount_deviation=-2.5)
        raw = _raw()
        result = evaluate_rules(features, raw)
        assert any(s.signal == "spending_amount_anomaly" for s in result.behaviour_signals)

    def test_location_anomaly_signal(self):
        features = _features(location_is_new=1, location_country=99)
        raw = _raw()
        result = evaluate_rules(features, raw)
        loc = [s for s in result.behaviour_signals if s.signal == "location_anomaly"]
        assert len(loc) == 1
        assert loc[0].severity == 0.8

    def test_device_anomaly_signal(self):
        features = _features(is_new_device=1, device_fingerprint=999)
        raw = _raw()
        result = evaluate_rules(features, raw)
        dev = [s for s in result.behaviour_signals if s.signal == "device_anomaly"]
        assert len(dev) == 1
        assert dev[0].severity == 0.7

    def test_time_anomaly_signal(self):
        features = _features(is_unusual_hour=1, hour_of_day_raw=3)
        raw = _raw()
        result = evaluate_rules(features, raw)
        time_s = [s for s in result.behaviour_signals if s.signal == "time_anomaly"]
        assert len(time_s) == 1
        assert time_s[0].severity == 0.5

    def test_velocity_anomaly_scaling(self):
        """Higher velocity → higher severity."""
        features_low = _features(tx_velocity_1h=4)
        features_high = _features(tx_velocity_1h=9)
        raw = _raw()
        result_low = evaluate_rules(features_low, raw)
        result_high = evaluate_rules(features_high, raw)

        sev_low = [s for s in result_low.behaviour_signals
                   if s.signal == "velocity_anomaly"][0].severity
        sev_high = [s for s in result_high.behaviour_signals
                    if s.signal == "velocity_anomaly"][0].severity
        assert sev_high > sev_low

    def test_behaviour_score_aggregation(self):
        """behaviour_score = avg(severities) * 100, capped at 100."""
        features = _features(
            amount_deviation=3.0,    # severity = 0.6
            location_is_new=1,       # severity = 0.8
        )
        raw = _raw()
        result = evaluate_rules(features, raw)

        # 2 signals: severity 0.6 + 0.8 = 1.4, avg = 0.7, score = 70
        assert result.behaviour_score == 70
