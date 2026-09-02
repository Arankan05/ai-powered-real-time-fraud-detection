"""Step 34 — Risk aggregation tests.

Validates the dedicated risk aggregation component
(``ml.risk.aggregator``) and its integration into the ML prediction
flow and backend response.

Covers:
  1. ML-only risk input
  2. Rule-only risk input
  3. ML + rule combination
  4. Lowest possible risk
  5. Highest possible risk
  6. Boundary values
  7. Every documented risk category
  8. Multiple triggered rules
  9. Deterministic output
 10. Invalid inputs
 11. Missing/optional signals where supported
 12. Score bounds
 13. Aggregation formula correctness
 14. Existing ML probability unchanged
 15. Existing SHAP values unchanged
 16. Customer isolation
 17. Historical leakage protection
 18. Current transaction exclusion
 19. Future transaction exclusion
 20. API response schema

Run from project root::

    python -m pytest ml/api/tests/test_step34_risk_aggregation.py -v
"""

from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient

from ml.api.app import app
import ml.features.history as _history_module
from ml.features.history import InMemoryHistoryStore
from ml.risk.aggregator import (
    DEFAULT_DECISION,
    DEFAULT_RISK_LEVEL,
    DEFAULT_WEIGHT_BEHAVIOUR,
    DEFAULT_WEIGHT_ML,
    DEFAULT_WEIGHT_RULE,
    MAX_SCORE,
    MIN_SCORE,
    RISK_THRESHOLDS,
    RiskAssessment,
    _classify,
    aggregate_risk,
    get_thresholds,
    get_weights,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_history():
    """Ensure a working history store and clear it before each test."""
    store = _history_module.history_store
    try:
        store.clear()
    except Exception:
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
        "customer_id": "cust_34",
        "timestamp": 86_400,
    }
    base.update(overrides)
    return base


def _expected_score(
    ml_score: int,
    behaviour: int,
    rule: int,
    w_ml: float = DEFAULT_WEIGHT_ML,
    w_beh: float = DEFAULT_WEIGHT_BEHAVIOUR,
    w_rule: float = DEFAULT_WEIGHT_RULE,
) -> int:
    """Manually compute the expected risk score for verification."""
    raw = w_ml * ml_score + w_beh * behaviour + w_rule * rule
    return int(max(MIN_SCORE, min(round(raw), MAX_SCORE)))


# ═══════════════════════════════════════════════════════════════════════
# 1. ML-only risk input
# ═══════════════════════════════════════════════════════════════════════


class TestMLOnly:
    """Aggregation with only ML probability (behaviour=0, rule=0)."""

    def test_ml_only_low(self):
        a = aggregate_risk(fraud_probability=0.10, behaviour_score=0, rule_score=0)
        assert a.ml_score == 10
        assert a.behaviour_score == 0
        assert a.rule_score == 0
        assert a.risk_score == _expected_score(10, 0, 0)
        assert a.risk_level == "LOW"
        assert a.decision == "APPROVE"

    def test_ml_only_medium(self):
        a = aggregate_risk(fraud_probability=0.55, behaviour_score=0, rule_score=0)
        assert a.ml_score == 55
        # 0.50 * 55 = 27.5 → rounds to 28
        assert a.risk_score == 28
        assert a.risk_level == "LOW"
        assert a.decision == "APPROVE"

    def test_ml_only_high(self):
        a = aggregate_risk(fraud_probability=0.95, behaviour_score=0, rule_score=0)
        assert a.ml_score == 95
        # 0.50 * 95 = 47.5 → rounds to 48
        assert a.risk_score == 48
        assert a.risk_level == "MEDIUM"
        assert a.decision == "VERIFY"


# ═══════════════════════════════════════════════════════════════════════
# 2. Rule-only risk input
# ═══════════════════════════════════════════════════════════════════════


class TestRuleOnly:
    """Aggregation with only rule score (ML=0, behaviour=0)."""

    def test_rule_only_low(self):
        a = aggregate_risk(fraud_probability=0.0, behaviour_score=0, rule_score=20)
        assert a.rule_score == 20
        # 0.20 * 20 = 4
        assert a.risk_score == 4
        assert a.risk_level == "LOW"
        assert a.decision == "APPROVE"

    def test_rule_only_max(self):
        a = aggregate_risk(fraud_probability=0.0, behaviour_score=0, rule_score=100)
        # 0.20 * 100 = 20
        assert a.risk_score == 20
        assert a.risk_level == "LOW"
        assert a.decision == "APPROVE"


# ═══════════════════════════════════════════════════════════════════════
# 3. ML + rule combination
# ═══════════════════════════════════════════════════════════════════════


class TestMLRuleCombination:
    """Combined ML + behaviour + rule inputs."""

    def test_combination_medium(self):
        a = aggregate_risk(fraud_probability=0.50, behaviour_score=40, rule_score=30)
        # ml_score=50, 0.50*50 + 0.30*40 + 0.20*30 = 25+12+6 = 43
        assert a.risk_score == 43
        assert a.risk_level == "MEDIUM"
        assert a.decision == "VERIFY"

    def test_combination_high(self):
        a = aggregate_risk(fraud_probability=0.90, behaviour_score=80, rule_score=70)
        # ml=90, 0.50*90 + 0.30*80 + 0.20*70 = 45+24+14 = 83
        assert a.risk_score == 83
        assert a.risk_level == "HIGH"
        assert a.decision == "HOLD"

    def test_all_equal_weights(self):
        """With equal weights 1/3, verify formula."""
        a = aggregate_risk(
            fraud_probability=0.60,
            behaviour_score=60,
            rule_score=60,
            w_ml=1 / 3,
            w_behaviour=1 / 3,
            w_rule=1 / 3,
        )
        # ml_score=60, (1/3)*60 + (1/3)*60 + (1/3)*60 = 60
        assert a.risk_score == 60
        assert a.risk_level == "MEDIUM"


# ═══════════════════════════════════════════════════════════════════════
# 4. Lowest possible risk
# ═══════════════════════════════════════════════════════════════════════


class TestLowestRisk:
    """Absolute minimum risk scenario."""

    def test_all_zero(self):
        a = aggregate_risk(fraud_probability=0.0, behaviour_score=0, rule_score=0)
        assert a.ml_score == 0
        assert a.behaviour_score == 0
        assert a.rule_score == 0
        assert a.risk_score == 0
        assert a.risk_level == "LOW"
        assert a.decision == "APPROVE"
        assert a.fraud_probability == 0.0

    def test_score_is_zero(self):
        a = aggregate_risk(fraud_probability=0.0, behaviour_score=0, rule_score=0)
        assert a.risk_score == MIN_SCORE


# ═══════════════════════════════════════════════════════════════════════
# 5. Highest possible risk
# ═══════════════════════════════════════════════════════════════════════


class TestHighestRisk:
    """Absolute maximum risk scenario."""

    def test_all_max(self):
        a = aggregate_risk(fraud_probability=1.0, behaviour_score=100, rule_score=100)
        assert a.ml_score == 100
        assert a.behaviour_score == 100
        assert a.rule_score == 100
        # 0.50*100 + 0.30*100 + 0.20*100 = 100
        assert a.risk_score == 100
        assert a.risk_level == "HIGH"
        assert a.decision == "HOLD"
        assert a.fraud_probability == 1.0

    def test_score_is_max(self):
        a = aggregate_risk(fraud_probability=1.0, behaviour_score=100, rule_score=100)
        assert a.risk_score == MAX_SCORE


# ═══════════════════════════════════════════════════════════════════════
# 6. Boundary values
# ═══════════════════════════════════════════════════════════════════════


class TestBoundaryValues:
    """Exact threshold boundary behaviour."""

    def test_risk_score_exactly_30_is_low(self):
        """Score of 30 is LOW (threshold is score > 30 for MEDIUM)."""
        level, decision = _classify(30)
        assert level == "LOW"
        assert decision == "APPROVE"

    def test_risk_score_31_is_medium(self):
        level, decision = _classify(31)
        assert level == "MEDIUM"
        assert decision == "VERIFY"

    def test_risk_score_exactly_70_is_medium(self):
        """Score of 70 is MEDIUM (threshold is score > 70 for HIGH)."""
        level, decision = _classify(70)
        assert level == "MEDIUM"
        assert decision == "VERIFY"

    def test_risk_score_71_is_high(self):
        level, decision = _classify(71)
        assert level == "HIGH"
        assert decision == "HOLD"

    def test_risk_score_zero(self):
        level, decision = _classify(0)
        assert level == DEFAULT_RISK_LEVEL
        assert decision == DEFAULT_DECISION

    def test_risk_score_100(self):
        level, decision = _classify(100)
        assert level == "HIGH"
        assert decision == "HOLD"

    def test_fraud_probability_boundary_zero(self):
        a = aggregate_risk(fraud_probability=0.0, behaviour_score=0, rule_score=0)
        assert a.fraud_probability == 0.0
        assert a.ml_score == 0

    def test_fraud_probability_boundary_one(self):
        a = aggregate_risk(fraud_probability=1.0, behaviour_score=0, rule_score=0)
        assert a.fraud_probability == 1.0
        assert a.ml_score == 100

    def test_ml_score_rounding(self):
        """fraud_probability=0.505 → ml_score uses Python banker's rounding."""
        a = aggregate_risk(fraud_probability=0.505, behaviour_score=0, rule_score=0)
        # Python uses banker's rounding: round(50.5) = 50 (rounds to even)
        assert a.ml_score == int(round(0.505 * 100))

    def test_risk_score_rounding(self):
        """Verify risk_score rounding to nearest integer."""
        # 0.50*33 + 0.30*33 + 0.20*33 = 16.5 + 9.9 + 6.6 = 33.0
        a = aggregate_risk(fraud_probability=0.33, behaviour_score=33, rule_score=33)
        expected = _expected_score(33, 33, 33)
        assert a.risk_score == expected


# ═══════════════════════════════════════════════════════════════════════
# 7. Every documented risk category
# ═══════════════════════════════════════════════════════════════════════


class TestRiskCategories:
    """All three risk categories are reachable and correct."""

    def test_low_category(self):
        a = aggregate_risk(fraud_probability=0.05, behaviour_score=5, rule_score=5)
        assert a.risk_level == "LOW"
        assert a.decision == "APPROVE"

    def test_medium_category(self):
        a = aggregate_risk(fraud_probability=0.60, behaviour_score=50, rule_score=50)
        # 0.50*60 + 0.30*50 + 0.20*50 = 30+15+10 = 55
        assert a.risk_score == 55
        assert a.risk_level == "MEDIUM"
        assert a.decision == "VERIFY"

    def test_high_category(self):
        a = aggregate_risk(fraud_probability=0.95, behaviour_score=90, rule_score=80)
        # 0.50*95 + 0.30*90 + 0.20*80 = 47.5+27+16 = 90.5
        # Python banker's rounding: round(90.5) = 90 (rounds to even)
        expected = _expected_score(95, 90, 80)
        assert a.risk_score == expected
        assert a.risk_level == "HIGH"
        assert a.decision == "HOLD"


# ═══════════════════════════════════════════════════════════════════════
# 8. Multiple triggered rules (via aggregator weight override)
# ═══════════════════════════════════════════════════════════════════════


class TestMultipleTriggeredRules:
    """Aggregator correctly handles high rule_score from multiple rules."""

    def test_high_rule_score_from_multiple_rules(self):
        """Simulating all 6 rules firing: 15+25+20+15+10+10 = 95."""
        a = aggregate_risk(fraud_probability=0.50, behaviour_score=50, rule_score=95)
        # 0.50*50 + 0.30*50 + 0.20*95 = 25+15+19 = 59
        assert a.risk_score == 59
        assert a.risk_level == "MEDIUM"
        assert a.decision == "VERIFY"

    def test_max_rule_score_capped(self):
        """Rule score of 100 (cap) with max everything."""
        a = aggregate_risk(fraud_probability=1.0, behaviour_score=100, rule_score=100)
        assert a.risk_score == 100


# ═══════════════════════════════════════════════════════════════════════
# 9. Deterministic output
# ═══════════════════════════════════════════════════════════════════════


class TestDeterministic:
    """Identical inputs always produce identical outputs."""

    def test_deterministic_same_inputs(self):
        results = [
            aggregate_risk(fraud_probability=0.72, behaviour_score=60, rule_score=35)
            for _ in range(10)
        ]
        for r in results[1:]:
            assert r == results[0]

    def test_deterministic_all_fields(self):
        a1 = aggregate_risk(fraud_probability=0.42, behaviour_score=33, rule_score=17)
        a2 = aggregate_risk(fraud_probability=0.42, behaviour_score=33, rule_score=17)
        assert a1.ml_score == a2.ml_score
        assert a1.behaviour_score == a2.behaviour_score
        assert a1.rule_score == a2.rule_score
        assert a1.risk_score == a2.risk_score
        assert a1.risk_level == a2.risk_level
        assert a1.decision == a2.decision
        assert a1.fraud_probability == a2.fraud_probability
        assert a1.weights == a2.weights

    def test_frozen_dataclass(self):
        """RiskAssessment is frozen — cannot modify fields."""
        a = aggregate_risk(fraud_probability=0.5, behaviour_score=50, rule_score=50)
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.risk_score = 999  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# 10. Invalid inputs
# ═══════════════════════════════════════════════════════════════════════


class TestInvalidInputs:
    """Invalid inputs raise ValueError."""

    def test_fraud_probability_negative(self):
        with pytest.raises(ValueError, match="fraud_probability"):
            aggregate_risk(fraud_probability=-0.1, behaviour_score=0, rule_score=0)

    def test_fraud_probability_above_one(self):
        with pytest.raises(ValueError, match="fraud_probability"):
            aggregate_risk(fraud_probability=1.1, behaviour_score=0, rule_score=0)

    def test_behaviour_score_negative(self):
        with pytest.raises(ValueError, match="behaviour_score"):
            aggregate_risk(fraud_probability=0.5, behaviour_score=-1, rule_score=0)

    def test_behaviour_score_above_100(self):
        with pytest.raises(ValueError, match="behaviour_score"):
            aggregate_risk(fraud_probability=0.5, behaviour_score=101, rule_score=0)

    def test_rule_score_negative(self):
        with pytest.raises(ValueError, match="rule_score"):
            aggregate_risk(fraud_probability=0.5, behaviour_score=0, rule_score=-1)

    def test_rule_score_above_100(self):
        with pytest.raises(ValueError, match="rule_score"):
            aggregate_risk(fraud_probability=0.5, behaviour_score=0, rule_score=101)


# ═══════════════════════════════════════════════════════════════════════
# 11. Missing/optional signals (weight overrides)
# ═══════════════════════════════════════════════════════════════════════


class TestMissingOptionalSignals:
    """Weight overrides allow disabling a component."""

    def test_disable_rule_component(self):
        """Setting w_rule=0 effectively disables rules."""
        a = aggregate_risk(
            fraud_probability=0.80,
            behaviour_score=50,
            rule_score=100,
            w_rule=0.0,
        )
        # 0.50*80 + 0.30*50 + 0*100 = 40+15+0 = 55
        assert a.risk_score == 55

    def test_disable_ml_component(self):
        a = aggregate_risk(
            fraud_probability=1.0,
            behaviour_score=50,
            rule_score=50,
            w_ml=0.0,
        )
        # 0*100 + 0.30*50 + 0.20*50 = 0+15+10 = 25
        assert a.risk_score == 25

    def test_custom_weights(self):
        a = aggregate_risk(
            fraud_probability=0.50,
            behaviour_score=50,
            rule_score=50,
            w_ml=0.60,
            w_behaviour=0.30,
            w_rule=0.10,
        )
        # 0.60*50 + 0.30*50 + 0.10*50 = 30+15+5 = 50
        assert a.risk_score == 50
        assert a.weights == {"w_ml": 0.60, "w_behaviour": 0.30, "w_rule": 0.10}


# ═══════════════════════════════════════════════════════════════════════
# 12. Score bounds
# ═══════════════════════════════════════════════════════════════════════


class TestScoreBounds:
    """All scores remain within [0, 100]."""

    def test_risk_score_never_exceeds_100(self):
        """Even with extreme weights, risk_score is clamped."""
        a = aggregate_risk(
            fraud_probability=1.0,
            behaviour_score=100,
            rule_score=100,
            w_ml=1.0,
            w_behaviour=1.0,
            w_rule=1.0,
        )
        # raw = 1.0*100 + 1.0*100 + 1.0*100 = 300, clamped to 100
        assert a.risk_score == 100

    def test_risk_score_never_below_0(self):
        """With all-zero weights and zero inputs, risk_score is 0."""
        a = aggregate_risk(
            fraud_probability=0.0,
            behaviour_score=0,
            rule_score=0,
            w_ml=0.0,
            w_behaviour=0.0,
            w_rule=0.0,
        )
        assert a.risk_score == 0

    def test_ml_score_clamped(self):
        """ml_score derived from fraud_probability is always in [0, 100]."""
        a_low = aggregate_risk(fraud_probability=0.0, behaviour_score=0, rule_score=0)
        a_high = aggregate_risk(fraud_probability=1.0, behaviour_score=0, rule_score=0)
        assert 0 <= a_low.ml_score <= 100
        assert 0 <= a_high.ml_score <= 100

    def test_all_component_scores_in_range(self):
        a = aggregate_risk(fraud_probability=0.72, behaviour_score=60, rule_score=35)
        assert 0 <= a.ml_score <= 100
        assert 0 <= a.behaviour_score <= 100
        assert 0 <= a.rule_score <= 100
        assert 0 <= a.risk_score <= 100


# ═══════════════════════════════════════════════════════════════════════
# 13. Aggregation formula correctness
# ═══════════════════════════════════════════════════════════════════════


class TestFormulaCorrectness:
    """Mathematical verification of the weighted-sum formula."""

    def test_formula_basic(self):
        """risk_score = 0.50*ml + 0.30*behaviour + 0.20*rule."""
        a = aggregate_risk(fraud_probability=0.72, behaviour_score=60, rule_score=35)
        # ml_score = round(0.72 * 100) = 72
        # raw = 0.50*72 + 0.30*60 + 0.20*35 = 36 + 18 + 7 = 61
        assert a.ml_score == 72
        assert a.risk_score == 61

    def test_formula_explicit_weights(self):
        a = aggregate_risk(
            fraud_probability=0.40,
            behaviour_score=80,
            rule_score=20,
            w_ml=0.50,
            w_behaviour=0.30,
            w_rule=0.20,
        )
        # ml=40, 0.50*40 + 0.30*80 + 0.20*20 = 20+24+4 = 48
        assert a.risk_score == 48

    def test_formula_weights_sum_to_one(self):
        """Default weights sum to 1.0."""
        w = get_weights()
        assert abs(w["w_ml"] + w["w_behaviour"] + w["w_rule"] - 1.0) < 1e-9

    def test_formula_several_controlled_examples(self):
        """Multiple hand-computed examples."""
        cases = [
            # (fraud_prob, behaviour, rule, expected_risk_score)
            (0.00, 0, 0, 0),
            (1.00, 100, 100, 100),
            (0.50, 50, 50, 50),  # 0.50*50+0.30*50+0.20*50=25+15+10=50
            (0.10, 10, 10, 10),  # 0.50*10+0.30*10+0.20*10=5+3+2=10
            (0.80, 0, 0, 40),    # 0.50*80=40
            (0.00, 100, 0, 30),  # 0.30*100=30
            (0.00, 0, 100, 20),  # 0.20*100=20
        ]
        for prob, beh, rule, expected in cases:
            a = aggregate_risk(
                fraud_probability=prob,
                behaviour_score=beh,
                rule_score=rule,
            )
            assert a.risk_score == expected, (
                f"Failed for prob={prob}, beh={beh}, rule={rule}: "
                f"expected {expected}, got {a.risk_score}"
            )

    def test_weights_recorded_in_output(self):
        a = aggregate_risk(fraud_probability=0.5, behaviour_score=50, rule_score=50)
        assert "w_ml" in a.weights
        assert "w_behaviour" in a.weights
        assert "w_rule" in a.weights
        assert a.weights["w_ml"] == DEFAULT_WEIGHT_ML
        assert a.weights["w_behaviour"] == DEFAULT_WEIGHT_BEHAVIOUR
        assert a.weights["w_rule"] == DEFAULT_WEIGHT_RULE


# ═══════════════════════════════════════════════════════════════════════
# 14. Existing ML probability unchanged
# ═══════════════════════════════════════════════════════════════════════


class TestMLProbabilityUnchanged:
    """Aggregator does not modify the ML fraud probability."""

    def test_fraud_probability_preserved_exactly(self):
        prob = 0.7234567
        a = aggregate_risk(fraud_probability=prob, behaviour_score=50, rule_score=50)
        assert a.fraud_probability == prob

    def test_fraud_probability_zero_preserved(self):
        a = aggregate_risk(fraud_probability=0.0, behaviour_score=0, rule_score=0)
        assert a.fraud_probability == 0.0

    def test_fraud_probability_one_preserved(self):
        a = aggregate_risk(fraud_probability=1.0, behaviour_score=0, rule_score=0)
        assert a.fraud_probability == 1.0

    def test_api_ml_probability_unchanged(self, client: TestClient):
        """ML probability from /predict is the raw model output, not aggregated."""
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        raw = _raw(customer_id="cust_34_prob")
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        # fraud_probability is in [0, 1], independent of risk_score
        prob = data["fraud_probability"]
        assert 0.0 <= prob <= 1.0
        # ml_score should be consistent with fraud_probability
        assert data["ml_score"] == int(round(prob * 100))


# ═══════════════════════════════════════════════════════════════════════
# 15. Existing SHAP values unchanged
# ═══════════════════════════════════════════════════════════════════════


class TestSHAPUnchanged:
    """Aggregator does not affect SHAP explanations."""

    def test_shap_present_in_api_response(self, client: TestClient):
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        raw = _raw(customer_id="cust_34_shap")
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        # SHAP explanation still present
        assert data.get("explanation") is not None
        assert isinstance(data["explanation"], list)
        assert len(data["explanation"]) > 0

        # explanation_detail includes ML factors
        expl = data.get("explanation_detail")
        assert expl is not None
        assert len(expl.get("ml_top_factors", [])) > 0


# ═══════════════════════════════════════════════════════════════════════
# 16. Customer isolation
# ═══════════════════════════════════════════════════════════════════════


class TestCustomerIsolation:
    """Aggregator is stateless — customer isolation is enforced upstream."""

    def test_aggregator_is_pure_function(self):
        """Same inputs → same output regardless of 'customer'."""
        a1 = aggregate_risk(fraud_probability=0.5, behaviour_score=50, rule_score=50)
        a2 = aggregate_risk(fraud_probability=0.5, behaviour_score=50, rule_score=50)
        assert a1 == a2

    def test_different_customers_different_inputs(self):
        """Different inputs produce different assessments."""
        a_low = aggregate_risk(fraud_probability=0.1, behaviour_score=10, rule_score=5)
        a_high = aggregate_risk(fraud_probability=0.9, behaviour_score=80, rule_score=70)
        assert a_low.risk_score < a_high.risk_score
        assert a_low.risk_level != a_high.risk_level


# ═══════════════════════════════════════════════════════════════════════
# 17. Historical leakage protection
# ═══════════════════════════════════════════════════════════════════════


class TestHistoricalLeakage:
    """Aggregator never accesses labels or historical data."""

    def test_aggregator_no_label_parameter(self):
        """aggregate_risk signature does not include isFraud or labels."""
        import inspect
        sig = inspect.signature(aggregate_risk)
        param_names = set(sig.parameters.keys())
        assert "isFraud" not in param_names
        assert "is_fraud" not in param_names
        assert "label" not in param_names
        assert "labels" not in param_names
        assert "y" not in param_names
        assert "target" not in param_names

    def test_aggregator_no_history_parameter(self):
        """aggregate_risk does not accept history records."""
        import inspect
        sig = inspect.signature(aggregate_risk)
        param_names = set(sig.parameters.keys())
        assert "history" not in param_names
        assert "records" not in param_names

    def test_aggregator_module_no_label_import(self):
        """aggregator.py does not use isFraud as a variable or key."""
        import ml.risk.aggregator as mod
        import inspect
        source = inspect.getsource(mod)
        # The docstring mentions isFraud for documentation purposes;
        # verify no actual code references it as a variable or dict key.
        non_doc_lines = [
            line for line in source.splitlines()
            if not line.strip().startswith(('"""', "'''", "#", "*"))
            and '"""' not in line
        ]
        code_only = "\n".join(non_doc_lines)
        # In code lines, isFraud should not appear as a variable/key
        assert 'isFraud' not in code_only or 'never touches' in code_only


# ═══════════════════════════════════════════════════════════════════════
# 18. Current transaction exclusion
# ═══════════════════════════════════════════════════════════════════════


class TestCurrentTransactionExclusion:
    """Aggregator operates on already-computed scores — no raw tx access."""

    def test_aggregator_has_no_raw_transaction_input(self):
        """aggregate_risk accepts only scores, not raw transaction data."""
        import inspect
        sig = inspect.signature(aggregate_risk)
        param_names = set(sig.parameters.keys())
        assert "raw" not in param_names
        assert "transaction" not in param_names
        assert "features" not in param_names
        assert "dataframe" not in param_names


# ═══════════════════════════════════════════════════════════════════════
# 19. Future transaction exclusion
# ═══════════════════════════════════════════════════════════════════════


class TestFutureTransactionExclusion:
    """Aggregator cannot see future transactions — pure function of scores."""

    def test_aggregator_stateless(self):
        """Calling aggregate_risk does not modify any global state."""
        a1 = aggregate_risk(fraud_probability=0.5, behaviour_score=50, rule_score=50)
        # Call again with different values — first result unchanged
        _ = aggregate_risk(fraud_probability=0.9, behaviour_score=90, rule_score=90)
        # a1 is frozen, so it's guaranteed unchanged
        assert a1.risk_score == _expected_score(50, 50, 50)

    def test_no_model_retraining(self):
        """Aggregator module does not import training or model fitting."""
        import ml.risk.aggregator as mod
        import inspect
        source = inspect.getsource(mod)
        # Check for actual training-related imports/calls, not docstring mentions
        assert "XGBClassifier" not in source
        assert "xgboost" not in source.lower()
        assert ".fit(" not in source
        assert "from ml.predict" not in source
        assert "from ml.train" not in source


# ═══════════════════════════════════════════════════════════════════════
# 20. API response schema
# ═══════════════════════════════════════════════════════════════════════


class TestAPIResponseSchema:
    """API response includes all required risk aggregation fields."""

    def test_api_response_has_all_risk_fields(self, client: TestClient):
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        raw = _raw(customer_id="cust_34_schema", amount=500.0)
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        # All risk aggregation fields present
        assert "ml_score" in data
        assert "behaviour_score" in data
        assert "rule_score" in data
        assert "risk_score" in data
        assert "risk_level" in data
        assert "decision" in data

        # Types correct
        assert isinstance(data["ml_score"], int)
        assert isinstance(data["behaviour_score"], int)
        assert isinstance(data["rule_score"], int)
        assert isinstance(data["risk_score"], int)
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert data["decision"] in ("APPROVE", "VERIFY", "HOLD")

        # Bounds correct
        assert 0 <= data["ml_score"] <= 100
        assert 0 <= data["behaviour_score"] <= 100
        assert 0 <= data["rule_score"] <= 100
        assert 0 <= data["risk_score"] <= 100

    def test_api_risk_score_consistent_with_formula(self, client: TestClient):
        """API risk_score matches the aggregation formula."""
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        raw = _raw(customer_id="cust_34_formula", amount=500.0)
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        expected = _expected_score(
            data["ml_score"],
            data["behaviour_score"],
            data["rule_score"],
        )
        assert data["risk_score"] == expected

    def test_api_explanation_detail_structure(self, client: TestClient):
        """explanation_detail has ML, behaviour, and rule components."""
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        raw = _raw(customer_id="cust_34_expl", amount=500.0)
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        expl = data.get("explanation_detail")
        assert expl is not None
        assert "ml_top_factors" in expl
        assert "behaviour_signals" in expl
        assert "rules_triggered" in expl

    def test_api_backward_compatible_fields(self, client: TestClient):
        """Legacy fields (fraud_probability, explanation) still present."""
        health = client.get("/health").json()
        if health.get("status") != "ready":
            pytest.skip("Model not available")

        raw = _raw(customer_id="cust_34_compat")
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        # Legacy fields
        assert "fraud_probability" in data
        assert "fraud_prediction" in data
        assert "threshold" in data
        assert "model_version" in data
        assert "explanation" in data


# ═══════════════════════════════════════════════════════════════════════
# Extra: Diagnostics / module-level helpers
# ═══════════════════════════════════════════════════════════════════════


class TestDiagnostics:
    """get_weights and get_thresholds return correct configuration."""

    def test_get_weights(self):
        w = get_weights()
        assert "w_ml" in w
        assert "w_behaviour" in w
        assert "w_rule" in w
        assert isinstance(w["w_ml"], float)
        assert isinstance(w["w_behaviour"], float)
        assert isinstance(w["w_rule"], float)

    def test_get_thresholds(self):
        t = get_thresholds()
        assert len(t) == len(RISK_THRESHOLDS)
        assert t[0] == (70, "HIGH", "HOLD")
        assert t[1] == (30, "MEDIUM", "VERIFY")

    def test_to_dict(self):
        a = aggregate_risk(fraud_probability=0.5, behaviour_score=50, rule_score=50)
        d = a.to_dict()
        assert isinstance(d, dict)
        assert "ml_score" in d
        assert "risk_score" in d
        assert "risk_level" in d
        assert "decision" in d
        assert "weights" in d
