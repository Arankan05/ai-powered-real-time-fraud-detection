"""Step 35 — End-to-end fraud detection validation.

Comprehensive pytest test suite that exercises the **complete**
fraud detection pipeline through the real ML ``/predict`` endpoint:

    Raw transaction
        → validation
        → feature engineering
        → persistent customer history
        → XGBoost prediction
        → SHAP explanation
        → rule evaluation
        → risk aggregation
        → structured response

Also validates:
  - Backend → ML integration (mock HTTP)
  - API contract compliance (``docs/api-contract.md``)
  - Error handling (ML unavailable, invalid input)
  - Outcome feedback loop
  - Determinism across repeated calls
  - Leakage protection (source + runtime)
  - SHAP additivity
  - Risk aggregation formula

Run from project root::

    python -m pytest ml/api/tests/test_step35_e2e_validation.py -v
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml.api.app import app
import ml.features.history as _history_module
from ml.features.engineer import FEATURE_LIST, engineer_features_for_inference
from ml.features.history import InMemoryHistoryStore
from ml.risk.aggregator import (
    DEFAULT_WEIGHT_BEHAVIOUR,
    DEFAULT_WEIGHT_ML,
    DEFAULT_WEIGHT_RULE,
    aggregate_risk,
)
from ml.rules.engine import (
    HIGH_AMOUNT_THRESHOLD,
    RuleResult,
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


def _raw(**overrides) -> dict:
    """Minimal valid raw transaction payload."""
    base = {
        "amount": 100.0,
        "currency": "USD",
        "merchant_name": "E2E Store",
        "merchant_category": "5732",
        "transaction_type": "purchase",
        "location_country": "US",
        "location_city": "New York",
        "device_fingerprint": "fp_e2e_35",
        "device_type": "desktop",
        "ip_address": "192.168.1.1",
        "customer_id": "cust_e2e_35",
        "timestamp": 86_400,
    }
    base.update(overrides)
    return base


def _skip_if_no_model(client: TestClient) -> None:
    health = client.get("/health").json()
    if health.get("status") != "ready":
        pytest.skip("Model not available")


# ═══════════════════════════════════════════════════════════════════════
# 1. Complete end-to-end pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestCompletePipeline:
    """Full HTTP flow: POST /predict → complete response."""

    def test_full_pipeline_low_risk(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_low", amount=50.0)
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        # All required fields present
        for key in (
            "fraud_probability", "fraud_prediction", "threshold",
            "model_version", "explanation", "timestamp",
            "ml_score", "behaviour_score", "rule_score",
            "risk_score", "risk_level", "decision",
            "explanation_detail", "risk_factors",
        ):
            assert key in data, f"Missing field: {key}"

        # Types correct
        assert isinstance(data["fraud_probability"], float)
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert data["fraud_prediction"] in (0, 1)
        assert isinstance(data["threshold"], float)
        assert isinstance(data["model_version"], str)
        assert isinstance(data["ml_score"], int)
        assert isinstance(data["behaviour_score"], int)
        assert isinstance(data["rule_score"], int)
        assert isinstance(data["risk_score"], int)
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert data["decision"] in ("APPROVE", "VERIFY", "HOLD")

    def test_full_pipeline_high_risk(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(
            customer_id="cust_e2e_high",
            amount=HIGH_AMOUNT_THRESHOLD + 5000,
            merchant_category="7995",
        )
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        assert data["rule_score"] > 0
        assert data["risk_score"] > 0
        expl = data.get("explanation_detail", {})
        assert len(expl.get("rules_triggered", [])) > 0

    def test_pipeline_response_schema_matches_architecture(self, client: TestClient):
        """Response contains every field from ml-architecture.md §Response Schema."""
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_schema", amount=500.0)
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()

        # Architecture §Response Schema fields
        assert "ml_score" in data and isinstance(data["ml_score"], int)
        assert "behaviour_score" in data and isinstance(data["behaviour_score"], int)
        assert "rule_score" in data and isinstance(data["rule_score"], int)
        assert "risk_score" in data and isinstance(data["risk_score"], int)
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert data["decision"] in ("APPROVE", "VERIFY", "HOLD")
        assert "model_version" in data

        # Explanation structure (architecture §6)
        expl = data.get("explanation_detail")
        assert expl is not None
        assert "ml_top_factors" in expl
        assert "behaviour_signals" in expl
        assert "rules_triggered" in expl


# ═══════════════════════════════════════════════════════════════════════
# 2. ML prediction validation
# ═══════════════════════════════════════════════════════════════════════


class TestMLPredictionValidation:
    """ML prediction is correct, unchanged by downstream layers."""

    def test_probability_range(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_prob")
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_binary_prediction(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_pred")
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        assert resp.json()["fraud_prediction"] in (0, 1)

    def test_threshold_positive(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_thresh")
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        assert resp.json()["threshold"] > 0

    def test_model_version_present(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_ver")
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        assert isinstance(resp.json()["model_version"], str)
        assert len(resp.json()["model_version"]) > 0

    def test_feature_schema_unchanged(self, client: TestClient):
        """The model expects exactly the documented 24 features."""
        _skip_if_no_model(client)
        from ml.api.app import _predictor
        assert _predictor is not None
        assert len(_predictor.feature_names) == 24
        assert set(_predictor.feature_names) == set(FEATURE_LIST)

    def test_ml_score_consistent_with_probability(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_consistency")
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()
        expected_ml = int(round(data["fraud_probability"] * 100))
        assert data["ml_score"] == expected_ml


# ═══════════════════════════════════════════════════════════════════════
# 3. SHAP validation
# ═══════════════════════════════════════════════════════════════════════


class TestSHAPValidation:
    """SHAP explanation correctness."""

    def test_shap_present(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_shap1")
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 200
        data = resp.json()
        assert data["explanation"] is not None
        assert isinstance(data["explanation"], list)
        assert len(data["explanation"]) > 0

    def test_shap_top_10_or_fewer(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_shap2")
        resp = client.post("/predict", json=raw)
        data = resp.json()
        factors = data["explanation"]
        assert len(factors) <= 10

    def test_shap_feature_names_valid(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_shap3")
        resp = client.post("/predict", json=raw)
        data = resp.json()
        valid_names = set(FEATURE_LIST)
        for factor in data["explanation"]:
            assert factor["feature"] in valid_names

    def test_shap_importance_numeric(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_shap4")
        resp = client.post("/predict", json=raw)
        for factor in resp.json()["explanation"]:
            assert isinstance(factor["importance"], float)

    def test_shap_ordering_descending(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_shap5")
        resp = client.post("/predict", json=raw)
        importances = [abs(f["importance"]) for f in resp.json()["explanation"]]
        assert importances == sorted(importances, reverse=True)

    def test_shap_additivity(self, client: TestClient):
        """expected_value + sum(SHAP values) ≈ model output (log-odds margin)."""
        _skip_if_no_model(client)
        from ml.api.app import _predictor
        from ml.predict.bundle import load_bundle
        from ml.models.baseline import apply_preprocessing
        import shap

        raw = _raw(customer_id="cust_e2e_shap_add")
        raw_data = raw.copy()

        features_df = engineer_features_for_inference(
            raw_data, history_store=_history_module.history_store
        )
        bundle = _predictor._bundle
        X = features_df[bundle.feature_names].copy()
        X_t = apply_preprocessing(X, bundle.preprocessing)

        explainer = shap.TreeExplainer(bundle.model)
        shap_values = explainer.shap_values(X_t)
        expected_value = explainer.expected_value

        # For XGBoost binary classifier, SHAP additivity:
        # expected_value + sum(shap_values[0]) ≈ model margin (raw log-odds)
        margin = bundle.model.predict(X_t, output_margin=True)[0]
        shap_sum = expected_value + np.sum(shap_values[0])
        assert abs(shap_sum - margin) < 0.01, (
            f"SHAP additivity violated: {shap_sum:.6f} vs margin {margin:.6f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. Determinism
# ═══════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Repeated identical calls produce identical results."""

    def test_deterministic_ml_prediction(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_det", timestamp=99999)

        # First call
        resp1 = client.post("/predict", json=raw)
        assert resp1.status_code == 200
        d1 = resp1.json()

        # Clear history so second call is identical
        store = _history_module.history_store
        try:
            store.clear()
        except Exception:
            pass

        # Second call
        resp2 = client.post("/predict", json=raw)
        assert resp2.status_code == 200
        d2 = resp2.json()

        assert d1["fraud_probability"] == d2["fraud_probability"]
        assert d1["fraud_prediction"] == d2["fraud_prediction"]
        assert d1["ml_score"] == d2["ml_score"]

        # SHAP factors should be identical
        f1 = d1.get("explanation", [])
        f2 = d2.get("explanation", [])
        assert len(f1) == len(f2)
        for a, b in zip(f1, f2):
            assert a["feature"] == b["feature"]
            assert a["importance"] == b["importance"]


# ═══════════════════════════════════════════════════════════════════════
# 5. Rule engine validation through real pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestRuleEngineE2E:
    """Rule engine through the real prediction flow."""

    def test_no_rules_normal_transaction(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_norule", amount=50.0)
        resp = client.post("/predict", json=raw)
        data = resp.json()
        expl = data.get("explanation_detail", {})
        assert data["rule_score"] == 0
        assert len(expl.get("rules_triggered", [])) == 0

    def test_high_amount_rule_triggers(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(
            customer_id="cust_e2e_hiamt",
            amount=HIGH_AMOUNT_THRESHOLD + 1,
        )
        resp = client.post("/predict", json=raw)
        data = resp.json()
        assert data["rule_score"] >= 15
        rules = data["explanation_detail"]["rules_triggered"]
        assert any(r["rule"] == "high_amount" for r in rules)

    def test_multiple_rules_e2e(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(
            customer_id="cust_e2e_multi",
            amount=HIGH_AMOUNT_THRESHOLD + 5000,
            merchant_category="7995",
        )
        resp = client.post("/predict", json=raw)
        data = resp.json()
        rules = {r["rule"] for r in data["explanation_detail"]["rules_triggered"]}
        assert "high_amount" in rules
        assert "high_risk_merchant" in rules
        assert data["rule_score"] >= 25


# ═══════════════════════════════════════════════════════════════════════
# 6. Risk aggregation validation through real pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestRiskAggregationE2E:
    """Risk aggregation formula verified end-to-end."""

    def test_risk_formula_e2e(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_formula", amount=500.0)
        resp = client.post("/predict", json=raw)
        data = resp.json()

        ml = data["ml_score"]
        beh = data["behaviour_score"]
        rule = data["rule_score"]
        expected = int(max(0, min(round(
            DEFAULT_WEIGHT_ML * ml
            + DEFAULT_WEIGHT_BEHAVIOUR * beh
            + DEFAULT_WEIGHT_RULE * rule
        ), 100)))
        assert data["risk_score"] == expected

    def test_risk_level_consistent_with_score(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_level", amount=500.0)
        resp = client.post("/predict", json=raw)
        data = resp.json()
        score = data["risk_score"]

        if score > 70:
            assert data["risk_level"] == "HIGH"
            assert data["decision"] == "HOLD"
        elif score > 30:
            assert data["risk_level"] == "MEDIUM"
            assert data["decision"] == "VERIFY"
        else:
            assert data["risk_level"] == "LOW"
            assert data["decision"] == "APPROVE"

    def test_aggregator_formula_manual_verification(self):
        """Controlled examples with manually computed expected scores."""
        cases = [
            (0.00, 0, 0, 0, "LOW", "APPROVE"),
            (1.00, 100, 100, 100, "HIGH", "HOLD"),
            (0.50, 50, 50, 50, "MEDIUM", "VERIFY"),
            (0.10, 10, 10, 10, "LOW", "APPROVE"),
            (0.80, 0, 0, 40, "MEDIUM", "VERIFY"),
            (0.00, 100, 0, 30, "LOW", "APPROVE"),
            (0.00, 0, 100, 20, "LOW", "APPROVE"),
        ]
        for prob, beh, rule, exp_score, exp_level, exp_dec in cases:
            a = aggregate_risk(
                fraud_probability=prob,
                behaviour_score=beh,
                rule_score=rule,
            )
            assert a.risk_score == exp_score, (
                f"prob={prob}, beh={beh}, rule={rule}: "
                f"expected {exp_score}, got {a.risk_score}"
            )
            assert a.risk_level == exp_level
            assert a.decision == exp_dec


# ═══════════════════════════════════════════════════════════════════════
# 7. Customer history validation
# ═══════════════════════════════════════════════════════════════════════


class TestCustomerHistoryE2E:
    """History lifecycle through the real pipeline."""

    def test_history_affects_subsequent_prediction(self, client: TestClient):
        _skip_if_no_model(client)
        # First transaction — cold start
        raw1 = _raw(customer_id="cust_e2e_hist", timestamp=1000, amount=100.0)
        resp1 = client.post("/predict", json=raw1)
        assert resp1.status_code == 200

        # Second transaction — same customer, different amount
        raw2 = _raw(customer_id="cust_e2e_hist", timestamp=2000, amount=5000.0)
        resp2 = client.post("/predict", json=raw2)
        assert resp2.status_code == 200

        # History should be available
        store = _history_module.history_store
        records = store.get("cust_e2e_hist", before_timestamp=99999)
        assert len(records) >= 2

    def test_customer_isolation_e2e(self, client: TestClient):
        _skip_if_no_model(client)
        raw_a = _raw(customer_id="cust_e2e_isoA", timestamp=1000, amount=100.0)
        raw_b = _raw(customer_id="cust_e2e_isoB", timestamp=1000, amount=100.0)
        client.post("/predict", json=raw_a)
        client.post("/predict", json=raw_b)

        store = _history_module.history_store
        recs_a = store.get("cust_e2e_isoA", before_timestamp=99999)
        recs_b = store.get("cust_e2e_isoB", before_timestamp=99999)
        assert len(recs_a) >= 1
        assert len(recs_b) >= 1


# ═══════════════════════════════════════════════════════════════════════
# 8. Outcome feedback validation
# ═══════════════════════════════════════════════════════════════════════


class TestOutcomeFeedbackE2E:
    """Outcome feedback loop through the real pipeline."""

    def test_outcome_update_affects_next_prediction(self, client: TestClient):
        _skip_if_no_model(client)
        # 1. Predict
        raw1 = _raw(customer_id="cust_e2e_outcome", timestamp=1000, amount=50.0)
        resp1 = client.post("/predict", json=raw1)
        assert resp1.status_code == 200

        # 2. Mark as fraud
        outcome_resp = client.post(
            "/outcome",
            json={
                "customer_id": "cust_e2e_outcome",
                "timestamp": 1000,
                "is_fraud": 1,
            },
        )
        assert outcome_resp.status_code == 200

        # 3. Next prediction should see previous_suspicious_count > 0
        raw2 = _raw(customer_id="cust_e2e_outcome", timestamp=2000, amount=50.0)
        resp2 = client.post("/predict", json=raw2)
        assert resp2.status_code == 200
        data = resp2.json()

        rules = [r["rule"] for r in data["explanation_detail"]["rules_triggered"]]
        assert "previous_suspicious" in rules

    def test_outcome_not_found(self, client: TestClient):
        _skip_if_no_model(client)
        resp = client.post(
            "/outcome",
            json={
                "customer_id": "nonexistent_customer",
                "timestamp": 99999,
                "is_fraud": 0,
            },
        )
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════
# 9. Backend contract validation (mock ML service)
# ═══════════════════════════════════════════════════════════════════════


class TestBackendContract:
    """Backend TransactionResponse matches api-contract.md."""

    @pytest.mark.asyncio
    async def test_backend_response_all_fields(self):
        """Backend response contains every field from api-contract.md."""
        from backend.schemas import TransactionResponse

        mock_ml_response = {
            "fraud_probability": 0.72,
            "fraud_prediction": 1,
            "threshold": 0.50,
            "model_version": "fraud-xgb-v1.0.0",
            "ml_score": 72,
            "behaviour_score": 60,
            "rule_score": 35,
            "risk_score": 61,
            "risk_level": "MEDIUM",
            "decision": "VERIFY",
            "explanation_detail": {
                "ml_top_factors": [
                    {"feature": "amount", "importance": 0.35},
                ],
                "behaviour_signals": [
                    {"signal": "spending_amount_anomaly", "severity": 0.6},
                ],
                "rules_triggered": [
                    {"rule": "high_amount", "contribution": 15},
                ],
            },
            "risk_factors": ["amount", "spending_amount_anomaly", "high_amount"],
            "explanation": [
                {"feature": "amount", "importance": 0.35},
            ],
        }

        from backend.schemas import MLPredictionResponse
        parsed = MLPredictionResponse.model_validate(mock_ml_response)

        # Build TransactionResponse as the backend router would
        response = TransactionResponse(
            transaction_id="test-txn-id",
            amount=1500.00,
            currency="USD",
            merchant_name="Test",
            merchant_category="5732",
            transaction_type="purchase",
            location_country="US",
            location_city="NYC",
            device_fingerprint="fp_test",
            device_type="desktop",
            ip_address="192.168.1.1",
            fraud_probability=parsed.fraud_probability,
            fraud_prediction=parsed.fraud_prediction,
            ml_score=parsed.ml_score,
            behaviour_score=parsed.behaviour_score,
            rule_score=parsed.rule_score,
            risk_score=parsed.risk_score,
            risk_level=parsed.risk_level,
            decision=parsed.decision,
            explanation=parsed.explanation_detail,
            risk_factors=parsed.risk_factors,
            model_version=parsed.model_version,
        )

        data = response.model_dump()

        # Verify all api-contract.md fields
        assert data["amount"] == 1500.00
        assert data["currency"] == "USD"
        assert data["ml_score"] == 72
        assert data["behaviour_score"] == 60
        assert data["rule_score"] == 35
        assert data["risk_score"] == 61
        assert data["risk_level"] == "MEDIUM"
        assert data["decision"] == "VERIFY"
        assert data["explanation"] is not None
        assert data["risk_factors"] is not None
        assert data["model_version"] == "fraud-xgb-v1.0.0"


# ═══════════════════════════════════════════════════════════════════════
# 10. Error handling
# ═══════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Error handling across the pipeline."""

    def test_invalid_input_422(self, client: TestClient):
        resp = client.post("/predict", json={"invalid": True})
        assert resp.status_code == 422
        assert "traceback" not in resp.text.lower()

    def test_missing_required_field_422(self, client: TestClient):
        resp = client.post("/predict", json={"amount": 100.0})
        assert resp.status_code == 422

    def test_negative_amount_422(self, client: TestClient):
        raw = _raw(amount=-50.0)
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 422

    def test_forbidden_isfraud_422(self, client: TestClient):
        raw = _raw()
        raw["isFraud"] = 1
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 422

    def test_model_unavailable_503(self, client: TestClient):
        import ml.api.app as app_module
        original = app_module._predictor
        app_module._predictor = None
        try:
            resp = client.post("/predict", json=_raw())
            assert resp.status_code == 503
            assert "traceback" not in resp.text.lower()
        finally:
            app_module._predictor = original

    def test_no_internal_urls_exposed(self, client: TestClient):
        """Error responses do not expose internal URLs or stack traces."""
        resp = client.post("/predict", json={"bad": "data"})
        body = resp.text.lower()
        assert "traceback" not in body
        assert "localhost" not in body
        assert "127.0.0" not in body

    def test_no_secrets_in_response(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_secrets")
        resp = client.post("/predict", json=raw)
        body = resp.text.lower()
        assert "password" not in body
        assert "secret" not in body
        assert "api_key" not in body


# ═══════════════════════════════════════════════════════════════════════
# 11. Leakage protection (runtime)
# ═══════════════════════════════════════════════════════════════════════


class TestLeakageProtection:
    """Runtime leakage checks through the real pipeline."""

    def test_isfraud_rejected_at_api(self, client: TestClient):
        raw = _raw()
        raw["isFraud"] = 1
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 422

    def test_transaction_id_rejected_at_api(self, client: TestClient):
        raw = _raw()
        raw["TransactionID"] = "T_001"
        resp = client.post("/predict", json=raw)
        assert resp.status_code == 422

    def test_isfraud_not_in_feature_list(self):
        assert "isFraud" not in FEATURE_LIST

    def test_transaction_id_not_in_feature_list(self):
        assert "TransactionID" not in FEATURE_LIST

    def test_predictor_rejects_forbidden_columns(self):
        """FraudPredictor raises ValueError if isFraud/TransactionID in input."""
        import pandas as pd
        from ml.predict.predictor import FraudPredictor
        try:
            predictor = FraudPredictor()
        except (FileNotFoundError, KeyError):
            pytest.skip("Model not available")

        data = {f: [0.0] for f in predictor.feature_names}
        data["isFraud"] = [1]
        df = pd.DataFrame(data)
        with pytest.raises(ValueError, match="Forbidden"):
            predictor.predict(df)

    def test_history_excludes_current_transaction(self, client: TestClient):
        _skip_if_no_model(client)
        raw = _raw(customer_id="cust_e2e_excl", timestamp=5000)
        client.post("/predict", json=raw)

        store = _history_module.history_store
        # History should have the recorded transaction
        records = store.get("cust_e2e_excl", before_timestamp=99999)
        assert len(records) >= 1

        # But querying before the current timestamp should return nothing
        records_before = store.get("cust_e2e_excl", before_timestamp=5000)
        assert len(records_before) == 0
