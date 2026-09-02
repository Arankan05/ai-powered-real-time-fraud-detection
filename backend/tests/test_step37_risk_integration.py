"""Step 37 — Complete Backend Risk Integration tests.

Validates that the backend transaction endpoint correctly consumes the
*complete* ML/Fraud Intelligence Service response — including fraud
probability, prediction, ML/rule/behaviour scores, aggregated risk
score, risk level, decision, SHAP explanation, behaviour signals
with ``reason``, triggered rules with ``reason``, risk factors,
model version, and timestamp.

Covers:
  1.  Successful integration — every ML field reaches the backend response
  2.  ML failure scenarios — connection, timeout, 4xx, 5xx, malformed
  3.  Backend validation — missing fields, invalid values, forbidden fields
  4.  No duplicated ML logic in the backend
  5.  End-to-end flow with real ML service (optional, skipped if unavailable)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import Response

from backend.services.ml_client import MLServiceClient
from backend.schemas import (
    MLBehaviourSignal,
    MLExplanation,
    MLPredictionResponse,
    MLRuleTrigger,
    TransactionResponse,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def complete_ml_response() -> dict:
    """Realistic ML service response with ALL fields populated.

    Mirrors the ML service ``PredictionResponse`` schema in
    ``ml/api/app.py`` — includes fraud probability, prediction,
    all risk scores, explanation with behaviour/rule reasons,
    risk_factors, model_version, and timestamp.
    """
    return {
        "fraud_probability": 0.8234,
        "fraud_prediction": 1,
        "threshold": 0.50,
        "model_version": "fraud-xgb-v1.0.0",
        "timestamp": 1725200000,
        # Legacy explanation (list of SHAP factors)
        "explanation": [
            {"feature": "amount_deviation", "importance": 0.45},
            {"feature": "is_new_device", "importance": 0.30},
            {"feature": "tx_velocity_7d", "importance": 0.22},
            {"feature": "device_fingerprint", "importance": -0.18},
            {"feature": "merchant_category", "importance": 0.12},
        ],
        # Risk scores
        "ml_score": 82,
        "behaviour_score": 65,
        "rule_score": 40,
        "risk_score": 68,
        "risk_level": "MEDIUM",
        "decision": "VERIFY",
        # Structured explanation
        "explanation_detail": {
            "ml_top_factors": [
                {"feature": "amount_deviation", "importance": 0.45},
                {"feature": "is_new_device", "importance": 0.30},
                {"feature": "tx_velocity_7d", "importance": 0.22},
                {"feature": "device_fingerprint", "importance": -0.18},
                {"feature": "merchant_category", "importance": 0.12},
            ],
            "behaviour_signals": [
                {
                    "signal": "spending_amount_anomaly",
                    "severity": 0.85,
                    "reason": "Transaction amount 3.2x above 30-day average",
                },
                {
                    "signal": "device_anomaly",
                    "severity": 0.60,
                    "reason": "Device fingerprint not seen in last 90 days",
                },
            ],
            "rules_triggered": [
                {
                    "rule": "new_device_high_amount",
                    "contribution": 15,
                    "reason": "New device with amount > $1000",
                },
                {
                    "rule": "velocity_limit",
                    "contribution": 20,
                    "reason": "5 transactions in last 24 hours",
                },
                {
                    "rule": "high_risk_merchant",
                    "contribution": 10,
                    "reason": "Merchant category 5732 flagged as high-risk",
                },
            ],
        },
        # Combined risk factors
        "risk_factors": [
            "amount_deviation",
            "is_new_device",
            "tx_velocity_7d",
            "spending_amount_anomaly",
            "device_anomaly",
            "new_device_high_amount",
            "velocity_limit",
            "high_risk_merchant",
        ],
    }


@pytest.fixture
def valid_transaction() -> dict:
    """Valid raw transaction payload matching TransactionCreate schema."""
    return {
        "amount": 5000.00,
        "currency": "USD",
        "merchant_name": "Offshore Trading Ltd",
        "merchant_category": "5732",
        "transaction_type": "purchase",
        "location_country": "KY",
        "location_city": "George Town",
        "device_fingerprint": "xyz789newdevice",
        "device_type": "desktop",
        "ip_address": "10.0.0.50",
    }


@pytest.fixture
def test_client(auth_override):
    """FastAPI TestClient with ML client configured."""
    from fastapi.testclient import TestClient
    from backend.app import app
    from backend.routers import transactions as txn_module

    ml_client = MLServiceClient(base_url="http://mock-ml:8001", timeout=2.0)
    txn_module.set_ml_client(ml_client)
    return TestClient(app), ml_client


# ── 1. Successful integration — complete ML response ──────────────────


class TestCompleteRiskIntegration:
    """Verify every ML field reaches the backend response correctly."""

    def test_fraud_probability(self, test_client, valid_transaction, complete_ml_response):
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        data = resp.json()
        assert data["fraud_probability"] == 0.8234

    def test_fraud_prediction(self, test_client, valid_transaction, complete_ml_response):
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        assert resp.json()["fraud_prediction"] == 1

    def test_ml_score(self, test_client, valid_transaction, complete_ml_response):
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        assert resp.json()["ml_score"] == 82

    def test_behaviour_score(self, test_client, valid_transaction, complete_ml_response):
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        assert resp.json()["behaviour_score"] == 65

    def test_rule_score(self, test_client, valid_transaction, complete_ml_response):
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        assert resp.json()["rule_score"] == 40

    def test_risk_score(self, test_client, valid_transaction, complete_ml_response):
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        assert resp.json()["risk_score"] == 68

    def test_risk_level(self, test_client, valid_transaction, complete_ml_response):
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        assert resp.json()["risk_level"] == "MEDIUM"

    def test_decision(self, test_client, valid_transaction, complete_ml_response):
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        assert resp.json()["decision"] == "VERIFY"

    def test_model_version(self, test_client, valid_transaction, complete_ml_response):
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        assert resp.json()["model_version"] == "fraud-xgb-v1.0.0"

    def test_timestamp(self, test_client, valid_transaction, complete_ml_response):
        """Timestamp from ML service reaches backend response."""
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        assert resp.json()["timestamp"] == 1725200000

    def test_risk_factors(self, test_client, valid_transaction, complete_ml_response):
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        data = resp.json()
        assert data["risk_factors"] == [
            "amount_deviation",
            "is_new_device",
            "tx_velocity_7d",
            "spending_amount_anomaly",
            "device_anomaly",
            "new_device_high_amount",
            "velocity_limit",
            "high_risk_merchant",
        ]

    def test_shap_explanation(self, test_client, valid_transaction, complete_ml_response):
        """SHAP top factors reach the explanation object."""
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        expl = resp.json()["explanation"]
        assert expl is not None
        factors = expl["ml_top_factors"]
        assert len(factors) == 5
        assert factors[0]["feature"] == "amount_deviation"
        assert factors[0]["importance"] == 0.45

    def test_behaviour_signals_with_reason(self, test_client, valid_transaction, complete_ml_response):
        """Behaviour signals include signal, severity, AND reason."""
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        signals = resp.json()["explanation"]["behaviour_signals"]
        assert len(signals) == 2
        assert signals[0]["signal"] == "spending_amount_anomaly"
        assert signals[0]["severity"] == 0.85
        assert signals[0]["reason"] == "Transaction amount 3.2x above 30-day average"
        assert signals[1]["signal"] == "device_anomaly"
        assert signals[1]["severity"] == 0.60
        assert signals[1]["reason"] == "Device fingerprint not seen in last 90 days"

    def test_rules_triggered_with_reason(self, test_client, valid_transaction, complete_ml_response):
        """Triggered rules include rule, contribution, AND reason."""
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        rules = resp.json()["explanation"]["rules_triggered"]
        assert len(rules) == 3
        assert rules[0]["rule"] == "new_device_high_amount"
        assert rules[0]["contribution"] == 15
        assert rules[0]["reason"] == "New device with amount > $1000"
        assert rules[1]["rule"] == "velocity_limit"
        assert rules[1]["contribution"] == 20
        assert rules[2]["rule"] == "high_risk_merchant"
        assert rules[2]["contribution"] == 10

    def test_transaction_fields_preserved(self, test_client, valid_transaction, complete_ml_response):
        """Original transaction fields are returned unchanged."""
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        data = resp.json()
        assert data["amount"] == 5000.00
        assert data["currency"] == "USD"
        assert data["merchant_name"] == "Offshore Trading Ltd"
        assert data["merchant_category"] == "5732"
        assert data["transaction_type"] == "purchase"
        assert data["location_country"] == "KY"
        assert data["location_city"] == "George Town"
        assert data["device_fingerprint"] == "xyz789newdevice"
        assert data["device_type"] == "desktop"
        assert data["ip_address"] == "10.0.0.50"

    def test_all_fields_in_single_response(self, test_client, valid_transaction, complete_ml_response):
        """Single assertion: all expected keys exist in the response."""
        tc, _ = test_client
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        data = resp.json()
        expected_keys = {
            "amount", "currency", "merchant_name", "merchant_category",
            "transaction_type", "location_country", "location_city",
            "device_fingerprint", "device_type", "ip_address",
            "fraud_probability", "fraud_prediction", "ml_score",
            "behaviour_score", "rule_score", "risk_score", "risk_level",
            "decision", "explanation", "risk_factors", "model_version",
            "timestamp",
        }
        assert expected_keys.issubset(set(data.keys())), (
            f"Missing keys: {expected_keys - set(data.keys())}"
        )


# ── 2. ML failure scenarios ──────────────────────────────────────────


class TestMLFailureScenarios:
    """Verify backend handles all ML service failure modes correctly."""

    def test_connection_refused(self, test_client, valid_transaction):
        """ML service unreachable → 503."""
        tc, _ = test_client
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()

    def test_timeout(self, test_client, valid_transaction):
        """ML service timeout → 503."""
        tc, _ = test_client
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ReadTimeout("Read timed out"),
        ):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 503
        assert "timed out" in resp.json()["detail"].lower()

    def test_ml_503(self, test_client, valid_transaction):
        """ML returns 503 (model unavailable) → backend 503."""
        tc, _ = test_client
        mock_resp = Response(503, json={"detail": "Model not available"})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 503

    def test_ml_422(self, test_client, valid_transaction):
        """ML returns 422 (validation error) → backend 502."""
        tc, _ = test_client
        mock_resp = Response(422, json={"detail": "Missing required features"})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 502

    def test_ml_500(self, test_client, valid_transaction):
        """ML returns 500 (internal error) → backend 502."""
        tc, _ = test_client
        mock_resp = Response(500, json={"detail": "Internal server error"})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 502

    def test_malformed_json(self, test_client, valid_transaction):
        """ML returns non-JSON body → backend 502."""
        tc, _ = test_client
        mock_resp = Response(200, text="<html>error</html>", headers={"content-type": "text/html"})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 502

    def test_incomplete_ml_response(self, test_client, valid_transaction):
        """ML response missing risk fields → backend still returns (with None for missing)."""
        tc, _ = test_client
        partial = {
            "fraud_probability": 0.15,
            "fraud_prediction": 0,
            "threshold": 0.50,
            "model_version": "fraud-xgb-v1.0.0",
        }
        mock_resp = Response(200, json=partial)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
        assert resp.status_code == 201
        data = resp.json()
        assert data["fraud_probability"] == 0.15
        assert data["fraud_prediction"] == 0
        # Missing fields default to None
        assert data["ml_score"] is None
        assert data["risk_score"] is None
        assert data["risk_level"] is None
        assert data["decision"] is None
        assert data["timestamp"] is None
        assert data["explanation"] is None

    def test_ml_no_client_configured(self, valid_transaction, auth_override):
        """ML client not configured → 503."""
        from fastapi.testclient import TestClient
        from backend.app import app
        from backend.routers import transactions as txn_module

        txn_module.set_ml_client(None)
        tc = TestClient(app)
        try:
            resp = tc.post("/api/v1/transactions", json=valid_transaction)
            assert resp.status_code == 503
        finally:
            txn_module.set_ml_client(MLServiceClient(base_url="http://mock-ml:8001"))


# ── 3. Backend validation ────────────────────────────────────────────


class TestBackendValidation:
    """Verify backend rejects invalid transactions before calling ML."""

    def test_missing_required_field(self, test_client):
        """Missing required field → 422."""
        tc, _ = test_client
        bad = {"amount": 100, "currency": "USD"}  # missing 8 fields
        resp = tc.post("/api/v1/transactions", json=bad)
        assert resp.status_code == 422

    def test_negative_amount(self, test_client, valid_transaction):
        """Negative amount → 422 (gt=0 constraint)."""
        tc, _ = test_client
        bad = dict(valid_transaction)
        bad["amount"] = -500
        resp = tc.post("/api/v1/transactions", json=bad)
        assert resp.status_code == 422

    def test_zero_amount(self, test_client, valid_transaction):
        """Zero amount → 422 (gt=0 constraint)."""
        tc, _ = test_client
        bad = dict(valid_transaction)
        bad["amount"] = 0
        resp = tc.post("/api/v1/transactions", json=bad)
        assert resp.status_code == 422

    def test_invalid_transaction_type(self, test_client, valid_transaction):
        """Invalid transaction_type → 422 (pattern constraint)."""
        tc, _ = test_client
        bad = dict(valid_transaction)
        bad["transaction_type"] = "refund"
        resp = tc.post("/api/v1/transactions", json=bad)
        assert resp.status_code == 422

    def test_invalid_device_type(self, test_client, valid_transaction):
        """Invalid device_type → 422 (pattern constraint)."""
        tc, _ = test_client
        bad = dict(valid_transaction)
        bad["device_type"] = "tablet"
        resp = tc.post("/api/v1/transactions", json=bad)
        assert resp.status_code == 422

    def test_invalid_currency_length(self, test_client, valid_transaction):
        """Currency must be 3 characters → 422."""
        tc, _ = test_client
        bad = dict(valid_transaction)
        bad["currency"] = "USDX"
        resp = tc.post("/api/v1/transactions", json=bad)
        assert resp.status_code == 422

    def test_invalid_ip_address(self, test_client, valid_transaction):
        """IP address too short → 422 (min_length=7)."""
        tc, _ = test_client
        bad = dict(valid_transaction)
        bad["ip_address"] = "1.2"
        resp = tc.post("/api/v1/transactions", json=bad)
        assert resp.status_code == 422

    def test_empty_merchant_name(self, test_client, valid_transaction):
        """Empty merchant name → 422 (min_length=1)."""
        tc, _ = test_client
        bad = dict(valid_transaction)
        bad["merchant_name"] = ""
        resp = tc.post("/api/v1/transactions", json=bad)
        assert resp.status_code == 422

    def test_wrong_field_type(self, test_client, valid_transaction):
        """Wrong field type (string instead of float for amount) → 422."""
        tc, _ = test_client
        bad = dict(valid_transaction)
        bad["amount"] = "not_a_number"
        resp = tc.post("/api/v1/transactions", json=bad)
        assert resp.status_code == 422

    def test_empty_body(self, test_client):
        """Empty request body → 422."""
        tc, _ = test_client
        resp = tc.post("/api/v1/transactions", json={})
        assert resp.status_code == 422


# ── 4. No duplicated ML logic ────────────────────────────────────────


class TestNoDuplicatedLogic:
    """Verify the backend does NOT duplicate ML calculations."""

    def test_backend_does_not_import_ml_modules(self):
        """Backend router must not import ml.* modules."""
        import backend.routers.transactions as txn_mod
        import inspect
        source = inspect.getsource(txn_mod)
        forbidden = [
            "from ml.", "import ml.",
            "from ml.features", "from ml.predict",
            "from ml.rules", "from ml.risk",
            "from ml.explainability",
        ]
        for pattern in forbidden:
            assert pattern not in source, (
                f"Backend router imports ML module: {pattern}"
            )

    def test_backend_does_not_compute_probability(self):
        """Backend router must not calculate fraud probability."""
        import backend.routers.transactions as txn_mod
        import inspect
        source = inspect.getsource(txn_mod)
        assert "predict_proba" not in source
        assert "fraud_probability =" not in source.replace("fraud_probability=ml_result", "")

    def test_backend_does_not_compute_shap(self):
        """Backend router must not calculate SHAP values."""
        import backend.routers.transactions as txn_mod
        import ast, inspect, textwrap
        tree = ast.parse(inspect.getsource(txn_mod))
        # Extract only non-comment, non-docstring code
        code_lines = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call, ast.Assign)):
                code_lines.append(ast.dump(node))
        code_str = " ".join(code_lines).lower()
        assert "TreeExplainer" not in code_str
        assert "shap" not in code_str

    def test_backend_does_not_evaluate_rules(self):
        """Backend router must not evaluate risk rules."""
        import backend.routers.transactions as txn_mod
        import inspect
        source = inspect.getsource(txn_mod)
        assert "evaluate_rules" not in source
        assert "rule_score =" not in source.replace("rule_score=ml_result", "")

    def test_backend_does_not_aggregate_risk(self):
        """Backend router must not aggregate risk scores."""
        import backend.routers.transactions as txn_mod
        import inspect
        source = inspect.getsource(txn_mod)
        assert "aggregate_risk" not in source
        assert "risk_score =" not in source.replace("risk_score=ml_result", "")

    def test_backend_does_not_engineer_features(self):
        """Backend router must not run feature engineering."""
        import backend.routers.transactions as txn_mod
        import inspect
        source = inspect.getsource(txn_mod)
        assert "engineer_features" not in source


# ── 5. Schema model tests ────────────────────────────────────────────


class TestSchemaModels:
    """Verify backend Pydantic models correctly parse ML responses."""

    def test_ml_prediction_response_complete(self, complete_ml_response):
        """MLPredictionResponse parses complete ML response with all fields."""
        parsed = MLPredictionResponse.model_validate(complete_ml_response)
        assert parsed.fraud_probability == 0.8234
        assert parsed.fraud_prediction == 1
        assert parsed.ml_score == 82
        assert parsed.behaviour_score == 65
        assert parsed.rule_score == 40
        assert parsed.risk_score == 68
        assert parsed.risk_level == "MEDIUM"
        assert parsed.decision == "VERIFY"
        assert parsed.model_version == "fraud-xgb-v1.0.0"
        assert parsed.timestamp == 1725200000
        assert len(parsed.risk_factors) == 8

    def test_ml_prediction_response_partial(self):
        """MLPredictionResponse handles partial response (legacy ML service)."""
        partial = {
            "fraud_probability": 0.15,
            "fraud_prediction": 0,
            "threshold": 0.50,
            "model_version": "fraud-xgb-v1.0.0",
        }
        parsed = MLPredictionResponse.model_validate(partial)
        assert parsed.fraud_probability == 0.15
        assert parsed.ml_score is None
        assert parsed.risk_score is None
        assert parsed.timestamp is None

    def test_behaviour_signal_with_reason(self):
        """MLBehaviourSignal captures reason field."""
        sig = MLBehaviourSignal(signal="spending_anomaly", severity=0.9, reason="3x above avg")
        assert sig.signal == "spending_anomaly"
        assert sig.severity == 0.9
        assert sig.reason == "3x above avg"

    def test_behaviour_signal_without_reason(self):
        """MLBehaviourSignal reason is optional for backward compatibility."""
        sig = MLBehaviourSignal(signal="test", severity=0.5)
        assert sig.reason is None

    def test_rule_trigger_with_reason(self):
        """MLRuleTrigger captures reason field."""
        rule = MLRuleTrigger(rule="high_amount", contribution=15, reason="Amount > $5000")
        assert rule.rule == "high_amount"
        assert rule.contribution == 15
        assert rule.reason == "Amount > $5000"

    def test_rule_trigger_without_reason(self):
        """MLRuleTrigger reason is optional for backward compatibility."""
        rule = MLRuleTrigger(rule="test", contribution=10)
        assert rule.reason is None

    def test_explanation_with_reasons(self, complete_ml_response):
        """MLExplanation correctly parses nested behaviour and rule reasons."""
        parsed = MLPredictionResponse.model_validate(complete_ml_response)
        expl = parsed.explanation_detail
        assert expl is not None
        # Behaviour signals
        assert len(expl.behaviour_signals) == 2
        assert expl.behaviour_signals[0].reason == "Transaction amount 3.2x above 30-day average"
        assert expl.behaviour_signals[1].reason == "Device fingerprint not seen in last 90 days"
        # Rules triggered
        assert len(expl.rules_triggered) == 3
        assert expl.rules_triggered[0].reason == "New device with amount > $1000"
        assert expl.rules_triggered[1].reason == "5 transactions in last 24 hours"
        assert expl.rules_triggered[2].reason == "Merchant category 5732 flagged as high-risk"

    def test_transaction_response_includes_timestamp(self):
        """TransactionResponse has timestamp field."""
        fields = TransactionResponse.model_fields
        assert "timestamp" in fields

    def test_extra_fields_allowed(self):
        """MLPredictionResponse allows extra fields for forward compatibility."""
        data = {"fraud_probability": 0.5, "new_future_field": "value"}
        parsed = MLPredictionResponse.model_validate(data)
        assert parsed.fraud_probability == 0.5


# ── 6. ML client parsing ─────────────────────────────────────────────


class TestMLClientParsing:
    """Verify MLServiceClient correctly parses the complete ML response."""

    @pytest.mark.asyncio
    async def test_client_parses_complete_response(self, complete_ml_response):
        """MLServiceClient.predict() returns typed MLPredictionResponse."""
        client = MLServiceClient(base_url="http://mock:8001")
        mock_resp = Response(200, json=complete_ml_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.predict({"amount": 5000})
        assert isinstance(result, MLPredictionResponse)
        assert result.fraud_probability == 0.8234
        assert result.ml_score == 82
        assert result.behaviour_score == 65
        assert result.rule_score == 40
        assert result.risk_score == 68
        assert result.risk_level == "MEDIUM"
        assert result.decision == "VERIFY"
        assert result.timestamp == 1725200000
        assert result.explanation_detail is not None
        assert len(result.explanation_detail.behaviour_signals) == 2
        assert result.explanation_detail.behaviour_signals[0].reason is not None
        assert len(result.explanation_detail.rules_triggered) == 3
        assert result.explanation_detail.rules_triggered[0].reason is not None
        assert len(result.risk_factors) == 8


# ── 7. End-to-end flow (requires running ML service) ─────────────────


class TestEndToEndFlow:
    """Real end-to-end test: Backend → ML /predict → full pipeline → response.

    These tests require the ML service running on localhost:8001.
    They are automatically skipped if the service is unavailable.
    """

    @pytest.fixture(autouse=True)
    def _check_ml_service(self):
        """Skip if ML service is not running."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            sock.connect(("localhost", 8001))
            sock.close()
        except (ConnectionRefusedError, OSError):
            pytest.skip("ML service not running on localhost:8001")

    def test_e2e_complete_flow(self, auth_override):
        """POST /api/v1/transactions → Backend → ML → full risk result."""
        from fastapi.testclient import TestClient
        from backend.app import app
        from backend.routers import transactions as txn_module
        from backend.config import get_settings

        settings = get_settings()
        ml_client = MLServiceClient(
            base_url=settings.ML_SERVICE_URL,
            timeout=float(settings.ML_REQUEST_TIMEOUT_SECONDS),
        )
        txn_module.set_ml_client(ml_client)
        tc = TestClient(app)

        transaction = {
            "amount": 5000.00,
            "currency": "USD",
            "merchant_name": "Test Merchant",
            "merchant_category": "5732",
            "transaction_type": "purchase",
            "location_country": "US",
            "location_city": "New York",
            "device_fingerprint": "e2e_test_device_001",
            "device_type": "mobile",
            "ip_address": "192.168.1.100",
        }

        resp = tc.post("/api/v1/transactions", json=transaction)
        assert resp.status_code == 201
        data = resp.json()

        # Verify all ML/risk fields are present
        assert data["fraud_probability"] is not None
        assert isinstance(data["fraud_probability"], float)
        assert 0 <= data["fraud_probability"] <= 1

        assert data["fraud_prediction"] is not None
        assert data["fraud_prediction"] in (0, 1)

        assert data["ml_score"] is not None
        assert isinstance(data["ml_score"], int)
        assert 0 <= data["ml_score"] <= 100

        assert data["behaviour_score"] is not None
        assert isinstance(data["behaviour_score"], int)
        assert 0 <= data["behaviour_score"] <= 100

        assert data["rule_score"] is not None
        assert isinstance(data["rule_score"], int)
        assert 0 <= data["rule_score"] <= 100

        assert data["risk_score"] is not None
        assert isinstance(data["risk_score"], int)
        assert 0 <= data["risk_score"] <= 100

        assert data["risk_level"] is not None
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")

        assert data["decision"] is not None
        assert data["decision"] in ("APPROVE", "VERIFY", "HOLD")

        assert data["model_version"] is not None
        assert isinstance(data["model_version"], str)

        assert data["timestamp"] is not None
        assert isinstance(data["timestamp"], int)

        # Explanation
        assert data["explanation"] is not None
        assert "ml_top_factors" in data["explanation"]
        assert isinstance(data["explanation"]["ml_top_factors"], list)
        assert len(data["explanation"]["ml_top_factors"]) > 0

        # Risk factors
        assert data["risk_factors"] is not None
        assert isinstance(data["risk_factors"], list)
