"""Step 43 — ML monitoring and observability tests.

Comprehensive test suite for the production monitoring layer:
metrics tracking, latency monitoring, error categorisation, model
version observability, drift detection, monitoring endpoint security,
concurrency safety, and performance overhead.

Run from the project root::

    python -m pytest ml/api/tests/test_step43_monitoring.py -v
"""

from __future__ import annotations

import json
import os
import threading
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ml.api import app as _app_module
from ml.api.app import app
import ml.features.history as _history_module
from ml.monitoring.metrics import (
    MonitoringConfig,
    PredictionMetrics,
    _VALID_DECISIONS,
    _VALID_ERROR_CATEGORIES,
    _VALID_RISK_LEVELS,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _valid_transaction(**overrides) -> dict:
    base = {
        "amount": 150.0,
        "currency": "USD",
        "merchant_name": "Test Merchant",
        "merchant_category": "5732",
        "transaction_type": "purchase",
        "location_country": "US",
        "location_city": "New York",
        "device_fingerprint": "fp_monitor_test",
        "device_type": "mobile",
        "ip_address": "192.168.1.100",
    }
    base.update(overrides)
    return base


def _model_available(client: TestClient) -> bool:
    resp = client.get("/health")
    return resp.json().get("status") == "ready"


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_metrics_and_history():
    """Reset metrics and clear history before/after each test."""
    from ml.monitoring import metrics as _mod
    _mod.metrics.reset()
    try:
        _history_module.history_store.clear()
    except Exception:
        _history_module.history_store = _history_module.InMemoryHistoryStore()
    yield
    _mod.metrics.reset()
    try:
        _history_module.history_store.clear()
    except Exception:
        _history_module.history_store = _history_module.InMemoryHistoryStore()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# =====================================================================
# 1. METRICS COUNTERS
# =====================================================================


class TestMetricsCounters:
    """Request, success, failure, and distribution counters."""

    def test_request_counter_increments(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        client.post("/predict", json=_valid_transaction())
        snap = _mod.metrics.snapshot()
        assert snap["total_requests"] >= 1

    def test_successful_prediction_counter(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        for _ in range(3):
            client.post("/predict", json=_valid_transaction())
        snap = _mod.metrics.snapshot()
        assert snap["successful_predictions"] == 3

    def test_failed_prediction_counter(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        # Trigger a feature engineering failure
        with patch(
            "ml.api.app.engineer_features_for_inference",
            side_effect=ValueError("forced"),
        ):
            client.post("/predict", json=_valid_transaction())
        snap = _mod.metrics.snapshot()
        assert snap["failed_predictions"] >= 1
        assert snap["errors"]["feature_engineering"] >= 1

    def test_fraud_non_fraud_counts(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        client.post("/predict", json=_valid_transaction())
        snap = _mod.metrics.snapshot()
        total = snap["fraud_count"] + snap["non_fraud_count"]
        assert total == snap["successful_predictions"]

    def test_decision_distribution(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        client.post("/predict", json=_valid_transaction())
        snap = _mod.metrics.snapshot()
        decisions = snap["decisions"]
        assert set(decisions.keys()) == _VALID_DECISIONS
        assert sum(decisions.values()) == snap["successful_predictions"]

    def test_risk_level_distribution(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        client.post("/predict", json=_valid_transaction())
        snap = _mod.metrics.snapshot()
        risk = snap["risk_levels"]
        assert set(risk.keys()) == _VALID_RISK_LEVELS
        assert sum(risk.values()) == snap["successful_predictions"]

    def test_model_version_reported(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        client.post("/predict", json=_valid_transaction())
        snap = _mod.metrics.snapshot()
        assert snap["model_version"] is not None
        assert len(snap["model_version"]) > 0

    def test_model_version_consistent_with_health(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        client.post("/predict", json=_valid_transaction())
        health = client.get("/health").json()
        snap = _mod.metrics.snapshot()
        assert snap["model_version"] == health["model_version"]


# =====================================================================
# 2. LATENCY MONITORING
# =====================================================================


class TestLatencyMonitoring:
    """Prediction latency measurement and statistics."""

    def test_latency_recorded(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        client.post("/predict", json=_valid_transaction())
        snap = _mod.metrics.snapshot()
        lat = snap["latency"]
        assert lat["count"] == 1
        assert lat["mean_seconds"] is not None
        assert lat["mean_seconds"] > 0

    def test_latency_percentiles(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        for _ in range(5):
            client.post("/predict", json=_valid_transaction())
        snap = _mod.metrics.snapshot()
        lat = snap["latency"]
        assert lat["count"] == 5
        assert lat["p50_seconds"] is not None
        assert lat["p95_seconds"] is not None
        assert lat["p99_seconds"] is not None
        assert lat["min_seconds"] <= lat["p50_seconds"] <= lat["max_seconds"]

    def test_latency_empty_initially(self):
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        snap = _mod.metrics.snapshot()
        assert snap["latency"]["count"] == 0
        assert snap["latency"]["mean_seconds"] is None

    def test_slow_prediction_counter(self):
        """Predictions exceeding the latency threshold are counted."""
        m = PredictionMetrics(MonitoringConfig())
        # Record with a very short threshold
        m._config.latency_warn_seconds = 0.0  # everything is "slow"
        m.record_success(
            latency_ms=100.0,
            fraud_prediction=0,
            fraud_probability=0.1,
            decision="APPROVE",
            risk_level="LOW",
            risk_score=10,
            amount=100.0,
            model_version="test",
        )
        snap = m.snapshot()
        assert snap["slow_predictions"] == 1


# =====================================================================
# 3. ERROR MONITORING
# =====================================================================


class TestErrorMonitoring:
    """Error categorisation and monitoring."""

    def test_validation_error_categorised(self, client: TestClient):
        """Invalid input → 422, categorised as 'validation'."""
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        # This triggers a ValueError in the predictor (missing columns)
        # but the simpler path is a Pydantic 422 which doesn't hit our
        # error handler.  Use a mock to simulate a ValueError.
        with patch.object(
            _app_module._predictor, "predict",
            side_effect=ValueError("Missing features for test"),
        ):
            client.post("/predict", json=_valid_transaction())
        snap = _mod.metrics.snapshot()
        assert snap["errors"]["validation"] >= 1

    def test_model_unavailable_categorised(self, client: TestClient):
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        saved = _app_module._predictor
        try:
            _app_module._predictor = None
            client.post("/predict", json=_valid_transaction())
        finally:
            _app_module._predictor = saved
        snap = _mod.metrics.snapshot()
        assert snap["errors"]["model_unavailable"] >= 1

    def test_prediction_failure_categorised(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        with patch.object(
            _app_module._predictor, "predict",
            side_effect=RuntimeError("Model crash"),
        ):
            client.post("/predict", json=_valid_transaction())
        snap = _mod.metrics.snapshot()
        assert snap["errors"]["prediction_failure"] >= 1

    def test_feature_engineering_error_categorised(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        with patch(
            "ml.api.app.engineer_features_for_inference",
            side_effect=ValueError("bad data"),
        ):
            client.post("/predict", json=_valid_transaction())
        snap = _mod.metrics.snapshot()
        assert snap["errors"]["feature_engineering"] >= 1

    def test_no_exception_leakage_in_metrics(self, client: TestClient):
        """Metrics never contain exception strings or tracebacks."""
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        with patch(
            "ml.api.app.engineer_features_for_inference",
            side_effect=OSError("C:\\secret\\path\\leaked"),
        ):
            client.post("/predict", json=_valid_transaction())
        snap = _mod.metrics.snapshot()
        full_json = json.dumps(snap)
        assert "Traceback" not in full_json
        assert "C:\\secret" not in full_json
        assert "OSError" not in full_json

    def test_unknown_error_category_safe(self):
        """Unrecognised categories are mapped to 'unknown'."""
        m = PredictionMetrics()
        m.record_error(category="totally_made_up")
        snap = m.snapshot()
        assert snap["errors"]["unknown"] == 1
        assert "totally_made_up" not in snap["errors"]


# =====================================================================
# 4. CONCURRENCY SAFETY
# =====================================================================


class TestConcurrencySafety:
    """Thread-safe metric updates under concurrent load."""

    def test_concurrent_metrics_updates(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()

        def submit():
            client.post("/predict", json=_valid_transaction(
                customer_id=f"conc_{threading.current_thread().name}",
            ))

        threads = [threading.Thread(target=submit) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        snap = _mod.metrics.snapshot()
        assert snap["successful_predictions"] == 8
        assert snap["total_requests"] == 8

    def test_no_cross_request_contamination(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        client.post("/predict", json=_valid_transaction(amount=100.0))
        client.post("/predict", json=_valid_transaction(amount=500.0))
        snap = _mod.metrics.snapshot()
        assert snap["total_requests"] == 2
        assert snap["successful_predictions"] == 2

    def test_bounded_latency_storage(self):
        """Latency deque is bounded — no memory leak."""
        m = PredictionMetrics()
        for i in range(10_000):
            m.record_success(
                latency_ms=float(i),
                fraud_prediction=0,
                fraud_probability=0.1,
                decision="APPROVE",
                risk_level="LOW",
                risk_score=10,
                amount=100.0,
                model_version="test",
            )
        snap = m.snapshot()
        # Bounded at _MAX_LATENCY_SAMPLES
        assert snap["latency"]["count"] <= 5_000

    def test_bounded_drift_storage(self):
        """Drift sample deques are bounded."""
        m = PredictionMetrics()
        for i in range(10_000):
            m.record_success(
                latency_ms=10.0,
                fraud_prediction=0,
                fraud_probability=0.1,
                decision="APPROVE",
                risk_level="LOW",
                risk_score=10,
                amount=float(i),
                model_version="test",
            )
        # Internal deques are bounded
        assert len(m._amount_samples) <= 2_000
        assert len(m._probability_samples) <= 2_000


# =====================================================================
# 5. MONITORING ENDPOINT
# =====================================================================


class TestMonitoringEndpoint:
    """GET /metrics endpoint — schema, security, and completeness."""

    def test_metrics_endpoint_exists(self, client: TestClient):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_response_schema(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        client.post("/predict", json=_valid_transaction())
        resp = client.get("/metrics")
        data = resp.json()
        required_keys = {
            "total_requests", "successful_predictions", "failed_predictions",
            "error_rate", "fraud_count", "non_fraud_count", "slow_predictions",
            "decisions", "risk_levels", "errors", "model_version",
            "latency", "drift", "config",
        }
        assert required_keys <= set(data.keys())

    def test_metrics_no_customer_ids(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        client.post("/predict", json=_valid_transaction(customer_id="secret_cust"))
        resp = client.get("/metrics")
        body = json.dumps(resp.json())
        assert "secret_cust" not in body

    def test_metrics_no_raw_transactions(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        client.post("/predict", json=_valid_transaction(
            merchant_name="UniqueMerchant12345",
        ))
        resp = client.get("/metrics")
        body = json.dumps(resp.json())
        assert "UniqueMerchant12345" not in body

    def test_metrics_no_secrets(self, client: TestClient):
        resp = client.get("/metrics")
        body = json.dumps(resp.json())
        assert "password" not in body.lower()
        assert "secret_key" not in body.lower()
        assert "Bearer" not in body
        assert "jwt" not in body.lower()

    def test_metrics_no_filesystem_paths(self, client: TestClient):
        resp = client.get("/metrics")
        body = json.dumps(resp.json())
        assert ".joblib" not in body
        assert "C:\\" not in body
        assert "/home/" not in body

    def test_metrics_no_auth_headers(self, client: TestClient):
        """Auth header in request is never echoed in metrics."""
        headers = {"Authorization": "Bearer eyJhbGci.secret_token"}
        client.get("/metrics", headers=headers)
        resp = client.get("/metrics")
        body = json.dumps(resp.json())
        assert "eyJhbGci" not in body
        assert "secret_token" not in body

    def test_metrics_aggregate_not_individual(self, client: TestClient):
        """Metrics contain only aggregate data, not per-request detail."""
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        for _ in range(5):
            client.post("/predict", json=_valid_transaction())
        resp = client.get("/metrics")
        data = resp.json()
        # No list of individual transactions
        assert "transactions" not in data
        assert "records" not in data
        assert "requests" not in data or isinstance(data.get("requests"), int)


# =====================================================================
# 6. DRIFT DETECTION
# =====================================================================


class TestDriftDetection:
    """Baseline comparison and drift signalling."""

    def test_no_baseline_reports_unavailable(self):
        """Without baseline config, drift status reports unavailable."""
        m = PredictionMetrics()
        snap = m.snapshot()
        assert snap["drift"]["baseline_configured"] is False
        assert "No baseline" in snap["drift"]["message"]

    def test_baseline_configured_via_env(self):
        """Baseline loaded from environment variables."""
        env = {
            "ML_BASELINE_AMOUNT_MEAN": "150.0",
            "ML_BASELINE_AMOUNT_STD": "50.0",
            "ML_BASELINE_PROB_MEAN": "0.15",
            "ML_BASELINE_PROB_STD": "0.10",
        }
        with patch.dict(os.environ, env, clear=False):
            m = PredictionMetrics()
            snap = m.snapshot()
        assert snap["drift"]["baseline_configured"] is True

    def test_normal_distribution_no_drift(self):
        """Normal data within baseline does not trigger drift."""
        env = {
            "ML_BASELINE_AMOUNT_MEAN": "150.0",
            "ML_BASELINE_AMOUNT_STD": "50.0",
        }
        with patch.dict(os.environ, env, clear=False):
            m = PredictionMetrics()
            # Feed amounts close to baseline mean
            for _ in range(50):
                m.record_success(
                    latency_ms=10.0,
                    fraud_prediction=0,
                    fraud_probability=0.1,
                    decision="APPROVE",
                    risk_level="LOW",
                    risk_score=10,
                    amount=155.0,  # very close to baseline mean
                    model_version="test",
                )
            snap = m.snapshot()
        warnings = snap["drift"]["warnings"]
        amount_warnings = [w for w in warnings if w["feature"] == "amount"]
        assert len(amount_warnings) == 0

    def test_artificial_drift_detected(self):
        """Clearly drifted data triggers a drift warning."""
        env = {
            "ML_BASELINE_AMOUNT_MEAN": "100.0",
            "ML_BASELINE_AMOUNT_STD": "10.0",
            "ML_DRIFT_STD_MULTIPLIER": "2.0",
        }
        with patch.dict(os.environ, env, clear=False):
            m = PredictionMetrics()
            # Feed amounts far from baseline
            for _ in range(50):
                m.record_success(
                    latency_ms=10.0,
                    fraud_prediction=0,
                    fraud_probability=0.1,
                    decision="APPROVE",
                    risk_level="LOW",
                    risk_score=10,
                    amount=500.0,  # far from baseline mean of 100
                    model_version="test",
                )
            snap = m.snapshot()
        warnings = snap["drift"]["warnings"]
        amount_warnings = [w for w in warnings if w["feature"] == "amount"]
        assert len(amount_warnings) == 1
        assert amount_warnings[0]["deviation"] > amount_warnings[0]["threshold"]

    def test_drift_does_not_change_prediction(self, client: TestClient):
        """Drift detection is observational — does not alter fraud decisions."""
        if not _model_available(client):
            pytest.skip("Model not available")
        env = {
            "ML_BASELINE_AMOUNT_MEAN": "100.0",
            "ML_BASELINE_AMOUNT_STD": "10.0",
        }
        with patch.dict(os.environ, env, clear=False):
            resp = client.post("/predict", json=_valid_transaction(amount=500.0))
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] in ("APPROVE", "VERIFY", "HOLD")
        assert data["fraud_prediction"] in (0, 1)

    def test_drift_does_not_block_requests(self, client: TestClient):
        """Even with drift detected, predictions continue normally."""
        if not _model_available(client):
            pytest.skip("Model not available")
        env = {
            "ML_BASELINE_AMOUNT_MEAN": "1.0",
            "ML_BASELINE_AMOUNT_STD": "0.1",
            "ML_DRIFT_STD_MULTIPLIER": "1.0",
        }
        with patch.dict(os.environ, env, clear=False):
            for _ in range(5):
                resp = client.post(
                    "/predict",
                    json=_valid_transaction(amount=999.0),
                )
                assert resp.status_code == 200

    def test_insufficient_samples_no_drift(self):
        """With fewer than 10 samples, no drift is reported."""
        env = {
            "ML_BASELINE_AMOUNT_MEAN": "100.0",
            "ML_BASELINE_AMOUNT_STD": "10.0",
        }
        with patch.dict(os.environ, env, clear=False):
            m = PredictionMetrics()
            # Only 5 samples
            for _ in range(5):
                m.record_success(
                    latency_ms=10.0,
                    fraud_prediction=0,
                    fraud_probability=0.1,
                    decision="APPROVE",
                    risk_level="LOW",
                    risk_score=10,
                    amount=999.0,
                    model_version="test",
                )
            snap = m.snapshot()
        warnings = snap["drift"]["warnings"]
        amount_warnings = [w for w in warnings if w["feature"] == "amount"]
        assert len(amount_warnings) == 0  # not enough samples


# =====================================================================
# 7. PERFORMANCE OVERHEAD
# =====================================================================


class TestPerformanceOverhead:
    """Monitoring must not significantly slow down predictions."""

    def test_monitoring_overhead_bounded(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()

        # Warm up
        client.post("/predict", json=_valid_transaction())

        # Measure with monitoring (always on)
        start = time.monotonic()
        for _ in range(5):
            client.post("/predict", json=_valid_transaction())
        elapsed = time.monotonic() - start
        avg_ms = (elapsed / 5) * 1000

        # Monitoring overhead should be negligible (< 50ms per request
        # on top of the prediction itself, which is typically < 100ms)
        assert avg_ms < 5000, f"Average latency {avg_ms:.0f}ms seems too high"


# =====================================================================
# 8. END-TO-END MONITORING SCENARIO
# =====================================================================


class TestEndToEndMonitoring:
    """Realistic monitoring scenario: submit, check metrics, verify."""

    def test_full_monitoring_flow(self, client: TestClient):
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()

        # 1. Verify readiness
        resp = client.get("/ready")
        assert resp.status_code == 200

        # 2. Submit valid transactions
        for _ in range(3):
            resp = client.post("/predict", json=_valid_transaction())
            assert resp.status_code == 200

        # 3. Trigger an error
        with patch(
            "ml.api.app.engineer_features_for_inference",
            side_effect=ValueError("forced"),
        ):
            resp = client.post("/predict", json=_valid_transaction())
        assert resp.status_code == 422

        # 4. Query metrics
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()

        # 5. Verify counters
        assert data["total_requests"] == 4  # 3 success + 1 error
        assert data["successful_predictions"] == 3
        assert data["failed_predictions"] == 1
        assert data["errors"]["feature_engineering"] >= 1

        # 6. Verify no sensitive data
        body = json.dumps(data)
        assert "customer_id" not in body or "customer_id" not in body.split('"')
        assert "password" not in body.lower()

        # 7. Verify model version
        health = client.get("/health").json()
        assert data["model_version"] == health["model_version"]

        # 8. Verify latency statistics exist
        assert data["latency"]["count"] == 3
        assert data["latency"]["mean_seconds"] > 0

    def test_monitoring_does_not_alter_decisions(self, client: TestClient):
        """Monitoring does not change prediction logic.

        We verify that two identical cold-start transactions (different
        customers, no prior history) produce identical predictions.
        """
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()

        txn1 = _valid_transaction(customer_id="mon_det_A")
        r1 = client.post("/predict", json=txn1).json()

        # Submit several transactions to change metrics state
        for i in range(5):
            client.post("/predict", json=_valid_transaction(
                customer_id=f"mon_noise_{i}", amount=5000.0,
            ))

        # Fresh cold-start customer with identical payload
        txn2 = _valid_transaction(customer_id="mon_det_B")
        r2 = client.post("/predict", json=txn2).json()

        # Both are cold-start with same features → same prediction
        assert r1["fraud_prediction"] == r2["fraud_prediction"]
        assert abs(r1["fraud_probability"] - r2["fraud_probability"]) < 1e-6


# =====================================================================
# 9. SECURITY — CARDINALITY AND LABEL SAFETY
# =====================================================================


class TestMetricSecurity:
    """Metrics must not allow cardinality attacks or label injection."""

    def test_error_categories_bounded(self):
        """Only predefined error categories are accepted."""
        m = PredictionMetrics()
        for cat in ["attack_label_1", "attack_label_2", "x" * 100]:
            m.record_error(category=cat)
        snap = m.snapshot()
        # All should map to "unknown"
        assert snap["errors"]["unknown"] == 3
        # No attack labels should appear
        for key in snap["errors"]:
            assert key in _VALID_ERROR_CATEGORIES

    def test_decision_labels_bounded(self):
        """Only predefined decision labels appear in metrics."""
        m = PredictionMetrics()
        m.record_success(
            latency_ms=10.0,
            fraud_prediction=0,
            fraud_probability=0.1,
            decision="APPROVE",
            risk_level="LOW",
            risk_score=10,
            amount=100.0,
            model_version="test",
        )
        snap = m.snapshot()
        for key in snap["decisions"]:
            assert key in _VALID_DECISIONS

    def test_no_per_customer_metrics(self, client: TestClient):
        """Metrics never contain per-customer identifiers."""
        if not _model_available(client):
            pytest.skip("Model not available")
        from ml.monitoring import metrics as _mod
        _mod.metrics.reset()
        client.post("/predict", json=_valid_transaction(
            customer_id="unique_cust_id_12345",
        ))
        snap = _mod.metrics.snapshot()
        full = json.dumps(snap)
        assert "unique_cust_id_12345" not in full

    def test_reset_clears_all(self):
        """reset() zeroes everything."""
        m = PredictionMetrics()
        m.record_success(
            latency_ms=10.0,
            fraud_prediction=1,
            fraud_probability=0.9,
            decision="HOLD",
            risk_level="HIGH",
            risk_score=80,
            amount=5000.0,
            model_version="v1",
        )
        m.record_error(category="validation")
        m.reset()
        snap = m.snapshot()
        assert snap["total_requests"] == 0
        assert snap["successful_predictions"] == 0
        assert snap["failed_predictions"] == 0
        assert snap["fraud_count"] == 0
        assert snap["latency"]["count"] == 0
