"""Tests for the ML service client and transaction integration.

Uses ``httpx`` mock transport so tests run without a real ML service.
Covers:
  - Successful ML prediction with explanation
  - ML service unavailable (connection refused)
  - ML service timeout
  - ML 4xx / 5xx responses
  - Invalid ML response body
  - Transaction flow with fraud result
  - Transaction flow with ML failure
  - Health check
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import Response

from backend.services.ml_client import (
    MLServiceClient,
    MLServiceResponseError,
    MLServiceTimeoutError,
    MLServiceUnavailableError,
)
from backend.schemas import MLPredictionResponse


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def client() -> MLServiceClient:
    return MLServiceClient(base_url="http://test-ml:8001", timeout=2.0)


@pytest.fixture
def ml_success_response() -> dict:
    """Realistic ML service response with SHAP explanation."""
    return {
        "fraud_probability": 0.2191,
        "fraud_prediction": 0,
        "threshold": 0.50,
        "model_version": "fraud-xgb-v1.0.0",
        "explanation": [
            {"feature": "device_fingerprint", "importance": -1.014},
            {"feature": "amount_deviation", "importance": 0.35},
            {"feature": "is_new_device", "importance": 0.22},
            {"feature": "tx_velocity_7d", "importance": 0.18},
            {"feature": "merchant_category", "importance": -0.15},
            {"feature": "previous_suspicious_count", "importance": 0.12},
            {"feature": "amount", "importance": 0.10},
            {"feature": "location_is_new", "importance": -0.08},
            {"feature": "hour_of_day_sin", "importance": 0.06},
            {"feature": "avg_spend_30d", "importance": -0.05},
        ],
    }


# ── ML Client unit tests ─────────────────────────────────────────────


class TestMLServiceClientPredict:
    """Tests for MLServiceClient.predict()."""

    @pytest.mark.asyncio
    async def test_successful_prediction(
        self, client: MLServiceClient, ml_success_response: dict
    ):
        """Valid ML response is parsed correctly."""
        mock_resp = Response(200, json=ml_success_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.predict({"amount": 100})

        assert isinstance(result, MLPredictionResponse)
        assert result.fraud_probability == 0.2191
        assert result.fraud_prediction == 0
        assert result.threshold == 0.50
        assert result.model_version == "fraud-xgb-v1.0.0"
        assert result.explanation is not None
        assert len(result.explanation) == 10
        assert result.explanation[0].feature == "device_fingerprint"
        assert result.explanation[0].importance == -1.014

    @pytest.mark.asyncio
    async def test_connection_refused(self, client: MLServiceClient):
        """Connection refused raises MLServiceUnavailableError."""
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with pytest.raises(MLServiceUnavailableError):
                await client.predict({"amount": 100})

    @pytest.mark.asyncio
    async def test_timeout(self, client: MLServiceClient):
        """Timeout raises MLServiceTimeoutError."""
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ReadTimeout("Read timed out"),
        ):
            with pytest.raises(MLServiceTimeoutError):
                await client.predict({"amount": 100})

    @pytest.mark.asyncio
    async def test_503_model_unavailable(self, client: MLServiceClient):
        """ML 503 raises MLServiceResponseError with status_code=503."""
        mock_resp = Response(
            503,
            json={"detail": "Model not available"},
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(MLServiceResponseError) as exc_info:
                await client.predict({"amount": 100})
            assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_422_validation_error(self, client: MLServiceClient):
        """ML 422 raises MLServiceResponseError with status_code=422."""
        mock_resp = Response(
            422,
            json={"detail": "Missing required features"},
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(MLServiceResponseError) as exc_info:
                await client.predict({"amount": 100})
            assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_500_internal_error(self, client: MLServiceClient):
        """ML 500 raises MLServiceResponseError."""
        mock_resp = Response(500, json={"detail": "Internal server error"})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(MLServiceResponseError) as exc_info:
                await client.predict({"amount": 100})
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_invalid_json_response(self, client: MLServiceClient):
        """Non-JSON 200 response raises MLServiceResponseError."""
        mock_resp = Response(200, text="not json", headers={"content-type": "text/plain"})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(MLServiceResponseError):
                await client.predict({"amount": 100})

    @pytest.mark.asyncio
    async def test_partial_response_fields(self, client: MLServiceClient):
        """ML response with only current fields (no behaviour/rules)."""
        partial = {
            "fraud_probability": 0.75,
            "fraud_prediction": 1,
            "threshold": 0.50,
            "model_version": "fraud-xgb-v1.0.0",
        }
        mock_resp = Response(200, json=partial)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.predict({"amount": 100})
        assert result.fraud_probability == 0.75
        assert result.fraud_prediction == 1
        assert result.explanation is None
        assert result.ml_score is None
        assert result.risk_score is None


class TestMLServiceClientHealth:
    """Tests for MLServiceClient.health()."""

    @pytest.mark.asyncio
    async def test_healthy(self, client: MLServiceClient):
        health_data = {"status": "ready", "model_version": "fraud-xgb-v1.0.0", "features": 24}
        mock_resp = Response(200, json=health_data)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.health()
        assert result["status"] == "ready"
        assert result["model_version"] == "fraud-xgb-v1.0.0"

    @pytest.mark.asyncio
    async def test_health_unavailable(self, client: MLServiceClient):
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            with pytest.raises(MLServiceUnavailableError):
                await client.health()


# ── Transaction endpoint integration tests (TestClient) ────────────────


class TestTransactionEndpoint:
    """Integration tests for POST /api/v1/transactions using TestClient."""

    @pytest.fixture
    def test_client(self):
        """Create a TestClient with a mocked ML client."""
        from fastapi.testclient import TestClient
        from backend.app import app
        from backend.routers import transactions as txn_module

        # Create a real MLServiceClient (won't be used — we mock its predict method)
        ml_client = MLServiceClient(base_url="http://mock:8001")
        txn_module.set_ml_client(ml_client)

        return TestClient(app), ml_client

    def test_transaction_success(
        self, test_client, valid_transaction, ml_success_response
    ):
        """Successful transaction with fraud scoring."""
        test_client, ml_client = test_client

        mock_resp = Response(200, json=ml_success_response)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = test_client.post("/api/v1/transactions", json=valid_transaction)

        assert resp.status_code == 201
        data = resp.json()
        assert data["amount"] == 1500.00
        assert data["fraud_probability"] == 0.2191
        assert data["fraud_prediction"] == 0
        assert data["model_version"] == "fraud-xgb-v1.0.0"
        assert data["explanation"] is not None
        assert len(data["explanation"]["ml_top_factors"]) == 10
        assert data["explanation"]["ml_top_factors"][0]["feature"] == "device_fingerprint"

    def test_transaction_ml_unavailable(self, test_client, valid_transaction):
        """ML service unavailable returns 503."""
        test_client, ml_client = test_client

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            resp = test_client.post("/api/v1/transactions", json=valid_transaction)

        assert resp.status_code == 503

    def test_transaction_ml_timeout(self, test_client, valid_transaction):
        """ML service timeout returns 503."""
        test_client, ml_client = test_client

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ReadTimeout("timed out"),
        ):
            resp = test_client.post("/api/v1/transactions", json=valid_transaction)

        assert resp.status_code == 503

    def test_transaction_ml_503(self, test_client, valid_transaction):
        """ML 503 (model unavailable) returns 503."""
        test_client, ml_client = test_client

        mock_resp = Response(503, json={"detail": "Model not available"})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = test_client.post("/api/v1/transactions", json=valid_transaction)

        assert resp.status_code == 503

    def test_transaction_ml_422(self, test_client, valid_transaction):
        """ML 422 (validation error) returns 502."""
        test_client, ml_client = test_client

        mock_resp = Response(422, json={"detail": "Missing features"})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            resp = test_client.post("/api/v1/transactions", json=valid_transaction)

        assert resp.status_code == 502

    def test_transaction_invalid_input(self, test_client):
        """Invalid transaction input returns 422 (Pydantic validation)."""
        test_client, _ = test_client

        bad = {"amount": -1, "currency": "X"}  # Missing many required fields
        resp = test_client.post("/api/v1/transactions", json=bad)
        assert resp.status_code == 422

    def test_transaction_negative_amount(self, test_client, valid_transaction):
        """Negative amount returns 422."""
        test_client, _ = test_client

        bad = dict(valid_transaction)
        bad["amount"] = -100
        resp = test_client.post("/api/v1/transactions", json=bad)
        assert resp.status_code == 422

    def test_transaction_invalid_type(self, test_client, valid_transaction):
        """Invalid transaction_type returns 422."""
        test_client, _ = test_client

        bad = dict(valid_transaction)
        bad["transaction_type"] = "invalid"
        resp = test_client.post("/api/v1/transactions", json=bad)
        assert resp.status_code == 422

    def test_transaction_no_ml_client(self, valid_transaction):
        """No ML client configured returns 503."""
        from fastapi.testclient import TestClient
        from backend.app import app
        from backend.routers import transactions as txn_module

        txn_module.set_ml_client(None)
        test_client = TestClient(app)
        # Reset client after test
        try:
            resp = test_client.post("/api/v1/transactions", json=valid_transaction)
            # Should be 503 because client is None
            assert resp.status_code == 503
        finally:
            txn_module.set_ml_client(MLServiceClient(base_url="http://mock:8001"))
