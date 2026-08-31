"""Shared pytest fixtures for backend tests."""

import pytest


@pytest.fixture
def valid_transaction() -> dict:
    """A valid raw transaction payload."""
    return {
        "amount": 1500.00,
        "currency": "USD",
        "merchant_name": "Acme Electronics",
        "merchant_category": "5732",
        "transaction_type": "purchase",
        "location_country": "US",
        "location_city": "New York",
        "device_fingerprint": "abc123def456",
        "device_type": "mobile",
        "ip_address": "192.168.1.100",
    }
