"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_transaction():
    """Return a sample valid transaction for testing."""
    return {
        "external_transaction_id": "TXN-2026-001",
        "account_id": "ACC-12345",
        "customer_id": "CUST-67890",
        "merchant_id": "MERCH-11111",
        "merchant_name": "Amazon Online Store",
        "merchant_category_code": "5411",
        "transaction_amount": 125.50,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": "credit",
        "card_last_four": "4242",
        "ip_address": "192.168.1.100",
        "device_id": "device-abc-123",
        "device_type": "mobile",
        "geo_latitude": 40.7128,
        "geo_longitude": -74.0060,
        "geo_country": "US",
        "geo_city": "New York",
        "is_international": False,
        "transaction_timestamp": "2026-06-15T10:30:00Z",
    }


@pytest.fixture
def sample_fraud_transaction():
    """Return a sample fraudulent transaction for testing."""
    return {
        "external_transaction_id": "TXN-2026-FRAUD-001",
        "account_id": "ACC-99999",
        "customer_id": "CUST-99999",
        "merchant_id": "MERCH-SUSPICIOUS",
        "merchant_name": "Unknown Merchant",
        "merchant_category_code": "7995",
        "transaction_amount": 9999.00,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": "credit",
        "card_last_four": "1111",
        "ip_address": "185.220.101.1",
        "device_id": "device-new-unknown",
        "device_type": "desktop",
        "geo_latitude": 55.7558,
        "geo_longitude": 37.6173,
        "geo_country": "RU",
        "geo_city": "Moscow",
        "is_international": True,
        "transaction_timestamp": "2026-06-15T03:15:00Z",
    }


@pytest.fixture
def kafka_config():
    """Return Kafka configuration for testing."""
    return {
        "bootstrap_servers": "localhost:9092",
        "consumer_group": "riskpulse-test",
        "auto_offset_reset": "earliest",
    }


@pytest.fixture
def postgres_config():
    """Return PostgreSQL configuration for testing."""
    return {
        "host": "localhost",
        "port": 5432,
        "database": "riskpulse_test",
        "user": "riskpulse",
        "password": "riskpulse_dev_password",
    }
