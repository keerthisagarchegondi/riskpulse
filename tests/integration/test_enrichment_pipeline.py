"""Integration tests for the full enrichment pipeline.

Tests the complete enrichment flow: geo → device → merchant → velocity,
verifying that all enrichers work together and produce consistent output.
"""

import time
from datetime import datetime, timedelta, timezone

import pytest

from src.enrichment.device_enricher import DeviceEnricher, InMemoryDeviceStore
from src.enrichment.geo_enricher import GeoEnricher
from src.enrichment.merchant_enricher import InMemoryMerchantStore, MerchantEnricher
from src.enrichment.velocity_calculator import VelocityCalculator


class EnrichmentPipeline:
    """Orchestrates the full enrichment pipeline for integration testing.

    Runs all enrichers in sequence and merges results into a single
    enriched transaction record.
    """

    def __init__(
        self,
        geo_enricher: GeoEnricher,
        device_enricher: DeviceEnricher,
        merchant_enricher: MerchantEnricher,
        velocity_calculator: VelocityCalculator,
    ) -> None:
        self._geo = geo_enricher
        self._device = device_enricher
        self._merchant = merchant_enricher
        self._velocity = velocity_calculator

    def enrich(
        self,
        transaction: dict,
        customer_profile: dict | None = None,
        last_transaction: dict | None = None,
    ) -> dict:
        """Run full enrichment pipeline on a transaction.

        Returns the transaction dict augmented with all enrichment fields.
        """
        enriched = dict(transaction)

        # Geo enrichment
        geo_result = self._geo.enrich(transaction, customer_profile, last_transaction)
        enriched.update(geo_result.to_dict())

        # Device enrichment
        device_result = self._device.enrich(transaction, customer_profile)
        enriched.update(device_result.to_dict())

        # Merchant enrichment
        merchant_result = self._merchant.enrich(transaction)
        enriched.update(merchant_result.to_dict())

        # Velocity calculation
        velocity_result = self._velocity.evaluate(transaction)
        enriched.update(velocity_result.to_dict())

        return enriched


@pytest.fixture
def device_store():
    return InMemoryDeviceStore()


@pytest.fixture
def merchant_store():
    return InMemoryMerchantStore()


@pytest.fixture
def pipeline(device_store, merchant_store):
    return EnrichmentPipeline(
        geo_enricher=GeoEnricher(),
        device_enricher=DeviceEnricher(device_store=device_store),
        merchant_enricher=MerchantEnricher(merchant_store=merchant_store),
        velocity_calculator=VelocityCalculator(),
    )


@pytest.fixture
def normal_transaction():
    return {
        "external_transaction_id": "TXN-2026-INT-001",
        "account_id": "ACC-12345",
        "customer_id": "CUST-67890",
        "merchant_id": "MERCH-11111",
        "merchant_name": "Local Grocery Store",
        "merchant_category_code": "5411",
        "transaction_amount": 45.50,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "pos",
        "card_type": "debit",
        "card_last_four": "4242",
        "ip_address": "192.168.1.100",
        "device_id": "device-regular-001",
        "device_type": "mobile",
        "geo_latitude": 40.7128,
        "geo_longitude": -74.0060,
        "geo_country": "US",
        "geo_city": "New York",
        "is_international": False,
        "transaction_timestamp": "2026-06-15T10:30:00Z",
    }


@pytest.fixture
def suspicious_transaction():
    return {
        "external_transaction_id": "TXN-2026-INT-002",
        "account_id": "ACC-99999",
        "customer_id": "CUST-99999",
        "merchant_id": "MERCH-CASINO",
        "merchant_name": "Online Casino Royal",
        "merchant_category_code": "7995",
        "transaction_amount": 5000.00,
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


class TestEnrichmentPipelineIntegration:
    """Integration tests for the complete enrichment pipeline."""

    def test_normal_transaction_full_enrichment(self, pipeline, normal_transaction):
        """Normal transaction should pass through all enrichers without high risk."""
        result = pipeline.enrich(normal_transaction)

        # Geo fields present
        assert result["geo_country_code"] == "US"
        assert result["geo_is_high_risk_country"] is False
        assert result["geo_is_impossible_travel"] is False

        # Device fields present
        assert result["device_id"] == "device-regular-001"
        assert result["device_type"] == "mobile"
        assert result["device_is_new"] is True  # First time

        # Merchant fields present
        assert result["merchant_mcc_risk_score"] <= 0.3
        assert result["merchant_is_new"] is True

        # Velocity fields present
        assert result["velocity_txn_count_1min"] == 1
        assert result["velocity_risk_score"] == 0.0

    def test_suspicious_transaction_high_risk(self, pipeline, suspicious_transaction):
        """Suspicious transaction should flag multiple risk indicators."""
        profile = {"home_latitude": 40.7128, "home_longitude": -74.0060}
        result = pipeline.enrich(suspicious_transaction, customer_profile=profile)

        # High-risk country
        assert result["geo_country_risk_score"] >= 0.5

        # Large distance from home
        assert result["geo_distance_from_home_miles"] is not None
        assert result["geo_distance_from_home_miles"] > 4000

        # High-risk MCC (gambling)
        assert result["merchant_mcc_risk_score"] >= 0.7

        # New device
        assert result["device_is_new"] is True
        assert result["device_is_known"] is False

    def test_impossible_travel_detection(self, pipeline):
        """Detect impossible travel between two transactions."""
        # First transaction in New York
        txn1 = {
            "external_transaction_id": "TXN-TRAVEL-1",
            "customer_id": "CUST-TRAVELER",
            "merchant_id": "MERCH-NY",
            "merchant_name": "NY Store",
            "merchant_category_code": "5411",
            "transaction_amount": 50.0,
            "device_id": "device-travel",
            "device_type": "mobile",
            "ip_address": "1.1.1.1",
            "geo_latitude": 40.7128,
            "geo_longitude": -74.0060,
            "geo_country": "US",
            "geo_city": "New York",
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }

        # Second transaction in Moscow 30 minutes later
        txn2 = {
            "external_transaction_id": "TXN-TRAVEL-2",
            "customer_id": "CUST-TRAVELER",
            "merchant_id": "MERCH-RU",
            "merchant_name": "Moscow Store",
            "merchant_category_code": "5411",
            "transaction_amount": 200.0,
            "device_id": "device-travel",
            "device_type": "mobile",
            "ip_address": "2.2.2.2",
            "geo_latitude": 55.7558,
            "geo_longitude": 37.6173,
            "geo_country": "RU",
            "geo_city": "Moscow",
            "transaction_timestamp": "2026-06-15T10:30:00Z",
        }

        # Enrich first
        pipeline.enrich(txn1)

        # Enrich second with last_transaction context
        last_txn = {
            "geo_latitude": 40.7128,
            "geo_longitude": -74.0060,
            "transaction_timestamp": "2026-06-15T10:00:00Z",
            "_current_timestamp": "2026-06-15T10:30:00Z",
        }
        result = pipeline.enrich(txn2, last_transaction=last_txn)

        assert result["geo_is_impossible_travel"] is True
        assert result["geo_travel_speed_mph"] > 500

    def test_velocity_breach_after_rapid_transactions(self, pipeline):
        """Rapid transactions should trigger velocity breaches."""
        base_ts = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        results = []

        for i in range(6):
            txn = {
                "external_transaction_id": f"TXN-RAPID-{i}",
                "customer_id": "CUST-RAPID",
                "merchant_id": f"MERCH-{i}",
                "merchant_name": f"Store {i}",
                "merchant_category_code": "5411",
                "transaction_amount": 100.0,
                "device_id": "device-rapid",
                "device_type": "mobile",
                "ip_address": "1.1.1.1",
                "geo_latitude": 40.7128,
                "geo_longitude": -74.0060,
                "geo_country": "US",
                "geo_city": "New York",
                "transaction_timestamp": (base_ts + timedelta(seconds=i * 8)).isoformat(),
            }
            results.append(pipeline.enrich(txn))

        last_result = results[-1]
        assert last_result["velocity_txn_count_1min"] == 6
        assert last_result["velocity_has_critical"] is True
        assert last_result["velocity_risk_score"] > 0.0

    def test_device_becomes_known_over_time(self, pipeline):
        """A device should transition from new to known after repeated use."""
        base_ts = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)

        txn_template = {
            "customer_id": "CUST-REPEAT",
            "merchant_id": "MERCH-001",
            "merchant_name": "Regular Store",
            "merchant_category_code": "5411",
            "transaction_amount": 25.0,
            "device_id": "device-my-phone",
            "device_type": "mobile",
            "ip_address": "1.1.1.1",
            "geo_latitude": 40.7128,
            "geo_longitude": -74.0060,
            "geo_country": "US",
            "geo_city": "New York",
        }

        # First transaction - device is new
        txn1 = {
            **txn_template,
            "external_transaction_id": "TXN-REPEAT-1",
            "transaction_timestamp": (base_ts + timedelta(hours=1)).isoformat(),
        }
        result1 = pipeline.enrich(txn1)
        assert result1["device_is_new"] is True

        # Second transaction - device is now known
        txn2 = {
            **txn_template,
            "external_transaction_id": "TXN-REPEAT-2",
            "transaction_timestamp": (base_ts + timedelta(hours=2)).isoformat(),
        }
        result2 = pipeline.enrich(txn2)
        assert result2["device_is_known"] is True

    def test_multi_account_device_detection(self, pipeline, device_store):
        """Detect when multiple customers use the same device."""
        base_ts = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)

        for i in range(4):
            txn = {
                "external_transaction_id": f"TXN-MULTI-{i}",
                "customer_id": f"CUST-{i:03d}",
                "merchant_id": "MERCH-001",
                "merchant_name": "Store",
                "merchant_category_code": "5411",
                "transaction_amount": 50.0,
                "device_id": "shared-device-id",
                "device_type": "desktop",
                "ip_address": "1.1.1.1",
                "geo_latitude": 40.7128,
                "geo_longitude": -74.0060,
                "geo_country": "US",
                "geo_city": "New York",
                "transaction_timestamp": (base_ts + timedelta(hours=i)).isoformat(),
            }
            result = pipeline.enrich(txn)

        # Last transaction should detect multi-account
        assert result["device_is_multi_account"] is True
        assert result["device_accounts_count"] >= 3

    def test_merchant_fraud_rate_tracking(self, pipeline, merchant_store):
        """Merchant fraud rate should be tracked and reported."""
        merchant_id = "MERCH-FRAUD-TRACK"
        first_seen = datetime.now(timezone.utc) - timedelta(days=90)
        merchant_store._merchants[merchant_id] = {
            "merchant_name": "Risky Merchant",
            "mcc": "5732",
            "first_seen": first_seen,
            "last_seen": datetime.now(timezone.utc),
            "total_transactions": 200,
            "fraud_count": 15,
            "fraud_rate": 0.075,
        }

        txn = {
            "external_transaction_id": "TXN-FRAUD-TRACK",
            "customer_id": "CUST-001",
            "merchant_id": merchant_id,
            "merchant_name": "Risky Merchant",
            "merchant_category_code": "5732",
            "transaction_amount": 300.0,
            "device_id": "device-001",
            "device_type": "mobile",
            "ip_address": "1.1.1.1",
            "geo_latitude": 40.7128,
            "geo_longitude": -74.0060,
            "geo_country": "US",
            "geo_city": "New York",
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }
        result = pipeline.enrich(txn)

        assert result["merchant_fraud_rate"] >= 0.05
        assert result["merchant_is_high_fraud"] is True
        assert result["merchant_is_new"] is False

    def test_enrichment_output_completeness(self, pipeline, normal_transaction):
        """Verify all expected enrichment fields are present in output."""
        result = pipeline.enrich(normal_transaction)

        # Geo fields
        geo_fields = [
            "geo_country_code", "geo_city", "geo_latitude", "geo_longitude",
            "geo_is_vpn", "geo_is_proxy", "geo_is_tor",
            "geo_distance_from_home_miles", "geo_country_risk_score",
            "geo_is_high_risk_country", "geo_travel_speed_mph",
            "geo_is_impossible_travel",
        ]
        for field in geo_fields:
            assert field in result, f"Missing geo field: {field}"

        # Device fields
        device_fields = [
            "device_id", "device_type", "device_fingerprint_hash",
            "device_is_known", "device_age_days", "device_is_new",
            "device_trust_score", "device_accounts_count",
            "device_is_multi_account", "device_risk_score",
        ]
        for field in device_fields:
            assert field in result, f"Missing device field: {field}"

        # Merchant fields
        merchant_fields = [
            "merchant_mcc_risk_score", "merchant_fraud_rate",
            "merchant_is_high_fraud", "merchant_is_new",
            "merchant_risk_score",
        ]
        for field in merchant_fields:
            assert field in result, f"Missing merchant field: {field}"

        # Velocity fields
        velocity_fields = [
            "velocity_txn_count_1min", "velocity_txn_count_5min",
            "velocity_txn_count_1hour", "velocity_amount_sum_1min",
            "velocity_risk_score", "velocity_breach_count",
            "velocity_has_warning", "velocity_has_critical",
        ]
        for field in velocity_fields:
            assert field in result, f"Missing velocity field: {field}"

    def test_pipeline_performance(self, pipeline, normal_transaction):
        """Full pipeline should complete within acceptable latency."""
        start = time.perf_counter()
        for _ in range(100):
            pipeline.enrich(normal_transaction)
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_ms = elapsed_ms / 100
        # Should be well under 50ms per transaction
        assert avg_ms < 50, f"Average enrichment latency {avg_ms:.2f}ms exceeds 50ms target"

    def test_pipeline_handles_missing_fields_gracefully(self, pipeline):
        """Pipeline should handle minimal transactions without errors."""
        minimal_txn = {
            "external_transaction_id": "TXN-MINIMAL",
            "customer_id": "CUST-001",
            "transaction_amount": 10.0,
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }
        result = pipeline.enrich(minimal_txn)

        # Should not raise, all fields should have defaults
        assert "geo_country_code" in result
        assert "device_risk_score" in result
        assert "merchant_risk_score" in result
        assert "velocity_txn_count_1min" in result

    def test_combined_risk_indicators(self, pipeline):
        """Transaction with multiple risk indicators should flag all of them."""
        risky_txn = {
            "external_transaction_id": "TXN-RISKY",
            "customer_id": "CUST-RISKY",
            "merchant_id": "MERCH-CASINO-2",
            "merchant_name": "Shady Casino",
            "merchant_category_code": "7995",  # Gambling
            "transaction_amount": 9999.00,
            "device_id": "",  # Missing device
            "device_type": "unknown",
            "ip_address": "185.220.101.1",
            "geo_latitude": 55.7558,
            "geo_longitude": 37.6173,
            "geo_country": "RU",  # High-risk country
            "geo_city": "Moscow",
            "transaction_timestamp": "2026-06-15T03:00:00Z",
        }
        profile = {"home_latitude": 40.7128, "home_longitude": -74.0060}
        result = pipeline.enrich(risky_txn, customer_profile=profile)

        # Multiple risk indicators should be triggered
        assert result["geo_country_risk_score"] >= 0.5
        assert result["merchant_mcc_risk_score"] >= 0.7
        assert result["device_risk_score"] >= 0.7
        assert result["geo_distance_from_home_miles"] > 4000
