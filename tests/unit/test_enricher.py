"""Unit tests for the data enrichment pipeline.

Tests cover:
- GeoEnricher: IP geolocation, country risk, travel velocity
- DeviceEnricher: Device fingerprint, known device, multi-account
- MerchantEnricher: MCC risk, fraud rate, new merchant detection
- VelocityCalculator: Windowed counts, threshold breaches
"""

import time
from datetime import datetime, timedelta, timezone

import pytest

from src.enrichment.device_enricher import (
    DeviceEnricher,
    InMemoryDeviceStore,
)
from src.enrichment.geo_enricher import (
    GeoEnricher,
    haversine_distance,
)
from src.enrichment.merchant_enricher import (
    InMemoryMerchantStore,
    MerchantEnricher,
)
from src.enrichment.velocity_calculator import (
    CustomerVelocityProfile,
    VelocityCalculator,
)

# =============================================================================
# Geo Enricher Tests
# =============================================================================


class TestHaversineDistance:
    """Test haversine distance calculation."""

    def test_same_point_returns_zero(self):
        distance = haversine_distance(40.7128, -74.0060, 40.7128, -74.0060)
        assert distance == 0.0

    def test_new_york_to_london(self):
        # NYC to London is approximately 3459 miles
        distance = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
        assert 3400 < distance < 3500

    def test_new_york_to_los_angeles(self):
        # NYC to LA is approximately 2451 miles
        distance = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        assert 2400 < distance < 2500

    def test_antipodal_points(self):
        # Max distance is half circumference (~12,450 miles)
        distance = haversine_distance(0, 0, 0, 180)
        assert 12400 < distance < 12500


class TestGeoEnricher:
    """Test GeoEnricher functionality."""

    @pytest.fixture
    def enricher(self):
        return GeoEnricher()

    @pytest.fixture
    def us_transaction(self):
        return {
            "external_transaction_id": "TXN-001",
            "customer_id": "CUST-001",
            "ip_address": "8.8.8.8",
            "geo_latitude": 40.7128,
            "geo_longitude": -74.0060,
            "geo_country": "US",
            "geo_city": "New York",
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }

    @pytest.fixture
    def ru_transaction(self):
        return {
            "external_transaction_id": "TXN-002",
            "customer_id": "CUST-001",
            "ip_address": "185.220.101.1",
            "geo_latitude": 55.7558,
            "geo_longitude": 37.6173,
            "geo_country": "RU",
            "geo_city": "Moscow",
            "transaction_timestamp": "2026-06-15T10:30:00Z",
        }

    def test_basic_enrichment(self, enricher, us_transaction):
        result = enricher.enrich(us_transaction)
        assert result.is_success
        assert result.location is not None
        assert result.location.country_code == "US"
        assert result.location.city == "New York"

    def test_country_risk_scoring_low(self, enricher, us_transaction):
        result = enricher.enrich(us_transaction)
        assert result.country_risk_score <= 0.3
        assert result.is_high_risk_country is False

    def test_country_risk_scoring_high(self, enricher, ru_transaction):
        result = enricher.enrich(ru_transaction)
        assert result.country_risk_score >= 0.5
        assert result.is_high_risk_country is False or result.country_risk_score >= 0.7

    def test_distance_from_home(self, enricher, us_transaction):
        profile = {"home_latitude": 40.7128, "home_longitude": -74.0060}
        result = enricher.enrich(us_transaction, customer_profile=profile)
        assert result.distance_from_home_miles is not None
        assert result.distance_from_home_miles < 1.0  # Same location

    def test_distance_from_home_far(self, enricher, ru_transaction):
        profile = {"home_latitude": 40.7128, "home_longitude": -74.0060}
        result = enricher.enrich(ru_transaction, customer_profile=profile)
        assert result.distance_from_home_miles is not None
        assert result.distance_from_home_miles > 4000  # NYC to Moscow

    def test_impossible_travel_detected(self, enricher):
        # Transaction in NYC
        last_txn = {
            "geo_latitude": 40.7128,
            "geo_longitude": -74.0060,
            "transaction_timestamp": "2026-06-15T10:00:00Z",
            "_current_timestamp": "2026-06-15T10:30:00Z",
        }
        # Transaction in London 30 minutes later (~3459 miles / 0.5 hours = ~6918 mph)
        current_txn = {
            "external_transaction_id": "TXN-003",
            "ip_address": "1.1.1.1",
            "geo_latitude": 51.5074,
            "geo_longitude": -0.1278,
            "geo_country": "GB",
            "geo_city": "London",
            "transaction_timestamp": "2026-06-15T10:30:00Z",
        }
        result = enricher.enrich(current_txn, last_transaction=last_txn)
        assert result.is_impossible_travel is True
        assert result.travel_speed_mph is not None
        assert result.travel_speed_mph > 500

    def test_normal_travel_not_flagged(self, enricher):
        # Transaction in NYC
        last_txn = {
            "geo_latitude": 40.7128,
            "geo_longitude": -74.0060,
            "transaction_timestamp": "2026-06-15T10:00:00Z",
            "_current_timestamp": "2026-06-15T18:00:00Z",
        }
        # Transaction in DC 8 hours later (~225 miles / 8 hours = ~28 mph)
        current_txn = {
            "external_transaction_id": "TXN-004",
            "ip_address": "1.1.1.1",
            "geo_latitude": 38.9072,
            "geo_longitude": -77.0369,
            "geo_country": "US",
            "geo_city": "Washington DC",
            "transaction_timestamp": "2026-06-15T18:00:00Z",
        }
        result = enricher.enrich(current_txn, last_transaction=last_txn)
        assert result.is_impossible_travel is False

    def test_missing_geo_data(self, enricher):
        txn = {
            "external_transaction_id": "TXN-005",
            "ip_address": "10.0.0.1",
        }
        result = enricher.enrich(txn)
        assert result.is_success
        assert result.location is None

    def test_to_dict(self, enricher, us_transaction):
        result = enricher.enrich(us_transaction)
        data = result.to_dict()
        assert "geo_country_code" in data
        assert "geo_is_impossible_travel" in data
        assert data["geo_country_code"] == "US"

    def test_batch_enrichment(self, enricher, us_transaction, ru_transaction):
        results = enricher.enrich_batch([us_transaction, ru_transaction])
        assert len(results) == 2
        assert results[0].location.country_code == "US"
        assert results[1].location.country_code == "RU"


# =============================================================================
# Device Enricher Tests
# =============================================================================


class TestDeviceEnricher:
    """Test DeviceEnricher functionality."""

    @pytest.fixture
    def store(self):
        return InMemoryDeviceStore()

    @pytest.fixture
    def enricher(self, store):
        return DeviceEnricher(device_store=store)

    @pytest.fixture
    def mobile_transaction(self):
        return {
            "external_transaction_id": "TXN-001",
            "customer_id": "CUST-001",
            "device_id": "device-abc-123",
            "device_type": "mobile",
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }

    def test_new_device_detection(self, enricher, mobile_transaction):
        result = enricher.enrich(mobile_transaction)
        assert result.is_success
        assert result.is_new_device is True
        assert result.is_known_device is False

    def test_known_device_after_repeat(self, enricher, mobile_transaction):
        # First transaction registers the device
        enricher.enrich(mobile_transaction)
        # Second transaction should see it as known
        result = enricher.enrich(mobile_transaction)
        assert result.is_known_device is True

    def test_device_type_classification(self, enricher, mobile_transaction):
        result = enricher.enrich(mobile_transaction)
        assert result.device_info is not None
        assert result.device_info.device_type == "mobile"

    def test_unknown_device_type(self, enricher):
        txn = {
            "external_transaction_id": "TXN-002",
            "customer_id": "CUST-001",
            "device_id": "device-xyz",
            "device_type": "smartwatch",
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }
        result = enricher.enrich(txn)
        assert result.device_info.device_type == "unknown"

    def test_multi_account_detection(self, enricher, store):
        # Three different customers using same device
        for i in range(3):
            txn = {
                "external_transaction_id": f"TXN-{i}",
                "customer_id": f"CUST-{i:03d}",
                "device_id": "shared-device",
                "device_type": "desktop",
                "transaction_timestamp": "2026-06-15T10:00:00Z",
            }
            enricher.enrich(txn)

        # Fourth customer on same device triggers multi-account
        txn = {
            "external_transaction_id": "TXN-MULTI",
            "customer_id": "CUST-NEW",
            "device_id": "shared-device",
            "device_type": "desktop",
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }
        result = enricher.enrich(txn)
        assert result.is_multi_account_device is True
        assert result.accounts_on_device >= 3

    def test_missing_device_id(self, enricher):
        txn = {
            "external_transaction_id": "TXN-003",
            "customer_id": "CUST-001",
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }
        result = enricher.enrich(txn)
        assert result.device_risk_score >= 0.7  # High risk for missing device

    def test_device_fingerprint_hash(self, enricher, mobile_transaction):
        result = enricher.enrich(mobile_transaction)
        assert result.device_info is not None
        assert len(result.device_info.fingerprint_hash) == 16  # Truncated SHA256

    def test_device_risk_score_new_device(self, enricher, mobile_transaction):
        result = enricher.enrich(mobile_transaction)
        assert result.device_risk_score > 0.0  # New device has some risk

    def test_to_dict(self, enricher, mobile_transaction):
        result = enricher.enrich(mobile_transaction)
        data = result.to_dict()
        assert "device_id" in data
        assert "device_is_known" in data
        assert "device_risk_score" in data

    def test_batch_enrichment(self, enricher, mobile_transaction):
        txns = [mobile_transaction, {**mobile_transaction, "customer_id": "CUST-002"}]
        results = enricher.enrich_batch(txns)
        assert len(results) == 2


# =============================================================================
# Merchant Enricher Tests
# =============================================================================


class TestMerchantEnricher:
    """Test MerchantEnricher functionality."""

    @pytest.fixture
    def store(self):
        return InMemoryMerchantStore()

    @pytest.fixture
    def enricher(self, store):
        return MerchantEnricher(merchant_store=store)

    @pytest.fixture
    def grocery_transaction(self):
        return {
            "external_transaction_id": "TXN-001",
            "merchant_id": "MERCH-GROCERY",
            "merchant_name": "Local Supermarket",
            "merchant_category_code": "5411",
            "transaction_amount": 45.00,
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }

    @pytest.fixture
    def gambling_transaction(self):
        return {
            "external_transaction_id": "TXN-002",
            "merchant_id": "MERCH-GAMBLING",
            "merchant_name": "Online Casino",
            "merchant_category_code": "7995",
            "transaction_amount": 500.00,
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }

    def test_low_risk_mcc(self, enricher, grocery_transaction):
        result = enricher.enrich(grocery_transaction)
        assert result.is_success
        assert result.mcc_risk_score <= 0.3
        assert result.merchant_info.category_name == "Grocery Stores/Supermarkets"

    def test_high_risk_mcc(self, enricher, gambling_transaction):
        result = enricher.enrich(gambling_transaction)
        assert result.is_success
        assert result.mcc_risk_score >= 0.7
        assert result.merchant_info.risk_category == "high"

    def test_new_merchant_detection(self, enricher, grocery_transaction):
        result = enricher.enrich(grocery_transaction)
        assert result.is_new_merchant is True

    def test_known_merchant_after_transactions(self, enricher, store):
        # Pre-populate merchant history
        merchant_id = "MERCH-KNOWN"
        first_seen = datetime.now(timezone.utc) - timedelta(days=60)
        store._merchants[merchant_id] = {
            "merchant_name": "Known Store",
            "mcc": "5411",
            "first_seen": first_seen,
            "last_seen": datetime.now(timezone.utc),
            "total_transactions": 1000,
            "fraud_count": 2,
            "fraud_rate": 0.002,
        }

        txn = {
            "external_transaction_id": "TXN-003",
            "merchant_id": merchant_id,
            "merchant_name": "Known Store",
            "merchant_category_code": "5411",
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }
        result = enricher.enrich(txn)
        assert result.is_new_merchant is False
        assert result.merchant_age_days > 30

    def test_high_fraud_rate_merchant(self, enricher, store):
        merchant_id = "MERCH-FRAUD"
        store._merchants[merchant_id] = {
            "merchant_name": "Fraudy Merchant",
            "mcc": "5999",
            "first_seen": datetime.now(timezone.utc) - timedelta(days=90),
            "last_seen": datetime.now(timezone.utc),
            "total_transactions": 100,
            "fraud_count": 10,
            "fraud_rate": 0.10,  # 10% fraud rate
        }

        txn = {
            "external_transaction_id": "TXN-004",
            "merchant_id": merchant_id,
            "merchant_name": "Fraudy Merchant",
            "merchant_category_code": "5999",
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }
        result = enricher.enrich(txn)
        assert result.is_high_fraud_merchant is True
        assert result.merchant_fraud_rate >= 0.05

    def test_missing_merchant_id(self, enricher):
        txn = {
            "external_transaction_id": "TXN-005",
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }
        result = enricher.enrich(txn)
        assert result.merchant_risk_score == 0.5

    def test_unknown_mcc(self, enricher):
        txn = {
            "external_transaction_id": "TXN-006",
            "merchant_id": "MERCH-UNKNOWN",
            "merchant_name": "Unknown Shop",
            "merchant_category_code": "9999",
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }
        result = enricher.enrich(txn)
        # Unknown MCC gets default risk
        assert result.mcc_risk_score == 0.3

    def test_to_dict(self, enricher, grocery_transaction):
        result = enricher.enrich(grocery_transaction)
        data = result.to_dict()
        assert "merchant_mcc_risk_score" in data
        assert "merchant_is_new" in data
        assert "merchant_risk_score" in data

    def test_batch_enrichment(self, enricher, grocery_transaction, gambling_transaction):
        results = enricher.enrich_batch([grocery_transaction, gambling_transaction])
        assert len(results) == 2
        assert results[0].mcc_risk_score < results[1].mcc_risk_score


# =============================================================================
# Velocity Calculator Tests
# =============================================================================


class TestCustomerVelocityProfile:
    """Test CustomerVelocityProfile sliding window."""

    def test_add_and_count(self):
        profile = CustomerVelocityProfile()
        base_time = time.time()
        profile.add_transaction(100.0, "MERCH-1", base_time)
        profile.add_transaction(200.0, "MERCH-2", base_time + 10)
        assert profile.record_count == 2

    def test_metrics_computation(self):
        profile = CustomerVelocityProfile()
        base_time = time.time()
        profile.add_transaction(100.0, "MERCH-1", base_time)
        profile.add_transaction(200.0, "MERCH-2", base_time + 10)

        metrics = profile.compute_metrics(base_time + 20)
        assert metrics.transaction_count_1min == 2
        assert metrics.amount_sum_1min == 300.0
        assert metrics.unique_merchants_1hour == 2

    def test_window_expiration(self):
        profile = CustomerVelocityProfile()
        base_time = time.time()
        # Add old transaction (2 minutes ago)
        profile.add_transaction(100.0, "MERCH-1", base_time - 120)
        # Add recent transaction (now)
        profile.add_transaction(200.0, "MERCH-2", base_time)

        metrics = profile.compute_metrics(base_time)
        # 1-min window should only have the recent one
        assert metrics.transaction_count_1min == 1
        assert metrics.amount_sum_1min == 200.0
        # 5-min window should have both
        assert metrics.transaction_count_5min == 2

    def test_24hour_eviction(self):
        profile = CustomerVelocityProfile()
        base_time = time.time()
        # Add transaction > 24 hours ago
        profile.add_transaction(100.0, "MERCH-1", base_time - 90000)
        # Add recent transaction
        profile.add_transaction(200.0, "MERCH-2", base_time)

        metrics = profile.compute_metrics(base_time)
        # Old transaction should be evicted
        assert metrics.transaction_count_24hour == 1


class TestVelocityCalculator:
    """Test VelocityCalculator threshold detection."""

    @pytest.fixture
    def calculator(self):
        return VelocityCalculator()

    @pytest.fixture
    def base_transaction(self):
        return {
            "external_transaction_id": "TXN-001",
            "customer_id": "CUST-001",
            "merchant_id": "MERCH-001",
            "transaction_amount": 100.0,
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }

    def test_single_transaction_no_breach(self, calculator, base_transaction):
        result = calculator.evaluate(base_transaction)
        assert result.is_success
        assert result.breach_count == 0
        assert result.velocity_risk_score == 0.0

    def test_rapid_transactions_trigger_warning(self, calculator):
        base_ts = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        # Simulate 4 transactions in 1 minute (warning threshold is 3)
        for i in range(4):
            txn = {
                "external_transaction_id": f"TXN-{i}",
                "customer_id": "CUST-RAPID",
                "merchant_id": f"MERCH-{i}",
                "transaction_amount": 100.0,
                "transaction_timestamp": (base_ts + timedelta(seconds=i * 10)).isoformat(),
            }
            result = calculator.evaluate(txn)

        assert result.has_warning is True
        assert any(b.metric_name == "transaction_count_1min" for b in result.breaches)

    def test_critical_breach_on_high_velocity(self, calculator):
        base_ts = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        # Simulate 6 transactions in 1 minute (critical threshold is 5)
        for i in range(6):
            txn = {
                "external_transaction_id": f"TXN-{i}",
                "customer_id": "CUST-BURST",
                "merchant_id": f"MERCH-{i}",
                "transaction_amount": 100.0,
                "transaction_timestamp": (base_ts + timedelta(seconds=i * 5)).isoformat(),
            }
            result = calculator.evaluate(txn)

        assert result.has_critical is True

    def test_amount_threshold_breach(self, calculator):
        base_ts = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        # Large transactions to breach amount_sum_1min (warning: $1000)
        for i in range(3):
            txn = {
                "external_transaction_id": f"TXN-{i}",
                "customer_id": "CUST-BIG",
                "merchant_id": "MERCH-001",
                "transaction_amount": 500.0,
                "transaction_timestamp": (base_ts + timedelta(seconds=i * 10)).isoformat(),
            }
            result = calculator.evaluate(txn)

        assert any(b.metric_name == "amount_sum_1min" for b in result.breaches)

    def test_risk_score_increases_with_breaches(self, calculator):
        base_ts = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        results = []
        for i in range(8):
            txn = {
                "external_transaction_id": f"TXN-{i}",
                "customer_id": "CUST-RISK",
                "merchant_id": f"MERCH-{i}",
                "transaction_amount": 2000.0,
                "transaction_timestamp": (base_ts + timedelta(seconds=i * 5)).isoformat(),
            }
            results.append(calculator.evaluate(txn))

        # Risk should increase over time
        assert results[-1].velocity_risk_score > results[0].velocity_risk_score

    def test_missing_customer_id(self, calculator):
        txn = {
            "external_transaction_id": "TXN-001",
            "transaction_amount": 100.0,
        }
        result = calculator.evaluate(txn)
        assert result.error is not None

    def test_profile_isolation(self, calculator):
        base_ts = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        # Customer A: many transactions
        for i in range(5):
            txn = {
                "external_transaction_id": f"TXN-A-{i}",
                "customer_id": "CUST-A",
                "merchant_id": "MERCH-001",
                "transaction_amount": 100.0,
                "transaction_timestamp": (base_ts + timedelta(seconds=i * 5)).isoformat(),
            }
            calculator.evaluate(txn)

        # Customer B: single transaction
        txn_b = {
            "external_transaction_id": "TXN-B-1",
            "customer_id": "CUST-B",
            "merchant_id": "MERCH-001",
            "transaction_amount": 100.0,
            "transaction_timestamp": base_ts.isoformat(),
        }
        result_b = calculator.evaluate(txn_b)
        assert result_b.metrics.transaction_count_1min == 1

    def test_to_dict(self, calculator, base_transaction):
        result = calculator.evaluate(base_transaction)
        data = result.to_dict()
        assert "velocity_txn_count_1min" in data
        assert "velocity_risk_score" in data
        assert "velocity_breaches" in data

    def test_batch_evaluation(self, calculator):
        base_ts = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        txns = [
            {
                "external_transaction_id": f"TXN-{i}",
                "customer_id": "CUST-BATCH",
                "merchant_id": "MERCH-001",
                "transaction_amount": 50.0,
                "transaction_timestamp": (base_ts + timedelta(seconds=i * 30)).isoformat(),
            }
            for i in range(3)
        ]
        results = calculator.evaluate_batch(txns)
        assert len(results) == 3

    def test_custom_thresholds(self):
        custom = {
            "transaction_count_1min": {"warning": 1, "critical": 2},
        }
        calculator = VelocityCalculator(thresholds=custom)
        base_ts = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)

        for i in range(3):
            txn = {
                "external_transaction_id": f"TXN-{i}",
                "customer_id": "CUST-CUSTOM",
                "merchant_id": "MERCH-001",
                "transaction_amount": 10.0,
                "transaction_timestamp": (base_ts + timedelta(seconds=i * 5)).isoformat(),
            }
            result = calculator.evaluate(txn)

        assert result.has_critical is True

    def test_get_profile_count(self, calculator):
        assert calculator.get_profile_count() == 0
        txn = {
            "external_transaction_id": "TXN-001",
            "customer_id": "CUST-001",
            "merchant_id": "MERCH-001",
            "transaction_amount": 100.0,
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }
        calculator.evaluate(txn)
        assert calculator.get_profile_count() == 1

    def test_reset(self, calculator, base_transaction):
        calculator.evaluate(base_transaction)
        assert calculator.get_profile_count() == 1
        calculator.reset()
        assert calculator.get_profile_count() == 0
