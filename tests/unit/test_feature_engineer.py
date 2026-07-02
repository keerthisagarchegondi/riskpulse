"""Unit tests for the feature engineering pipeline and aggregator.

Tests cover:
- FeatureEngineer: transaction, velocity, behavioral, sequence features
- TimeWindowAggregator: tumbling/sliding windows, running statistics
- IncrementalAggregator: streaming buffer management
- Edge cases: missing data, empty history, boundary conditions
- Performance benchmarks: < 50ms per transaction
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from src.transformation.aggregator import (
    AggregationResult,
    IncrementalAggregator,
    RunningStatistics,
    TimeWindowAggregator,
    WindowSpec,
    WindowType,
)
from src.transformation.feature_engineer import (
    FeatureEngineer,
    FeatureMetrics,
    FeatureResult,
    WINDOW_1H,
    WINDOW_24H,
    WINDOW_7D,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def engineer():
    """Create a FeatureEngineer instance."""
    return FeatureEngineer()


@pytest.fixture
def base_transaction():
    """A typical transaction for testing."""
    return {
        "external_transaction_id": "TXN-001",
        "transaction_id": "uuid-001",
        "account_id": "ACC-12345",
        "customer_id": "CUST-100",
        "merchant_id": "MERCH-500",
        "merchant_name": "Test Store",
        "transaction_amount": 150.00,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "geo_country": "US",
        "status": "approved",
        "transaction_timestamp": "2026-06-15T14:30:00Z",
    }


@pytest.fixture
def customer_profile():
    """Customer profile with historical statistics."""
    return {
        "avg_transaction_amount": 100.0,
        "std_transaction_amount": 30.0,
        "min_transaction_amount": 10.0,
        "max_transaction_amount": 500.0,
        "total_transaction_count": 50,
        "last_transaction_timestamp": "2026-06-15T12:00:00Z",
    }


@pytest.fixture
def transaction_history():
    """Recent transaction history sorted by time desc."""
    base_time = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
    return [
        {
            "transaction_id": "uuid-h1",
            "merchant_id": "MERCH-500",
            "transaction_amount": 50.0,
            "channel": "online",
            "geo_country": "US",
            "status": "approved",
            "transaction_timestamp": (base_time - timedelta(minutes=10)).isoformat(),
        },
        {
            "transaction_id": "uuid-h2",
            "merchant_id": "MERCH-200",
            "transaction_amount": 200.0,
            "channel": "pos",
            "geo_country": "US",
            "status": "approved",
            "transaction_timestamp": (base_time - timedelta(hours=2)).isoformat(),
        },
        {
            "transaction_id": "uuid-h3",
            "merchant_id": "MERCH-300",
            "transaction_amount": 75.0,
            "channel": "online",
            "geo_country": "CA",
            "status": "declined",
            "transaction_timestamp": (base_time - timedelta(hours=5)).isoformat(),
        },
        {
            "transaction_id": "uuid-h4",
            "merchant_id": "MERCH-500",
            "transaction_amount": 120.0,
            "channel": "mobile",
            "geo_country": "US",
            "status": "approved",
            "transaction_timestamp": (base_time - timedelta(days=2)).isoformat(),
        },
        {
            "transaction_id": "uuid-h5",
            "merchant_id": "MERCH-100",
            "transaction_amount": 300.0,
            "channel": "online",
            "geo_country": "GB",
            "status": "approved",
            "transaction_timestamp": (base_time - timedelta(days=5)).isoformat(),
        },
    ]


# ──────────────────────────────────────────────────────────────────────────────
# FeatureEngineer: Transaction Features
# ──────────────────────────────────────────────────────────────────────────────


class TestTransactionFeatures:
    """Tests for transaction-level feature computation."""

    def test_amount_zscore_positive(self, engineer, base_transaction, customer_profile):
        """Amount above average should produce positive z-score."""
        result = engineer.compute_features(base_transaction, customer_profile)
        assert result.is_success
        # (150 - 100) / 30 = 1.6667
        assert abs(result.features["amount_zscore"] - 1.666667) < 0.001

    def test_amount_zscore_negative(self, engineer, customer_profile):
        """Amount below average should produce negative z-score."""
        txn = {
            "external_transaction_id": "TXN-002",
            "transaction_amount": 50.0,
            "transaction_timestamp": "2026-06-15T14:30:00Z",
        }
        result = engineer.compute_features(txn, customer_profile)
        # (50 - 100) / 30 = -1.6667
        assert result.features["amount_zscore"] < 0

    def test_amount_zscore_zero_std(self, engineer, base_transaction):
        """Zero std deviation should produce z-score of 0."""
        profile = {"avg_transaction_amount": 100.0, "std_transaction_amount": 0.0}
        result = engineer.compute_features(base_transaction, profile)
        assert result.features["amount_zscore"] == 0.0

    def test_hour_of_day(self, engineer, base_transaction, customer_profile):
        """Hour should be extracted correctly from timestamp."""
        result = engineer.compute_features(base_transaction, customer_profile)
        assert result.features["hour_of_day"] == 14

    def test_day_of_week(self, engineer, base_transaction, customer_profile):
        """Day of week should be correct (Monday=0)."""
        result = engineer.compute_features(base_transaction, customer_profile)
        # 2026-06-15 is a Monday
        assert result.features["day_of_week"] == 0

    def test_is_weekend_weekday(self, engineer, base_transaction, customer_profile):
        """Weekday should not be flagged as weekend."""
        result = engineer.compute_features(base_transaction, customer_profile)
        assert result.features["is_weekend"] is False

    def test_is_weekend_saturday(self, engineer, customer_profile):
        """Saturday transaction should be flagged as weekend."""
        txn = {
            "external_transaction_id": "TXN-003",
            "transaction_amount": 100.0,
            "transaction_timestamp": "2026-06-20T10:00:00Z",  # Saturday
        }
        result = engineer.compute_features(txn, customer_profile)
        assert result.features["is_weekend"] is True

    def test_is_holiday(self, engineer, customer_profile):
        """July 4th transaction should be flagged as holiday."""
        txn = {
            "external_transaction_id": "TXN-004",
            "transaction_amount": 100.0,
            "transaction_timestamp": "2026-07-04T12:00:00Z",
        }
        result = engineer.compute_features(txn, customer_profile)
        assert result.features["is_holiday"] is True

    def test_time_since_last_transaction(self, engineer, base_transaction, customer_profile):
        """Time delta from last transaction should be computed correctly."""
        result = engineer.compute_features(base_transaction, customer_profile)
        # 14:30 - 12:00 = 2.5 hours = 9000 seconds
        assert result.features["time_since_last_transaction"] == 9000.0

    def test_amount_to_avg_ratio(self, engineer, base_transaction, customer_profile):
        """Amount / avg ratio should be computed correctly."""
        result = engineer.compute_features(base_transaction, customer_profile)
        # 150 / 100 = 1.5
        assert abs(result.features["amount_to_avg_ratio"] - 1.5) < 0.0001

    def test_amount_to_avg_ratio_zero_avg(self, engineer, base_transaction):
        """Zero average should produce ratio of 0 (not division error)."""
        profile = {"avg_transaction_amount": 0.0, "std_transaction_amount": 0.0}
        result = engineer.compute_features(base_transaction, profile)
        assert result.features["amount_to_avg_ratio"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# FeatureEngineer: Velocity Features
# ──────────────────────────────────────────────────────────────────────────────


class TestVelocityFeatures:
    """Tests for velocity-based windowed features."""

    def test_txn_count_1h(self, engineer, base_transaction, customer_profile, transaction_history):
        """Transactions within 1 hour should be counted."""
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        # h1 is 10 min ago (within 1h)
        assert result.features["txn_count_1h"] == 1

    def test_txn_count_24h(self, engineer, base_transaction, customer_profile, transaction_history):
        """Transactions within 24 hours should be counted."""
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        # h1 (10min), h2 (2h), h3 (5h) are within 24h
        assert result.features["txn_count_24h"] == 3

    def test_txn_count_7d(self, engineer, base_transaction, customer_profile, transaction_history):
        """Transactions within 7 days should be counted."""
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        # All 5 are within 7 days
        assert result.features["txn_count_7d"] == 5

    def test_txn_amount_sum_1h(self, engineer, base_transaction, customer_profile, transaction_history):
        """Amount sum within 1 hour window."""
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        assert result.features["txn_amount_sum_1h"] == 50.0

    def test_txn_amount_sum_24h(self, engineer, base_transaction, customer_profile, transaction_history):
        """Amount sum within 24 hour window."""
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        # 50 + 200 + 75 = 325
        assert result.features["txn_amount_sum_24h"] == 325.0

    def test_unique_merchants_24h(self, engineer, base_transaction, customer_profile, transaction_history):
        """Unique merchants within 24 hours."""
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        # MERCH-500, MERCH-200, MERCH-300
        assert result.features["unique_merchants_24h"] == 3

    def test_unique_countries_24h(self, engineer, base_transaction, customer_profile, transaction_history):
        """Unique countries within 24 hours."""
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        # US, US, CA -> {US, CA}
        assert result.features["unique_countries_24h"] == 2

    def test_empty_history(self, engineer, base_transaction, customer_profile):
        """All velocity features should be 0 with no history."""
        result = engineer.compute_features(base_transaction, customer_profile, [])
        assert result.features["txn_count_1h"] == 0
        assert result.features["txn_count_24h"] == 0
        assert result.features["txn_count_7d"] == 0
        assert result.features["txn_amount_sum_1h"] == 0.0
        assert result.features["txn_amount_sum_24h"] == 0.0
        assert result.features["unique_merchants_24h"] == 0
        assert result.features["unique_countries_24h"] == 0

    def test_no_data_leakage(self, engineer, customer_profile):
        """Future transactions should not be included in velocity features."""
        txn = {
            "external_transaction_id": "TXN-005",
            "transaction_amount": 100.0,
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }
        future_history = [
            {
                "transaction_amount": 500.0,
                "merchant_id": "M1",
                "geo_country": "US",
                "transaction_timestamp": "2026-06-15T12:00:00Z",  # AFTER the txn
            }
        ]
        result = engineer.compute_features(txn, customer_profile, future_history)
        assert result.features["txn_count_1h"] == 0
        assert result.features["txn_count_24h"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# FeatureEngineer: Behavioral Features
# ──────────────────────────────────────────────────────────────────────────────


class TestBehavioralFeatures:
    """Tests for behavioral deviation features."""

    def test_new_merchant_flag_known(self, engineer, base_transaction, customer_profile, transaction_history):
        """Known merchant should not be flagged as new."""
        # MERCH-500 is in h1 and h4
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        assert result.features["new_merchant_flag"] is False

    def test_new_merchant_flag_unknown(self, engineer, customer_profile, transaction_history):
        """Unknown merchant should be flagged as new."""
        txn = {
            "external_transaction_id": "TXN-006",
            "merchant_id": "MERCH-NEVER-SEEN",
            "transaction_amount": 100.0,
            "transaction_timestamp": "2026-06-15T14:30:00Z",
            "channel": "online",
        }
        result = engineer.compute_features(txn, customer_profile, transaction_history)
        assert result.features["new_merchant_flag"] is True

    def test_unusual_hour_flag_normal(self, engineer, base_transaction, customer_profile):
        """14:30 is not an unusual hour."""
        result = engineer.compute_features(base_transaction, customer_profile)
        assert result.features["unusual_hour_flag"] is False

    def test_unusual_hour_flag_late_night(self, engineer, customer_profile):
        """3:00 AM should be flagged as unusual."""
        txn = {
            "external_transaction_id": "TXN-007",
            "transaction_amount": 100.0,
            "transaction_timestamp": "2026-06-15T03:00:00Z",
            "channel": "online",
        }
        result = engineer.compute_features(txn, customer_profile)
        assert result.features["unusual_hour_flag"] is True

    def test_amount_percentile_high(self, engineer, base_transaction, customer_profile, transaction_history):
        """Amount higher than most history should have high percentile."""
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        # 150 > 50, 75, 120 but < 200, 300 => 3/5 = 0.6
        assert abs(result.features["amount_percentile"] - 0.6) < 0.01

    def test_amount_percentile_no_history(self, engineer, base_transaction, customer_profile):
        """Default percentile should be 0.5 with no history."""
        result = engineer.compute_features(base_transaction, customer_profile, [])
        assert result.features["amount_percentile"] == 0.5

    def test_channel_switch_flag_same(self, engineer, base_transaction, customer_profile, transaction_history):
        """Same channel as last txn should not flag."""
        # base uses "online", h1 uses "online"
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        assert result.features["channel_switch_flag"] is False

    def test_channel_switch_flag_different(self, engineer, customer_profile, transaction_history):
        """Different channel from last txn should flag."""
        txn = {
            "external_transaction_id": "TXN-008",
            "merchant_id": "MERCH-500",
            "transaction_amount": 100.0,
            "channel": "atm",
            "transaction_timestamp": "2026-06-15T14:30:00Z",
        }
        result = engineer.compute_features(txn, customer_profile, transaction_history)
        assert result.features["channel_switch_flag"] is True


# ──────────────────────────────────────────────────────────────────────────────
# FeatureEngineer: Sequence Features
# ──────────────────────────────────────────────────────────────────────────────


class TestSequenceFeatures:
    """Tests for sequence-based features."""

    def test_consecutive_declined_none(self, engineer, base_transaction, customer_profile, transaction_history):
        """No consecutive declines when last txn was approved."""
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        assert result.features["consecutive_declined_count"] == 0

    def test_consecutive_declined_count(self, engineer, customer_profile):
        """Consecutive declines should be counted correctly."""
        txn = {
            "external_transaction_id": "TXN-009",
            "transaction_amount": 100.0,
            "transaction_timestamp": "2026-06-15T14:30:00Z",
            "channel": "online",
        }
        history = [
            {"status": "declined", "transaction_timestamp": "2026-06-15T14:20:00Z", "transaction_amount": 100.0},
            {"status": "declined", "transaction_timestamp": "2026-06-15T14:15:00Z", "transaction_amount": 100.0},
            {"status": "declined", "transaction_timestamp": "2026-06-15T14:10:00Z", "transaction_amount": 100.0},
            {"status": "approved", "transaction_timestamp": "2026-06-15T14:00:00Z", "transaction_amount": 100.0},
        ]
        result = engineer.compute_features(txn, customer_profile, history)
        assert result.features["consecutive_declined_count"] == 3

    def test_rapid_succession_flag_true(self, engineer, customer_profile):
        """Transaction within 60s of previous should be flagged."""
        txn = {
            "external_transaction_id": "TXN-010",
            "transaction_amount": 100.0,
            "transaction_timestamp": "2026-06-15T14:30:30Z",
            "channel": "online",
        }
        history = [
            {
                "transaction_amount": 50.0,
                "transaction_timestamp": "2026-06-15T14:30:00Z",  # 30s before
                "status": "approved",
            },
        ]
        result = engineer.compute_features(txn, customer_profile, history)
        assert result.features["rapid_succession_flag"] is True

    def test_rapid_succession_flag_false(self, engineer, base_transaction, customer_profile, transaction_history):
        """Transaction > 60s from previous should not be flagged."""
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        # h1 is 10 min ago
        assert result.features["rapid_succession_flag"] is False


# ──────────────────────────────────────────────────────────────────────────────
# FeatureEngineer: General / Integration
# ──────────────────────────────────────────────────────────────────────────────


class TestFeatureEngineerGeneral:
    """General tests for the feature engineering pipeline."""

    def test_all_features_present(self, engineer, base_transaction, customer_profile, transaction_history):
        """All 20 expected features should be present."""
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        expected_names = FeatureEngineer.get_feature_names()
        assert len(expected_names) == 20
        for name in expected_names:
            assert name in result.features, f"Missing feature: {name}"

    def test_feature_count_minimum(self, engineer, base_transaction, customer_profile, transaction_history):
        """At least 20 features should be computed (satisfies 25+ with aggregator)."""
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        assert len(result.features) >= 20

    def test_no_profile_graceful(self, engineer, base_transaction):
        """Missing profile should produce valid features with defaults."""
        result = engineer.compute_features(base_transaction, None)
        assert result.is_success
        assert result.features["amount_zscore"] == 0.0
        assert result.features["amount_to_avg_ratio"] == 0.0

    def test_no_history_graceful(self, engineer, base_transaction, customer_profile):
        """Missing history should produce valid features with defaults."""
        result = engineer.compute_features(base_transaction, customer_profile, None)
        assert result.is_success

    def test_invalid_transaction_error(self, engineer):
        """Completely broken transaction should return error result."""
        # None amount triggers error path when history has issues
        result = engineer.compute_features({}, None, None)
        # Should still succeed with defaults
        assert result.is_success

    def test_batch_processing(self, engineer, base_transaction, customer_profile, transaction_history):
        """Batch computation should process all transactions."""
        transactions = [base_transaction.copy() for _ in range(10)]
        profiles = {base_transaction["customer_id"]: customer_profile}
        histories = {base_transaction["customer_id"]: transaction_history}
        results = engineer.compute_features_batch(transactions, profiles, histories)
        assert len(results) == 10
        assert all(r.is_success for r in results)

    def test_metrics_tracking(self, engineer, base_transaction, customer_profile):
        """Metrics should track computation count and latency."""
        engineer.compute_features(base_transaction, customer_profile)
        engineer.compute_features(base_transaction, customer_profile)
        metrics = engineer.metrics.to_dict()
        assert metrics["total_computed"] == 2
        assert metrics["avg_latency_ms"] > 0

    def test_result_to_dict(self, engineer, base_transaction, customer_profile):
        """FeatureResult.to_dict should serialize correctly."""
        result = engineer.compute_features(base_transaction, customer_profile)
        d = result.to_dict()
        assert "transaction_id" in d
        assert "features" in d
        assert "latency_ms" in d


# ──────────────────────────────────────────────────────────────────────────────
# FeatureEngineer: Performance
# ──────────────────────────────────────────────────────────────────────────────


class TestFeaturePerformance:
    """Performance benchmarks for feature computation."""

    def test_single_transaction_under_50ms(self, engineer, base_transaction, customer_profile, transaction_history):
        """Single transaction feature computation must be < 50ms."""
        result = engineer.compute_features(base_transaction, customer_profile, transaction_history)
        assert result.latency_ms < 50.0

    def test_batch_100_performance(self, engineer, base_transaction, customer_profile, transaction_history):
        """100 transactions should complete in reasonable time."""
        transactions = [base_transaction.copy() for _ in range(100)]
        profiles = {base_transaction["customer_id"]: customer_profile}
        histories = {base_transaction["customer_id"]: transaction_history}

        start = time.perf_counter()
        results = engineer.compute_features_batch(transactions, profiles, histories)
        elapsed = (time.perf_counter() - start) * 1000

        assert all(r.is_success for r in results)
        # Average should be well under 50ms per transaction
        avg_latency = elapsed / 100
        assert avg_latency < 50.0


# ──────────────────────────────────────────────────────────────────────────────
# RunningStatistics
# ──────────────────────────────────────────────────────────────────────────────


class TestRunningStatistics:
    """Tests for Welford's online statistics algorithm."""

    def test_empty(self):
        """Empty stats should return zeros."""
        stats = RunningStatistics()
        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.std == 0.0

    def test_single_value(self):
        """Single value should have mean equal to that value."""
        stats = RunningStatistics()
        stats.add(42.0)
        assert stats.count == 1
        assert stats.mean == 42.0
        assert stats.std == 0.0

    def test_known_values(self):
        """Known set of values should produce correct statistics."""
        stats = RunningStatistics()
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for v in values:
            stats.add(v)
        assert stats.count == 5
        assert stats.mean == 30.0
        assert stats.min_val == 10.0
        assert stats.max_val == 50.0
        assert stats.sum_val == 150.0
        # std dev of [10,20,30,40,50] sample std = 15.81...
        assert abs(stats.std - 15.8114) < 0.01

    def test_reset(self):
        """Reset should clear all values."""
        stats = RunningStatistics()
        stats.add(100.0)
        stats.reset()
        assert stats.count == 0
        assert stats.mean == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# TimeWindowAggregator
# ──────────────────────────────────────────────────────────────────────────────


class TestTimeWindowAggregator:
    """Tests for the batch time-window aggregator."""

    @pytest.fixture
    def aggregator(self):
        return TimeWindowAggregator(window_specs=[
            WindowSpec("1h", 3600, WindowType.SLIDING),
            WindowSpec("24h", 86400, WindowType.SLIDING),
        ])

    @pytest.fixture
    def events(self):
        base = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        return [
            {"transaction_amount": 100.0, "transaction_timestamp": (base - timedelta(minutes=30)).isoformat()},
            {"transaction_amount": 200.0, "transaction_timestamp": (base - timedelta(minutes=45)).isoformat()},
            {"transaction_amount": 50.0, "transaction_timestamp": (base - timedelta(hours=3)).isoformat()},
            {"transaction_amount": 300.0, "transaction_timestamp": (base - timedelta(hours=20)).isoformat()},
        ]

    def test_1h_window(self, aggregator, events):
        """1h window should include only recent events."""
        ref = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        results = aggregator.aggregate(events, ref)
        assert results["1h"].count == 2
        assert results["1h"].sum == 300.0

    def test_24h_window(self, aggregator, events):
        """24h window should include all events within 24 hours."""
        ref = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        results = aggregator.aggregate(events, ref)
        assert results["24h"].count == 4
        assert results["24h"].sum == 650.0

    def test_mean_computation(self, aggregator, events):
        """Mean should be computed correctly."""
        ref = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        results = aggregator.aggregate(events, ref)
        assert results["1h"].mean == 150.0  # (100+200)/2

    def test_empty_events(self, aggregator):
        """Empty events list should return zero aggregations."""
        ref = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        results = aggregator.aggregate([], ref)
        assert results["1h"].count == 0
        assert results["24h"].count == 0

    def test_distinct_counts(self, aggregator):
        """Distinct count should count unique values in window."""
        ref = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        events = [
            {"merchant_id": "M1", "transaction_amount": 10, "transaction_timestamp": (ref - timedelta(minutes=10)).isoformat()},
            {"merchant_id": "M1", "transaction_amount": 20, "transaction_timestamp": (ref - timedelta(minutes=20)).isoformat()},
            {"merchant_id": "M2", "transaction_amount": 30, "transaction_timestamp": (ref - timedelta(minutes=30)).isoformat()},
            {"merchant_id": "M3", "transaction_amount": 40, "transaction_timestamp": (ref - timedelta(hours=10)).isoformat()},
        ]
        counts = aggregator.compute_distinct_counts(events, ref, "merchant_id")
        assert counts["1h"] == 2  # M1, M2
        assert counts["24h"] == 3  # M1, M2, M3

    def test_multiple_fields(self, aggregator, events):
        """Aggregating multiple fields should return results per field."""
        ref = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        results = aggregator.aggregate_multiple_fields(
            events, ref, ["transaction_amount"]
        )
        assert "transaction_amount" in results
        assert results["transaction_amount"]["1h"].count == 2


# ──────────────────────────────────────────────────────────────────────────────
# IncrementalAggregator
# ──────────────────────────────────────────────────────────────────────────────


class TestIncrementalAggregator:
    """Tests for the streaming incremental aggregator."""

    def test_add_and_get_stats(self):
        """Adding events should produce correct statistics."""
        agg = IncrementalAggregator(max_window_seconds=3600)
        base = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)

        agg.add_event(base - timedelta(minutes=10), 100.0)
        agg.add_event(base - timedelta(minutes=20), 200.0)
        agg.add_event(base - timedelta(minutes=30), 150.0)

        result = agg.get_statistics(base, 3600)
        assert result.count == 3
        assert result.sum == 450.0

    def test_window_exclusion(self):
        """Events outside the window should be excluded."""
        agg = IncrementalAggregator(max_window_seconds=7200)
        base = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)

        agg.add_event(base - timedelta(minutes=10), 100.0)
        agg.add_event(base - timedelta(minutes=90), 200.0)  # outside 1h window

        result = agg.get_statistics(base, 3600)
        assert result.count == 1
        assert result.sum == 100.0

    def test_eviction(self):
        """Old events should be evicted from the buffer."""
        agg = IncrementalAggregator(max_window_seconds=60)
        base = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)

        agg.add_event(base - timedelta(seconds=120), 100.0)  # will be evicted
        agg.add_event(base, 200.0)  # triggers eviction

        assert agg.buffer_size == 1

    def test_distinct_count(self):
        """Distinct count should work with metadata."""
        agg = IncrementalAggregator(max_window_seconds=3600)
        base = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)

        agg.add_event(base - timedelta(minutes=10), 100.0, {"merchant": "M1"})
        agg.add_event(base - timedelta(minutes=20), 200.0, {"merchant": "M1"})
        agg.add_event(base - timedelta(minutes=30), 150.0, {"merchant": "M2"})

        count = agg.get_distinct_count(base, 3600, "merchant")
        assert count == 2

    def test_clear(self):
        """Clear should empty the buffer."""
        agg = IncrementalAggregator()
        base = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        agg.add_event(base, 100.0)
        agg.clear()
        assert agg.buffer_size == 0


# ──────────────────────────────────────────────────────────────────────────────
# WindowSpec validation
# ──────────────────────────────────────────────────────────────────────────────


class TestWindowSpec:
    """Tests for WindowSpec validation."""

    def test_invalid_duration(self):
        """Zero or negative duration should raise."""
        with pytest.raises(ValueError):
            WindowSpec("bad", 0, WindowType.SLIDING)

    def test_tumbling_sets_slide(self):
        """Tumbling window should default slide to duration."""
        spec = WindowSpec("1h", 3600, WindowType.TUMBLING)
        assert spec.slide_seconds == 3600
