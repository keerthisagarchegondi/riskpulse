"""Comprehensive tests for the Alert Generation Framework."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from src.alerting.alert_manager import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertStatus,
    AlertType,
    DeduplicationEngine,
    SuppressionEngine,
    ThrottleEngine,
)
from src.alerting.alert_templates import (
    AlertTemplateRenderer,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def scoring_result_critical():
    """Critical scoring result from the pipeline."""
    return {
        "transaction_id": "TXN-001",
        "final_score": 0.97,
        "risk_classification": "critical",
        "alert_recommended": True,
        "auto_block_recommended": True,
        "methods_succeeded": 3,
        "methods_failed": 0,
        "scoring_version": "1.0.0",
        "method_scores": [
            {
                "method": "rule_engine",
                "raw_score": 0.9,
                "normalized_score": 0.9,
                "weight": 0.3,
                "weighted_score": 0.27,
                "latency_ms": 5.0,
                "success": True,
                "error": None,
                "details": {"triggered_rules": [{"rule_id": "R001", "rule_name": "High Amount"}]},
            },
            {
                "method": "anomaly_detection",
                "raw_score": 0.85,
                "normalized_score": 0.85,
                "weight": 0.3,
                "weighted_score": 0.255,
                "latency_ms": 8.0,
                "success": True,
                "error": None,
            },
            {
                "method": "ml_model",
                "raw_score": 0.95,
                "normalized_score": 0.95,
                "weight": 0.4,
                "weighted_score": 0.38,
                "latency_ms": 12.0,
                "success": True,
                "error": None,
            },
        ],
    }


@pytest.fixture
def scoring_result_high():
    """High-severity scoring result."""
    return {
        "transaction_id": "TXN-002",
        "final_score": 0.82,
        "risk_classification": "high",
        "alert_recommended": True,
        "auto_block_recommended": False,
        "methods_succeeded": 3,
        "methods_failed": 0,
        "scoring_version": "1.0.0",
        "method_scores": [
            {
                "method": "rule_engine",
                "raw_score": 0.7,
                "normalized_score": 0.7,
                "weight": 0.3,
                "weighted_score": 0.21,
                "latency_ms": 4.0,
                "success": True,
                "error": None,
                "details": {
                    "triggered_rules": [{"rule_id": "R003", "rule_name": "Velocity Check"}]
                },
            },
            {
                "method": "anomaly_detection",
                "raw_score": 0.6,
                "normalized_score": 0.6,
                "weight": 0.3,
                "weighted_score": 0.18,
                "latency_ms": 7.0,
                "success": True,
                "error": None,
            },
            {
                "method": "ml_model",
                "raw_score": 0.85,
                "normalized_score": 0.85,
                "weight": 0.4,
                "weighted_score": 0.34,
                "latency_ms": 10.0,
                "success": True,
                "error": None,
            },
        ],
    }


@pytest.fixture
def scoring_result_low():
    """Low-severity scoring result (should not usually generate alerts)."""
    return {
        "transaction_id": "TXN-003",
        "final_score": 0.35,
        "risk_classification": "low",
        "alert_recommended": False,
        "auto_block_recommended": False,
        "methods_succeeded": 3,
        "methods_failed": 0,
        "scoring_version": "1.0.0",
        "method_scores": [],
    }


@pytest.fixture
def sample_transaction():
    """Sample transaction for alert generation."""
    return {
        "external_transaction_id": "TXN-2026-001",
        "account_id": "ACC-12345",
        "customer_id": "CUST-67890",
        "merchant_id": "MERCH-11111",
        "merchant_name": "Suspicious Merchant",
        "merchant_category_code": "7995",
        "transaction_amount": 5000.00,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": "credit",
        "card_last_four": "4242",
        "ip_address": "185.220.101.1",
        "device_id": "device-xyz-999",
        "device_type": "desktop",
        "geo_latitude": 55.7558,
        "geo_longitude": 37.6173,
        "geo_country": "RU",
        "geo_city": "Moscow",
        "is_international": True,
        "transaction_timestamp": "2026-06-15T03:15:00Z",
    }


@pytest.fixture
def alert_manager():
    """AlertManager instance with default config."""
    manager = AlertManager()
    yield manager
    manager.clear_all()


@pytest.fixture
def alert_manager_with_kafka():
    """AlertManager with mock Kafka producer."""
    mock_producer = MagicMock()
    manager = AlertManager(kafka_producer=mock_producer)
    yield manager, mock_producer
    manager.clear_all()


# ── Deduplication Engine Tests ───────────────────────────────────────────────


class TestDeduplicationEngine:
    """Tests for alert deduplication logic."""

    def test_first_alert_is_not_duplicate(self):
        engine = DeduplicationEngine(window_minutes=60)
        assert engine.is_duplicate("ACC-001", "R001", "rule_based") is False

    def test_same_alert_within_window_is_duplicate(self):
        engine = DeduplicationEngine(window_minutes=60)
        engine.is_duplicate("ACC-001", "R001", "rule_based")
        assert engine.is_duplicate("ACC-001", "R001", "rule_based") is True

    def test_different_account_is_not_duplicate(self):
        engine = DeduplicationEngine(window_minutes=60)
        engine.is_duplicate("ACC-001", "R001", "rule_based")
        assert engine.is_duplicate("ACC-002", "R001", "rule_based") is False

    def test_different_rule_is_not_duplicate(self):
        engine = DeduplicationEngine(window_minutes=60)
        engine.is_duplicate("ACC-001", "R001", "rule_based")
        assert engine.is_duplicate("ACC-001", "R002", "rule_based") is False

    def test_different_type_is_not_duplicate(self):
        engine = DeduplicationEngine(window_minutes=60)
        engine.is_duplicate("ACC-001", "R001", "rule_based")
        assert engine.is_duplicate("ACC-001", "R001", "anomaly") is False

    def test_expired_window_allows_new_alert(self):
        engine = DeduplicationEngine(window_minutes=1)
        engine.is_duplicate("ACC-001", "R001", "rule_based")

        # Manually expire by manipulating internal state
        key = engine._generate_dedup_key("ACC-001", "R001", "rule_based")
        engine._seen[key] = [time.time() - 120]  # 2 minutes ago

        assert engine.is_duplicate("ACC-001", "R001", "rule_based") is False

    def test_clear_resets_state(self):
        engine = DeduplicationEngine(window_minutes=60)
        engine.is_duplicate("ACC-001", "R001", "rule_based")
        assert engine.size == 1
        engine.clear()
        assert engine.size == 0

    def test_none_rule_id_handled(self):
        engine = DeduplicationEngine(window_minutes=60)
        assert engine.is_duplicate("ACC-001", None, "anomaly") is False
        assert engine.is_duplicate("ACC-001", None, "anomaly") is True


# ── Suppression Engine Tests ─────────────────────────────────────────────────


class TestSuppressionEngine:
    """Tests for alert suppression logic."""

    def test_unsuppressed_account_not_suppressed(self):
        engine = SuppressionEngine(cooldown_minutes=120)
        assert engine.is_suppressed("ACC-001") is False

    def test_suppressed_account_is_suppressed(self):
        engine = SuppressionEngine(cooldown_minutes=120)
        engine.add_suppression("ACC-001", "under investigation")
        assert engine.is_suppressed("ACC-001") is True

    def test_suppression_reason_returned(self):
        engine = SuppressionEngine(cooldown_minutes=120)
        engine.add_suppression("ACC-001", "alert resolved")
        assert engine.get_suppression_reason("ACC-001") == "alert resolved"

    def test_suppression_expires_after_cooldown(self):
        engine = SuppressionEngine(cooldown_minutes=1)
        engine.add_suppression("ACC-001", "resolved")

        # Manually expire
        engine._suppressed_accounts["ACC-001"] = ("resolved", time.time() - 120)
        assert engine.is_suppressed("ACC-001") is False

    def test_remove_suppression(self):
        engine = SuppressionEngine(cooldown_minutes=120)
        engine.add_suppression("ACC-001", "investigating")
        engine.remove_suppression("ACC-001")
        assert engine.is_suppressed("ACC-001") is False

    def test_clear_resets_state(self):
        engine = SuppressionEngine(cooldown_minutes=120)
        engine.add_suppression("ACC-001", "test")
        engine.add_suppression("ACC-002", "test")
        engine.clear()
        assert engine.size == 0


# ── Throttle Engine Tests ────────────────────────────────────────────────────


class TestThrottleEngine:
    """Tests for alert throttling logic."""

    def test_first_alert_not_throttled(self):
        engine = ThrottleEngine(
            max_per_account_per_hour=5,
            max_total_per_minute=100,
        )
        throttled, reason = engine.should_throttle("ACC-001", "R001")
        assert throttled is False
        assert reason == ""

    def test_account_rate_limit_enforced(self):
        engine = ThrottleEngine(max_per_account_per_hour=3)
        for _ in range(3):
            engine.should_throttle("ACC-001", None)

        throttled, reason = engine.should_throttle("ACC-001", None)
        assert throttled is True
        assert "account_rate_exceeded" in reason

    def test_different_accounts_independent(self):
        engine = ThrottleEngine(max_per_account_per_hour=2)
        engine.should_throttle("ACC-001", None)
        engine.should_throttle("ACC-001", None)

        # ACC-002 should not be affected
        throttled, _ = engine.should_throttle("ACC-002", None)
        assert throttled is False

    def test_rule_rate_limit_enforced(self):
        engine = ThrottleEngine(max_per_rule_per_hour=2)
        engine.should_throttle("ACC-001", "R001")
        engine.should_throttle("ACC-002", "R001")

        throttled, reason = engine.should_throttle("ACC-003", "R001")
        assert throttled is True
        assert "rule_rate_exceeded" in reason

    def test_total_rate_limit_enforced(self):
        engine = ThrottleEngine(max_total_per_minute=3)
        for i in range(3):
            engine.should_throttle(f"ACC-{i}", None)

        throttled, reason = engine.should_throttle("ACC-99", None)
        assert throttled is True
        assert "total_rate_exceeded" in reason

    def test_storm_detection(self):
        engine = ThrottleEngine(
            max_total_per_minute=1000,
            storm_threshold_per_minute=5,
        )
        for i in range(5):
            engine.should_throttle(f"ACC-{i}", None)

        throttled, reason = engine.should_throttle("ACC-99", None)
        assert throttled is True
        assert "storm" in reason
        assert engine.is_storm_active is True

    def test_storm_reset(self):
        engine = ThrottleEngine(storm_threshold_per_minute=2)
        engine.should_throttle("ACC-001", None)
        engine.should_throttle("ACC-002", None)
        engine.should_throttle("ACC-003", None)  # Triggers storm

        engine.reset_storm()
        assert engine.is_storm_active is False


# ── Alert Manager Tests ──────────────────────────────────────────────────────


class TestAlertManager:
    """Tests for the AlertManager class."""

    def test_generate_alert_critical(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.status == AlertStatus.OPEN
        assert alert.account_id == "ACC-12345"
        assert alert.risk_score == 0.97

    def test_generate_alert_high(self, alert_manager, scoring_result_high, sample_transaction):
        alert = alert_manager.generate_alert(scoring_result_high, sample_transaction)
        assert alert is not None
        assert alert.severity == AlertSeverity.HIGH

    def test_generate_alert_extracts_rule_id(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        assert alert is not None
        assert alert.rule_id == "R001"

    def test_generate_alert_deduplication(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert1 = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        alert2 = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        assert alert1 is not None
        assert alert2 is None  # Deduplicated

    def test_generate_alert_suppression(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        # Add account to suppression
        alert_manager._suppression_engine.add_suppression("ACC-12345", "under investigation")
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        assert alert is None

    def test_generate_alert_with_enrichment(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        enrichment = {"customer_history": {"total_transactions": 500, "previous_alerts": 2}}
        alert = alert_manager.generate_alert(
            scoring_result_critical, sample_transaction, enrichment
        )
        assert alert is not None
        assert alert.enrichment == enrichment

    def test_generate_alert_publishes_to_kafka(
        self, alert_manager_with_kafka, scoring_result_critical, sample_transaction
    ):
        manager, mock_producer = alert_manager_with_kafka
        alert = manager.generate_alert(scoring_result_critical, sample_transaction)
        assert alert is not None
        mock_producer.produce.assert_called_once()

    def test_generate_alerts_from_batch(
        self, alert_manager, scoring_result_critical, scoring_result_high
    ):
        txn1 = {
            "external_transaction_id": "TXN-B1",
            "account_id": "ACC-001",
            "merchant_name": "Test",
            "transaction_amount": 1000,
            "transaction_currency": "USD",
            "channel": "online",
        }
        txn2 = {
            "external_transaction_id": "TXN-B2",
            "account_id": "ACC-002",
            "merchant_name": "Test2",
            "transaction_amount": 2000,
            "transaction_currency": "USD",
            "channel": "pos",
        }
        batch = [
            (scoring_result_critical, txn1),
            (scoring_result_high, txn2),
        ]
        alerts = alert_manager.generate_alerts_from_batch(batch)
        assert len(alerts) == 2

    def test_batch_skips_non_recommended(self, alert_manager, scoring_result_low):
        txn = {
            "external_transaction_id": "TXN-LOW",
            "account_id": "ACC-LOW",
            "merchant_name": "Normal",
            "transaction_amount": 50,
            "transaction_currency": "USD",
            "channel": "pos",
        }
        batch = [(scoring_result_low, txn)]
        alerts = alert_manager.generate_alerts_from_batch(batch)
        assert len(alerts) == 0

    def test_channels_based_on_severity(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        assert alert is not None
        # Critical should have multiple channels
        assert len(alert.channels) > 0

    def test_alert_description_populated(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        assert alert is not None
        assert "CRITICAL" in alert.description
        assert "5000.00" in alert.description

    def test_alert_details_contain_scoring(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        assert alert is not None
        assert "scoring" in alert.details
        assert "transaction" in alert.details
        assert alert.details["scoring"]["final_score"] == 0.97


# ── Alert Lifecycle Tests ────────────────────────────────────────────────────


class TestAlertLifecycle:
    """Tests for alert lifecycle management."""

    def test_transition_open_to_investigating(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        updated = alert_manager.transition_alert(
            alert.alert_id,
            AlertStatus.INVESTIGATING,
            assigned_to="analyst-1",
        )
        assert updated is not None
        assert updated.status == AlertStatus.INVESTIGATING
        assert updated.assigned_to == "analyst-1"

    def test_transition_investigating_to_resolved(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        alert_manager.transition_alert(alert.alert_id, AlertStatus.INVESTIGATING)
        updated = alert_manager.transition_alert(
            alert.alert_id,
            AlertStatus.RESOLVED,
            resolution_notes="Confirmed legitimate transaction",
        )
        assert updated is not None
        assert updated.status == AlertStatus.RESOLVED
        assert updated.resolved_at is not None
        assert updated.resolution_notes == "Confirmed legitimate transaction"

    def test_transition_to_false_positive(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        alert_manager.transition_alert(alert.alert_id, AlertStatus.INVESTIGATING)
        updated = alert_manager.transition_alert(
            alert.alert_id,
            AlertStatus.FALSE_POSITIVE,
            resolution_notes="Known customer behavior",
        )
        assert updated is not None
        assert updated.status == AlertStatus.FALSE_POSITIVE

    def test_invalid_transition_rejected(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        # Resolve first
        alert_manager.transition_alert(alert.alert_id, AlertStatus.INVESTIGATING)
        alert_manager.transition_alert(alert.alert_id, AlertStatus.RESOLVED)

        # Try to transition resolved → investigating (invalid)
        result = alert_manager.transition_alert(alert.alert_id, AlertStatus.INVESTIGATING)
        assert result is None

    def test_transition_adds_suppression(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        alert_manager.transition_alert(alert.alert_id, AlertStatus.INVESTIGATING)

        # Account should now be suppressed
        assert alert_manager._suppression_engine.is_suppressed("ACC-12345") is True

    def test_get_alert_by_id(self, alert_manager, scoring_result_critical, sample_transaction):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        retrieved = alert_manager.get_alert(alert.alert_id)
        assert retrieved is not None
        assert retrieved.alert_id == alert.alert_id

    def test_get_nonexistent_alert(self, alert_manager):
        assert alert_manager.get_alert("nonexistent-id") is None

    def test_get_alerts_by_account(
        self, alert_manager, scoring_result_critical, scoring_result_high
    ):
        txn1 = {
            "external_transaction_id": "TXN-A1",
            "account_id": "ACC-MULTI",
            "merchant_name": "Test",
            "transaction_amount": 5000,
            "transaction_currency": "USD",
            "channel": "online",
        }
        alert_manager.generate_alert(scoring_result_critical, txn1)

        alerts = alert_manager.get_alerts_by_account("ACC-MULTI")
        assert len(alerts) == 1

    def test_get_open_alerts_ordered_by_severity(self, alert_manager):
        # Generate alerts with different severities
        txns_and_scores = [
            (
                {
                    "final_score": 0.35,
                    "risk_classification": "low",
                    "alert_recommended": True,
                    "method_scores": [],
                },
                {
                    "external_transaction_id": "T1",
                    "account_id": "A1",
                    "merchant_name": "M",
                    "transaction_amount": 100,
                    "transaction_currency": "USD",
                    "channel": "online",
                },
            ),
            (
                {
                    "final_score": 0.97,
                    "risk_classification": "critical",
                    "alert_recommended": True,
                    "method_scores": [],
                },
                {
                    "external_transaction_id": "T2",
                    "account_id": "A2",
                    "merchant_name": "M",
                    "transaction_amount": 9000,
                    "transaction_currency": "USD",
                    "channel": "online",
                },
            ),
            (
                {
                    "final_score": 0.6,
                    "risk_classification": "medium",
                    "alert_recommended": True,
                    "method_scores": [],
                },
                {
                    "external_transaction_id": "T3",
                    "account_id": "A3",
                    "merchant_name": "M",
                    "transaction_amount": 500,
                    "transaction_currency": "USD",
                    "channel": "pos",
                },
            ),
        ]
        for score, txn in txns_and_scores:
            alert_manager.generate_alert(score, txn)

        open_alerts = alert_manager.get_open_alerts()
        assert len(open_alerts) == 3
        # Critical should be first
        assert open_alerts[0].severity == AlertSeverity.CRITICAL


# ── Alert Enrichment Tests ───────────────────────────────────────────────────


class TestAlertEnrichment:
    """Tests for alert enrichment."""

    def test_enrich_with_customer_history(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        enriched = alert_manager.enrich_alert(
            alert,
            customer_history={
                "total_transactions": 500,
                "account_age_days": 365,
                "previous_alerts": 2,
                "average_transaction_amount": 150.0,
            },
        )
        assert "customer_history" in enriched.enrichment
        assert enriched.enrichment["customer_history"]["total_transactions"] == 500

    def test_enrich_with_recent_transactions(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        recent_txns = [
            {
                "transaction_id": "TXN-R1",
                "transaction_amount": 50.0,
                "transaction_timestamp": "2026-06-14T10:00:00Z",
                "merchant_name": "Grocery Store",
                "channel": "pos",
            }
        ]
        enriched = alert_manager.enrich_alert(alert, recent_transactions=recent_txns)
        assert "recent_transactions" in enriched.enrichment
        assert len(enriched.enrichment["recent_transactions"]) == 1

    def test_enrich_with_risk_profile(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        enriched = alert_manager.enrich_alert(
            alert,
            risk_profile={
                "risk_tier": "elevated",
                "lifetime_risk_score": 0.45,
                "flagged_count": 3,
            },
        )
        assert "risk_profile" in enriched.enrichment
        assert enriched.enrichment["risk_profile"]["risk_tier"] == "elevated"

    def test_enrich_limits_recent_transactions(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        # Provide 20 transactions, should be limited to 10
        recent_txns = [
            {
                "transaction_id": f"TXN-R{i}",
                "transaction_amount": float(i * 10),
                "transaction_timestamp": "2026-06-14T10:00:00Z",
                "merchant_name": f"Merchant {i}",
                "channel": "online",
            }
            for i in range(20)
        ]
        enriched = alert_manager.enrich_alert(alert, recent_transactions=recent_txns)
        assert len(enriched.enrichment["recent_transactions"]) == 10


# ── Alert Statistics Tests ───────────────────────────────────────────────────


class TestAlertStatistics:
    """Tests for alert statistics tracking."""

    def test_statistics_after_generation(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        stats = alert_manager.get_statistics_snapshot()
        assert stats["total_generated"] == 1
        assert stats["by_severity"]["critical"] == 1
        assert stats["by_status"]["open"] == 1

    def test_statistics_after_deduplication(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        stats = alert_manager.get_statistics_snapshot()
        assert stats["total_generated"] == 1
        assert stats["total_deduplicated"] == 1

    def test_statistics_after_suppression(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert_manager._suppression_engine.add_suppression("ACC-12345", "test")
        alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        stats = alert_manager.get_statistics_snapshot()
        assert stats["total_suppressed"] == 1

    def test_statistics_status_change(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        alert_manager.transition_alert(alert.alert_id, AlertStatus.INVESTIGATING)
        stats = alert_manager.get_statistics_snapshot()
        assert stats["by_status"]["investigating"] == 1
        assert stats["by_status"]["open"] == 0

    def test_statistics_kafka_publish_count(
        self, alert_manager_with_kafka, scoring_result_critical, sample_transaction
    ):
        manager, _ = alert_manager_with_kafka
        manager.generate_alert(scoring_result_critical, sample_transaction)
        stats = manager.get_statistics_snapshot()
        assert stats["total_published"] == 1


# ── Alert Serialization Tests ────────────────────────────────────────────────


class TestAlertSerialization:
    """Tests for alert serialization/deserialization."""

    def test_alert_to_dict(self, alert_manager, scoring_result_critical, sample_transaction):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        data = alert.to_dict()
        assert data["alert_id"] == alert.alert_id
        assert data["severity"] == "critical"
        assert data["status"] == "open"
        assert isinstance(data["created_at"], str)

    def test_alert_from_dict_roundtrip(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        data = alert.to_dict()
        restored = Alert.from_dict(data)
        assert restored.alert_id == alert.alert_id
        assert restored.severity == alert.severity
        assert restored.status == alert.status
        assert restored.risk_score == alert.risk_score


# ── Template Renderer Tests ──────────────────────────────────────────────────


class TestAlertTemplateRenderer:
    """Tests for alert template rendering."""

    @pytest.fixture
    def renderer(self):
        return AlertTemplateRenderer(locale="en_US")

    @pytest.fixture
    def sample_alert(self):
        return Alert(
            alert_id="ALT-001",
            transaction_id="TXN-001",
            account_id="ACC-12345",
            alert_type=AlertType.RULE_BASED,
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.OPEN,
            risk_score=0.97,
            rule_id="R001",
            description="Rule-based detection triggered [CRITICAL] - Risk score: 0.9700",
            details={
                "transaction": {
                    "amount": 5000.00,
                    "currency": "USD",
                    "merchant": "Suspicious Merchant",
                    "channel": "online",
                    "timestamp": "2026-06-15T03:15:00Z",
                },
                "scoring": {"final_score": 0.97},
            },
            channels=["email", "sms", "dashboard", "webhook"],
        )

    def test_render_email(self, renderer, sample_alert):
        rendered = renderer.render(sample_alert, "email")
        assert rendered.channel == "email"
        assert "[CRITICAL]" in rendered.subject
        assert "ACC-12345" in rendered.subject
        assert "0.9700" in rendered.body
        assert rendered.html_body is not None
        assert rendered.priority == "urgent"

    def test_render_sms(self, renderer, sample_alert):
        rendered = renderer.render(sample_alert, "sms")
        assert rendered.channel == "sms"
        assert len(rendered.body) <= 160
        assert "CRITICAL" in rendered.body
        assert rendered.priority == "urgent"

    def test_render_dashboard(self, renderer, sample_alert):
        rendered = renderer.render(sample_alert, "dashboard")
        assert rendered.channel == "dashboard"
        assert "alert_id" in rendered.body
        assert "ALT-001" in rendered.body
        assert rendered.metadata.get("widget_type") == "alert_card"

    def test_render_webhook(self, renderer, sample_alert):
        rendered = renderer.render(sample_alert, "webhook")
        assert rendered.channel == "webhook"
        assert "fraud.alert.created" in rendered.body
        assert rendered.metadata.get("content_type") == "application/json"

    def test_render_all_channels(self, renderer, sample_alert):
        rendered_list = renderer.render_all_channels(sample_alert)
        assert len(rendered_list) == 4
        channels = {r.channel for r in rendered_list}
        assert channels == {"email", "sms", "dashboard", "webhook"}

    def test_locale_spanish(self, sample_alert):
        renderer = AlertTemplateRenderer(locale="es_ES")
        rendered = renderer.render(sample_alert, "email")
        assert "Alerta de Fraude" in rendered.body

    def test_locale_french(self, sample_alert):
        renderer = AlertTemplateRenderer(locale="fr_FR")
        rendered = renderer.render(sample_alert, "email")
        assert "Alerte de Fraude" in rendered.body

    def test_locale_german(self, sample_alert):
        renderer = AlertTemplateRenderer(locale="de_DE")
        rendered = renderer.render(sample_alert, "email")
        assert "Betrugswarnung" in rendered.body

    def test_locale_portuguese(self, sample_alert):
        renderer = AlertTemplateRenderer(locale="pt_BR")
        rendered = renderer.render(sample_alert, "email")
        assert "Alerta de Fraude" in rendered.body

    def test_unsupported_locale_falls_back(self, sample_alert):
        renderer = AlertTemplateRenderer(locale="xx_XX")
        assert renderer.locale == "en_US"

    def test_priority_mapping(self, renderer):
        assert renderer._get_priority(AlertSeverity.LOW) == "low"
        assert renderer._get_priority(AlertSeverity.MEDIUM) == "normal"
        assert renderer._get_priority(AlertSeverity.HIGH) == "high"
        assert renderer._get_priority(AlertSeverity.CRITICAL) == "urgent"

    def test_email_html_contains_severity_color(self, renderer, sample_alert):
        rendered = renderer.render(sample_alert, "email")
        # Critical color
        assert "#9C27B0" in rendered.html_body

    def test_sms_truncation_for_long_content(self, renderer):
        alert = Alert(
            alert_id="ALT-LONG-ID-THAT-IS-QUITE-VERBOSE",
            transaction_id="TXN-VERY-LONG-TRANSACTION-ID",
            account_id="ACC-EXTREMELY-LONG-ACCOUNT-IDENTIFIER",
            alert_type=AlertType.ENSEMBLE,
            severity=AlertSeverity.HIGH,
            status=AlertStatus.OPEN,
            risk_score=0.85,
            details={
                "transaction": {
                    "amount": 99999.99,
                    "currency": "USD",
                }
            },
            channels=["sms"],
        )
        rendered = renderer.render(alert, "sms")
        assert len(rendered.body) <= 160


# ── Integration Tests ────────────────────────────────────────────────────────


class TestAlertManagerIntegration:
    """Integration tests combining multiple components."""

    def test_full_alert_lifecycle(self, alert_manager, scoring_result_critical, sample_transaction):
        # Generate
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        assert alert is not None
        assert alert.status == AlertStatus.OPEN

        # Enrich
        enriched = alert_manager.enrich_alert(
            alert,
            customer_history={"total_transactions": 100, "previous_alerts": 1},
            recent_transactions=[
                {
                    "transaction_id": "T1",
                    "transaction_amount": 50.0,
                    "transaction_timestamp": "2026-06-14T10:00:00Z",
                    "merchant_name": "Store",
                    "channel": "pos",
                }
            ],
        )
        assert "customer_history" in enriched.enrichment

        # Investigate
        updated = alert_manager.transition_alert(
            alert.alert_id,
            AlertStatus.INVESTIGATING,
            assigned_to="analyst-1",
        )
        assert updated.status == AlertStatus.INVESTIGATING

        # Resolve
        resolved = alert_manager.transition_alert(
            alert.alert_id,
            AlertStatus.RESOLVED,
            resolution_notes="Confirmed fraud, card blocked",
        )
        assert resolved.status == AlertStatus.RESOLVED
        assert resolved.resolved_at is not None

    def test_dedup_then_different_rule_generates(self, alert_manager, sample_transaction):
        score1 = {
            "final_score": 0.85,
            "risk_classification": "high",
            "alert_recommended": True,
            "method_scores": [
                {
                    "method": "rule_engine",
                    "success": True,
                    "weighted_score": 0.3,
                    "details": {"triggered_rules": [{"rule_id": "R001"}]},
                }
            ],
        }
        score2 = {
            "final_score": 0.85,
            "risk_classification": "high",
            "alert_recommended": True,
            "method_scores": [
                {
                    "method": "rule_engine",
                    "success": True,
                    "weighted_score": 0.3,
                    "details": {"triggered_rules": [{"rule_id": "R002"}]},
                }
            ],
        }

        alert1 = alert_manager.generate_alert(score1, sample_transaction)
        alert2 = alert_manager.generate_alert(score2, sample_transaction)

        assert alert1 is not None
        assert alert2 is not None  # Different rule, not deduplicated

    def test_render_generated_alert(
        self, alert_manager, scoring_result_critical, sample_transaction
    ):
        alert = alert_manager.generate_alert(scoring_result_critical, sample_transaction)
        renderer = AlertTemplateRenderer()
        rendered_list = renderer.render_all_channels(alert)
        assert len(rendered_list) > 0
        for rendered in rendered_list:
            assert rendered.alert_id == alert.alert_id
            assert rendered.body != ""
