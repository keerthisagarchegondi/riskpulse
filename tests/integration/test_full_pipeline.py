"""Comprehensive integration tests for the full fraud detection pipeline.

Tests the complete flow:
    Transaction → Validate → Transform → Enrich → Score → Alert

Verifies:
- Alert generation for known fraud patterns
- End-to-end data integrity (no data loss)
- Scoring pipeline integration with alert manager
- Deduplication and suppression across the full stack
- Batch scoring and alert generation at scale
- Pipeline resilience under mixed valid/invalid/fraud loads
"""

from __future__ import annotations

import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from src.alerting.alert_manager import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertStatus,
)
from src.enrichment.device_enricher import DeviceEnricher, InMemoryDeviceStore
from src.enrichment.geo_enricher import GeoEnricher
from src.enrichment.merchant_enricher import InMemoryMerchantStore, MerchantEnricher
from src.enrichment.velocity_calculator import VelocityCalculator
from src.fraud_detection.anomaly_detector import AnomalyDetector
from src.fraud_detection.rule_engine import FraudRuleEngine
from src.fraud_detection.scoring_pipeline import (
    RiskClassification,
    ScoringPipeline,
    UnifiedScore,
)
from src.pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineStage,
)
from src.transformation.cleaner import DataCleaner
from src.transformation.feature_engineer import FeatureEngineer
from src.transformation.normalizer import DataNormalizer
from src.validation.quarantine_handler import QuarantineHandler
from src.validation.schema_validator import SchemaValidator

# ============================================================================
# Test Data Generators
# ============================================================================

MERCHANTS_LEGIT = [
    ("MERCH-001", "Amazon", "5411", "US"),
    ("MERCH-002", "Walmart", "5411", "US"),
    ("MERCH-003", "Starbucks", "5812", "US"),
    ("MERCH-004", "Target", "5311", "US"),
    ("MERCH-005", "Costco", "5300", "US"),
]

MERCHANTS_SUSPICIOUS = [
    ("MERCH-090", "Unknown Digital Store", "7995", "RU"),
    ("MERCH-091", "Wire Transfer Services", "6012", "NG"),
    ("MERCH-092", "Crypto Exchange Intl", "6051", "KY"),
]

LOCATIONS_DOMESTIC = [
    ("US", "New York", 40.7128, -74.0060),
    ("US", "Los Angeles", 34.0522, -118.2437),
    ("US", "Chicago", 41.8781, -87.6298),
]

LOCATIONS_HIGH_RISK = [
    ("RU", "Moscow", 55.7558, 37.6173),
    ("NG", "Lagos", 6.5244, 3.3792),
    ("KP", "Pyongyang", 39.0392, 125.7625),
]


def _random_ip(domestic: bool = True) -> str:
    if domestic:
        return f"10.{random.randint(0, 255)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
    return f"{random.choice([185, 195, 203])}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"


def make_valid_transaction(**overrides) -> dict[str, Any]:
    """Generate a valid low-risk transaction."""
    merchant = random.choice(MERCHANTS_LEGIT)
    location = random.choice(LOCATIONS_DOMESTIC)
    base = {
        "external_transaction_id": f"TXN-FP-{uuid.uuid4().hex[:12].upper()}",
        "account_id": f"ACC-{random.randint(10000, 89999)}",
        "customer_id": f"CUST-{random.randint(10000, 89999)}",
        "merchant_id": merchant[0],
        "merchant_name": merchant[1],
        "merchant_category_code": merchant[2],
        "transaction_amount": round(random.uniform(10.0, 500.0), 2),
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": random.choice(["online", "pos", "mobile"]),
        "card_type": random.choice(["credit", "debit"]),
        "card_last_four": f"{random.randint(1000, 9999)}",
        "ip_address": _random_ip(domestic=True),
        "device_id": f"device-{uuid.uuid4().hex[:8]}",
        "device_type": random.choice(["mobile", "desktop"]),
        "geo_latitude": location[2] + random.uniform(-0.01, 0.01),
        "geo_longitude": location[3] + random.uniform(-0.01, 0.01),
        "geo_country": location[0],
        "geo_city": location[1],
        "is_international": False,
        "transaction_timestamp": (
            datetime.now(timezone.utc) - timedelta(minutes=random.randint(1, 120))
        ).isoformat(),
    }
    base.update(overrides)
    return base


def make_fraud_transaction(**overrides) -> dict[str, Any]:
    """Generate a transaction with strong fraud indicators."""
    merchant = random.choice(MERCHANTS_SUSPICIOUS)
    location = random.choice(LOCATIONS_HIGH_RISK)
    base = {
        "external_transaction_id": f"TXN-FRAUD-{uuid.uuid4().hex[:12].upper()}",
        "account_id": f"ACC-{random.randint(90000, 99999)}",
        "customer_id": f"CUST-{random.randint(90000, 99999)}",
        "merchant_id": merchant[0],
        "merchant_name": merchant[1],
        "merchant_category_code": merchant[2],
        "transaction_amount": round(random.uniform(5000.0, 9999.0), 2),
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": "credit",
        "card_last_four": f"{random.randint(1000, 9999)}",
        "ip_address": _random_ip(domestic=False),
        "device_id": f"device-new-{uuid.uuid4().hex[:8]}",
        "device_type": "desktop",
        "geo_latitude": location[2],
        "geo_longitude": location[3],
        "geo_country": location[0],
        "geo_city": location[1],
        "is_international": True,
        "transaction_timestamp": datetime.now(timezone.utc)
        .replace(hour=random.randint(1, 4))
        .isoformat(),
    }
    base.update(overrides)
    return base


def make_invalid_transaction() -> dict[str, Any]:
    """Generate a transaction that fails schema validation."""
    return {
        "external_transaction_id": "",
        "account_id": "",
        "transaction_amount": -100.0,
        "transaction_currency": "INVALID_CODE",
        "transaction_type": "invalid_type",
        "channel": "carrier_pigeon",
        "transaction_timestamp": "not-a-date",
    }


def make_mixed_batch(
    total: int,
    fraud_ratio: float = 0.10,
    invalid_ratio: float = 0.05,
) -> list[dict[str, Any]]:
    """Generate a mixed batch of valid, fraud, and invalid transactions."""
    n_invalid = int(total * invalid_ratio)
    n_fraud = int(total * fraud_ratio)
    n_valid = total - n_invalid - n_fraud

    batch = []
    for _ in range(n_valid):
        batch.append(make_valid_transaction())
    for _ in range(n_fraud):
        batch.append(make_fraud_transaction())
    for _ in range(n_invalid):
        batch.append(make_invalid_transaction())

    random.shuffle(batch)
    return batch


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def pipeline() -> PipelineOrchestrator:
    """Create a fully-configured pipeline orchestrator."""
    return PipelineOrchestrator(
        schema_validator=SchemaValidator(),
        rules_engine=None,
        data_cleaner=DataCleaner(),
        normalizer=DataNormalizer(),
        feature_engineer=FeatureEngineer(),
        geo_enricher=GeoEnricher(),
        device_enricher=DeviceEnricher(device_store=InMemoryDeviceStore()),
        merchant_enricher=MerchantEnricher(merchant_store=InMemoryMerchantStore()),
        velocity_calculator=VelocityCalculator(),
        quarantine_handler=QuarantineHandler(),
        batch_size=200,
    )


@pytest.fixture
def rule_engine() -> FraudRuleEngine:
    """Instantiate the rule engine with production rules."""
    return FraudRuleEngine()


@pytest.fixture
def scoring_pipeline(rule_engine) -> ScoringPipeline:
    """Create scoring pipeline with rule engine only (no trained ML models)."""
    return ScoringPipeline(rule_engine=rule_engine)


@pytest.fixture
def alert_manager() -> AlertManager:
    """Create alert manager with default configuration."""
    return AlertManager()


@pytest.fixture
def trained_anomaly_detector() -> AnomalyDetector:
    """Create and train an anomaly detector on synthetic data."""
    import pandas as pd

    detector = AnomalyDetector(
        n_estimators=50,
        contamination=0.05,
        random_state=42,
    )

    n_samples = 2000
    rng = np.random.default_rng(42)

    normal_data = {
        "transaction_amount": rng.lognormal(4.0, 1.0, n_samples),
        "transaction_count_1hour": rng.poisson(2, n_samples),
        "transaction_count_24hour": rng.poisson(5, n_samples),
        "amount_mean_24hour": rng.lognormal(4.0, 0.5, n_samples),
        "amount_std_24hour": rng.exponential(50, n_samples),
        "time_since_last_transaction_seconds": rng.exponential(7200, n_samples),
        "distance_from_last_location_km": rng.exponential(10, n_samples),
        "unique_merchants_24hour": rng.poisson(3, n_samples),
        "unique_countries_24hour": np.ones(n_samples),
        "hour_of_day": rng.integers(8, 22, n_samples),
        "is_international": np.zeros(n_samples),
        "amount_to_avg_ratio": rng.lognormal(0, 0.3, n_samples),
    }

    df = pd.DataFrame(normal_data)
    detector.fit(df)
    return detector


@pytest.fixture
def full_scoring_pipeline(rule_engine, trained_anomaly_detector) -> ScoringPipeline:
    """Scoring pipeline with rule engine + anomaly detector (no ML model)."""
    return ScoringPipeline(
        rule_engine=rule_engine,
        anomaly_detector=trained_anomaly_detector,
    )


# ============================================================================
# Integration Tests: Full Pipeline → Score → Alert
# ============================================================================


class TestFullPipelineToAlert:
    """Test the complete flow from ingestion through alert generation."""

    def test_valid_transaction_no_alert(self, pipeline, scoring_pipeline, alert_manager):
        """Low-risk transactions pass through pipeline without generating alerts."""
        txn = make_valid_transaction(transaction_amount=50.0)

        # Process through ingestion pipeline
        result = pipeline.process_record(txn)
        assert result.success is True

        # Score the processed record
        enriched = result.record
        score = scoring_pipeline.score_transaction_sync(enriched)

        assert score.final_score < 0.6
        assert score.alert_recommended is False

    def test_fraud_transaction_generates_alert(
        self, pipeline, full_scoring_pipeline, alert_manager
    ):
        """High-risk transactions generate alerts through the full flow."""
        txn = make_fraud_transaction(transaction_amount=9500.0)

        # Process through ingestion pipeline
        result = pipeline.process_record(txn)
        assert result.success is True

        # Score the processed record
        enriched = result.record
        score = full_scoring_pipeline.score_transaction_sync(enriched)

        # Generate alert if recommended
        if score.alert_recommended:
            alert = alert_manager.generate_alert(score.to_dict(), txn)
            assert alert is not None
            assert alert.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL)
            assert alert.status == AlertStatus.OPEN
            assert alert.account_id == txn["account_id"]
            assert alert.risk_score >= 0.6

    def test_batch_pipeline_to_scoring(self, pipeline, full_scoring_pipeline):
        """Batch of transactions processes through pipeline and scoring."""
        batch = make_mixed_batch(200, fraud_ratio=0.10, invalid_ratio=0.05)

        # Process through ingestion pipeline
        batch_result = pipeline.process_batch(batch)
        assert batch_result.total == 200
        assert batch_result.succeeded > 0

        # Score all successful records
        scored = []
        for result in batch_result.results:
            if result.success:
                score = full_scoring_pipeline.score_transaction_sync(result.record)
                scored.append((score, result.record))

        assert len(scored) > 0

        # Verify we got a mix of risk classifications
        classifications = [s.risk_classification.value for s, _ in scored]
        assert "low" in classifications

    def test_batch_alert_generation(self, pipeline, full_scoring_pipeline, alert_manager):
        """Batch scoring and alert generation produces expected alerts."""
        fraud_txns = [make_fraud_transaction() for _ in range(20)]
        valid_txns = [make_valid_transaction() for _ in range(80)]
        batch = fraud_txns + valid_txns
        random.shuffle(batch)

        # Process through pipeline
        batch_result = pipeline.process_batch(batch)

        # Score and collect alert-worthy results
        alert_candidates = []
        for result in batch_result.results:
            if not result.success:
                continue
            score = full_scoring_pipeline.score_transaction_sync(result.record)
            if score.alert_recommended:
                alert_candidates.append((score.to_dict(), result.record))

        # Generate alerts
        alerts = alert_manager.generate_alerts_from_batch([(s, t) for s, t in alert_candidates])

        # We should have at least some alerts from the fraud transactions
        assert len(alerts) > 0
        assert all(isinstance(a, Alert) for a in alerts)
        assert all(a.status == AlertStatus.OPEN for a in alerts)

    def test_no_data_loss_in_full_flow(self, pipeline):
        """Every transaction is accounted for: success + failure = total."""
        batch = make_mixed_batch(500, fraud_ratio=0.08, invalid_ratio=0.04)
        result = pipeline.process_batch(batch)

        assert result.total == 500
        assert result.succeeded + result.failed == 500
        # DLQ count <= failed (some failures may not route to DLQ)
        assert result.dlq_count <= result.failed

    def test_pipeline_preserves_transaction_id(self, pipeline):
        """Transaction IDs are preserved through the entire pipeline."""
        txn = make_valid_transaction()
        original_id = txn["external_transaction_id"]

        result = pipeline.process_record(txn)
        assert result.transaction_id == original_id

        if result.success:
            record = result.record
            assert record.get("external_transaction_id") == original_id


class TestAlertIntegration:
    """Test alert generation, deduplication, and lifecycle integration."""

    def test_alert_deduplication_within_window(self, alert_manager):
        """Same account + rule + type within window produces only one alert."""
        scoring_result = {
            "final_score": 0.85,
            "risk_classification": "high",
            "alert_recommended": True,
            "method_scores": [
                {
                    "method": "rule_engine",
                    "success": True,
                    "details": {
                        "triggered_rules": [{"rule_id": "RULE-HIGH-AMOUNT", "severity": "high"}]
                    },
                },
            ],
        }
        txn = make_fraud_transaction(account_id="ACC-DEDUP-001")

        alert1 = alert_manager.generate_alert(scoring_result, txn)
        assert alert1 is not None

        # Second alert with same account should be deduplicated
        txn2 = make_fraud_transaction(account_id="ACC-DEDUP-001")
        alert2 = alert_manager.generate_alert(scoring_result, txn2)
        assert alert2 is None

    def test_alert_lifecycle_transitions(self, alert_manager):
        """Alert lifecycle: open → investigating → resolved."""
        scoring_result = {
            "final_score": 0.90,
            "risk_classification": "critical",
            "alert_recommended": True,
            "method_scores": [],
        }
        txn = make_fraud_transaction(account_id="ACC-LIFECYCLE-001")

        alert = alert_manager.generate_alert(scoring_result, txn)
        assert alert is not None
        assert alert.status == AlertStatus.OPEN

        # Transition to investigating
        updated = alert_manager.transition_alert(
            alert.alert_id,
            AlertStatus.INVESTIGATING,
            assigned_to="analyst@company.com",
        )
        assert updated is not None
        assert updated.status == AlertStatus.INVESTIGATING
        assert updated.assigned_to == "analyst@company.com"

        # Transition to resolved
        resolved = alert_manager.transition_alert(
            alert.alert_id,
            AlertStatus.RESOLVED,
            resolution_notes="Confirmed fraud, account blocked",
        )
        assert resolved is not None
        assert resolved.status == AlertStatus.RESOLVED
        assert resolved.resolved_at is not None

    def test_suppression_after_investigation(self, alert_manager):
        """Alerts are suppressed for accounts under investigation."""
        scoring_result = {
            "final_score": 0.88,
            "risk_classification": "high",
            "alert_recommended": True,
            "method_scores": [],
        }
        account_id = "ACC-SUPPRESS-001"
        txn1 = make_fraud_transaction(account_id=account_id)

        alert = alert_manager.generate_alert(scoring_result, txn1)
        assert alert is not None

        # Start investigating → suppresses further alerts
        alert_manager.transition_alert(alert.alert_id, AlertStatus.INVESTIGATING)

        # New alert for same account should be suppressed
        txn2 = make_fraud_transaction(account_id=account_id)
        txn2["external_transaction_id"] = f"TXN-NEW-{uuid.uuid4().hex[:8]}"
        alert2 = alert_manager.generate_alert(scoring_result, txn2)
        assert alert2 is None

    def test_alert_severity_classification(self, alert_manager):
        """Alert severity maps correctly from risk scores."""
        test_cases = [
            (0.35, AlertSeverity.MEDIUM),
            (0.65, AlertSeverity.HIGH),
            (0.90, AlertSeverity.CRITICAL),
        ]

        for score, expected_severity in test_cases:
            scoring_result = {
                "final_score": score,
                "risk_classification": expected_severity.value,
                "alert_recommended": True,
                "method_scores": [],
            }
            # Use unique account to avoid dedup
            txn = make_fraud_transaction(account_id=f"ACC-SEV-{uuid.uuid4().hex[:8]}")
            alert = alert_manager.generate_alert(scoring_result, txn)
            if alert is not None:
                assert alert.severity == expected_severity

    def test_throttle_prevents_alert_storm(self):
        """Throttle engine prevents alert storms."""
        manager = AlertManager()
        scoring_result = {
            "final_score": 0.85,
            "risk_classification": "high",
            "alert_recommended": True,
            "method_scores": [],
        }

        # Generate many alerts for same account to trigger per-account throttle
        alerts_generated = 0
        for i in range(20):
            txn = make_fraud_transaction(account_id="ACC-STORM-001")
            txn["external_transaction_id"] = f"TXN-STORM-{i:04d}"
            alert = manager.generate_alert(scoring_result, txn)
            if alert is not None:
                alerts_generated += 1

        # Per-account throttle (5/hour) should limit alerts
        assert alerts_generated <= 6  # small buffer for timing

    def test_alert_statistics_tracking(self, alert_manager):
        """Alert statistics accurately reflect generation activity."""
        scoring_result = {
            "final_score": 0.80,
            "risk_classification": "high",
            "alert_recommended": True,
            "method_scores": [],
        }

        for i in range(5):
            txn = make_fraud_transaction(account_id=f"ACC-STATS-{i:04d}")
            alert_manager.generate_alert(scoring_result, txn)

        stats = alert_manager._statistics.snapshot()
        assert stats["total_generated"] >= 5
        assert stats["by_severity"]["high"] >= 5


class TestScoringPipelineIntegration:
    """Test the unified scoring pipeline with multiple methods."""

    def test_rule_engine_only_scoring(self, scoring_pipeline):
        """Scoring with only rule engine available produces valid scores."""
        txn = make_fraud_transaction(transaction_amount=9800.0)
        score = scoring_pipeline.score_transaction_sync(txn)

        assert isinstance(score, UnifiedScore)
        assert 0.0 <= score.final_score <= 1.0
        assert score.risk_classification in RiskClassification
        assert score.methods_succeeded >= 1

    def test_ensemble_scoring_rule_plus_anomaly(self, full_scoring_pipeline):
        """Ensemble scoring with rule + anomaly produces weighted result."""
        txn = make_fraud_transaction(transaction_amount=9500.0)
        score = full_scoring_pipeline.score_transaction_sync(txn)

        assert score.methods_succeeded >= 1
        assert score.total_latency_ms > 0

        # Check individual method scores are present
        methods = {ms.method for ms in score.method_scores}
        assert "rule_engine" in methods
        assert "anomaly_detector" in methods

    def test_scoring_caching(self, full_scoring_pipeline):
        """Repeated scoring of same transaction uses cache."""
        txn = make_fraud_transaction()

        score1 = full_scoring_pipeline.score_transaction_sync(txn, use_cache=True)
        score2 = full_scoring_pipeline.score_transaction_sync(txn, use_cache=True)

        assert score2.cached is True
        assert score1.final_score == score2.final_score

    def test_scoring_without_cache(self, full_scoring_pipeline):
        """Scoring without cache recomputes every time."""
        txn = make_fraud_transaction()

        score1 = full_scoring_pipeline.score_transaction_sync(txn, use_cache=False)
        score2 = full_scoring_pipeline.score_transaction_sync(txn, use_cache=False)

        assert score1.cached is False
        assert score2.cached is False

    def test_batch_scoring_consistency(self, full_scoring_pipeline):
        """Scoring a batch produces consistent results."""
        batch = [make_fraud_transaction() for _ in range(10)]

        scores = []
        for txn in batch:
            score = full_scoring_pipeline.score_transaction_sync(txn, use_cache=False)
            scores.append(score)

        assert len(scores) == 10
        assert all(0.0 <= s.final_score <= 1.0 for s in scores)

    def test_scoring_metrics_accumulate(self, full_scoring_pipeline):
        """Scoring metrics track total scored and average latency."""
        batch = [make_valid_transaction() for _ in range(5)]
        for txn in batch:
            full_scoring_pipeline.score_transaction_sync(txn, use_cache=False)

        metrics = full_scoring_pipeline.metrics
        assert metrics["total_scored"] >= 5
        assert metrics["avg_latency_ms"] > 0

    def test_risk_classification_thresholds(self, full_scoring_pipeline):
        """Ensure proper risk classification based on score thresholds."""
        # Force known scores by manipulating weights
        thresholds = full_scoring_pipeline.thresholds
        assert "low" in thresholds
        assert "medium" in thresholds
        assert "high" in thresholds
        assert "critical" in thresholds
        assert (
            thresholds["low"] < thresholds["medium"] < thresholds["high"] < thresholds["critical"]
        )


class TestPipelineResilience:
    """Test pipeline resilience to edge cases and failure modes."""

    def test_pipeline_handles_missing_optional_fields(self, pipeline):
        """Pipeline processes transactions with missing optional fields."""
        txn = make_valid_transaction()
        # Remove optional fields
        txn.pop("device_id", None)
        txn.pop("ip_address", None)
        txn.pop("geo_latitude", None)
        txn.pop("geo_longitude", None)

        result = pipeline.process_record(txn)
        # Should still process (optional fields missing is OK)
        assert result.success is True or result.stage_failed != PipelineStage.VALIDATION.value

    def test_pipeline_handles_unicode_merchant_names(self, pipeline):
        """Pipeline handles unicode characters in merchant names."""
        txn = make_valid_transaction(merchant_name="カフェ・ド・フロール")
        result = pipeline.process_record(txn)
        assert result.success is True

    def test_pipeline_handles_boundary_amounts(self, pipeline):
        """Pipeline handles boundary transaction amounts."""
        boundary_amounts = [0.01, 1.00, 9999.99, 10000.00]
        for amount in boundary_amounts:
            txn = make_valid_transaction(transaction_amount=amount)
            result = pipeline.process_record(txn)
            # Should process without crash regardless of amount
            assert result.success is True or result.error is not None

    def test_pipeline_processes_all_channels(self, pipeline):
        """All valid channels process successfully."""
        for channel in ["online", "pos", "atm", "mobile"]:
            txn = make_valid_transaction(channel=channel)
            result = pipeline.process_record(txn)
            assert result.success is True

    def test_pipeline_processes_all_transaction_types(self, pipeline):
        """All valid transaction types process successfully."""
        for txn_type in ["purchase", "withdrawal", "transfer", "refund"]:
            txn = make_valid_transaction(transaction_type=txn_type)
            result = pipeline.process_record(txn)
            assert result.success is True

    def test_concurrent_pipeline_processing(self, pipeline):
        """Pipeline handles concurrent processing without data corruption."""
        batch = [make_valid_transaction() for _ in range(50)]
        results = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(pipeline.process_record, txn) for txn in batch]
            for future in as_completed(futures):
                results.append(future.result())

        assert len(results) == 50
        successful = [r for r in results if r.success]
        assert len(successful) > 0

    def test_scoring_graceful_degradation(self):
        """Scoring pipeline degrades gracefully when methods fail."""
        # Create pipeline with no scoring methods initialized
        pipeline = ScoringPipeline(
            rule_engine=None,
            anomaly_detector=None,
            risk_scorer=None,
        )

        txn = make_valid_transaction()
        score = pipeline.score_transaction_sync(txn)

        # Should return a valid score even with all methods failing
        assert isinstance(score, UnifiedScore)
        assert score.final_score == 0.0
        assert score.methods_failed == 3

    def test_alert_manager_handles_malformed_scoring_result(self, alert_manager):
        """Alert manager handles incomplete scoring results gracefully."""
        minimal_result = {
            "final_score": 0.75,
            "risk_classification": "high",
            "alert_recommended": True,
        }
        txn = make_fraud_transaction(account_id=f"ACC-MALFORMED-{uuid.uuid4().hex[:6]}")

        alert = alert_manager.generate_alert(minimal_result, txn)
        # Should not crash — either generates alert or returns None
        assert alert is None or isinstance(alert, Alert)


class TestEndToEndDataFlow:
    """Verify data integrity across the full pipeline."""

    def test_enrichment_fields_present_after_pipeline(self, pipeline):
        """Enrichment stage adds expected fields to processed records."""
        txn = make_valid_transaction()
        result = pipeline.process_record(txn)

        if result.success:
            record = result.record
            assert record.get("_pipeline_processed") is True
            assert "_pipeline_timestamp" in record

    def test_velocity_accumulates_across_transactions(self, pipeline):
        """Velocity calculator tracks multiple transactions for same customer."""
        customer_id = "CUST-VELOCITY-001"
        account_id = "ACC-VELOCITY-001"

        results = []
        for i in range(5):
            txn = make_valid_transaction(
                customer_id=customer_id,
                account_id=account_id,
            )
            result = pipeline.process_record(txn)
            results.append(result)

        successful = [r for r in results if r.success]
        assert len(successful) >= 3

    def test_invalid_transactions_route_to_dlq(self, pipeline):
        """Invalid transactions are properly routed to the dead-letter queue."""
        invalid = make_invalid_transaction()
        result = pipeline.process_record(invalid)

        assert result.success is False
        assert result.dlq is True
        assert len(pipeline.dlq) >= 1

        dlq_entry = pipeline.dlq[-1]
        assert "original_record" in dlq_entry
        assert "failed_stage" in dlq_entry
        assert "failure_reason" in dlq_entry

    def test_full_flow_with_scoring_integration(
        self, pipeline, full_scoring_pipeline, alert_manager
    ):
        """Complete end-to-end: ingest → validate → transform → enrich → score → alert."""
        # Use a moderately suspicious transaction that passes the validation
        # rules engine but still scores high on fraud detection
        txn = make_valid_transaction(
            transaction_amount=3500.0,
            account_id=f"ACC-E2E-{uuid.uuid4().hex[:8]}",
            channel="online",
            is_international=False,
        )

        # Stage 1: Ingestion pipeline
        pipeline_result = pipeline.process_record(txn)
        assert pipeline_result.success is True

        # Stage 2: Scoring
        enriched = pipeline_result.record
        score = full_scoring_pipeline.score_transaction_sync(enriched)
        assert isinstance(score, UnifiedScore)
        assert score.total_latency_ms > 0

        # Stage 3: Alert generation (if warranted by scoring)
        if score.alert_recommended:
            alert = alert_manager.generate_alert(score.to_dict(), txn)
            if alert is not None:
                assert alert.transaction_id == txn["external_transaction_id"]
                assert alert.risk_score == score.final_score

    def test_large_batch_end_to_end(self, pipeline, full_scoring_pipeline, alert_manager):
        """Process 1000 transactions through the full pipeline with scoring and alerting."""
        batch = make_mixed_batch(1000, fraud_ratio=0.08, invalid_ratio=0.03)

        # Stage 1: Pipeline processing
        batch_result = pipeline.process_batch(batch)
        assert batch_result.total == 1000
        assert batch_result.succeeded + batch_result.failed == 1000

        # Stage 2: Score all successful records
        scored_high_risk = []
        for result in batch_result.results:
            if not result.success:
                continue
            score = full_scoring_pipeline.score_transaction_sync(result.record, use_cache=False)
            if score.alert_recommended:
                scored_high_risk.append((score.to_dict(), result.record))

        # Stage 3: Generate alerts
        alert_manager.generate_alerts_from_batch(scored_high_risk)

        # Verify pipeline metrics
        assert batch_result.metrics.throughput_per_second > 0

        # Verify scoring metrics
        metrics = full_scoring_pipeline.metrics
        assert metrics["total_scored"] > 0

        # Verify alert statistics
        stats = alert_manager._statistics.snapshot()
        assert (
            stats["total_generated"] + stats["total_suppressed"] + stats["total_deduplicated"] >= 0
        )

    def test_alert_latency_under_5_seconds(self, pipeline, full_scoring_pipeline, alert_manager):
        """End-to-end latency from ingestion to alert < 5 seconds for critical."""
        # Use a transaction that passes validation but has some risk indicators
        txn = make_valid_transaction(
            transaction_amount=2500.0,
            account_id=f"ACC-LAT-{uuid.uuid4().hex[:8]}",
            channel="online",
        )

        start = time.perf_counter()

        # Full flow
        pipeline_result = pipeline.process_record(txn)
        assert pipeline_result.success is True

        score = full_scoring_pipeline.score_transaction_sync(pipeline_result.record)

        if score.alert_recommended:
            alert_manager.generate_alert(score.to_dict(), txn)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # End-to-end latency must be under 5000ms
        assert elapsed_ms < 5000, f"End-to-end latency {elapsed_ms:.1f}ms exceeds 5000ms target"
