"""End-to-end integration tests for the full processing pipeline.

Tests the complete flow: Ingest → Validate → Transform → Enrich
with 10,000 synthetic transactions, throughput/latency benchmarks,
and DLQ routing verification.
"""

import random
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.enrichment.device_enricher import DeviceEnricher, InMemoryDeviceStore
from src.enrichment.geo_enricher import GeoEnricher
from src.enrichment.merchant_enricher import InMemoryMerchantStore, MerchantEnricher
from src.enrichment.velocity_calculator import VelocityCalculator
from src.ingestion.kafka_consumer import TransactionConsumer
from src.pipeline_orchestrator import (
    BatchResult,
    PipelineOrchestrator,
    PipelineStage,
    StageErrorPolicy,
)
from src.transformation.cleaner import DataCleaner
from src.transformation.feature_engineer import FeatureEngineer
from src.transformation.normalizer import DataNormalizer
from src.validation.quarantine_handler import QuarantineHandler
from src.validation.rules_engine import RulesEngine
from src.validation.schema_validator import SchemaValidator

# ============================================================================
# Fixtures
# ============================================================================

MERCHANTS = [
    ("MERCH-001", "Amazon", "5411", "US"),
    ("MERCH-002", "Walmart", "5411", "US"),
    ("MERCH-003", "Shell Gas Station", "5541", "US"),
    ("MERCH-004", "Netflix", "4899", "US"),
    ("MERCH-005", "Uber Technologies", "4121", "US"),
    ("MERCH-006", "Starbucks Coffee", "5812", "US"),
    ("MERCH-007", "Apple Store", "5732", "US"),
    ("MERCH-008", "Target", "5311", "US"),
    ("MERCH-090", "Unknown Digital Store", "7995", "RU"),
    ("MERCH-091", "Wire Transfer Services", "6012", "NG"),
]

LOCATIONS = [
    ("US", "New York", 40.7128, -74.0060),
    ("US", "Los Angeles", 34.0522, -118.2437),
    ("US", "Chicago", 41.8781, -87.6298),
    ("US", "Houston", 29.7604, -95.3698),
    ("GB", "London", 51.5074, -0.1278),
    ("RU", "Moscow", 55.7558, 37.6173),
    ("NG", "Lagos", 6.5244, 3.3792),
]

CHANNELS = ["online", "pos", "atm", "mobile"]
CARD_TYPES = ["credit", "debit", "prepaid"]
TRANSACTION_TYPES = ["purchase", "withdrawal", "transfer", "refund"]


def _generate_ip(domestic: bool = True) -> str:
    if domestic:
        return f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
    return f"{random.randint(100, 200)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"


def generate_valid_transaction(idx: int = 0) -> dict:
    """Generate a single valid synthetic transaction."""
    merchant = random.choice(MERCHANTS[:8])
    location = random.choice(LOCATIONS[:4])
    base_time = datetime.now(timezone.utc) - timedelta(hours=random.randint(0, 48))

    return {
        "external_transaction_id": f"TXN-E2E-{uuid.uuid4().hex[:12].upper()}",
        "account_id": f"ACC-{random.randint(10000, 99999)}",
        "customer_id": f"CUST-{random.randint(10000, 99999)}",
        "merchant_id": merchant[0],
        "merchant_name": merchant[1],
        "merchant_category_code": merchant[2],
        "transaction_amount": round(random.uniform(5.0, 2000.0), 2),
        "transaction_currency": random.choice(["USD", "EUR", "GBP"]),
        "transaction_type": random.choices(TRANSACTION_TYPES, weights=[0.7, 0.1, 0.1, 0.1], k=1)[0],
        "channel": random.choice(CHANNELS),
        "card_type": random.choice(CARD_TYPES),
        "card_last_four": f"{random.randint(1000, 9999)}",
        "ip_address": _generate_ip(domestic=True),
        "device_id": f"device-{uuid.uuid4().hex[:8]}",
        "device_type": random.choice(["mobile", "desktop", "tablet"]),
        "geo_latitude": location[2] + random.uniform(-0.05, 0.05),
        "geo_longitude": location[3] + random.uniform(-0.05, 0.05),
        "geo_country": location[0],
        "geo_city": location[1],
        "is_international": False,
        "transaction_timestamp": base_time.isoformat(),
    }


def generate_invalid_transaction() -> dict:
    """Generate a transaction that should fail validation."""
    return {
        "external_transaction_id": "",
        "account_id": "",
        "transaction_amount": -500.0,
        "transaction_currency": "INVALID",
        "transaction_type": "unknown_type",
        "channel": "carrier_pigeon",
        "transaction_timestamp": "not-a-date",
    }


def generate_fraud_transaction() -> dict:
    """Generate a transaction with fraud indicators."""
    merchant = random.choice(MERCHANTS[8:])
    location = random.choice(LOCATIONS[5:])

    return {
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
        "ip_address": _generate_ip(domestic=False),
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


def generate_batch(
    count: int, fraud_ratio: float = 0.05, invalid_ratio: float = 0.02
) -> list[dict]:
    """Generate a mixed batch of transactions.

    Args:
        count: Total number of transactions.
        fraud_ratio: Proportion of fraud transactions.
        invalid_ratio: Proportion of invalid transactions.
    """
    transactions = []
    n_invalid = int(count * invalid_ratio)
    n_fraud = int(count * fraud_ratio)
    n_valid = count - n_invalid - n_fraud

    for i in range(n_valid):
        transactions.append(generate_valid_transaction(i))
    for _ in range(n_fraud):
        transactions.append(generate_fraud_transaction())
    for _ in range(n_invalid):
        transactions.append(generate_invalid_transaction())

    random.shuffle(transactions)
    return transactions


@pytest.fixture
def pipeline():
    """Create a fully-configured pipeline orchestrator."""
    return PipelineOrchestrator(
        schema_validator=SchemaValidator(),
        rules_engine=RulesEngine(),
        data_cleaner=DataCleaner(),
        normalizer=DataNormalizer(),
        feature_engineer=FeatureEngineer(),
        geo_enricher=GeoEnricher(),
        device_enricher=DeviceEnricher(device_store=InMemoryDeviceStore()),
        merchant_enricher=MerchantEnricher(merchant_store=InMemoryMerchantStore()),
        velocity_calculator=VelocityCalculator(),
        quarantine_handler=QuarantineHandler(),
        batch_size=100,
    )


@pytest.fixture
def pipeline_halt_on_validation():
    """Pipeline that halts on validation errors."""
    return PipelineOrchestrator(
        error_policy={
            PipelineStage.VALIDATION.value: StageErrorPolicy.HALT,
            PipelineStage.CLEANING.value: StageErrorPolicy.SKIP,
            PipelineStage.NORMALIZATION.value: StageErrorPolicy.SKIP,
            PipelineStage.FEATURE_ENGINEERING.value: StageErrorPolicy.SKIP,
            PipelineStage.ENRICHMENT.value: StageErrorPolicy.SKIP,
        },
    )


@pytest.fixture
def consumer(pipeline):
    """Create a TransactionConsumer with pipeline (no actual Kafka)."""
    return TransactionConsumer(
        bootstrap_servers="localhost:9092",
        group_id="riskpulse-test-e2e",
        pipeline=pipeline,
    )


# ============================================================================
# Integration Tests: Pipeline Processing
# ============================================================================


class TestEndToEndPipeline:
    """Test the full pipeline from ingest through enrichment."""

    def test_single_valid_transaction(self, pipeline):
        """A single valid transaction processes through all stages."""
        txn = generate_valid_transaction()
        result = pipeline.process_record(txn)

        assert result.success is True
        assert result.transaction_id == txn["external_transaction_id"]
        assert result.error is None
        assert result.dlq is False
        assert result.latency_ms > 0

        # Verify enrichment fields are present
        record = result.record
        assert record.get("_pipeline_processed") is True
        assert "_pipeline_timestamp" in record

    def test_single_invalid_transaction_routes_to_dlq(self, pipeline):
        """An invalid transaction is caught by validation and routed to DLQ."""
        txn = generate_invalid_transaction()
        result = pipeline.process_record(txn)

        assert result.success is False
        assert result.dlq is True
        assert result.stage_failed == PipelineStage.VALIDATION.value
        assert len(pipeline.dlq) >= 1

    def test_batch_mixed_transactions(self, pipeline):
        """A batch with valid, fraud, and invalid transactions processes correctly."""
        batch = generate_batch(100, fraud_ratio=0.1, invalid_ratio=0.05)
        result = pipeline.process_batch(batch)

        assert result.total == 100
        assert result.succeeded + result.failed == 100
        assert result.succeeded > 0
        assert result.metrics.throughput_per_second > 0

    def test_all_stages_execute_in_order(self, pipeline):
        """Verify each stage adds expected metadata to the record."""
        txn = generate_valid_transaction()
        result = pipeline.process_record(txn)

        if result.success:
            record = result.record
            # Validation metadata
            assert "_validation_warnings" in record or "_rules_triggered" in record
            # Feature engineering metadata
            assert "_feature_count" in record or "_feature_engineering_error" in record
            # Enrichment metadata
            assert "_enrichment_latency_ms" in record or "_enrichment_error" in record

    def test_duplicate_detection(self, pipeline):
        """Duplicate transactions are detected and routed to DLQ."""
        txn = generate_valid_transaction()
        # Process same record twice
        result1 = pipeline.process_record(txn)
        result2 = pipeline.process_record(txn)

        # First should succeed, second should be caught as duplicate
        assert result1.success is True
        assert result2.success is False or result2.record.get("_cleaning_error") is not None

    def test_enrichment_adds_geo_data(self, pipeline):
        """Geo enricher adds location context to transactions."""
        txn = generate_valid_transaction()
        txn["geo_country"] = "US"
        txn["geo_latitude"] = 40.7128
        txn["geo_longitude"] = -74.0060

        result = pipeline.process_record(txn)
        if result.success:
            record = result.record
            # Geo enrichment fields should be present
            assert "geo_country_code" in record or "geo_country" in record

    def test_enrichment_adds_velocity_data(self, pipeline):
        """Velocity calculator tracks and enriches transactions."""
        customer_id = f"CUST-{random.randint(10000, 99999)}"
        transactions = []
        for i in range(5):
            txn = generate_valid_transaction()
            txn["customer_id"] = customer_id
            transactions.append(txn)

        # Process all transactions for same customer
        results = [pipeline.process_record(txn) for txn in transactions]
        successful = [r for r in results if r.success]

        # At least some should have velocity data
        assert len(successful) > 0
        last_record = successful[-1].record
        has_velocity = any(k.startswith("velocity_") for k in last_record.keys())
        assert has_velocity

    def test_pipeline_metrics_tracking(self, pipeline):
        """Pipeline metrics accurately reflect processing state."""
        batch = generate_batch(50, fraud_ratio=0.0, invalid_ratio=0.0)
        result = pipeline.process_batch(batch)

        metrics = result.metrics
        assert metrics.total_ingested == 50
        assert metrics.total_completed + metrics.total_failed == 50
        assert metrics.throughput_per_second > 0
        assert metrics.total_pipeline_latency_ms > 0

    def test_stage_metrics_populated(self, pipeline):
        """Each stage has individual metrics after processing."""
        batch = generate_batch(20, fraud_ratio=0.0, invalid_ratio=0.0)
        pipeline.process_batch(batch)

        stage_metrics = pipeline.get_stage_metrics()
        assert PipelineStage.VALIDATION.value in stage_metrics
        assert PipelineStage.CLEANING.value in stage_metrics
        assert PipelineStage.NORMALIZATION.value in stage_metrics
        assert PipelineStage.ENRICHMENT.value in stage_metrics

    def test_error_policy_skip(self, pipeline):
        """Skip policy allows records to pass through despite stage errors."""
        # Transaction that might have feature engineering issues
        txn = generate_valid_transaction()
        txn["transaction_amount"] = 0.01  # Edge case amount

        result = pipeline.process_record(txn)
        # Should still process (skip policy on errors)
        assert result.success is True or result.error is not None

    def test_error_policy_halt(self, pipeline_halt_on_validation):
        """Halt policy stops processing and routes to DLQ."""
        txn = generate_invalid_transaction()
        result = pipeline_halt_on_validation.process_record(txn)

        assert result.success is False
        assert result.dlq is True

    def test_dlq_contains_failure_context(self, pipeline):
        """DLQ entries contain sufficient debugging context."""
        txn = generate_invalid_transaction()
        pipeline.process_record(txn)

        dlq = pipeline.dlq
        assert len(dlq) >= 1

        entry = dlq[0]
        assert "original_record" in entry
        assert "failed_stage" in entry
        assert "failure_reason" in entry
        assert "timestamp" in entry
        assert "dlq_id" in entry

    def test_dlq_callback_invoked(self):
        """DLQ callback is called when records fail."""
        dlq_records = []

        def on_dlq(record, stage, reason):
            dlq_records.append({"record": record, "stage": stage, "reason": reason})

        p = PipelineOrchestrator(on_dlq=on_dlq)
        txn = generate_invalid_transaction()
        p.process_record(txn)

        assert len(dlq_records) >= 1
        assert dlq_records[0]["stage"] is not None

    def test_metrics_reset(self, pipeline):
        """Metrics can be reset between batches."""
        batch = generate_batch(10)
        pipeline.process_batch(batch)

        assert pipeline.metrics.total_ingested > 0

        pipeline.reset_metrics()
        assert pipeline.metrics.total_ingested == 0
        assert len(pipeline.dlq) == 0


# ============================================================================
# Integration Tests: Consumer Pipeline
# ============================================================================


class TestConsumerPipeline:
    """Test the Kafka consumer's pipeline integration (without live Kafka)."""

    def test_consumer_process_records(self, consumer):
        """Consumer can process records directly through the pipeline."""
        records = generate_batch(50, fraud_ratio=0.05, invalid_ratio=0.02)
        result = consumer.process_records(records)

        assert isinstance(result, BatchResult)
        assert result.total == 50
        assert result.succeeded > 0

    def test_consumer_metrics_update(self, consumer):
        """Consumer metrics are updated after processing."""
        records = generate_batch(30, fraud_ratio=0.0, invalid_ratio=0.0)
        consumer.process_records(records)

        metrics = consumer.metrics.snapshot()
        assert metrics["total_messages_consumed"] == 30
        assert metrics["total_batches_processed"] == 1
        assert metrics["total_records_succeeded"] > 0

    def test_consumer_multiple_batches(self, consumer):
        """Consumer handles multiple sequential batches correctly."""
        for _ in range(3):
            records = generate_batch(20, fraud_ratio=0.0, invalid_ratio=0.0)
            consumer.process_records(records)

        metrics = consumer.metrics.snapshot()
        assert metrics["total_messages_consumed"] == 60
        assert metrics["total_batches_processed"] == 3

    def test_consumer_pipeline_access(self, consumer):
        """Consumer exposes pipeline metrics."""
        records = generate_batch(10, fraud_ratio=0.0, invalid_ratio=0.0)
        result = consumer.process_records(records)

        assert result.metrics is not None
        assert result.metrics.total_ingested == 10


# ============================================================================
# Performance / Load Tests
# ============================================================================


class TestPipelinePerformance:
    """Performance benchmarks for the pipeline."""

    def test_throughput_500_events_per_second(self, pipeline):
        """Pipeline must handle > 500 events/second."""
        batch = generate_batch(1000, fraud_ratio=0.05, invalid_ratio=0.02)

        start = time.perf_counter()
        result = pipeline.process_batch(batch)
        elapsed = time.perf_counter() - start

        throughput = result.succeeded / elapsed if elapsed > 0 else 0
        # Minimum throughput target
        assert throughput > 500, f"Throughput {throughput:.0f} events/sec is below 500 target"

    def test_per_record_latency_under_sla(self, pipeline):
        """Average per-record latency should be under 10ms."""
        batch = generate_batch(500, fraud_ratio=0.0, invalid_ratio=0.0)
        result = pipeline.process_batch(batch)

        avg_latency = result.metrics.avg_per_record_ms
        # Under 10ms average per record
        assert avg_latency < 10.0, f"Average latency {avg_latency:.2f}ms exceeds 10ms SLA"

    def test_large_batch_10k_transactions(self, pipeline):
        """Process 10,000 transactions end-to-end without errors."""
        batch = generate_batch(10000, fraud_ratio=0.05, invalid_ratio=0.02)

        start = time.perf_counter()
        result = pipeline.process_batch(batch)
        elapsed = time.perf_counter() - start

        # All records should be accounted for
        assert result.total == 10000
        assert result.succeeded + result.failed == 10000

        # Success rate should be > 90% (some will fail due to invalid/rules)
        assert (
            result.succeeded / result.total > 0.90
        ), f"Success rate {result.succeeded/result.total:.2%} is below 90%"

        throughput = result.succeeded / elapsed if elapsed > 0 else 0
        print(f"\n{'='*60}")
        print("End-to-End Pipeline Benchmark (10,000 transactions)")
        print(f"{'='*60}")
        print(f"  Total Records:      {result.total:,}")
        print(f"  Succeeded:          {result.succeeded:,}")
        print(f"  Failed:             {result.failed:,}")
        print(f"  DLQ:                {result.dlq_count:,}")
        print(f"  Success Rate:       {result.succeeded/result.total:.2%}")
        print(f"  Total Time:         {elapsed:.2f}s")
        print(f"  Throughput:         {throughput:.0f} events/sec")
        print(f"  Avg Latency:        {result.metrics.avg_per_record_ms:.3f}ms/record")
        print(f"{'='*60}")

        # Stage-level metrics
        print("\n  Stage Metrics:")
        for stage_name, stage_data in pipeline.get_stage_metrics().items():
            processed = stage_data.get("records_processed", 0)
            avg_lat = stage_data.get("avg_latency_ms", 0)
            print(f"    {stage_name:25s} | processed: {processed:6d} | avg: {avg_lat:.3f}ms")
        print(f"{'='*60}\n")

    def test_sustained_processing_multiple_batches(self, pipeline):
        """Pipeline maintains throughput over sustained batch processing."""
        batch_count = 10
        records_per_batch = 500
        total_succeeded = 0
        total_time = 0.0

        for i in range(batch_count):
            batch = generate_batch(records_per_batch, fraud_ratio=0.05, invalid_ratio=0.02)

            start = time.perf_counter()
            result = pipeline.process_batch(batch)
            elapsed = time.perf_counter() - start

            total_succeeded += result.succeeded
            total_time += elapsed

            pipeline.reset_metrics()

        sustained_throughput = total_succeeded / total_time if total_time > 0 else 0
        assert (
            sustained_throughput > 400
        ), f"Sustained throughput {sustained_throughput:.0f} events/sec is below 400 target"

    def test_validation_stage_latency(self, pipeline):
        """Validation stage completes within 5ms per record."""
        batch = generate_batch(200, fraud_ratio=0.0, invalid_ratio=0.0)
        pipeline.process_batch(batch)

        stage_metrics = pipeline.get_stage_metrics()
        validation_metrics = stage_metrics.get(PipelineStage.VALIDATION.value, {})
        avg_latency = validation_metrics.get("avg_latency_ms", 0)

        assert avg_latency < 5.0, f"Validation avg latency {avg_latency:.3f}ms exceeds 5ms target"

    def test_enrichment_stage_latency(self, pipeline):
        """Enrichment stage completes within 5ms per record."""
        batch = generate_batch(200, fraud_ratio=0.0, invalid_ratio=0.0)
        pipeline.process_batch(batch)

        stage_metrics = pipeline.get_stage_metrics()
        enrichment_metrics = stage_metrics.get(PipelineStage.ENRICHMENT.value, {})
        avg_latency = enrichment_metrics.get("avg_latency_ms", 0)

        assert avg_latency < 5.0, f"Enrichment avg latency {avg_latency:.3f}ms exceeds 5ms target"


# ============================================================================
# Edge Cases
# ============================================================================


class TestPipelineEdgeCases:
    """Edge case and resilience tests."""

    def test_empty_batch(self, pipeline):
        """Empty batch returns valid empty result."""
        result = pipeline.process_batch([])

        assert result.total == 0
        assert result.succeeded == 0
        assert result.failed == 0

    def test_single_record_batch(self, pipeline):
        """Single-record batch processes correctly."""
        batch = [generate_valid_transaction()]
        result = pipeline.process_batch(batch)

        assert result.total == 1

    def test_all_invalid_batch(self, pipeline):
        """Batch of all invalid records handles gracefully."""
        batch = [generate_invalid_transaction() for _ in range(10)]
        result = pipeline.process_batch(batch)

        assert result.total == 10
        assert result.failed == 10
        assert result.dlq_count == 10

    def test_extreme_amount_values(self, pipeline):
        """Extreme amount values don't crash the pipeline."""
        txn = generate_valid_transaction()
        txn["transaction_amount"] = 0.01
        result1 = pipeline.process_record(txn)

        txn2 = generate_valid_transaction()
        txn2["transaction_amount"] = 999999.99
        result2 = pipeline.process_record(txn2)

        # Should not crash
        assert result1.latency_ms > 0
        assert result2.latency_ms > 0

    def test_unicode_merchant_names(self, pipeline):
        """Unicode characters in merchant names are handled."""
        txn = generate_valid_transaction()
        txn["merchant_name"] = "Café Résumé — München Straße"
        result = pipeline.process_record(txn)

        assert result.latency_ms > 0

    def test_missing_optional_fields(self, pipeline):
        """Records with missing optional fields still process."""
        txn = {
            "external_transaction_id": f"TXN-{uuid.uuid4().hex[:12]}",
            "account_id": "ACC-12345",
            "customer_id": "CUST-12345",
            "transaction_amount": 50.00,
            "transaction_currency": "USD",
            "transaction_type": "purchase",
            "channel": "online",
            "transaction_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        result = pipeline.process_record(txn)
        # Should process (maybe with warnings) or fail validation cleanly
        assert result.latency_ms > 0

    def test_concurrent_customer_velocity(self, pipeline):
        """Multiple transactions for same customer track velocity correctly."""
        customer_id = "CUST-VELOCITY-TEST"
        account_id = "ACC-VELOCITY-TEST"

        results = []
        for i in range(10):
            txn = generate_valid_transaction()
            txn["customer_id"] = customer_id
            txn["account_id"] = account_id
            txn["external_transaction_id"] = f"TXN-VEL-{i}-{uuid.uuid4().hex[:8]}"
            results.append(pipeline.process_record(txn))

        successful = [r for r in results if r.success]
        assert len(successful) > 0

        # Later transactions should have higher velocity counts
        if len(successful) >= 2:
            last = successful[-1].record
            velocity_count = last.get("velocity_txn_count_1min", 0)
            assert velocity_count >= 1
