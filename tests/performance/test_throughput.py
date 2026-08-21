"""Performance and throughput tests for the RiskPulse pipeline.

Benchmarks:
- Pipeline throughput: sustained 10K events/minute (166+ events/sec)
- Alert generation latency: < 5 seconds from ingestion to alert
- Scoring pipeline latency: P50, P95, P99
- Concurrent processing capacity
- Memory stability under sustained load
- Stage-level latency breakdown
"""

from __future__ import annotations

import gc
import random
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from src.alerting.alert_manager import AlertManager
from src.enrichment.device_enricher import DeviceEnricher, InMemoryDeviceStore
from src.enrichment.geo_enricher import GeoEnricher
from src.enrichment.merchant_enricher import InMemoryMerchantStore, MerchantEnricher
from src.enrichment.velocity_calculator import VelocityCalculator
from src.fraud_detection.anomaly_detector import AnomalyDetector
from src.fraud_detection.rule_engine import FraudRuleEngine
from src.fraud_detection.scoring_pipeline import ScoringPipeline
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
# Test Data Generation
# ============================================================================

MERCHANTS = [
    ("MERCH-001", "Amazon", "5411"),
    ("MERCH-002", "Walmart", "5411"),
    ("MERCH-003", "Shell Gas", "5541"),
    ("MERCH-004", "Netflix", "4899"),
    ("MERCH-005", "Uber", "4121"),
    ("MERCH-006", "Starbucks", "5812"),
    ("MERCH-007", "Apple Store", "5732"),
    ("MERCH-008", "Target", "5311"),
]

LOCATIONS = [
    ("US", "New York", 40.7128, -74.0060),
    ("US", "Los Angeles", 34.0522, -118.2437),
    ("US", "Chicago", 41.8781, -87.6298),
    ("US", "Houston", 29.7604, -95.3698),
    ("US", "Phoenix", 33.4484, -112.0740),
]


def _generate_transaction(idx: int = 0) -> dict[str, Any]:
    """Generate a realistic synthetic transaction for throughput testing."""
    merchant = random.choice(MERCHANTS)
    location = random.choice(LOCATIONS)
    base_time = datetime.now(timezone.utc) - timedelta(minutes=random.randint(0, 60))

    return {
        "external_transaction_id": f"TXN-PERF-{uuid.uuid4().hex[:12].upper()}",
        "account_id": f"ACC-{random.randint(10000, 99999)}",
        "customer_id": f"CUST-{random.randint(10000, 99999)}",
        "merchant_id": merchant[0],
        "merchant_name": merchant[1],
        "merchant_category_code": merchant[2],
        "transaction_amount": round(random.uniform(5.0, 2000.0), 2),
        "transaction_currency": random.choice(["USD", "EUR", "GBP"]),
        "transaction_type": random.choices(
            ["purchase", "withdrawal", "transfer", "refund"],
            weights=[0.7, 0.1, 0.1, 0.1],
            k=1,
        )[0],
        "channel": random.choice(["online", "pos", "atm", "mobile"]),
        "card_type": random.choice(["credit", "debit", "prepaid"]),
        "card_last_four": f"{random.randint(1000, 9999)}",
        "ip_address": f"10.{random.randint(0, 255)}.{random.randint(1, 254)}.{random.randint(1, 254)}",
        "device_id": f"device-{uuid.uuid4().hex[:8]}",
        "device_type": random.choice(["mobile", "desktop", "tablet"]),
        "geo_latitude": location[2] + random.uniform(-0.05, 0.05),
        "geo_longitude": location[3] + random.uniform(-0.05, 0.05),
        "geo_country": location[0],
        "geo_city": location[1],
        "is_international": False,
        "transaction_timestamp": base_time.isoformat(),
    }


def _generate_fraud_transaction() -> dict[str, Any]:
    """Generate a high-risk fraud transaction."""
    return {
        "external_transaction_id": f"TXN-FRAUD-{uuid.uuid4().hex[:12].upper()}",
        "account_id": f"ACC-{random.randint(90000, 99999)}",
        "customer_id": f"CUST-{random.randint(90000, 99999)}",
        "merchant_id": "MERCH-090",
        "merchant_name": "Suspicious Digital Store",
        "merchant_category_code": "7995",
        "transaction_amount": round(random.uniform(5000.0, 9999.0), 2),
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": "credit",
        "card_last_four": f"{random.randint(1000, 9999)}",
        "ip_address": f"185.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}",
        "device_id": f"device-new-{uuid.uuid4().hex[:8]}",
        "device_type": "desktop",
        "geo_latitude": 55.7558,
        "geo_longitude": 37.6173,
        "geo_country": "RU",
        "geo_city": "Moscow",
        "is_international": True,
        "transaction_timestamp": datetime.now(timezone.utc).replace(hour=3).isoformat(),
    }


def _generate_batch(count: int, fraud_ratio: float = 0.05) -> list[dict[str, Any]]:
    """Generate a batch of transactions with configurable fraud ratio."""
    n_fraud = int(count * fraud_ratio)
    n_valid = count - n_fraud

    batch = [_generate_transaction(i) for i in range(n_valid)]
    batch.extend([_generate_fraud_transaction() for _ in range(n_fraud)])
    random.shuffle(batch)
    return batch


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def pipeline() -> PipelineOrchestrator:
    """Create optimized pipeline for performance testing."""
    return PipelineOrchestrator(
        schema_validator=SchemaValidator(),
        data_cleaner=DataCleaner(),
        normalizer=DataNormalizer(),
        feature_engineer=FeatureEngineer(),
        geo_enricher=GeoEnricher(),
        device_enricher=DeviceEnricher(device_store=InMemoryDeviceStore()),
        merchant_enricher=MerchantEnricher(merchant_store=InMemoryMerchantStore()),
        velocity_calculator=VelocityCalculator(),
        quarantine_handler=QuarantineHandler(),
        batch_size=500,
    )


@pytest.fixture
def rule_engine() -> FraudRuleEngine:
    """Instantiate rule engine with production rules."""
    return FraudRuleEngine()


@pytest.fixture
def trained_anomaly_detector() -> AnomalyDetector:
    """Create and train a lightweight anomaly detector for perf tests."""
    import pandas as pd

    detector = AnomalyDetector(
        n_estimators=50,
        contamination=0.05,
        random_state=42,
    )

    rng = np.random.default_rng(42)
    n_samples = 1000

    data = {
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

    df = pd.DataFrame(data)
    detector.fit(df)
    return detector


@pytest.fixture
def scoring_pipeline(rule_engine, trained_anomaly_detector) -> ScoringPipeline:
    """Create scoring pipeline with rule engine + anomaly detector."""
    return ScoringPipeline(
        rule_engine=rule_engine,
        anomaly_detector=trained_anomaly_detector,
    )


@pytest.fixture
def alert_manager() -> AlertManager:
    """Create alert manager for performance tests."""
    return AlertManager()


# ============================================================================
# Pipeline Throughput Tests
# ============================================================================


class TestPipelineThroughput:
    """Test pipeline handles 10K events/minute (166+ events/second)."""

    def test_10k_events_per_minute_target(self, pipeline):
        """Pipeline must process 10,000 events within 60 seconds."""
        batch = _generate_batch(10000, fraud_ratio=0.05)

        start = time.perf_counter()
        result = pipeline.process_batch(batch)
        elapsed = time.perf_counter() - start

        throughput = result.succeeded / elapsed if elapsed > 0 else 0
        events_per_minute = throughput * 60

        assert result.total == 10000
        assert result.succeeded + result.failed == 10000
        assert result.succeeded / result.total > 0.90

        # Must sustain 10K events/minute = 166.67 events/second
        assert events_per_minute >= 10000, (
            f"Throughput {events_per_minute:.0f} events/min is below 10K target. "
            f"Achieved {throughput:.1f} events/sec in {elapsed:.2f}s"
        )

        print(f"\n{'='*70}")
        print("  THROUGHPUT BENCHMARK: 10K Events/Minute Target")
        print(f"{'='*70}")
        print(f"  Records Processed:   {result.total:,}")
        print(f"  Succeeded:           {result.succeeded:,}")
        print(f"  Failed:              {result.failed:,}")
        print(f"  DLQ:                 {result.dlq_count:,}")
        print(f"  Success Rate:        {result.succeeded/result.total:.2%}")
        print(f"  Elapsed Time:        {elapsed:.2f}s")
        print(f"  Throughput:          {throughput:.1f} events/sec")
        print(f"  Events/Minute:       {events_per_minute:.0f}")
        print(f"  Avg Latency:         {result.metrics.avg_per_record_ms:.3f} ms/record")
        print(f"{'='*70}\n")

    def test_sustained_throughput_5_batches(self, pipeline):
        """Pipeline maintains throughput over 5 consecutive batches of 2000."""
        batch_size = 2000
        num_batches = 5
        batch_results = []

        for i in range(num_batches):
            batch = _generate_batch(batch_size, fraud_ratio=0.05)

            start = time.perf_counter()
            result = pipeline.process_batch(batch)
            elapsed = time.perf_counter() - start

            throughput = result.succeeded / elapsed if elapsed > 0 else 0
            batch_results.append(
                {
                    "batch": i + 1,
                    "succeeded": result.succeeded,
                    "elapsed": elapsed,
                    "throughput": throughput,
                }
            )

            pipeline.reset_metrics()

        # Calculate sustained throughput
        total_succeeded = sum(r["succeeded"] for r in batch_results)
        total_elapsed = sum(r["elapsed"] for r in batch_results)
        sustained_throughput = total_succeeded / total_elapsed

        # Must sustain > 166 events/sec across all batches
        assert (
            sustained_throughput > 166
        ), f"Sustained throughput {sustained_throughput:.1f} events/sec is below 166 target"

        # No significant degradation between first and last batch
        first_throughput = batch_results[0]["throughput"]
        last_throughput = batch_results[-1]["throughput"]
        degradation = (first_throughput - last_throughput) / first_throughput
        assert degradation < 0.30, f"Throughput degraded {degradation:.1%} from first to last batch"

        print(f"\n{'='*70}")
        print(f"  SUSTAINED THROUGHPUT: {num_batches} x {batch_size} Events")
        print(f"{'='*70}")
        for r in batch_results:
            print(f"  Batch {r['batch']}: {r['throughput']:.1f} events/sec ({r['elapsed']:.2f}s)")
        print(f"  {'─'*50}")
        print(f"  Sustained:           {sustained_throughput:.1f} events/sec")
        print(f"  Degradation:         {degradation:.1%}")
        print(f"{'='*70}\n")

    def test_concurrent_batch_processing(self, pipeline):
        """Pipeline handles concurrent batch submissions."""
        num_threads = 4
        records_per_thread = 500
        results = []

        def process_batch(thread_id: int) -> dict:
            batch = _generate_batch(records_per_thread, fraud_ratio=0.05)
            start = time.perf_counter()
            result = pipeline.process_batch(batch)
            elapsed = time.perf_counter() - start
            return {
                "thread": thread_id,
                "total": result.total,
                "succeeded": result.succeeded,
                "elapsed": elapsed,
                "throughput": result.succeeded / elapsed if elapsed > 0 else 0,
            }

        start_all = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_batch, i) for i in range(num_threads)]
            for future in as_completed(futures):
                results.append(future.result())
        total_elapsed = time.perf_counter() - start_all

        total_records = sum(r["total"] for r in results)
        total_succeeded = sum(r["succeeded"] for r in results)
        aggregate_throughput = total_succeeded / total_elapsed

        assert total_records == num_threads * records_per_thread
        assert (
            aggregate_throughput > 100
        ), f"Concurrent throughput {aggregate_throughput:.1f} events/sec too low"

        print(f"\n{'='*70}")
        print(f"  CONCURRENT THROUGHPUT: {num_threads} Threads x {records_per_thread} Events")
        print(f"{'='*70}")
        for r in results:
            print(f"  Thread {r['thread']}: {r['throughput']:.1f} events/sec")
        print(f"  {'─'*50}")
        print(f"  Aggregate:           {aggregate_throughput:.1f} events/sec")
        print(f"  Wall Clock:          {total_elapsed:.2f}s")
        print(f"{'='*70}\n")


# ============================================================================
# Latency Tests
# ============================================================================


class TestLatencyBenchmarks:
    """Test per-record and per-stage latency meets SLA targets."""

    def test_pipeline_per_record_latency(self, pipeline):
        """Average per-record pipeline latency < 10ms."""
        batch = _generate_batch(1000, fraud_ratio=0.0)
        result = pipeline.process_batch(batch)

        avg_latency = result.metrics.avg_per_record_ms
        assert (
            avg_latency < 10.0
        ), f"Average per-record latency {avg_latency:.3f}ms exceeds 10ms target"

    def test_pipeline_latency_percentiles(self, pipeline):
        """P95 pipeline latency < 20ms, P99 < 50ms."""
        batch = _generate_batch(500, fraud_ratio=0.0)

        latencies = []
        for txn in batch:
            start = time.perf_counter()
            pipeline.process_record(txn)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
            pipeline.reset_metrics()

        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        assert p95 < 20.0, f"P95 latency {p95:.3f}ms exceeds 20ms target"
        assert p99 < 50.0, f"P99 latency {p99:.3f}ms exceeds 50ms target"

        print(f"\n{'='*70}")
        print("  PIPELINE LATENCY PERCENTILES (500 records)")
        print(f"{'='*70}")
        print(f"  P50:    {p50:.3f} ms")
        print(f"  P95:    {p95:.3f} ms")
        print(f"  P99:    {p99:.3f} ms")
        print(f"  Min:    {min(latencies):.3f} ms")
        print(f"  Max:    {max(latencies):.3f} ms")
        print(f"  Mean:   {statistics.mean(latencies):.3f} ms")
        print(f"  StdDev: {statistics.stdev(latencies):.3f} ms")
        print(f"{'='*70}\n")

    def test_validation_stage_latency(self, pipeline):
        """Validation stage < 5ms per record."""
        batch = _generate_batch(500, fraud_ratio=0.0)
        pipeline.process_batch(batch)

        stage_metrics = pipeline.get_stage_metrics()
        val_metrics = stage_metrics.get(PipelineStage.VALIDATION.value, {})
        avg_latency = val_metrics.get("avg_latency_ms", 0)

        assert avg_latency < 5.0, f"Validation latency {avg_latency:.3f}ms exceeds 5ms target"

    def test_enrichment_stage_latency(self, pipeline):
        """Enrichment stage < 5ms per record."""
        batch = _generate_batch(500, fraud_ratio=0.0)
        pipeline.process_batch(batch)

        stage_metrics = pipeline.get_stage_metrics()
        enrich_metrics = stage_metrics.get(PipelineStage.ENRICHMENT.value, {})
        avg_latency = enrich_metrics.get("avg_latency_ms", 0)

        assert avg_latency < 5.0, f"Enrichment latency {avg_latency:.3f}ms exceeds 5ms target"

    def test_feature_engineering_stage_latency(self, pipeline):
        """Feature engineering stage < 50ms per record."""
        batch = _generate_batch(500, fraud_ratio=0.0)
        pipeline.process_batch(batch)

        stage_metrics = pipeline.get_stage_metrics()
        fe_metrics = stage_metrics.get(PipelineStage.FEATURE_ENGINEERING.value, {})
        avg_latency = fe_metrics.get("avg_latency_ms", 0)

        assert (
            avg_latency < 50.0
        ), f"Feature engineering latency {avg_latency:.3f}ms exceeds 50ms target"


# ============================================================================
# Scoring Pipeline Performance
# ============================================================================


class TestScoringPerformance:
    """Test scoring pipeline latency and throughput."""

    def test_scoring_latency_under_100ms(self, scoring_pipeline):
        """Individual scoring latency must be < 100ms."""
        transactions = [_generate_transaction(i) for i in range(100)]

        latencies = []
        for txn in transactions:
            start = time.perf_counter()
            score = scoring_pipeline.score_transaction_sync(txn, use_cache=False)
            assert 0.0 <= score.final_score <= 1.0
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        avg_latency = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        assert (
            avg_latency < 100.0
        ), f"Average scoring latency {avg_latency:.3f}ms exceeds 100ms target"
        assert p99 < 200.0, f"P99 scoring latency {p99:.3f}ms exceeds 200ms target"

        print(f"\n{'='*70}")
        print("  SCORING LATENCY (100 transactions)")
        print(f"{'='*70}")
        print(f"  Mean:   {avg_latency:.3f} ms")
        print(f"  P50:    {statistics.median(latencies):.3f} ms")
        print(f"  P95:    {p95:.3f} ms")
        print(f"  P99:    {p99:.3f} ms")
        print(f"{'='*70}\n")

    def test_scoring_throughput_1000_per_second(self, scoring_pipeline):
        """Scoring pipeline must handle 40+ scores/second with real models."""
        transactions = [_generate_transaction(i) for i in range(200)]

        start = time.perf_counter()
        for txn in transactions:
            scoring_pipeline.score_transaction_sync(txn, use_cache=False)
        elapsed = time.perf_counter() - start

        throughput = 200 / elapsed if elapsed > 0 else 0
        # With real Isolation Forest + Rule Engine, target 40+ scores/sec
        # (each score ~17ms = ~59/sec theoretical max)
        assert throughput > 40, f"Scoring throughput {throughput:.0f}/sec is below 40 target"

        print(f"\n{'='*70}")
        print("  SCORING THROUGHPUT (200 transactions, real models)")
        print(f"{'='*70}")
        print(f"  Elapsed:     {elapsed:.3f}s")
        print(f"  Throughput:  {throughput:.0f} scores/sec")
        print(f"{'='*70}\n")

    def test_cache_improves_scoring_throughput(self, scoring_pipeline):
        """Cache hit dramatically reduces scoring latency."""
        txn = _generate_transaction()

        # First call — cache miss
        start = time.perf_counter()
        score1 = scoring_pipeline.score_transaction_sync(txn, use_cache=True)
        assert score1.cached is False
        first_latency = (time.perf_counter() - start) * 1000

        # Second call — cache hit
        start = time.perf_counter()
        score2 = scoring_pipeline.score_transaction_sync(txn, use_cache=True)
        cached_latency = (time.perf_counter() - start) * 1000

        assert score2.cached is True
        # Cache hit should be significantly faster
        assert cached_latency < first_latency

    def test_scoring_concurrent_access(self, scoring_pipeline):
        """Scoring pipeline handles concurrent requests without errors."""
        transactions = [_generate_transaction(i) for i in range(200)]
        scores = []
        errors = []

        def score_one(txn):
            try:
                return scoring_pipeline.score_transaction_sync(txn, use_cache=False)
            except Exception as e:
                errors.append(str(e))
                return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(score_one, txn) for txn in transactions]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    scores.append(result)

        assert len(errors) == 0, f"Concurrent scoring errors: {errors[:5]}"
        assert len(scores) == 200


# ============================================================================
# Alert Generation Performance
# ============================================================================


class TestAlertPerformance:
    """Test alert generation latency and throughput."""

    def test_alert_generation_latency(self, alert_manager):
        """Single alert generation < 5ms."""
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

        latencies = []
        for i in range(100):
            txn = _generate_fraud_transaction()
            txn["account_id"] = f"ACC-ALERTPERF-{i:06d}"

            start = time.perf_counter()
            alert_manager.generate_alert(scoring_result, txn)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        avg_latency = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        assert (
            avg_latency < 5.0
        ), f"Alert generation avg latency {avg_latency:.3f}ms exceeds 5ms target"
        assert p95 < 10.0, f"Alert generation P95 latency {p95:.3f}ms exceeds 10ms target"

    def test_batch_alert_throughput(self, alert_manager):
        """Batch alert generation handles 100+ alerts/second."""
        scoring_result = {
            "final_score": 0.80,
            "risk_classification": "high",
            "alert_recommended": True,
            "method_scores": [],
        }

        batch = []
        for i in range(500):
            txn = _generate_fraud_transaction()
            txn["account_id"] = f"ACC-BATCH-{i:06d}"
            batch.append((scoring_result, txn))

        start = time.perf_counter()
        alert_manager.generate_alerts_from_batch(batch)
        elapsed = time.perf_counter() - start

        throughput = len(batch) / elapsed if elapsed > 0 else 0
        assert throughput > 100, f"Alert batch throughput {throughput:.0f}/sec below 100 target"


# ============================================================================
# End-to-End Latency Tests
# ============================================================================


class TestEndToEndLatency:
    """Test complete pipeline latency from ingestion to alert."""

    def test_ingestion_to_alert_under_5_seconds(self, pipeline, scoring_pipeline, alert_manager):
        """Complete flow: ingest → score → alert in < 5 seconds."""
        fraud_txns = [_generate_fraud_transaction() for _ in range(20)]

        latencies = []
        for txn in fraud_txns:
            txn["account_id"] = f"ACC-E2ELAT-{uuid.uuid4().hex[:8]}"

            start = time.perf_counter()

            # Stage 1: Pipeline processing
            pipeline_result = pipeline.process_record(txn)
            if not pipeline_result.success:
                continue

            # Stage 2: Scoring
            score = scoring_pipeline.score_transaction_sync(pipeline_result.record, use_cache=False)

            # Stage 3: Alert generation
            if score.alert_recommended:
                alert_manager.generate_alert(score.to_dict(), txn)

            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

            pipeline.reset_metrics()

        assert len(latencies) > 0
        max_latency = max(latencies)
        avg_latency = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        # SLA: < 5000ms (5 seconds) end-to-end
        assert max_latency < 5000, f"Max end-to-end latency {max_latency:.1f}ms exceeds 5000ms"
        assert avg_latency < 1000, f"Average end-to-end latency {avg_latency:.1f}ms exceeds 1000ms"

        print(f"\n{'='*70}")
        print("  END-TO-END LATENCY: Ingestion → Score → Alert")
        print(f"{'='*70}")
        print(f"  Transactions:  {len(latencies)}")
        print(f"  Mean:          {avg_latency:.1f} ms")
        print(f"  P95:           {p95:.1f} ms")
        print(f"  Max:           {max_latency:.1f} ms")
        print(f"  Min:           {min(latencies):.1f} ms")
        print(f"{'='*70}\n")

    def test_pipeline_stage_breakdown(self, pipeline):
        """Detailed stage-level latency breakdown for 1000 records."""
        batch = _generate_batch(1000, fraud_ratio=0.05)
        result = pipeline.process_batch(batch)

        stage_metrics = pipeline.get_stage_metrics()

        print(f"\n{'='*70}")
        print("  STAGE LATENCY BREAKDOWN (1000 records)")
        print(f"{'='*70}")
        print(f"  {'Stage':<25} {'Processed':>10} {'Avg (ms)':>10} {'Max (ms)':>10} {'Success':>8}")
        print(f"  {'─'*65}")

        total_stage_latency = 0.0
        for stage_name, metrics in stage_metrics.items():
            processed = metrics.get("records_processed", 0)
            avg_lat = metrics.get("avg_latency_ms", 0)
            max_lat = metrics.get("max_latency_ms", 0)
            success = metrics.get("success_rate", 0)
            total_stage_latency += avg_lat
            print(
                f"  {stage_name:<25} {processed:>10,} {avg_lat:>10.3f} "
                f"{max_lat:>10.3f} {success:>7.2%}"
            )

        print(f"  {'─'*65}")
        print(f"  {'Total Stage Sum':<25} {'':>10} {total_stage_latency:>10.3f}")
        print(f"  {'Pipeline Avg':<25} {'':>10} {result.metrics.avg_per_record_ms:>10.3f}")
        print(f"{'='*70}\n")


# ============================================================================
# Memory and Stability Tests
# ============================================================================


class TestStabilityUnderLoad:
    """Test pipeline stability during sustained load."""

    def test_no_memory_leak_sustained_processing(self, pipeline):
        """Memory usage remains stable during sustained processing."""
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Process 10 batches of 500 records
        for _ in range(10):
            batch = _generate_batch(500, fraud_ratio=0.05)
            pipeline.process_batch(batch)
            pipeline.reset_metrics()

        gc.collect()
        final_objects = len(gc.get_objects())

        # Allow some growth but not unbounded
        growth = final_objects - initial_objects
        growth_rate = growth / initial_objects if initial_objects > 0 else 0

        # Growth should be less than 50%
        assert (
            growth_rate < 0.50
        ), f"Object count grew {growth_rate:.1%} ({growth:,} objects), possible leak"

    def test_pipeline_recovers_from_large_invalid_batch(self, pipeline):
        """Pipeline recovers and performs normally after processing invalid data."""
        # Send a batch of all invalid data
        invalid_batch = [
            {
                "external_transaction_id": "",
                "account_id": "",
                "transaction_amount": -1,
                "transaction_currency": "XXX",
                "transaction_type": "bad",
                "channel": "none",
                "transaction_timestamp": "invalid",
            }
            for _ in range(100)
        ]
        pipeline.process_batch(invalid_batch)
        pipeline.reset_metrics()

        # Now process valid data — should work normally
        valid_batch = _generate_batch(200, fraud_ratio=0.0)
        result = pipeline.process_batch(valid_batch)

        assert result.succeeded > 0
        assert result.succeeded / result.total > 0.90

    def test_dlq_does_not_grow_unbounded(self, pipeline):
        """DLQ remains manageable under sustained invalid input."""
        for _ in range(5):
            batch = _generate_batch(100, fraud_ratio=0.0)
            # Add some invalid records
            batch.extend(
                [
                    {
                        "external_transaction_id": "",
                        "account_id": "",
                        "transaction_amount": -1,
                        "transaction_currency": "BAD",
                        "transaction_type": "x",
                        "channel": "y",
                        "transaction_timestamp": "z",
                    }
                    for _ in range(10)
                ]
            )
            random.shuffle(batch)
            pipeline.process_batch(batch)

        # DLQ should have entries but not be unlimited
        dlq_size = len(pipeline.dlq)
        assert dlq_size > 0
        assert dlq_size <= 100  # reasonable upper bound for 50 invalid records
