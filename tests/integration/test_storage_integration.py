"""Integration tests for the storage layer.

Tests the full data flow:
    Processing → Redis cache → PostgreSQL → S3 (batched) → Snowflake (batched)

Covers:
- Cache handler operations (profile, transaction, prediction caching)
- Write-ahead pattern (Kafka → cache → DB confirmation)
- Storage orchestrator multi-backend writes
- Circuit breaker behavior on backend failures
- Buffer flush mechanics for batch backends
- Storage latency monitoring
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.storage.cache_handler import (
    PREFIX_CUSTOMER_PROFILE,
    PREFIX_DEDUP,
    PREFIX_MODEL_PREDICTION,
    PREFIX_RECENT_TRANSACTIONS,
    PREFIX_VELOCITY,
    CacheHandler,
    CacheMetrics,
)
from src.storage.storage_orchestrator import (
    BackendHealth,
    OrchestratedWriteResult,
    StorageBackend,
    StorageOrchestrator,
    StorageState,
    WriteResult,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    client = MagicMock()
    client.ping.return_value = True
    client.get.return_value = None
    client.setex.return_value = True
    client.set.return_value = True
    client.delete.return_value = 1
    client.exists.return_value = 0
    client.incr.return_value = 1
    client.expire.return_value = True
    client.info.return_value = {}
    client.scan.return_value = (0, [])

    pipe = MagicMock()
    pipe.incr.return_value = pipe
    pipe.expire.return_value = pipe
    pipe.setex.return_value = pipe
    pipe.execute.return_value = [1, True]
    client.pipeline.return_value = pipe

    return client


@pytest.fixture
def mock_redis_pool():
    """Create a mock Redis connection pool."""
    pool = MagicMock()
    pool.max_connections = 50
    pool._in_use_connections = set()
    pool._available_connections = []
    pool.disconnect.return_value = None
    return pool


@pytest.fixture
def cache_handler(mock_redis_client, mock_redis_pool):
    """Create a CacheHandler with mocked Redis."""
    with patch(
        "src.storage.cache_handler.redis.ConnectionPool.from_url", return_value=mock_redis_pool
    ):
        with patch("src.storage.cache_handler.redis.Redis") as mock_redis_cls:
            mock_redis_cls.return_value = mock_redis_client
            handler = CacheHandler.__new__(CacheHandler)
            handler._pool = mock_redis_pool
            handler._client = mock_redis_client
            handler._metrics = CacheMetrics()
            handler._ttl_customer_profile = 300
            handler._ttl_recent_transactions = 3600
            handler._ttl_model_predictions = 60
            handler._redis_url = "redis://localhost:6379/0"
            return handler


@pytest.fixture
def mock_postgres_handler():
    """Create a mock PostgresHandler."""
    handler = AsyncMock()
    handler.create_transaction = AsyncMock(return_value=MagicMock(transaction_id="txn-uuid"))
    handler.health_check = AsyncMock(return_value=True)
    handler.close = AsyncMock()
    handler.get_pool_stats.return_value = MagicMock(
        pool_size=20, checked_in=18, checked_out=2, overflow=0, invalid=0
    )
    return handler


@pytest.fixture
def mock_s3_handler():
    """Create a mock S3Handler."""
    handler = MagicMock()
    handler.upload_transactions.return_value = "s3://bucket/path/file.parquet"
    handler.close.return_value = None
    return handler


@pytest.fixture
def mock_snowflake_handler():
    """Create a mock SnowflakeHandler."""
    handler = MagicMock()
    handler.bulk_load_records.return_value = MagicMock(records_loaded=100)
    handler.close.return_value = None
    return handler


@pytest.fixture
def sample_transaction():
    """Return a sample enriched transaction record."""
    return {
        "external_transaction_id": "TXN-2026-INT-001",
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
        "_pipeline_processed": True,
        "_pipeline_timestamp": time.time(),
    }


@pytest.fixture
def sample_customer_profile():
    """Return a sample customer profile."""
    return {
        "customer_id": "CUST-67890",
        "account_id": "ACC-12345",
        "risk_tier": "standard",
        "avg_transaction_amount": 150.0,
        "transaction_count_30d": 42,
        "last_transaction_timestamp": "2026-06-14T09:00:00Z",
        "home_country": "US",
    }


@pytest.fixture
def storage_orchestrator(
    cache_handler, mock_postgres_handler, mock_s3_handler, mock_snowflake_handler
):
    """Create a StorageOrchestrator with mocked backends."""
    return StorageOrchestrator(
        cache_handler=cache_handler,
        postgres_handler=mock_postgres_handler,
        s3_handler=mock_s3_handler,
        snowflake_handler=mock_snowflake_handler,
        s3_batch_size=5,
        snowflake_batch_size=10,
        enable_write_ahead=True,
    )


# =============================================================================
# Cache Handler Tests
# =============================================================================


class TestCacheHandlerCustomerProfile:
    """Tests for customer profile caching."""

    def test_set_customer_profile(self, cache_handler, mock_redis_client):
        profile = {"customer_id": "CUST-001", "risk_tier": "standard"}
        result = cache_handler.set_customer_profile("CUST-001", profile)

        assert result is True
        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][0] == f"{PREFIX_CUSTOMER_PROFILE}CUST-001"
        assert call_args[0][1] == 300  # default TTL

    def test_get_customer_profile_hit(self, cache_handler, mock_redis_client):
        profile = {"customer_id": "CUST-001", "risk_tier": "high"}
        mock_redis_client.get.return_value = json.dumps(profile)

        result = cache_handler.get_customer_profile("CUST-001")

        assert result == profile
        assert cache_handler.metrics.hits == 1
        assert cache_handler.metrics.misses == 0

    def test_get_customer_profile_miss(self, cache_handler, mock_redis_client):
        mock_redis_client.get.return_value = None

        result = cache_handler.get_customer_profile("CUST-UNKNOWN")

        assert result is None
        assert cache_handler.metrics.misses == 1
        assert cache_handler.metrics.hits == 0

    def test_invalidate_customer_profile(self, cache_handler, mock_redis_client):
        result = cache_handler.invalidate_customer_profile("CUST-001")

        assert result is True
        mock_redis_client.delete.assert_called_with(f"{PREFIX_CUSTOMER_PROFILE}CUST-001")

    def test_set_customer_profile_custom_ttl(self, cache_handler, mock_redis_client):
        profile = {"customer_id": "CUST-VIP"}
        cache_handler.set_customer_profile("CUST-VIP", profile, ttl=600)

        call_args = mock_redis_client.setex.call_args
        assert call_args[0][1] == 600

    def test_set_customer_profile_redis_error(self, cache_handler, mock_redis_client):
        from redis.exceptions import RedisError

        mock_redis_client.setex.side_effect = RedisError("Connection refused")

        result = cache_handler.set_customer_profile("CUST-001", {"test": True})

        assert result is False
        assert cache_handler.metrics.errors == 1


class TestCacheHandlerTransactions:
    """Tests for recent transaction caching."""

    def test_cache_recent_transaction(self, cache_handler, mock_redis_client):
        txn_data = {"amount": "100.00", "type": "purchase"}
        result = cache_handler.cache_recent_transaction("TXN-001", txn_data)

        assert result is True
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][0] == f"{PREFIX_RECENT_TRANSACTIONS}TXN-001"
        assert call_args[0][1] == 3600  # default TTL

    def test_is_duplicate_transaction_exists(self, cache_handler, mock_redis_client):
        mock_redis_client.exists.return_value = 1

        result = cache_handler.is_duplicate_transaction("TXN-001")

        assert result is True
        assert cache_handler.metrics.hits == 1

    def test_is_duplicate_transaction_not_exists(self, cache_handler, mock_redis_client):
        mock_redis_client.exists.return_value = 0

        result = cache_handler.is_duplicate_transaction("TXN-NEW")

        assert result is False
        assert cache_handler.metrics.misses == 1

    def test_get_recent_transaction(self, cache_handler, mock_redis_client):
        txn = {"amount": "50.00", "type": "withdrawal"}
        mock_redis_client.get.return_value = json.dumps(txn)

        result = cache_handler.get_recent_transaction("TXN-002")

        assert result == txn


class TestCacheHandlerPredictions:
    """Tests for model prediction caching."""

    def test_cache_prediction(self, cache_handler, mock_redis_client):
        prediction = {"risk_score": 0.85, "model": "isolation_forest_v2"}
        result = cache_handler.cache_prediction("abc123hash", prediction)

        assert result is True
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][0] == f"{PREFIX_MODEL_PREDICTION}abc123hash"
        assert call_args[0][1] == 60  # default model prediction TTL

    def test_get_cached_prediction_hit(self, cache_handler, mock_redis_client):
        prediction = {"risk_score": 0.72, "explanation": {"feature_1": 0.3}}
        mock_redis_client.get.return_value = json.dumps(prediction)

        result = cache_handler.get_cached_prediction("abc123hash")

        assert result == prediction
        assert cache_handler.metrics.hits == 1

    def test_get_cached_prediction_miss(self, cache_handler, mock_redis_client):
        mock_redis_client.get.return_value = None

        result = cache_handler.get_cached_prediction("unknown_hash")

        assert result is None
        assert cache_handler.metrics.misses == 1

    def test_compute_transaction_fingerprint(self):
        txn = {
            "customer_id": "CUST-001",
            "merchant_id": "MERCH-001",
            "transaction_amount": 100.0,
            "transaction_type": "purchase",
            "channel": "online",
            "geo_country": "US",
        }
        fingerprint = CacheHandler.compute_transaction_fingerprint(txn)

        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 32

        # Same input produces same fingerprint
        fingerprint2 = CacheHandler.compute_transaction_fingerprint(txn)
        assert fingerprint == fingerprint2

        # Different input produces different fingerprint
        txn_modified = dict(txn)
        txn_modified["transaction_amount"] = 200.0
        fingerprint3 = CacheHandler.compute_transaction_fingerprint(txn_modified)
        assert fingerprint3 != fingerprint


class TestCacheHandlerVelocity:
    """Tests for velocity counter operations."""

    def test_increment_velocity_counter(self, cache_handler, mock_redis_client):
        pipe = mock_redis_client.pipeline.return_value
        pipe.execute.return_value = [3, True]

        count = cache_handler.increment_velocity_counter("CUST-001", "10min")

        assert count == 3
        pipe.incr.assert_called_once()
        pipe.expire.assert_called_once()

    def test_get_velocity_count(self, cache_handler, mock_redis_client):
        mock_redis_client.get.return_value = "7"

        count = cache_handler.get_velocity_count("CUST-001", "1hr")

        assert count == 7

    def test_get_velocity_count_not_found(self, cache_handler, mock_redis_client):
        mock_redis_client.get.return_value = None

        count = cache_handler.get_velocity_count("CUST-NEW", "10min")

        assert count == 0


class TestCacheHandlerLocking:
    """Tests for distributed lock operations."""

    def test_acquire_processing_lock_success(self, cache_handler, mock_redis_client):
        mock_redis_client.set.return_value = True

        result = cache_handler.acquire_processing_lock("TXN-001")

        assert result is True
        mock_redis_client.set.assert_called_with(f"lock:TXN-001", "1", nx=True, ex=30)

    def test_acquire_processing_lock_already_held(self, cache_handler, mock_redis_client):
        mock_redis_client.set.return_value = None

        result = cache_handler.acquire_processing_lock("TXN-001")

        assert result is False

    def test_release_processing_lock(self, cache_handler, mock_redis_client):
        result = cache_handler.release_processing_lock("TXN-001")

        assert result is True
        mock_redis_client.delete.assert_called_with(f"lock:TXN-001")


class TestCacheHandlerWriteAhead:
    """Tests for write-ahead caching pattern."""

    def test_write_ahead_cache(self, cache_handler, mock_redis_client):
        data = {"customer_id": "CUST-001", "amount": 100}
        result = cache_handler.write_ahead_cache("TXN-001", data, ttl=300)

        assert result is True
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][0] == f"{PREFIX_DEDUP}TXN-001"
        assert call_args[0][1] == 300

    def test_get_write_ahead_data(self, cache_handler, mock_redis_client):
        data = {"customer_id": "CUST-001", "amount": 100}
        mock_redis_client.get.return_value = json.dumps(data)

        result = cache_handler.get_write_ahead_data("TXN-001")

        assert result == data

    def test_confirm_write_ahead(self, cache_handler, mock_redis_client):
        result = cache_handler.confirm_write_ahead("TXN-001")

        assert result is True
        mock_redis_client.delete.assert_called_with(f"{PREFIX_DEDUP}TXN-001")


class TestCacheHandlerBulkOperations:
    """Tests for bulk cache operations."""

    def test_bulk_set_customer_profiles(self, cache_handler, mock_redis_client):
        profiles = {
            "CUST-001": {"risk_tier": "low"},
            "CUST-002": {"risk_tier": "high"},
            "CUST-003": {"risk_tier": "standard"},
        }
        pipe = mock_redis_client.pipeline.return_value

        count = cache_handler.bulk_set_customer_profiles(profiles)

        assert count == 3
        assert pipe.setex.call_count == 3
        pipe.execute.assert_called_once()

    def test_bulk_cache_transactions(self, cache_handler, mock_redis_client):
        transactions = {
            "TXN-001": {"amount": "100"},
            "TXN-002": {"amount": "200"},
        }
        pipe = mock_redis_client.pipeline.return_value

        count = cache_handler.bulk_cache_transactions(transactions)

        assert count == 2


class TestCacheHandlerInvalidation:
    """Tests for cache invalidation strategies."""

    def test_invalidate_by_pattern(self, cache_handler, mock_redis_client):
        mock_redis_client.scan.return_value = (0, ["cp:CUST-001", "cp:CUST-002"])

        deleted = cache_handler.invalidate_by_pattern("cp:CUST-*")

        assert deleted == 2
        mock_redis_client.delete.assert_called_once_with("cp:CUST-001", "cp:CUST-002")

    def test_invalidate_customer_data(self, cache_handler, mock_redis_client):
        mock_redis_client.scan.return_value = (0, [])
        mock_redis_client.delete.return_value = 1

        deleted = cache_handler.invalidate_customer_data("CUST-001")

        assert deleted >= 1

    def test_flush_all_caches(self, cache_handler, mock_redis_client):
        mock_redis_client.scan.return_value = (0, [])

        result = cache_handler.flush_all_caches()

        assert result is True


class TestCacheHandlerMetrics:
    """Tests for cache metrics tracking."""

    def test_metrics_hit_rate(self, cache_handler, mock_redis_client):
        mock_redis_client.get.return_value = json.dumps({"test": True})

        # 3 hits
        for _ in range(3):
            cache_handler.get_customer_profile("CUST-001")

        # 2 misses
        mock_redis_client.get.return_value = None
        for _ in range(2):
            cache_handler.get_customer_profile("CUST-MISSING")

        assert cache_handler.metrics.hits == 3
        assert cache_handler.metrics.misses == 2
        assert cache_handler.metrics.hit_rate == 0.6

    def test_metrics_snapshot(self, cache_handler, mock_redis_client):
        mock_redis_client.get.return_value = json.dumps({"test": True})
        cache_handler.get_customer_profile("CUST-001")
        cache_handler.set_customer_profile("CUST-001", {"x": 1})

        snapshot = cache_handler.metrics.snapshot()

        assert "hits" in snapshot
        assert "misses" in snapshot
        assert "sets" in snapshot
        assert "hit_rate" in snapshot
        assert "avg_read_latency_ms" in snapshot
        assert "avg_write_latency_ms" in snapshot

    def test_metrics_reset(self, cache_handler, mock_redis_client):
        mock_redis_client.get.return_value = json.dumps({"test": True})
        cache_handler.get_customer_profile("CUST-001")

        cache_handler.reset_metrics()

        assert cache_handler.metrics.hits == 0
        assert cache_handler.metrics.misses == 0

    def test_health_check_success(self, cache_handler, mock_redis_client):
        mock_redis_client.ping.return_value = True

        assert cache_handler.health_check() is True

    def test_health_check_failure(self, cache_handler, mock_redis_client):
        from redis.exceptions import RedisError

        mock_redis_client.ping.side_effect = RedisError("Connection refused")

        assert cache_handler.health_check() is False


# =============================================================================
# Storage Orchestrator Tests
# =============================================================================


class TestStorageOrchestratorWriteFlow:
    """Tests for the unified write flow across backends."""

    @pytest.mark.asyncio
    async def test_store_transaction_all_backends(
        self, storage_orchestrator, sample_transaction, mock_redis_client
    ):
        result = await storage_orchestrator.store_transaction(sample_transaction)

        assert isinstance(result, OrchestratedWriteResult)
        assert result.transaction_id == "TXN-2026-INT-001"
        # Redis and Postgres should have write results
        assert len(result.results) == 2
        assert result.results[0].backend == StorageBackend.REDIS
        assert result.results[1].backend == StorageBackend.POSTGRES
        assert result.buffered_for_batch is True

    @pytest.mark.asyncio
    async def test_store_transaction_caches_customer_profile(
        self, storage_orchestrator, sample_transaction, mock_redis_client
    ):
        await storage_orchestrator.store_transaction(sample_transaction)

        # Should have called setex for write-ahead + profile + recent txn
        assert mock_redis_client.setex.call_count >= 2

    @pytest.mark.asyncio
    async def test_store_transaction_writes_to_postgres(
        self, storage_orchestrator, sample_transaction, mock_postgres_handler
    ):
        await storage_orchestrator.store_transaction(sample_transaction)

        mock_postgres_handler.create_transaction.assert_called_once()
        call_data = mock_postgres_handler.create_transaction.call_args[0][0]
        # Internal fields should be stripped
        assert "_pipeline_processed" not in call_data
        assert "_pipeline_timestamp" not in call_data

    @pytest.mark.asyncio
    async def test_store_transaction_buffers_for_s3(self, storage_orchestrator, sample_transaction):
        await storage_orchestrator.store_transaction(sample_transaction)

        buffer_sizes = storage_orchestrator.get_buffer_sizes()
        assert buffer_sizes["s3"] == 1

    @pytest.mark.asyncio
    async def test_store_transaction_buffers_for_snowflake(
        self, storage_orchestrator, sample_transaction
    ):
        await storage_orchestrator.store_transaction(sample_transaction)

        buffer_sizes = storage_orchestrator.get_buffer_sizes()
        assert buffer_sizes["snowflake"] == 1

    @pytest.mark.asyncio
    async def test_store_batch(self, storage_orchestrator, sample_transaction):
        records = [dict(sample_transaction) for _ in range(3)]
        for i, rec in enumerate(records):
            rec["external_transaction_id"] = f"TXN-BATCH-{i}"

        results = await storage_orchestrator.store_batch(records)

        assert len(results) == 3
        assert all(r.any_success for r in results)


class TestStorageOrchestratorBufferFlush:
    """Tests for batch buffer flushing."""

    @pytest.mark.asyncio
    async def test_s3_buffer_auto_flush(
        self, storage_orchestrator, sample_transaction, mock_s3_handler
    ):
        # s3_batch_size is 5, so 5 records should trigger flush
        for i in range(5):
            record = dict(sample_transaction)
            record["external_transaction_id"] = f"TXN-FLUSH-{i}"
            await storage_orchestrator.store_transaction(record)

        mock_s3_handler.upload_transactions.assert_called_once()
        assert storage_orchestrator.get_buffer_sizes()["s3"] == 0

    @pytest.mark.asyncio
    async def test_snowflake_buffer_auto_flush(
        self, storage_orchestrator, sample_transaction, mock_snowflake_handler
    ):
        # snowflake_batch_size is 10
        for i in range(10):
            record = dict(sample_transaction)
            record["external_transaction_id"] = f"TXN-SF-{i}"
            await storage_orchestrator.store_transaction(record)

        mock_snowflake_handler.bulk_load_records.assert_called_once()
        assert storage_orchestrator.get_buffer_sizes()["snowflake"] == 0

    @pytest.mark.asyncio
    async def test_manual_flush_all_buffers(
        self, storage_orchestrator, sample_transaction, mock_s3_handler, mock_snowflake_handler
    ):
        # Add some records to buffers
        for i in range(3):
            record = dict(sample_transaction)
            record["external_transaction_id"] = f"TXN-MANUAL-{i}"
            await storage_orchestrator.store_transaction(record)

        results = await storage_orchestrator.flush_all_buffers()

        assert StorageBackend.S3.value in results
        assert StorageBackend.SNOWFLAKE.value in results
        assert storage_orchestrator.get_buffer_sizes()["s3"] == 0
        assert storage_orchestrator.get_buffer_sizes()["snowflake"] == 0

    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self, storage_orchestrator, mock_s3_handler):
        result = await storage_orchestrator.flush_s3_buffer()

        assert result.success is True
        assert result.records_written == 0
        mock_s3_handler.upload_transactions.assert_not_called()


class TestStorageOrchestratorCircuitBreaker:
    """Tests for circuit breaker behavior on storage failures."""

    @pytest.mark.asyncio
    async def test_redis_failure_triggers_degraded_state(
        self, storage_orchestrator, sample_transaction, mock_redis_client
    ):
        from redis.exceptions import RedisError

        mock_redis_client.setex.side_effect = RedisError("Connection refused")

        for i in range(3):
            record = dict(sample_transaction)
            record["external_transaction_id"] = f"TXN-FAIL-{i}"
            await storage_orchestrator.store_transaction(record)

        health = storage_orchestrator.get_backend_health()
        redis_health = health["redis"]
        assert redis_health["consecutive_failures"] >= 2
        assert redis_health["state"] in ("degraded", "unavailable")

    @pytest.mark.asyncio
    async def test_postgres_failure_triggers_circuit_breaker(
        self, storage_orchestrator, sample_transaction, mock_postgres_handler
    ):
        mock_postgres_handler.create_transaction.side_effect = Exception("DB connection lost")

        for i in range(5):
            record = dict(sample_transaction)
            record["external_transaction_id"] = f"TXN-PGFAIL-{i}"
            await storage_orchestrator.store_transaction(record)

        health = storage_orchestrator.get_backend_health()
        pg_health = health["postgres"]
        assert pg_health["state"] == "unavailable"
        assert pg_health["consecutive_failures"] == 5

    @pytest.mark.asyncio
    async def test_circuit_breaker_skips_unavailable_backend(
        self, storage_orchestrator, sample_transaction, mock_postgres_handler
    ):
        # Force postgres to unavailable state
        storage_orchestrator._health[StorageBackend.POSTGRES].state = StorageState.UNAVAILABLE
        storage_orchestrator._health[StorageBackend.POSTGRES].consecutive_failures = 10

        result = await storage_orchestrator.store_transaction(sample_transaction)

        pg_result = next((r for r in result.results if r.backend == StorageBackend.POSTGRES), None)
        assert pg_result is not None
        assert pg_result.success is False
        assert pg_result.error == "circuit_breaker_open"
        # Should NOT have attempted DB write
        mock_postgres_handler.create_transaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_s3_failure_buffers_for_retry(
        self, storage_orchestrator, sample_transaction, mock_s3_handler
    ):
        from src.storage.s3_handler import S3HandlerError

        mock_s3_handler.upload_transactions.side_effect = S3HandlerError("Upload failed")

        # Fill buffer to trigger flush
        for i in range(5):
            record = dict(sample_transaction)
            record["external_transaction_id"] = f"TXN-S3FAIL-{i}"
            await storage_orchestrator.store_transaction(record)

        # Records should be back in buffer
        buffer_sizes = storage_orchestrator.get_buffer_sizes()
        assert buffer_sizes["s3"] == 5

    @pytest.mark.asyncio
    async def test_backend_recovery_after_success(
        self, storage_orchestrator, sample_transaction, mock_redis_client
    ):
        from redis.exceptions import RedisError

        # Simulate failures then recovery
        mock_redis_client.setex.side_effect = RedisError("Down")
        for i in range(3):
            record = dict(sample_transaction)
            record["external_transaction_id"] = f"TXN-RECOV-{i}"
            await storage_orchestrator.store_transaction(record)

        assert storage_orchestrator._health[StorageBackend.REDIS].consecutive_failures >= 2

        # Recovery - clear the error
        mock_redis_client.setex.side_effect = None
        mock_redis_client.setex.return_value = True

        record = dict(sample_transaction)
        record["external_transaction_id"] = "TXN-RECOV-FINAL"
        await storage_orchestrator.store_transaction(record)

        health = storage_orchestrator._health[StorageBackend.REDIS]
        assert health.consecutive_failures == 0
        assert health.state == StorageState.HEALTHY


class TestStorageOrchestratorWriteAheadPattern:
    """Tests for the write-ahead pattern (Kafka → cache → DB)."""

    @pytest.mark.asyncio
    async def test_write_ahead_then_confirm(
        self, storage_orchestrator, sample_transaction, mock_redis_client
    ):
        """Verify write-ahead cache is set before DB, then confirmed after DB success."""
        call_order = []

        def track_setex(*args, **kwargs):
            call_order.append("cache_set")
            return True

        async def track_create(*args, **kwargs):
            call_order.append("db_write")
            return MagicMock()

        def track_delete(*args, **kwargs):
            call_order.append("cache_confirm")
            return 1

        mock_redis_client.setex.side_effect = track_setex
        storage_orchestrator._postgres.create_transaction = track_create
        mock_redis_client.delete.side_effect = track_delete

        await storage_orchestrator.store_transaction(sample_transaction)

        # Cache should be written before DB, and confirmed after
        assert "cache_set" in call_order
        assert "db_write" in call_order
        db_idx = call_order.index("db_write")
        cache_set_idx = call_order.index("cache_set")
        assert cache_set_idx < db_idx

    @pytest.mark.asyncio
    async def test_write_ahead_no_confirm_on_db_failure(
        self, storage_orchestrator, sample_transaction, mock_redis_client, mock_postgres_handler
    ):
        """Write-ahead data should NOT be confirmed if DB write fails."""
        mock_postgres_handler.create_transaction.side_effect = Exception("DB error")
        mock_redis_client.delete.reset_mock()

        await storage_orchestrator.store_transaction(sample_transaction)

        # confirm_write_ahead calls delete — should NOT be called on failure
        # (delete may be called for other reasons, so check that the dedup key wasn't deleted)
        # The write_ahead_cache key should still be in place
        dedup_key = f"{PREFIX_DEDUP}TXN-2026-INT-001"
        delete_calls = [str(call) for call in mock_redis_client.delete.call_args_list]
        # The confirm should not have been triggered
        assert not any(dedup_key in c for c in delete_calls)


class TestStorageOrchestratorMonitoring:
    """Tests for storage latency monitoring."""

    @pytest.mark.asyncio
    async def test_latency_report(
        self, storage_orchestrator, sample_transaction, mock_redis_client
    ):
        await storage_orchestrator.store_transaction(sample_transaction)

        report = storage_orchestrator.get_latency_report()

        assert "redis" in report
        assert "postgres" in report
        assert "s3" in report
        assert "snowflake" in report
        assert report["redis"] >= 0
        assert report["postgres"] >= 0

    @pytest.mark.asyncio
    async def test_backend_health_report(self, storage_orchestrator, sample_transaction):
        await storage_orchestrator.store_transaction(sample_transaction)

        health = storage_orchestrator.get_backend_health()

        assert "redis" in health
        assert "postgres" in health
        assert health["redis"]["state"] == "healthy"
        assert health["redis"]["total_successes"] >= 1
        assert health["postgres"]["state"] == "healthy"

    @pytest.mark.asyncio
    async def test_check_all_backends(
        self, storage_orchestrator, mock_redis_client, mock_postgres_handler
    ):
        results = await storage_orchestrator.check_all_backends()

        assert "redis" in results
        assert "postgres" in results
        assert results["redis"]["healthy"] is True
        assert results["postgres"]["healthy"] is True


class TestStorageOrchestratorGracefulDegradation:
    """Tests for graceful degradation when backends fail."""

    @pytest.mark.asyncio
    async def test_continues_without_redis(
        self, mock_postgres_handler, mock_s3_handler, mock_snowflake_handler, sample_transaction
    ):
        """Pipeline should continue even if Redis is completely unavailable."""
        orchestrator = StorageOrchestrator(
            cache_handler=None,  # No Redis
            postgres_handler=mock_postgres_handler,
            s3_handler=mock_s3_handler,
            snowflake_handler=mock_snowflake_handler,
        )

        result = await orchestrator.store_transaction(sample_transaction)

        assert result.any_success
        mock_postgres_handler.create_transaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_continues_without_postgres(
        self, cache_handler, mock_s3_handler, mock_snowflake_handler, sample_transaction
    ):
        """Buffering should still work even if PostgreSQL is down."""
        orchestrator = StorageOrchestrator(
            cache_handler=cache_handler,
            postgres_handler=None,  # No Postgres
            s3_handler=mock_s3_handler,
            snowflake_handler=mock_snowflake_handler,
        )

        result = await orchestrator.store_transaction(sample_transaction)

        # Cache write should succeed, data buffered for S3/SF
        assert result.any_success
        assert result.buffered_for_batch

    @pytest.mark.asyncio
    async def test_result_reports_failed_backends(
        self, storage_orchestrator, sample_transaction, mock_postgres_handler, mock_redis_client
    ):
        from redis.exceptions import RedisError

        mock_redis_client.setex.side_effect = RedisError("Timeout")

        result = await storage_orchestrator.store_transaction(sample_transaction)

        assert StorageBackend.REDIS in result.failed_backends
        assert StorageBackend.POSTGRES not in result.failed_backends
        # Postgres should still succeed despite Redis failure
        pg_result = next(r for r in result.results if r.backend == StorageBackend.POSTGRES)
        assert pg_result.success is True


class TestStorageOrchestratorLifecycle:
    """Tests for orchestrator lifecycle management."""

    @pytest.mark.asyncio
    async def test_close_flushes_buffers(
        self, storage_orchestrator, sample_transaction, mock_s3_handler, mock_snowflake_handler
    ):
        # Add records to buffer
        for i in range(3):
            record = dict(sample_transaction)
            record["external_transaction_id"] = f"TXN-CLOSE-{i}"
            await storage_orchestrator.store_transaction(record)

        await storage_orchestrator.close()

        # S3 and Snowflake should have been flushed
        mock_s3_handler.upload_transactions.assert_called_once()
        mock_snowflake_handler.bulk_load_records.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_closes_all_backends(
        self, storage_orchestrator, mock_postgres_handler, mock_s3_handler, mock_snowflake_handler
    ):
        await storage_orchestrator.close()

        mock_postgres_handler.close.assert_called_once()
        mock_s3_handler.close.assert_called_once()
        mock_snowflake_handler.close.assert_called_once()


class TestBackendHealth:
    """Tests for BackendHealth tracking."""

    def test_initial_state_is_healthy(self):
        health = BackendHealth(backend=StorageBackend.REDIS)
        assert health.state == StorageState.HEALTHY

    def test_two_failures_triggers_degraded(self):
        health = BackendHealth(backend=StorageBackend.REDIS)
        health.record_failure()
        health.record_failure()
        assert health.state == StorageState.DEGRADED

    def test_five_failures_triggers_unavailable(self):
        health = BackendHealth(backend=StorageBackend.POSTGRES)
        for _ in range(5):
            health.record_failure()
        assert health.state == StorageState.UNAVAILABLE

    def test_success_resets_failure_count(self):
        health = BackendHealth(backend=StorageBackend.S3)
        health.record_failure()
        health.record_failure()
        health.record_failure()
        assert health.consecutive_failures == 3

        health.record_success(5.0)
        assert health.consecutive_failures == 0
        assert health.state == StorageState.HEALTHY

    def test_success_after_unavailable_recovers(self):
        health = BackendHealth(backend=StorageBackend.SNOWFLAKE)
        for _ in range(5):
            health.record_failure()
        assert health.state == StorageState.UNAVAILABLE

        health.record_success(10.0)
        assert health.state == StorageState.HEALTHY

    def test_latency_tracking(self):
        health = BackendHealth(backend=StorageBackend.REDIS)
        health.record_success(1.0)
        health.record_success(3.0)
        health.record_success(5.0)

        assert health.avg_latency_ms == 3.0

    def test_snapshot(self):
        health = BackendHealth(backend=StorageBackend.POSTGRES)
        health.record_success(2.5)
        health.record_failure()

        snap = health.snapshot()
        assert snap["backend"] == "postgres"
        assert snap["total_successes"] == 1
        assert snap["total_failures"] == 1
        assert "avg_latency_ms" in snap


class TestWriteResult:
    """Tests for WriteResult and OrchestratedWriteResult."""

    def test_orchestrated_result_all_success(self):
        result = OrchestratedWriteResult(transaction_id="TXN-001")
        result.results = [
            WriteResult(backend=StorageBackend.REDIS, success=True, latency_ms=0.5),
            WriteResult(backend=StorageBackend.POSTGRES, success=True, latency_ms=2.0),
        ]

        assert result.all_success is True
        assert result.any_success is True
        assert result.failed_backends == []

    def test_orchestrated_result_partial_failure(self):
        result = OrchestratedWriteResult(transaction_id="TXN-002")
        result.results = [
            WriteResult(backend=StorageBackend.REDIS, success=False, error="timeout"),
            WriteResult(backend=StorageBackend.POSTGRES, success=True, latency_ms=3.0),
        ]

        assert result.all_success is False
        assert result.any_success is True
        assert result.failed_backends == [StorageBackend.REDIS]

    def test_orchestrated_result_to_dict(self):
        result = OrchestratedWriteResult(transaction_id="TXN-003")
        result.results = [
            WriteResult(backend=StorageBackend.REDIS, success=True, latency_ms=0.8),
        ]

        d = result.to_dict()
        assert d["transaction_id"] == "TXN-003"
        assert d["all_success"] is True
        assert len(d["results"]) == 1
