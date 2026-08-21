"""Unified storage orchestrator for RiskPulse data flow.

Coordinates writes across all storage backends:
- Redis (hot data, real-time lookups)
- PostgreSQL (operational, transactional)
- S3 (raw/processed data lake)
- Snowflake (analytical warehouse, batch)

Implements:
- Write-ahead pattern (Kafka → cache → DB)
- Circuit breaker for storage failures
- Storage latency monitoring
- Graceful degradation on backend failures
- Buffered batch writes for efficiency
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any

import structlog

from src.storage.cache_handler import CacheHandler, CacheHandlerError
from src.storage.postgres_handler import PostgresHandler, PostgresHandlerError
from src.storage.s3_handler import S3Handler, S3HandlerError
from src.storage.snowflake_handler import SnowflakeHandler, SnowflakeHandlerError

logger = structlog.get_logger(__name__)


class StorageBackend(str, Enum):
    """Enumeration of storage backends."""

    REDIS = "redis"
    POSTGRES = "postgres"
    S3 = "s3"
    SNOWFLAKE = "snowflake"


class StorageState(str, Enum):
    """State of a storage backend."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class BackendHealth:
    """Health status of a single storage backend."""

    backend: StorageBackend
    state: StorageState = StorageState.HEALTHY
    last_success: float = 0.0
    last_failure: float = 0.0
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    avg_latency_ms: float = 0.0
    _latency_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record_success(self, latency_ms: float) -> None:
        with self._lock:
            self.last_success = time.time()
            self.consecutive_failures = 0
            self.total_successes += 1
            self._latency_samples.append(latency_ms)
            self.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)
            if self.state != StorageState.HEALTHY:
                self.state = StorageState.HEALTHY
                logger.info("storage_backend_recovered", backend=self.backend.value)

    def record_failure(self) -> None:
        with self._lock:
            self.last_failure = time.time()
            self.consecutive_failures += 1
            self.total_failures += 1
            if self.consecutive_failures >= 5:
                self.state = StorageState.UNAVAILABLE
            elif self.consecutive_failures >= 2:
                self.state = StorageState.DEGRADED

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": self.backend.value,
                "state": self.state.value,
                "consecutive_failures": self.consecutive_failures,
                "total_failures": self.total_failures,
                "total_successes": self.total_successes,
                "avg_latency_ms": round(self.avg_latency_ms, 4),
                "last_success": self.last_success,
                "last_failure": self.last_failure,
            }


@dataclass
class WriteResult:
    """Result of writing to a storage backend."""

    backend: StorageBackend
    success: bool
    latency_ms: float = 0.0
    error: str | None = None
    records_written: int = 0


@dataclass
class OrchestratedWriteResult:
    """Aggregate result of writing to all configured backends."""

    transaction_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    results: list[WriteResult] = field(default_factory=list)
    buffered_for_batch: bool = False

    @property
    def all_success(self) -> bool:
        return all(r.success for r in self.results)

    @property
    def any_success(self) -> bool:
        return any(r.success for r in self.results)

    @property
    def failed_backends(self) -> list[StorageBackend]:
        return [r.backend for r in self.results if not r.success]

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp.isoformat(),
            "all_success": self.all_success,
            "failed_backends": [b.value for b in self.failed_backends],
            "results": [
                {
                    "backend": r.backend.value,
                    "success": r.success,
                    "latency_ms": round(r.latency_ms, 4),
                    "error": r.error,
                }
                for r in self.results
            ],
        }


# Circuit breaker configuration
_CB_FAILURE_THRESHOLD = 5
_CB_RECOVERY_TIMEOUT = 30
_CB_EXPECTED_EXCEPTION = Exception


class StorageOrchestrator:
    """Unified storage orchestrator coordinating writes across all backends.

    Implements the write-ahead pattern:
        1. Write to Redis (immediate availability for real-time queries)
        2. Write to PostgreSQL (durable operational store)
        3. Buffer for S3 (batch upload as Parquet)
        4. Buffer for Snowflake (batch load via stage)

    Circuit breakers protect against cascading failures when any
    individual backend becomes unavailable.

    Usage:
        orchestrator = StorageOrchestrator(
            cache_handler=cache,
            postgres_handler=postgres,
            s3_handler=s3,
            snowflake_handler=snowflake,
        )
        result = await orchestrator.store_transaction(enriched_record)
        await orchestrator.flush_batch_buffers()
        orchestrator.close()
    """

    def __init__(
        self,
        cache_handler: CacheHandler | None = None,
        postgres_handler: PostgresHandler | None = None,
        s3_handler: S3Handler | None = None,
        snowflake_handler: SnowflakeHandler | None = None,
        s3_batch_size: int = 500,
        snowflake_batch_size: int = 1000,
        enable_write_ahead: bool = True,
        circuit_breaker_threshold: int = _CB_FAILURE_THRESHOLD,
        circuit_breaker_timeout: int = _CB_RECOVERY_TIMEOUT,
    ) -> None:
        self._cache = cache_handler
        self._postgres = postgres_handler
        self._s3 = s3_handler
        self._snowflake = snowflake_handler

        self._s3_batch_size = s3_batch_size
        self._snowflake_batch_size = snowflake_batch_size
        self._enable_write_ahead = enable_write_ahead

        # Batch buffers
        self._s3_buffer: list[dict[str, Any]] = []
        self._snowflake_buffer: list[dict[str, Any]] = []
        self._buffer_lock = Lock()

        # Backend health tracking
        self._health: dict[StorageBackend, BackendHealth] = {
            StorageBackend.REDIS: BackendHealth(backend=StorageBackend.REDIS),
            StorageBackend.POSTGRES: BackendHealth(backend=StorageBackend.POSTGRES),
            StorageBackend.S3: BackendHealth(backend=StorageBackend.S3),
            StorageBackend.SNOWFLAKE: BackendHealth(backend=StorageBackend.SNOWFLAKE),
        }

        # Circuit breaker settings
        self._cb_threshold = circuit_breaker_threshold
        self._cb_timeout = circuit_breaker_timeout

        logger.info(
            "storage_orchestrator_initialized",
            backends_configured={
                "redis": cache_handler is not None,
                "postgres": postgres_handler is not None,
                "s3": s3_handler is not None,
                "snowflake": snowflake_handler is not None,
            },
            s3_batch_size=s3_batch_size,
            snowflake_batch_size=snowflake_batch_size,
            write_ahead=enable_write_ahead,
        )

    # -------------------------------------------------------------------------
    # Main Write Path
    # -------------------------------------------------------------------------

    async def store_transaction(
        self,
        record: dict[str, Any],
    ) -> OrchestratedWriteResult:
        """Store a processed transaction across all configured backends.

        Write order (write-ahead pattern):
            1. Redis cache (immediate, non-blocking)
            2. PostgreSQL (durable, operational)
            3. S3 buffer (batch upload when threshold reached)
            4. Snowflake buffer (batch load when threshold reached)

        Args:
            record: Fully processed/enriched transaction record.

        Returns:
            OrchestratedWriteResult with per-backend outcomes.
        """
        transaction_id = record.get(
            "external_transaction_id",
            record.get("transaction_id", str(uuid.uuid4())),
        )
        result = OrchestratedWriteResult(transaction_id=transaction_id)

        # Step 1: Write-ahead cache (Redis)
        if self._cache and self._enable_write_ahead:
            cache_result = await self._write_to_cache(transaction_id, record)
            result.results.append(cache_result)

        # Step 2: PostgreSQL (operational store)
        if self._postgres:
            pg_result = await self._write_to_postgres(transaction_id, record)
            result.results.append(pg_result)

            # Confirm write-ahead on successful DB write
            if pg_result.success and self._cache and self._enable_write_ahead:
                self._cache.confirm_write_ahead(transaction_id)

        # Step 3: Buffer for S3 batch write
        if self._s3:
            self._buffer_for_s3(record)
            result.buffered_for_batch = True

        # Step 4: Buffer for Snowflake batch load
        if self._snowflake:
            self._buffer_for_snowflake(record)

        # Auto-flush if buffers are full
        await self._check_and_flush_buffers()

        if not result.all_success and result.any_success:
            logger.warning(
                "partial_storage_write",
                transaction_id=transaction_id,
                failed_backends=[b.value for b in result.failed_backends],
            )
        elif not result.any_success and result.results:
            logger.error(
                "complete_storage_failure",
                transaction_id=transaction_id,
            )

        return result

    async def store_batch(
        self,
        records: list[dict[str, Any]],
    ) -> list[OrchestratedWriteResult]:
        """Store a batch of processed transactions.

        Args:
            records: List of processed transaction records.

        Returns:
            List of per-record orchestration results.
        """
        results = []
        for record in records:
            result = await self.store_transaction(record)
            results.append(result)
        return results

    # -------------------------------------------------------------------------
    # Backend Write Implementations
    # -------------------------------------------------------------------------

    async def _write_to_cache(self, transaction_id: str, record: dict[str, Any]) -> WriteResult:
        """Write transaction to Redis cache with circuit breaker."""
        start = time.perf_counter()
        health = self._health[StorageBackend.REDIS]
        if self._cache is None:
            return WriteResult(
                backend=StorageBackend.REDIS,
                success=False,
                error="cache_backend_not_configured",
            )

        if health.state == StorageState.UNAVAILABLE:
            return WriteResult(
                backend=StorageBackend.REDIS,
                success=False,
                error="circuit_breaker_open",
            )

        try:
            # Write-ahead cache
            wa_ok = self._cache.write_ahead_cache(transaction_id, record)

            # Cache customer profile for velocity lookups
            profile_ok = True
            customer_id = record.get("customer_id")
            if customer_id:
                profile = self._extract_customer_profile(record)
                profile_ok = self._cache.set_customer_profile(customer_id, profile)

            # Cache recent transaction for dedup
            txn_ok = self._cache.cache_recent_transaction(
                transaction_id,
                {
                    "customer_id": record.get("customer_id"),
                    "amount": str(record.get("transaction_amount")),
                    "timestamp": record.get("transaction_timestamp"),
                    "type": record.get("transaction_type"),
                },
            )

            # If any cache operation failed, treat as failure
            if not (wa_ok and profile_ok and txn_ok):
                latency = (time.perf_counter() - start) * 1000
                health.record_failure()
                return WriteResult(
                    backend=StorageBackend.REDIS,
                    success=False,
                    latency_ms=latency,
                    error="one_or_more_cache_operations_failed",
                )

            latency = (time.perf_counter() - start) * 1000
            health.record_success(latency)
            return WriteResult(
                backend=StorageBackend.REDIS,
                success=True,
                latency_ms=latency,
                records_written=1,
            )
        except (CacheHandlerError, Exception) as e:
            latency = (time.perf_counter() - start) * 1000
            health.record_failure()
            logger.error(
                "cache_write_failed",
                transaction_id=transaction_id,
                error=str(e),
            )
            return WriteResult(
                backend=StorageBackend.REDIS,
                success=False,
                latency_ms=latency,
                error=str(e),
            )

    async def _write_to_postgres(self, transaction_id: str, record: dict[str, Any]) -> WriteResult:
        """Write transaction to PostgreSQL with circuit breaker."""
        start = time.perf_counter()
        health = self._health[StorageBackend.POSTGRES]

        if health.state == StorageState.UNAVAILABLE:
            return WriteResult(
                backend=StorageBackend.POSTGRES,
                success=False,
                error="circuit_breaker_open",
            )

        try:
            pg_data = self._prepare_postgres_record(record)
            if self._postgres is None:
                return WriteResult(
                    backend=StorageBackend.POSTGRES,
                    success=False,
                    error="postgres_backend_not_configured",
                )
            await self._postgres.create_transaction(pg_data)

            latency = (time.perf_counter() - start) * 1000
            health.record_success(latency)
            return WriteResult(
                backend=StorageBackend.POSTGRES,
                success=True,
                latency_ms=latency,
                records_written=1,
            )
        except (PostgresHandlerError, Exception) as e:
            latency = (time.perf_counter() - start) * 1000
            health.record_failure()
            logger.error(
                "postgres_write_failed",
                transaction_id=transaction_id,
                error=str(e),
            )
            return WriteResult(
                backend=StorageBackend.POSTGRES,
                success=False,
                latency_ms=latency,
                error=str(e),
            )

    def _buffer_for_s3(self, record: dict[str, Any]) -> None:
        """Add record to S3 batch buffer."""
        with self._buffer_lock:
            self._s3_buffer.append(record)

    def _buffer_for_snowflake(self, record: dict[str, Any]) -> None:
        """Add record to Snowflake batch buffer."""
        with self._buffer_lock:
            self._snowflake_buffer.append(record)

    async def _check_and_flush_buffers(self) -> None:
        """Flush batch buffers if they exceed configured thresholds."""
        s3_flush = False
        sf_flush = False

        with self._buffer_lock:
            if len(self._s3_buffer) >= self._s3_batch_size:
                s3_flush = True
            if len(self._snowflake_buffer) >= self._snowflake_batch_size:
                sf_flush = True

        if s3_flush:
            await self.flush_s3_buffer()
        if sf_flush:
            await self.flush_snowflake_buffer()

    async def flush_s3_buffer(self) -> WriteResult:
        """Flush buffered records to S3 as a Parquet batch.

        Returns:
            WriteResult with batch write outcome.
        """
        with self._buffer_lock:
            if not self._s3_buffer:
                return WriteResult(backend=StorageBackend.S3, success=True, records_written=0)
            batch = list(self._s3_buffer)
            self._s3_buffer.clear()

        start = time.perf_counter()
        health = self._health[StorageBackend.S3]

        if health.state == StorageState.UNAVAILABLE:
            # Put records back in buffer for retry
            with self._buffer_lock:
                self._s3_buffer = batch + self._s3_buffer
            return WriteResult(
                backend=StorageBackend.S3,
                success=False,
                error="circuit_breaker_open",
            )

        try:
            if self._s3 is None:
                with self._buffer_lock:
                    self._s3_buffer = batch + self._s3_buffer
                return WriteResult(
                    backend=StorageBackend.S3,
                    success=False,
                    error="s3_backend_not_configured",
                )
            self._s3.upload_transactions(batch)
            latency = (time.perf_counter() - start) * 1000
            health.record_success(latency)
            logger.info("s3_batch_flushed", records=len(batch), latency_ms=round(latency, 2))
            return WriteResult(
                backend=StorageBackend.S3,
                success=True,
                latency_ms=latency,
                records_written=len(batch),
            )
        except (S3HandlerError, Exception) as e:
            latency = (time.perf_counter() - start) * 1000
            health.record_failure()
            # Put records back for retry
            with self._buffer_lock:
                self._s3_buffer = batch + self._s3_buffer
            logger.error("s3_batch_flush_failed", records=len(batch), error=str(e))
            return WriteResult(
                backend=StorageBackend.S3,
                success=False,
                latency_ms=latency,
                error=str(e),
            )

    async def flush_snowflake_buffer(self) -> WriteResult:
        """Flush buffered records to Snowflake via stage loading.

        Returns:
            WriteResult with batch load outcome.
        """
        with self._buffer_lock:
            if not self._snowflake_buffer:
                return WriteResult(
                    backend=StorageBackend.SNOWFLAKE, success=True, records_written=0
                )
            batch = list(self._snowflake_buffer)
            self._snowflake_buffer.clear()

        start = time.perf_counter()
        health = self._health[StorageBackend.SNOWFLAKE]

        if health.state == StorageState.UNAVAILABLE:
            with self._buffer_lock:
                self._snowflake_buffer = batch + self._snowflake_buffer
            return WriteResult(
                backend=StorageBackend.SNOWFLAKE,
                success=False,
                error="circuit_breaker_open",
            )

        try:
            if self._snowflake is None:
                with self._buffer_lock:
                    self._snowflake_buffer = batch + self._snowflake_buffer
                return WriteResult(
                    backend=StorageBackend.SNOWFLAKE,
                    success=False,
                    error="snowflake_backend_not_configured",
                )
            self._snowflake.bulk_load_records(
                records=batch,
                table_name="TRANSACTIONS",
                schema="RAW",
            )
            latency = (time.perf_counter() - start) * 1000
            health.record_success(latency)
            logger.info(
                "snowflake_batch_flushed",
                records=len(batch),
                latency_ms=round(latency, 2),
            )
            return WriteResult(
                backend=StorageBackend.SNOWFLAKE,
                success=True,
                latency_ms=latency,
                records_written=len(batch),
            )
        except (SnowflakeHandlerError, Exception) as e:
            latency = (time.perf_counter() - start) * 1000
            health.record_failure()
            with self._buffer_lock:
                self._snowflake_buffer = batch + self._snowflake_buffer
            logger.error("snowflake_batch_flush_failed", records=len(batch), error=str(e))
            return WriteResult(
                backend=StorageBackend.SNOWFLAKE,
                success=False,
                latency_ms=latency,
                error=str(e),
            )

    async def flush_all_buffers(self) -> dict[str, WriteResult]:
        """Flush all pending batch buffers.

        Returns:
            Dict mapping backend name to write result.
        """
        s3_result = await self.flush_s3_buffer()
        sf_result = await self.flush_snowflake_buffer()
        return {
            StorageBackend.S3.value: s3_result,
            StorageBackend.SNOWFLAKE.value: sf_result,
        }

    # -------------------------------------------------------------------------
    # Health & Monitoring
    # -------------------------------------------------------------------------

    async def check_all_backends(self) -> dict[str, dict[str, Any]]:
        """Check health of all configured storage backends.

        Returns:
            Dict mapping backend name to health status.
        """
        results = {}

        if self._cache:
            healthy = self._cache.health_check()
            self._health[StorageBackend.REDIS].state = (
                StorageState.HEALTHY if healthy else StorageState.UNAVAILABLE
            )
            results["redis"] = {"healthy": healthy}

        if self._postgres:
            healthy = await self._postgres.health_check()
            self._health[StorageBackend.POSTGRES].state = (
                StorageState.HEALTHY if healthy else StorageState.UNAVAILABLE
            )
            results["postgres"] = {"healthy": healthy}

        return results

    def get_backend_health(self) -> dict[str, dict[str, Any]]:
        """Get health status for all backends."""
        return {k.value: v.snapshot() for k, v in self._health.items()}

    def get_latency_report(self) -> dict[str, float]:
        """Get average latency for each backend."""
        return {k.value: round(v.avg_latency_ms, 4) for k, v in self._health.items()}

    def get_buffer_sizes(self) -> dict[str, int]:
        """Get current batch buffer sizes."""
        with self._buffer_lock:
            return {
                "s3": len(self._s3_buffer),
                "snowflake": len(self._snowflake_buffer),
            }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_customer_profile(record: dict[str, Any]) -> dict[str, Any]:
        """Extract customer profile data from a transaction record for caching."""
        return {
            "customer_id": record.get("customer_id"),
            "account_id": record.get("account_id"),
            "last_transaction_amount": str(record.get("transaction_amount")),
            "last_transaction_timestamp": record.get("transaction_timestamp"),
            "last_channel": record.get("channel"),
            "last_geo_country": record.get("geo_country"),
            "last_geo_city": record.get("geo_city"),
            "last_device_type": record.get("device_type"),
            "is_international": record.get("is_international"),
        }

    @staticmethod
    def _prepare_postgres_record(record: dict[str, Any]) -> dict[str, Any]:
        """Prepare a record for PostgreSQL insertion.

        Strips internal pipeline metadata fields and maps to DB columns.
        """
        # Fields that should not be persisted to the transactions table
        internal_prefixes = ("_pipeline_", "_validation_", "_rules_", "_enrichment_")
        pg_record = {}

        for key, value in record.items():
            if any(key.startswith(prefix) for prefix in internal_prefixes):
                continue
            pg_record[key] = value

        return pg_record

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def close(self) -> None:
        """Flush buffers and close all backend connections."""
        logger.info("storage_orchestrator_shutting_down")

        # Flush any remaining buffered data
        await self.flush_all_buffers()

        if self._cache:
            self._cache.close()

        if self._postgres:
            await self._postgres.close()

        if self._s3:
            self._s3.close()

        if self._snowflake:
            self._snowflake.close()

        logger.info("storage_orchestrator_closed")


def create_storage_orchestrator(
    cache_handler: CacheHandler | None = None,
    postgres_handler: PostgresHandler | None = None,
    s3_handler: S3Handler | None = None,
    snowflake_handler: SnowflakeHandler | None = None,
    **kwargs: Any,
) -> StorageOrchestrator:
    """Factory function to create a configured StorageOrchestrator.

    Args:
        cache_handler: Optional Redis cache handler.
        postgres_handler: Optional PostgreSQL handler.
        s3_handler: Optional S3 handler.
        snowflake_handler: Optional Snowflake handler.
        **kwargs: Additional configuration overrides.

    Returns:
        Configured StorageOrchestrator instance.
    """
    return StorageOrchestrator(
        cache_handler=cache_handler,
        postgres_handler=postgres_handler,
        s3_handler=s3_handler,
        snowflake_handler=snowflake_handler,
        **kwargs,
    )
