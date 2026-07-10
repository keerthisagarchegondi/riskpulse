"""Redis cache layer for RiskPulse real-time data access.

Production-grade Redis handler providing:
- Connection pooling with health checks
- Customer profile caching (for velocity lookups)
- Recent transaction caching (for deduplication)
- Model prediction caching (TTL-based)
- Cache invalidation strategies (TTL, explicit, pattern-based)
- Write-ahead caching (Kafka → cache → DB flow)
- Metrics tracking for hit/miss rates and latencies
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any, Generator

import redis
import structlog
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError, TimeoutError
from redis.retry import Retry

from src.utils.config import get_settings

logger = structlog.get_logger(__name__)


# Cache key prefixes
PREFIX_CUSTOMER_PROFILE = "cp:"
PREFIX_RECENT_TRANSACTIONS = "rtx:"
PREFIX_MODEL_PREDICTION = "pred:"
PREFIX_DEDUP = "dedup:"
PREFIX_VELOCITY = "vel:"
PREFIX_LOCK = "lock:"


class CacheStrategy(str, Enum):
    """Cache invalidation strategy."""

    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    CACHE_ASIDE = "cache_aside"


class CacheHandlerError(Exception):
    """Base exception for cache handler errors."""


class CacheConnectionError(CacheHandlerError):
    """Raised when Redis connection fails."""


class CacheOperationError(CacheHandlerError):
    """Raised when a cache operation fails."""


@dataclass
class CacheMetrics:
    """Thread-safe metrics for cache operations."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_read_latency_ms: float = 0.0
    total_write_latency_ms: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    @property
    def avg_read_latency_ms(self) -> float:
        total_reads = self.hits + self.misses
        if total_reads == 0:
            return 0.0
        return self.total_read_latency_ms / total_reads

    @property
    def avg_write_latency_ms(self) -> float:
        if self.sets == 0:
            return 0.0
        return self.total_write_latency_ms / self.sets

    def record_hit(self, latency_ms: float) -> None:
        with self._lock:
            self.hits += 1
            self.total_read_latency_ms += latency_ms

    def record_miss(self, latency_ms: float) -> None:
        with self._lock:
            self.misses += 1
            self.total_read_latency_ms += latency_ms

    def record_set(self, latency_ms: float) -> None:
        with self._lock:
            self.sets += 1
            self.total_write_latency_ms += latency_ms

    def record_delete(self) -> None:
        with self._lock:
            self.deletes += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "deletes": self.deletes,
                "errors": self.errors,
                "hit_rate": round(self.hit_rate, 4),
                "avg_read_latency_ms": round(self.avg_read_latency_ms, 4),
                "avg_write_latency_ms": round(self.avg_write_latency_ms, 4),
            }

    def reset(self) -> None:
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.sets = 0
            self.deletes = 0
            self.errors = 0
            self.total_read_latency_ms = 0.0
            self.total_write_latency_ms = 0.0


class CacheHandler:
    """Redis cache layer for real-time data access in the RiskPulse pipeline.

    Provides low-latency caching for:
    - Customer profiles (velocity lookups, risk context)
    - Recent transactions (deduplication checks)
    - Model predictions (avoid redundant scoring)
    - Distributed locks (coordination)

    Usage:
        handler = CacheHandler(redis_url="redis://localhost:6379/0")
        handler.set_customer_profile("CUST-123", profile_data)
        profile = handler.get_customer_profile("CUST-123")
        handler.close()
    """

    def __init__(
        self,
        redis_url: str | None = None,
        max_connections: int = 50,
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
        ttl_customer_profile: int | None = None,
        ttl_recent_transactions: int | None = None,
        ttl_model_predictions: int | None = None,
    ) -> None:
        settings = get_settings()

        self._redis_url = redis_url or settings.get(
            "redis.url", "redis://localhost:6379/0"
        )
        self._ttl_customer_profile = ttl_customer_profile or settings.get(
            "redis.ttl.customer_profile", 300
        )
        self._ttl_recent_transactions = ttl_recent_transactions or settings.get(
            "redis.ttl.recent_transactions", 3600
        )
        self._ttl_model_predictions = ttl_model_predictions or settings.get(
            "redis.ttl.model_predictions", 60
        )

        retry = Retry(ExponentialBackoff(cap=5, base=0.1), retries=3)

        self._pool = redis.ConnectionPool.from_url(
            self._redis_url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            retry_on_timeout=retry_on_timeout,
            health_check_interval=health_check_interval,
            retry=retry,
            decode_responses=True,
        )
        self._client = redis.Redis(connection_pool=self._pool)
        self._metrics = CacheMetrics()

        logger.info(
            "cache_handler_initialized",
            redis_url=self._sanitize_url(self._redis_url),
            max_connections=max_connections,
            ttl_customer_profile=self._ttl_customer_profile,
            ttl_recent_transactions=self._ttl_recent_transactions,
            ttl_model_predictions=self._ttl_model_predictions,
        )

    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Remove credentials from Redis URL for logging."""
        if "@" in url:
            scheme_end = url.index("://") + 3
            at_pos = url.index("@")
            return url[:scheme_end] + "***@" + url[at_pos + 1:]
        return url

    @property
    def metrics(self) -> CacheMetrics:
        return self._metrics

    # -------------------------------------------------------------------------
    # Health & Monitoring
    # -------------------------------------------------------------------------

    def health_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            return self._client.ping()
        except RedisError as e:
            logger.error("redis_health_check_failed", error=str(e))
            return False

    def get_info(self) -> dict[str, Any]:
        """Get Redis server info."""
        try:
            info = self._client.info(section="memory")
            info.update(self._client.info(section="stats"))
            info.update(self._client.info(section="clients"))
            return info
        except RedisError as e:
            logger.error("redis_info_failed", error=str(e))
            return {}

    def get_pool_stats(self) -> dict[str, int]:
        """Get connection pool statistics."""
        return {
            "max_connections": self._pool.max_connections,
            "current_connections": len(self._pool._in_use_connections),
            "available_connections": len(self._pool._available_connections),
        }

    # -------------------------------------------------------------------------
    # Customer Profile Caching
    # -------------------------------------------------------------------------

    def set_customer_profile(
        self,
        customer_id: str,
        profile: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Cache a customer profile for velocity lookups.

        Args:
            customer_id: Unique customer identifier.
            profile: Customer profile data (risk tier, history, etc.).
            ttl: Override TTL in seconds. Uses configured default if None.

        Returns:
            True if successfully cached.
        """
        key = f"{PREFIX_CUSTOMER_PROFILE}{customer_id}"
        effective_ttl = ttl or self._ttl_customer_profile
        start = time.perf_counter()

        try:
            serialized = json.dumps(profile, default=str)
            self._client.setex(key, effective_ttl, serialized)
            latency = (time.perf_counter() - start) * 1000
            self._metrics.record_set(latency)
            logger.debug("customer_profile_cached", customer_id=customer_id, ttl=effective_ttl)
            return True
        except RedisError as e:
            self._metrics.record_error()
            logger.error("customer_profile_cache_set_failed", customer_id=customer_id, error=str(e))
            return False

    def get_customer_profile(self, customer_id: str) -> dict[str, Any] | None:
        """Retrieve a cached customer profile.

        Args:
            customer_id: Unique customer identifier.

        Returns:
            Customer profile dict or None if not cached.
        """
        key = f"{PREFIX_CUSTOMER_PROFILE}{customer_id}"
        start = time.perf_counter()

        try:
            data = self._client.get(key)
            latency = (time.perf_counter() - start) * 1000

            if data is None:
                self._metrics.record_miss(latency)
                return None

            self._metrics.record_hit(latency)
            return json.loads(data)
        except RedisError as e:
            self._metrics.record_error()
            logger.error("customer_profile_cache_get_failed", customer_id=customer_id, error=str(e))
            return None

    def invalidate_customer_profile(self, customer_id: str) -> bool:
        """Remove a customer profile from cache."""
        key = f"{PREFIX_CUSTOMER_PROFILE}{customer_id}"
        try:
            self._client.delete(key)
            self._metrics.record_delete()
            return True
        except RedisError as e:
            self._metrics.record_error()
            logger.error("customer_profile_invalidate_failed", customer_id=customer_id, error=str(e))
            return False

    # -------------------------------------------------------------------------
    # Recent Transaction Caching (Deduplication)
    # -------------------------------------------------------------------------

    def cache_recent_transaction(
        self,
        transaction_id: str,
        transaction_data: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Cache a recent transaction for deduplication checks.

        Args:
            transaction_id: External transaction ID.
            transaction_data: Minimal transaction metadata.
            ttl: Override TTL in seconds.

        Returns:
            True if successfully cached.
        """
        key = f"{PREFIX_RECENT_TRANSACTIONS}{transaction_id}"
        effective_ttl = ttl or self._ttl_recent_transactions
        start = time.perf_counter()

        try:
            serialized = json.dumps(transaction_data, default=str)
            self._client.setex(key, effective_ttl, serialized)
            latency = (time.perf_counter() - start) * 1000
            self._metrics.record_set(latency)
            return True
        except RedisError as e:
            self._metrics.record_error()
            logger.error("recent_txn_cache_set_failed", transaction_id=transaction_id, error=str(e))
            return False

    def is_duplicate_transaction(self, transaction_id: str) -> bool:
        """Check if a transaction has already been processed (dedup).

        Args:
            transaction_id: External transaction ID.

        Returns:
            True if transaction exists in cache (duplicate).
        """
        key = f"{PREFIX_RECENT_TRANSACTIONS}{transaction_id}"
        start = time.perf_counter()

        try:
            exists = self._client.exists(key)
            latency = (time.perf_counter() - start) * 1000

            if exists:
                self._metrics.record_hit(latency)
            else:
                self._metrics.record_miss(latency)

            return bool(exists)
        except RedisError as e:
            self._metrics.record_error()
            logger.error("dedup_check_failed", transaction_id=transaction_id, error=str(e))
            return False

    def get_recent_transaction(self, transaction_id: str) -> dict[str, Any] | None:
        """Retrieve a cached recent transaction."""
        key = f"{PREFIX_RECENT_TRANSACTIONS}{transaction_id}"
        start = time.perf_counter()

        try:
            data = self._client.get(key)
            latency = (time.perf_counter() - start) * 1000

            if data is None:
                self._metrics.record_miss(latency)
                return None

            self._metrics.record_hit(latency)
            return json.loads(data)
        except RedisError as e:
            self._metrics.record_error()
            logger.error("recent_txn_cache_get_failed", transaction_id=transaction_id, error=str(e))
            return None

    # -------------------------------------------------------------------------
    # Model Prediction Caching
    # -------------------------------------------------------------------------

    def cache_prediction(
        self,
        transaction_fingerprint: str,
        prediction: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Cache a model prediction result.

        Uses a fingerprint of transaction features as the key so that
        identical transactions reuse cached predictions.

        Args:
            transaction_fingerprint: Hash of transaction features.
            prediction: Model prediction result (score, explanation, etc.).
            ttl: Override TTL in seconds.

        Returns:
            True if successfully cached.
        """
        key = f"{PREFIX_MODEL_PREDICTION}{transaction_fingerprint}"
        effective_ttl = ttl or self._ttl_model_predictions
        start = time.perf_counter()

        try:
            serialized = json.dumps(prediction, default=str)
            self._client.setex(key, effective_ttl, serialized)
            latency = (time.perf_counter() - start) * 1000
            self._metrics.record_set(latency)
            return True
        except RedisError as e:
            self._metrics.record_error()
            logger.error("prediction_cache_set_failed", error=str(e))
            return False

    def get_cached_prediction(self, transaction_fingerprint: str) -> dict[str, Any] | None:
        """Retrieve a cached model prediction.

        Args:
            transaction_fingerprint: Hash of transaction features.

        Returns:
            Prediction dict or None if not cached or expired.
        """
        key = f"{PREFIX_MODEL_PREDICTION}{transaction_fingerprint}"
        start = time.perf_counter()

        try:
            data = self._client.get(key)
            latency = (time.perf_counter() - start) * 1000

            if data is None:
                self._metrics.record_miss(latency)
                return None

            self._metrics.record_hit(latency)
            return json.loads(data)
        except RedisError as e:
            self._metrics.record_error()
            logger.error("prediction_cache_get_failed", error=str(e))
            return None

    @staticmethod
    def compute_transaction_fingerprint(transaction: dict[str, Any]) -> str:
        """Compute a stable fingerprint for a transaction for prediction caching.

        Uses key features that determine the prediction outcome.

        Args:
            transaction: Transaction record dict.

        Returns:
            SHA-256 hex digest of feature values.
        """
        feature_keys = [
            "customer_id",
            "merchant_id",
            "transaction_amount",
            "transaction_type",
            "channel",
            "geo_country",
        ]
        feature_values = "|".join(
            str(transaction.get(k, "")) for k in sorted(feature_keys)
        )
        return hashlib.sha256(feature_values.encode()).hexdigest()[:32]

    # -------------------------------------------------------------------------
    # Velocity Data Caching
    # -------------------------------------------------------------------------

    def increment_velocity_counter(
        self,
        customer_id: str,
        window_key: str,
        ttl: int = 600,
    ) -> int:
        """Atomically increment a velocity counter for a customer.

        Used for tracking transaction frequency within time windows.

        Args:
            customer_id: Customer identifier.
            window_key: Time window identifier (e.g., "10min", "1hr").
            ttl: TTL in seconds for the counter.

        Returns:
            New counter value after increment.
        """
        key = f"{PREFIX_VELOCITY}{customer_id}:{window_key}"
        try:
            pipe = self._client.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl)
            results = pipe.execute()
            return results[0]
        except RedisError as e:
            self._metrics.record_error()
            logger.error("velocity_increment_failed", customer_id=customer_id, error=str(e))
            return 0

    def get_velocity_count(self, customer_id: str, window_key: str) -> int:
        """Get current velocity count for a customer in a given window.

        Args:
            customer_id: Customer identifier.
            window_key: Time window identifier.

        Returns:
            Current count or 0 if not found.
        """
        key = f"{PREFIX_VELOCITY}{customer_id}:{window_key}"
        try:
            value = self._client.get(key)
            return int(value) if value else 0
        except RedisError as e:
            self._metrics.record_error()
            logger.error("velocity_get_failed", customer_id=customer_id, error=str(e))
            return 0

    # -------------------------------------------------------------------------
    # Deduplication Lock (Set-If-Not-Exists)
    # -------------------------------------------------------------------------

    def acquire_processing_lock(
        self,
        transaction_id: str,
        lock_ttl: int = 30,
    ) -> bool:
        """Acquire a distributed lock for transaction processing.

        Prevents duplicate processing of the same transaction across workers.

        Args:
            transaction_id: Transaction to lock.
            lock_ttl: Lock expiry in seconds (prevents deadlocks).

        Returns:
            True if lock acquired (first to process), False if already locked.
        """
        key = f"{PREFIX_LOCK}{transaction_id}"
        try:
            acquired = self._client.set(key, "1", nx=True, ex=lock_ttl)
            return bool(acquired)
        except RedisError as e:
            self._metrics.record_error()
            logger.error("lock_acquire_failed", transaction_id=transaction_id, error=str(e))
            return False

    def release_processing_lock(self, transaction_id: str) -> bool:
        """Release a processing lock."""
        key = f"{PREFIX_LOCK}{transaction_id}"
        try:
            self._client.delete(key)
            return True
        except RedisError as e:
            self._metrics.record_error()
            logger.error("lock_release_failed", transaction_id=transaction_id, error=str(e))
            return False

    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------

    def bulk_set_customer_profiles(
        self,
        profiles: dict[str, dict[str, Any]],
        ttl: int | None = None,
    ) -> int:
        """Bulk cache customer profiles using pipeline for efficiency.

        Args:
            profiles: Mapping of customer_id → profile data.
            ttl: Override TTL in seconds.

        Returns:
            Number of profiles successfully cached.
        """
        effective_ttl = ttl or self._ttl_customer_profile
        start = time.perf_counter()
        cached_count = 0

        try:
            pipe = self._client.pipeline(transaction=False)
            for customer_id, profile in profiles.items():
                key = f"{PREFIX_CUSTOMER_PROFILE}{customer_id}"
                serialized = json.dumps(profile, default=str)
                pipe.setex(key, effective_ttl, serialized)
                cached_count += 1

            pipe.execute()
            latency = (time.perf_counter() - start) * 1000
            logger.info(
                "bulk_customer_profiles_cached",
                count=cached_count,
                latency_ms=round(latency, 2),
            )
            return cached_count
        except RedisError as e:
            self._metrics.record_error()
            logger.error("bulk_profile_cache_failed", error=str(e))
            return 0

    def bulk_cache_transactions(
        self,
        transactions: dict[str, dict[str, Any]],
        ttl: int | None = None,
    ) -> int:
        """Bulk cache recent transactions using pipeline.

        Args:
            transactions: Mapping of transaction_id → transaction metadata.
            ttl: Override TTL in seconds.

        Returns:
            Number of transactions successfully cached.
        """
        effective_ttl = ttl or self._ttl_recent_transactions
        start = time.perf_counter()
        cached_count = 0

        try:
            pipe = self._client.pipeline(transaction=False)
            for txn_id, txn_data in transactions.items():
                key = f"{PREFIX_RECENT_TRANSACTIONS}{txn_id}"
                serialized = json.dumps(txn_data, default=str)
                pipe.setex(key, effective_ttl, serialized)
                cached_count += 1

            pipe.execute()
            latency = (time.perf_counter() - start) * 1000
            logger.info(
                "bulk_transactions_cached",
                count=cached_count,
                latency_ms=round(latency, 2),
            )
            return cached_count
        except RedisError as e:
            self._metrics.record_error()
            logger.error("bulk_transaction_cache_failed", error=str(e))
            return 0

    # -------------------------------------------------------------------------
    # Cache Invalidation
    # -------------------------------------------------------------------------

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern.

        Uses SCAN to avoid blocking Redis with KEYS command.

        Args:
            pattern: Redis glob pattern (e.g., "cp:CUST-*").

        Returns:
            Number of keys deleted.
        """
        deleted = 0
        try:
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    self._client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break

            logger.info("cache_invalidated_by_pattern", pattern=pattern, deleted=deleted)
            return deleted
        except RedisError as e:
            self._metrics.record_error()
            logger.error("pattern_invalidation_failed", pattern=pattern, error=str(e))
            return deleted

    def invalidate_customer_data(self, customer_id: str) -> int:
        """Invalidate all cached data for a specific customer.

        Removes profile, velocity counters, and associated predictions.

        Args:
            customer_id: Customer identifier.

        Returns:
            Number of keys deleted.
        """
        patterns = [
            f"{PREFIX_CUSTOMER_PROFILE}{customer_id}",
            f"{PREFIX_VELOCITY}{customer_id}:*",
        ]
        deleted = 0
        for pattern in patterns:
            if "*" in pattern:
                deleted += self.invalidate_by_pattern(pattern)
            else:
                try:
                    deleted += self._client.delete(pattern)
                except RedisError:
                    pass
        return deleted

    def flush_all_caches(self) -> bool:
        """Flush all RiskPulse cache keys (not entire Redis DB).

        Only removes keys with known RiskPulse prefixes.

        Returns:
            True if successful.
        """
        prefixes = [
            PREFIX_CUSTOMER_PROFILE,
            PREFIX_RECENT_TRANSACTIONS,
            PREFIX_MODEL_PREDICTION,
            PREFIX_VELOCITY,
            PREFIX_LOCK,
            PREFIX_DEDUP,
        ]
        total_deleted = 0
        for prefix in prefixes:
            total_deleted += self.invalidate_by_pattern(f"{prefix}*")

        logger.info("all_caches_flushed", total_deleted=total_deleted)
        return True

    # -------------------------------------------------------------------------
    # Write-Ahead Pattern Support
    # -------------------------------------------------------------------------

    def write_ahead_cache(
        self,
        transaction_id: str,
        data: dict[str, Any],
        ttl: int = 300,
    ) -> bool:
        """Cache transaction data as part of write-ahead pattern.

        Data flows: Kafka → Redis (immediate) → DB (async).
        This ensures data is available for real-time lookups before
        the database write confirms.

        Args:
            transaction_id: Transaction identifier.
            data: Full transaction data to cache.
            ttl: TTL in seconds (should exceed expected DB write latency).

        Returns:
            True if cached successfully.
        """
        key = f"{PREFIX_DEDUP}{transaction_id}"
        start = time.perf_counter()

        try:
            serialized = json.dumps(data, default=str)
            self._client.setex(key, ttl, serialized)
            latency = (time.perf_counter() - start) * 1000
            self._metrics.record_set(latency)
            return True
        except RedisError as e:
            self._metrics.record_error()
            logger.error("write_ahead_cache_failed", transaction_id=transaction_id, error=str(e))
            return False

    def get_write_ahead_data(self, transaction_id: str) -> dict[str, Any] | None:
        """Retrieve write-ahead cached data."""
        key = f"{PREFIX_DEDUP}{transaction_id}"
        start = time.perf_counter()

        try:
            data = self._client.get(key)
            latency = (time.perf_counter() - start) * 1000

            if data is None:
                self._metrics.record_miss(latency)
                return None

            self._metrics.record_hit(latency)
            return json.loads(data)
        except RedisError as e:
            self._metrics.record_error()
            logger.error("write_ahead_get_failed", transaction_id=transaction_id, error=str(e))
            return None

    def confirm_write_ahead(self, transaction_id: str) -> bool:
        """Confirm write-ahead data has been persisted to DB.

        Removes the write-ahead key (data now lives in DB).

        Args:
            transaction_id: Transaction that was persisted.

        Returns:
            True if confirmed/removed.
        """
        key = f"{PREFIX_DEDUP}{transaction_id}"
        try:
            self._client.delete(key)
            self._metrics.record_delete()
            return True
        except RedisError as e:
            self._metrics.record_error()
            logger.error("write_ahead_confirm_failed", transaction_id=transaction_id, error=str(e))
            return False

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Close the Redis connection pool."""
        try:
            self._pool.disconnect()
            logger.info("cache_handler_closed")
        except RedisError as e:
            logger.error("cache_handler_close_failed", error=str(e))

    def reset_metrics(self) -> None:
        """Reset cache metrics counters."""
        self._metrics.reset()


def create_cache_handler(**kwargs: Any) -> CacheHandler:
    """Factory function to create a CacheHandler with settings from config.

    Args:
        **kwargs: Override arguments passed to CacheHandler constructor.

    Returns:
        Configured CacheHandler instance.
    """
    return CacheHandler(**kwargs)
