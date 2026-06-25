"""Token bucket rate limiter middleware for RiskPulse API.

Implements per-API-key rate limiting using Redis as the backing store.
Falls back to in-memory rate limiting when Redis is unavailable.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import structlog
from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.utils.config import get_settings
from src.utils.constants import DEFAULT_BURST_SIZE, DEFAULT_RATE_LIMIT

logger = structlog.get_logger(__name__)


class TokenBucket:
    """In-memory token bucket implementation for a single key."""

    __slots__ = ("capacity", "refill_rate", "tokens", "last_refill")

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens. Returns True if successful."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    @property
    def remaining(self) -> int:
        """Approximate remaining tokens."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        return int(min(self.capacity, self.tokens + elapsed * self.refill_rate))


class InMemoryRateLimiter:
    """In-memory rate limiter using token bucket algorithm.

    Used as a fallback when Redis is unavailable.
    Note: This does not work across multiple API instances.
    """

    def __init__(self, default_rate: int = DEFAULT_RATE_LIMIT, burst_size: int = DEFAULT_BURST_SIZE) -> None:
        self._default_rate = default_rate
        self._burst_size = burst_size
        self._buckets: dict[str, TokenBucket] = {}

    def _get_bucket(self, key: str, custom_rate: int | None = None) -> TokenBucket:
        """Get or create a token bucket for the given key."""
        if key not in self._buckets:
            rate = custom_rate or self._default_rate
            # Capacity is the burst size, refill rate converts req/min to tokens/sec
            self._buckets[key] = TokenBucket(
                capacity=self._burst_size + rate,
                refill_rate=rate / 60.0,
            )
        return self._buckets[key]

    def is_allowed(self, key: str, custom_rate: int | None = None) -> tuple[bool, int, int]:
        """Check if a request is allowed for the given key.

        Returns:
            Tuple of (allowed, remaining_tokens, retry_after_seconds)
        """
        bucket = self._get_bucket(key, custom_rate)
        allowed = bucket.consume()
        remaining = bucket.remaining
        retry_after = 0 if allowed else int(1.0 / bucket.refill_rate) + 1
        return allowed, remaining, retry_after

    def reset(self, key: str) -> None:
        """Reset rate limit for a specific key."""
        self._buckets.pop(key, None)

    def clear(self) -> None:
        """Clear all rate limit state."""
        self._buckets.clear()


class RedisRateLimiter:
    """Redis-backed rate limiter using sliding window counters.

    Provides distributed rate limiting across multiple API instances.
    """

    def __init__(self, redis_client: Any, default_rate: int = DEFAULT_RATE_LIMIT, window_seconds: int = 60) -> None:
        self._redis = redis_client
        self._default_rate = default_rate
        self._window_seconds = window_seconds

    async def is_allowed(self, key: str, custom_rate: int | None = None) -> tuple[bool, int, int]:
        """Check if a request is allowed using Redis sliding window.

        Returns:
            Tuple of (allowed, remaining_requests, retry_after_seconds)
        """
        rate_limit = custom_rate or self._default_rate
        now = time.time()
        window_start = now - self._window_seconds
        redis_key = f"ratelimit:{key}"

        try:
            pipe = self._redis.pipeline()
            # Remove expired entries
            pipe.zremrangebyscore(redis_key, 0, window_start)
            # Count current window requests
            pipe.zcard(redis_key)
            # Add current request
            pipe.zadd(redis_key, {str(now): now})
            # Set TTL on the key
            pipe.expire(redis_key, self._window_seconds + 1)
            results = await pipe.execute()

            current_count = results[1]
            remaining = max(0, rate_limit - current_count - 1)

            if current_count >= rate_limit:
                # Remove the entry we just added since request is denied
                await self._redis.zrem(redis_key, str(now))
                retry_after = self._window_seconds - int(now - window_start)
                return False, 0, max(1, retry_after)

            return True, remaining, 0
        except Exception as exc:
            logger.error("redis_rate_limit_error", error=str(exc))
            # Fail open - allow the request if Redis is down
            return True, rate_limit, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that enforces per-API-key rate limits.

    Rate limit info is returned in response headers:
    - X-RateLimit-Limit: Maximum requests per window
    - X-RateLimit-Remaining: Remaining requests in current window
    - X-RateLimit-Reset: Seconds until the rate limit resets
    """

    # Paths exempt from rate limiting
    EXEMPT_PATHS = frozenset({"/health", "/health/ready", "/health/live", "/docs", "/openapi.json", "/redoc"})

    def __init__(self, app: Any, redis_client: Any | None = None) -> None:
        super().__init__(app)
        settings = get_settings()
        self._rate_limit = settings.get("api.rate_limit_per_minute", DEFAULT_RATE_LIMIT)

        if redis_client is not None:
            self._limiter: RedisRateLimiter | InMemoryRateLimiter = RedisRateLimiter(
                redis_client=redis_client,
                default_rate=self._rate_limit,
            )
        else:
            self._limiter = InMemoryRateLimiter(default_rate=self._rate_limit)
            logger.warning("rate_limiter_fallback", msg="Using in-memory rate limiter (not suitable for production)")

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply rate limiting based on API key identity."""
        # Skip rate limiting for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Determine rate limit key (API key name or IP address)
        key = self._get_rate_limit_key(request)
        custom_rate = self._get_custom_rate(request)

        allowed, remaining, retry_after = await self._check_limit(key, custom_rate)

        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                key=key,
                path=request.url.path,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please retry later.",
                headers={
                    "X-RateLimit-Limit": str(self._rate_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(retry_after),
                    "Retry-After": str(retry_after),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(custom_rate or self._rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    async def _check_limit(self, key: str, custom_rate: int | None) -> tuple[bool, int, int]:
        """Check rate limit, handling both sync and async limiters."""
        if isinstance(self._limiter, RedisRateLimiter):
            return await self._limiter.is_allowed(key, custom_rate)
        return self._limiter.is_allowed(key, custom_rate)

    @staticmethod
    def _get_rate_limit_key(request: Request) -> str:
        """Extract the rate limit key from request state or client IP."""
        # Prefer API key name set by auth middleware
        if hasattr(request.state, "api_key_name"):
            return f"apikey:{request.state.api_key_name}"
        # Fallback to client IP
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    @staticmethod
    def _get_custom_rate(request: Request) -> int | None:
        """Get custom rate limit if the API key has one configured."""
        if hasattr(request.state, "api_key_rate_limit"):
            return request.state.api_key_rate_limit
        return None
