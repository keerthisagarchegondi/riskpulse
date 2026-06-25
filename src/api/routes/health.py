"""Health check endpoints for RiskPulse API.

Provides liveness and readiness probes for container orchestration,
as well as detailed service dependency status.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter, status
from pydantic import BaseModel

from src.utils.config import get_settings
from src.utils.constants import APP_NAME, APP_VERSION

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Health"])


class DependencyStatus(BaseModel):
    """Status of a single service dependency."""

    name: str
    status: str  # "healthy", "unhealthy", "degraded"
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    """Comprehensive health check response."""

    status: str  # "healthy", "unhealthy", "degraded"
    service: str
    version: str
    environment: str
    timestamp: str
    uptime_seconds: float
    dependencies: list[DependencyStatus] = []


# Track service start time
_start_time = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service health check",
    description="Returns the overall health status of the API service and its dependencies.",
)
async def health_check() -> HealthResponse:
    """Comprehensive health check with dependency status."""
    settings = get_settings()
    dependencies: list[DependencyStatus] = []
    overall_status = "healthy"

    # Check Kafka connectivity
    kafka_status = await _check_kafka()
    dependencies.append(kafka_status)
    if kafka_status.status == "unhealthy":
        overall_status = "degraded"

    # Check PostgreSQL connectivity
    postgres_status = await _check_postgres()
    dependencies.append(postgres_status)
    if postgres_status.status == "unhealthy":
        overall_status = "degraded"

    # Check Redis connectivity
    redis_status = await _check_redis()
    dependencies.append(redis_status)
    if redis_status.status == "unhealthy":
        # Redis being down is non-critical (rate limiter fallback exists)
        if overall_status == "healthy":
            overall_status = "healthy"

    return HealthResponse(
        status=overall_status,
        service=APP_NAME,
        version=APP_VERSION,
        environment=settings.environment,
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=round(time.time() - _start_time, 2),
        dependencies=dependencies,
    )


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Simple liveness check for Kubernetes/Docker health probes.",
)
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe — returns 200 if the process is running."""
    return {"status": "alive"}


@router.get(
    "/health/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Readiness check — returns 200 only if the service can accept traffic.",
)
async def readiness() -> dict[str, str]:
    """Kubernetes readiness probe — checks if service is ready to accept traffic."""
    # Check critical dependencies (Kafka must be available for ingestion)
    kafka_status = await _check_kafka()
    if kafka_status.status == "unhealthy":
        return {"status": "not_ready", "reason": "Kafka unavailable"}
    return {"status": "ready"}


async def _check_kafka() -> DependencyStatus:
    """Check Kafka broker connectivity."""
    start = time.perf_counter()
    try:
        from confluent_kafka.admin import AdminClient

        settings = get_settings()
        admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
        metadata = admin.list_topics(timeout=5.0)
        latency = (time.perf_counter() - start) * 1000
        topic_count = len(metadata.topics)
        return DependencyStatus(
            name="kafka",
            status="healthy",
            latency_ms=round(latency, 2),
            detail=f"{topic_count} topics available",
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return DependencyStatus(
            name="kafka",
            status="unhealthy",
            latency_ms=round(latency, 2),
            detail=str(exc)[:200],
        )


async def _check_postgres() -> DependencyStatus:
    """Check PostgreSQL connectivity."""
    start = time.perf_counter()
    try:
        import asyncpg

        settings = get_settings()
        host = settings.get("database.host", "localhost")
        port = settings.get("database.port", 5432)
        dbname = settings.get("database.name", "riskpulse")
        user = settings.get("database.user", "riskpulse")
        password = settings.get("database.password", "riskpulse")

        conn = await asyncpg.connect(
            host=host, port=port, database=dbname, user=user, password=password, timeout=5.0
        )
        await conn.fetchval("SELECT 1")
        await conn.close()
        latency = (time.perf_counter() - start) * 1000
        return DependencyStatus(
            name="postgresql",
            status="healthy",
            latency_ms=round(latency, 2),
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return DependencyStatus(
            name="postgresql",
            status="unhealthy",
            latency_ms=round(latency, 2),
            detail=str(exc)[:200],
        )


async def _check_redis() -> DependencyStatus:
    """Check Redis connectivity."""
    start = time.perf_counter()
    try:
        import redis.asyncio as aioredis

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url, socket_connect_timeout=3)
        await r.ping()
        await r.aclose()
        latency = (time.perf_counter() - start) * 1000
        return DependencyStatus(
            name="redis",
            status="healthy",
            latency_ms=round(latency, 2),
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return DependencyStatus(
            name="redis",
            status="unhealthy",
            latency_ms=round(latency, 2),
            detail=str(exc)[:200],
        )
