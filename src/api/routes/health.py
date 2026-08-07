"""Health check endpoints for RiskPulse API.

Provides liveness and readiness probes for container orchestration,
as well as detailed service dependency status.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from src.monitoring.health_checker import DependencyCheckResult, HealthChecker
from src.utils.config import get_settings
from src.utils.constants import APP_NAME, APP_VERSION

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Health"])
_health_checker = HealthChecker()


class DependencyStatus(BaseModel):
    """Status of a single service dependency."""

    name: str
    status: str
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    """Comprehensive health check response."""

    status: str
    service: str
    version: str
    environment: str
    timestamp: str
    uptime_seconds: float
    dependencies: list[DependencyStatus] = Field(default_factory=list)


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
    dependencies = [
        await _check_kafka(),
        await _check_postgres(),
        await _check_redis(),
        await _check_snowflake(),
    ]
    critical_failures = [
        item for item in dependencies if item.name in {"kafka", "postgresql"} and item.status == "unhealthy"
    ]
    overall_status = "degraded" if critical_failures else "healthy"

    return HealthResponse(
        status=overall_status,
        service=APP_NAME,
        version=APP_VERSION,
        environment=settings.environment,
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=_health_checker.build_health([]).uptime_seconds,
        dependencies=dependencies,
    )


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Simple liveness check for Kubernetes/Docker health probes.",
)
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe - returns 200 if the process is running."""
    return _health_checker.liveness()


@router.get(
    "/health/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Readiness check - returns 200 only if the service can accept traffic.",
)
async def readiness() -> dict[str, object]:
    """Kubernetes readiness probe - checks critical dependencies."""
    dependencies = [
        await _check_kafka(),
        await _check_postgres(),
    ]
    failing = [item.name for item in dependencies if item.status == "unhealthy"]
    if failing:
        return {"status": "not_ready", "failed_dependencies": failing}
    return {"status": "ready"}


async def _check_kafka() -> DependencyStatus:
    """Check Kafka broker connectivity."""
    return _to_api_dependency(await _health_checker.check_kafka())


async def _check_postgres() -> DependencyStatus:
    """Check PostgreSQL connectivity."""
    return _to_api_dependency(await _health_checker.check_postgres())


async def _check_redis() -> DependencyStatus:
    """Check Redis connectivity."""
    return _to_api_dependency(await _health_checker.check_redis())


async def _check_snowflake() -> DependencyStatus:
    """Check Snowflake warehouse connectivity."""
    return _to_api_dependency(await _health_checker.check_snowflake())


def _to_api_dependency(result: DependencyCheckResult) -> DependencyStatus:
    return DependencyStatus(
        name=result.name,
        status=result.status,
        latency_ms=result.latency_ms,
        detail=result.detail,
    )
