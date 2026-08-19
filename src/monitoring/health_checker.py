"""Dependency health checks and readiness/liveness models for RiskPulse."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.utils.config import get_settings
from src.utils.constants import APP_NAME, APP_VERSION

CheckCallable = Callable[[], bool | Awaitable[bool]]


@dataclass(frozen=True)
class DependencyCheckResult:
    """Status of a single dependency check."""

    name: str
    status: str
    latency_ms: float
    critical: bool = True
    detail: str | None = None

    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"


@dataclass(frozen=True)
class ServiceHealth:
    """Full service health payload."""

    status: str
    service: str
    version: str
    environment: str
    timestamp: str
    uptime_seconds: float
    dependencies: list[DependencyCheckResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "service": self.service,
            "version": self.version,
            "environment": self.environment,
            "timestamp": self.timestamp,
            "uptime_seconds": self.uptime_seconds,
            "dependencies": [
                {
                    "name": item.name,
                    "status": item.status,
                    "latency_ms": item.latency_ms,
                    "detail": item.detail,
                }
                for item in self.dependencies
            ],
        }


class HealthChecker:
    """Runs service dependency checks with timeouts and severity handling."""

    def __init__(
        self,
        *,
        service_name: str = APP_NAME,
        version: str = APP_VERSION,
        environment: str | None = None,
        start_time: float | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        settings = get_settings()
        self.service_name = service_name
        self.version = version
        self.environment = environment or settings.environment
        self.start_time = start_time or time.time()
        self.timeout_seconds = timeout_seconds

    async def check_all(self) -> ServiceHealth:
        checks = await asyncio.gather(
            self.check_kafka(),
            self.check_postgres(),
            self.check_redis(),
            self.check_snowflake(),
        )
        return self.build_health(checks)

    def build_health(self, dependencies: list[DependencyCheckResult]) -> ServiceHealth:
        critical_failures = [item for item in dependencies if item.critical and not item.is_healthy]
        noncritical_failures = [
            item for item in dependencies if not item.critical and not item.is_healthy
        ]

        if critical_failures:
            status = "degraded"
        elif noncritical_failures:
            status = "degraded"
        else:
            status = "healthy"

        return ServiceHealth(
            status=status,
            service=self.service_name,
            version=self.version,
            environment=self.environment,
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=round(time.time() - self.start_time, 2),
            dependencies=dependencies,
        )

    async def readiness(self) -> dict[str, Any]:
        kafka, postgres = await asyncio.gather(self.check_kafka(), self.check_postgres())
        failing = [item.name for item in (kafka, postgres) if not item.is_healthy]
        if failing:
            return {"status": "not_ready", "failed_dependencies": failing}
        return {"status": "ready"}

    def liveness(self) -> dict[str, str]:
        return {"status": "alive"}

    async def check_kafka(self) -> DependencyCheckResult:
        settings = get_settings()

        def _probe() -> bool:
            from confluent_kafka.admin import AdminClient

            admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
            metadata = admin.list_topics(timeout=self.timeout_seconds)
            return metadata is not None

        return await self._run_check("kafka", _probe, critical=True)

    async def check_postgres(self) -> DependencyCheckResult:
        async def _probe() -> bool:
            import asyncpg

            settings = get_settings()
            host = settings.get("database.host", "localhost")
            port = settings.get("database.port", 5432)
            dbname = settings.get("database.name", "riskpulse")
            user = settings.get("database.user", "riskpulse")
            password = settings.get("database.password", "riskpulse")
            conn = await asyncpg.connect(
                host=host,
                port=port,
                database=dbname,
                user=user,
                password=password,
                timeout=self.timeout_seconds,
            )
            try:
                await conn.fetchval("SELECT 1")
            finally:
                await conn.close()
            return True

        return await self._run_check("postgresql", _probe, critical=True)

    async def check_redis(self) -> DependencyCheckResult:
        async def _probe() -> bool:
            import redis.asyncio as aioredis

            settings = get_settings()
            client = aioredis.from_url(
                settings.redis_url,
                socket_connect_timeout=self.timeout_seconds,
                socket_timeout=self.timeout_seconds,
            )
            try:
                return bool(await client.ping())
            finally:
                await client.aclose()

        return await self._run_check("redis", _probe, critical=False)

    async def check_snowflake(self) -> DependencyCheckResult:
        def _probe() -> bool:
            from src.storage.snowflake_handler import SnowflakeHandler

            handler = SnowflakeHandler(pool_size=1)
            try:
                handler.connect()
                result = handler.execute_query("SELECT 1 AS HEALTH_CHECK")
                return result.row_count >= 1
            finally:
                handler.close()

        return await self._run_check("snowflake", _probe, critical=False)

    async def _run_check(
        self,
        name: str,
        check: CheckCallable,
        *,
        critical: bool,
    ) -> DependencyCheckResult:
        start = time.perf_counter()
        try:
            if inspect.iscoroutinefunction(check):
                healthy = await asyncio.wait_for(check(), timeout=self.timeout_seconds)
            else:
                healthy = await asyncio.wait_for(
                    asyncio.to_thread(lambda: bool(check())),
                    timeout=self.timeout_seconds,
                )
            status = "healthy" if healthy else "unhealthy"
            detail = None if healthy else "probe returned false"
        except Exception as exc:
            status = "unhealthy"
            detail = str(exc)[:200]

        return DependencyCheckResult(
            name=name,
            status=status,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            critical=critical,
            detail=detail,
        )
