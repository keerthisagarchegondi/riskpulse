"""PostgreSQL data access layer with connection pooling, CRUD, and query optimization."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, AsyncGenerator, Sequence

import structlog
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import AsyncAdaptedQueuePool

from src.storage.models import (
    AuditLog,
    Base,
    CustomerProfile,
    FraudAlert,
    RiskScore,
    Transaction,
)

logger = structlog.get_logger(__name__)


def _rowcount(result: Any) -> int:
    """Return SQLAlchemy rowcount for DML results, defaulting safely for drivers without it."""
    return int(getattr(result, "rowcount", 0) or 0)


@dataclass
class QueryMetrics:
    """Tracks query performance metrics."""

    query_name: str
    duration_ms: float
    rows_affected: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PoolStats:
    """Connection pool statistics."""

    pool_size: int
    checked_in: int
    checked_out: int
    overflow: int
    invalid: int


class PostgresHandlerError(Exception):
    """Base exception for PostgreSQL handler errors."""


class ConnectionError(PostgresHandlerError):
    """Raised when database connection fails."""


class QueryError(PostgresHandlerError):
    """Raised when a query operation fails."""


class BulkOperationError(PostgresHandlerError):
    """Raised when a bulk operation fails."""


class PostgresHandler:
    """Async PostgreSQL data access layer with connection pooling and query optimization."""

    def __init__(
        self,
        connection_url: str,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
    ) -> None:
        self._connection_url = connection_url
        self._engine = create_async_engine(
            connection_url,
            poolclass=AsyncAdaptedQueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,
            echo=echo,
        )
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self._metrics: list[QueryMetrics] = []
        self._max_metrics_history = 1000

    async def initialize(self) -> None:
        """Create all tables (for development/testing). In production, use migrations."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("postgres_handler_initialized")

    async def close(self) -> None:
        """Dispose of the connection pool."""
        await self._engine.dispose()
        logger.info("postgres_handler_closed")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide a transactional session scope."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    @asynccontextmanager
    async def atomic(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide an atomic transaction context with nested savepoint support."""
        async with self._session_factory() as session:
            async with session.begin():
                yield session

    # -------------------------------------------------------------------------
    # Health & Monitoring
    # -------------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error("postgres_health_check_failed", error=str(e))
            return False

    def get_pool_stats(self) -> PoolStats:
        """Return current connection pool statistics."""
        pool = self._engine.pool
        return PoolStats(
            pool_size=int(getattr(pool, "size", lambda: 0)()),
            checked_in=int(getattr(pool, "checkedin", lambda: 0)()),
            checked_out=int(getattr(pool, "checkedout", lambda: 0)()),
            overflow=int(getattr(pool, "overflow", lambda: 0)()),
            invalid=int(getattr(pool, "invalidated_count", 0)),
        )

    def get_query_metrics(self, last_n: int = 100) -> list[QueryMetrics]:
        """Return recent query performance metrics."""
        return self._metrics[-last_n:]

    def _record_metric(self, query_name: str, duration_ms: float, rows_affected: int) -> None:
        """Record a query performance metric."""
        metric = QueryMetrics(
            query_name=query_name, duration_ms=duration_ms, rows_affected=rows_affected
        )
        self._metrics.append(metric)
        if len(self._metrics) > self._max_metrics_history:
            self._metrics = self._metrics[-self._max_metrics_history :]
        if duration_ms > 500:
            logger.warning(
                "slow_query_detected",
                query_name=query_name,
                duration_ms=duration_ms,
                rows_affected=rows_affected,
            )

    # -------------------------------------------------------------------------
    # Transaction CRUD
    # -------------------------------------------------------------------------

    async def create_transaction(self, data: dict[str, Any]) -> Transaction:
        """Insert a single transaction record."""
        start = time.perf_counter()
        async with self.session() as session:
            txn = Transaction(**data)
            session.add(txn)
            await session.flush()
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("create_transaction", duration_ms, 1)
            logger.debug("transaction_created", transaction_id=str(txn.transaction_id))
            return txn

    async def get_transaction(self, transaction_id: uuid.UUID) -> Transaction | None:
        """Retrieve a transaction by ID."""
        start = time.perf_counter()
        async with self.session() as session:
            result = await session.get(Transaction, transaction_id)
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("get_transaction", duration_ms, 1 if result else 0)
            return result

    async def get_transaction_by_external_id(self, external_id: str) -> Transaction | None:
        """Retrieve a transaction by external transaction ID."""
        start = time.perf_counter()
        async with self.session() as session:
            stmt = select(Transaction).where(Transaction.external_transaction_id == external_id)
            result = await session.execute(stmt)
            txn = result.scalar_one_or_none()
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("get_transaction_by_external_id", duration_ms, 1 if txn else 0)
            return txn

    async def update_transaction_status(self, transaction_id: uuid.UUID, status: str) -> bool:
        """Update transaction status."""
        start = time.perf_counter()
        async with self.session() as session:
            stmt = (
                update(Transaction)
                .where(Transaction.transaction_id == transaction_id)
                .values(status=status)
            )
            result = await session.execute(stmt)
            duration_ms = (time.perf_counter() - start) * 1000
            rows = _rowcount(result)
            self._record_metric("update_transaction_status", duration_ms, rows)
            return rows > 0

    async def query_transactions(
        self,
        *,
        customer_id: str | None = None,
        account_id: str | None = None,
        status: str | None = None,
        channel: str | None = None,
        min_amount: Decimal | None = None,
        max_amount: Decimal | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Transaction]:
        """Query transactions with filters."""
        start = time.perf_counter()
        async with self.session() as session:
            stmt = select(Transaction)

            if customer_id:
                stmt = stmt.where(Transaction.customer_id == customer_id)
            if account_id:
                stmt = stmt.where(Transaction.account_id == account_id)
            if status:
                stmt = stmt.where(Transaction.status == status)
            if channel:
                stmt = stmt.where(Transaction.channel == channel)
            if min_amount is not None:
                stmt = stmt.where(Transaction.transaction_amount >= min_amount)
            if max_amount is not None:
                stmt = stmt.where(Transaction.transaction_amount <= max_amount)
            if start_time:
                stmt = stmt.where(Transaction.transaction_timestamp >= start_time)
            if end_time:
                stmt = stmt.where(Transaction.transaction_timestamp <= end_time)

            stmt = stmt.order_by(Transaction.transaction_timestamp.desc())
            stmt = stmt.limit(limit).offset(offset)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("query_transactions", duration_ms, len(rows))
            return rows

    async def count_transactions(
        self,
        *,
        customer_id: str | None = None,
        status: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """Count transactions matching filters."""
        start = time.perf_counter()
        async with self.session() as session:
            stmt = select(func.count(Transaction.transaction_id))

            if customer_id:
                stmt = stmt.where(Transaction.customer_id == customer_id)
            if status:
                stmt = stmt.where(Transaction.status == status)
            if start_time:
                stmt = stmt.where(Transaction.transaction_timestamp >= start_time)
            if end_time:
                stmt = stmt.where(Transaction.transaction_timestamp <= end_time)

            result = await session.execute(stmt)
            count = result.scalar_one()
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("count_transactions", duration_ms, count)
            return count

    # -------------------------------------------------------------------------
    # Fraud Alert CRUD
    # -------------------------------------------------------------------------

    async def create_alert(self, data: dict[str, Any]) -> FraudAlert:
        """Insert a single fraud alert."""
        start = time.perf_counter()
        async with self.session() as session:
            alert = FraudAlert(**data)
            session.add(alert)
            await session.flush()
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("create_alert", duration_ms, 1)
            return alert

    async def get_alert(self, alert_id: uuid.UUID) -> FraudAlert | None:
        """Retrieve an alert by ID."""
        start = time.perf_counter()
        async with self.session() as session:
            result = await session.get(FraudAlert, alert_id)
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("get_alert", duration_ms, 1 if result else 0)
            return result

    async def update_alert_status(
        self,
        alert_id: uuid.UUID,
        status: str,
        resolution_notes: str | None = None,
        assigned_to: str | None = None,
    ) -> bool:
        """Update alert status and optional resolution fields."""
        start = time.perf_counter()
        async with self.session() as session:
            values: dict[str, Any] = {"status": status}
            if resolution_notes:
                values["resolution_notes"] = resolution_notes
            if assigned_to:
                values["assigned_to"] = assigned_to
            if status in ("resolved", "false_positive"):
                values["resolved_at"] = func.now()

            stmt = update(FraudAlert).where(FraudAlert.alert_id == alert_id).values(**values)
            result = await session.execute(stmt)
            duration_ms = (time.perf_counter() - start) * 1000
            rows = _rowcount(result)
            self._record_metric("update_alert_status", duration_ms, rows)
            return rows > 0

    async def query_alerts(
        self,
        *,
        transaction_id: uuid.UUID | None = None,
        severity: str | None = None,
        status: str | None = None,
        alert_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[FraudAlert]:
        """Query fraud alerts with filters."""
        start = time.perf_counter()
        async with self.session() as session:
            stmt = select(FraudAlert)

            if transaction_id:
                stmt = stmt.where(FraudAlert.transaction_id == transaction_id)
            if severity:
                stmt = stmt.where(FraudAlert.severity == severity)
            if status:
                stmt = stmt.where(FraudAlert.status == status)
            if alert_type:
                stmt = stmt.where(FraudAlert.alert_type == alert_type)
            if start_time:
                stmt = stmt.where(FraudAlert.created_at >= start_time)
            if end_time:
                stmt = stmt.where(FraudAlert.created_at <= end_time)

            stmt = stmt.order_by(FraudAlert.created_at.desc())
            stmt = stmt.limit(limit).offset(offset)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("query_alerts", duration_ms, len(rows))
            return rows

    # -------------------------------------------------------------------------
    # Risk Score CRUD
    # -------------------------------------------------------------------------

    async def create_risk_score(self, data: dict[str, Any]) -> RiskScore:
        """Insert a risk score record."""
        start = time.perf_counter()
        async with self.session() as session:
            score = RiskScore(**data)
            session.add(score)
            await session.flush()
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("create_risk_score", duration_ms, 1)
            return score

    async def get_risk_scores_for_transaction(
        self, transaction_id: uuid.UUID
    ) -> Sequence[RiskScore]:
        """Get all risk scores for a transaction."""
        start = time.perf_counter()
        async with self.session() as session:
            stmt = (
                select(RiskScore)
                .where(RiskScore.transaction_id == transaction_id)
                .order_by(RiskScore.scoring_timestamp.desc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("get_risk_scores_for_transaction", duration_ms, len(rows))
            return rows

    async def get_high_risk_scores(
        self, threshold: Decimal = Decimal("0.8"), limit: int = 100
    ) -> Sequence[RiskScore]:
        """Get risk scores above threshold."""
        start = time.perf_counter()
        async with self.session() as session:
            stmt = (
                select(RiskScore)
                .where(RiskScore.overall_score >= threshold)
                .order_by(RiskScore.overall_score.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("get_high_risk_scores", duration_ms, len(rows))
            return rows

    # -------------------------------------------------------------------------
    # Customer Profile CRUD
    # -------------------------------------------------------------------------

    async def upsert_customer_profile(self, data: dict[str, Any]) -> CustomerProfile:
        """Upsert a customer profile (insert or update on conflict)."""
        start = time.perf_counter()
        async with self.session() as session:
            stmt = pg_insert(CustomerProfile).values(**data)
            update_cols = {k: v for k, v in data.items() if k != "customer_id"}
            update_cols["updated_at"] = func.now()
            stmt = stmt.on_conflict_do_update(
                index_elements=["customer_id"],
                set_=update_cols,
            )
            await session.execute(stmt)
            await session.flush()

            # Fetch the resulting profile
            profile = await session.get(CustomerProfile, data["customer_id"])
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("upsert_customer_profile", duration_ms, 1)
            return profile  # type: ignore[return-value]

    async def get_customer_profile(self, customer_id: str) -> CustomerProfile | None:
        """Retrieve a customer profile."""
        start = time.perf_counter()
        async with self.session() as session:
            result = await session.get(CustomerProfile, customer_id)
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("get_customer_profile", duration_ms, 1 if result else 0)
            return result

    # -------------------------------------------------------------------------
    # Audit Log
    # -------------------------------------------------------------------------

    async def create_audit_log(self, data: dict[str, Any]) -> AuditLog:
        """Insert an audit log entry."""
        start = time.perf_counter()
        async with self.session() as session:
            log_entry = AuditLog(**data)
            session.add(log_entry)
            await session.flush()
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("create_audit_log", duration_ms, 1)
            return log_entry

    async def query_audit_logs(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        event_type: str | None = None,
        actor: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[AuditLog]:
        """Query audit logs with filters."""
        start = time.perf_counter()
        async with self.session() as session:
            stmt = select(AuditLog)

            if entity_type:
                stmt = stmt.where(AuditLog.entity_type == entity_type)
            if entity_id:
                stmt = stmt.where(AuditLog.entity_id == entity_id)
            if event_type:
                stmt = stmt.where(AuditLog.event_type == event_type)
            if actor:
                stmt = stmt.where(AuditLog.actor == actor)
            if start_time:
                stmt = stmt.where(AuditLog.created_at >= start_time)
            if end_time:
                stmt = stmt.where(AuditLog.created_at <= end_time)

            stmt = stmt.order_by(AuditLog.created_at.desc())
            stmt = stmt.limit(limit).offset(offset)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("query_audit_logs", duration_ms, len(rows))
            return rows

    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------

    async def bulk_upsert_transactions(
        self, records: list[dict[str, Any]], batch_size: int = 1000
    ) -> int:
        """Bulk upsert transactions using ON CONFLICT DO UPDATE.

        Processes records in batches for memory efficiency.
        Returns total number of rows affected.
        """
        start = time.perf_counter()
        total_rows = 0

        async with self.session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                stmt = pg_insert(Transaction).values(batch)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["external_transaction_id"],
                    set_={
                        "status": stmt.excluded.status,
                        "transaction_amount": stmt.excluded.transaction_amount,
                        "processed_timestamp": func.now(),
                        "updated_at": func.now(),
                    },
                )
                result = await session.execute(stmt)
                total_rows += _rowcount(result)

        duration_ms = (time.perf_counter() - start) * 1000
        self._record_metric("bulk_upsert_transactions", duration_ms, total_rows)
        logger.info(
            "bulk_upsert_transactions_complete",
            total_records=len(records),
            rows_affected=total_rows,
            duration_ms=round(duration_ms, 2),
        )
        return total_rows

    async def bulk_insert_alerts(
        self, records: list[dict[str, Any]], batch_size: int = 1000
    ) -> int:
        """Bulk insert fraud alerts. Skips conflicts."""
        start = time.perf_counter()
        total_rows = 0

        async with self.session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                stmt = pg_insert(FraudAlert).values(batch)
                stmt = stmt.on_conflict_do_nothing()
                result = await session.execute(stmt)
                total_rows += _rowcount(result)

        duration_ms = (time.perf_counter() - start) * 1000
        self._record_metric("bulk_insert_alerts", duration_ms, total_rows)
        logger.info(
            "bulk_insert_alerts_complete",
            total_records=len(records),
            rows_affected=total_rows,
            duration_ms=round(duration_ms, 2),
        )
        return total_rows

    async def bulk_upsert_risk_scores(
        self, records: list[dict[str, Any]], batch_size: int = 1000
    ) -> int:
        """Bulk upsert risk scores."""
        start = time.perf_counter()
        total_rows = 0

        async with self.session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                stmt = pg_insert(RiskScore).values(batch)
                stmt = stmt.on_conflict_do_nothing()
                result = await session.execute(stmt)
                total_rows += _rowcount(result)

        duration_ms = (time.perf_counter() - start) * 1000
        self._record_metric("bulk_upsert_risk_scores", duration_ms, total_rows)
        logger.info(
            "bulk_upsert_risk_scores_complete",
            total_records=len(records),
            rows_affected=total_rows,
            duration_ms=round(duration_ms, 2),
        )
        return total_rows

    async def bulk_upsert_customer_profiles(
        self, records: list[dict[str, Any]], batch_size: int = 500
    ) -> int:
        """Bulk upsert customer profiles."""
        start = time.perf_counter()
        total_rows = 0

        async with self.session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                stmt = pg_insert(CustomerProfile).values(batch)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["customer_id"],
                    set_={
                        "total_transactions_24h": stmt.excluded.total_transactions_24h,
                        "total_amount_24h": stmt.excluded.total_amount_24h,
                        "total_transactions_7d": stmt.excluded.total_transactions_7d,
                        "total_amount_7d": stmt.excluded.total_amount_7d,
                        "avg_transaction_amount": stmt.excluded.avg_transaction_amount,
                        "max_transaction_amount": stmt.excluded.max_transaction_amount,
                        "unique_merchants_7d": stmt.excluded.unique_merchants_7d,
                        "unique_countries_7d": stmt.excluded.unique_countries_7d,
                        "last_transaction_timestamp": stmt.excluded.last_transaction_timestamp,
                        "risk_tier": stmt.excluded.risk_tier,
                        "updated_at": func.now(),
                    },
                )
                result = await session.execute(stmt)
                total_rows += _rowcount(result)

        duration_ms = (time.perf_counter() - start) * 1000
        self._record_metric("bulk_upsert_customer_profiles", duration_ms, total_rows)
        logger.info(
            "bulk_upsert_customer_profiles_complete",
            total_records=len(records),
            rows_affected=total_rows,
            duration_ms=round(duration_ms, 2),
        )
        return total_rows

    # -------------------------------------------------------------------------
    # Delete Operations
    # -------------------------------------------------------------------------

    async def delete_transaction(self, transaction_id: uuid.UUID) -> bool:
        """Delete a transaction and cascade to related records."""
        start = time.perf_counter()
        async with self.session() as session:
            stmt = delete(Transaction).where(Transaction.transaction_id == transaction_id)
            result = await session.execute(stmt)
            duration_ms = (time.perf_counter() - start) * 1000
            rows = _rowcount(result)
            self._record_metric("delete_transaction", duration_ms, rows)
            return rows > 0

    # -------------------------------------------------------------------------
    # Aggregation Queries
    # -------------------------------------------------------------------------

    async def get_transaction_stats(
        self,
        customer_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Get aggregated transaction statistics for a customer."""
        start = time.perf_counter()
        async with self.session() as session:
            stmt = select(
                func.count(Transaction.transaction_id).label("total_count"),
                func.sum(Transaction.transaction_amount).label("total_amount"),
                func.avg(Transaction.transaction_amount).label("avg_amount"),
                func.max(Transaction.transaction_amount).label("max_amount"),
                func.min(Transaction.transaction_amount).label("min_amount"),
            ).where(Transaction.customer_id == customer_id)

            if start_time:
                stmt = stmt.where(Transaction.transaction_timestamp >= start_time)
            if end_time:
                stmt = stmt.where(Transaction.transaction_timestamp <= end_time)

            result = await session.execute(stmt)
            row = result.one()
            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("get_transaction_stats", duration_ms, 1)

            return {
                "total_count": row.total_count or 0,
                "total_amount": float(row.total_amount or 0),
                "avg_amount": float(row.avg_amount or 0),
                "max_amount": float(row.max_amount or 0),
                "min_amount": float(row.min_amount or 0),
            }

    async def get_alert_summary(self) -> dict[str, Any]:
        """Get alert counts by severity and status."""
        start = time.perf_counter()
        async with self.session() as session:
            # By severity
            severity_stmt = select(FraudAlert.severity, func.count(FraudAlert.alert_id)).group_by(
                FraudAlert.severity
            )
            severity_result = await session.execute(severity_stmt)
            by_severity = {row[0]: row[1] for row in severity_result.all()}

            # By status
            status_stmt = select(FraudAlert.status, func.count(FraudAlert.alert_id)).group_by(
                FraudAlert.status
            )
            status_result = await session.execute(status_stmt)
            by_status = {row[0]: row[1] for row in status_result.all()}

            duration_ms = (time.perf_counter() - start) * 1000
            self._record_metric("get_alert_summary", duration_ms, 1)

            return {"by_severity": by_severity, "by_status": by_status}

    # -------------------------------------------------------------------------
    # Query Performance Analysis
    # -------------------------------------------------------------------------

    async def explain_query(self, stmt: Any) -> str:
        """Run EXPLAIN ANALYZE on a query statement (for development use)."""
        async with self._engine.connect() as conn:
            compiled = stmt.compile(
                dialect=self._engine.dialect, compile_kwargs={"literal_binds": True}
            )
            explain_stmt = text(f"EXPLAIN ANALYZE {compiled.string}")
            result = await conn.execute(explain_stmt)
            return "\n".join(row[0] for row in result.all())

    def get_performance_summary(self) -> dict[str, Any]:
        """Get aggregated query performance summary."""
        if not self._metrics:
            return {"total_queries": 0}

        durations = [m.duration_ms for m in self._metrics]
        by_query: dict[str, list[float]] = {}
        for m in self._metrics:
            by_query.setdefault(m.query_name, []).append(m.duration_ms)

        return {
            "total_queries": len(self._metrics),
            "avg_duration_ms": sum(durations) / len(durations),
            "max_duration_ms": max(durations),
            "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)],
            "slow_queries": sum(1 for d in durations if d > 500),
            "by_query": {
                name: {
                    "count": len(times),
                    "avg_ms": sum(times) / len(times),
                    "max_ms": max(times),
                }
                for name, times in by_query.items()
            },
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_postgres_handler(
    host: str = "localhost",
    port: int = 5432,
    database: str = "riskpulse",
    user: str = "riskpulse",
    password: str = "",
    pool_size: int = 20,
    max_overflow: int = 10,
    pool_timeout: int = 30,
    pool_recycle: int = 3600,
    echo: bool = False,
) -> PostgresHandler:
    """Create a PostgresHandler instance from individual connection parameters."""
    connection_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
    return PostgresHandler(
        connection_url=connection_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        echo=echo,
    )
