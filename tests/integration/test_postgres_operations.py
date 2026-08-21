"""Integration tests for PostgreSQL data access layer.

Requires a running PostgreSQL instance. Uses testcontainers for isolation
when available, otherwise connects to the configured test database.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from src.storage.models import Base, Transaction
from src.storage.postgres_handler import PostgresHandler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_DB_URL = "postgresql+asyncpg://riskpulse:riskpulse_dev_password@localhost:5432/riskpulse_test"


@pytest_asyncio.fixture
async def pg_handler() -> AsyncGenerator[PostgresHandler, None]:
    """Create a fresh PostgresHandler with clean tables for each test."""
    handler = PostgresHandler(connection_url=TEST_DB_URL, pool_size=5, max_overflow=5, echo=False)
    # Create tables
    await handler.initialize()
    # Clean all tables before test
    async with handler._engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())
    yield handler
    # Cleanup after test
    async with handler._engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())
    await handler.close()


def _make_transaction_data(suffix: str = "001") -> dict:
    """Generate transaction test data."""
    return {
        "external_transaction_id": f"TXN-TEST-{suffix}-{uuid.uuid4().hex[:8]}",
        "account_id": f"ACC-{suffix}",
        "customer_id": f"CUST-{suffix}",
        "merchant_id": f"MERCH-{suffix}",
        "merchant_name": f"Test Merchant {suffix}",
        "merchant_category_code": "5411",
        "transaction_amount": Decimal("125.50"),
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": "credit",
        "card_last_four": "4242",
        "ip_address": "192.168.1.100",
        "device_id": f"device-{suffix}",
        "device_type": "mobile",
        "geo_latitude": Decimal("40.71280000"),
        "geo_longitude": Decimal("-74.00600000"),
        "geo_country": "USA",
        "geo_city": "New York",
        "is_international": False,
        "transaction_timestamp": datetime.now(timezone.utc),
        "status": "pending",
    }


def _make_alert_data(transaction_id: uuid.UUID) -> dict:
    """Generate fraud alert test data."""
    return {
        "transaction_id": transaction_id,
        "alert_type": "rule_based",
        "rule_id": "RULE-001",
        "risk_score": Decimal("0.8500"),
        "severity": "high",
        "status": "open",
        "description": "High-value transaction from unusual location",
        "details": {"triggered_rules": ["high_amount", "new_location"]},
    }


def _make_risk_score_data(transaction_id: uuid.UUID) -> dict:
    """Generate risk score test data."""
    return {
        "transaction_id": transaction_id,
        "model_version": "1.0.0",
        "overall_score": Decimal("0.7500"),
        "rule_score": Decimal("0.6000"),
        "anomaly_score": Decimal("0.8000"),
        "ml_score": Decimal("0.8200"),
        "feature_contributions": {"amount_zscore": 0.3, "velocity_1h": 0.25},
        "latency_ms": 45,
    }


# ---------------------------------------------------------------------------
# Connection & Health Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestConnectionAndHealth:
    @pytest.mark.asyncio
    async def test_health_check_passes(self, pg_handler: PostgresHandler) -> None:
        result = await pg_handler.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_pool_stats_available(self, pg_handler: PostgresHandler) -> None:
        stats = pg_handler.get_pool_stats()
        assert stats.pool_size == 5
        assert stats.checked_in >= 0
        assert stats.checked_out >= 0

    @pytest.mark.asyncio
    async def test_health_check_with_bad_connection(self) -> None:
        bad_handler = PostgresHandler(
            connection_url="postgresql+asyncpg://bad:bad@localhost:9999/nonexistent",
            pool_size=1,
        )
        result = await bad_handler.health_check()
        assert result is False
        await bad_handler.close()


# ---------------------------------------------------------------------------
# Transaction CRUD Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTransactionCRUD:
    @pytest.mark.asyncio
    async def test_create_transaction(self, pg_handler: PostgresHandler) -> None:
        data = _make_transaction_data("create")
        txn = await pg_handler.create_transaction(data)
        assert txn.transaction_id is not None
        assert txn.external_transaction_id == data["external_transaction_id"]
        assert txn.transaction_amount == Decimal("125.50")
        assert txn.status == "pending"

    @pytest.mark.asyncio
    async def test_get_transaction(self, pg_handler: PostgresHandler) -> None:
        data = _make_transaction_data("get")
        created = await pg_handler.create_transaction(data)
        fetched = await pg_handler.get_transaction(created.transaction_id)
        assert fetched is not None
        assert fetched.transaction_id == created.transaction_id
        assert fetched.customer_id == data["customer_id"]

    @pytest.mark.asyncio
    async def test_get_transaction_not_found(self, pg_handler: PostgresHandler) -> None:
        result = await pg_handler.get_transaction(uuid.uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_transaction_by_external_id(self, pg_handler: PostgresHandler) -> None:
        data = _make_transaction_data("ext")
        created = await pg_handler.create_transaction(data)
        fetched = await pg_handler.get_transaction_by_external_id(data["external_transaction_id"])
        assert fetched is not None
        assert fetched.transaction_id == created.transaction_id

    @pytest.mark.asyncio
    async def test_update_transaction_status(self, pg_handler: PostgresHandler) -> None:
        data = _make_transaction_data("update")
        created = await pg_handler.create_transaction(data)
        success = await pg_handler.update_transaction_status(created.transaction_id, "approved")
        assert success is True
        fetched = await pg_handler.get_transaction(created.transaction_id)
        assert fetched is not None
        assert fetched.status == "approved"

    @pytest.mark.asyncio
    async def test_query_transactions_by_customer(self, pg_handler: PostgresHandler) -> None:
        # Create multiple transactions for the same customer
        for i in range(5):
            data = _make_transaction_data(f"query-{i}")
            data["customer_id"] = "CUST-QUERY"
            await pg_handler.create_transaction(data)

        results = await pg_handler.query_transactions(customer_id="CUST-QUERY")
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_query_transactions_with_amount_filter(self, pg_handler: PostgresHandler) -> None:
        for i, amount in enumerate([50, 150, 250, 350, 450]):
            data = _make_transaction_data(f"amount-{i}")
            data["customer_id"] = "CUST-AMT"
            data["transaction_amount"] = Decimal(str(amount))
            await pg_handler.create_transaction(data)

        results = await pg_handler.query_transactions(
            customer_id="CUST-AMT",
            min_amount=Decimal("200"),
            max_amount=Decimal("400"),
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_transactions_with_time_filter(self, pg_handler: PostgresHandler) -> None:
        now = datetime.now(timezone.utc)
        for i in range(5):
            data = _make_transaction_data(f"time-{i}")
            data["customer_id"] = "CUST-TIME"
            data["transaction_timestamp"] = now - timedelta(hours=i)
            await pg_handler.create_transaction(data)

        results = await pg_handler.query_transactions(
            customer_id="CUST-TIME",
            start_time=now - timedelta(hours=2, minutes=30),
            end_time=now,
        )
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_count_transactions(self, pg_handler: PostgresHandler) -> None:
        for i in range(3):
            data = _make_transaction_data(f"count-{i}")
            data["customer_id"] = "CUST-COUNT"
            await pg_handler.create_transaction(data)

        count = await pg_handler.count_transactions(customer_id="CUST-COUNT")
        assert count == 3

    @pytest.mark.asyncio
    async def test_delete_transaction(self, pg_handler: PostgresHandler) -> None:
        data = _make_transaction_data("delete")
        created = await pg_handler.create_transaction(data)
        deleted = await pg_handler.delete_transaction(created.transaction_id)
        assert deleted is True
        assert await pg_handler.get_transaction(created.transaction_id) is None


# ---------------------------------------------------------------------------
# Fraud Alert CRUD Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFraudAlertCRUD:
    @pytest.mark.asyncio
    async def test_create_alert(self, pg_handler: PostgresHandler) -> None:
        txn = await pg_handler.create_transaction(_make_transaction_data("alert-create"))
        alert_data = _make_alert_data(txn.transaction_id)
        alert = await pg_handler.create_alert(alert_data)
        assert alert.alert_id is not None
        assert alert.severity == "high"
        assert alert.status == "open"

    @pytest.mark.asyncio
    async def test_get_alert(self, pg_handler: PostgresHandler) -> None:
        txn = await pg_handler.create_transaction(_make_transaction_data("alert-get"))
        alert_data = _make_alert_data(txn.transaction_id)
        created = await pg_handler.create_alert(alert_data)
        fetched = await pg_handler.get_alert(created.alert_id)
        assert fetched is not None
        assert fetched.alert_id == created.alert_id

    @pytest.mark.asyncio
    async def test_update_alert_status_resolved(self, pg_handler: PostgresHandler) -> None:
        txn = await pg_handler.create_transaction(_make_transaction_data("alert-resolve"))
        alert_data = _make_alert_data(txn.transaction_id)
        created = await pg_handler.create_alert(alert_data)

        success = await pg_handler.update_alert_status(
            created.alert_id,
            status="resolved",
            resolution_notes="Confirmed legitimate transaction",
            assigned_to="analyst@riskpulse.com",
        )
        assert success is True

    @pytest.mark.asyncio
    async def test_query_alerts_by_severity(self, pg_handler: PostgresHandler) -> None:
        txn = await pg_handler.create_transaction(_make_transaction_data("alert-sev"))
        for severity in ["low", "medium", "high", "critical"]:
            data = _make_alert_data(txn.transaction_id)
            data["severity"] = severity
            await pg_handler.create_alert(data)

        high_alerts = await pg_handler.query_alerts(severity="high")
        assert len(high_alerts) == 1

    @pytest.mark.asyncio
    async def test_query_alerts_by_status(self, pg_handler: PostgresHandler) -> None:
        txn = await pg_handler.create_transaction(_make_transaction_data("alert-stat"))
        alert_data = _make_alert_data(txn.transaction_id)
        await pg_handler.create_alert(alert_data)

        open_alerts = await pg_handler.query_alerts(status="open")
        assert len(open_alerts) >= 1


# ---------------------------------------------------------------------------
# Risk Score CRUD Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRiskScoreCRUD:
    @pytest.mark.asyncio
    async def test_create_risk_score(self, pg_handler: PostgresHandler) -> None:
        txn = await pg_handler.create_transaction(_make_transaction_data("score-create"))
        score_data = _make_risk_score_data(txn.transaction_id)
        score = await pg_handler.create_risk_score(score_data)
        assert score.score_id is not None
        assert score.overall_score == Decimal("0.7500")

    @pytest.mark.asyncio
    async def test_get_risk_scores_for_transaction(self, pg_handler: PostgresHandler) -> None:
        txn = await pg_handler.create_transaction(_make_transaction_data("score-get"))
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            data = _make_risk_score_data(txn.transaction_id)
            data["model_version"] = version
            await pg_handler.create_risk_score(data)

        scores = await pg_handler.get_risk_scores_for_transaction(txn.transaction_id)
        assert len(scores) == 3

    @pytest.mark.asyncio
    async def test_get_high_risk_scores(self, pg_handler: PostgresHandler) -> None:
        txn = await pg_handler.create_transaction(_make_transaction_data("score-high"))
        for score_val in ["0.5000", "0.7000", "0.8500", "0.9500"]:
            data = _make_risk_score_data(txn.transaction_id)
            data["overall_score"] = Decimal(score_val)
            data["model_version"] = f"v-{score_val}"
            await pg_handler.create_risk_score(data)

        high_scores = await pg_handler.get_high_risk_scores(threshold=Decimal("0.8"))
        assert len(high_scores) == 2
        assert all(s.overall_score >= Decimal("0.8") for s in high_scores)


# ---------------------------------------------------------------------------
# Customer Profile Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCustomerProfileCRUD:
    @pytest.mark.asyncio
    async def test_upsert_customer_profile_insert(self, pg_handler: PostgresHandler) -> None:
        data = {
            "customer_id": "CUST-PROFILE-001",
            "total_transactions_24h": 5,
            "total_amount_24h": Decimal("500.00"),
            "total_transactions_7d": 25,
            "total_amount_7d": Decimal("2500.00"),
            "avg_transaction_amount": Decimal("100.00"),
            "max_transaction_amount": Decimal("250.00"),
            "unique_merchants_7d": 8,
            "unique_countries_7d": 2,
            "risk_tier": "standard",
        }
        profile = await pg_handler.upsert_customer_profile(data)
        assert profile.customer_id == "CUST-PROFILE-001"
        assert profile.total_transactions_24h == 5

    @pytest.mark.asyncio
    async def test_upsert_customer_profile_update(self, pg_handler: PostgresHandler) -> None:
        data = {
            "customer_id": "CUST-PROFILE-UPD",
            "total_transactions_24h": 5,
            "total_amount_24h": Decimal("500.00"),
            "total_transactions_7d": 25,
            "total_amount_7d": Decimal("2500.00"),
            "avg_transaction_amount": Decimal("100.00"),
            "max_transaction_amount": Decimal("250.00"),
            "unique_merchants_7d": 8,
            "unique_countries_7d": 2,
            "risk_tier": "standard",
        }
        await pg_handler.upsert_customer_profile(data)

        # Update with new values
        data["total_transactions_24h"] = 10
        data["risk_tier"] = "elevated"
        updated = await pg_handler.upsert_customer_profile(data)
        assert updated.total_transactions_24h == 10
        assert updated.risk_tier == "elevated"

    @pytest.mark.asyncio
    async def test_get_customer_profile(self, pg_handler: PostgresHandler) -> None:
        data = {
            "customer_id": "CUST-PROFILE-GET",
            "total_transactions_24h": 3,
            "total_amount_24h": Decimal("300.00"),
            "total_transactions_7d": 15,
            "total_amount_7d": Decimal("1500.00"),
            "unique_merchants_7d": 5,
            "unique_countries_7d": 1,
            "risk_tier": "low",
        }
        await pg_handler.upsert_customer_profile(data)
        profile = await pg_handler.get_customer_profile("CUST-PROFILE-GET")
        assert profile is not None
        assert profile.risk_tier == "low"

    @pytest.mark.asyncio
    async def test_get_customer_profile_not_found(self, pg_handler: PostgresHandler) -> None:
        result = await pg_handler.get_customer_profile("NONEXISTENT")
        assert result is None


# ---------------------------------------------------------------------------
# Audit Log Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAuditLogCRUD:
    @pytest.mark.asyncio
    async def test_create_audit_log(self, pg_handler: PostgresHandler) -> None:
        data = {
            "event_type": "transaction.scored",
            "entity_type": "transaction",
            "entity_id": "TXN-001",
            "action": "score_computed",
            "actor": "system:scoring_engine",
            "details": {"score": 0.85, "model": "v1.0"},
        }
        log = await pg_handler.create_audit_log(data)
        assert log.log_id is not None
        assert log.event_type == "transaction.scored"

    @pytest.mark.asyncio
    async def test_query_audit_logs_by_entity(self, pg_handler: PostgresHandler) -> None:
        for i in range(3):
            data = {
                "event_type": "alert.created",
                "entity_type": "alert",
                "entity_id": f"ALERT-{i}",
                "action": "created",
                "actor": "system:fraud_engine",
            }
            await pg_handler.create_audit_log(data)

        logs = await pg_handler.query_audit_logs(entity_type="alert")
        assert len(logs) == 3


# ---------------------------------------------------------------------------
# Bulk Operations Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBulkOperations:
    @pytest.mark.asyncio
    async def test_bulk_upsert_transactions(self, pg_handler: PostgresHandler) -> None:
        records = [_make_transaction_data(f"bulk-{i}") for i in range(100)]
        rows = await pg_handler.bulk_upsert_transactions(records)
        assert rows == 100

        # Verify count
        count = await pg_handler.count_transactions(customer_id="CUST-bulk-0")
        assert count == 1

    @pytest.mark.asyncio
    async def test_bulk_upsert_transactions_handles_conflicts(
        self, pg_handler: PostgresHandler
    ) -> None:
        records = [_make_transaction_data(f"conflict-{i}") for i in range(10)]
        await pg_handler.bulk_upsert_transactions(records)

        # Update status in records and upsert again
        for r in records:
            r["status"] = "approved"
        rows = await pg_handler.bulk_upsert_transactions(records)
        assert rows == 10

    @pytest.mark.asyncio
    async def test_bulk_upsert_1000_transactions_performance(
        self, pg_handler: PostgresHandler
    ) -> None:
        """Verify 1000 records bulk upsert completes in under 2 seconds."""
        records = [_make_transaction_data(f"perf-{i}") for i in range(1000)]

        start = time.perf_counter()
        rows = await pg_handler.bulk_upsert_transactions(records, batch_size=1000)
        elapsed = time.perf_counter() - start

        assert rows == 1000
        assert elapsed < 2.0, f"Bulk upsert took {elapsed:.2f}s, expected < 2s"

    @pytest.mark.asyncio
    async def test_bulk_insert_alerts(self, pg_handler: PostgresHandler) -> None:
        # Create transactions first
        txn_records = [_make_transaction_data(f"bulk-alert-{i}") for i in range(50)]
        await pg_handler.bulk_upsert_transactions(txn_records)

        # Get transaction IDs
        txns = await pg_handler.query_transactions(limit=50)
        alert_records = [_make_alert_data(t.transaction_id) for t in txns[:50]]
        rows = await pg_handler.bulk_insert_alerts(alert_records)
        assert rows == 50

    @pytest.mark.asyncio
    async def test_bulk_upsert_customer_profiles(self, pg_handler: PostgresHandler) -> None:
        records = [
            {
                "customer_id": f"CUST-BULK-{i}",
                "total_transactions_24h": i,
                "total_amount_24h": Decimal(str(i * 100)),
                "total_transactions_7d": i * 5,
                "total_amount_7d": Decimal(str(i * 500)),
                "unique_merchants_7d": i % 10,
                "unique_countries_7d": 1,
                "risk_tier": "standard",
            }
            for i in range(100)
        ]
        rows = await pg_handler.bulk_upsert_customer_profiles(records)
        assert rows == 100


# ---------------------------------------------------------------------------
# Aggregation & Analytics Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAggregations:
    @pytest.mark.asyncio
    async def test_get_transaction_stats(self, pg_handler: PostgresHandler) -> None:
        for i in range(5):
            data = _make_transaction_data(f"stats-{i}")
            data["customer_id"] = "CUST-STATS"
            data["transaction_amount"] = Decimal(str((i + 1) * 100))
            await pg_handler.create_transaction(data)

        stats = await pg_handler.get_transaction_stats("CUST-STATS")
        assert stats["total_count"] == 5
        assert stats["total_amount"] == 1500.0
        assert stats["avg_amount"] == 300.0
        assert stats["max_amount"] == 500.0
        assert stats["min_amount"] == 100.0

    @pytest.mark.asyncio
    async def test_get_alert_summary(self, pg_handler: PostgresHandler) -> None:
        txn = await pg_handler.create_transaction(_make_transaction_data("summary"))
        for severity in ["low", "medium", "high", "high", "critical"]:
            data = _make_alert_data(txn.transaction_id)
            data["severity"] = severity
            await pg_handler.create_alert(data)

        summary = await pg_handler.get_alert_summary()
        assert summary["by_severity"]["high"] == 2
        assert summary["by_severity"]["critical"] == 1
        assert summary["by_status"]["open"] == 5


# ---------------------------------------------------------------------------
# Transaction Management Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTransactionManagement:
    @pytest.mark.asyncio
    async def test_atomic_context_commits_on_success(self, pg_handler: PostgresHandler) -> None:
        async with pg_handler.atomic() as session:
            txn = Transaction(**_make_transaction_data("atomic-ok"))
            session.add(txn)

        # Verify committed
        fetched = await pg_handler.get_transaction_by_external_id(txn.external_transaction_id)
        assert fetched is not None

    @pytest.mark.asyncio
    async def test_atomic_context_rollbacks_on_error(self, pg_handler: PostgresHandler) -> None:
        ext_id = f"TXN-ATOMIC-FAIL-{uuid.uuid4().hex[:8]}"
        with pytest.raises(ValueError):
            async with pg_handler.atomic() as session:
                txn = Transaction(**_make_transaction_data("atomic-fail"))
                txn.external_transaction_id = ext_id
                session.add(txn)
                await session.flush()
                raise ValueError("Simulated failure")

        # Verify rolled back
        fetched = await pg_handler.get_transaction_by_external_id(ext_id)
        assert fetched is None

    @pytest.mark.asyncio
    async def test_session_rollback_on_constraint_violation(
        self, pg_handler: PostgresHandler
    ) -> None:
        data = _make_transaction_data("constraint")
        await pg_handler.create_transaction(data)

        # Attempt duplicate insert should raise
        with pytest.raises(Exception):
            await pg_handler.create_transaction(data)


# ---------------------------------------------------------------------------
# Query Performance Monitoring Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestQueryPerformanceMonitoring:
    @pytest.mark.asyncio
    async def test_metrics_recorded(self, pg_handler: PostgresHandler) -> None:
        await pg_handler.create_transaction(_make_transaction_data("metrics"))
        metrics = pg_handler.get_query_metrics(last_n=10)
        assert len(metrics) >= 1
        assert metrics[-1].query_name == "create_transaction"
        assert metrics[-1].duration_ms >= 0

    @pytest.mark.asyncio
    async def test_performance_summary(self, pg_handler: PostgresHandler) -> None:
        # Generate some queries
        for i in range(10):
            await pg_handler.create_transaction(_make_transaction_data(f"summary-{i}"))

        summary = pg_handler.get_performance_summary()
        assert summary["total_queries"] >= 10
        assert "avg_duration_ms" in summary
        assert "max_duration_ms" in summary
        assert "p95_duration_ms" in summary
        assert "create_transaction" in summary["by_query"]

    @pytest.mark.asyncio
    async def test_pool_stats_after_operations(self, pg_handler: PostgresHandler) -> None:
        await pg_handler.create_transaction(_make_transaction_data("pool"))
        stats = pg_handler.get_pool_stats()
        assert stats.pool_size == 5
