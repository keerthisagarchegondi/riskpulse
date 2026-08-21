"""Integration tests for Snowflake analytical data loading pipeline.

Tests cover:
- Connection management and pooling
- Stage-based data loading (PUT → COPY INTO)
- Incremental loading with watermarks
- RAW → STAGING → ANALYTICS → REPORTING pipeline
- SCD Type 2 for customer dimension
- Query caching
- Schema management
- Full pipeline orchestration
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.storage.snowflake_handler import (
    SCHEMA_ANALYTICS,
    SCHEMA_RAW,
    SCHEMA_REPORTING,
    SCHEMA_STAGING,
    FileFormat,
    LoadMetrics,
    LoadStrategy,
    QueryResult,
    SnowflakeConnectionError,
    SnowflakeHandler,
    SnowflakeLoadError,
    SnowflakeMetrics,
    SnowflakeQueryError,
    WatermarkState,
    create_snowflake_handler,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_snowflake_connector():
    """Mock snowflake.connector module."""
    with patch("src.storage.snowflake_handler.snowflake") as mock_sf:
        mock_conn = MagicMock()
        mock_conn.is_closed.return_value = False
        mock_sf.connector.connect.return_value = mock_conn
        yield mock_sf, mock_conn


@pytest.fixture
def handler(mock_snowflake_connector):
    """Create a SnowflakeHandler with mocked connection."""
    mock_sf, mock_conn = mock_snowflake_connector
    h = SnowflakeHandler(
        account="test_account",
        user="test_user",
        password="test_pass",
        warehouse="TEST_WH",
        database="TEST_DB",
        role="TEST_ROLE",
        pool_size=2,
        query_cache_ttl=60,
    )
    h.connect()
    return h


@pytest.fixture
def sample_transactions():
    """Sample transaction records for loading."""
    return [
        {
            "transaction_id": f"TXN-{uuid.uuid4().hex[:8]}",
            "external_transaction_id": f"EXT-{uuid.uuid4().hex[:8]}",
            "account_id": "ACC-12345",
            "customer_id": "CUST-67890",
            "merchant_id": "MERCH-11111",
            "merchant_name": "Test Merchant",
            "merchant_category_code": "5411",
            "transaction_amount": 150.00,
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
            "transaction_timestamp": "2026-07-01T10:30:00Z",
            "status": "completed",
        }
        for _ in range(5)
    ]


@pytest.fixture
def sample_customer_profiles():
    """Sample customer profile data for SCD Type 2 testing."""
    return [
        {
            "customer_id": "CUST-67890",
            "risk_tier": "medium",
            "total_transactions_7d": 15,
            "avg_transaction_amount": 125.50,
            "max_transaction_amount": 500.00,
        },
        {
            "customer_id": "CUST-11111",
            "risk_tier": "low",
            "total_transactions_7d": 3,
            "avg_transaction_amount": 45.00,
            "max_transaction_amount": 100.00,
        },
    ]


# =============================================================================
# CONNECTION MANAGEMENT TESTS
# =============================================================================


class TestSnowflakeConnection:
    """Test connection pool management."""

    def test_connect_creates_pool(self, mock_snowflake_connector):
        """Test that connect() creates connections up to pool size."""
        mock_sf, mock_conn = mock_snowflake_connector
        handler = SnowflakeHandler(
            account="test_acct",
            user="test_user",
            password="test_pass",
            pool_size=3,
        )
        handler.connect()

        assert handler.is_connected
        assert mock_sf.connector.connect.call_count == 3

    def test_connect_raises_without_credentials(self):
        """Test that connect() raises error when credentials are missing."""
        with patch("src.storage.snowflake_handler.snowflake"):
            handler = SnowflakeHandler(account="", user="")
            with pytest.raises(
                SnowflakeConnectionError, match="account and user must be configured"
            ):
                handler.connect()

    def test_connect_raises_when_snowflake_not_installed(self):
        """Test that connect() raises when snowflake-connector is not installed."""
        with patch("src.storage.snowflake_handler.snowflake", None):
            handler = SnowflakeHandler(account="test", user="test")
            with pytest.raises(SnowflakeConnectionError, match="not installed"):
                handler.connect()

    def test_close_releases_all_connections(self, handler, mock_snowflake_connector):
        """Test that close() properly releases all pooled connections."""
        handler.close()
        assert not handler.is_connected

    def test_connection_recovery_on_closed_connection(self, mock_snowflake_connector):
        """Test that a closed connection is replaced with a new one."""
        mock_sf, mock_conn = mock_snowflake_connector
        mock_conn.is_closed.return_value = True

        handler = SnowflakeHandler(
            account="test",
            user="test",
            password="pass",
            pool_size=1,
        )
        handler.connect()

        # Execute a query which requires getting a connection
        mock_cursor = MagicMock()
        mock_cursor.description = [("COL1",), ("COL2",)]
        mock_cursor.fetchall.return_value = [{"COL1": "val1", "COL2": "val2"}]
        mock_cursor.sfqid = "query-123"

        new_conn = MagicMock()
        new_conn.is_closed.return_value = False
        new_conn.cursor.return_value = mock_cursor
        mock_sf.connector.connect.return_value = new_conn

        result = handler.execute_query("SELECT 1")
        assert result.row_count == 1


# =============================================================================
# QUERY EXECUTION TESTS
# =============================================================================


class TestQueryExecution:
    """Test query execution and result handling."""

    def test_execute_query_returns_result(self, handler, mock_snowflake_connector):
        """Test basic query execution returns QueryResult."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("ID",), ("NAME",), ("VALUE",)]
        mock_cursor.fetchall.return_value = [
            {"ID": 1, "NAME": "test", "VALUE": 100},
            {"ID": 2, "NAME": "test2", "VALUE": 200},
        ]
        mock_cursor.sfqid = "qid-abc-123"
        mock_conn.cursor.return_value = mock_cursor

        result = handler.execute_query("SELECT * FROM test_table")

        assert isinstance(result, QueryResult)
        assert result.row_count == 2
        assert result.columns == ["ID", "NAME", "VALUE"]
        assert result.query_id == "qid-abc-123"
        assert result.duration_ms > 0
        assert not result.cached

    def test_execute_query_with_schema_context(self, handler, mock_snowflake_connector):
        """Test query execution sets schema context."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("COUNT",)]
        mock_cursor.fetchall.return_value = [{"COUNT": 42}]
        mock_cursor.sfqid = "qid-456"
        mock_conn.cursor.return_value = mock_cursor

        handler.execute_query("SELECT COUNT(*) FROM t", schema="ANALYTICS")

        # Should have called USE SCHEMA
        calls = mock_cursor.execute.call_args_list
        assert any("USE SCHEMA" in str(c) for c in calls)

    def test_execute_query_caching(self, handler, mock_snowflake_connector):
        """Test that query results are cached when use_cache=True."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("RESULT",)]
        mock_cursor.fetchall.return_value = [{"RESULT": "cached_value"}]
        mock_cursor.sfqid = "qid-cache-1"
        mock_conn.cursor.return_value = mock_cursor

        # First call - cache miss
        result1 = handler.execute_query("SELECT 'cached_value'", use_cache=True)
        assert not result1.cached
        assert handler.metrics.cache_misses == 1

        # Second call - cache hit
        result2 = handler.execute_query("SELECT 'cached_value'", use_cache=True)
        assert result2.cached
        assert handler.metrics.cache_hits == 1
        assert result2.rows == result1.rows

    def test_execute_query_raises_on_programming_error(self, handler, mock_snowflake_connector):
        """Test that ProgrammingError is wrapped in SnowflakeQueryError."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()

        from src.storage.snowflake_handler import ProgrammingError

        mock_cursor.execute.side_effect = ProgrammingError("Syntax error")
        mock_conn.cursor.return_value = mock_cursor

        with pytest.raises(SnowflakeQueryError, match="Query execution failed"):
            handler.execute_query("INVALID SQL")

    def test_execute_non_query_returns_rows_affected(self, handler, mock_snowflake_connector):
        """Test DML execution returns row count."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 5
        mock_conn.cursor.return_value = mock_cursor

        rows = handler.execute_non_query("UPDATE t SET col=1")
        assert rows == 5

    def test_call_procedure(self, handler, mock_snowflake_connector):
        """Test stored procedure invocation."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("RESULT",)]
        mock_cursor.fetchall.return_value = [
            {"RESULT": "Successfully loaded 100 transactions for batch BATCH-001"}
        ]
        mock_cursor.sfqid = "qid-proc-1"
        mock_conn.cursor.return_value = mock_cursor

        result = handler.call_procedure("STAGING.LOAD_TRANSACTIONS", ["BATCH-001"])
        assert "100" in result
        assert "BATCH-001" in result


# =============================================================================
# DATA LOADING TESTS
# =============================================================================


class TestDataLoading:
    """Test stage-based and PUT/COPY data loading."""

    def test_load_from_s3_stage(self, handler, mock_snowflake_connector):
        """Test loading data from S3 external stage."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("rows_loaded",), ("errors_seen",)]
        mock_cursor.fetchall.return_value = [{"rows_loaded": 1000, "errors_seen": 2}]
        mock_cursor.sfqid = "qid-load-1"
        mock_conn.cursor.return_value = mock_cursor

        metrics = handler.load_from_s3_stage(
            table_name="TRANSACTIONS",
            stage_name="RAW.S3_STAGE",
            file_pattern=".*2026-07-01.*",
            batch_id="BATCH-S3-001",
        )

        assert isinstance(metrics, LoadMetrics)
        assert metrics.batch_id == "BATCH-S3-001"
        assert metrics.rows_loaded == 1000
        assert metrics.rows_rejected == 2
        assert metrics.target_schema == SCHEMA_RAW
        assert metrics.strategy == LoadStrategy.APPEND

    def test_load_from_s3_stage_with_file_format(self, handler, mock_snowflake_connector):
        """Test S3 stage loading with JSON format."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("rows_loaded",), ("errors_seen",)]
        mock_cursor.fetchall.return_value = [{"rows_loaded": 500, "errors_seen": 0}]
        mock_cursor.sfqid = "qid-json-1"
        mock_conn.cursor.return_value = mock_cursor

        metrics = handler.load_from_s3_stage(
            table_name="TRANSACTIONS",
            file_format=FileFormat.JSON,
        )
        assert metrics.rows_loaded == 500

    def test_put_and_copy_with_data(self, handler, mock_snowflake_connector, sample_transactions):
        """Test PUT/COPY loading pattern with actual data."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.rowcount = len(sample_transactions)
        mock_conn.cursor.return_value = mock_cursor

        metrics = handler.put_and_copy(
            data=sample_transactions,
            target_table="TRANSACTIONS",
            schema=SCHEMA_RAW,
            batch_id="BATCH-PUT-001",
        )

        assert metrics.batch_id == "BATCH-PUT-001"
        assert metrics.rows_loaded == len(sample_transactions)
        assert metrics.target_schema == SCHEMA_RAW
        assert metrics.duration_ms > 0

    def test_put_and_copy_empty_data(self, handler):
        """Test PUT/COPY with empty data returns zero metrics."""
        metrics = handler.put_and_copy(
            data=[],
            target_table="TRANSACTIONS",
            batch_id="BATCH-EMPTY",
        )
        assert metrics.rows_loaded == 0
        assert metrics.duration_ms == 0

    def test_load_from_s3_stage_error_handling(self, handler, mock_snowflake_connector):
        """Test that S3 stage load errors are properly wrapped."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()

        from src.storage.snowflake_handler import ProgrammingError

        mock_cursor.execute.side_effect = ProgrammingError("Stage not found")
        mock_conn.cursor.return_value = mock_cursor

        with pytest.raises(SnowflakeLoadError, match="Failed to load"):
            handler.load_from_s3_stage(table_name="TRANSACTIONS")


# =============================================================================
# INCREMENTAL LOADING TESTS
# =============================================================================


class TestIncrementalLoading:
    """Test watermark-based incremental loading."""

    def test_get_watermark_from_cache(self, handler):
        """Test watermark retrieval from in-memory cache."""
        handler._watermarks["STAGING.STG_TRANSACTIONS.LOAD_TIMESTAMP"] = WatermarkState(
            table_name="STAGING.STG_TRANSACTIONS",
            column_name="LOAD_TIMESTAMP",
            last_value="2026-07-01T00:00:00Z",
        )

        watermark = handler.get_watermark("STAGING.STG_TRANSACTIONS", "LOAD_TIMESTAMP")
        assert watermark == "2026-07-01T00:00:00Z"

    def test_get_watermark_from_snowflake(self, handler, mock_snowflake_connector):
        """Test watermark retrieval from Snowflake metadata table."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("LAST_VALUE",)]
        mock_cursor.fetchall.return_value = [{"LAST_VALUE": "2026-06-30T23:00:00Z"}]
        mock_cursor.sfqid = "qid-wm-1"
        mock_conn.cursor.return_value = mock_cursor

        watermark = handler.get_watermark("STAGING.STG_TRANSACTIONS", "LOAD_TIMESTAMP")
        assert watermark == "2026-06-30T23:00:00Z"

    def test_get_watermark_returns_none_for_new_table(self, handler, mock_snowflake_connector):
        """Test that None is returned for tables without watermarks."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("LAST_VALUE",)]
        mock_cursor.fetchall.return_value = []
        mock_cursor.sfqid = "qid-wm-2"
        mock_conn.cursor.return_value = mock_cursor

        watermark = handler.get_watermark("NEW_TABLE", "LOAD_TIMESTAMP")
        assert watermark is None

    def test_update_watermark(self, handler, mock_snowflake_connector):
        """Test watermark update persists to Snowflake and cache."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn.cursor.return_value = mock_cursor

        handler.update_watermark(
            "STAGING.STG_TRANSACTIONS",
            "LOAD_TIMESTAMP",
            "2026-07-01T12:00:00Z",
        )

        key = "STAGING.STG_TRANSACTIONS.LOAD_TIMESTAMP"
        assert key in handler._watermarks
        assert handler._watermarks[key].last_value == "2026-07-01T12:00:00Z"

    def test_load_incremental(self, handler, mock_snowflake_connector):
        """Test incremental loading applies watermark filter."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 50
        mock_cursor.description = [("MAX_VAL",)]
        mock_cursor.fetchall.return_value = [{"MAX_VAL": "2026-07-01T15:00:00Z"}]
        mock_cursor.sfqid = "qid-incr-1"
        mock_conn.cursor.return_value = mock_cursor

        # Set initial watermark
        handler._watermarks["STAGING.STG_TRANSACTIONS.LOAD_TIMESTAMP"] = WatermarkState(
            table_name="STAGING.STG_TRANSACTIONS",
            column_name="LOAD_TIMESTAMP",
            last_value="2026-07-01T00:00:00Z",
        )

        metrics = handler.load_incremental(
            source_table="TRANSACTIONS",
            target_table="STG_TRANSACTIONS",
            watermark_column="LOAD_TIMESTAMP",
            batch_id="BATCH-INCR-001",
        )

        assert metrics.rows_loaded == 50
        assert metrics.strategy == LoadStrategy.INCREMENTAL


# =============================================================================
# PIPELINE ORCHESTRATION TESTS
# =============================================================================


class TestPipelineOrchestration:
    """Test full ETL pipeline flow."""

    def test_process_raw_to_staging(self, handler, mock_snowflake_connector):
        """Test RAW → STAGING transformation step."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("RESULT",)]
        mock_cursor.fetchall.return_value = [
            {"RESULT": "Successfully loaded 200 records for batch BATCH-001"}
        ]
        mock_cursor.sfqid = "qid-r2s-1"
        mock_conn.cursor.return_value = mock_cursor

        results = handler.process_raw_to_staging("BATCH-001")

        assert "STG_TRANSACTIONS" in results
        assert "STG_FRAUD_ALERTS" in results
        assert "STG_RISK_SCORES" in results
        assert all(m.source_schema == SCHEMA_RAW for m in results.values())
        assert all(m.target_schema == SCHEMA_STAGING for m in results.values())

    def test_process_staging_to_analytics(self, handler, mock_snowflake_connector):
        """Test STAGING → ANALYTICS transformation step."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("RESULT",)]
        mock_cursor.fetchall.return_value = [
            {"RESULT": "Loaded 150 rows into target for batch BATCH-002"}
        ]
        mock_cursor.sfqid = "qid-s2a-1"
        mock_cursor.rowcount = 10
        mock_conn.cursor.return_value = mock_cursor

        results = handler.process_staging_to_analytics("BATCH-002")

        assert "DIM_CUSTOMER" in results
        assert "DIM_MERCHANT" in results
        assert "DIM_GEOGRAPHY" in results
        assert "FACT_TRANSACTIONS" in results
        assert all(m.target_schema == SCHEMA_ANALYTICS for m in results.values())

    def test_refresh_reporting(self, handler, mock_snowflake_connector):
        """Test ANALYTICS → REPORTING aggregation refresh."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("RESULT",)]
        mock_cursor.fetchall.return_value = [
            {"RESULT": "All reporting tables refreshed for 2026-07-01"}
        ]
        mock_cursor.sfqid = "qid-rpt-1"
        mock_conn.cursor.return_value = mock_cursor

        results = handler.refresh_reporting("2026-07-01")

        assert "REPORTING_ALL" in results
        assert results["REPORTING_ALL"].target_schema == SCHEMA_REPORTING

    def test_run_full_pipeline(self, handler, mock_snowflake_connector):
        """Test complete pipeline execution: RAW → STAGING → ANALYTICS → REPORTING."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("RESULT",)]
        mock_cursor.fetchall.return_value = [
            {"RESULT": "Successfully processed 100 records for batch FULL-001"}
        ]
        mock_cursor.sfqid = "qid-full-1"
        mock_cursor.rowcount = 10
        mock_conn.cursor.return_value = mock_cursor

        results = handler.run_full_pipeline("FULL-001", summary_date="2026-07-01")

        assert "raw_to_staging" in results
        assert "staging_to_analytics" in results
        assert "reporting" in results

        # Verify all layers were processed
        assert len(results["raw_to_staging"]) == 3  # transactions, alerts, scores
        assert len(results["staging_to_analytics"]) == 4  # dims + fact
        assert len(results["reporting"]) == 1


# =============================================================================
# SCD TYPE 2 TESTS
# =============================================================================


class TestSCDType2:
    """Test Slowly Changing Dimension Type 2 logic."""

    def test_apply_scd_type2(self, handler, mock_snowflake_connector):
        """Test SCD Type 2 closes old records and inserts new ones."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 3
        mock_conn.cursor.return_value = mock_cursor

        inserted = handler.apply_scd_type2(
            staging_table="STG_CUSTOMER_PROFILES",
            dimension_table="DIM_CUSTOMER",
            business_key="CUSTOMER_ID",
            tracked_columns=["RISK_TIER", "AVG_TRANSACTION_AMOUNT"],
            batch_id="BATCH-SCD-001",
        )

        assert inserted == 3
        # Verify UPDATE (close) and INSERT (new version) were called
        execute_calls = mock_cursor.execute.call_args_list
        assert len(execute_calls) >= 2

    def test_apply_scd_type2_no_changes(self, handler, mock_snowflake_connector):
        """Test SCD Type 2 with no dimension changes."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn.cursor.return_value = mock_cursor

        inserted = handler.apply_scd_type2(
            staging_table="STG_CUSTOMER_PROFILES",
            dimension_table="DIM_CUSTOMER",
            business_key="CUSTOMER_ID",
            tracked_columns=["RISK_TIER"],
            batch_id="BATCH-NO-CHANGE",
        )

        assert inserted == 0


# =============================================================================
# SCHEMA MANAGEMENT TESTS
# =============================================================================


class TestSchemaManagement:
    """Test schema and table management operations."""

    def test_ensure_schema_exists(self, handler, mock_snowflake_connector):
        """Test schema creation."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn.cursor.return_value = mock_cursor

        handler.ensure_schema_exists("NEW_SCHEMA")
        mock_cursor.execute.assert_called()

    def test_initialize_schemas(self, handler, mock_snowflake_connector):
        """Test that all four schemas are initialized."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn.cursor.return_value = mock_cursor

        handler.initialize_schemas()

        execute_calls = [str(c) for c in mock_cursor.execute.call_args_list]
        # Should create RAW, STAGING, ANALYTICS, REPORTING schemas + watermarks table
        assert len(execute_calls) >= 5

    def test_get_table_columns(self, handler, mock_snowflake_connector):
        """Test table column metadata retrieval."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ("COLUMN_NAME",),
            ("DATA_TYPE",),
            ("IS_NULLABLE",),
            ("CHARACTER_MAXIMUM_LENGTH",),
            ("NUMERIC_PRECISION",),
            ("NUMERIC_SCALE",),
        ]
        mock_cursor.fetchall.return_value = [
            {
                "COLUMN_NAME": "TRANSACTION_ID",
                "DATA_TYPE": "VARCHAR",
                "IS_NULLABLE": "NO",
                "CHARACTER_MAXIMUM_LENGTH": 64,
                "NUMERIC_PRECISION": None,
                "NUMERIC_SCALE": None,
            },
            {
                "COLUMN_NAME": "AMOUNT",
                "DATA_TYPE": "NUMBER",
                "IS_NULLABLE": "NO",
                "CHARACTER_MAXIMUM_LENGTH": None,
                "NUMERIC_PRECISION": 15,
                "NUMERIC_SCALE": 2,
            },
        ]
        mock_cursor.sfqid = "qid-cols-1"
        mock_conn.cursor.return_value = mock_cursor

        columns = handler.get_table_columns("STAGING", "STG_TRANSACTIONS")
        assert len(columns) == 2
        assert columns[0]["COLUMN_NAME"] == "TRANSACTION_ID"
        assert columns[1]["DATA_TYPE"] == "NUMBER"

    def test_add_column(self, handler, mock_snowflake_connector):
        """Test adding a column to existing table."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn.cursor.return_value = mock_cursor

        handler.add_column(
            schema="STAGING",
            table_name="STG_TRANSACTIONS",
            column_name="NEW_FLAG",
            data_type="BOOLEAN",
            default="FALSE",
        )
        mock_cursor.execute.assert_called()


# =============================================================================
# METRICS AND MONITORING TESTS
# =============================================================================


class TestMetrics:
    """Test metrics collection and reporting."""

    def test_metrics_tracking(self, handler, mock_snowflake_connector):
        """Test that metrics are properly tracked across operations."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("RESULT",)]
        mock_cursor.fetchall.return_value = [{"RESULT": 1}]
        mock_cursor.sfqid = "qid-met-1"
        mock_conn.cursor.return_value = mock_cursor

        handler.execute_query("SELECT 1")
        handler.execute_query("SELECT 2")

        snapshot = handler.metrics.snapshot()
        assert snapshot["queries_executed"] == 2
        assert snapshot["total_query_time_ms"] > 0
        assert snapshot["avg_query_time_ms"] > 0

    def test_metrics_error_tracking(self, handler, mock_snowflake_connector):
        """Test that query errors are counted."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()

        from src.storage.snowflake_handler import ProgrammingError

        mock_cursor.execute.side_effect = ProgrammingError("bad query")
        mock_conn.cursor.return_value = mock_cursor

        with pytest.raises(SnowflakeQueryError):
            handler.execute_query("BAD SQL")

        assert handler.metrics.queries_failed >= 1

    def test_load_metrics_dataclass(self):
        """Test LoadMetrics dataclass initialization."""
        metrics = LoadMetrics(
            batch_id="test-batch",
            source_schema="RAW",
            target_schema="STAGING",
            table_name="STG_TRANSACTIONS",
            rows_loaded=1000,
            rows_rejected=5,
            duration_ms=2500.0,
            strategy=LoadStrategy.INCREMENTAL,
        )
        assert metrics.rows_loaded == 1000
        assert metrics.strategy == LoadStrategy.INCREMENTAL
        assert metrics.timestamp is not None

    def test_snowflake_metrics_thread_safety(self):
        """Test that metrics are thread-safe."""
        import threading

        metrics = SnowflakeMetrics()

        def increment_queries():
            for _ in range(100):
                metrics.record_query(1.0)

        threads = [threading.Thread(target=increment_queries) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.queries_executed == 1000
        assert metrics.total_query_time_ms == 1000.0


# =============================================================================
# DYNAMIC TABLE / MATERIALIZED VIEW TESTS
# =============================================================================


class TestDynamicTables:
    """Test dynamic table (materialized view) management."""

    def test_create_dynamic_table(self, handler, mock_snowflake_connector):
        """Test dynamic table creation."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn.cursor.return_value = mock_cursor

        handler.create_or_replace_dynamic_table(
            schema="REPORTING",
            table_name="DT_TEST",
            query="SELECT * FROM ANALYTICS.FACT_TRANSACTIONS",
            lag="1 hour",
        )
        mock_cursor.execute.assert_called()

    def test_setup_reporting_dynamic_tables(self, handler, mock_snowflake_connector):
        """Test that all reporting dynamic tables are set up."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn.cursor.return_value = mock_cursor

        handler.setup_reporting_dynamic_tables()

        # Should have created at least 3 dynamic tables
        execute_calls = mock_cursor.execute.call_args_list
        dynamic_table_calls = [c for c in execute_calls if "DYNAMIC TABLE" in str(c)]
        assert len(dynamic_table_calls) >= 3


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactory:
    """Test factory function."""

    def test_create_snowflake_handler(self):
        """Test factory function creates handler with defaults."""
        with patch.dict(
            os.environ,
            {
                "SNOWFLAKE_ACCOUNT": "test_account",
                "SNOWFLAKE_USER": "test_user",
                "SNOWFLAKE_PASSWORD": "test_pass",
            },
        ):
            handler = create_snowflake_handler(pool_size=3, query_cache_ttl=120)
            assert handler._pool_size == 3
            assert handler._query_cache_ttl == 120
            assert handler._account == "test_account"


# =============================================================================
# PERFORMANCE / WAREHOUSE TESTS
# =============================================================================


class TestPerformanceUtilities:
    """Test performance and warehouse management utilities."""

    def test_get_load_history(self, handler, mock_snowflake_connector):
        """Test load history retrieval."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.description = [("TABLE_NAME",), ("ROWS_LOADED",), ("LAST_LOAD_TIME",)]
        mock_cursor.fetchall.return_value = [
            {"TABLE_NAME": "TRANSACTIONS", "ROWS_LOADED": 5000, "LAST_LOAD_TIME": "2026-07-01"},
        ]
        mock_cursor.sfqid = "qid-hist-1"
        mock_conn.cursor.return_value = mock_cursor

        history = handler.get_load_history("TRANSACTIONS", days=7)
        assert len(history) == 1
        assert history[0]["ROWS_LOADED"] == 5000

    def test_suspend_warehouse(self, handler, mock_snowflake_connector):
        """Test warehouse suspension."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn.cursor.return_value = mock_cursor

        handler.suspend_warehouse()
        mock_cursor.execute.assert_called()

    def test_resume_warehouse(self, handler, mock_snowflake_connector):
        """Test warehouse resumption."""
        _, mock_conn = mock_snowflake_connector
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn.cursor.return_value = mock_cursor

        handler.resume_warehouse()
        mock_cursor.execute.assert_called()
