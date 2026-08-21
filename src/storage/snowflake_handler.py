"""Snowflake analytical data warehouse handler.

Production-grade Snowflake handler providing:
- Connection pooling with automatic reconnection
- Stage-based data loading (PUT → COPY INTO)
- Incremental loading with timestamp-based watermarks
- Schema management (create/alter tables dynamically)
- Query execution with result caching
- Four-layer architecture: RAW → STAGING → ANALYTICS → REPORTING
- SCD Type 2 for slowly changing dimensions
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Generator

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    import snowflake.connector
    from snowflake.connector import DictCursor, SnowflakeConnection
    from snowflake.connector.errors import (
        DatabaseError,
        InterfaceError,
        OperationalError,
        ProgrammingError,
    )
except ImportError:
    snowflake = None  # type: ignore[assignment]

    # Define stub exception classes so the module can be imported without snowflake installed
    class OperationalError(Exception):  # type: ignore[no-redef]
        pass

    class InterfaceError(Exception):  # type: ignore[no-redef]
        pass

    class ProgrammingError(Exception):  # type: ignore[no-redef]
        pass

    class DatabaseError(Exception):  # type: ignore[no-redef]
        pass

    class DictCursor:  # type: ignore[no-redef]
        pass

    class SnowflakeConnection:  # type: ignore[no-redef]
        pass


logger = structlog.get_logger(__name__)


# Schema layer constants
SCHEMA_RAW = "RAW"
SCHEMA_STAGING = "STAGING"
SCHEMA_ANALYTICS = "ANALYTICS"
SCHEMA_REPORTING = "REPORTING"

_SNOWFLAKE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_TRANSFORM_SQL_BLOCKLIST_RE = re.compile(
    r"\b(ALTER|CALL|COPY|CREATE|DELETE|DROP|GRANT|INSERT|MERGE|REVOKE|TRUNCATE|UPDATE)\b",
    re.IGNORECASE,
)


def _validate_identifier(identifier: str, *, name: str = "identifier") -> str:
    """Validate a Snowflake identifier before interpolating it into SQL."""
    if not isinstance(identifier, str) or not _SNOWFLAKE_IDENTIFIER_RE.fullmatch(identifier):
        raise SnowflakeQueryError(f"Unsafe Snowflake {name}: {identifier!r}")
    return identifier


def _validate_qualified_name(schema: str, table: str) -> str:
    return (
        f"{_validate_identifier(schema, name='schema')}.{_validate_identifier(table, name='table')}"
    )


def _validate_transform_sql(transform_sql: str | None) -> str:
    if transform_sql is None:
        return "*"

    normalized = transform_sql.strip()
    if not normalized:
        raise SnowflakeQueryError("Transform SQL cannot be empty")
    if any(token in normalized for token in (";", "--", "/*", "*/")):
        raise SnowflakeQueryError("Transform SQL contains unsafe SQL delimiters")
    if _TRANSFORM_SQL_BLOCKLIST_RE.search(normalized):
        raise SnowflakeQueryError("Transform SQL must be a SELECT expression only")
    return normalized


def _bounded_int(value: int, *, name: str, minimum: int, maximum: int) -> int:
    bounded = int(value)
    if bounded < minimum or bounded > maximum:
        raise SnowflakeQueryError(f"{name} must be between {minimum} and {maximum}")
    return bounded


class LoadStrategy(str, Enum):
    """Data loading strategy."""

    FULL = "full"
    INCREMENTAL = "incremental"
    APPEND = "append"


class FileFormat(str, Enum):
    """Supported file formats for stage loading."""

    PARQUET = "parquet"
    JSON = "json"
    CSV = "csv"


class SnowflakeHandlerError(Exception):
    """Base exception for Snowflake handler errors."""


class SnowflakeConnectionError(SnowflakeHandlerError):
    """Raised when Snowflake connection fails."""


class SnowflakeQueryError(SnowflakeHandlerError):
    """Raised when a query execution fails."""


class SnowflakeLoadError(SnowflakeHandlerError):
    """Raised when data loading fails."""


class SnowflakeSchemaError(SnowflakeHandlerError):
    """Raised when schema management operations fail."""


@dataclass
class LoadMetrics:
    """Metrics for a data loading operation."""

    batch_id: str
    source_schema: str
    target_schema: str
    table_name: str
    rows_loaded: int
    rows_rejected: int
    duration_ms: float
    strategy: LoadStrategy
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class QueryResult:
    """Wraps a Snowflake query result with metadata."""

    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    duration_ms: float
    query_id: str
    cached: bool = False


@dataclass
class WatermarkState:
    """Tracks incremental loading watermark."""

    table_name: str
    column_name: str
    last_value: str
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SnowflakeMetrics:
    """Thread-safe metrics for Snowflake operations."""

    queries_executed: int = 0
    queries_failed: int = 0
    rows_loaded: int = 0
    rows_rejected: int = 0
    bytes_staged: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_query_time_ms: float = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record_query(self, duration_ms: float) -> None:
        with self._lock:
            self.queries_executed += 1
            self.total_query_time_ms += duration_ms

    def record_query_error(self) -> None:
        with self._lock:
            self.queries_failed += 1

    def record_load(self, rows: int, rejected: int, bytes_size: int) -> None:
        with self._lock:
            self.rows_loaded += rows
            self.rows_rejected += rejected
            self.bytes_staged += bytes_size

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self.cache_misses += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queries_executed": self.queries_executed,
                "queries_failed": self.queries_failed,
                "rows_loaded": self.rows_loaded,
                "rows_rejected": self.rows_rejected,
                "bytes_staged": self.bytes_staged,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "total_query_time_ms": self.total_query_time_ms,
                "avg_query_time_ms": (
                    self.total_query_time_ms / self.queries_executed
                    if self.queries_executed > 0
                    else 0
                ),
            }


class SnowflakeHandler:
    """Production Snowflake data warehouse handler.

    Manages the four-layer analytical architecture:
    - RAW: Landing zone for semi-structured data (VARIANT)
    - STAGING: Parsed and typed data with validation
    - ANALYTICS: Star schema with fact and dimension tables
    - REPORTING: Pre-aggregated materialized views

    Usage:
        handler = SnowflakeHandler()
        handler.connect()

        # Stage-based loading
        batch_id = handler.load_from_s3_stage("transactions", "2026-07-01")

        # Incremental processing
        handler.process_raw_to_staging(batch_id)
        handler.process_staging_to_analytics(batch_id)
        handler.refresh_reporting("2026-07-01")

        handler.close()
    """

    def __init__(
        self,
        account: str | None = None,
        user: str | None = None,
        password: str | None = None,
        private_key_path: str | None = None,
        warehouse: str | None = None,
        database: str | None = None,
        role: str | None = None,
        pool_size: int = 5,
        query_cache_ttl: int = 300,
    ) -> None:
        self._account = account or os.environ.get("SNOWFLAKE_ACCOUNT", "")
        self._user = user or os.environ.get("SNOWFLAKE_USER", "")
        self._password = password or os.environ.get("SNOWFLAKE_PASSWORD")
        self._private_key_path = private_key_path or os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH")
        self._warehouse = warehouse or os.environ.get("SNOWFLAKE_WAREHOUSE", "RISKPULSE_WH")
        self._database = database or os.environ.get("SNOWFLAKE_DATABASE", "RISKPULSE")
        self._role = role or os.environ.get("SNOWFLAKE_ROLE", "RISKPULSE_ROLE")
        self._pool_size = pool_size
        self._query_cache_ttl = query_cache_ttl

        self._pool: list[SnowflakeConnection] = []
        self._pool_lock = Lock()
        self._metrics = SnowflakeMetrics()
        self._query_cache: dict[str, tuple[QueryResult, float]] = {}
        self._watermarks: dict[str, WatermarkState] = {}
        self._connected = False

    @property
    def metrics(self) -> SnowflakeMetrics:
        return self._metrics

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """Initialize connection pool to Snowflake."""
        if snowflake is None:
            raise SnowflakeConnectionError(
                "snowflake-connector-python is not installed. "
                "Install with: pip install snowflake-connector-python"
            )

        if not self._account or not self._user:
            raise SnowflakeConnectionError(
                "Snowflake account and user must be configured via environment "
                "variables (SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER) or constructor params."
            )

        try:
            for _ in range(self._pool_size):
                conn = self._create_connection()
                self._pool.append(conn)
            self._connected = True
            logger.info(
                "snowflake_connected",
                account=self._account,
                warehouse=self._warehouse,
                database=self._database,
                pool_size=self._pool_size,
            )
        except Exception as e:
            raise SnowflakeConnectionError(f"Failed to connect to Snowflake: {e}") from e

    def _create_connection(self) -> SnowflakeConnection:
        """Create a single Snowflake connection."""
        connect_params: dict[str, Any] = {
            "account": self._account,
            "user": self._user,
            "warehouse": self._warehouse,
            "database": self._database,
            "role": self._role,
            "client_session_keep_alive": True,
            "network_timeout": 30,
            "login_timeout": 15,
        }

        if self._private_key_path:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization

            key_path = Path(self._private_key_path)
            with open(key_path, "rb") as key_file:
                p_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "").encode()
                    or None,
                    backend=default_backend(),
                )
            pkb = p_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            connect_params["private_key"] = pkb
        elif self._password:
            connect_params["password"] = self._password

        return snowflake.connector.connect(**connect_params)

    @contextmanager
    def _get_connection(self) -> Generator[SnowflakeConnection, None, None]:
        """Acquire a connection from the pool."""
        conn = None
        with self._pool_lock:
            if self._pool:
                conn = self._pool.pop()

        if conn is None:
            conn = self._create_connection()

        try:
            if conn.is_closed():
                conn = self._create_connection()
            yield conn
        except (InterfaceError, OperationalError):
            conn = self._create_connection()
            yield conn
        finally:
            with self._pool_lock:
                if len(self._pool) < self._pool_size:
                    self._pool.append(conn)
                else:
                    try:
                        conn.close()
                    except Exception:
                        pass

    def close(self) -> None:
        """Close all connections in the pool."""
        with self._pool_lock:
            for conn in self._pool:
                try:
                    conn.close()
                except Exception:
                    pass
            self._pool.clear()
        self._connected = False
        logger.info("snowflake_disconnected")

    # =========================================================================
    # QUERY EXECUTION
    # =========================================================================

    @retry(
        retry=retry_if_exception_type((OperationalError, InterfaceError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        use_cache: bool = False,
        schema: str | None = None,
    ) -> QueryResult:
        """Execute a SQL query against Snowflake.

        Args:
            query: SQL query string
            params: Query parameters for binding
            use_cache: Whether to use query result caching
            schema: Target schema context for the query

        Returns:
            QueryResult with rows and metadata
        """
        cache_key = None
        if use_cache:
            cache_key = self._compute_cache_key(query, params)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                self._metrics.record_cache_hit()
                return cached
            self._metrics.record_cache_miss()

        start_time = time.perf_counter()

        try:
            with self._get_connection() as conn:
                if schema:
                    conn.cursor().execute(f"USE SCHEMA {self._database}.{schema}")

                cursor = conn.cursor(DictCursor)
                cursor.execute(query, params or {})

                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                query_id = cursor.sfqid or ""

                duration_ms = (time.perf_counter() - start_time) * 1000
                self._metrics.record_query(duration_ms)

                result = QueryResult(
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    duration_ms=duration_ms,
                    query_id=query_id,
                )

                if cache_key:
                    self._put_in_cache(cache_key, result)

                logger.debug(
                    "snowflake_query_executed",
                    query_id=query_id,
                    duration_ms=round(duration_ms, 2),
                    row_count=len(rows),
                )

                return result

        except ProgrammingError as e:
            self._metrics.record_query_error()
            raise SnowflakeQueryError(f"Query execution failed: {e}") from e
        except DatabaseError as e:
            self._metrics.record_query_error()
            raise SnowflakeQueryError(f"Database error: {e}") from e

    def execute_non_query(
        self,
        statement: str,
        params: dict[str, Any] | None = None,
        schema: str | None = None,
    ) -> int:
        """Execute a DML statement (INSERT, UPDATE, DELETE, MERGE).

        Returns:
            Number of rows affected
        """
        start_time = time.perf_counter()

        try:
            with self._get_connection() as conn:
                if schema:
                    conn.cursor().execute(f"USE SCHEMA {self._database}.{schema}")

                cursor = conn.cursor()
                cursor.execute(statement, params or {})
                rows_affected = cursor.rowcount or 0

                duration_ms = (time.perf_counter() - start_time) * 1000
                self._metrics.record_query(duration_ms)

                logger.debug(
                    "snowflake_dml_executed",
                    rows_affected=rows_affected,
                    duration_ms=round(duration_ms, 2),
                )

                return rows_affected

        except (ProgrammingError, DatabaseError) as e:
            self._metrics.record_query_error()
            raise SnowflakeQueryError(f"DML execution failed: {e}") from e

    def call_procedure(
        self,
        procedure_name: str,
        args: list[Any] | None = None,
        schema: str | None = None,
    ) -> str:
        """Call a Snowflake stored procedure.

        Args:
            procedure_name: Fully qualified or relative procedure name
            args: Procedure arguments
            schema: Schema context

        Returns:
            Procedure return value as string
        """
        args_str = ", ".join(f"'{a}'" if isinstance(a, str) else str(a) for a in (args or []))
        query = f"CALL {procedure_name}({args_str})"

        result = self.execute_query(query, schema=schema)
        if result.rows:
            first_row = result.rows[0]
            return str(list(first_row.values())[0]) if first_row else ""
        return ""

    # =========================================================================
    # STAGE-BASED DATA LOADING
    # =========================================================================

    def load_from_s3_stage(
        self,
        table_name: str,
        stage_name: str = "RAW.S3_STAGE",
        file_pattern: str | None = None,
        file_format: FileFormat = FileFormat.PARQUET,
        batch_id: str | None = None,
    ) -> LoadMetrics:
        """Load data from an S3 external stage into RAW schema.

        Uses COPY INTO for efficient bulk loading from external stage.

        Args:
            table_name: Target table in RAW schema
            stage_name: External stage name
            file_pattern: Regex pattern to match files in stage
            file_format: File format specification
            batch_id: Optional batch identifier

        Returns:
            LoadMetrics with loading statistics
        """
        batch_id = batch_id or uuid.uuid4().hex[:16]
        start_time = time.perf_counter()

        format_name = f"RAW.{file_format.value.upper()}_FORMAT"

        copy_query = f"""
            COPY INTO {SCHEMA_RAW}.{table_name}
            FROM @{stage_name}
            FILE_FORMAT = (FORMAT_NAME = '{format_name}')
            ON_ERROR = 'CONTINUE'
            PURGE = FALSE
        """

        if file_pattern:
            copy_query += f"\n            PATTERN = '{file_pattern}'"

        try:
            result = self.execute_query(copy_query)
            rows_loaded = 0
            rows_rejected = 0

            for row in result.rows:
                rows_loaded += row.get("rows_loaded", 0)
                rows_rejected += row.get("errors_seen", 0)

            duration_ms = (time.perf_counter() - start_time) * 1000
            self._metrics.record_load(rows_loaded, rows_rejected, 0)

            metrics = LoadMetrics(
                batch_id=batch_id,
                source_schema="S3_STAGE",
                target_schema=SCHEMA_RAW,
                table_name=table_name,
                rows_loaded=rows_loaded,
                rows_rejected=rows_rejected,
                duration_ms=duration_ms,
                strategy=LoadStrategy.APPEND,
            )

            logger.info(
                "snowflake_s3_load_complete",
                table=table_name,
                batch_id=batch_id,
                rows_loaded=rows_loaded,
                rows_rejected=rows_rejected,
                duration_ms=round(duration_ms, 2),
            )

            return metrics

        except SnowflakeQueryError as e:
            raise SnowflakeLoadError(
                f"Failed to load from stage {stage_name} into {table_name}: {e}"
            ) from e

    def put_and_copy(
        self,
        data: list[dict[str, Any]],
        target_table: str,
        schema: str = SCHEMA_RAW,
        batch_id: str | None = None,
    ) -> LoadMetrics:
        """Load data via local file PUT → COPY INTO pattern.

        Serializes data to a temporary file, PUTs to internal stage,
        then COPY INTO the target table.

        Args:
            data: List of records to load
            target_table: Target table name
            schema: Target schema
            batch_id: Optional batch identifier

        Returns:
            LoadMetrics with loading statistics
        """
        if not data:
            return LoadMetrics(
                batch_id=batch_id or "",
                source_schema="LOCAL",
                target_schema=schema,
                table_name=target_table,
                rows_loaded=0,
                rows_rejected=0,
                duration_ms=0,
                strategy=LoadStrategy.APPEND,
            )

        batch_id = batch_id or uuid.uuid4().hex[:16]
        start_time = time.perf_counter()

        # Write data to temporary NDJSON file
        temp_dir = tempfile.mkdtemp(prefix="riskpulse_sf_")
        file_path = Path(temp_dir) / f"{batch_id}.json.gz"

        try:
            import gzip

            with gzip.open(file_path, "wt", encoding="utf-8") as f:
                for record in data:
                    record_with_meta = {
                        **record,
                        "_batch_id": batch_id,
                        "_load_timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    f.write(json.dumps(record_with_meta, default=str) + "\n")

            file_size = file_path.stat().st_size

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # PUT file to internal stage
                internal_stage = f"@{self._database}.{schema}.%{target_table}"
                put_query = (
                    f"PUT 'file://{file_path.as_posix()}' {internal_stage} "
                    f"AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
                )
                cursor.execute(put_query)

                # COPY INTO target table
                copy_query = f"""
                    COPY INTO {schema}.{target_table}
                    FROM {internal_stage}
                    FILE_FORMAT = (TYPE = 'JSON' COMPRESSION = 'GZIP' STRIP_OUTER_ARRAY = FALSE)
                    ON_ERROR = 'CONTINUE'
                    PURGE = TRUE
                    MATCH_BY_COLUMN_NAME = 'CASE_INSENSITIVE'
                """
                cursor.execute(copy_query)

                rows_loaded = cursor.rowcount or len(data)

            duration_ms = (time.perf_counter() - start_time) * 1000
            self._metrics.record_load(rows_loaded, 0, file_size)

            metrics = LoadMetrics(
                batch_id=batch_id,
                source_schema="LOCAL",
                target_schema=schema,
                table_name=target_table,
                rows_loaded=rows_loaded,
                rows_rejected=0,
                duration_ms=duration_ms,
                strategy=LoadStrategy.APPEND,
            )

            logger.info(
                "snowflake_put_copy_complete",
                table=target_table,
                batch_id=batch_id,
                rows_loaded=rows_loaded,
                file_size_bytes=file_size,
                duration_ms=round(duration_ms, 2),
            )

            return metrics

        finally:
            # Clean up temp file
            if file_path.exists():
                file_path.unlink()
            Path(temp_dir).rmdir()

    def bulk_load_records(
        self,
        records: list[dict[str, Any]],
        table_name: str,
        schema: str = SCHEMA_RAW,
        batch_id: str | None = None,
    ) -> LoadMetrics:
        """Compatibility wrapper for orchestrated buffered loads."""
        return self.put_and_copy(
            data=records,
            target_table=table_name,
            schema=schema,
            batch_id=batch_id,
        )

    # =========================================================================
    # INCREMENTAL LOADING (Watermark-based)
    # =========================================================================

    def get_watermark(self, table_name: str, column_name: str = "LOAD_TIMESTAMP") -> str | None:
        """Get the current watermark value for incremental loading.

        Args:
            table_name: Table to get watermark for
            column_name: Timestamp column used as watermark

        Returns:
            Last watermark value as ISO string, or None if never loaded
        """
        key = f"{table_name}.{column_name}"
        if key in self._watermarks:
            return self._watermarks[key].last_value

        # Try to get from Snowflake metadata table
        try:
            result = self.execute_query(
                """
                SELECT LAST_VALUE FROM RAW.LOAD_WATERMARKS
                WHERE TABLE_NAME = %(table_name)s AND COLUMN_NAME = %(column_name)s
                """,
                params={"table_name": table_name, "column_name": column_name},
            )
            if result.rows:
                value = str(result.rows[0]["LAST_VALUE"])
                self._watermarks[key] = WatermarkState(
                    table_name=table_name,
                    column_name=column_name,
                    last_value=value,
                )
                return value
        except SnowflakeQueryError:
            pass

        return None

    def update_watermark(self, table_name: str, column_name: str, value: str) -> None:
        """Update the watermark after successful incremental load.

        Args:
            table_name: Table name
            column_name: Watermark column name
            value: New watermark value (ISO timestamp)
        """
        key = f"{table_name}.{column_name}"
        self._watermarks[key] = WatermarkState(
            table_name=table_name,
            column_name=column_name,
            last_value=value,
        )

        self.execute_non_query(
            """
            MERGE INTO RAW.LOAD_WATERMARKS tgt
            USING (
                SELECT %(table_name)s AS TABLE_NAME,
                       %(column_name)s AS COLUMN_NAME,
                       %(value)s AS LAST_VALUE
            ) src
            ON tgt.TABLE_NAME = src.TABLE_NAME AND tgt.COLUMN_NAME = src.COLUMN_NAME
            WHEN MATCHED THEN UPDATE SET
                LAST_VALUE = src.LAST_VALUE,
                UPDATED_AT = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT (TABLE_NAME, COLUMN_NAME, LAST_VALUE, UPDATED_AT)
            VALUES (src.TABLE_NAME, src.COLUMN_NAME, src.LAST_VALUE, CURRENT_TIMESTAMP())
            """,
            params={"table_name": table_name, "column_name": column_name, "value": value},
        )

        logger.info(
            "watermark_updated",
            table=table_name,
            column=column_name,
            value=value,
        )

    def load_incremental(
        self,
        source_table: str,
        target_table: str,
        watermark_column: str = "LOAD_TIMESTAMP",
        source_schema: str = SCHEMA_RAW,
        target_schema: str = SCHEMA_STAGING,
        transform_sql: str | None = None,
        batch_id: str | None = None,
    ) -> LoadMetrics:
        """Perform incremental load using watermark-based strategy.

        Only processes records newer than the last watermark value.

        Args:
            source_table: Source table name
            target_table: Target table name
            watermark_column: Column to use for watermarking
            source_schema: Source schema
            target_schema: Target schema
            transform_sql: Optional SQL transformation expression
            batch_id: Batch identifier

        Returns:
            LoadMetrics with loading statistics
        """
        batch_id = batch_id or uuid.uuid4().hex[:16]
        start_time = time.perf_counter()

        source_name = _validate_qualified_name(source_schema, source_table)
        target_name = _validate_qualified_name(target_schema, target_table)
        watermark_column = _validate_identifier(watermark_column, name="watermark column")
        select_expr = _validate_transform_sql(transform_sql)

        watermark = self.get_watermark(f"{target_schema}.{target_table}", watermark_column)
        watermark_filter = ""
        query_params: dict[str, Any] = {}
        if watermark:
            watermark_filter = f"WHERE {watermark_column} > %(watermark)s"
            query_params["watermark"] = watermark

        # Identifiers and optional transform SQL are validated before interpolation.
        query = f"""
                INSERT INTO {target_name}
                SELECT {select_expr}
                FROM {source_name}
                {watermark_filter}
                ORDER BY {watermark_column}
            """  # nosec B608

        try:
            rows_affected = self.execute_non_query(query, params=query_params)

            # Update watermark to max value in loaded batch
            new_watermark_result = self.execute_query(
                f"SELECT MAX({watermark_column})::VARCHAR AS max_val "  # nosec B608
                f"FROM {source_name} {watermark_filter}",
                params=query_params,
            )
            if new_watermark_result.rows and new_watermark_result.rows[0].get("MAX_VAL"):
                self.update_watermark(
                    f"{target_schema}.{target_table}",
                    watermark_column,
                    new_watermark_result.rows[0]["MAX_VAL"],
                )

            duration_ms = (time.perf_counter() - start_time) * 1000

            return LoadMetrics(
                batch_id=batch_id,
                source_schema=source_schema,
                target_schema=target_schema,
                table_name=target_table,
                rows_loaded=rows_affected,
                rows_rejected=0,
                duration_ms=duration_ms,
                strategy=LoadStrategy.INCREMENTAL,
            )

        except SnowflakeQueryError as e:
            raise SnowflakeLoadError(
                f"Incremental load failed for {source_table} → {target_table}: {e}"
            ) from e

    # =========================================================================
    # PIPELINE ORCHESTRATION (RAW → STAGING → ANALYTICS → REPORTING)
    # =========================================================================

    def process_raw_to_staging(self, batch_id: str) -> dict[str, LoadMetrics]:
        """Transform RAW VARIANT data into typed STAGING tables.

        Calls stored procedures for parsing raw JSON into structured columns.

        Args:
            batch_id: Batch identifier to process

        Returns:
            Dictionary of table name → LoadMetrics
        """
        results: dict[str, LoadMetrics] = {}
        start_time = time.perf_counter()

        # Load transactions
        proc_result = self.call_procedure("STAGING.LOAD_TRANSACTIONS", [batch_id])
        duration_ms = (time.perf_counter() - start_time) * 1000
        results["STG_TRANSACTIONS"] = LoadMetrics(
            batch_id=batch_id,
            source_schema=SCHEMA_RAW,
            target_schema=SCHEMA_STAGING,
            table_name="STG_TRANSACTIONS",
            rows_loaded=self._parse_rows_from_result(proc_result),
            rows_rejected=0,
            duration_ms=duration_ms,
            strategy=LoadStrategy.APPEND,
        )

        # Load fraud alerts
        start_time = time.perf_counter()
        proc_result = self.call_procedure("STAGING.LOAD_FRAUD_ALERTS", [batch_id])
        duration_ms = (time.perf_counter() - start_time) * 1000
        results["STG_FRAUD_ALERTS"] = LoadMetrics(
            batch_id=batch_id,
            source_schema=SCHEMA_RAW,
            target_schema=SCHEMA_STAGING,
            table_name="STG_FRAUD_ALERTS",
            rows_loaded=self._parse_rows_from_result(proc_result),
            rows_rejected=0,
            duration_ms=duration_ms,
            strategy=LoadStrategy.APPEND,
        )

        # Load risk scores
        start_time = time.perf_counter()
        proc_result = self.call_procedure("STAGING.LOAD_RISK_SCORES", [batch_id])
        duration_ms = (time.perf_counter() - start_time) * 1000
        results["STG_RISK_SCORES"] = LoadMetrics(
            batch_id=batch_id,
            source_schema=SCHEMA_RAW,
            target_schema=SCHEMA_STAGING,
            table_name="STG_RISK_SCORES",
            rows_loaded=self._parse_rows_from_result(proc_result),
            rows_rejected=0,
            duration_ms=duration_ms,
            strategy=LoadStrategy.APPEND,
        )

        logger.info(
            "raw_to_staging_complete",
            batch_id=batch_id,
            tables_processed=len(results),
            total_rows=sum(m.rows_loaded for m in results.values()),
        )

        return results

    def process_staging_to_analytics(self, batch_id: str) -> dict[str, LoadMetrics]:
        """Transform STAGING data into ANALYTICS star schema.

        Populates fact and dimension tables from staging layer.

        Args:
            batch_id: Batch identifier to process

        Returns:
            Dictionary of table name → LoadMetrics
        """
        results: dict[str, LoadMetrics] = {}

        # Update customer dimension (SCD Type 2)
        start_time = time.perf_counter()
        proc_result = self.call_procedure("ANALYTICS.UPDATE_DIM_CUSTOMER", [batch_id])
        duration_ms = (time.perf_counter() - start_time) * 1000
        results["DIM_CUSTOMER"] = LoadMetrics(
            batch_id=batch_id,
            source_schema=SCHEMA_STAGING,
            target_schema=SCHEMA_ANALYTICS,
            table_name="DIM_CUSTOMER",
            rows_loaded=self._parse_rows_from_result(proc_result),
            rows_rejected=0,
            duration_ms=duration_ms,
            strategy=LoadStrategy.INCREMENTAL,
        )

        # Update merchant dimension
        start_time = time.perf_counter()
        rows = self._upsert_merchant_dimension(batch_id)
        duration_ms = (time.perf_counter() - start_time) * 1000
        results["DIM_MERCHANT"] = LoadMetrics(
            batch_id=batch_id,
            source_schema=SCHEMA_STAGING,
            target_schema=SCHEMA_ANALYTICS,
            table_name="DIM_MERCHANT",
            rows_loaded=rows,
            rows_rejected=0,
            duration_ms=duration_ms,
            strategy=LoadStrategy.INCREMENTAL,
        )

        # Update geography dimension
        start_time = time.perf_counter()
        rows = self._upsert_geography_dimension(batch_id)
        duration_ms = (time.perf_counter() - start_time) * 1000
        results["DIM_GEOGRAPHY"] = LoadMetrics(
            batch_id=batch_id,
            source_schema=SCHEMA_STAGING,
            target_schema=SCHEMA_ANALYTICS,
            table_name="DIM_GEOGRAPHY",
            rows_loaded=rows,
            rows_rejected=0,
            duration_ms=duration_ms,
            strategy=LoadStrategy.INCREMENTAL,
        )

        # Load fact transactions
        start_time = time.perf_counter()
        proc_result = self.call_procedure("ANALYTICS.LOAD_FACT_TRANSACTIONS", [batch_id])
        duration_ms = (time.perf_counter() - start_time) * 1000
        results["FACT_TRANSACTIONS"] = LoadMetrics(
            batch_id=batch_id,
            source_schema=SCHEMA_STAGING,
            target_schema=SCHEMA_ANALYTICS,
            table_name="FACT_TRANSACTIONS",
            rows_loaded=self._parse_rows_from_result(proc_result),
            rows_rejected=0,
            duration_ms=duration_ms,
            strategy=LoadStrategy.APPEND,
        )

        logger.info(
            "staging_to_analytics_complete",
            batch_id=batch_id,
            tables_processed=len(results),
            total_rows=sum(m.rows_loaded for m in results.values()),
        )

        return results

    def refresh_reporting(self, summary_date: str) -> dict[str, LoadMetrics]:
        """Refresh REPORTING layer aggregations for a given date.

        Args:
            summary_date: Date string (YYYY-MM-DD) to refresh

        Returns:
            Dictionary of table name → LoadMetrics
        """
        results: dict[str, LoadMetrics] = {}

        start_time = time.perf_counter()
        self.call_procedure("REPORTING.REFRESH_ALL", [summary_date])
        duration_ms = (time.perf_counter() - start_time) * 1000

        results["REPORTING_ALL"] = LoadMetrics(
            batch_id=f"report_{summary_date}",
            source_schema=SCHEMA_ANALYTICS,
            target_schema=SCHEMA_REPORTING,
            table_name="ALL_REPORTING_TABLES",
            rows_loaded=0,
            rows_rejected=0,
            duration_ms=duration_ms,
            strategy=LoadStrategy.FULL,
        )

        logger.info(
            "reporting_refresh_complete",
            summary_date=summary_date,
            duration_ms=round(duration_ms, 2),
        )

        return results

    def run_full_pipeline(
        self,
        batch_id: str,
        summary_date: str | None = None,
    ) -> dict[str, dict[str, LoadMetrics]]:
        """Execute the full ETL pipeline: RAW → STAGING → ANALYTICS → REPORTING.

        Args:
            batch_id: Batch identifier
            summary_date: Date for reporting refresh (defaults to today)

        Returns:
            Nested dict of layer → table → LoadMetrics
        """
        summary_date = summary_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        pipeline_results: dict[str, dict[str, LoadMetrics]] = {}

        # Stage 1: RAW → STAGING
        pipeline_results["raw_to_staging"] = self.process_raw_to_staging(batch_id)

        # Stage 2: STAGING → ANALYTICS
        pipeline_results["staging_to_analytics"] = self.process_staging_to_analytics(batch_id)

        # Stage 3: ANALYTICS → REPORTING
        pipeline_results["reporting"] = self.refresh_reporting(summary_date)

        total_rows = sum(
            m.rows_loaded for layer in pipeline_results.values() for m in layer.values()
        )

        logger.info(
            "full_pipeline_complete",
            batch_id=batch_id,
            layers_processed=len(pipeline_results),
            total_rows_processed=total_rows,
        )

        return pipeline_results

    # =========================================================================
    # SCD TYPE 2 MANAGEMENT
    # =========================================================================

    def apply_scd_type2(
        self,
        staging_table: str,
        dimension_table: str,
        business_key: str,
        tracked_columns: list[str],
        batch_id: str,
    ) -> int:
        """Apply SCD Type 2 logic for slowly changing dimensions.

        Compares staging data with current dimension records,
        closes changed records and inserts new versions.

        Args:
            staging_table: Source staging table
            dimension_table: Target dimension table
            business_key: Business key column for matching
            tracked_columns: Columns to track for changes
            batch_id: Current batch identifier

        Returns:
            Number of new dimension records created
        """
        staging_table = _validate_identifier(staging_table, name="staging table")
        dimension_table = _validate_identifier(dimension_table, name="dimension table")
        business_key = _validate_identifier(business_key, name="business key")
        tracked_columns = [
            _validate_identifier(column, name="tracked column") for column in tracked_columns
        ]
        change_conditions = " OR ".join(f"dim.{col} != stg.{col}" for col in tracked_columns)

        # Step 1: Close existing records that have changed
        close_query = f"""
                UPDATE {SCHEMA_ANALYTICS}.{dimension_table} dim
                SET EFFECTIVE_TO = CURRENT_TIMESTAMP(),
                    IS_CURRENT = FALSE
                FROM {SCHEMA_STAGING}.{staging_table} stg
                WHERE dim.{business_key} = stg.{business_key}
                  AND dim.IS_CURRENT = TRUE
                  AND stg.BATCH_ID = %(batch_id)s
                  AND ({change_conditions})
            """  # nosec B608
        closed_count = self.execute_non_query(close_query, params={"batch_id": batch_id})

        # Step 2: Insert new current records for changed/new entries
        insert_columns = ", ".join(tracked_columns)
        stg_columns = ", ".join(f"stg.{col}" for col in tracked_columns)

        insert_query = f"""
                INSERT INTO {SCHEMA_ANALYTICS}.{dimension_table}
                    ({business_key}, {insert_columns}, EFFECTIVE_FROM, EFFECTIVE_TO, IS_CURRENT)
                SELECT
                    stg.{business_key},
                    {stg_columns},
                    CURRENT_TIMESTAMP(),
                    '9999-12-31'::TIMESTAMPTZ,
                    TRUE
                FROM {SCHEMA_STAGING}.{staging_table} stg
                LEFT JOIN {SCHEMA_ANALYTICS}.{dimension_table} dim
                    ON stg.{business_key} = dim.{business_key} AND dim.IS_CURRENT = TRUE
                WHERE stg.BATCH_ID = %(batch_id)s
                  AND (dim.{business_key} IS NULL OR {change_conditions})
            """  # nosec B608
        inserted_count = self.execute_non_query(insert_query, params={"batch_id": batch_id})

        logger.info(
            "scd_type2_applied",
            dimension=dimension_table,
            batch_id=batch_id,
            records_closed=closed_count,
            records_inserted=inserted_count,
        )

        return inserted_count

    # =========================================================================
    # SCHEMA MANAGEMENT
    # =========================================================================

    def ensure_schema_exists(self, schema_name: str) -> None:
        """Create schema if it does not exist."""
        schema_name = _validate_identifier(schema_name, name="schema")
        self.execute_non_query(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

    def ensure_table_exists(self, schema: str, table_name: str, ddl: str) -> None:
        """Create table if it does not exist.

        Args:
            schema: Target schema
            table_name: Table name
            ddl: CREATE TABLE DDL statement
        """
        self.execute_non_query(ddl, schema=schema)
        logger.debug("table_ensured", schema=schema, table=table_name)

    def get_table_columns(self, schema: str, table_name: str) -> list[dict[str, Any]]:
        """Get column metadata for a table.

        Returns:
            List of column info dicts with name, type, nullable keys
        """
        result = self.execute_query(
            """
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH,
                   NUMERIC_PRECISION, NUMERIC_SCALE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %(schema)s AND TABLE_NAME = %(table_name)s
            ORDER BY ORDINAL_POSITION
            """,
            params={"schema": schema, "table_name": table_name},
        )
        return result.rows

    def add_column(
        self,
        schema: str,
        table_name: str,
        column_name: str,
        data_type: str,
        default: str | None = None,
    ) -> None:
        """Add a column to an existing table."""
        stmt = f"ALTER TABLE {schema}.{table_name} ADD COLUMN {column_name} {data_type}"
        if default is not None:
            stmt += f" DEFAULT {default}"
        self.execute_non_query(stmt)
        logger.info("column_added", schema=schema, table=table_name, column=column_name)

    def initialize_schemas(self) -> None:
        """Create all four schema layers if they don't exist."""
        for schema in [SCHEMA_RAW, SCHEMA_STAGING, SCHEMA_ANALYTICS, SCHEMA_REPORTING]:
            self.ensure_schema_exists(schema)

        # Create watermark tracking table
        self.execute_non_query("""
            CREATE TABLE IF NOT EXISTS RAW.LOAD_WATERMARKS (
                TABLE_NAME VARCHAR(200) NOT NULL,
                COLUMN_NAME VARCHAR(100) NOT NULL,
                LAST_VALUE VARCHAR(500) NOT NULL,
                UPDATED_AT TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP(),
                PRIMARY KEY (TABLE_NAME, COLUMN_NAME)
            )
            """)

        logger.info("schemas_initialized")

    # =========================================================================
    # MATERIALIZED VIEW MANAGEMENT
    # =========================================================================

    def create_or_replace_dynamic_table(
        self,
        schema: str,
        table_name: str,
        query: str,
        warehouse: str | None = None,
        lag: str = "1 hour",
    ) -> None:
        """Create or replace a Snowflake Dynamic Table (materialized view equivalent).

        Args:
            schema: Target schema
            table_name: Dynamic table name
            query: Source query for the dynamic table
            warehouse: Warehouse for refresh (defaults to handler warehouse)
            lag: Target lag for data freshness (e.g., '1 hour', '1 day')
        """
        wh = warehouse or self._warehouse
        stmt = f"""
            CREATE OR REPLACE DYNAMIC TABLE {schema}.{table_name}
            TARGET_LAG = '{lag}'
            WAREHOUSE = {wh}
            AS
            {query}
        """
        self.execute_non_query(stmt)
        logger.info("dynamic_table_created", schema=schema, table=table_name, lag=lag)

    def setup_reporting_dynamic_tables(self) -> None:
        """Set up dynamic tables for common reporting queries."""
        # Real-time transaction velocity
        self.create_or_replace_dynamic_table(
            schema=SCHEMA_REPORTING,
            table_name="DT_TRANSACTION_VELOCITY",
            query="""
                SELECT
                    DATE_TRUNC('HOUR', TRANSACTION_TIMESTAMP) AS HOUR_BUCKET,
                    COUNT(*) AS TRANSACTION_COUNT,
                    SUM(TRANSACTION_AMOUNT) AS TOTAL_AMOUNT,
                    AVG(TRANSACTION_AMOUNT) AS AVG_AMOUNT,
                    COUNT(CASE WHEN IS_FRAUD THEN 1 END) AS FRAUD_COUNT,
                    AVG(RISK_SCORE) AS AVG_RISK_SCORE
                FROM ANALYTICS.FACT_TRANSACTIONS
                WHERE TRANSACTION_TIMESTAMP >= DATEADD(DAY, -7, CURRENT_TIMESTAMP())
                GROUP BY 1
            """,
            lag="30 minutes",
        )

        # Top risky merchants
        self.create_or_replace_dynamic_table(
            schema=SCHEMA_REPORTING,
            table_name="DT_RISKY_MERCHANTS",
            query="""
                SELECT
                    m.MERCHANT_ID,
                    m.MERCHANT_NAME,
                    m.CATEGORY_CODE,
                    COUNT(*) AS TOTAL_TRANSACTIONS,
                    SUM(CASE WHEN ft.IS_FRAUD THEN 1 ELSE 0 END) AS FRAUD_COUNT,
                    CASE WHEN COUNT(*) > 0
                        THEN SUM(CASE WHEN ft.IS_FRAUD THEN 1 ELSE 0 END)::FLOAT / COUNT(*)
                        ELSE 0 END AS FRAUD_RATE,
                    AVG(ft.RISK_SCORE) AS AVG_RISK_SCORE,
                    SUM(ft.TRANSACTION_AMOUNT) AS TOTAL_AMOUNT
                FROM ANALYTICS.FACT_TRANSACTIONS ft
                JOIN ANALYTICS.DIM_MERCHANT m ON ft.MERCHANT_KEY = m.MERCHANT_KEY
                WHERE ft.TRANSACTION_TIMESTAMP >= DATEADD(DAY, -30, CURRENT_TIMESTAMP())
                GROUP BY m.MERCHANT_ID, m.MERCHANT_NAME, m.CATEGORY_CODE
                HAVING COUNT(*) >= 10
            """,
            lag="1 hour",
        )

        # Customer risk profile summary
        self.create_or_replace_dynamic_table(
            schema=SCHEMA_REPORTING,
            table_name="DT_CUSTOMER_RISK_SUMMARY",
            query="""
                SELECT
                    c.CUSTOMER_ID,
                    c.RISK_TIER,
                    COUNT(*) AS TOTAL_TRANSACTIONS_30D,
                    SUM(ft.TRANSACTION_AMOUNT) AS TOTAL_AMOUNT_30D,
                    AVG(ft.RISK_SCORE) AS AVG_RISK_SCORE,
                    MAX(ft.RISK_SCORE) AS MAX_RISK_SCORE,
                    SUM(CASE WHEN ft.IS_FRAUD THEN 1 ELSE 0 END) AS FRAUD_COUNT_30D,
                    COUNT(DISTINCT ft.MERCHANT_KEY) AS UNIQUE_MERCHANTS_30D
                FROM ANALYTICS.FACT_TRANSACTIONS ft
                JOIN ANALYTICS.DIM_CUSTOMER c ON ft.CUSTOMER_KEY = c.CUSTOMER_KEY
                WHERE c.IS_CURRENT = TRUE
                  AND ft.TRANSACTION_TIMESTAMP >= DATEADD(DAY, -30, CURRENT_TIMESTAMP())
                GROUP BY c.CUSTOMER_ID, c.RISK_TIER
            """,
            lag="1 hour",
        )

        logger.info("reporting_dynamic_tables_configured")

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _upsert_merchant_dimension(self, batch_id: str) -> int:
        """Upsert merchant dimension from staging data."""
        return self.execute_non_query(
            """
            MERGE INTO ANALYTICS.DIM_MERCHANT tgt
            USING (
                SELECT DISTINCT
                    MERCHANT_ID,
                    MERCHANT_NAME,
                    MERCHANT_CATEGORY_CODE AS CATEGORY_CODE
                FROM STAGING.STG_TRANSACTIONS
                WHERE BATCH_ID = %(batch_id)s
                  AND MERCHANT_ID IS NOT NULL
            ) src
            ON tgt.MERCHANT_ID = src.MERCHANT_ID
            WHEN MATCHED THEN UPDATE SET
                tgt.MERCHANT_NAME = src.MERCHANT_NAME,
                tgt.CATEGORY_CODE = src.CATEGORY_CODE,
                tgt.TOTAL_TRANSACTIONS = tgt.TOTAL_TRANSACTIONS + 1,
                tgt.LAST_UPDATED = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT (
                MERCHANT_ID, MERCHANT_NAME, CATEGORY_CODE, TOTAL_TRANSACTIONS, LAST_UPDATED
            ) VALUES (
                src.MERCHANT_ID, src.MERCHANT_NAME, src.CATEGORY_CODE, 1, CURRENT_TIMESTAMP()
            )
            """,
            params={"batch_id": batch_id},
        )

    def _upsert_geography_dimension(self, batch_id: str) -> int:
        """Upsert geography dimension from staging data."""
        return self.execute_non_query(
            """
            MERGE INTO ANALYTICS.DIM_GEOGRAPHY tgt
            USING (
                SELECT DISTINCT
                    GEO_COUNTRY AS COUNTRY_CODE,
                    GEO_CITY AS CITY
                FROM STAGING.STG_TRANSACTIONS
                WHERE BATCH_ID = %(batch_id)s
                  AND GEO_COUNTRY IS NOT NULL
            ) src
            ON tgt.COUNTRY_CODE = src.COUNTRY_CODE
               AND COALESCE(tgt.CITY, '') = COALESCE(src.CITY, '')
            WHEN NOT MATCHED THEN INSERT (COUNTRY_CODE, COUNTRY_NAME, CITY)
            VALUES (src.COUNTRY_CODE, src.COUNTRY_CODE, src.CITY)
            """,
            params={"batch_id": batch_id},
        )

    def _parse_rows_from_result(self, result_str: str) -> int:
        """Extract row count from stored procedure result string."""
        import re

        match = re.search(r"(\d+)", result_str)
        return int(match.group(1)) if match else 0

    def _compute_cache_key(self, query: str, params: dict[str, Any] | None) -> str:
        """Compute a cache key for a query + params combination."""
        key_data = query + json.dumps(params or {}, sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> QueryResult | None:
        """Get result from cache if not expired."""
        if key in self._query_cache:
            result, cached_at = self._query_cache[key]
            if time.time() - cached_at < self._query_cache_ttl:
                result_copy = QueryResult(
                    columns=result.columns,
                    rows=result.rows,
                    row_count=result.row_count,
                    duration_ms=result.duration_ms,
                    query_id=result.query_id,
                    cached=True,
                )
                return result_copy
            else:
                del self._query_cache[key]
        return None

    def _put_in_cache(self, key: str, result: QueryResult) -> None:
        """Store result in cache."""
        self._query_cache[key] = (result, time.time())

    # =========================================================================
    # PERFORMANCE UTILITIES
    # =========================================================================

    def get_load_history(self, table_name: str, days: int = 7) -> list[dict[str, Any]]:
        """Get recent COPY INTO load history for a table.

        Args:
            table_name: Table name to check
            days: Number of days of history

        Returns:
            List of load history records
        """
        table_name = _validate_identifier(table_name, name="table")
        days = _bounded_int(days, name="days", minimum=1, maximum=90)
        result = self.execute_query(
            """
            SELECT *
            FROM TABLE(INFORMATION_SCHEMA.COPY_HISTORY(
                TABLE_NAME => %(table_name)s,
                START_TIME => DATEADD(DAYS, -%(days)s, CURRENT_TIMESTAMP())
            ))
            ORDER BY LAST_LOAD_TIME DESC
            LIMIT 50
            """,
            params={"table_name": table_name, "days": days},
        )
        return result.rows

    def get_query_history(self, hours: int = 24, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent query execution history.

        Args:
            hours: Lookback period in hours
            limit: Maximum results

        Returns:
            List of query history records
        """
        hours = _bounded_int(hours, name="hours", minimum=1, maximum=168)
        limit = _bounded_int(limit, name="limit", minimum=1, maximum=1000)
        result = self.execute_query(
            """
                SELECT QUERY_ID, QUERY_TEXT, DATABASE_NAME, SCHEMA_NAME,
                       EXECUTION_STATUS, TOTAL_ELAPSED_TIME, ROWS_PRODUCED,
                       BYTES_SCANNED, START_TIME, END_TIME
                FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY(
                    DATEADD(HOURS, %(hours_offset)s, CURRENT_TIMESTAMP()),
                    CURRENT_TIMESTAMP()
                ))
                ORDER BY START_TIME DESC
                LIMIT %(limit)s
            """,
            params={"hours_offset": -hours, "limit": limit},
        )
        return result.rows

    def get_warehouse_usage(self) -> dict[str, Any]:
        """Get current warehouse credit consumption metrics."""
        result = self.execute_query(
            """
            SELECT
                WAREHOUSE_NAME,
                SUM(CREDITS_USED) AS TOTAL_CREDITS,
                SUM(CREDITS_USED_COMPUTE) AS COMPUTE_CREDITS,
                SUM(CREDITS_USED_CLOUD_SERVICES) AS CLOUD_CREDITS,
                COUNT(*) AS QUERY_COUNT
            FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
            WHERE START_TIME >= DATEADD(DAY, -7, CURRENT_TIMESTAMP())
            GROUP BY WAREHOUSE_NAME
            """,
            use_cache=True,
        )
        return {"warehouses": result.rows}

    def suspend_warehouse(self) -> None:
        """Suspend warehouse to save credits when idle."""
        self.execute_non_query(f"ALTER WAREHOUSE {self._warehouse} SUSPEND")
        logger.info("warehouse_suspended", warehouse=self._warehouse)

    def resume_warehouse(self) -> None:
        """Resume warehouse before running queries."""
        self.execute_non_query(f"ALTER WAREHOUSE {self._warehouse} RESUME")
        logger.info("warehouse_resumed", warehouse=self._warehouse)


def create_snowflake_handler(
    pool_size: int = 5,
    query_cache_ttl: int = 300,
) -> SnowflakeHandler:
    """Factory function to create a configured SnowflakeHandler.

    Reads configuration from environment variables:
    - SNOWFLAKE_ACCOUNT
    - SNOWFLAKE_USER
    - SNOWFLAKE_PASSWORD or SNOWFLAKE_PRIVATE_KEY_PATH
    - SNOWFLAKE_WAREHOUSE
    - SNOWFLAKE_DATABASE
    - SNOWFLAKE_ROLE
    """
    handler = SnowflakeHandler(
        pool_size=pool_size,
        query_cache_ttl=query_cache_ttl,
    )
    return handler
