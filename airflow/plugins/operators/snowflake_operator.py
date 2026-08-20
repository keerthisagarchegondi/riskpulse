"""Custom Snowflake operators for the RiskPulse Airflow platform.

Provides reusable operators for common Snowflake ETL patterns:
- COPY INTO from external stages (S3)
- MERGE (upsert) operations
- Materialized view refresh
"""

from __future__ import annotations

import re
import time
from typing import Any, Sequence

from airflow.exceptions import AirflowException
from airflow.models import BaseOperator
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from airflow.utils.context import Context

from src.utils.logger import get_logger

logger = get_logger(__name__, component="snowflake_operator")

_SNOWFLAKE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _validate_snowflake_identifier(identifier: str, *, name: str = "identifier") -> str:
    """Validate Snowflake identifiers before interpolating them into SQL."""
    if not isinstance(identifier, str) or not _SNOWFLAKE_IDENTIFIER_RE.fullmatch(identifier):
        raise AirflowException(f"Invalid Snowflake {name}: {identifier!r}")
    return identifier


def _validate_identifier_list(values: Sequence[str], *, name: str) -> list[str]:
    return [_validate_snowflake_identifier(value, name=name) for value in values]


class SnowflakeCopyIntoOperator(BaseOperator):
    """Load data from an external S3 stage into a Snowflake table using COPY INTO.

    Supports partitioned loading, error handling, and load metrics.

    Parameters
    ----------
    snowflake_conn_id : str
        Airflow connection ID for Snowflake.
    database : str
        Snowflake database name.
    schema : str
        Target schema name.
    table : str
        Target table name.
    stage : str
        External stage path (e.g., @DB.SCHEMA.STAGE_NAME).
    partition_path : str | None
        Optional sub-path within the stage for partition-based loading.
    file_format : str
        File format specification. Defaults to Parquet with snappy.
    on_error : str
        Error handling policy. One of CONTINUE, SKIP_FILE, ABORT_STATEMENT.
    purge : bool
        Whether to purge files after successful load.
    pattern : str | None
        Optional regex pattern to filter files in the stage.
    warehouse : str | None
        Snowflake warehouse to use. Falls back to connection default.
    """

    template_fields: Sequence[str] = (
        "partition_path",
        "stage",
        "table",
        "schema",
        "database",
    )

    def __init__(
        self,
        *,
        snowflake_conn_id: str = "riskpulse_snowflake",
        database: str = "RISKPULSE",
        schema: str = "RAW",
        table: str,
        stage: str,
        partition_path: str | None = None,
        file_format: str = "TYPE = 'PARQUET' SNAPPY_COMPRESSION = TRUE",
        on_error: str = "CONTINUE",
        purge: bool = False,
        pattern: str | None = None,
        warehouse: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.snowflake_conn_id = snowflake_conn_id
        self.database = database
        self.schema = schema
        self.table = table
        self.stage = stage
        self.partition_path = partition_path
        self.file_format = file_format
        self.on_error = on_error
        self.purge = purge
        self.pattern = pattern
        self.warehouse = warehouse

    def execute(self, context: Context) -> dict[str, Any]:
        hook = SnowflakeHook(snowflake_conn_id=self.snowflake_conn_id)
        start_time = time.monotonic()

        # Build stage path
        stage_path = self.stage
        if self.partition_path:
            stage_path = f"{self.stage}/{self.partition_path}/"

        # Build COPY INTO SQL
        sql_parts = [
            f"COPY INTO {self.database}.{self.schema}.{self.table}",
            f"FROM {stage_path}",
            f"FILE_FORMAT = ({self.file_format})",
            "MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE",
            f"ON_ERROR = '{self.on_error}'",
            f"PURGE = {str(self.purge).upper()}",
        ]

        if self.pattern:
            sql_parts.append(f"PATTERN = '{self.pattern}'")

        sql = "\n".join(sql_parts) + ";"

        logger.info(
            "Executing COPY INTO",
            table=f"{self.database}.{self.schema}.{self.table}",
            stage=stage_path,
        )

        try:
            if self.warehouse:
                hook.run(f"USE WAREHOUSE {self.warehouse};", autocommit=True)

            result = hook.run(sql, autocommit=True)

            elapsed_ms = (time.monotonic() - start_time) * 1000
            rows_loaded = 0
            files_loaded = 0
            errors_seen = 0

            if result:
                for row in result:
                    if hasattr(row, "__iter__") and len(row) >= 2:
                        files_loaded += 1
                        rows_loaded += row[0] if row[0] else 0
                        errors_seen += row[1] if len(row) > 1 and row[1] else 0

            summary = {
                "table": f"{self.database}.{self.schema}.{self.table}",
                "stage": stage_path,
                "files_loaded": files_loaded,
                "rows_loaded": rows_loaded,
                "errors": errors_seen,
                "elapsed_ms": round(elapsed_ms, 2),
            }

            if errors_seen > 0:
                logger.warning("COPY INTO completed with errors", **summary)
            else:
                logger.info("COPY INTO completed successfully", **summary)

            return summary

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "COPY INTO failed",
                table=f"{self.database}.{self.schema}.{self.table}",
                error=str(exc),
                elapsed_ms=round(elapsed_ms, 2),
            )
            raise AirflowException(f"COPY INTO failed: {exc}") from exc


class SnowflakeMergeOperator(BaseOperator):
    """Execute a MERGE (upsert) operation in Snowflake.

    Merges data from a source table/query into a target table with
    configurable match keys, update columns, and insert behavior.

    Parameters
    ----------
    snowflake_conn_id : str
        Airflow connection ID for Snowflake.
    database : str
        Snowflake database name.
    target_schema : str
        Target table schema.
    target_table : str
        Target table name.
    source_query : str
        SQL query that provides the source data for the merge.
    merge_keys : list[str]
        Columns to match on (JOIN condition).
    update_columns : list[str] | None
        Columns to update on match. If None, updates all non-key columns.
    insert_columns : list[str] | None
        Columns to insert on no-match. If None, inserts all source columns.
    warehouse : str | None
        Snowflake warehouse to use.
    """

    template_fields: Sequence[str] = (
        "source_query",
        "target_table",
        "target_schema",
        "database",
    )

    def __init__(
        self,
        *,
        snowflake_conn_id: str = "riskpulse_snowflake",
        database: str = "RISKPULSE",
        target_schema: str,
        target_table: str,
        source_query: str,
        merge_keys: list[str],
        update_columns: list[str] | None = None,
        insert_columns: list[str] | None = None,
        warehouse: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.snowflake_conn_id = snowflake_conn_id
        self.database = database
        self.target_schema = target_schema
        self.target_table = target_table
        self.source_query = source_query
        self.merge_keys = merge_keys
        self.update_columns = update_columns
        self.insert_columns = insert_columns
        self.warehouse = warehouse

    def execute(self, context: Context) -> dict[str, Any]:
        hook = SnowflakeHook(snowflake_conn_id=self.snowflake_conn_id)
        start_time = time.monotonic()

        database = _validate_snowflake_identifier(self.database, name="database")
        target_schema = _validate_snowflake_identifier(self.target_schema, name="schema")
        target_table = _validate_snowflake_identifier(self.target_table, name="table")
        merge_keys = _validate_identifier_list(self.merge_keys, name="merge key")
        update_columns = (
            _validate_identifier_list(self.update_columns, name="update column")
            if self.update_columns
            else None
        )
        insert_columns = (
            _validate_identifier_list(self.insert_columns, name="insert column")
            if self.insert_columns
            else None
        )

        target = f"{database}.{target_schema}.{target_table}"

        # Build MERGE SQL
        join_condition = " AND ".join(f"target.{key} = source.{key}" for key in merge_keys)

        sql_parts = [
            f"MERGE INTO {target} AS target",
            f"USING ({self.source_query}) AS source",
            f"ON {join_condition}",
        ]

        # WHEN MATCHED — update
        if update_columns:
            update_set = ", ".join(f"target.{col} = source.{col}" for col in update_columns)
            sql_parts.append(f"WHEN MATCHED THEN UPDATE SET {update_set}")  # nosec B608

        # WHEN NOT MATCHED — insert
        if insert_columns:
            cols = ", ".join(insert_columns)
            vals = ", ".join(f"source.{col}" for col in insert_columns)
            sql_parts.append(f"WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ({vals})")

        sql = "\n".join(sql_parts) + ";"

        logger.info("Executing MERGE", target=target)

        try:
            if self.warehouse:
                hook.run(f"USE WAREHOUSE {self.warehouse};", autocommit=True)

            hook.run(sql, autocommit=True)
            elapsed_ms = (time.monotonic() - start_time) * 1000

            summary = {
                "target": target,
                "merge_keys": self.merge_keys,
                "elapsed_ms": round(elapsed_ms, 2),
            }
            logger.info("MERGE completed", **summary)
            return summary

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "MERGE failed", target=target, error=str(exc), elapsed_ms=round(elapsed_ms, 2)
            )
            raise AirflowException(f"MERGE failed for {target}: {exc}") from exc


class SnowflakeRefreshViewsOperator(BaseOperator):
    """Refresh materialized views (CREATE OR REPLACE TABLE) in Snowflake.

    Executes a series of CREATE OR REPLACE TABLE statements that act as
    materialized views in Snowflake's architecture.

    Parameters
    ----------
    snowflake_conn_id : str
        Airflow connection ID for Snowflake.
    database : str
        Snowflake database name.
    schema : str
        Schema containing the views.
    view_definitions : dict[str, str]
        Mapping of view name → SELECT query that defines the view content.
    warehouse : str | None
        Snowflake warehouse to use.
    fail_on_error : bool
        If False, continue refreshing other views when one fails.
    """

    template_fields: Sequence[str] = ("database", "schema")

    def __init__(
        self,
        *,
        snowflake_conn_id: str = "riskpulse_snowflake",
        database: str = "RISKPULSE",
        schema: str = "ANALYTICS",
        view_definitions: dict[str, str],
        warehouse: str | None = None,
        fail_on_error: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.snowflake_conn_id = snowflake_conn_id
        self.database = database
        self.schema = schema
        self.view_definitions = view_definitions
        self.warehouse = warehouse
        self.fail_on_error = fail_on_error

    def execute(self, context: Context) -> dict[str, Any]:
        hook = SnowflakeHook(snowflake_conn_id=self.snowflake_conn_id)
        start_time = time.monotonic()

        if self.warehouse:
            hook.run(f"USE WAREHOUSE {self.warehouse};", autocommit=True)

        results: dict[str, dict[str, Any]] = {}
        failures = 0

        for view_name, select_query in self.view_definitions.items():
            view_start = time.monotonic()
            full_name = f"{self.database}.{self.schema}.{view_name}"

            sql = f"CREATE OR REPLACE TABLE {full_name} AS\n{select_query};"

            try:
                hook.run(sql, autocommit=True)
                view_elapsed = (time.monotonic() - view_start) * 1000
                results[view_name] = {
                    "status": "success",
                    "elapsed_ms": round(view_elapsed, 2),
                }
                logger.info("View refreshed", view=full_name, elapsed_ms=round(view_elapsed, 2))
            except Exception as exc:
                view_elapsed = (time.monotonic() - view_start) * 1000
                failures += 1
                results[view_name] = {
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_ms": round(view_elapsed, 2),
                }
                logger.error("View refresh failed", view=full_name, error=str(exc))
                if self.fail_on_error:
                    raise AirflowException(f"View refresh failed for {full_name}: {exc}") from exc

        elapsed_ms = (time.monotonic() - start_time) * 1000

        summary = {
            "views_total": len(self.view_definitions),
            "views_succeeded": len(self.view_definitions) - failures,
            "views_failed": failures,
            "elapsed_ms": round(elapsed_ms, 2),
            "details": results,
        }
        logger.info(
            "View refresh batch complete", **{k: v for k, v in summary.items() if k != "details"}
        )

        if failures > 0 and self.fail_on_error:
            raise AirflowException(f"{failures} view(s) failed to refresh")

        return summary
