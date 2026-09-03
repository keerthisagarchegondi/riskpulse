"""Local analytical warehouse fallback for development and CI."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from src.storage.snowflake_handler import (
    SCHEMA_RAW,
    FileFormat,
    LoadMetrics,
    LoadStrategy,
    QueryResult,
    SnowflakeMetrics,
)
from src.utils.config import get_settings


class LocalWarehouseHandler:
    """Small SnowflakeHandler-compatible warehouse backed by local JSONL files."""

    def __init__(
        self,
        root_path: str | os.PathLike[str] | None = None,
        *_: Any,
        **__: Any,
    ) -> None:
        settings = get_settings()
        configured_root = (
            root_path
            or os.environ.get("RISKPULSE_LOCAL_WAREHOUSE_ROOT")
            or settings.get("warehouse.local.root", ".local_storage/warehouse")
        )
        self.root_path = Path(configured_root).expanduser()
        if not self.root_path.is_absolute():
            self.root_path = Path.cwd() / self.root_path
        self.root_path.mkdir(parents=True, exist_ok=True)
        self._metrics = SnowflakeMetrics()
        self._connected = False
        self._lock = Lock()

    @property
    def metrics(self) -> SnowflakeMetrics:
        return self._metrics

    def connect(self) -> None:
        self.root_path.mkdir(parents=True, exist_ok=True)
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> QueryResult:
        del params, use_cache
        start = time.perf_counter()
        normalized = " ".join(query.upper().split())

        if normalized.startswith("SELECT 1"):
            rows = [{"HEALTH_CHECK": 1}]
            columns = ["HEALTH_CHECK"]
        else:
            rows = []
            columns = []

        duration_ms = (time.perf_counter() - start) * 1000
        self._metrics.record_query(duration_ms)
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            duration_ms=duration_ms,
            query_id=f"local-{uuid.uuid4().hex[:12]}",
            cached=False,
        )

    def execute_non_query(self, query: str, params: dict[str, Any] | None = None) -> int:
        del query, params
        self._metrics.record_query(0)
        return 0

    def bulk_load_records(
        self,
        records: list[dict[str, Any]],
        table_name: str,
        schema: str = SCHEMA_RAW,
        batch_id: str | None = None,
    ) -> LoadMetrics:
        return self.put_and_copy(records, target_table=table_name, schema=schema, batch_id=batch_id)

    def put_and_copy(
        self,
        data: list[dict[str, Any]],
        target_table: str,
        schema: str = SCHEMA_RAW,
        batch_id: str | None = None,
    ) -> LoadMetrics:
        batch_id = batch_id or uuid.uuid4().hex[:16]
        start = time.perf_counter()
        table_path = self._table_path(schema, target_table)
        table_path.parent.mkdir(parents=True, exist_ok=True)

        bytes_written = 0
        with self._lock, table_path.open("a", encoding="utf-8") as handle:
            for record in data:
                payload = {
                    **record,
                    "_batch_id": batch_id,
                    "_load_timestamp": datetime.now(timezone.utc).isoformat(),
                }
                line = json.dumps(payload, default=str)
                bytes_written += len(line.encode("utf-8")) + 1
                handle.write(line)
                handle.write("\n")

        duration_ms = (time.perf_counter() - start) * 1000
        self._metrics.record_load(len(data), 0, bytes_written)
        return LoadMetrics(
            batch_id=batch_id,
            source_schema="LOCAL",
            target_schema=schema,
            table_name=target_table,
            rows_loaded=len(data),
            rows_rejected=0,
            duration_ms=duration_ms,
            strategy=LoadStrategy.APPEND,
        )

    def load_from_s3_stage(
        self,
        table_name: str,
        stage_name: str = "LOCAL.STAGE",
        file_pattern: str | None = None,
        file_format: FileFormat = FileFormat.PARQUET,
        batch_id: str | None = None,
    ) -> LoadMetrics:
        del stage_name, file_pattern, file_format
        return LoadMetrics(
            batch_id=batch_id or uuid.uuid4().hex[:16],
            source_schema="LOCAL_STAGE",
            target_schema=SCHEMA_RAW,
            table_name=table_name,
            rows_loaded=0,
            rows_rejected=0,
            duration_ms=0,
            strategy=LoadStrategy.APPEND,
        )

    def get_watermark(self, table_name: str, column_name: str = "LOAD_TIMESTAMP") -> str | None:
        watermark_path = self._watermark_path(table_name, column_name)
        if not watermark_path.exists():
            return None
        return watermark_path.read_text(encoding="utf-8").strip() or None

    def update_watermark(
        self,
        table_name: str,
        column_name: str,
        value: str,
    ) -> None:
        watermark_path = self._watermark_path(table_name, column_name)
        watermark_path.parent.mkdir(parents=True, exist_ok=True)
        watermark_path.write_text(value, encoding="utf-8")

    def _table_path(self, schema: str, table_name: str) -> Path:
        safe_schema = self._safe_name(schema)
        safe_table = self._safe_name(table_name)
        return self.root_path / safe_schema / f"{safe_table}.jsonl"

    def _watermark_path(self, table_name: str, column_name: str) -> Path:
        return (
            self.root_path
            / "_watermarks"
            / f"{self._safe_name(table_name)}.{self._safe_name(column_name)}"
        )

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
        return safe or "unknown"
