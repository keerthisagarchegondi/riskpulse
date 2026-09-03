from __future__ import annotations

import json

from src.storage.local_warehouse_handler import LocalWarehouseHandler
from src.storage.snowflake_handler import SCHEMA_RAW, create_snowflake_handler


def test_local_warehouse_bulk_loads_records_to_jsonl(tmp_path) -> None:
    handler = LocalWarehouseHandler(root_path=tmp_path)
    records = [{"transaction_id": "txn-1", "amount": 99.5}]

    metrics = handler.bulk_load_records(records, table_name="transactions", schema=SCHEMA_RAW)

    stored_path = tmp_path / SCHEMA_RAW / "transactions.jsonl"
    payload = json.loads(stored_path.read_text(encoding="utf-8").splitlines()[0])
    assert metrics.rows_loaded == 1
    assert payload["transaction_id"] == "txn-1"
    assert payload["_batch_id"] == metrics.batch_id


def test_local_warehouse_health_query_returns_row(tmp_path) -> None:
    handler = LocalWarehouseHandler(root_path=tmp_path)
    handler.connect()

    result = handler.execute_query("SELECT 1 AS HEALTH_CHECK")

    assert result.row_count == 1
    assert result.rows[0]["HEALTH_CHECK"] == 1


def test_snowflake_factory_uses_local_warehouse(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RISKPULSE_WAREHOUSE_BACKEND", "local")
    monkeypatch.setenv("RISKPULSE_LOCAL_WAREHOUSE_ROOT", str(tmp_path))

    handler = create_snowflake_handler()

    assert isinstance(handler, LocalWarehouseHandler)
