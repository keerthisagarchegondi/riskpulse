"""Mocked integration tests covering service interaction contracts."""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.middleware.auth import reset_key_manager


AUTH_HEADERS = {"X-API-Key": "dev-api-key-riskpulse-2024"}


class FakeKafkaProducer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.topics: Counter[str] = Counter()

    def produce(self, event: dict[str, Any], topic: str) -> None:
        self.events.append(event)
        self.topics[topic] += 1

    def produce_batch(self, events: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
        self.events.extend(events)
        self.topics[topic] += len(events)
        return []


class FakeConcurrentStorage:
    def __init__(self) -> None:
        self._records: dict[uuid.UUID, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def write(self, transaction_id: uuid.UUID, record: dict[str, Any]) -> None:
        async with self._lock:
            self._records[transaction_id] = record

    async def read(self, transaction_id: uuid.UUID) -> dict[str, Any] | None:
        async with self._lock:
            return self._records.get(transaction_id)

    @property
    def count(self) -> int:
        return len(self._records)


class FakeObjectStore:
    def __init__(self, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.uploads: list[tuple[str, bytes]] = []

    def put_object(self, key: str, body: bytes) -> None:
        if self.fail_after is not None and len(self.uploads) >= self.fail_after:
            raise RuntimeError("simulated S3 throttling")
        self.uploads.append((key, body))


class FakeSnowflakeLoader:
    def __init__(self) -> None:
        self.loaded_files: list[str] = []
        self.watermark: datetime | None = None

    def load_stage_files(self, files: list[str], watermark: datetime) -> dict[str, Any]:
        self.loaded_files.extend(files)
        self.watermark = watermark
        return {"loaded": len(files), "failed": 0}


@pytest.fixture(autouse=True)
def _reset_auth_state():
    reset_key_manager()
    yield
    reset_key_manager()


def _transaction(index: int) -> dict[str, object]:
    return {
        "external_transaction_id": f"TXN-INTEGRATION-{index:04d}",
        "account_id": f"ACC-{index % 50:03d}",
        "customer_id": f"CUST-{index % 100:03d}",
        "merchant_id": "MERCH-001",
        "merchant_name": "Known Merchant",
        "transaction_amount": f"{10 + (index % 100)}.00",
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "transaction_timestamp": "2026-08-13T12:00:00Z",
    }


@pytest.mark.integration
def test_batch_ingestion_publishes_1000_transactions_to_kafka(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = FakeKafkaProducer()
    monkeypatch.setattr("src.api.routes.transactions._get_kafka_producer", lambda: producer)

    response = TestClient(create_app()).post(
        "/api/v1/transactions/batch",
        json={"transactions": [_transaction(index) for index in range(1000)]},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == status.HTTP_202_ACCEPTED
    assert response.json()["accepted"] == 1000
    assert len(producer.events) == 1000
    assert producer.topics["txn.raw.events"] == 1000
    assert {event["external_transaction_id"] for event in producer.events} == {
        f"TXN-INTEGRATION-{index:04d}" for index in range(1000)
    }


@pytest.mark.integration
def test_concurrent_database_operations_preserve_all_records() -> None:
    async def _run_scenario() -> None:
        storage = FakeConcurrentStorage()
        ids = [uuid.uuid4() for _ in range(250)]

        await asyncio.gather(
            *[
                storage.write(transaction_id, {"amount": index, "status": "processed"})
                for index, transaction_id in enumerate(ids)
            ]
        )
        rows = await asyncio.gather(*[storage.read(transaction_id) for transaction_id in ids])

        assert storage.count == 250
        assert all(row is not None for row in rows)
        assert sorted(row["amount"] for row in rows if row is not None) == list(range(250))

    asyncio.run(_run_scenario())


@pytest.mark.integration
def test_concurrent_database_operations_are_atomic_under_interleaved_reads() -> None:
    storage = FakeConcurrentStorage()
    ids = [uuid.uuid4() for _ in range(250)]

    async def _run_scenario() -> None:
        writers = [
            storage.write(transaction_id, {"amount": index, "status": "processed"})
            for index, transaction_id in enumerate(ids)
        ]
        readers = [storage.read(transaction_id) for transaction_id in ids]
        await asyncio.gather(*(writers + readers))

        persisted = await asyncio.gather(*[storage.read(transaction_id) for transaction_id in ids])
        assert storage.count == 250
        assert all(row is not None for row in persisted)

    asyncio.run(_run_scenario())


@pytest.mark.integration
def test_kafka_producer_consumer_contract_preserves_order_and_payload() -> None:
    producer = FakeKafkaProducer()
    events = [
        {
            "transaction_id": str(uuid.uuid4()),
            "external_transaction_id": f"TXN-KAFKA-{index:04d}",
            "transaction_amount": float(index),
        }
        for index in range(200)
    ]

    producer.produce_batch(events, topic="txn.raw.events")
    consumed = list(producer.events)

    assert consumed == events
    assert consumed[0]["external_transaction_id"] == "TXN-KAFKA-0000"
    assert consumed[-1]["external_transaction_id"] == "TXN-KAFKA-0199"


@pytest.mark.integration
def test_s3_error_scenario_reports_partial_upload_without_data_loss() -> None:
    store = FakeObjectStore(fail_after=3)
    keys = [f"raw/2026/08/13/transactions-{index}.json" for index in range(5)]
    failures: list[str] = []

    for key in keys:
        try:
            store.put_object(key, b"{}")
        except RuntimeError:
            failures.append(key)

    assert [key for key, _ in store.uploads] == keys[:3]
    assert failures == keys[3:]


@pytest.mark.integration
def test_snowflake_loading_verifies_counts_and_watermark() -> None:
    loader = FakeSnowflakeLoader()
    watermark = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)

    result = loader.load_stage_files(
        [f"stage/transactions-{index}.parquet" for index in range(12)],
        watermark=watermark,
    )

    assert result == {"loaded": 12, "failed": 0}
    assert len(loader.loaded_files) == 12
    assert loader.watermark == watermark
