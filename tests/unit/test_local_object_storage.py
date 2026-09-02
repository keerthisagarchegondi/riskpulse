from __future__ import annotations

import pytest

from src.storage.local_object_storage import LocalObjectStorageError, LocalObjectStorageHandler
from src.storage.s3_handler import S3_BUCKET_RAW, StorageLayer, get_s3_handler


def test_local_storage_writes_and_reads_batch(tmp_path) -> None:
    handler = LocalObjectStorageHandler(root_path=tmp_path)
    records = [
        {"transaction_id": "txn-1", "amount": 25.5, "is_fraud": False},
        {"transaction_id": "txn-2", "amount": 210.0, "is_fraud": True},
    ]

    key = handler.write_batch(
        records=records,
        storage_layer=StorageLayer.RAW,
        partition_path="validated/year=2026/month=09/day=02/hour=10",
        metadata={"source": "unit-test"},
    )

    assert key.endswith(".parquet")
    assert handler.file_exists(S3_BUCKET_RAW, key)
    assert handler.read_batch(StorageLayer.RAW, "validated/year=2026/month=09/day=02") == records
    assert handler.get_file_metadata(S3_BUCKET_RAW, key)["metadata"]["source"] == "unit-test"


def test_local_storage_s3_client_compatibility(tmp_path) -> None:
    handler = LocalObjectStorageHandler(root_path=tmp_path)
    handler.upload_raw_file(b"abcdef", "samples/input.json", bucket=S3_BUCKET_RAW)

    response = handler._s3_client.get_object(
        Bucket=S3_BUCKET_RAW,
        Key="samples/input.json",
        Range="bytes=1-3",
    )
    assert response["Body"].read() == b"bcd"

    pages = list(
        handler._s3_client.get_paginator("list_objects_v2").paginate(
            Bucket=S3_BUCKET_RAW,
            Prefix="samples/",
        )
    )
    assert pages[0]["Contents"][0]["Key"] == "samples/input.json"


def test_local_storage_empty_batch_writes_marker_file(tmp_path) -> None:
    handler = LocalObjectStorageHandler(root_path=tmp_path)

    key = handler.write_batch([], StorageLayer.RAW, "empty/partition")

    assert key.endswith(".empty.json")
    assert handler.file_exists(S3_BUCKET_RAW, key)
    assert handler.read_batch(StorageLayer.RAW, "empty/partition") == []


def test_local_storage_blocks_path_traversal(tmp_path) -> None:
    handler = LocalObjectStorageHandler(root_path=tmp_path)

    with pytest.raises(LocalObjectStorageError):
        handler._s3_client.put_object(Bucket=S3_BUCKET_RAW, Key="../escape.txt", Body=b"x")


def test_storage_factory_uses_local_backend(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RISKPULSE_STORAGE_BACKEND", "local")
    monkeypatch.setenv("RISKPULSE_LOCAL_STORAGE_ROOT", str(tmp_path))

    handler = get_s3_handler()

    assert isinstance(handler, LocalObjectStorageHandler)
