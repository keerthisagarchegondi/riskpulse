"""Filesystem-backed object storage for local RiskPulse deployments."""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from src.storage.s3_handler import (
    MULTIPART_CHUNK_SIZE,
    S3_BUCKET_RAW,
    S3DownloadError,
    S3Metrics,
    S3UploadError,
    StorageLayer,
    _bucket_for_storage_layer,
    _generate_file_key,
    _normalize_object_prefix,
)
from src.utils.config import get_settings

logger = structlog.get_logger(__name__)


class LocalObjectStorageError(Exception):
    """Base exception for local object storage failures."""


class LocalObjectNotFoundError(LocalObjectStorageError):
    """Raised when a local object key is not present."""


class _LocalObjectPaginator:
    def __init__(self, client: LocalObjectStorageClient) -> None:
        self._client = client

    def paginate(self, *, Bucket: str, Prefix: str = "", **_: Any) -> Generator[dict[str, Any]]:
        yield self._client.list_objects_v2(Bucket=Bucket, Prefix=Prefix)


class LocalObjectStorageClient:
    """Small S3-client-compatible facade backed by files on disk."""

    _METADATA_SUFFIX = ".metadata.json"

    def __init__(self, root_path: str | os.PathLike[str]) -> None:
        self.root_path = Path(root_path).expanduser()
        if not self.root_path.is_absolute():
            self.root_path = Path.cwd() / self.root_path
        self.root_path.mkdir(parents=True, exist_ok=True)

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes | str | io.BufferedIOBase,
        ContentType: str = "application/octet-stream",
        Metadata: dict[str, str] | None = None,
        **_: Any,
    ) -> dict[str, str]:
        data = self._coerce_body(Body)
        object_path = self._object_path(Bucket, Key)
        object_path.parent.mkdir(parents=True, exist_ok=True)
        object_path.write_bytes(data)
        self._write_metadata(object_path, ContentType, Metadata or {})
        return {"ETag": f'"{len(data):x}-{int(datetime.now(timezone.utc).timestamp())}"'}

    def get_object(
        self, *, Bucket: str, Key: str, Range: str | None = None, **_: Any
    ) -> dict[str, Any]:
        object_path = self._object_path(Bucket, Key)
        if not object_path.exists():
            raise LocalObjectNotFoundError(f"Object not found: local://{Bucket}/{Key}")

        data = object_path.read_bytes()
        if Range:
            data = self._slice_range(data, Range)

        metadata = self._read_metadata(object_path)
        return {
            "Body": io.BytesIO(data),
            "ContentLength": len(data),
            "ContentType": metadata.get("content_type", "application/octet-stream"),
            "Metadata": metadata.get("metadata", {}),
        }

    def head_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        object_path = self._object_path(Bucket, Key)
        if not object_path.exists():
            raise LocalObjectNotFoundError(f"Object not found: local://{Bucket}/{Key}")

        metadata = self._read_metadata(object_path)
        stat = object_path.stat()
        return {
            "ContentLength": stat.st_size,
            "ContentType": metadata.get("content_type", "application/octet-stream"),
            "LastModified": datetime.fromtimestamp(stat.st_mtime, timezone.utc),
            "Metadata": metadata.get("metadata", {}),
        }

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = "",
        MaxKeys: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        bucket_root = self._bucket_root(Bucket)
        prefix = _normalize_object_prefix(Prefix)
        contents: list[dict[str, Any]] = []

        if bucket_root.exists():
            for path in sorted(bucket_root.rglob("*")):
                if not path.is_file() or path.name.endswith(self._METADATA_SUFFIX):
                    continue
                key = path.relative_to(bucket_root).as_posix()
                if prefix and not key.startswith(prefix):
                    continue
                contents.append({"Key": key, "Size": path.stat().st_size})
                if MaxKeys and len(contents) >= MaxKeys:
                    break

        return {
            "Contents": contents,
            "KeyCount": len(contents),
            "IsTruncated": False,
        }

    def get_paginator(self, operation_name: str) -> _LocalObjectPaginator:
        if operation_name != "list_objects_v2":
            raise LocalObjectStorageError(f"Unsupported paginator operation: {operation_name}")
        return _LocalObjectPaginator(self)

    def put_bucket_notification_configuration(
        self,
        *,
        Bucket: str,
        NotificationConfiguration: dict[str, Any],
        **_: Any,
    ) -> None:
        bucket_root = self._bucket_root(Bucket)
        bucket_root.mkdir(parents=True, exist_ok=True)
        config_path = bucket_root / ".notifications.json"
        config_path.write_text(json.dumps(NotificationConfiguration, indent=2), encoding="utf-8")

    def _bucket_root(self, bucket: str) -> Path:
        if not bucket or "/" in bucket or "\\" in bucket or bucket in {".", ".."}:
            raise LocalObjectStorageError(f"Invalid bucket name: {bucket!r}")
        return self.root_path / bucket

    def _object_path(self, bucket: str, key: str) -> Path:
        if not key or Path(key).is_absolute():
            raise LocalObjectStorageError(f"Invalid object key: {key!r}")

        parts = [part for part in key.replace("\\", "/").split("/") if part]
        if any(part == ".." for part in parts):
            raise LocalObjectStorageError(f"Object key escapes storage root: {key!r}")

        bucket_root = self._bucket_root(bucket).resolve()
        object_path = (bucket_root / Path(*parts)).resolve()
        if object_path != bucket_root and bucket_root not in object_path.parents:
            raise LocalObjectStorageError(f"Object key escapes storage root: {key!r}")
        return object_path

    def _metadata_path(self, object_path: Path) -> Path:
        return object_path.with_name(f"{object_path.name}{self._METADATA_SUFFIX}")

    def _write_metadata(
        self,
        object_path: Path,
        content_type: str,
        metadata: dict[str, str],
    ) -> None:
        payload = {
            "content_type": content_type,
            "metadata": metadata,
            "last_modified": datetime.now(timezone.utc).isoformat(),
        }
        self._metadata_path(object_path).write_text(json.dumps(payload), encoding="utf-8")

    def _read_metadata(self, object_path: Path) -> dict[str, Any]:
        metadata_path = self._metadata_path(object_path)
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    @staticmethod
    def _coerce_body(body: bytes | str | io.BufferedIOBase) -> bytes:
        if isinstance(body, bytes):
            return body
        if isinstance(body, str):
            return body.encode("utf-8")
        return body.read()

    @staticmethod
    def _slice_range(data: bytes, byte_range: str) -> bytes:
        if not byte_range.startswith("bytes="):
            return data
        start_text, _, end_text = byte_range.removeprefix("bytes=").partition("-")
        start = int(start_text or 0)
        end = int(end_text) if end_text else len(data) - 1
        return data[start : end + 1]


class LocalObjectStorageHandler:
    """S3Handler-compatible local storage implementation."""

    def __init__(self, root_path: str | os.PathLike[str] | None = None) -> None:
        settings = get_settings()
        configured_root = (
            root_path
            or os.environ.get("RISKPULSE_LOCAL_STORAGE_ROOT")
            or settings.get("storage.local.root", ".local_storage")
        )
        self._root_path = Path(configured_root)
        self._s3_client = LocalObjectStorageClient(self._root_path)
        self._metrics = S3Metrics()
        logger.info("local_object_storage_initialized", root_path=str(self._s3_client.root_path))

    @property
    def metrics(self) -> S3Metrics:
        return self._metrics

    def upload_transactions(
        self,
        transactions: list[dict[str, Any]],
        timestamp: datetime | None = None,
        bucket: str = S3_BUCKET_RAW,
        prefix: str = "transactions",
    ) -> str:
        if not transactions:
            raise ValueError("Cannot upload empty transaction list")

        from src.storage.s3_handler import _build_partition_path

        partition_path = _build_partition_path(prefix, timestamp)
        key = _generate_file_key(partition_path, "parquet")
        parquet_buffer = self._records_to_parquet(transactions)
        return self.upload_raw_file(
            parquet_buffer.getvalue(),
            key,
            bucket=bucket,
            content_type="application/x-parquet",
            metadata={
                "record_count": str(len(transactions)),
                "partition_timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
                "compression": "snappy",
            },
        )

    def upload_raw_file(
        self,
        data: bytes,
        s3_key: str,
        bucket: str = S3_BUCKET_RAW,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        try:
            self._s3_client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=data,
                ContentType=content_type,
                Metadata=metadata or {},
            )
            self._metrics.record_upload(len(data))
            return s3_key
        except Exception as exc:
            self._metrics.record_upload_error()
            raise S3UploadError(f"Failed to write local://{bucket}/{s3_key}: {exc}") from exc

    def upload_large_batch(
        self,
        transactions: list[dict[str, Any]],
        timestamp: datetime | None = None,
        bucket: str = S3_BUCKET_RAW,
        prefix: str = "transactions",
        max_records_per_file: int = 100_000,
    ) -> list[str]:
        if not transactions:
            raise ValueError("Cannot upload empty transaction list")
        return [
            self.upload_transactions(
                transactions[start : start + max_records_per_file], timestamp, bucket, prefix
            )
            for start in range(0, len(transactions), max_records_per_file)
        ]

    def write_batch(
        self,
        records: list[dict[str, Any]],
        storage_layer: StorageLayer,
        partition_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        bucket = _bucket_for_storage_layer(storage_layer)
        normalized_partition = _normalize_object_prefix(partition_path)
        object_metadata = {
            "record_count": str(len(records)),
            "storage_layer": storage_layer.value,
        }
        if metadata:
            object_metadata.update({str(key): str(value) for key, value in metadata.items()})

        if not records:
            key = _generate_file_key(normalized_partition or storage_layer.value, "empty.json")
            return self.upload_raw_file(
                b"[]",
                key,
                bucket=bucket,
                content_type="application/json",
                metadata=object_metadata,
            )

        key = _generate_file_key(normalized_partition or storage_layer.value, "parquet")
        parquet_buffer = self._records_to_parquet(records)
        return self.upload_raw_file(
            parquet_buffer.getvalue(),
            key,
            bucket=bucket,
            content_type="application/x-parquet",
            metadata=object_metadata,
        )

    def download_parquet(self, bucket: str, key: str) -> pa.Table:
        try:
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
            body = response["Body"].read()
            self._metrics.record_download(len(body))
            return pq.read_table(io.BytesIO(body))
        except Exception as exc:
            self._metrics.record_download_error()
            raise S3DownloadError(f"Failed to read local://{bucket}/{key}: {exc}") from exc

    def read_batch(self, storage_layer: StorageLayer, partition_path: str) -> list[dict[str, Any]]:
        bucket = _bucket_for_storage_layer(storage_layer)
        prefix = _normalize_object_prefix(partition_path)
        records: list[dict[str, Any]] = []
        response = self._s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        for obj in response.get("Contents", []):
            key = obj.get("Key", "")
            if not key.endswith(".parquet"):
                continue
            records.extend(self.download_parquet(bucket, key).to_pylist())

        return records

    def stream_download(
        self,
        bucket: str,
        key: str,
        chunk_size: int = MULTIPART_CHUNK_SIZE,
    ) -> Generator[bytes, None, None]:
        response = self._s3_client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        total_bytes = 0
        while True:
            chunk = body.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            yield chunk
        self._metrics.record_download(total_bytes)

    def list_partition(
        self,
        bucket: str,
        prefix: str,
        timestamp: datetime | None = None,
    ) -> list[str]:
        from src.storage.s3_handler import _build_partition_path

        partition_path = _build_partition_path(prefix, timestamp)
        response = self._s3_client.list_objects_v2(Bucket=bucket, Prefix=partition_path)
        return [obj["Key"] for obj in response.get("Contents", [])]

    def file_exists(self, bucket: str, key: str) -> bool:
        try:
            self._s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except LocalObjectNotFoundError:
            return False

    def get_file_metadata(self, bucket: str, key: str) -> dict[str, Any]:
        response = self._s3_client.head_object(Bucket=bucket, Key=key)
        return {
            "content_length": response["ContentLength"],
            "content_type": response.get("ContentType", ""),
            "last_modified": response["LastModified"],
            "metadata": response.get("Metadata", {}),
        }

    def configure_event_notifications(
        self,
        bucket: str,
        sqs_queue_arn: str,
        prefix_filter: str = "",
        suffix_filter: str = ".parquet",
        events: list[str] | None = None,
    ) -> None:
        self._s3_client.put_bucket_notification_configuration(
            Bucket=bucket,
            NotificationConfiguration={
                "local_notification_target": sqs_queue_arn,
                "prefix_filter": prefix_filter,
                "suffix_filter": suffix_filter,
                "events": events or ["s3:ObjectCreated:*"],
            },
        )

    def close(self) -> None:
        logger.info("local_object_storage_closed", metrics=self._metrics.snapshot())

    @staticmethod
    def _records_to_parquet(records: list[dict[str, Any]]) -> io.BytesIO:
        table = pa.Table.from_pylist(records)
        buffer = io.BytesIO()
        pq.write_table(
            table,
            buffer,
            compression="snappy",
            use_dictionary=True,
            write_statistics=True,
        )
        buffer.seek(0)
        return buffer
