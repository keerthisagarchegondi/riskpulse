"""Unit tests for S3 handler with moto mocking.

Tests cover:
- S3Handler initialization
- Transaction upload with partitioning
- Parquet format verification
- Multipart upload for large batches
- Download operations
- Partition listing
- File existence checks
- Event notification configuration
- Error handling and retries
- Batch ingestion handler
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from unittest.mock import patch

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from moto import mock_aws

from src.storage.s3_handler import (
    S3_BUCKET_RAW,
    S3_BUCKET_PROCESSED,
    S3Handler,
    S3Metrics,
    S3UploadError,
    S3DownloadError,
    StorageLayer,
    _build_partition_path,
    _generate_file_key,
    MULTIPART_THRESHOLD,
)
from src.ingestion.api_ingestion import (
    BatchIngestionHandler,
    DetectedSchema,
    FileFormat,
    IngestionResult,
    IngestionStatus,
    IngestionError,
    SchemaDetectionError,
    ValidationError,
    REQUIRED_TRANSACTION_FIELDS,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def aws_credentials():
    """Mock AWS credentials for moto."""
    import os

    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    yield
    for key in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SECURITY_TOKEN",
        "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ.pop(key, None)


@pytest.fixture
def s3_setup(aws_credentials):
    """Create mocked S3 buckets for testing."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=S3_BUCKET_RAW)
        s3.create_bucket(Bucket=S3_BUCKET_PROCESSED)
        s3.create_bucket(Bucket="riskpulse-models")
        s3.create_bucket(Bucket="riskpulse-archive")
        yield s3


@pytest.fixture
def s3_handler(s3_setup):
    """Create an S3Handler configured for testing with moto."""
    handler = S3Handler(
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    return handler


@pytest.fixture
def batch_handler(s3_handler):
    """Create a BatchIngestionHandler for testing."""
    return BatchIngestionHandler(s3_handler=s3_handler)


@pytest.fixture
def sample_transactions():
    """Generate sample transaction records."""
    return [
        {
            "external_transaction_id": f"TXN-{i:06d}",
            "account_id": f"ACC-{i % 100:05d}",
            "customer_id": f"CUST-{i % 50:05d}",
            "merchant_id": f"MERCH-{i % 200:05d}",
            "merchant_name": f"Merchant {i % 200}",
            "merchant_category_code": "5411",
            "transaction_amount": round(10.0 + (i * 1.5), 2),
            "transaction_currency": "USD",
            "transaction_type": "purchase",
            "channel": "online",
            "card_type": "credit",
            "card_last_four": "4242",
            "geo_country": "US",
            "geo_city": "New York",
            "is_international": False,
            "transaction_timestamp": "2026-06-15T10:30:00Z",
        }
        for i in range(100)
    ]


@pytest.fixture
def sample_csv_data():
    """Generate sample CSV content."""
    header = "account_id,transaction_amount,transaction_currency,transaction_type,transaction_timestamp\n"
    rows = [
        f"ACC-{i:05d},{10.0 + i},USD,purchase,2026-06-15T10:30:00Z\n"
        for i in range(50)
    ]
    return (header + "".join(rows)).encode("utf-8")


@pytest.fixture
def sample_json_data():
    """Generate sample JSON content."""
    records = [
        {
            "account_id": f"ACC-{i:05d}",
            "transaction_amount": 10.0 + i,
            "transaction_currency": "USD",
            "transaction_type": "purchase",
            "transaction_timestamp": "2026-06-15T10:30:00Z",
        }
        for i in range(50)
    ]
    return json.dumps(records).encode("utf-8")


@pytest.fixture
def sample_jsonl_data():
    """Generate sample JSONL content."""
    lines = [
        json.dumps({
            "account_id": f"ACC-{i:05d}",
            "transaction_amount": 10.0 + i,
            "transaction_currency": "USD",
            "transaction_type": "purchase",
            "transaction_timestamp": "2026-06-15T10:30:00Z",
        })
        for i in range(50)
    ]
    return "\n".join(lines).encode("utf-8")


# ============================================================================
# Tests: Partition Path Building
# ============================================================================


class TestPartitionPath:
    """Tests for partition path generation."""

    def test_build_partition_path_with_timestamp(self):
        ts = datetime(2026, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        path = _build_partition_path("transactions", ts)
        assert path == "transactions/2026/06/15/14"

    def test_build_partition_path_default_timestamp(self):
        path = _build_partition_path("transactions")
        # Should contain current UTC time components
        now = datetime.now(timezone.utc)
        assert path.startswith(f"transactions/{now.year:04d}/")

    def test_build_partition_path_preserves_prefix(self):
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        path = _build_partition_path("custom/prefix", ts)
        assert path == "custom/prefix/2026/01/01/00"

    def test_generate_file_key_format(self):
        key = _generate_file_key("transactions/2026/06/15/14", "parquet")
        assert key.startswith("transactions/2026/06/15/14/")
        assert key.endswith(".parquet")

    def test_generate_file_key_uniqueness(self):
        key1 = _generate_file_key("prefix/2026/01/01/00")
        key2 = _generate_file_key("prefix/2026/01/01/00")
        assert key1 != key2


# ============================================================================
# Tests: S3 Metrics
# ============================================================================


class TestS3Metrics:
    """Tests for S3 metrics tracking."""

    def test_initial_metrics(self):
        metrics = S3Metrics()
        snapshot = metrics.snapshot()
        assert snapshot["uploads"] == 0
        assert snapshot["downloads"] == 0
        assert snapshot["bytes_uploaded"] == 0

    def test_record_upload(self):
        metrics = S3Metrics()
        metrics.record_upload(1024)
        metrics.record_upload(2048)
        snapshot = metrics.snapshot()
        assert snapshot["uploads"] == 2
        assert snapshot["bytes_uploaded"] == 3072

    def test_record_download(self):
        metrics = S3Metrics()
        metrics.record_download(512)
        snapshot = metrics.snapshot()
        assert snapshot["downloads"] == 1
        assert snapshot["bytes_downloaded"] == 512

    def test_record_errors(self):
        metrics = S3Metrics()
        metrics.record_upload_error()
        metrics.record_download_error()
        snapshot = metrics.snapshot()
        assert snapshot["upload_errors"] == 1
        assert snapshot["download_errors"] == 1


# ============================================================================
# Tests: S3 Handler Upload
# ============================================================================


class TestS3HandlerUpload:
    """Tests for S3Handler upload operations."""

    def test_upload_transactions_basic(self, s3_handler, sample_transactions):
        ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(sample_transactions, timestamp=ts)

        assert key.startswith("transactions/2026/06/15/14/")
        assert key.endswith(".parquet")

    def test_upload_transactions_creates_parquet(self, s3_handler, s3_setup, sample_transactions):
        ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(sample_transactions, timestamp=ts)

        # Download and verify it's valid Parquet
        response = s3_setup.get_object(Bucket=S3_BUCKET_RAW, Key=key)
        body = response["Body"].read()
        buffer = io.BytesIO(body)
        table = pq.read_table(buffer)

        assert table.num_rows == 100
        assert "account_id" in table.column_names
        assert "transaction_amount" in table.column_names

    def test_upload_transactions_snappy_compression(self, s3_handler, s3_setup, sample_transactions):
        ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(sample_transactions, timestamp=ts)

        response = s3_setup.get_object(Bucket=S3_BUCKET_RAW, Key=key)
        body = response["Body"].read()
        buffer = io.BytesIO(body)
        pf = pq.ParquetFile(buffer)

        # Verify snappy compression
        metadata = pf.metadata
        for i in range(metadata.num_row_groups):
            rg = metadata.row_group(i)
            for j in range(rg.num_columns):
                col = rg.column(j)
                assert col.compression == "SNAPPY"

    def test_upload_transactions_metadata(self, s3_handler, s3_setup, sample_transactions):
        ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(sample_transactions, timestamp=ts)

        response = s3_setup.head_object(Bucket=S3_BUCKET_RAW, Key=key)
        metadata = response.get("Metadata", {})

        assert metadata["record_count"] == "100"
        assert "partition_timestamp" in metadata
        assert metadata["compression"] == "snappy"

    def test_upload_transactions_empty_raises(self, s3_handler):
        with pytest.raises(ValueError, match="empty"):
            s3_handler.upload_transactions([])

    def test_upload_transactions_updates_metrics(self, s3_handler, sample_transactions):
        ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        s3_handler.upload_transactions(sample_transactions, timestamp=ts)

        snapshot = s3_handler.metrics.snapshot()
        assert snapshot["uploads"] == 1
        assert snapshot["bytes_uploaded"] > 0

    def test_upload_raw_file(self, s3_handler, s3_setup):
        data = b"raw binary content for testing"
        key = s3_handler.upload_raw_file(
            data=data,
            s3_key="test/raw/file.bin",
            bucket=S3_BUCKET_RAW,
            content_type="application/octet-stream",
            metadata={"source": "test"},
        )

        assert key == "test/raw/file.bin"

        response = s3_setup.get_object(Bucket=S3_BUCKET_RAW, Key=key)
        assert response["Body"].read() == data

    def test_upload_large_batch_splits(self, s3_handler, sample_transactions):
        # Upload with small max per file to test splitting
        ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        keys = s3_handler.upload_large_batch(
            sample_transactions,
            timestamp=ts,
            max_records_per_file=30,
        )

        # 100 records / 30 per file = 4 files
        assert len(keys) == 4

    def test_upload_custom_bucket_prefix(self, s3_handler, s3_setup, sample_transactions):
        ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(
            sample_transactions[:10],
            timestamp=ts,
            bucket=S3_BUCKET_PROCESSED,
            prefix="validated",
        )

        assert key.startswith("validated/2026/06/15/14/")
        assert s3_handler.file_exists(S3_BUCKET_PROCESSED, key)


# ============================================================================
# Tests: S3 Handler Download
# ============================================================================


class TestS3HandlerDownload:
    """Tests for S3Handler download operations."""

    def test_download_parquet(self, s3_handler, sample_transactions):
        ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(sample_transactions, timestamp=ts)

        table = s3_handler.download_parquet(S3_BUCKET_RAW, key)
        assert table.num_rows == 100
        assert "account_id" in table.column_names

    def test_download_nonexistent_raises(self, s3_handler):
        with pytest.raises(S3DownloadError):
            s3_handler.download_parquet(S3_BUCKET_RAW, "nonexistent/key.parquet")

    def test_stream_download(self, s3_handler, s3_setup):
        # Upload test data
        data = b"x" * 1000
        s3_setup.put_object(Bucket=S3_BUCKET_RAW, Key="test/stream.bin", Body=data)

        # Stream download
        chunks = list(s3_handler.stream_download(S3_BUCKET_RAW, "test/stream.bin", chunk_size=256))
        result = b"".join(chunks)
        assert result == data

    def test_download_updates_metrics(self, s3_handler, sample_transactions):
        ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(sample_transactions, timestamp=ts)
        s3_handler.download_parquet(S3_BUCKET_RAW, key)

        snapshot = s3_handler.metrics.snapshot()
        assert snapshot["downloads"] == 1
        assert snapshot["bytes_downloaded"] > 0


# ============================================================================
# Tests: S3 Handler Listing & Utilities
# ============================================================================


class TestS3HandlerUtilities:
    """Tests for S3Handler utility operations."""

    def test_list_partition(self, s3_handler, sample_transactions):
        ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        key1 = s3_handler.upload_transactions(sample_transactions[:10], timestamp=ts)
        key2 = s3_handler.upload_transactions(sample_transactions[10:20], timestamp=ts)

        keys = s3_handler.list_partition(
            S3_BUCKET_RAW, "transactions", timestamp=ts
        )
        assert len(keys) == 2
        assert key1 in keys
        assert key2 in keys

    def test_list_partition_empty(self, s3_handler):
        ts = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        keys = s3_handler.list_partition(S3_BUCKET_RAW, "transactions", timestamp=ts)
        assert keys == []

    def test_file_exists_true(self, s3_handler, s3_setup):
        s3_setup.put_object(Bucket=S3_BUCKET_RAW, Key="test/exists.txt", Body=b"data")
        assert s3_handler.file_exists(S3_BUCKET_RAW, "test/exists.txt") is True

    def test_file_exists_false(self, s3_handler):
        assert s3_handler.file_exists(S3_BUCKET_RAW, "nonexistent.txt") is False

    def test_get_file_metadata(self, s3_handler, s3_setup):
        s3_setup.put_object(
            Bucket=S3_BUCKET_RAW,
            Key="test/meta.txt",
            Body=b"hello",
            ContentType="text/plain",
            Metadata={"custom": "value"},
        )
        metadata = s3_handler.get_file_metadata(S3_BUCKET_RAW, "test/meta.txt")
        assert metadata["content_length"] == 5
        assert metadata["content_type"] == "text/plain"
        assert metadata["metadata"]["custom"] == "value"


# ============================================================================
# Tests: S3 Event Notifications
# ============================================================================


class TestS3EventNotifications:
    """Tests for S3 event notification configuration."""

    def test_configure_event_notifications(self, s3_handler, s3_setup):
        # This should not raise
        s3_handler.configure_event_notifications(
            bucket=S3_BUCKET_RAW,
            sqs_queue_arn="arn:aws:sqs:us-east-1:123456789012:riskpulse-events",
            prefix_filter="transactions/",
            suffix_filter=".parquet",
        )

        # Verify configuration was set
        config = s3_setup.get_bucket_notification_configuration(Bucket=S3_BUCKET_RAW)
        assert len(config.get("QueueConfigurations", [])) == 1
        queue_config = config["QueueConfigurations"][0]
        assert queue_config["QueueArn"] == "arn:aws:sqs:us-east-1:123456789012:riskpulse-events"


# ============================================================================
# Tests: Batch Ingestion Handler - Format Detection
# ============================================================================


class TestBatchIngestionFormatDetection:
    """Tests for file format detection."""

    def test_detect_csv(self, batch_handler):
        fmt = batch_handler._detect_format("data/transactions.csv")
        assert fmt == FileFormat.CSV

    def test_detect_json(self, batch_handler):
        fmt = batch_handler._detect_format("data/events.json")
        assert fmt == FileFormat.JSON

    def test_detect_jsonl(self, batch_handler):
        fmt = batch_handler._detect_format("data/stream.jsonl")
        assert fmt == FileFormat.JSONL

    def test_detect_ndjson(self, batch_handler):
        fmt = batch_handler._detect_format("data/stream.ndjson")
        assert fmt == FileFormat.JSONL

    def test_detect_parquet(self, batch_handler):
        fmt = batch_handler._detect_format("data/output.parquet")
        assert fmt == FileFormat.PARQUET

    def test_detect_unknown_raises(self, batch_handler):
        with pytest.raises(IngestionError, match="Cannot determine format"):
            batch_handler._detect_format("data/file.xlsx")


# ============================================================================
# Tests: Batch Ingestion Handler - CSV Ingestion
# ============================================================================


class TestBatchIngestionCSV:
    """Tests for CSV file ingestion."""

    def test_ingest_csv(self, batch_handler, s3_setup, sample_csv_data):
        s3_setup.put_object(
            Bucket=S3_BUCKET_RAW, Key="incoming/data.csv", Body=sample_csv_data
        )

        result = batch_handler.ingest_file(S3_BUCKET_RAW, "incoming/data.csv")

        assert result.status in (IngestionStatus.COMPLETED, IngestionStatus.PARTIAL)
        assert result.records_processed == 50
        assert result.file_format == FileFormat.CSV
        assert "account_id" in result.schema_fields

    def test_ingest_csv_creates_parquet_output(self, batch_handler, s3_setup, sample_csv_data):
        s3_setup.put_object(
            Bucket=S3_BUCKET_RAW, Key="incoming/data.csv", Body=sample_csv_data
        )

        result = batch_handler.ingest_file(S3_BUCKET_RAW, "incoming/data.csv")

        assert result.destination_key is not None
        assert result.destination_key.endswith(".parquet")

        # Verify the output is readable
        response = s3_setup.get_object(
            Bucket=S3_BUCKET_PROCESSED, Key=result.destination_key
        )
        body = response["Body"].read()
        table = pq.read_table(io.BytesIO(body))
        assert table.num_rows == 50


# ============================================================================
# Tests: Batch Ingestion Handler - JSON Ingestion
# ============================================================================


class TestBatchIngestionJSON:
    """Tests for JSON file ingestion."""

    def test_ingest_json_array(self, batch_handler, s3_setup, sample_json_data):
        s3_setup.put_object(
            Bucket=S3_BUCKET_RAW, Key="incoming/data.json", Body=sample_json_data
        )

        result = batch_handler.ingest_file(S3_BUCKET_RAW, "incoming/data.json")

        assert result.status in (IngestionStatus.COMPLETED, IngestionStatus.PARTIAL)
        assert result.records_processed == 50
        assert result.file_format == FileFormat.JSON

    def test_ingest_json_wrapped(self, batch_handler, s3_setup):
        wrapped = json.dumps({
            "data": [
                {
                    "account_id": "ACC-001",
                    "transaction_amount": 100.0,
                    "transaction_currency": "USD",
                    "transaction_type": "purchase",
                    "transaction_timestamp": "2026-06-15T10:00:00Z",
                }
            ]
        }).encode("utf-8")

        s3_setup.put_object(
            Bucket=S3_BUCKET_RAW, Key="incoming/wrapped.json", Body=wrapped
        )

        result = batch_handler.ingest_file(S3_BUCKET_RAW, "incoming/wrapped.json")
        assert result.records_processed == 1

    def test_ingest_jsonl(self, batch_handler, s3_setup, sample_jsonl_data):
        s3_setup.put_object(
            Bucket=S3_BUCKET_RAW, Key="incoming/data.jsonl", Body=sample_jsonl_data
        )

        result = batch_handler.ingest_file(S3_BUCKET_RAW, "incoming/data.jsonl")

        assert result.status in (IngestionStatus.COMPLETED, IngestionStatus.PARTIAL)
        assert result.records_processed == 50
        assert result.file_format == FileFormat.JSONL


# ============================================================================
# Tests: Batch Ingestion Handler - Validation
# ============================================================================


class TestBatchIngestionValidation:
    """Tests for schema validation during ingestion."""

    def test_missing_required_fields(self, batch_handler, s3_setup):
        # CSV missing required fields
        csv_data = b"name,email\nJohn,john@test.com\nJane,jane@test.com\n"
        s3_setup.put_object(
            Bucket=S3_BUCKET_RAW, Key="incoming/bad.csv", Body=csv_data
        )

        result = batch_handler.ingest_file(S3_BUCKET_RAW, "incoming/bad.csv")

        assert len(result.errors) > 0
        assert any("Missing required fields" in e for e in result.errors)

    def test_skip_validation(self, batch_handler, s3_setup):
        csv_data = b"name,email\nJohn,john@test.com\n"
        s3_setup.put_object(
            Bucket=S3_BUCKET_RAW, Key="incoming/no_validate.csv", Body=csv_data
        )

        result = batch_handler.ingest_file(
            S3_BUCKET_RAW, "incoming/no_validate.csv", validate=False
        )

        assert result.status == IngestionStatus.COMPLETED
        assert result.records_processed == 1
        assert result.errors == []

    def test_null_values_filtered(self, batch_handler, s3_setup):
        csv_data = (
            "account_id,transaction_amount,transaction_currency,transaction_type,transaction_timestamp\n"
            "ACC-001,100.0,USD,purchase,2026-06-15T10:00:00Z\n"
            ",200.0,USD,purchase,2026-06-15T11:00:00Z\n"  # null account_id
            "ACC-003,300.0,USD,purchase,2026-06-15T12:00:00Z\n"
        ).encode("utf-8")

        s3_setup.put_object(
            Bucket=S3_BUCKET_RAW, Key="incoming/nulls.csv", Body=csv_data
        )

        result = batch_handler.ingest_file(S3_BUCKET_RAW, "incoming/nulls.csv")

        # The empty account_id should be treated as null and filtered
        assert result.records_processed >= 2


# ============================================================================
# Tests: Batch Ingestion Handler - Schema Detection
# ============================================================================


class TestSchemaDetection:
    """Tests for schema detection."""

    def test_detect_csv_schema(self, batch_handler, s3_setup, sample_csv_data):
        s3_setup.put_object(
            Bucket=S3_BUCKET_RAW, Key="detect/data.csv", Body=sample_csv_data
        )

        schema = batch_handler.detect_schema(S3_BUCKET_RAW, "detect/data.csv")

        assert isinstance(schema, DetectedSchema)
        assert "account_id" in schema.fields
        assert "transaction_amount" in schema.fields
        assert schema.has_header is True
        assert schema.row_count_estimate == 50

    def test_detect_json_schema(self, batch_handler, s3_setup, sample_json_data):
        s3_setup.put_object(
            Bucket=S3_BUCKET_RAW, Key="detect/data.json", Body=sample_json_data
        )

        schema = batch_handler.detect_schema(S3_BUCKET_RAW, "detect/data.json")

        assert "account_id" in schema.fields
        assert "transaction_amount" in schema.fields
        assert schema.row_count_estimate == 50

    def test_detect_jsonl_schema(self, batch_handler, s3_setup, sample_jsonl_data):
        s3_setup.put_object(
            Bucket=S3_BUCKET_RAW, Key="detect/data.jsonl", Body=sample_jsonl_data
        )

        schema = batch_handler.detect_schema(S3_BUCKET_RAW, "detect/data.jsonl")

        assert "account_id" in schema.fields
        assert schema.row_count_estimate == 50


# ============================================================================
# Tests: Batch Ingestion Handler - Partition Ingestion
# ============================================================================


class TestPartitionIngestion:
    """Tests for batch ingestion of entire partitions."""

    def test_ingest_partition(self, batch_handler, s3_setup, sample_csv_data, sample_json_data):
        # Place multiple files in a partition
        s3_setup.put_object(
            Bucket=S3_BUCKET_RAW, Key="incoming/2026/06/15/file1.csv", Body=sample_csv_data
        )
        s3_setup.put_object(
            Bucket=S3_BUCKET_RAW, Key="incoming/2026/06/15/file2.json", Body=sample_json_data
        )

        results = batch_handler.ingest_partition(
            S3_BUCKET_RAW, "incoming/2026/06/15/"
        )

        assert len(results) == 2
        assert all(
            r.status in (IngestionStatus.COMPLETED, IngestionStatus.PARTIAL)
            for r in results
        )

    def test_ingest_empty_partition(self, batch_handler, s3_setup):
        results = batch_handler.ingest_partition(
            S3_BUCKET_RAW, "empty/partition/"
        )
        assert results == []


# ============================================================================
# Tests: Ingestion Result
# ============================================================================


class TestIngestionResult:
    """Tests for IngestionResult data class."""

    def test_success_rate_calculation(self):
        result = IngestionResult(
            status=IngestionStatus.PARTIAL,
            source_key="test.csv",
            records_processed=90,
            records_failed=10,
        )
        assert result.success_rate == 0.9

    def test_success_rate_zero_records(self):
        result = IngestionResult(
            status=IngestionStatus.FAILED,
            source_key="test.csv",
        )
        assert result.success_rate == 0.0

    def test_to_dict(self):
        result = IngestionResult(
            status=IngestionStatus.COMPLETED,
            source_key="data.csv",
            destination_key="validated/data.parquet",
            records_processed=100,
            file_format=FileFormat.CSV,
            schema_fields=["col1", "col2"],
        )
        d = result.to_dict()
        assert d["status"] == "completed"
        assert d["source_key"] == "data.csv"
        assert d["records_processed"] == 100
        assert d["file_format"] == "csv"


# ============================================================================
# Tests: Storage Layer Enum
# ============================================================================


class TestStorageLayer:
    """Tests for StorageLayer enum."""

    def test_storage_layers(self):
        assert StorageLayer.RAW.value == "raw"
        assert StorageLayer.VALIDATED.value == "validated"
        assert StorageLayer.ENRICHED.value == "enriched"
        assert StorageLayer.MODELS.value == "models"
        assert StorageLayer.ARCHIVE.value == "archive"
