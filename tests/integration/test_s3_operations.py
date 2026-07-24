"""Integration tests for S3 operations using LocalStack.

Tests cover:
- Bucket creation and configuration
- Object upload/download with partitioning
- Lifecycle policy application
- Event notifications via SQS
- Encryption enforcement
- Cross-service access patterns
- Large batch uploads (multipart)
"""

from __future__ import annotations

import io
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Generator

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from src.storage.s3_handler import (
    S3Handler,
    S3_BUCKET_ARCHIVE,
    S3_BUCKET_MODELS,
    S3_BUCKET_PROCESSED,
    S3_BUCKET_RAW,
    StorageLayer,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCALSTACK_ENDPOINT = "http://localhost:4566"
TEST_REGION = "us-east-1"

BUCKET_NAMES = [
    S3_BUCKET_RAW,
    S3_BUCKET_PROCESSED,
    S3_BUCKET_MODELS,
    S3_BUCKET_ARCHIVE,
]

SQS_QUEUE_NAME = "riskpulse-s3-events-test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def aws_credentials() -> dict[str, str]:
    """Provide dummy AWS credentials for LocalStack."""
    return {
        "aws_access_key_id": "testing",
        "aws_secret_access_key": "testing",
        "region_name": TEST_REGION,
    }


@pytest.fixture(scope="module")
def s3_client(aws_credentials: dict[str, str]) -> Any:
    """Create a boto3 S3 client pointing to LocalStack."""
    return boto3.client(
        "s3",
        endpoint_url=LOCALSTACK_ENDPOINT,
        **aws_credentials,
        config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}),
    )


@pytest.fixture(scope="module")
def sqs_client(aws_credentials: dict[str, str]) -> Any:
    """Create a boto3 SQS client pointing to LocalStack."""
    return boto3.client(
        "sqs",
        endpoint_url=LOCALSTACK_ENDPOINT,
        **aws_credentials,
    )


@pytest.fixture(scope="module")
def kms_client(aws_credentials: dict[str, str]) -> Any:
    """Create a boto3 KMS client pointing to LocalStack."""
    return boto3.client(
        "kms",
        endpoint_url=LOCALSTACK_ENDPOINT,
        **aws_credentials,
    )


@pytest.fixture(scope="module")
def kms_key_arn(kms_client: Any) -> str:
    """Create a KMS key in LocalStack and return its ARN."""
    response = kms_client.create_key(
        Description="RiskPulse S3 test encryption key",
        KeyUsage="ENCRYPT_DECRYPT",
    )
    return response["KeyMetadata"]["Arn"]


@pytest.fixture(scope="module")
def sqs_queue_url(sqs_client: Any) -> str:
    """Create an SQS queue in LocalStack for S3 event notifications."""
    response = sqs_client.create_queue(
        QueueName=SQS_QUEUE_NAME,
        Attributes={
            "VisibilityTimeout": "30",
            "MessageRetentionPeriod": "86400",
        },
    )
    return response["QueueUrl"]


@pytest.fixture(scope="module")
def sqs_queue_arn(sqs_client: Any, sqs_queue_url: str) -> str:
    """Get the ARN of the test SQS queue."""
    attrs = sqs_client.get_queue_attributes(
        QueueUrl=sqs_queue_url,
        AttributeNames=["QueueArn"],
    )
    return attrs["Attributes"]["QueueArn"]


@pytest.fixture(scope="module")
def s3_buckets(s3_client: Any) -> list[str]:
    """Create all data lake buckets in LocalStack."""
    for bucket_name in BUCKET_NAMES:
        try:
            s3_client.create_bucket(Bucket=bucket_name)
        except ClientError as e:
            if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                raise
    return BUCKET_NAMES


@pytest.fixture(scope="module")
def s3_handler(s3_buckets: list[str]) -> Generator[S3Handler, None, None]:
    """Create an S3Handler instance configured for LocalStack."""
    handler = S3Handler(
        region_name=TEST_REGION,
        endpoint_url=LOCALSTACK_ENDPOINT,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    yield handler
    handler.close()


@pytest.fixture
def sample_transactions() -> list[dict[str, Any]]:
    """Generate sample transaction records."""
    return [
        {
            "transaction_id": str(uuid.uuid4()),
            "account_id": f"ACC-{i:05d}",
            "customer_id": f"CUST-{i:05d}",
            "merchant_id": f"MERCH-{i % 100:04d}",
            "merchant_name": f"Test Merchant {i % 100}",
            "transaction_amount": round(50.0 + (i * 7.31 % 950), 2),
            "transaction_currency": "USD",
            "transaction_type": ["purchase", "withdrawal", "transfer", "refund"][i % 4],
            "channel": ["online", "pos", "atm", "mobile"][i % 4],
            "geo_country": ["US", "GB", "DE", "FR", "CA"][i % 5],
            "geo_city": ["New York", "London", "Berlin", "Paris", "Toronto"][i % 5],
            "is_international": i % 5 != 0,
            "transaction_timestamp": datetime(2026, 7, 15, 10, i % 60, 0).isoformat(),
        }
        for i in range(100)
    ]


@pytest.fixture
def large_batch_transactions() -> list[dict[str, Any]]:
    """Generate a large batch of transactions for multipart upload testing."""
    return [
        {
            "transaction_id": str(uuid.uuid4()),
            "account_id": f"ACC-{i:06d}",
            "customer_id": f"CUST-{i:06d}",
            "merchant_id": f"MERCH-{i % 500:04d}",
            "transaction_amount": round(10.0 + (i * 3.14 % 5000), 2),
            "transaction_currency": "USD",
            "transaction_type": "purchase",
            "channel": "online",
            "geo_country": "US",
            "is_international": False,
            "transaction_timestamp": datetime(2026, 7, 15, i % 24, i % 60, 0).isoformat(),
        }
        for i in range(5000)
    ]


# ---------------------------------------------------------------------------
# Tests: Bucket Architecture
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBucketArchitecture:
    """Validate S3 bucket creation and zone separation."""

    def test_all_buckets_exist(self, s3_client: Any, s3_buckets: list[str]) -> None:
        """All four data lake zones should have corresponding buckets."""
        response = s3_client.list_buckets()
        existing = {b["Name"] for b in response["Buckets"]}

        for bucket in BUCKET_NAMES:
            assert bucket in existing, f"Bucket {bucket} was not created"

    def test_bucket_zone_isolation(
        self, s3_handler: S3Handler, sample_transactions: list[dict[str, Any]]
    ) -> None:
        """Data written to raw zone should not appear in processed zone."""
        ts = datetime(2026, 7, 20, 14, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(
            sample_transactions[:10],
            timestamp=ts,
            bucket=S3_BUCKET_RAW,
        )

        # Verify file exists in raw
        assert s3_handler.file_exists(S3_BUCKET_RAW, key)

        # Verify file does NOT exist in processed
        assert not s3_handler.file_exists(S3_BUCKET_PROCESSED, key)


# ---------------------------------------------------------------------------
# Tests: Upload Operations
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestUploadOperations:
    """Validate object upload with partitioning and Parquet format."""

    def test_upload_transactions_creates_parquet(
        self, s3_handler: S3Handler, sample_transactions: list[dict[str, Any]]
    ) -> None:
        """Uploading transactions should create a valid Parquet file."""
        ts = datetime(2026, 7, 15, 10, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(
            sample_transactions,
            timestamp=ts,
            bucket=S3_BUCKET_RAW,
        )

        assert key is not None
        assert key.endswith(".parquet")
        assert "2026/07/15/10" in key

    def test_upload_partitioning_by_timestamp(
        self, s3_handler: S3Handler, sample_transactions: list[dict[str, Any]]
    ) -> None:
        """Files should be partitioned by year/month/day/hour."""
        timestamps = [
            datetime(2026, 1, 15, 8, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 6, 20, 14, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 12, 31, 23, 0, 0, tzinfo=timezone.utc),
        ]

        for ts in timestamps:
            key = s3_handler.upload_transactions(
                sample_transactions[:5],
                timestamp=ts,
                bucket=S3_BUCKET_RAW,
            )
            expected_partition = f"{ts.year:04d}/{ts.month:02d}/{ts.day:02d}/{ts.hour:02d}"
            assert expected_partition in key

    def test_upload_returns_unique_keys(
        self, s3_handler: S3Handler, sample_transactions: list[dict[str, Any]]
    ) -> None:
        """Multiple uploads at the same timestamp should produce unique keys."""
        ts = datetime(2026, 7, 15, 10, 0, 0, tzinfo=timezone.utc)
        keys = set()

        for _ in range(5):
            key = s3_handler.upload_transactions(
                sample_transactions[:5],
                timestamp=ts,
                bucket=S3_BUCKET_RAW,
            )
            keys.add(key)

        assert len(keys) == 5

    def test_upload_empty_list_raises(self, s3_handler: S3Handler) -> None:
        """Uploading an empty list should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            s3_handler.upload_transactions([], bucket=S3_BUCKET_RAW)

    def test_upload_raw_file(self, s3_handler: S3Handler) -> None:
        """Uploading raw bytes should store the file at the specified key."""
        content = b'{"test": "data", "value": 42}'
        key = "test/raw_upload_test.json"

        result_key = s3_handler.upload_raw_file(
            data=content,
            s3_key=key,
            bucket=S3_BUCKET_RAW,
            content_type="application/json",
            metadata={"source": "integration_test"},
        )

        assert result_key == key
        assert s3_handler.file_exists(S3_BUCKET_RAW, key)

    def test_upload_large_batch_splits_files(
        self, s3_handler: S3Handler, large_batch_transactions: list[dict[str, Any]]
    ) -> None:
        """Large batches should be split across multiple files."""
        ts = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
        keys = s3_handler.upload_large_batch(
            large_batch_transactions,
            timestamp=ts,
            bucket=S3_BUCKET_RAW,
            max_records_per_file=2000,
        )

        assert len(keys) == 3  # 5000 / 2000 = 3 files (ceil)

        # Verify all files exist
        for key in keys:
            assert s3_handler.file_exists(S3_BUCKET_RAW, key)


# ---------------------------------------------------------------------------
# Tests: Download Operations
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDownloadOperations:
    """Validate object download and Parquet reading."""

    def test_download_parquet_roundtrip(
        self, s3_handler: S3Handler, sample_transactions: list[dict[str, Any]]
    ) -> None:
        """Data should survive upload/download roundtrip in Parquet format."""
        ts = datetime(2026, 7, 16, 9, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(
            sample_transactions,
            timestamp=ts,
            bucket=S3_BUCKET_RAW,
        )

        table = s3_handler.download_parquet(S3_BUCKET_RAW, key)

        assert table.num_rows == len(sample_transactions)
        assert "transaction_id" in table.column_names
        assert "account_id" in table.column_names
        assert "transaction_amount" in table.column_names

    def test_download_nonexistent_key_raises(self, s3_handler: S3Handler) -> None:
        """Downloading a nonexistent key should raise S3DownloadError."""
        from src.storage.s3_handler import S3DownloadError

        with pytest.raises(S3DownloadError):
            s3_handler.download_parquet(S3_BUCKET_RAW, "nonexistent/key.parquet")

    def test_stream_download(
        self, s3_handler: S3Handler, sample_transactions: list[dict[str, Any]]
    ) -> None:
        """Stream download should return all data in chunks."""
        ts = datetime(2026, 7, 16, 11, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(
            sample_transactions[:50],
            timestamp=ts,
            bucket=S3_BUCKET_RAW,
        )

        chunks = list(s3_handler.stream_download(S3_BUCKET_RAW, key, chunk_size=1024))
        assert len(chunks) > 0

        # Reassemble and verify it's valid Parquet
        full_data = b"".join(chunks)
        table = pq.read_table(io.BytesIO(full_data))
        assert table.num_rows == 50


# ---------------------------------------------------------------------------
# Tests: Listing & Metadata
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestListingAndMetadata:
    """Validate partition listing and file metadata."""

    def test_list_partition(
        self, s3_handler: S3Handler, sample_transactions: list[dict[str, Any]]
    ) -> None:
        """Listing a partition should return all files in that partition."""
        ts = datetime(2026, 8, 1, 15, 0, 0, tzinfo=timezone.utc)

        # Upload 3 files to the same partition
        for _ in range(3):
            s3_handler.upload_transactions(
                sample_transactions[:10],
                timestamp=ts,
                bucket=S3_BUCKET_RAW,
            )

        keys = s3_handler.list_partition(
            bucket=S3_BUCKET_RAW,
            prefix="transactions",
            timestamp=ts,
        )

        assert len(keys) >= 3

    def test_file_metadata(
        self, s3_handler: S3Handler, sample_transactions: list[dict[str, Any]]
    ) -> None:
        """File metadata should include record count and compression info."""
        ts = datetime(2026, 8, 2, 10, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(
            sample_transactions[:20],
            timestamp=ts,
            bucket=S3_BUCKET_RAW,
        )

        metadata = s3_handler.get_file_metadata(S3_BUCKET_RAW, key)

        assert metadata["content_length"] > 0
        assert metadata["content_type"] == "application/x-parquet"
        assert metadata["metadata"]["record_count"] == "20"
        assert metadata["metadata"]["compression"] == "snappy"

    def test_file_exists_returns_false_for_missing(self, s3_handler: S3Handler) -> None:
        """file_exists should return False for nonexistent keys."""
        assert not s3_handler.file_exists(S3_BUCKET_RAW, "does/not/exist.parquet")


# ---------------------------------------------------------------------------
# Tests: Encryption
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEncryption:
    """Validate server-side encryption on uploaded objects."""

    def test_raw_bucket_uses_sse_s3(
        self,
        s3_client: Any,
        s3_handler: S3Handler,
        s3_buckets: list[str],
        sample_transactions: list[dict[str, Any]],
    ) -> None:
        """Objects in raw bucket should be encrypted with SSE-S3 (AES256)."""
        # Configure default encryption on raw bucket
        s3_client.put_bucket_encryption(
            Bucket=S3_BUCKET_RAW,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256"
                        }
                    }
                ]
            },
        )

        ts = datetime(2026, 8, 3, 10, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(
            sample_transactions[:5],
            timestamp=ts,
            bucket=S3_BUCKET_RAW,
        )

        # Verify encryption
        response = s3_client.head_object(Bucket=S3_BUCKET_RAW, Key=key)
        assert response.get("ServerSideEncryption") == "AES256"

    def test_processed_bucket_uses_sse_kms(
        self,
        s3_client: Any,
        s3_handler: S3Handler,
        s3_buckets: list[str],
        kms_key_arn: str,
        sample_transactions: list[dict[str, Any]],
    ) -> None:
        """Objects in processed bucket should be encrypted with SSE-KMS."""
        # Configure default encryption with KMS on processed bucket
        s3_client.put_bucket_encryption(
            Bucket=S3_BUCKET_PROCESSED,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "aws:kms",
                            "KMSMasterKeyID": kms_key_arn,
                        },
                        "BucketKeyEnabled": True,
                    }
                ]
            },
        )

        ts = datetime(2026, 8, 3, 11, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(
            sample_transactions[:5],
            timestamp=ts,
            bucket=S3_BUCKET_PROCESSED,
            prefix="validated",
        )

        response = s3_client.head_object(Bucket=S3_BUCKET_PROCESSED, Key=key)
        assert response.get("ServerSideEncryption") == "aws:kms"

    def test_models_bucket_uses_sse_kms(
        self,
        s3_client: Any,
        s3_handler: S3Handler,
        s3_buckets: list[str],
        kms_key_arn: str,
    ) -> None:
        """Objects in models bucket should be encrypted with SSE-KMS."""
        s3_client.put_bucket_encryption(
            Bucket=S3_BUCKET_MODELS,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "aws:kms",
                            "KMSMasterKeyID": kms_key_arn,
                        },
                        "BucketKeyEnabled": True,
                    }
                ]
            },
        )

        model_data = b"fake_model_binary_data_for_testing"
        key = "isolation_forest/v1.0.0/model.pkl"

        s3_handler.upload_raw_file(
            data=model_data,
            s3_key=key,
            bucket=S3_BUCKET_MODELS,
            content_type="application/octet-stream",
        )

        response = s3_client.head_object(Bucket=S3_BUCKET_MODELS, Key=key)
        assert response.get("ServerSideEncryption") == "aws:kms"


# ---------------------------------------------------------------------------
# Tests: Lifecycle Policy Configuration
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLifecyclePolicies:
    """Validate lifecycle policy configuration on buckets."""

    def test_raw_bucket_lifecycle_configured(
        self, s3_client: Any, s3_buckets: list[str]
    ) -> None:
        """Raw bucket should have lifecycle rules for IA and Glacier transitions."""
        rules = [
            {
                "ID": "raw-to-ia",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "Transitions": [{"Days": 30, "StorageClass": "STANDARD_IA"}],
            },
            {
                "ID": "raw-to-glacier",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "Transitions": [{"Days": 90, "StorageClass": "GLACIER"}],
            },
        ]

        s3_client.put_bucket_lifecycle_configuration(
            Bucket=S3_BUCKET_RAW,
            LifecycleConfiguration={"Rules": rules},
        )

        response = s3_client.get_bucket_lifecycle_configuration(Bucket=S3_BUCKET_RAW)
        configured_rules = response["Rules"]

        rule_ids = {r["ID"] for r in configured_rules}
        assert "raw-to-ia" in rule_ids
        assert "raw-to-glacier" in rule_ids

        glacier_rule = next(r for r in configured_rules if r["ID"] == "raw-to-glacier")
        assert glacier_rule["Transitions"][0]["Days"] == 90
        assert glacier_rule["Transitions"][0]["StorageClass"] == "GLACIER"

    def test_processed_bucket_lifecycle_configured(
        self, s3_client: Any, s3_buckets: list[str]
    ) -> None:
        """Processed bucket should transition to Glacier after 180 days."""
        rules = [
            {
                "ID": "processed-to-ia",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "Transitions": [{"Days": 60, "StorageClass": "STANDARD_IA"}],
            },
            {
                "ID": "processed-to-glacier",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "Transitions": [{"Days": 180, "StorageClass": "GLACIER"}],
            },
        ]

        s3_client.put_bucket_lifecycle_configuration(
            Bucket=S3_BUCKET_PROCESSED,
            LifecycleConfiguration={"Rules": rules},
        )

        response = s3_client.get_bucket_lifecycle_configuration(
            Bucket=S3_BUCKET_PROCESSED
        )
        configured_rules = response["Rules"]

        glacier_rule = next(
            r for r in configured_rules if r["ID"] == "processed-to-glacier"
        )
        assert glacier_rule["Transitions"][0]["Days"] == 180

    def test_archive_bucket_lifecycle_deep_archive(
        self, s3_client: Any, s3_buckets: list[str]
    ) -> None:
        """Archive bucket should transition to Deep Archive after 365 days."""
        rules = [
            {
                "ID": "archive-to-deep-archive",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "Transitions": [{"Days": 365, "StorageClass": "DEEP_ARCHIVE"}],
            },
        ]

        s3_client.put_bucket_lifecycle_configuration(
            Bucket=S3_BUCKET_ARCHIVE,
            LifecycleConfiguration={"Rules": rules},
        )

        response = s3_client.get_bucket_lifecycle_configuration(
            Bucket=S3_BUCKET_ARCHIVE
        )
        configured_rules = response["Rules"]

        deep_rule = next(
            r for r in configured_rules if r["ID"] == "archive-to-deep-archive"
        )
        assert deep_rule["Transitions"][0]["Days"] == 365
        assert deep_rule["Transitions"][0]["StorageClass"] == "DEEP_ARCHIVE"


# ---------------------------------------------------------------------------
# Tests: S3 Event Notifications
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEventNotifications:
    """Validate S3 event notifications to SQS."""

    def test_configure_event_notifications(
        self,
        s3_handler: S3Handler,
        s3_buckets: list[str],
        sqs_queue_arn: str,
    ) -> None:
        """Event notification configuration should succeed without errors."""
        s3_handler.configure_event_notifications(
            bucket=S3_BUCKET_RAW,
            sqs_queue_arn=sqs_queue_arn,
            prefix_filter="transactions/",
            suffix_filter=".parquet",
        )

    def test_upload_triggers_sqs_notification(
        self,
        s3_client: Any,
        s3_handler: S3Handler,
        s3_buckets: list[str],
        sqs_client: Any,
        sqs_queue_url: str,
        sqs_queue_arn: str,
        sample_transactions: list[dict[str, Any]],
    ) -> None:
        """Uploading a file should trigger an SQS notification."""
        # Configure notification
        s3_client.put_bucket_notification_configuration(
            Bucket=S3_BUCKET_RAW,
            NotificationConfiguration={
                "QueueConfigurations": [
                    {
                        "Id": "test-notification",
                        "QueueArn": sqs_queue_arn,
                        "Events": ["s3:ObjectCreated:*"],
                        "Filter": {
                            "Key": {
                                "FilterRules": [
                                    {"Name": "suffix", "Value": ".parquet"}
                                ]
                            }
                        },
                    }
                ]
            },
        )

        # Purge existing messages
        try:
            sqs_client.purge_queue(QueueUrl=sqs_queue_url)
            time.sleep(1)
        except ClientError:
            pass

        # Upload a file
        ts = datetime(2026, 8, 5, 10, 0, 0, tzinfo=timezone.utc)
        key = s3_handler.upload_transactions(
            sample_transactions[:5],
            timestamp=ts,
            bucket=S3_BUCKET_RAW,
        )

        # Poll SQS for message (LocalStack delivers quickly)
        time.sleep(2)
        response = sqs_client.receive_message(
            QueueUrl=sqs_queue_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=5,
        )

        messages = response.get("Messages", [])
        assert len(messages) > 0

        # Verify the notification references our upload
        body = json.loads(messages[0]["Body"])
        if "Records" in body:
            record = body["Records"][0]
            assert record["eventSource"] == "aws:s3"
            assert record["s3"]["bucket"]["name"] == S3_BUCKET_RAW
            assert record["s3"]["object"]["key"] == key


# ---------------------------------------------------------------------------
# Tests: Versioning
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestVersioning:
    """Validate bucket versioning."""

    def test_bucket_versioning_enabled(
        self, s3_client: Any, s3_buckets: list[str]
    ) -> None:
        """All buckets should support versioning when enabled."""
        s3_client.put_bucket_versioning(
            Bucket=S3_BUCKET_RAW,
            VersioningConfiguration={"Status": "Enabled"},
        )

        response = s3_client.get_bucket_versioning(Bucket=S3_BUCKET_RAW)
        assert response["Status"] == "Enabled"

    def test_versioning_preserves_previous_versions(
        self, s3_client: Any, s3_handler: S3Handler, s3_buckets: list[str]
    ) -> None:
        """Overwriting an object should preserve the previous version."""
        s3_client.put_bucket_versioning(
            Bucket=S3_BUCKET_RAW,
            VersioningConfiguration={"Status": "Enabled"},
        )

        key = "test/version_test.json"

        # Upload v1
        s3_handler.upload_raw_file(
            data=b'{"version": 1}',
            s3_key=key,
            bucket=S3_BUCKET_RAW,
            content_type="application/json",
        )

        # Upload v2 (overwrite)
        s3_handler.upload_raw_file(
            data=b'{"version": 2}',
            s3_key=key,
            bucket=S3_BUCKET_RAW,
            content_type="application/json",
        )

        # List versions
        response = s3_client.list_object_versions(
            Bucket=S3_BUCKET_RAW, Prefix=key
        )
        versions = response.get("Versions", [])
        assert len(versions) >= 2


# ---------------------------------------------------------------------------
# Tests: Cross-Service Access Patterns
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCrossServiceAccess:
    """Validate cross-service data access patterns."""

    def test_write_raw_read_for_processing(
        self, s3_handler: S3Handler, sample_transactions: list[dict[str, Any]]
    ) -> None:
        """Simulate pipeline: write to raw, read back for processing."""
        ts = datetime(2026, 8, 6, 8, 0, 0, tzinfo=timezone.utc)

        # Ingestion service writes raw data
        key = s3_handler.upload_transactions(
            sample_transactions,
            timestamp=ts,
            bucket=S3_BUCKET_RAW,
        )

        # Processing service reads raw data
        table = s3_handler.download_parquet(S3_BUCKET_RAW, key)
        assert table.num_rows == len(sample_transactions)

        # Processing service writes to processed bucket
        processed_key = s3_handler.upload_transactions(
            sample_transactions,
            timestamp=ts,
            bucket=S3_BUCKET_PROCESSED,
            prefix="validated",
        )

        assert s3_handler.file_exists(S3_BUCKET_PROCESSED, processed_key)

    def test_model_artifact_lifecycle(
        self, s3_handler: S3Handler
    ) -> None:
        """Simulate model registry: upload model, retrieve for serving."""
        model_data = b"serialized_isolation_forest_model_v2"
        config_data = json.dumps({
            "model_name": "isolation_forest",
            "version": "2.0.0",
            "parameters": {"n_estimators": 200, "contamination": 0.01},
            "metrics": {"auc_roc": 0.94, "recall": 0.87},
        }).encode()

        # Upload model artifact
        model_key = "isolation_forest/v2.0.0/model.pkl"
        config_key = "isolation_forest/v2.0.0/config.json"

        s3_handler.upload_raw_file(
            data=model_data,
            s3_key=model_key,
            bucket=S3_BUCKET_MODELS,
            content_type="application/octet-stream",
            metadata={"model_version": "2.0.0"},
        )
        s3_handler.upload_raw_file(
            data=config_data,
            s3_key=config_key,
            bucket=S3_BUCKET_MODELS,
            content_type="application/json",
        )

        # Serving service retrieves model
        assert s3_handler.file_exists(S3_BUCKET_MODELS, model_key)
        assert s3_handler.file_exists(S3_BUCKET_MODELS, config_key)

        meta = s3_handler.get_file_metadata(S3_BUCKET_MODELS, model_key)
        assert meta["metadata"]["model_version"] == "2.0.0"

    def test_archive_workflow(
        self, s3_handler: S3Handler, sample_transactions: list[dict[str, Any]]
    ) -> None:
        """Simulate archival: move old data to archive bucket."""
        ts = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Write to archive zone
        key = s3_handler.upload_transactions(
            sample_transactions[:10],
            timestamp=ts,
            bucket=S3_BUCKET_ARCHIVE,
            prefix="archive/transactions",
        )

        assert s3_handler.file_exists(S3_BUCKET_ARCHIVE, key)
        assert "2025/01/15/10" in key


# ---------------------------------------------------------------------------
# Tests: Public Access Block
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPublicAccessBlock:
    """Validate public access is blocked on all buckets."""

    def test_public_access_blocked(
        self, s3_client: Any, s3_buckets: list[str]
    ) -> None:
        """All buckets should have public access blocked."""
        for bucket in BUCKET_NAMES:
            s3_client.put_public_access_block(
                Bucket=bucket,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True,
                },
            )

            response = s3_client.get_public_access_block(Bucket=bucket)
            config = response["PublicAccessBlockConfiguration"]

            assert config["BlockPublicAcls"] is True
            assert config["IgnorePublicAcls"] is True
            assert config["BlockPublicPolicy"] is True
            assert config["RestrictPublicBuckets"] is True


# ---------------------------------------------------------------------------
# Tests: Metrics
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMetrics:
    """Validate S3 handler metrics collection."""

    def test_upload_metrics_tracked(
        self, s3_handler: S3Handler, sample_transactions: list[dict[str, Any]]
    ) -> None:
        """Upload operations should increment metrics."""
        initial = s3_handler.metrics.snapshot()

        s3_handler.upload_transactions(
            sample_transactions[:10],
            bucket=S3_BUCKET_RAW,
        )

        updated = s3_handler.metrics.snapshot()
        assert updated["uploads"] > initial["uploads"]
        assert updated["bytes_uploaded"] > initial["bytes_uploaded"]

    def test_download_metrics_tracked(
        self, s3_handler: S3Handler, sample_transactions: list[dict[str, Any]]
    ) -> None:
        """Download operations should increment metrics."""
        key = s3_handler.upload_transactions(
            sample_transactions[:10],
            bucket=S3_BUCKET_RAW,
        )

        initial = s3_handler.metrics.snapshot()
        s3_handler.download_parquet(S3_BUCKET_RAW, key)
        updated = s3_handler.metrics.snapshot()

        assert updated["downloads"] > initial["downloads"]
        assert updated["bytes_downloaded"] > initial["bytes_downloaded"]
