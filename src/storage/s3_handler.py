"""S3 storage handler for raw data landing and batch operations.

Production-grade S3 handler with:
- Partitioned storage (year/month/day/hour)
- Parquet file format with snappy compression
- Multipart uploads for large batches
- S3 event notification configuration
- Retry logic with exponential backoff
- Metrics tracking for observability
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any, Generator

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.config import get_settings
from src.utils.constants import S3_RAW_PREFIX, S3_PROCESSED_PREFIX, S3_MODELS_PREFIX, S3_ARCHIVE_PREFIX

logger = structlog.get_logger(__name__)


# S3 bucket names
S3_BUCKET_RAW = "riskpulse-raw"
S3_BUCKET_PROCESSED = "riskpulse-processed"
S3_BUCKET_MODELS = "riskpulse-models"
S3_BUCKET_ARCHIVE = "riskpulse-archive"

# Multipart upload threshold (100 MB)
MULTIPART_THRESHOLD = 100 * 1024 * 1024

# Multipart chunk size (8 MB)
MULTIPART_CHUNK_SIZE = 8 * 1024 * 1024

# Maximum records per Parquet file
MAX_RECORDS_PER_FILE = 100_000


class StorageLayer(str, Enum):
    """S3 storage layer classification."""

    RAW = "raw"
    VALIDATED = "validated"
    ENRICHED = "enriched"
    MODELS = "models"
    ARCHIVE = "archive"


class S3HandlerError(Exception):
    """Base exception for S3 handler errors."""


class S3UploadError(S3HandlerError):
    """Raised when an upload operation fails."""


class S3DownloadError(S3HandlerError):
    """Raised when a download operation fails."""


class S3ConfigError(S3HandlerError):
    """Raised when S3 configuration is invalid."""


@dataclass
class S3Metrics:
    """Thread-safe metrics for S3 operations."""

    uploads: int = 0
    downloads: int = 0
    bytes_uploaded: int = 0
    bytes_downloaded: int = 0
    upload_errors: int = 0
    download_errors: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record_upload(self, byte_size: int) -> None:
        with self._lock:
            self.uploads += 1
            self.bytes_uploaded += byte_size

    def record_download(self, byte_size: int) -> None:
        with self._lock:
            self.downloads += 1
            self.bytes_downloaded += byte_size

    def record_upload_error(self) -> None:
        with self._lock:
            self.upload_errors += 1

    def record_download_error(self) -> None:
        with self._lock:
            self.download_errors += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uploads": self.uploads,
                "downloads": self.downloads,
                "bytes_uploaded": self.bytes_uploaded,
                "bytes_downloaded": self.bytes_downloaded,
                "upload_errors": self.upload_errors,
                "download_errors": self.download_errors,
            }


def _build_partition_path(
    prefix: str,
    timestamp: datetime | None = None,
) -> str:
    """Build a partitioned S3 key path: prefix/YYYY/MM/DD/HH/

    Args:
        prefix: Base prefix (e.g., "transactions")
        timestamp: Event timestamp for partitioning. Defaults to UTC now.

    Returns:
        Partitioned path string
    """
    ts = timestamp or datetime.now(timezone.utc)
    return f"{prefix}/{ts.year:04d}/{ts.month:02d}/{ts.day:02d}/{ts.hour:02d}"


def _generate_file_key(partition_path: str, file_format: str = "parquet") -> str:
    """Generate a unique file key within a partition.

    Args:
        partition_path: The partition prefix
        file_format: File extension (parquet, json, csv)

    Returns:
        Full S3 key with unique filename
    """
    batch_id = uuid.uuid4().hex[:12]
    timestamp_suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{partition_path}/{timestamp_suffix}_{batch_id}.{file_format}"


class S3Handler:
    """Production-grade S3 handler for raw data landing.

    Manages upload/download of transaction data to S3 with:
    - Time-based partitioning (year/month/day/hour)
    - Parquet format with snappy compression
    - Multipart uploads for large datasets
    - Retry logic for transient failures
    - Metrics collection

    Usage:
        handler = S3Handler()
        key = handler.upload_transactions(transactions, timestamp)
        data = handler.download_parquet(bucket, key)
        handler.close()
    """

    def __init__(
        self,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        settings = get_settings()

        self._region = region_name or settings.get("aws.region", "us-east-1")
        self._endpoint_url = endpoint_url or settings.get("aws.s3.endpoint_url")
        self._metrics = S3Metrics()

        boto_config = BotoConfig(
            region_name=self._region,
            retries={"max_attempts": 3, "mode": "adaptive"},
            max_pool_connections=50,
        )

        session_kwargs: dict[str, Any] = {}
        if aws_access_key_id:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key

        session = boto3.Session(**session_kwargs)

        client_kwargs: dict[str, Any] = {"config": boto_config}
        if self._endpoint_url:
            client_kwargs["endpoint_url"] = self._endpoint_url

        self._s3_client = session.client("s3", **client_kwargs)
        self._s3_resource = session.resource("s3", **client_kwargs)

        logger.info(
            "s3_handler_initialized",
            region=self._region,
            endpoint_url=self._endpoint_url,
        )

    @property
    def metrics(self) -> S3Metrics:
        return self._metrics

    # =========================================================================
    # Upload Operations
    # =========================================================================

    @retry(
        retry=retry_if_exception_type(ClientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def upload_transactions(
        self,
        transactions: list[dict[str, Any]],
        timestamp: datetime | None = None,
        bucket: str = S3_BUCKET_RAW,
        prefix: str = S3_RAW_PREFIX,
    ) -> str:
        """Upload a batch of transactions as a Parquet file to S3.

        Args:
            transactions: List of transaction dictionaries
            timestamp: Event timestamp for partitioning (defaults to UTC now)
            bucket: Target S3 bucket
            prefix: Key prefix within bucket

        Returns:
            S3 key of the uploaded file

        Raises:
            S3UploadError: If upload fails after retries
            ValueError: If transactions list is empty
        """
        if not transactions:
            raise ValueError("Cannot upload empty transaction list")

        partition_path = _build_partition_path(prefix, timestamp)
        s3_key = _generate_file_key(partition_path, "parquet")

        try:
            # Convert to Parquet bytes
            parquet_buffer = self._transactions_to_parquet(transactions)
            parquet_bytes = parquet_buffer.getvalue()
            byte_size = len(parquet_bytes)

            logger.info(
                "uploading_transactions",
                record_count=len(transactions),
                byte_size=byte_size,
                bucket=bucket,
                key=s3_key,
            )

            # Choose upload strategy based on size
            if byte_size >= MULTIPART_THRESHOLD:
                self._multipart_upload(bucket, s3_key, parquet_bytes)
            else:
                self._s3_client.put_object(
                    Bucket=bucket,
                    Key=s3_key,
                    Body=parquet_bytes,
                    ContentType="application/x-parquet",
                    Metadata={
                        "record_count": str(len(transactions)),
                        "partition_timestamp": (
                            timestamp or datetime.now(timezone.utc)
                        ).isoformat(),
                        "compression": "snappy",
                    },
                )

            self._metrics.record_upload(byte_size)

            logger.info(
                "transactions_uploaded",
                key=s3_key,
                record_count=len(transactions),
                byte_size=byte_size,
            )

            return s3_key

        except ClientError as e:
            self._metrics.record_upload_error()
            logger.error(
                "s3_upload_failed",
                error=str(e),
                bucket=bucket,
                key=s3_key,
            )
            raise S3UploadError(
                f"Failed to upload to s3://{bucket}/{s3_key}: {e}"
            ) from e

    @retry(
        retry=retry_if_exception_type(ClientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def upload_raw_file(
        self,
        data: bytes,
        s3_key: str,
        bucket: str = S3_BUCKET_RAW,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload raw bytes to S3 with specified key.

        Args:
            data: Raw bytes to upload
            s3_key: Full S3 key path
            bucket: Target bucket
            content_type: MIME type of the data
            metadata: Optional metadata dict

        Returns:
            S3 key of the uploaded file

        Raises:
            S3UploadError: If upload fails
        """
        try:
            put_kwargs: dict[str, Any] = {
                "Bucket": bucket,
                "Key": s3_key,
                "Body": data,
                "ContentType": content_type,
            }
            if metadata:
                put_kwargs["Metadata"] = metadata

            if len(data) >= MULTIPART_THRESHOLD:
                self._multipart_upload(bucket, s3_key, data)
            else:
                self._s3_client.put_object(**put_kwargs)

            self._metrics.record_upload(len(data))

            logger.info(
                "raw_file_uploaded",
                key=s3_key,
                byte_size=len(data),
                content_type=content_type,
            )
            return s3_key

        except ClientError as e:
            self._metrics.record_upload_error()
            logger.error("s3_raw_upload_failed", error=str(e), key=s3_key)
            raise S3UploadError(
                f"Failed to upload to s3://{bucket}/{s3_key}: {e}"
            ) from e

    def upload_large_batch(
        self,
        transactions: list[dict[str, Any]],
        timestamp: datetime | None = None,
        bucket: str = S3_BUCKET_RAW,
        prefix: str = S3_RAW_PREFIX,
        max_records_per_file: int = MAX_RECORDS_PER_FILE,
    ) -> list[str]:
        """Upload a large batch of transactions split across multiple Parquet files.

        Splits data into chunks of max_records_per_file and uploads each
        as a separate Parquet file in the same partition.

        Args:
            transactions: List of transaction dictionaries
            timestamp: Event timestamp for partitioning
            bucket: Target S3 bucket
            prefix: Key prefix
            max_records_per_file: Maximum records per file

        Returns:
            List of S3 keys for all uploaded files
        """
        if not transactions:
            raise ValueError("Cannot upload empty transaction list")

        keys: list[str] = []
        for i in range(0, len(transactions), max_records_per_file):
            chunk = transactions[i : i + max_records_per_file]
            key = self.upload_transactions(chunk, timestamp, bucket, prefix)
            keys.append(key)

        logger.info(
            "large_batch_uploaded",
            total_records=len(transactions),
            file_count=len(keys),
        )
        return keys

    # =========================================================================
    # Download Operations
    # =========================================================================

    @retry(
        retry=retry_if_exception_type(ClientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def download_parquet(
        self, bucket: str, key: str
    ) -> pa.Table:
        """Download and read a Parquet file from S3.

        Args:
            bucket: Source bucket
            key: S3 key of the Parquet file

        Returns:
            PyArrow Table with the file contents

        Raises:
            S3DownloadError: If download or parsing fails
        """
        try:
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
            body = response["Body"].read()
            self._metrics.record_download(len(body))

            buffer = io.BytesIO(body)
            table = pq.read_table(buffer)

            logger.info(
                "parquet_downloaded",
                bucket=bucket,
                key=key,
                row_count=table.num_rows,
            )
            return table

        except ClientError as e:
            self._metrics.record_download_error()
            logger.error("s3_download_failed", error=str(e), bucket=bucket, key=key)
            raise S3DownloadError(
                f"Failed to download s3://{bucket}/{key}: {e}"
            ) from e
        except Exception as e:
            self._metrics.record_download_error()
            logger.error("parquet_parse_failed", error=str(e), key=key)
            raise S3DownloadError(
                f"Failed to parse Parquet file s3://{bucket}/{key}: {e}"
            ) from e

    def stream_download(
        self, bucket: str, key: str, chunk_size: int = MULTIPART_CHUNK_SIZE
    ) -> Generator[bytes, None, None]:
        """Stream download a file from S3 in chunks.

        Args:
            bucket: Source bucket
            key: S3 key
            chunk_size: Size of each chunk in bytes

        Yields:
            Bytes chunks of the file

        Raises:
            S3DownloadError: If download fails
        """
        try:
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

        except ClientError as e:
            self._metrics.record_download_error()
            logger.error("s3_stream_download_failed", error=str(e), key=key)
            raise S3DownloadError(
                f"Failed to stream s3://{bucket}/{key}: {e}"
            ) from e

    # =========================================================================
    # Listing & Utilities
    # =========================================================================

    def list_partition(
        self,
        bucket: str,
        prefix: str,
        timestamp: datetime | None = None,
    ) -> list[str]:
        """List all files in a given partition.

        Args:
            bucket: S3 bucket name
            prefix: Base prefix (e.g., "transactions")
            timestamp: Partition timestamp (defaults to current hour)

        Returns:
            List of S3 keys in the partition
        """
        partition_path = _build_partition_path(prefix, timestamp)

        keys: list[str] = []
        paginator = self._s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket, Prefix=partition_path):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])

        return keys

    def file_exists(self, bucket: str, key: str) -> bool:
        """Check if a file exists in S3.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            True if the object exists
        """
        try:
            self._s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def get_file_metadata(self, bucket: str, key: str) -> dict[str, Any]:
        """Retrieve metadata for an S3 object.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            Dict with ContentLength, ContentType, LastModified, Metadata
        """
        response = self._s3_client.head_object(Bucket=bucket, Key=key)
        return {
            "content_length": response["ContentLength"],
            "content_type": response.get("ContentType", ""),
            "last_modified": response["LastModified"],
            "metadata": response.get("Metadata", {}),
        }

    # =========================================================================
    # S3 Event Notifications
    # =========================================================================

    def configure_event_notifications(
        self,
        bucket: str,
        sqs_queue_arn: str,
        prefix_filter: str = "",
        suffix_filter: str = ".parquet",
        events: list[str] | None = None,
    ) -> None:
        """Configure S3 event notifications for new file arrivals.

        Args:
            bucket: Target bucket name
            sqs_queue_arn: ARN of the SQS queue to notify
            prefix_filter: Key prefix filter for notifications
            suffix_filter: Key suffix filter (e.g., ".parquet")
            events: List of S3 event types (defaults to ObjectCreated)
        """
        if events is None:
            events = ["s3:ObjectCreated:*"]

        filter_rules: list[dict[str, str]] = []
        if prefix_filter:
            filter_rules.append({"Name": "prefix", "Value": prefix_filter})
        if suffix_filter:
            filter_rules.append({"Name": "suffix", "Value": suffix_filter})

        notification_config = {
            "QueueConfigurations": [
                {
                    "Id": f"riskpulse-{bucket}-notification",
                    "QueueArn": sqs_queue_arn,
                    "Events": events,
                    "Filter": {
                        "Key": {"FilterRules": filter_rules}
                    },
                }
            ]
        }

        self._s3_client.put_bucket_notification_configuration(
            Bucket=bucket,
            NotificationConfiguration=notification_config,
        )

        logger.info(
            "s3_event_notifications_configured",
            bucket=bucket,
            sqs_queue_arn=sqs_queue_arn,
            events=events,
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _transactions_to_parquet(
        self, transactions: list[dict[str, Any]]
    ) -> io.BytesIO:
        """Convert transaction dicts to a Parquet buffer with snappy compression.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            BytesIO buffer containing Parquet data
        """
        table = pa.Table.from_pylist(transactions)

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

    def _multipart_upload(
        self, bucket: str, key: str, data: bytes
    ) -> None:
        """Perform a multipart upload for large files.

        Args:
            bucket: Target bucket
            key: S3 object key
            data: Raw bytes to upload
        """
        mpu = self._s3_client.create_multipart_upload(
            Bucket=bucket,
            Key=key,
            ContentType="application/x-parquet",
        )
        upload_id = mpu["UploadId"]
        parts: list[dict[str, Any]] = []

        try:
            part_number = 1
            offset = 0
            while offset < len(data):
                chunk = data[offset : offset + MULTIPART_CHUNK_SIZE]
                response = self._s3_client.upload_part(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=chunk,
                )
                parts.append({
                    "PartNumber": part_number,
                    "ETag": response["ETag"],
                })
                offset += MULTIPART_CHUNK_SIZE
                part_number += 1

            self._s3_client.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )

            logger.info(
                "multipart_upload_completed",
                key=key,
                parts=len(parts),
                total_bytes=len(data),
            )

        except (ClientError, Exception) as e:
            # Abort the multipart upload on failure
            self._s3_client.abort_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
            )
            logger.error(
                "multipart_upload_aborted",
                key=key,
                upload_id=upload_id,
                error=str(e),
            )
            raise

    def close(self) -> None:
        """Clean up resources."""
        logger.info("s3_handler_closed", metrics=self._metrics.snapshot())


def get_s3_handler(
    region_name: str | None = None,
    endpoint_url: str | None = None,
) -> S3Handler:
    """Factory function to create an S3Handler instance.

    Args:
        region_name: AWS region override
        endpoint_url: Endpoint URL override (for LocalStack)

    Returns:
        Configured S3Handler instance
    """
    return S3Handler(region_name=region_name, endpoint_url=endpoint_url)
