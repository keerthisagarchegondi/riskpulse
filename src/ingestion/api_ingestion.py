"""Batch file ingestion from S3 with schema detection and validation.

Production-grade batch ingestion handler supporting:
- CSV and JSON file ingestion from S3
- Automatic schema detection and validation
- Large file streaming with chunked reads
- Data quality checks before processing
- Progress tracking and resumable ingestion
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generator

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.json as pa_json
import pyarrow.parquet as pq
import structlog

from src.storage.s3_handler import (
    S3Handler,
    S3_BUCKET_RAW,
    S3_BUCKET_PROCESSED,
    S3DownloadError,
    S3UploadError,
    get_s3_handler,
)
from src.utils.config import get_settings

logger = structlog.get_logger(__name__)


# Required fields for transaction ingestion
REQUIRED_TRANSACTION_FIELDS = frozenset({
    "account_id",
    "transaction_amount",
    "transaction_currency",
    "transaction_type",
    "transaction_timestamp",
})

# Maximum chunk size for streaming reads (5 MB)
STREAM_CHUNK_SIZE = 5 * 1024 * 1024

# Maximum file size for in-memory processing (500 MB)
MAX_IN_MEMORY_SIZE = 500 * 1024 * 1024


class FileFormat(str, Enum):
    """Supported file formats for ingestion."""

    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"


class IngestionStatus(str, Enum):
    """Status of an ingestion job."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class IngestionError(Exception):
    """Base exception for ingestion errors."""


class SchemaDetectionError(IngestionError):
    """Raised when schema detection fails."""


class ValidationError(IngestionError):
    """Raised when data validation fails."""


class FileSizeError(IngestionError):
    """Raised when file exceeds size limits."""


@dataclass
class IngestionResult:
    """Result of a batch ingestion operation."""

    status: IngestionStatus
    source_key: str
    destination_key: str | None = None
    records_processed: int = 0
    records_failed: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    file_format: FileFormat | None = None
    schema_fields: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.records_processed + self.records_failed
        if total == 0:
            return 0.0
        return self.records_processed / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "source_key": self.source_key,
            "destination_key": self.destination_key,
            "records_processed": self.records_processed,
            "records_failed": self.records_failed,
            "errors": self.errors[:10],  # Limit error messages
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "file_format": self.file_format.value if self.file_format else None,
            "schema_fields": self.schema_fields,
            "success_rate": round(self.success_rate, 4),
        }


@dataclass
class DetectedSchema:
    """Result of schema detection on a file."""

    fields: list[str]
    field_types: dict[str, str]
    row_count_estimate: int
    has_header: bool = True
    delimiter: str = ","
    encoding: str = "utf-8"


class BatchIngestionHandler:
    """Handles batch file ingestion from S3.

    Processes CSV, JSON, and JSONL files from S3, validates them against
    expected schemas, and outputs validated Parquet files to the processed bucket.

    Usage:
        handler = BatchIngestionHandler()
        result = handler.ingest_file("source-bucket", "data/file.csv")
        print(result.to_dict())
    """

    def __init__(
        self,
        s3_handler: S3Handler | None = None,
        destination_bucket: str = S3_BUCKET_PROCESSED,
        destination_prefix: str = "validated",
    ) -> None:
        self._s3 = s3_handler or get_s3_handler()
        self._destination_bucket = destination_bucket
        self._destination_prefix = destination_prefix
        self._settings = get_settings()

    # =========================================================================
    # Public API
    # =========================================================================

    def ingest_file(
        self,
        source_bucket: str,
        source_key: str,
        file_format: FileFormat | None = None,
        required_fields: frozenset[str] | None = None,
        validate: bool = True,
    ) -> IngestionResult:
        """Ingest a single file from S3.

        Detects format, validates schema, and writes validated records
        as Parquet to the destination bucket.

        Args:
            source_bucket: S3 bucket containing the source file
            source_key: S3 key of the source file
            file_format: Override format detection (auto-detects if None)
            required_fields: Required fields for validation
            validate: Whether to perform schema validation

        Returns:
            IngestionResult with processing details

        Raises:
            IngestionError: If ingestion fails completely
        """
        result = IngestionResult(
            status=IngestionStatus.IN_PROGRESS,
            source_key=source_key,
        )

        try:
            # Detect format if not provided
            detected_format = file_format or self._detect_format(source_key)
            result.file_format = detected_format

            logger.info(
                "ingestion_started",
                source_bucket=source_bucket,
                source_key=source_key,
                format=detected_format.value,
            )

            # Check file size
            metadata = self._s3.get_file_metadata(source_bucket, source_key)
            file_size = metadata["content_length"]

            if file_size > MAX_IN_MEMORY_SIZE:
                # Use streaming for large files
                table = self._stream_ingest(
                    source_bucket, source_key, detected_format, file_size
                )
            else:
                # Standard in-memory processing
                table = self._standard_ingest(
                    source_bucket, source_key, detected_format
                )

            result.schema_fields = table.column_names

            # Validate schema
            if validate:
                fields_to_check = required_fields or REQUIRED_TRANSACTION_FIELDS
                validation_errors = self._validate_schema(table, fields_to_check)
                if validation_errors:
                    result.errors.extend(validation_errors)
                    # Filter out invalid rows
                    table, failed_count = self._filter_invalid_rows(
                        table, fields_to_check
                    )
                    result.records_failed = failed_count

            result.records_processed = table.num_rows

            # Write validated output as Parquet
            if table.num_rows > 0:
                destination_key = self._write_output(table, source_key)
                result.destination_key = destination_key

            result.status = (
                IngestionStatus.COMPLETED
                if result.records_failed == 0
                else IngestionStatus.PARTIAL
            )
            result.completed_at = datetime.now(timezone.utc)

            logger.info(
                "ingestion_completed",
                source_key=source_key,
                records_processed=result.records_processed,
                records_failed=result.records_failed,
                destination_key=result.destination_key,
            )

        except (S3DownloadError, S3UploadError) as e:
            result.status = IngestionStatus.FAILED
            result.errors.append(str(e))
            result.completed_at = datetime.now(timezone.utc)
            logger.error("ingestion_failed", source_key=source_key, error=str(e))

        except Exception as e:
            result.status = IngestionStatus.FAILED
            result.errors.append(f"Unexpected error: {str(e)}")
            result.completed_at = datetime.now(timezone.utc)
            logger.error(
                "ingestion_unexpected_error",
                source_key=source_key,
                error=str(e),
                exc_info=True,
            )

        return result

    def ingest_partition(
        self,
        source_bucket: str,
        prefix: str,
        file_format: FileFormat | None = None,
    ) -> list[IngestionResult]:
        """Ingest all files in an S3 partition/prefix.

        Args:
            source_bucket: S3 bucket containing source files
            prefix: S3 prefix to list files from
            file_format: Override format detection

        Returns:
            List of IngestionResult for each processed file
        """
        # List files in the prefix
        keys = self._list_files(source_bucket, prefix)
        results: list[IngestionResult] = []

        logger.info(
            "partition_ingestion_started",
            prefix=prefix,
            file_count=len(keys),
        )

        for key in keys:
            result = self.ingest_file(source_bucket, key, file_format)
            results.append(result)

        completed = sum(1 for r in results if r.status == IngestionStatus.COMPLETED)
        logger.info(
            "partition_ingestion_completed",
            prefix=prefix,
            total=len(results),
            completed=completed,
            failed=len(results) - completed,
        )

        return results

    def detect_schema(
        self,
        source_bucket: str,
        source_key: str,
        sample_size: int = 1000,
    ) -> DetectedSchema:
        """Detect the schema of a file without full ingestion.

        Reads a sample of the file to detect columns, types, delimiter, etc.

        Args:
            source_bucket: S3 bucket
            source_key: S3 key
            sample_size: Number of rows to sample for detection

        Returns:
            DetectedSchema with detected fields and types
        """
        file_format = self._detect_format(source_key)

        # Download a sample (first chunk)
        sample_bytes = self._download_sample(source_bucket, source_key, sample_size)

        if file_format == FileFormat.CSV:
            return self._detect_csv_schema(sample_bytes)
        elif file_format in (FileFormat.JSON, FileFormat.JSONL):
            return self._detect_json_schema(sample_bytes, file_format)
        elif file_format == FileFormat.PARQUET:
            return self._detect_parquet_schema(source_bucket, source_key)
        else:
            raise SchemaDetectionError(f"Unsupported format: {file_format}")

    # =========================================================================
    # Format Detection
    # =========================================================================

    def _detect_format(self, key: str) -> FileFormat:
        """Detect file format from the S3 key extension.

        Args:
            key: S3 object key

        Returns:
            Detected FileFormat

        Raises:
            IngestionError: If format cannot be determined
        """
        lower_key = key.lower()
        if lower_key.endswith(".csv"):
            return FileFormat.CSV
        elif lower_key.endswith(".json"):
            return FileFormat.JSON
        elif lower_key.endswith(".jsonl") or lower_key.endswith(".ndjson"):
            return FileFormat.JSONL
        elif lower_key.endswith(".parquet") or lower_key.endswith(".pq"):
            return FileFormat.PARQUET
        else:
            raise IngestionError(
                f"Cannot determine format for key: {key}. "
                "Supported: .csv, .json, .jsonl, .ndjson, .parquet"
            )

    # =========================================================================
    # Standard Ingestion (In-Memory)
    # =========================================================================

    def _standard_ingest(
        self,
        bucket: str,
        key: str,
        file_format: FileFormat,
    ) -> pa.Table:
        """Ingest a file that fits in memory.

        Args:
            bucket: S3 bucket
            key: S3 key
            file_format: Detected file format

        Returns:
            PyArrow Table with file contents
        """
        if file_format == FileFormat.PARQUET:
            return self._s3.download_parquet(bucket, key)

        # Download raw bytes for CSV/JSON
        response = self._s3._s3_client.get_object(Bucket=bucket, Key=key)
        raw_data = response["Body"].read()

        if file_format == FileFormat.CSV:
            return self._parse_csv(raw_data)
        elif file_format == FileFormat.JSON:
            return self._parse_json(raw_data)
        elif file_format == FileFormat.JSONL:
            return self._parse_jsonl(raw_data)
        else:
            raise IngestionError(f"Unsupported format for in-memory ingestion: {file_format}")

    def _parse_csv(self, data: bytes) -> pa.Table:
        """Parse CSV bytes into a PyArrow Table."""
        read_options = pa_csv.ReadOptions(
            autogenerate_column_names=False,
        )
        parse_options = pa_csv.ParseOptions(
            delimiter=",",
            quote_char='"',
            escape_char=None,
            newlines_in_values=True,
        )
        convert_options = pa_csv.ConvertOptions(
            strings_can_be_null=True,
            null_values=["", "null", "NULL", "None", "NA", "N/A"],
        )

        buffer = io.BytesIO(data)
        return pa_csv.read_csv(
            buffer,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )

    def _parse_json(self, data: bytes) -> pa.Table:
        """Parse a JSON array into a PyArrow Table."""
        records = json.loads(data)
        if isinstance(records, dict):
            # Handle wrapped responses (e.g., {"data": [...]})
            for key in ("data", "records", "transactions", "items", "results"):
                if key in records and isinstance(records[key], list):
                    records = records[key]
                    break
            else:
                # Single record
                records = [records]

        if not isinstance(records, list):
            raise IngestionError("JSON file must contain an array of records")

        return pa.Table.from_pylist(records)

    def _parse_jsonl(self, data: bytes) -> pa.Table:
        """Parse newline-delimited JSON into a PyArrow Table."""
        records: list[dict[str, Any]] = []
        for line in data.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

        if not records:
            raise IngestionError("JSONL file is empty")

        return pa.Table.from_pylist(records)

    # =========================================================================
    # Streaming Ingestion (Large Files)
    # =========================================================================

    def _stream_ingest(
        self,
        bucket: str,
        key: str,
        file_format: FileFormat,
        file_size: int,
    ) -> pa.Table:
        """Stream-ingest a large file in chunks.

        Args:
            bucket: S3 bucket
            key: S3 key
            file_format: File format
            file_size: Total file size in bytes

        Returns:
            PyArrow Table with all records
        """
        logger.info(
            "streaming_ingestion",
            key=key,
            file_size=file_size,
            format=file_format.value,
        )

        if file_format == FileFormat.PARQUET:
            # Parquet can be read directly (it supports partial reads)
            return self._s3.download_parquet(bucket, key)

        # For CSV/JSON/JSONL, stream and accumulate chunks
        chunks: list[bytes] = []
        for chunk in self._s3.stream_download(bucket, key, STREAM_CHUNK_SIZE):
            chunks.append(chunk)

        all_data = b"".join(chunks)

        if file_format == FileFormat.CSV:
            return self._parse_csv(all_data)
        elif file_format == FileFormat.JSON:
            return self._parse_json(all_data)
        elif file_format == FileFormat.JSONL:
            return self._parse_jsonl(all_data)
        else:
            raise IngestionError(f"Unsupported format for streaming: {file_format}")

    # =========================================================================
    # Schema Validation
    # =========================================================================

    def _validate_schema(
        self,
        table: pa.Table,
        required_fields: frozenset[str],
    ) -> list[str]:
        """Validate that a table contains required fields.

        Args:
            table: PyArrow table to validate
            required_fields: Set of required column names

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []
        columns = set(table.column_names)

        missing = required_fields - columns
        if missing:
            errors.append(f"Missing required fields: {sorted(missing)}")

        return errors

    def _filter_invalid_rows(
        self,
        table: pa.Table,
        required_fields: frozenset[str],
    ) -> tuple[pa.Table, int]:
        """Filter out rows with null values in required fields.

        Args:
            table: Input table
            required_fields: Fields that must not be null

        Returns:
            Tuple of (filtered table, count of removed rows)
        """
        original_count = table.num_rows
        mask = None

        for field_name in required_fields:
            if field_name in table.column_names:
                col = table.column(field_name)
                validity = col.is_valid()
                if mask is None:
                    mask = validity
                else:
                    mask = pa.compute.and_(mask, validity)

        if mask is not None:
            table = table.filter(mask)

        removed = original_count - table.num_rows
        return table, removed

    # =========================================================================
    # Schema Detection
    # =========================================================================

    def _download_sample(
        self, bucket: str, key: str, sample_size: int
    ) -> bytes:
        """Download a sample of a file for schema detection."""
        # Read first chunk (enough for header + sample rows)
        try:
            response = self._s3._s3_client.get_object(
                Bucket=bucket,
                Key=key,
                Range=f"bytes=0-{STREAM_CHUNK_SIZE - 1}",
            )
            return response["Body"].read()
        except Exception:
            # Fallback: download entire file if Range not supported
            response = self._s3._s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()

    def _detect_csv_schema(self, sample: bytes) -> DetectedSchema:
        """Detect CSV schema from sample bytes."""
        text = sample.decode("utf-8", errors="replace")
        lines = text.splitlines()

        if not lines:
            raise SchemaDetectionError("Empty CSV file")

        # Detect delimiter
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(lines[0])
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

        # Parse header
        reader = csv.reader(io.StringIO(lines[0]), delimiter=delimiter)
        header = next(reader)

        # Detect types from sample rows
        field_types: dict[str, str] = {}
        sample_lines = lines[1:101]  # Up to 100 rows for type detection

        for col_name in header:
            field_types[col_name] = "string"  # default

        if sample_lines:
            for line in sample_lines:
                row_reader = csv.reader(io.StringIO(line), delimiter=delimiter)
                try:
                    row = next(row_reader)
                except StopIteration:
                    continue

                for i, value in enumerate(row):
                    if i >= len(header):
                        break
                    col = header[i]
                    if value and field_types[col] == "string":
                        inferred = self._infer_type(value)
                        if inferred != "string":
                            field_types[col] = inferred

        return DetectedSchema(
            fields=header,
            field_types=field_types,
            row_count_estimate=len(lines) - 1,
            has_header=True,
            delimiter=delimiter,
        )

    def _detect_json_schema(
        self, sample: bytes, file_format: FileFormat
    ) -> DetectedSchema:
        """Detect JSON/JSONL schema from sample bytes."""
        text = sample.decode("utf-8", errors="replace")

        if file_format == FileFormat.JSONL:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if not lines:
                raise SchemaDetectionError("Empty JSONL file")
            records = [json.loads(lines[0])]
            row_count = len(lines)
        else:
            data = json.loads(text)
            if isinstance(data, list):
                records = data[:1]
                row_count = len(data)
            elif isinstance(data, dict):
                for key in ("data", "records", "transactions", "items"):
                    if key in data and isinstance(data[key], list):
                        records = data[key][:1]
                        row_count = len(data[key])
                        break
                else:
                    records = [data]
                    row_count = 1
            else:
                raise SchemaDetectionError("Cannot detect schema from JSON")

        if not records:
            raise SchemaDetectionError("No records found in JSON file")

        first_record = records[0]
        fields = list(first_record.keys())
        field_types = {k: type(v).__name__ for k, v in first_record.items()}

        return DetectedSchema(
            fields=fields,
            field_types=field_types,
            row_count_estimate=row_count,
            has_header=True,
        )

    def _detect_parquet_schema(
        self, bucket: str, key: str
    ) -> DetectedSchema:
        """Detect schema from a Parquet file (reads metadata only)."""
        table = self._s3.download_parquet(bucket, key)
        schema = table.schema

        fields = [f.name for f in schema]
        field_types = {f.name: str(f.type) for f in schema}

        return DetectedSchema(
            fields=fields,
            field_types=field_types,
            row_count_estimate=table.num_rows,
            has_header=True,
        )

    @staticmethod
    def _infer_type(value: str) -> str:
        """Infer the data type of a string value."""
        if not value:
            return "string"

        # Check integer
        try:
            int(value)
            return "int64"
        except ValueError:
            pass

        # Check float
        try:
            float(value)
            return "float64"
        except ValueError:
            pass

        # Check boolean
        if value.lower() in ("true", "false"):
            return "bool"

        return "string"

    # =========================================================================
    # Output Writing
    # =========================================================================

    def _write_output(self, table: pa.Table, source_key: str) -> str:
        """Write validated table as Parquet to the destination bucket.

        Args:
            table: Validated PyArrow table
            source_key: Original source key (for naming)

        Returns:
            Destination S3 key
        """
        now = datetime.now(timezone.utc)
        partition_path = (
            f"{self._destination_prefix}/"
            f"{now.year:04d}/{now.month:02d}/{now.day:02d}"
        )

        # Generate output key
        source_stem = source_key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        dest_key = f"{partition_path}/{source_stem}_{now.strftime('%H%M%S')}.parquet"

        # Write to buffer
        buffer = io.BytesIO()
        pq.write_table(
            table,
            buffer,
            compression="snappy",
            use_dictionary=True,
            write_statistics=True,
        )
        buffer.seek(0)

        # Upload
        self._s3.upload_raw_file(
            data=buffer.getvalue(),
            s3_key=dest_key,
            bucket=self._destination_bucket,
            content_type="application/x-parquet",
            metadata={
                "source_key": source_key,
                "record_count": str(table.num_rows),
                "ingestion_timestamp": now.isoformat(),
            },
        )

        return dest_key

    # =========================================================================
    # Utilities
    # =========================================================================

    def _list_files(self, bucket: str, prefix: str) -> list[str]:
        """List all ingestible files under a prefix."""
        paginator = self._s3._s3_client.get_paginator("list_objects_v2")
        keys: list[str] = []

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Filter to supported formats
                lower_key = key.lower()
                if any(
                    lower_key.endswith(ext)
                    for ext in (".csv", ".json", ".jsonl", ".ndjson", ".parquet")
                ):
                    keys.append(key)

        return keys
