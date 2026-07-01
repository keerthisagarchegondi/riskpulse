"""Data cleaning pipeline for raw transaction records.

Handles deduplication, missing value imputation, string standardization,
date parsing, PII masking, and field-level cleaning. Designed for
high-throughput processing with per-record metrics tracking.

Performance target: < 5ms per record.
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__, component="data_cleaner")


# --- PII Masking ---

_DEFAULT_PII_FIELDS = frozenset({
    "card_number",
    "card_last_four",
    "ssn",
    "social_security_number",
    "email",
    "email_address",
    "phone",
    "phone_number",
    "date_of_birth",
    "dob",
    "name",
    "first_name",
    "last_name",
    "full_name",
    "customer_name",
    "address",
    "street_address",
    "zip_code",
    "postal_code",
    "password",
    "secret",
    "token",
    "api_key",
    "ip_address",
    "device_id",
})

# Fields to partially mask (show last N chars)
_PARTIAL_MASK_FIELDS: dict[str, int] = {
    "card_number": 4,
    "card_last_four": 4,
    "phone": 4,
    "phone_number": 4,
    "email": 0,  # special handling
    "email_address": 0,
}


# --- Date Formats ---

_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y%m%d%H%M%S",
    "%Y%m%d",
)


# --- Metrics ---


@dataclass
class CleaningMetrics:
    """Thread-safe metrics tracking for the cleaning pipeline."""

    total_records_processed: int = 0
    total_records_deduplicated: int = 0
    total_fields_imputed: int = 0
    total_fields_standardized: int = 0
    total_fields_masked: int = 0
    total_dates_parsed: int = 0
    total_dates_failed: int = 0
    total_whitespace_trimmed: int = 0
    total_encoding_fixed: int = 0
    total_errors: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def increment(self, **kwargs: int) -> None:
        with self._lock:
            for key, value in kwargs.items():
                current = getattr(self, key, 0)
                setattr(self, key, current + value)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_records_processed": self.total_records_processed,
                "total_records_deduplicated": self.total_records_deduplicated,
                "total_fields_imputed": self.total_fields_imputed,
                "total_fields_standardized": self.total_fields_standardized,
                "total_fields_masked": self.total_fields_masked,
                "total_dates_parsed": self.total_dates_parsed,
                "total_dates_failed": self.total_dates_failed,
                "total_whitespace_trimmed": self.total_whitespace_trimmed,
                "total_encoding_fixed": self.total_encoding_fixed,
                "total_errors": self.total_errors,
            }

    def reset(self) -> None:
        with self._lock:
            self.total_records_processed = 0
            self.total_records_deduplicated = 0
            self.total_fields_imputed = 0
            self.total_fields_standardized = 0
            self.total_fields_masked = 0
            self.total_dates_parsed = 0
            self.total_dates_failed = 0
            self.total_whitespace_trimmed = 0
            self.total_encoding_fixed = 0
            self.total_errors = 0


# --- Cleaning Result ---


@dataclass
class CleaningResult:
    """Result of cleaning a single transaction record."""

    record: dict[str, Any]
    is_duplicate: bool = False
    changes: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record,
            "is_duplicate": self.is_duplicate,
            "changes": self.changes,
            "latency_ms": round(self.latency_ms, 4),
        }


# --- Dedup Cache ---


class _LRUDedup:
    """Bounded LRU cache for idempotency-key deduplication."""

    def __init__(self, max_size: int = 100000) -> None:
        self._max_size = max_size
        self._lock = Lock()
        self._cache: OrderedDict[str, float] = OrderedDict()

    def check_and_add(self, key: str) -> bool:
        """Return True if key is a duplicate, False otherwise."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return True
            self._cache[key] = time.time()
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


# --- Default Imputation Strategies ---

_DEFAULT_IMPUTATION: dict[str, Any] = {
    "transaction_currency": "USD",
    "is_international": False,
    "channel": "online",
    "card_type": None,  # None = leave as-is (nullable field)
    "geo_country": None,
    "geo_city": None,
    "device_type": "unknown",
}

# --- String Fields to Standardize ---

_LOWERCASE_FIELDS = frozenset({
    "transaction_type",
    "channel",
    "card_type",
    "device_type",
})

_TRIM_FIELDS = frozenset({
    "external_transaction_id",
    "account_id",
    "customer_id",
    "merchant_id",
    "merchant_name",
    "merchant_category_code",
    "transaction_currency",
    "card_last_four",
    "ip_address",
    "device_id",
    "geo_country",
    "geo_city",
})

_UPPERCASE_FIELDS = frozenset({
    "transaction_currency",
    "geo_country",
})


# --- Data Cleaner ---


class DataCleaner:
    """Production data cleaning pipeline for transaction records.

    Applies a sequence of cleaning steps in order:
    1. Deduplication (idempotency-key based)
    2. Whitespace trimming and encoding normalization
    3. String standardization (case, special chars)
    4. Date parsing and validation
    5. Missing value imputation
    6. PII masking for logs/metrics

    Thread-safe for concurrent processing.
    """

    def __init__(
        self,
        dedup_cache_size: int = 100000,
        pii_fields: frozenset[str] | None = None,
        imputation_defaults: dict[str, Any] | None = None,
        idempotency_fields: list[str] | None = None,
    ) -> None:
        self._dedup = _LRUDedup(max_size=dedup_cache_size)
        self._pii_fields = pii_fields or _DEFAULT_PII_FIELDS
        self._imputation = imputation_defaults or _DEFAULT_IMPUTATION
        self._idempotency_fields = idempotency_fields or [
            "external_transaction_id",
            "account_id",
            "transaction_amount",
            "transaction_timestamp",
        ]
        self.metrics = CleaningMetrics()

    # --- Public API ---

    def clean(self, record: dict[str, Any]) -> CleaningResult:
        """Clean a single transaction record.

        Returns a CleaningResult with the cleaned record and metadata.
        Duplicate records are returned with is_duplicate=True.
        """
        start = time.perf_counter()
        changes: list[str] = []

        # Work on a copy
        cleaned = dict(record)

        # 1. Deduplication check
        dedup_key = self._compute_dedup_key(cleaned)
        if self._dedup.check_and_add(dedup_key):
            self.metrics.increment(
                total_records_processed=1,
                total_records_deduplicated=1,
            )
            return CleaningResult(
                record=cleaned,
                is_duplicate=True,
                changes=["duplicate_detected"],
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # 2. Fix encoding issues
        encoding_fixes = self._fix_encoding(cleaned)
        changes.extend(encoding_fixes)

        # 3. Trim whitespace
        trim_fixes = self._trim_whitespace(cleaned)
        changes.extend(trim_fixes)

        # 4. Standardize strings (case normalization)
        string_fixes = self._standardize_strings(cleaned)
        changes.extend(string_fixes)

        # 5. Parse and validate dates
        date_fixes = self._parse_dates(cleaned)
        changes.extend(date_fixes)

        # 6. Impute missing values
        imputed = self._impute_missing(cleaned)
        changes.extend(imputed)

        # 7. Clean numeric fields
        numeric_fixes = self._clean_numerics(cleaned)
        changes.extend(numeric_fixes)

        # 8. Sanitize string content
        sanitize_fixes = self._sanitize_strings(cleaned)
        changes.extend(sanitize_fixes)

        elapsed = (time.perf_counter() - start) * 1000

        self.metrics.increment(total_records_processed=1)

        return CleaningResult(
            record=cleaned,
            is_duplicate=False,
            changes=changes,
            latency_ms=elapsed,
        )

    def clean_batch(self, records: list[dict[str, Any]]) -> list[CleaningResult]:
        """Clean a batch of records. Returns results in order."""
        return [self.clean(record) for record in records]

    def mask_pii(self, record: dict[str, Any]) -> dict[str, Any]:
        """Create a PII-masked copy of a record for logging/metrics.

        Does NOT modify the original record. Returns a new dict with
        sensitive fields masked.
        """
        masked = dict(record)
        masked_count = 0
        for field_name in list(masked.keys()):
            lower_name = field_name.lower()
            if lower_name in self._pii_fields:
                masked[field_name] = self._mask_value(field_name, masked[field_name])
                masked_count += 1
        if masked_count > 0:
            self.metrics.increment(total_fields_masked=masked_count)
        return masked

    def reset_dedup_cache(self) -> None:
        """Clear the deduplication cache."""
        self._dedup.clear()

    @property
    def dedup_cache_size(self) -> int:
        return self._dedup.size

    # --- Internal Cleaning Steps ---

    def _compute_dedup_key(self, record: dict[str, Any]) -> str:
        """Compute idempotency key from configured fields."""
        parts = []
        for field_name in self._idempotency_fields:
            val = record.get(field_name, "")
            parts.append(str(val).strip().lower() if val is not None else "")
        composite = "|".join(parts)
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()

    def _fix_encoding(self, record: dict[str, Any]) -> list[str]:
        """Normalize Unicode encoding (NFC form) and fix mojibake patterns."""
        changes = []
        for key, value in record.items():
            if not isinstance(value, str):
                continue
            normalized = unicodedata.normalize("NFC", value)
            # Remove null bytes
            normalized = normalized.replace("\x00", "")
            # Remove other control characters except newline/tab
            normalized = "".join(
                c for c in normalized
                if c in ("\n", "\t") or not unicodedata.category(c).startswith("C")
            )
            if normalized != value:
                record[key] = normalized
                changes.append(f"encoding_fixed:{key}")
                self.metrics.increment(total_encoding_fixed=1)
        return changes

    def _trim_whitespace(self, record: dict[str, Any]) -> list[str]:
        """Trim leading/trailing whitespace and collapse internal whitespace."""
        changes = []
        for key in _TRIM_FIELDS:
            value = record.get(key)
            if not isinstance(value, str):
                continue
            trimmed = value.strip()
            # Collapse multiple spaces
            trimmed = re.sub(r"\s+", " ", trimmed)
            if trimmed != value:
                record[key] = trimmed
                changes.append(f"whitespace_trimmed:{key}")
                self.metrics.increment(total_whitespace_trimmed=1)
        return changes

    def _standardize_strings(self, record: dict[str, Any]) -> list[str]:
        """Apply case standardization per field."""
        changes = []
        for key in _LOWERCASE_FIELDS:
            value = record.get(key)
            if isinstance(value, str) and value != value.lower():
                record[key] = value.lower()
                changes.append(f"lowercased:{key}")
                self.metrics.increment(total_fields_standardized=1)

        for key in _UPPERCASE_FIELDS:
            value = record.get(key)
            if isinstance(value, str) and value != value.upper():
                record[key] = value.upper()
                changes.append(f"uppercased:{key}")
                self.metrics.increment(total_fields_standardized=1)
        return changes

    def _parse_dates(self, record: dict[str, Any]) -> list[str]:
        """Parse and normalize date fields to ISO 8601 UTC."""
        changes = []
        date_fields = ("transaction_timestamp",)
        for key in date_fields:
            value = record.get(key)
            if value is None or value == "":
                continue
            if isinstance(value, datetime):
                # Already a datetime — normalize to ISO string
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                record[key] = value.strftime("%Y-%m-%dT%H:%M:%SZ")
                changes.append(f"datetime_to_iso:{key}")
                self.metrics.increment(total_dates_parsed=1)
                continue

            if not isinstance(value, str):
                continue

            parsed = self._try_parse_date(value)
            if parsed is not None:
                # Convert to UTC
                utc_dt = parsed.astimezone(timezone.utc)
                record[key] = utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                if record[key] != value:
                    changes.append(f"date_parsed:{key}")
                    self.metrics.increment(total_dates_parsed=1)
            else:
                self.metrics.increment(total_dates_failed=1)
                logger.warning(
                    "date_parse_failed",
                    field=key,
                    value=value[:50],
                )
        return changes

    @staticmethod
    def _try_parse_date(value: str) -> datetime | None:
        """Attempt to parse a date string using known formats."""
        value = value.strip()

        # Handle epoch timestamps (seconds or milliseconds)
        if value.isdigit():
            ts = int(value)
            # Milliseconds if > 10 billion
            if ts > 1e10:
                ts = ts / 1000
            try:
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except (OSError, ValueError, OverflowError):
                return None

        for fmt in _DATE_FORMATS:
            try:
                dt = datetime.strptime(value, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        return None

    def _impute_missing(self, record: dict[str, Any]) -> list[str]:
        """Impute missing values using configured defaults."""
        changes = []
        for field_name, default_value in self._imputation.items():
            if default_value is None:
                continue
            current = record.get(field_name)
            if current is None or (isinstance(current, str) and current.strip() == ""):
                record[field_name] = default_value
                changes.append(f"imputed:{field_name}={default_value}")
                self.metrics.increment(total_fields_imputed=1)
        return changes

    def _clean_numerics(self, record: dict[str, Any]) -> list[str]:
        """Clean and validate numeric fields."""
        changes = []
        amount = record.get("transaction_amount")
        if amount is not None:
            cleaned_amount = self._parse_amount(amount)
            if cleaned_amount is not None and cleaned_amount != amount:
                record["transaction_amount"] = cleaned_amount
                changes.append(f"amount_cleaned:{amount}->{cleaned_amount}")

        # Validate latitude/longitude ranges
        lat = record.get("geo_latitude")
        if lat is not None:
            try:
                lat_f = float(lat)
                if lat_f < -90 or lat_f > 90:
                    record["geo_latitude"] = None
                    changes.append("invalid_latitude_cleared")
            except (ValueError, TypeError):
                record["geo_latitude"] = None
                changes.append("unparseable_latitude_cleared")

        lon = record.get("geo_longitude")
        if lon is not None:
            try:
                lon_f = float(lon)
                if lon_f < -180 or lon_f > 180:
                    record["geo_longitude"] = None
                    changes.append("invalid_longitude_cleared")
            except (ValueError, TypeError):
                record["geo_longitude"] = None
                changes.append("unparseable_longitude_cleared")

        return changes

    @staticmethod
    def _parse_amount(value: Any) -> float | None:
        """Parse amount handling various formats (commas, currency symbols)."""
        if isinstance(value, (int, float)):
            return round(float(value), 4)
        if not isinstance(value, str):
            return None
        # Strip currency symbols and spaces (but keep commas and dots)
        cleaned = re.sub(r"[$ €£¥₹\s]", "", value)
        # Handle European format: 1.234,56 → 1234.56
        if re.match(r"^\d{1,3}(\.\d{3})+(,\d+)?$", cleaned):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        # Handle simple comma as decimal: 123,45 → 123.45
        elif "," in cleaned and "." not in cleaned:
            cleaned = cleaned.replace(",", ".")
        # Handle comma as thousands separator with dot decimal: 1,234.56
        elif "," in cleaned and "." in cleaned:
            cleaned = cleaned.replace(",", "")
        try:
            return round(float(cleaned), 4)
        except ValueError:
            return None

    def _sanitize_strings(self, record: dict[str, Any]) -> list[str]:
        """Remove potentially dangerous characters from string fields."""
        changes = []
        for key, value in record.items():
            if not isinstance(value, str):
                continue
            # Remove HTML/script tags
            sanitized = re.sub(r"<[^>]+>", "", value)
            if sanitized != value:
                record[key] = sanitized
                changes.append(f"html_stripped:{key}")
        return changes

    @staticmethod
    def _mask_value(field_name: str, value: Any) -> str:
        """Mask a PII field value appropriately."""
        if value is None:
            return "***"
        str_val = str(value)
        if not str_val:
            return "***"

        lower = field_name.lower()

        # Email: show first char + domain
        if lower in ("email", "email_address") and "@" in str_val:
            parts = str_val.split("@", 1)
            return f"{parts[0][0]}***@{parts[1]}"

        # Show last N characters for phone/card
        keep = _PARTIAL_MASK_FIELDS.get(lower)
        if keep and keep > 0 and len(str_val) > keep:
            return "*" * (len(str_val) - keep) + str_val[-keep:]

        # IP address: mask last octet
        if lower == "ip_address" and "." in str_val:
            parts = str_val.rsplit(".", 1)
            return f"{parts[0]}.***"

        # Device ID: show first 4 + last 4
        if lower == "device_id" and len(str_val) > 8:
            return f"{str_val[:4]}***{str_val[-4:]}"

        # Default: full mask
        return "***MASKED***"
