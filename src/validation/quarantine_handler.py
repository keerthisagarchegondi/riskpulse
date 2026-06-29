"""Quarantine handler for invalid transaction records.

Routes records that fail schema validation to a quarantine store,
tracks failure reasons, provides metrics, and supports re-processing.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from src.utils.logger import get_logger
from src.validation.schema_validator import ValidationResult, ValidationSeverity

logger = get_logger(__name__, component="quarantine_handler")


@dataclass
class QuarantinedRecord:
    """A single record that failed validation and was quarantined."""

    quarantine_id: str
    record: dict[str, Any]
    failure_reasons: list[dict[str, str]]
    quarantined_at: str
    retry_count: int = 0
    status: str = "quarantined"  # quarantined | reprocessed | discarded

    def to_dict(self) -> dict[str, Any]:
        return {
            "quarantine_id": self.quarantine_id,
            "record": self.record,
            "failure_reasons": self.failure_reasons,
            "quarantined_at": self.quarantined_at,
            "retry_count": self.retry_count,
            "status": self.status,
        }


@dataclass
class QuarantineMetrics:
    """Thread-safe quarantine metrics tracking."""

    total_quarantined: int = 0
    total_reprocessed: int = 0
    total_discarded: int = 0
    reasons_breakdown: dict[str, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record_quarantine(self, failure_reasons: list[dict[str, str]]) -> None:
        with self._lock:
            self.total_quarantined += 1
            for reason in failure_reasons:
                key = reason.get("rule", "unknown")
                self.reasons_breakdown[key] = self.reasons_breakdown.get(key, 0) + 1

    def record_reprocess(self) -> None:
        with self._lock:
            self.total_reprocessed += 1

    def record_discard(self) -> None:
        with self._lock:
            self.total_discarded += 1

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_quarantined": self.total_quarantined,
                "total_reprocessed": self.total_reprocessed,
                "total_discarded": self.total_discarded,
                "active_quarantined": self.total_quarantined - self.total_reprocessed - self.total_discarded,
                "reasons_breakdown": dict(self.reasons_breakdown),
            }

    def reset(self) -> None:
        with self._lock:
            self.total_quarantined = 0
            self.total_reprocessed = 0
            self.total_discarded = 0
            self.reasons_breakdown.clear()


class QuarantineHandler:
    """Handles quarantining of invalid records with tracking and re-processing.

    Records that fail validation are stored in an in-memory quarantine
    with full failure context. In production, this would persist to a
    PostgreSQL quarantine table. Supports retry/re-processing and
    automatic discard after max retries.

    Usage:
        handler = QuarantineHandler(max_retry_attempts=3)
        handler.quarantine(record, validation_result)

        # Re-process with a validator
        results = handler.reprocess(validator)
    """

    def __init__(
        self,
        max_retry_attempts: int = 3,
        retention_days: int = 30,
    ) -> None:
        self._max_retry_attempts = max_retry_attempts
        self._retention_days = retention_days
        self._store: dict[str, QuarantinedRecord] = {}
        self._metrics = QuarantineMetrics()
        self._lock = Lock()
        logger.info(
            "Quarantine handler initialized",
            max_retry_attempts=max_retry_attempts,
            retention_days=retention_days,
        )

    @property
    def metrics(self) -> QuarantineMetrics:
        return self._metrics

    def quarantine(
        self,
        record: dict[str, Any],
        validation_result: ValidationResult,
    ) -> QuarantinedRecord:
        """Quarantine a record that failed validation.

        Args:
            record: The original transaction record.
            validation_result: The validation result with errors.

        Returns:
            The quarantined record entry.
        """
        failure_reasons = [
            {
                "field": err.field,
                "rule": err.rule,
                "message": err.message,
                "severity": err.severity.value,
            }
            for err in validation_result.errors
        ]

        quarantine_id = str(uuid.uuid4())
        entry = QuarantinedRecord(
            quarantine_id=quarantine_id,
            record=record,
            failure_reasons=failure_reasons,
            quarantined_at=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._store[quarantine_id] = entry

        self._metrics.record_quarantine(failure_reasons)

        logger.warning(
            "Record quarantined",
            quarantine_id=quarantine_id,
            error_count=len(failure_reasons),
            rules=[r["rule"] for r in failure_reasons],
        )

        return entry

    def get(self, quarantine_id: str) -> QuarantinedRecord | None:
        """Retrieve a quarantined record by ID."""
        with self._lock:
            return self._store.get(quarantine_id)

    def list_quarantined(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> list[QuarantinedRecord]:
        """List quarantined records, optionally filtered by status.

        Args:
            status: Filter by status (quarantined, reprocessed, discarded).
            limit: Maximum number of records to return.

        Returns:
            List of quarantined records.
        """
        with self._lock:
            records = list(self._store.values())

        if status:
            records = [r for r in records if r.status == status]

        # Sort by quarantined_at descending (newest first)
        records.sort(key=lambda r: r.quarantined_at, reverse=True)

        return records[:limit]

    def reprocess(
        self,
        validator: Any,
        quarantine_id: str | None = None,
    ) -> list[tuple[QuarantinedRecord, ValidationResult]]:
        """Re-process quarantined records through validation.

        Args:
            validator: SchemaValidator instance to re-validate records.
            quarantine_id: Optional specific record to reprocess.
                If None, reprocesses all eligible records.

        Returns:
            List of (QuarantinedRecord, ValidationResult) tuples.
        """
        results: list[tuple[QuarantinedRecord, ValidationResult]] = []

        with self._lock:
            if quarantine_id:
                candidates = [self._store.get(quarantine_id)]
                candidates = [c for c in candidates if c is not None]
            else:
                candidates = [
                    r for r in self._store.values()
                    if r.status == "quarantined"
                    and r.retry_count < self._max_retry_attempts
                ]

        for entry in candidates:
            entry.retry_count += 1
            result = validator.validate(entry.record)

            if result.is_valid:
                entry.status = "reprocessed"
                self._metrics.record_reprocess()
                logger.info(
                    "Quarantined record reprocessed successfully",
                    quarantine_id=entry.quarantine_id,
                    retry_count=entry.retry_count,
                )
            elif entry.retry_count >= self._max_retry_attempts:
                entry.status = "discarded"
                self._metrics.record_discard()
                logger.warning(
                    "Quarantined record discarded after max retries",
                    quarantine_id=entry.quarantine_id,
                    retry_count=entry.retry_count,
                )
            else:
                # Update failure reasons with latest attempt
                entry.failure_reasons = [
                    {
                        "field": err.field,
                        "rule": err.rule,
                        "message": err.message,
                        "severity": err.severity.value,
                    }
                    for err in result.errors
                ]

            results.append((entry, result))

        return results

    def discard(self, quarantine_id: str) -> bool:
        """Manually discard a quarantined record.

        Args:
            quarantine_id: ID of the record to discard.

        Returns:
            True if record was found and discarded, False otherwise.
        """
        with self._lock:
            entry = self._store.get(quarantine_id)
            if entry is None or entry.status != "quarantined":
                return False
            entry.status = "discarded"

        self._metrics.record_discard()
        logger.info("Record manually discarded", quarantine_id=quarantine_id)
        return True

    def purge_expired(self) -> int:
        """Remove records older than retention period.

        Returns:
            Number of records purged.
        """
        cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(
            days=self._retention_days
        )
        cutoff_iso = cutoff.isoformat()
        purged = 0

        with self._lock:
            to_remove = [
                qid
                for qid, entry in self._store.items()
                if entry.quarantined_at < cutoff_iso
                and entry.status in ("reprocessed", "discarded")
            ]
            for qid in to_remove:
                del self._store[qid]
                purged += 1

        if purged > 0:
            logger.info("Purged expired quarantine records", count=purged)

        return purged

    @property
    def count(self) -> int:
        """Number of currently quarantined (active) records."""
        with self._lock:
            return sum(
                1 for r in self._store.values() if r.status == "quarantined"
            )

    @property
    def total_count(self) -> int:
        """Total number of records in quarantine store."""
        with self._lock:
            return len(self._store)
