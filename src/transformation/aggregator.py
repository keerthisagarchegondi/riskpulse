"""Time-window aggregation framework for feature engineering.

Provides tumbling and sliding window aggregations with running statistics
(mean, std, min, max, sum, count) over configurable time periods.
Designed for efficient incremental computation.

Performance target: aggregate 1000 records in < 20ms.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__, component="aggregator")


class WindowType(Enum):
    """Supported window types for time-based aggregation."""

    TUMBLING = "tumbling"
    SLIDING = "sliding"


@dataclass
class WindowSpec:
    """Specification for a time window.

    Attributes:
        name: Human-readable name for the window.
        duration_seconds: Window size in seconds.
        window_type: TUMBLING (non-overlapping) or SLIDING (per-event).
        slide_seconds: Slide interval for sliding windows (ignored for tumbling).
    """

    name: str
    duration_seconds: int
    window_type: WindowType = WindowType.SLIDING
    slide_seconds: int = 0

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError(f"Window duration must be positive, got {self.duration_seconds}")
        if self.window_type == WindowType.TUMBLING and self.slide_seconds == 0:
            self.slide_seconds = self.duration_seconds


@dataclass
class AggregationResult:
    """Result of a window aggregation computation."""

    window_name: str
    count: int = 0
    sum: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    distinct_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            f"{self.window_name}_count": self.count,
            f"{self.window_name}_sum": round(self.sum, 2),
            f"{self.window_name}_mean": round(self.mean, 4),
            f"{self.window_name}_std": round(self.std, 4),
            f"{self.window_name}_min": round(self.min, 2),
            f"{self.window_name}_max": round(self.max, 2),
        }


class RunningStatistics:
    """Welford's online algorithm for running mean and variance.

    Efficiently computes mean and standard deviation in a single pass
    without storing all values. Supports adding and removing values
    for sliding window use cases.
    """

    __slots__ = ("_count", "_mean", "_m2", "_min", "_max", "_sum")

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0
        self._min: float = float("inf")
        self._max: float = float("-inf")
        self._sum: float = 0.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._mean if self._count > 0 else 0.0

    @property
    def variance(self) -> float:
        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)

    @property
    def std(self) -> float:
        return float(self.variance**0.5)

    @property
    def min_val(self) -> float:
        return self._min if self._count > 0 else 0.0

    @property
    def max_val(self) -> float:
        return self._max if self._count > 0 else 0.0

    @property
    def sum_val(self) -> float:
        return self._sum

    def add(self, value: float) -> None:
        """Add a new value to the running statistics."""
        self._count += 1
        self._sum += value
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

    def to_aggregation_result(self, window_name: str) -> AggregationResult:
        """Convert running statistics to an AggregationResult."""
        return AggregationResult(
            window_name=window_name,
            count=self._count,
            sum=self._sum,
            mean=self.mean,
            std=self.std,
            min=self.min_val,
            max=self.max_val,
        )

    def reset(self) -> None:
        """Reset all statistics."""
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = float("inf")
        self._max = float("-inf")
        self._sum = 0.0


class TimeWindowAggregator:
    """Aggregates numeric values over configurable time windows.

    Supports both tumbling (non-overlapping, fixed-size) and sliding
    (per-event, lookback) windows. Maintains a buffer of timestamped
    events for efficient window computation.

    Usage:
        aggregator = TimeWindowAggregator(window_specs=[
            WindowSpec("1h", 3600, WindowType.SLIDING),
            WindowSpec("24h", 86400, WindowType.SLIDING),
        ])
        results = aggregator.aggregate(events, reference_time)
    """

    def __init__(self, window_specs: list[WindowSpec] | None = None) -> None:
        if window_specs is None:
            window_specs = [
                WindowSpec("1h", 3600, WindowType.SLIDING),
                WindowSpec("24h", 86400, WindowType.SLIDING),
                WindowSpec("7d", 604800, WindowType.SLIDING),
            ]
        self._windows = window_specs

    @property
    def windows(self) -> list[WindowSpec]:
        return self._windows

    def aggregate(
        self,
        events: list[dict[str, Any]],
        reference_time: datetime,
        value_field: str = "transaction_amount",
        time_field: str = "transaction_timestamp",
    ) -> dict[str, AggregationResult]:
        """Compute aggregations over all configured windows.

        Args:
            events: List of event dictionaries (sorted by time desc preferred).
            reference_time: The point in time to look back from.
            value_field: The field name to aggregate.
            time_field: The field containing the timestamp.

        Returns:
            Mapping of window_name -> AggregationResult.
        """
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)

        results: dict[str, AggregationResult] = {}

        for spec in self._windows:
            stats = RunningStatistics()
            window_start = reference_time.timestamp() - spec.duration_seconds
            ref_ts = reference_time.timestamp()

            for event in events:
                event_time = self._extract_timestamp(event.get(time_field))
                if event_time is None:
                    continue

                event_ts = event_time.timestamp()
                # Only include events within the window and before reference
                if window_start <= event_ts < ref_ts:
                    value = event.get(value_field)
                    if value is not None:
                        try:
                            stats.add(float(value))
                        except (ValueError, TypeError):
                            continue

            results[spec.name] = stats.to_aggregation_result(spec.name)

        return results

    def aggregate_multiple_fields(
        self,
        events: list[dict[str, Any]],
        reference_time: datetime,
        value_fields: list[str],
        time_field: str = "transaction_timestamp",
    ) -> dict[str, dict[str, AggregationResult]]:
        """Compute aggregations for multiple fields over all windows.

        Args:
            events: List of event dictionaries.
            reference_time: The point in time to look back from.
            value_fields: List of field names to aggregate.
            time_field: The field containing the timestamp.

        Returns:
            Mapping of field_name -> {window_name -> AggregationResult}.
        """
        results: dict[str, dict[str, AggregationResult]] = {}
        for field_name in value_fields:
            results[field_name] = self.aggregate(
                events, reference_time, value_field=field_name, time_field=time_field
            )
        return results

    def compute_distinct_counts(
        self,
        events: list[dict[str, Any]],
        reference_time: datetime,
        count_field: str,
        time_field: str = "transaction_timestamp",
    ) -> dict[str, int]:
        """Count distinct values of a field within each time window.

        Args:
            events: List of event dictionaries.
            reference_time: The point in time to look back from.
            count_field: The field to count distinct values of.
            time_field: The field containing the timestamp.

        Returns:
            Mapping of window_name -> distinct count.
        """
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)

        results: dict[str, int] = {}
        ref_ts = reference_time.timestamp()

        for spec in self._windows:
            window_start = ref_ts - spec.duration_seconds
            distinct_values: set[str] = set()

            for event in events:
                event_time = self._extract_timestamp(event.get(time_field))
                if event_time is None:
                    continue

                event_ts = event_time.timestamp()
                if window_start <= event_ts < ref_ts:
                    value = event.get(count_field)
                    if value is not None and value != "":
                        distinct_values.add(str(value))

            results[spec.name] = len(distinct_values)

        return results

    @staticmethod
    def _extract_timestamp(value: Any) -> datetime | None:
        """Parse a timestamp value into timezone-aware datetime."""
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                return None
        return None


class IncrementalAggregator:
    """Maintains an incremental sliding window buffer for streaming use cases.

    Events are added one at a time and expired events are automatically
    evicted based on the configured maximum window duration.

    Thread-safe for concurrent reads and writes.
    """

    def __init__(self, max_window_seconds: int = 604800) -> None:
        self._max_window = max_window_seconds
        self._buffer: deque[tuple[float, float, dict[str, Any]]] = deque()
        self._lock = Lock()

    def add_event(
        self,
        timestamp: datetime,
        value: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an event to the incremental buffer.

        Args:
            timestamp: Event timestamp.
            value: Numeric value to aggregate.
            metadata: Additional event metadata for distinct counting.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        ts = timestamp.timestamp()
        with self._lock:
            self._buffer.append((ts, value, metadata or {}))
            self._evict_expired(ts)

    def get_statistics(
        self,
        reference_time: datetime,
        window_seconds: int,
    ) -> AggregationResult:
        """Get running statistics for a specific window from reference time.

        Args:
            reference_time: The point in time to look back from.
            window_seconds: How many seconds to look back.

        Returns:
            Aggregation result for the specified window.
        """
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)

        ref_ts = reference_time.timestamp()
        cutoff = ref_ts - window_seconds
        stats = RunningStatistics()

        with self._lock:
            for ts, value, _ in self._buffer:
                if cutoff <= ts < ref_ts:
                    stats.add(value)

        return stats.to_aggregation_result(f"{window_seconds}s")

    def get_distinct_count(
        self,
        reference_time: datetime,
        window_seconds: int,
        metadata_key: str,
    ) -> int:
        """Count distinct values of a metadata field within a time window.

        Args:
            reference_time: The point in time to look back from.
            window_seconds: How many seconds to look back.
            metadata_key: The metadata key to count distinct values of.

        Returns:
            Number of distinct values.
        """
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)

        ref_ts = reference_time.timestamp()
        cutoff = ref_ts - window_seconds
        distinct: set[str] = set()

        with self._lock:
            for ts, _, meta in self._buffer:
                if cutoff <= ts < ref_ts:
                    val = meta.get(metadata_key)
                    if val is not None and val != "":
                        distinct.add(str(val))

        return len(distinct)

    def _evict_expired(self, current_ts: float) -> None:
        """Remove events older than max window. Must be called under lock."""
        cutoff = current_ts - self._max_window
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
