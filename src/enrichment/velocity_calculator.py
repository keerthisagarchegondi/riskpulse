"""Real-time velocity calculation module.

Computes transaction velocity metrics and detects threshold breaches:
- Transaction count velocity (per time window)
- Amount velocity (total spend per window)
- Unique merchant velocity
- Customer profile updates
- Threshold breach detection with configurable limits
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__, component="velocity_calculator")

# Default velocity windows (seconds)
WINDOW_1_MIN = 60
WINDOW_5_MIN = 300
WINDOW_1_HOUR = 3600
WINDOW_24_HOUR = 86400

# Default thresholds
_DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "transaction_count_1min": {"warning": 3, "critical": 5},
    "transaction_count_5min": {"warning": 8, "critical": 15},
    "transaction_count_1hour": {"warning": 20, "critical": 50},
    "transaction_count_24hour": {"warning": 100, "critical": 200},
    "amount_sum_1min": {"warning": 1000.0, "critical": 5000.0},
    "amount_sum_5min": {"warning": 3000.0, "critical": 10000.0},
    "amount_sum_1hour": {"warning": 10000.0, "critical": 50000.0},
    "amount_sum_24hour": {"warning": 50000.0, "critical": 100000.0},
    "unique_merchants_1hour": {"warning": 5, "critical": 10},
    "unique_merchants_24hour": {"warning": 15, "critical": 30},
}


@dataclass
class VelocityMetrics:
    """Computed velocity metrics for a single evaluation."""

    transaction_count_1min: int = 0
    transaction_count_5min: int = 0
    transaction_count_1hour: int = 0
    transaction_count_24hour: int = 0
    amount_sum_1min: float = 0.0
    amount_sum_5min: float = 0.0
    amount_sum_1hour: float = 0.0
    amount_sum_24hour: float = 0.0
    unique_merchants_1hour: int = 0
    unique_merchants_24hour: int = 0
    avg_amount_1hour: float = 0.0
    max_amount_1hour: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "velocity_txn_count_1min": self.transaction_count_1min,
            "velocity_txn_count_5min": self.transaction_count_5min,
            "velocity_txn_count_1hour": self.transaction_count_1hour,
            "velocity_txn_count_24hour": self.transaction_count_24hour,
            "velocity_amount_sum_1min": self.amount_sum_1min,
            "velocity_amount_sum_5min": self.amount_sum_5min,
            "velocity_amount_sum_1hour": self.amount_sum_1hour,
            "velocity_amount_sum_24hour": self.amount_sum_24hour,
            "velocity_unique_merchants_1hour": self.unique_merchants_1hour,
            "velocity_unique_merchants_24hour": self.unique_merchants_24hour,
            "velocity_avg_amount_1hour": self.avg_amount_1hour,
            "velocity_max_amount_1hour": self.max_amount_1hour,
        }


@dataclass
class ThresholdBreach:
    """A detected threshold breach."""

    metric_name: str
    current_value: float
    threshold_value: float
    severity: str  # "warning" or "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "severity": self.severity,
        }


@dataclass
class VelocityResult:
    """Result of velocity calculation for a transaction."""

    customer_id: str
    metrics: VelocityMetrics = field(default_factory=VelocityMetrics)
    breaches: list[ThresholdBreach] = field(default_factory=list)
    velocity_risk_score: float = 0.0
    has_warning: bool = False
    has_critical: bool = False
    enrichment_latency_ms: float = 0.0
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None

    @property
    def breach_count(self) -> int:
        return len(self.breaches)

    def to_dict(self) -> dict[str, Any]:
        result = self.metrics.to_dict()
        result.update({
            "velocity_risk_score": self.velocity_risk_score,
            "velocity_breach_count": self.breach_count,
            "velocity_has_warning": self.has_warning,
            "velocity_has_critical": self.has_critical,
            "velocity_breaches": [b.to_dict() for b in self.breaches],
        })
        return result


@dataclass
class _TransactionRecord:
    """Internal record for a transaction in the sliding window."""

    timestamp: float  # Unix timestamp
    amount: float
    merchant_id: str


class CustomerVelocityProfile:
    """Thread-safe sliding-window velocity profile for a single customer.

    Maintains a deque of recent transactions and computes velocity metrics
    over configurable time windows.
    """

    def __init__(self, max_window_seconds: int = WINDOW_24_HOUR) -> None:
        self._records: deque[_TransactionRecord] = deque()
        self._max_window = max_window_seconds
        self._lock = Lock()

    def add_transaction(
        self, amount: float, merchant_id: str, timestamp: float | None = None
    ) -> None:
        """Add a transaction to the velocity window.

        Args:
            amount: Transaction amount.
            merchant_id: Merchant identifier.
            timestamp: Unix timestamp (defaults to now).
        """
        ts = timestamp or time.time()
        record = _TransactionRecord(timestamp=ts, amount=amount, merchant_id=merchant_id)

        with self._lock:
            self._records.append(record)
            self._evict_expired(ts)

    def compute_metrics(self, current_time: float | None = None) -> VelocityMetrics:
        """Compute velocity metrics over all time windows.

        Args:
            current_time: Reference time (unix timestamp). Defaults to now.

        Returns:
            VelocityMetrics with all windowed computations.
        """
        now = current_time or time.time()
        metrics = VelocityMetrics()

        with self._lock:
            self._evict_expired(now)

            records_1min: list[_TransactionRecord] = []
            records_5min: list[_TransactionRecord] = []
            records_1hour: list[_TransactionRecord] = []
            records_24hour: list[_TransactionRecord] = []

            for record in self._records:
                age = now - record.timestamp
                if age <= WINDOW_1_MIN:
                    records_1min.append(record)
                if age <= WINDOW_5_MIN:
                    records_5min.append(record)
                if age <= WINDOW_1_HOUR:
                    records_1hour.append(record)
                if age <= WINDOW_24_HOUR:
                    records_24hour.append(record)

        # Transaction counts
        metrics.transaction_count_1min = len(records_1min)
        metrics.transaction_count_5min = len(records_5min)
        metrics.transaction_count_1hour = len(records_1hour)
        metrics.transaction_count_24hour = len(records_24hour)

        # Amount sums
        metrics.amount_sum_1min = sum(r.amount for r in records_1min)
        metrics.amount_sum_5min = sum(r.amount for r in records_5min)
        metrics.amount_sum_1hour = sum(r.amount for r in records_1hour)
        metrics.amount_sum_24hour = sum(r.amount for r in records_24hour)

        # Unique merchants
        merchants_1h = {r.merchant_id for r in records_1hour if r.merchant_id}
        merchants_24h = {r.merchant_id for r in records_24hour if r.merchant_id}
        metrics.unique_merchants_1hour = len(merchants_1h)
        metrics.unique_merchants_24hour = len(merchants_24h)

        # Amount statistics for 1 hour window
        if records_1hour:
            amounts = [r.amount for r in records_1hour]
            metrics.avg_amount_1hour = sum(amounts) / len(amounts)
            metrics.max_amount_1hour = max(amounts)

        return metrics

    def _evict_expired(self, current_time: float) -> None:
        """Remove records older than the max window."""
        cutoff = current_time - self._max_window
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()

    @property
    def record_count(self) -> int:
        with self._lock:
            return len(self._records)


class VelocityCalculator:
    """Real-time velocity calculator with threshold breach detection.

    Maintains per-customer velocity profiles and evaluates transactions
    against configurable thresholds.

    Usage:
        calculator = VelocityCalculator()
        result = calculator.evaluate(transaction)
    """

    def __init__(
        self,
        thresholds: dict[str, dict[str, float]] | None = None,
        max_profiles: int = 100000,
    ) -> None:
        self._thresholds = thresholds or _DEFAULT_THRESHOLDS
        self._profiles: dict[str, CustomerVelocityProfile] = {}
        self._max_profiles = max_profiles
        self._lock = Lock()

    def evaluate(self, transaction: dict[str, Any]) -> VelocityResult:
        """Evaluate a transaction's velocity and detect threshold breaches.

        Args:
            transaction: Transaction record with customer_id, amount, merchant_id, timestamp.

        Returns:
            VelocityResult with metrics and any threshold breaches.
        """
        start = time.perf_counter()
        customer_id = transaction.get("customer_id", "")
        result = VelocityResult(customer_id=customer_id)

        try:
            if not customer_id:
                result.error = "Missing customer_id"
                result.enrichment_latency_ms = (time.perf_counter() - start) * 1000
                return result

            amount = float(transaction.get("transaction_amount", 0.0))
            merchant_id = transaction.get("merchant_id", "")
            timestamp = self._parse_unix_timestamp(
                transaction.get("transaction_timestamp")
            )

            # Get or create customer profile
            profile = self._get_or_create_profile(customer_id)

            # Add current transaction to profile
            profile.add_transaction(amount, merchant_id, timestamp)

            # Compute metrics
            result.metrics = profile.compute_metrics(timestamp)

            # Check thresholds
            result.breaches = self._check_thresholds(result.metrics)
            result.has_warning = any(
                b.severity == "warning" for b in result.breaches
            )
            result.has_critical = any(
                b.severity == "critical" for b in result.breaches
            )

            # Compute velocity risk score
            result.velocity_risk_score = self._compute_risk_score(result)

        except Exception as e:
            result.error = str(e)
            logger.error(
                "Velocity calculation failed",
                error=str(e),
                customer_id=customer_id,
                transaction_id=transaction.get("external_transaction_id"),
            )

        result.enrichment_latency_ms = (time.perf_counter() - start) * 1000
        return result

    def _get_or_create_profile(self, customer_id: str) -> CustomerVelocityProfile:
        """Get existing or create new velocity profile for a customer.

        Implements basic LRU eviction when max_profiles is reached.
        """
        with self._lock:
            if customer_id in self._profiles:
                return self._profiles[customer_id]

            # Evict oldest if at capacity
            if len(self._profiles) >= self._max_profiles:
                # Remove first entry (oldest by insertion order)
                oldest_key = next(iter(self._profiles))
                del self._profiles[oldest_key]

            profile = CustomerVelocityProfile()
            self._profiles[customer_id] = profile
            return profile

    def _check_thresholds(self, metrics: VelocityMetrics) -> list[ThresholdBreach]:
        """Check all configured thresholds against current metrics.

        Args:
            metrics: Computed velocity metrics.

        Returns:
            List of threshold breaches detected.
        """
        breaches: list[ThresholdBreach] = []

        metrics_map = {
            "transaction_count_1min": metrics.transaction_count_1min,
            "transaction_count_5min": metrics.transaction_count_5min,
            "transaction_count_1hour": metrics.transaction_count_1hour,
            "transaction_count_24hour": metrics.transaction_count_24hour,
            "amount_sum_1min": metrics.amount_sum_1min,
            "amount_sum_5min": metrics.amount_sum_5min,
            "amount_sum_1hour": metrics.amount_sum_1hour,
            "amount_sum_24hour": metrics.amount_sum_24hour,
            "unique_merchants_1hour": metrics.unique_merchants_1hour,
            "unique_merchants_24hour": metrics.unique_merchants_24hour,
        }

        for metric_name, current_value in metrics_map.items():
            threshold_config = self._thresholds.get(metric_name)
            if not threshold_config:
                continue

            critical_threshold = threshold_config.get("critical")
            warning_threshold = threshold_config.get("warning")

            # Check critical first (higher severity)
            if critical_threshold is not None and current_value >= critical_threshold:
                breaches.append(
                    ThresholdBreach(
                        metric_name=metric_name,
                        current_value=float(current_value),
                        threshold_value=float(critical_threshold),
                        severity="critical",
                    )
                )
            elif warning_threshold is not None and current_value >= warning_threshold:
                breaches.append(
                    ThresholdBreach(
                        metric_name=metric_name,
                        current_value=float(current_value),
                        threshold_value=float(warning_threshold),
                        severity="warning",
                    )
                )

        return breaches

    def _compute_risk_score(self, result: VelocityResult) -> float:
        """Compute velocity risk score from 0.0 to 1.0.

        Based on number and severity of threshold breaches.
        """
        if not result.breaches:
            return 0.0

        critical_count = sum(1 for b in result.breaches if b.severity == "critical")
        warning_count = sum(1 for b in result.breaches if b.severity == "warning")

        # Critical breaches are heavily weighted
        score = (critical_count * 0.3) + (warning_count * 0.1)
        return min(1.0, score)

    def get_customer_profile(self, customer_id: str) -> CustomerVelocityProfile | None:
        """Get the velocity profile for a customer (read-only access).

        Args:
            customer_id: Customer identifier.

        Returns:
            CustomerVelocityProfile or None if not tracked.
        """
        with self._lock:
            return self._profiles.get(customer_id)

    def get_profile_count(self) -> int:
        """Get the number of tracked customer profiles."""
        with self._lock:
            return len(self._profiles)

    def reset(self) -> None:
        """Clear all tracked profiles."""
        with self._lock:
            self._profiles.clear()

    @staticmethod
    def _parse_unix_timestamp(ts: Any) -> float:
        """Parse timestamp to unix float."""
        if ts is None:
            return time.time()
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, datetime):
            return ts.timestamp()
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.timestamp()
        return time.time()

    def evaluate_batch(
        self, transactions: list[dict[str, Any]]
    ) -> list[VelocityResult]:
        """Evaluate a batch of transactions.

        Args:
            transactions: List of transaction records.

        Returns:
            List of VelocityResult in same order as input.
        """
        return [self.evaluate(txn) for txn in transactions]
