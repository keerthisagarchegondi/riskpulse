"""Feature engineering pipeline for ML-ready fraud detection features.

Computes 25+ features per transaction across four categories:
- Transaction features (amount z-score, temporal, ratio)
- Velocity features (windowed counts, sums, uniques)
- Behavioral features (new merchant, unusual hour, percentile)
- Sequence features (consecutive declines, rapid succession)

Performance target: < 50ms per transaction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__, component="feature_engineer")

# Business hours and holidays (configurable)
_BUSINESS_HOUR_START = 6
_BUSINESS_HOUR_END = 22
_UNUSUAL_HOURS = frozenset(range(0, 6))  # midnight to 6am

# US federal holidays (month, day) - simplified; production would use a calendar lib
_HOLIDAYS: frozenset[tuple[int, int]] = frozenset(
    {
        (1, 1),
        (1, 20),
        (2, 17),
        (5, 26),
        (6, 19),
        (7, 4),
        (9, 1),
        (10, 13),
        (11, 11),
        (11, 27),
        (12, 25),
    }
)

# Velocity windows in seconds
WINDOW_1H = 3600
WINDOW_24H = 86400
WINDOW_7D = 604800

# Rapid succession threshold (seconds)
RAPID_SUCCESSION_THRESHOLD = 60


@dataclass
class FeatureMetrics:
    """Thread-safe metrics for feature computation."""

    total_computed: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, latency_ms: float, error: bool = False) -> None:
        with self._lock:
            self.total_computed += 1
            self.total_latency_ms += latency_ms
            if latency_ms < self.min_latency_ms:
                self.min_latency_ms = latency_ms
            if latency_ms > self.max_latency_ms:
                self.max_latency_ms = latency_ms
            if error:
                self.total_errors += 1

    @property
    def avg_latency_ms(self) -> float:
        with self._lock:
            if self.total_computed == 0:
                return 0.0
            return self.total_latency_ms / self.total_computed

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            avg_latency_ms = (
                self.total_latency_ms / self.total_computed if self.total_computed else 0.0
            )
            return {
                "total_computed": self.total_computed,
                "total_errors": self.total_errors,
                "avg_latency_ms": round(avg_latency_ms, 4),
                "min_latency_ms": (
                    round(self.min_latency_ms, 4) if self.min_latency_ms != float("inf") else 0.0
                ),
                "max_latency_ms": round(self.max_latency_ms, 4),
            }

    def reset(self) -> None:
        with self._lock:
            self.total_computed = 0
            self.total_errors = 0
            self.total_latency_ms = 0.0
            self.min_latency_ms = float("inf")
            self.max_latency_ms = 0.0


@dataclass
class FeatureResult:
    """Result container for computed features."""

    transaction_id: str
    features: dict[str, Any]
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "features": self.features,
            "latency_ms": round(self.latency_ms, 4),
            "error": self.error,
        }


class FeatureEngineer:
    """Computes ML-ready features from transaction data with customer context.

    Requires a customer profile (historical statistics) for velocity and
    behavioral features. If no profile is available, features degrade
    gracefully to defaults (0 or False).

    Usage:
        engineer = FeatureEngineer()
        result = engineer.compute_features(transaction, customer_profile, history)
    """

    def __init__(
        self,
        unusual_hours: frozenset[int] | None = None,
        holidays: frozenset[tuple[int, int]] | None = None,
        rapid_threshold_s: int = RAPID_SUCCESSION_THRESHOLD,
    ) -> None:
        self._unusual_hours = unusual_hours or _UNUSUAL_HOURS
        self._holidays = holidays or _HOLIDAYS
        self._rapid_threshold_s = rapid_threshold_s
        self._metrics = FeatureMetrics()

    @property
    def metrics(self) -> FeatureMetrics:
        return self._metrics

    def compute_features(
        self,
        transaction: dict[str, Any],
        customer_profile: dict[str, Any] | None = None,
        transaction_history: list[dict[str, Any]] | None = None,
    ) -> FeatureResult:
        """Compute all features for a single transaction.

        Args:
            transaction: The current transaction record.
            customer_profile: Customer statistics (avg_amount, std_amount, etc.)
            transaction_history: Recent transactions for this customer (sorted by time desc).

        Returns:
            FeatureResult with 25+ computed features.
        """
        start = time.perf_counter()
        txn_id = transaction.get(
            "external_transaction_id", transaction.get("transaction_id", "unknown")
        )

        try:
            profile = customer_profile or {}
            history = transaction_history or []

            features: dict[str, Any] = {}

            # Transaction features
            features.update(self._compute_transaction_features(transaction, profile))

            # Velocity features
            features.update(self._compute_velocity_features(transaction, history))

            # Behavioral features
            features.update(self._compute_behavioral_features(transaction, profile, history))

            # Sequence features
            features.update(self._compute_sequence_features(transaction, history))

            elapsed_ms = (time.perf_counter() - start) * 1000
            self._metrics.record(elapsed_ms)

            return FeatureResult(
                transaction_id=txn_id,
                features=features,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._metrics.record(elapsed_ms, error=True)
            logger.error(
                "Feature computation failed",
                transaction_id=txn_id,
                error=str(e),
            )
            return FeatureResult(
                transaction_id=txn_id,
                features={},
                latency_ms=elapsed_ms,
                error=str(e),
            )

    def compute_features_batch(
        self,
        transactions: list[dict[str, Any]],
        profiles: dict[str, dict[str, Any]] | None = None,
        histories: dict[str, list[dict[str, Any]]] | None = None,
    ) -> list[FeatureResult]:
        """Compute features for a batch of transactions.

        Args:
            transactions: List of transaction records.
            profiles: Mapping of customer_id -> customer_profile.
            histories: Mapping of customer_id -> recent transaction history.

        Returns:
            List of FeatureResult for each transaction.
        """
        profiles = profiles or {}
        histories = histories or {}
        results = []

        for txn in transactions:
            customer_id = txn.get("customer_id", "")
            profile = profiles.get(customer_id)
            history = histories.get(customer_id)
            results.append(self.compute_features(txn, profile, history))

        return results

    # --- Transaction Features ---

    def _compute_transaction_features(
        self,
        transaction: dict[str, Any],
        profile: dict[str, Any],
    ) -> dict[str, float | int | bool]:
        """Compute transaction-level features."""
        amount = float(transaction.get("transaction_amount", 0))
        ts = self._parse_timestamp(transaction.get("transaction_timestamp"))

        # Amount z-score
        avg_amount = float(profile.get("avg_transaction_amount", 0))
        std_amount = float(profile.get("std_transaction_amount", 0))
        if std_amount > 0:
            amount_zscore = (amount - avg_amount) / std_amount
        else:
            amount_zscore = 0.0

        # Temporal features
        hour_of_day = ts.hour if ts else 0
        day_of_week = ts.weekday() if ts else 0  # Monday=0
        is_weekend = day_of_week >= 5 if ts else False
        is_holiday = (ts.month, ts.day) in self._holidays if ts else False

        # Time since last transaction (seconds)
        last_txn_ts = profile.get("last_transaction_timestamp")
        if ts and last_txn_ts:
            last_ts = self._parse_timestamp(last_txn_ts)
            time_since_last = (ts - last_ts).total_seconds() if last_ts else 0.0
        else:
            time_since_last = 0.0

        # Amount to average ratio
        amount_to_avg_ratio = (amount / avg_amount) if avg_amount > 0 else 0.0

        return {
            "amount_zscore": round(amount_zscore, 6),
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "time_since_last_transaction": round(time_since_last, 2),
            "amount_to_avg_ratio": round(amount_to_avg_ratio, 6),
        }

    # --- Velocity Features ---

    def _compute_velocity_features(
        self,
        transaction: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, float | int]:
        """Compute velocity-based features using windowed aggregations."""
        ts = self._parse_timestamp(transaction.get("transaction_timestamp"))
        if not ts or not history:
            return {
                "txn_count_1h": 0,
                "txn_count_24h": 0,
                "txn_count_7d": 0,
                "txn_amount_sum_1h": 0.0,
                "txn_amount_sum_24h": 0.0,
                "unique_merchants_24h": 0,
                "unique_countries_24h": 0,
            }

        count_1h = 0
        count_24h = 0
        count_7d = 0
        sum_1h = 0.0
        sum_24h = 0.0
        merchants_24h: set[str] = set()
        countries_24h: set[str] = set()

        for h_txn in history:
            h_ts = self._parse_timestamp(h_txn.get("transaction_timestamp"))
            if not h_ts:
                continue

            delta_s = (ts - h_ts).total_seconds()
            if delta_s < 0:
                # Future transactions shouldn't exist - skip to prevent leakage
                continue

            h_amount = float(h_txn.get("transaction_amount", 0))

            if delta_s <= WINDOW_1H:
                count_1h += 1
                sum_1h += h_amount
            if delta_s <= WINDOW_24H:
                count_24h += 1
                sum_24h += h_amount
                merchant = h_txn.get("merchant_id", "")
                if merchant:
                    merchants_24h.add(merchant)
                country = h_txn.get("geo_country", "")
                if country:
                    countries_24h.add(country)
            if delta_s <= WINDOW_7D:
                count_7d += 1

        return {
            "txn_count_1h": count_1h,
            "txn_count_24h": count_24h,
            "txn_count_7d": count_7d,
            "txn_amount_sum_1h": round(sum_1h, 2),
            "txn_amount_sum_24h": round(sum_24h, 2),
            "unique_merchants_24h": len(merchants_24h),
            "unique_countries_24h": len(countries_24h),
        }

    # --- Behavioral Features ---

    def _compute_behavioral_features(
        self,
        transaction: dict[str, Any],
        profile: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, float | bool]:
        """Compute behavioral deviation features."""
        amount = float(transaction.get("transaction_amount", 0))
        ts = self._parse_timestamp(transaction.get("transaction_timestamp"))
        merchant_id = transaction.get("merchant_id", "")
        channel = transaction.get("channel", "")

        # New merchant flag - merchant not seen in history
        known_merchants: set[str] = set()
        for h_txn in history:
            m = h_txn.get("merchant_id", "")
            if m:
                known_merchants.add(m)
        new_merchant_flag = bool(merchant_id and merchant_id not in known_merchants)

        # Unusual hour flag
        unusual_hour_flag = ts.hour in self._unusual_hours if ts else False

        # Amount percentile within customer history
        historical_amounts = [
            float(h.get("transaction_amount", 0))
            for h in history
            if h.get("transaction_amount") is not None
        ]
        if historical_amounts:
            below_count = sum(1 for a in historical_amounts if a <= amount)
            amount_percentile = below_count / len(historical_amounts)
        else:
            amount_percentile = 0.5  # default to median if no history

        # Channel switch flag - different channel from last transaction
        channel_switch_flag = False
        if history and channel:
            last_channel = history[0].get("channel", "")
            channel_switch_flag = bool(last_channel and last_channel != channel)

        return {
            "new_merchant_flag": new_merchant_flag,
            "unusual_hour_flag": unusual_hour_flag,
            "amount_percentile": round(amount_percentile, 6),
            "channel_switch_flag": channel_switch_flag,
        }

    # --- Sequence Features ---

    def _compute_sequence_features(
        self,
        transaction: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, int | bool]:
        """Compute sequence-based features from transaction ordering."""
        ts = self._parse_timestamp(transaction.get("transaction_timestamp"))

        # Consecutive declined count (how many recent declines in a row before this)
        consecutive_declined = 0
        for h_txn in history:
            if h_txn.get("status") == "declined":
                consecutive_declined += 1
            else:
                break  # stop at first non-declined

        # Rapid succession flag - check if there's a transaction < 60s before this
        rapid_succession_flag = False
        if ts and history:
            for h_txn in history:
                h_ts = self._parse_timestamp(h_txn.get("transaction_timestamp"))
                if not h_ts:
                    continue
                delta_s = (ts - h_ts).total_seconds()
                if 0 < delta_s <= self._rapid_threshold_s:
                    rapid_succession_flag = True
                    break
                if delta_s > self._rapid_threshold_s:
                    break  # history is sorted desc, no need to check further

        return {
            "consecutive_declined_count": consecutive_declined,
            "rapid_succession_flag": rapid_succession_flag,
        }

    # --- Utilities ---

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        """Parse a timestamp value into a timezone-aware datetime."""
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str):
            # ISO format parsing
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def get_feature_names() -> list[str]:
        """Return ordered list of all feature names produced by this engineer."""
        return [
            # Transaction features
            "amount_zscore",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_holiday",
            "time_since_last_transaction",
            "amount_to_avg_ratio",
            # Velocity features
            "txn_count_1h",
            "txn_count_24h",
            "txn_count_7d",
            "txn_amount_sum_1h",
            "txn_amount_sum_24h",
            "unique_merchants_24h",
            "unique_countries_24h",
            # Behavioral features
            "new_merchant_flag",
            "unusual_hour_flag",
            "amount_percentile",
            "channel_switch_flag",
            # Sequence features
            "consecutive_declined_count",
            "rapid_succession_flag",
        ]
