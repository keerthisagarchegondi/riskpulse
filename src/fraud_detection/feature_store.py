"""Feature store for real-time ML scoring and feature retrieval.

Provides low-latency feature retrieval for real-time fraud risk scoring,
feature freshness validation, and graceful handling of missing features.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Complete feature catalog with 50+ features for the risk scorer
FEATURE_CATALOG = [
    # Transaction features
    "transaction_amount",
    "amount_zscore",
    "amount_to_avg_ratio",
    "amount_percentile",
    "amount_log",
    # Temporal features
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "time_since_last_transaction",
    "minutes_since_midnight",
    # Velocity features (1h window)
    "txn_count_1h",
    "txn_amount_sum_1h",
    "txn_amount_avg_1h",
    "txn_amount_max_1h",
    # Velocity features (24h window)
    "txn_count_24h",
    "txn_amount_sum_24h",
    "txn_amount_avg_24h",
    "txn_amount_max_24h",
    "txn_amount_std_24h",
    # Velocity features (7d window)
    "txn_count_7d",
    "txn_amount_sum_7d",
    "txn_amount_avg_7d",
    # Merchant features
    "unique_merchants_24h",
    "unique_merchants_7d",
    "new_merchant_flag",
    "merchant_risk_score",
    "merchant_txn_count_30d",
    "merchant_fraud_rate",
    # Geographic features
    "unique_countries_24h",
    "is_international",
    "country_risk_score",
    "distance_from_last_location_km",
    "impossible_travel_flag",
    # Device features
    "known_device_flag",
    "device_age_days",
    "device_txn_count",
    "multiple_accounts_on_device",
    # Behavioral features
    "unusual_hour_flag",
    "channel_switch_flag",
    "amount_deviation_from_mean",
    "amount_deviation_from_median",
    "txn_frequency_deviation",
    # Sequence features
    "consecutive_declined_count",
    "rapid_succession_flag",
    "time_since_last_decline",
    "decline_rate_24h",
    # Account features
    "account_age_days",
    "account_total_txn_count",
    "account_avg_amount",
    "account_std_amount",
    "account_max_amount",
    # Cross features
    "amount_x_velocity",
    "amount_x_hour_risk",
    "international_x_new_merchant",
    "high_amount_x_unusual_hour",
]

# Default values for features when not available
FEATURE_DEFAULTS: dict[str, float] = {
    "transaction_amount": 0.0,
    "amount_zscore": 0.0,
    "amount_to_avg_ratio": 1.0,
    "amount_percentile": 0.5,
    "amount_log": 0.0,
    "hour_of_day": 12.0,
    "day_of_week": 3.0,
    "is_weekend": 0.0,
    "is_holiday": 0.0,
    "time_since_last_transaction": 3600.0,
    "minutes_since_midnight": 720.0,
    "txn_count_1h": 1.0,
    "txn_amount_sum_1h": 0.0,
    "txn_amount_avg_1h": 0.0,
    "txn_amount_max_1h": 0.0,
    "txn_count_24h": 5.0,
    "txn_amount_sum_24h": 0.0,
    "txn_amount_avg_24h": 0.0,
    "txn_amount_max_24h": 0.0,
    "txn_amount_std_24h": 0.0,
    "txn_count_7d": 20.0,
    "txn_amount_sum_7d": 0.0,
    "txn_amount_avg_7d": 0.0,
    "unique_merchants_24h": 2.0,
    "unique_merchants_7d": 5.0,
    "new_merchant_flag": 0.0,
    "merchant_risk_score": 0.1,
    "merchant_txn_count_30d": 100.0,
    "merchant_fraud_rate": 0.001,
    "unique_countries_24h": 1.0,
    "is_international": 0.0,
    "country_risk_score": 0.1,
    "distance_from_last_location_km": 0.0,
    "impossible_travel_flag": 0.0,
    "known_device_flag": 1.0,
    "device_age_days": 365.0,
    "device_txn_count": 100.0,
    "multiple_accounts_on_device": 0.0,
    "unusual_hour_flag": 0.0,
    "channel_switch_flag": 0.0,
    "amount_deviation_from_mean": 0.0,
    "amount_deviation_from_median": 0.0,
    "txn_frequency_deviation": 0.0,
    "consecutive_declined_count": 0.0,
    "rapid_succession_flag": 0.0,
    "time_since_last_decline": 86400.0,
    "decline_rate_24h": 0.0,
    "account_age_days": 365.0,
    "account_total_txn_count": 500.0,
    "account_avg_amount": 50.0,
    "account_std_amount": 30.0,
    "account_max_amount": 200.0,
    "amount_x_velocity": 0.0,
    "amount_x_hour_risk": 0.0,
    "international_x_new_merchant": 0.0,
    "high_amount_x_unusual_hour": 0.0,
}

# Freshness thresholds per feature group
FRESHNESS_THRESHOLDS: dict[str, timedelta] = {
    "velocity_1h": timedelta(minutes=5),
    "velocity_24h": timedelta(minutes=15),
    "velocity_7d": timedelta(hours=1),
    "customer_profile": timedelta(hours=1),
    "merchant_profile": timedelta(hours=6),
    "device_profile": timedelta(hours=24),
}


@dataclass
class FeatureVector:
    """Container for a computed feature vector with metadata."""

    transaction_id: str
    customer_id: str
    features: dict[str, float]
    feature_count: int = 0
    missing_features: list[str] = field(default_factory=list)
    stale_features: list[str] = field(default_factory=list)
    computation_time_ms: float = 0.0
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_array(self, feature_names: list[str]) -> np.ndarray:
        """Convert to numpy array in specified feature order."""
        return np.array([self.features.get(f, FEATURE_DEFAULTS.get(f, 0.0)) for f in feature_names])

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "customer_id": self.customer_id,
            "features": self.features,
            "feature_count": self.feature_count,
            "missing_features": self.missing_features,
            "stale_features": self.stale_features,
            "computation_time_ms": self.computation_time_ms,
            "computed_at": self.computed_at.isoformat(),
        }


@dataclass
class FeatureFreshnessReport:
    """Report on feature freshness status."""

    total_features: int = 0
    fresh_features: int = 0
    stale_features: int = 0
    missing_features: int = 0
    is_acceptable: bool = True
    details: dict[str, str] = field(default_factory=dict)


class FeatureStore:
    """Feature store for real-time ML model scoring.

    Retrieves precomputed features from the database/cache layer,
    validates freshness, handles missing features gracefully, and
    computes derived cross-features on the fly.
    """

    def __init__(
        self,
        db_handler: Any | None = None,
        cache_handler: Any | None = None,
        feature_names: list[str] | None = None,
        freshness_check_enabled: bool = True,
        max_missing_feature_ratio: float = 0.2,
    ) -> None:
        self._db = db_handler
        self._cache = cache_handler
        self._feature_names = feature_names or FEATURE_CATALOG
        self._freshness_check_enabled = freshness_check_enabled
        self._max_missing_ratio = max_missing_feature_ratio
        self._feature_defaults = FEATURE_DEFAULTS.copy()
        self._stats = {
            "total_retrievals": 0,
            "cache_hits": 0,
            "db_hits": 0,
            "fallbacks_to_default": 0,
        }

    def get_features(
        self,
        transaction_id: str,
        customer_id: str,
        transaction_data: dict[str, Any],
    ) -> FeatureVector:
        """Retrieve or compute features for a transaction.

        Attempts cache first, then database, then computes on-the-fly.
        Missing features are filled with defaults and flagged.

        Args:
            transaction_id: Unique transaction identifier.
            customer_id: Customer identifier for profile lookup.
            transaction_data: Raw transaction data for on-the-fly computation.

        Returns:
            FeatureVector with all required features populated.
        """
        start = time.perf_counter()
        self._stats["total_retrievals"] += 1

        features: dict[str, float] = {}
        missing: list[str] = []
        stale: list[str] = []

        # Attempt to retrieve from cache
        cached_features = self._get_from_cache(transaction_id, customer_id)
        if cached_features:
            self._stats["cache_hits"] += 1
            features.update(cached_features)

        # Fill remaining from database
        remaining = [f for f in self._feature_names if f not in features]
        if remaining:
            db_features = self._get_from_db(transaction_id, customer_id, remaining)
            if db_features:
                self._stats["db_hits"] += 1
                features.update(db_features)

        # Compute transaction-level features on-the-fly
        computed = self._compute_transaction_features(transaction_data, features)
        features.update(computed)

        # Compute cross-features
        cross_features = self._compute_cross_features(features)
        features.update(cross_features)

        # Identify missing features and apply defaults
        for feat_name in self._feature_names:
            if feat_name not in features or features[feat_name] is None:
                missing.append(feat_name)
                features[feat_name] = self._feature_defaults.get(feat_name, 0.0)
                self._stats["fallbacks_to_default"] += 1

        # Freshness check
        if self._freshness_check_enabled:
            stale = self._check_freshness(features, customer_id)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        vector = FeatureVector(
            transaction_id=transaction_id,
            customer_id=customer_id,
            features=features,
            feature_count=len(features),
            missing_features=missing,
            stale_features=stale,
            computation_time_ms=elapsed_ms,
        )

        if missing:
            logger.warning(
                "Feature retrieval had %d missing features for txn=%s",
                len(missing),
                transaction_id,
            )

        return vector

    def get_features_batch(
        self,
        transactions: list[dict[str, Any]],
    ) -> list[FeatureVector]:
        """Retrieve features for a batch of transactions."""
        results = []
        for txn in transactions:
            vector = self.get_features(
                transaction_id=txn.get("transaction_id", txn.get("external_transaction_id", "")),
                customer_id=txn.get("customer_id", ""),
                transaction_data=txn,
            )
            results.append(vector)
        return results

    def check_feature_freshness(self, customer_id: str) -> FeatureFreshnessReport:
        """Check freshness of all features for a customer."""
        report = FeatureFreshnessReport(total_features=len(self._feature_names))

        if not self._freshness_check_enabled:
            report.fresh_features = report.total_features
            return report

        stale = self._check_freshness({}, customer_id)
        report.stale_features = len(stale)
        report.fresh_features = report.total_features - report.stale_features
        report.is_acceptable = (report.stale_features / max(report.total_features, 1)) < 0.3

        for feat in stale:
            report.details[feat] = "stale"

        return report

    def is_feature_set_valid(self, feature_vector: FeatureVector) -> bool:
        """Check if a feature vector has acceptable quality for scoring."""
        if not self._feature_names:
            return True
        missing_ratio = len(feature_vector.missing_features) / len(self._feature_names)
        return missing_ratio <= self._max_missing_ratio

    def get_stats(self) -> dict[str, Any]:
        """Return retrieval statistics."""
        total = max(self._stats["total_retrievals"], 1)
        return {
            **self._stats,
            "cache_hit_rate": self._stats["cache_hits"] / total,
            "default_fallback_rate": self._stats["fallbacks_to_default"]
            / (total * len(self._feature_names)),
        }

    def _get_from_cache(self, transaction_id: str, customer_id: str) -> dict[str, float]:
        """Attempt to retrieve features from cache layer."""
        if self._cache is None:
            return {}
        try:
            cached = self._cache.get(f"features:{transaction_id}")
            if cached and isinstance(cached, dict):
                return {k: float(v) for k, v in cached.items() if k in self._feature_names}
            # Try customer profile cache
            profile = self._cache.get(f"customer_profile:{customer_id}")
            if profile and isinstance(profile, dict):
                return {k: float(v) for k, v in profile.items() if k in self._feature_names}
        except Exception as e:
            logger.debug("Cache retrieval failed: %s", e)
        return {}

    def _get_from_db(
        self, transaction_id: str, customer_id: str, feature_names: list[str]
    ) -> dict[str, float]:
        """Retrieve features from the database."""
        if self._db is None:
            return {}
        try:
            result = self._db.get_transaction_features(transaction_id)
            if result and isinstance(result, dict):
                return {
                    k: float(v) for k, v in result.items() if k in feature_names and v is not None
                }
        except Exception as e:
            logger.debug("DB retrieval failed: %s", e)
        return {}

    def _compute_transaction_features(
        self, transaction_data: dict[str, Any], existing: dict[str, float]
    ) -> dict[str, float]:
        """Compute features directly from the raw transaction data."""
        computed: dict[str, float] = {}

        amount = transaction_data.get("transaction_amount")
        if amount is not None:
            amount = float(amount)
            computed["transaction_amount"] = amount
            computed["amount_log"] = float(np.log1p(amount))

            avg = existing.get(
                "account_avg_amount", self._feature_defaults.get("account_avg_amount", 50.0)
            )
            std = existing.get(
                "account_std_amount", self._feature_defaults.get("account_std_amount", 30.0)
            )
            if std > 0:
                computed["amount_zscore"] = (amount - avg) / std
            else:
                computed["amount_zscore"] = 0.0

            if avg > 0:
                computed["amount_to_avg_ratio"] = amount / avg
            computed["amount_deviation_from_mean"] = amount - avg

        # Temporal features
        timestamp = transaction_data.get("transaction_timestamp")
        if timestamp:
            if isinstance(timestamp, str):
                try:
                    from datetime import datetime as dt

                    ts = dt.fromisoformat(timestamp.replace("Z", "+00:00"))
                    computed["hour_of_day"] = float(ts.hour)
                    computed["day_of_week"] = float(ts.weekday())
                    computed["is_weekend"] = float(ts.weekday() >= 5)
                    computed["minutes_since_midnight"] = float(ts.hour * 60 + ts.minute)
                    computed["unusual_hour_flag"] = float(ts.hour < 6 or ts.hour > 23)
                except (ValueError, TypeError):
                    pass

        # Geographic features
        is_intl = transaction_data.get("is_international")
        if is_intl is not None:
            computed["is_international"] = float(bool(is_intl))

        # Channel features
        channel = transaction_data.get("channel")
        last_channel = transaction_data.get("last_channel")
        if channel and last_channel and channel != last_channel:
            computed["channel_switch_flag"] = 1.0

        return computed

    def _compute_cross_features(self, features: dict[str, float]) -> dict[str, float]:
        """Compute interaction/cross features from existing features."""
        cross: dict[str, float] = {}

        amount = features.get("transaction_amount", 0.0)
        velocity_1h = features.get("txn_count_1h", 0.0)
        hour_risk = 1.0 if features.get("unusual_hour_flag", 0.0) > 0 else 0.0
        is_intl = features.get("is_international", 0.0)
        new_merchant = features.get("new_merchant_flag", 0.0)

        cross["amount_x_velocity"] = amount * velocity_1h
        cross["amount_x_hour_risk"] = amount * hour_risk
        cross["international_x_new_merchant"] = is_intl * new_merchant
        cross["high_amount_x_unusual_hour"] = (
            float(amount > features.get("account_avg_amount", 50.0) * 3) * hour_risk
        )

        return cross

    def _check_freshness(self, features: dict[str, float], customer_id: str) -> list[str]:
        """Check which features are potentially stale."""
        stale: list[str] = []

        if self._cache is None:
            return stale

        try:
            last_update = self._cache.get(f"feature_ts:{customer_id}")
            if last_update is None:
                return stale

            now = datetime.now(timezone.utc)
            if isinstance(last_update, str):
                from datetime import datetime as dt

                last_update_dt = dt.fromisoformat(last_update)
            elif isinstance(last_update, (int, float)):
                last_update_dt = datetime.fromtimestamp(float(last_update), tz=timezone.utc)
            else:
                return stale

            age = now - last_update_dt

            # Check velocity features freshness
            velocity_features = [f for f in self._feature_names if "1h" in f]
            if age > FRESHNESS_THRESHOLDS["velocity_1h"]:
                stale.extend(velocity_features)

            velocity_24h_features = [f for f in self._feature_names if "24h" in f]
            if age > FRESHNESS_THRESHOLDS["velocity_24h"]:
                stale.extend(velocity_24h_features)

        except Exception as e:
            logger.debug("Freshness check failed: %s", e)

        return stale
