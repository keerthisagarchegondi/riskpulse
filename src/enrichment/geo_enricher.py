"""Geo-enrichment module for IP geolocation and travel velocity detection.

Enriches transactions with:
- IP → geolocation (country, city, lat/lng)
- Distance from customer's usual location
- Country risk scoring
- Impossible travel detection (> 500 mph between transactions)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__, component="geo_enricher")

# Earth radius in miles
_EARTH_RADIUS_MILES = 3958.8

# Default impossible travel speed threshold (mph)
_IMPOSSIBLE_TRAVEL_SPEED_MPH = 500


def _load_risk_countries() -> dict[str, dict[str, Any]]:
    """Load high-risk country configuration from YAML."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "geo_risk_countries.yaml"
    if not config_path.exists():
        logger.warning("geo_risk_countries.yaml not found, using empty risk list")
        return {}
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("countries", {})


@dataclass(frozen=True)
class GeoLocation:
    """Resolved geolocation data."""

    country_code: str
    country_name: str
    city: str
    latitude: float
    longitude: float
    is_vpn: bool = False
    is_proxy: bool = False
    is_tor: bool = False


@dataclass
class GeoEnrichmentResult:
    """Result of geo-enrichment for a transaction."""

    location: GeoLocation | None = None
    distance_from_home_miles: float | None = None
    country_risk_score: float = 0.0
    is_high_risk_country: bool = False
    travel_speed_mph: float | None = None
    is_impossible_travel: bool = False
    enrichment_latency_ms: float = 0.0
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "geo_country_code": self.location.country_code if self.location else None,
            "geo_country_name": self.location.country_name if self.location else None,
            "geo_city": self.location.city if self.location else None,
            "geo_latitude": self.location.latitude if self.location else None,
            "geo_longitude": self.location.longitude if self.location else None,
            "geo_is_vpn": self.location.is_vpn if self.location else False,
            "geo_is_proxy": self.location.is_proxy if self.location else False,
            "geo_is_tor": self.location.is_tor if self.location else False,
            "geo_distance_from_home_miles": self.distance_from_home_miles,
            "geo_country_risk_score": self.country_risk_score,
            "geo_is_high_risk_country": self.is_high_risk_country,
            "geo_travel_speed_mph": self.travel_speed_mph,
            "geo_is_impossible_travel": self.is_impossible_travel,
        }
        return result


class GeoIPProvider:
    """Abstract interface for GeoIP lookups.

    Production implementations can wrap MaxMind GeoLite2, ip-api, or any
    geolocation service. The default implementation uses transaction-provided
    geo fields as a fallback when no external provider is configured.
    """

    def lookup(self, ip_address: str) -> GeoLocation | None:
        """Resolve an IP address to a GeoLocation.

        Args:
            ip_address: IPv4 or IPv6 address string.

        Returns:
            GeoLocation if resolved, None otherwise.
        """
        raise NotImplementedError


class TransactionFieldGeoProvider(GeoIPProvider):
    """Fallback provider that extracts geo data from the transaction itself.

    Used when no external GeoIP database is configured. Relies on geo fields
    already present in the transaction payload (geo_latitude, geo_longitude,
    geo_country, geo_city).
    """

    def lookup(
        self, ip_address: str, transaction: dict[str, Any] | None = None
    ) -> GeoLocation | None:
        if transaction is None:
            return None

        lat = transaction.get("geo_latitude")
        lng = transaction.get("geo_longitude")
        country = transaction.get("geo_country", "")
        city = transaction.get("geo_city", "")

        if lat is None or lng is None:
            return None

        return GeoLocation(
            country_code=country,
            country_name=country,
            city=city,
            latitude=float(lat),
            longitude=float(lng),
        )


class GeoEnricher:
    """Enriches transactions with geolocation data and travel velocity analysis.

    Features:
    - IP → geolocation resolution
    - Haversine distance from customer's home location
    - Country risk scoring based on configurable risk list
    - Impossible travel detection between consecutive transactions

    Usage:
        enricher = GeoEnricher()
        result = enricher.enrich(transaction, customer_profile, last_transaction)
    """

    def __init__(
        self,
        geo_provider: GeoIPProvider | None = None,
        impossible_travel_speed_mph: float = _IMPOSSIBLE_TRAVEL_SPEED_MPH,
    ) -> None:
        self._provider = geo_provider or TransactionFieldGeoProvider()
        self._impossible_travel_speed_mph = impossible_travel_speed_mph
        self._risk_countries = _load_risk_countries()

    def enrich(
        self,
        transaction: dict[str, Any],
        customer_profile: dict[str, Any] | None = None,
        last_transaction: dict[str, Any] | None = None,
    ) -> GeoEnrichmentResult:
        """Enrich a transaction with geolocation context.

        Args:
            transaction: Current transaction record.
            customer_profile: Customer's profile with home_latitude, home_longitude.
            last_transaction: Previous transaction for travel velocity calculation.

        Returns:
            GeoEnrichmentResult with all geo-enrichment fields.
        """
        start = time.perf_counter()
        result = GeoEnrichmentResult()

        try:
            # Resolve geolocation
            ip_address = transaction.get("ip_address", "")
            if isinstance(self._provider, TransactionFieldGeoProvider):
                location = self._provider.lookup(ip_address, transaction=transaction)
            else:
                location = self._provider.lookup(ip_address)

            result.location = location

            if location:
                # Country risk scoring
                result.country_risk_score = self._score_country_risk(location.country_code)
                result.is_high_risk_country = result.country_risk_score >= 0.7

                # Distance from home
                profile = customer_profile or {}
                home_lat = profile.get("home_latitude")
                home_lng = profile.get("home_longitude")
                if home_lat is not None and home_lng is not None:
                    result.distance_from_home_miles = haversine_distance(
                        home_lat, home_lng, location.latitude, location.longitude
                    )

                # Travel velocity check
                if last_transaction:
                    speed = self._calculate_travel_speed(location, last_transaction)
                    if speed is not None:
                        result.travel_speed_mph = speed
                        result.is_impossible_travel = speed > self._impossible_travel_speed_mph

        except Exception as e:
            result.error = str(e)
            logger.error(
                "Geo-enrichment failed",
                error=str(e),
                transaction_id=transaction.get("external_transaction_id"),
            )

        result.enrichment_latency_ms = (time.perf_counter() - start) * 1000
        return result

    def _score_country_risk(self, country_code: str) -> float:
        """Score country risk from 0.0 (safe) to 1.0 (highest risk).

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            Risk score float between 0.0 and 1.0.
        """
        if not country_code:
            return 0.5  # Unknown country gets moderate risk

        country_info = self._risk_countries.get(country_code.upper())
        if country_info is None:
            return 0.1  # Unlisted countries get low risk

        return float(country_info.get("risk_score", 0.1))

    def _calculate_travel_speed(
        self, current_location: GeoLocation, last_transaction: dict[str, Any]
    ) -> float | None:
        """Calculate travel speed between current and last transaction.

        Args:
            current_location: Resolved location of current transaction.
            last_transaction: Previous transaction with geo and timestamp fields.

        Returns:
            Speed in mph, or None if calculation not possible.
        """
        last_lat = last_transaction.get("geo_latitude")
        last_lng = last_transaction.get("geo_longitude")
        last_ts = last_transaction.get("transaction_timestamp")
        current_ts_str = last_transaction.get("_current_timestamp")

        if last_lat is None or last_lng is None or last_ts is None:
            return None

        # Calculate distance
        distance = haversine_distance(
            float(last_lat),
            float(last_lng),
            current_location.latitude,
            current_location.longitude,
        )

        # Calculate time difference
        if isinstance(last_ts, str):
            last_time = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        elif isinstance(last_ts, datetime):
            last_time = last_ts
        else:
            return None

        # Use current transaction timestamp if available
        current_time = datetime.now(timezone.utc)
        if current_ts_str:
            if isinstance(current_ts_str, str):
                current_time = datetime.fromisoformat(current_ts_str.replace("Z", "+00:00"))
            elif isinstance(current_ts_str, datetime):
                current_time = current_ts_str

        time_diff_hours = abs((current_time - last_time).total_seconds()) / 3600.0

        if time_diff_hours < 0.001:  # Less than ~3.6 seconds
            return None

        return distance / time_diff_hours

    def enrich_batch(
        self,
        transactions: list[dict[str, Any]],
        customer_profiles: dict[str, dict[str, Any]] | None = None,
        last_transactions: dict[str, dict[str, Any]] | None = None,
    ) -> list[GeoEnrichmentResult]:
        """Enrich a batch of transactions.

        Args:
            transactions: List of transaction records.
            customer_profiles: Map of customer_id → profile.
            last_transactions: Map of customer_id → last transaction.

        Returns:
            List of GeoEnrichmentResult in same order as input.
        """
        profiles = customer_profiles or {}
        last_txns = last_transactions or {}
        results = []

        for txn in transactions:
            customer_id = txn.get("customer_id", "")
            profile = profiles.get(customer_id)
            last_txn = last_txns.get(customer_id)
            results.append(self.enrich(txn, profile, last_txn))

        return results


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points using Haversine formula.

    Args:
        lat1: Latitude of point 1 in degrees.
        lon1: Longitude of point 1 in degrees.
        lat2: Latitude of point 2 in degrees.
        lon2: Longitude of point 2 in degrees.

    Returns:
        Distance in miles.
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return _EARTH_RADIUS_MILES * c
