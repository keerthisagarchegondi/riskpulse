"""Device intelligence enrichment module.

Enriches transactions with device-related context:
- Device fingerprint analysis
- Known device detection (seen before for this customer)
- Device type classification
- Multiple accounts per device detection
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__, component="device_enricher")

# Device age thresholds
_NEW_DEVICE_THRESHOLD_DAYS = 1
_SUSPICIOUS_MULTI_ACCOUNT_THRESHOLD = 3


@dataclass
class DeviceInfo:
    """Parsed device intelligence data."""

    device_id: str
    device_type: str  # mobile, desktop, tablet, unknown
    fingerprint_hash: str
    os_family: str
    browser_family: str
    is_emulator: bool = False
    is_rooted: bool = False


@dataclass
class DeviceEnrichmentResult:
    """Result of device enrichment for a transaction."""

    device_info: DeviceInfo | None = None
    is_known_device: bool = False
    device_age_days: float | None = None
    is_new_device: bool = True
    device_trust_score: float = 0.0
    accounts_on_device: int = 0
    is_multi_account_device: bool = False
    transactions_on_device: int = 0
    device_risk_score: float = 0.0
    enrichment_latency_ms: float = 0.0
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_info.device_id if self.device_info else None,
            "device_type": self.device_info.device_type if self.device_info else "unknown",
            "device_fingerprint_hash": (
                self.device_info.fingerprint_hash if self.device_info else None
            ),
            "device_os_family": self.device_info.os_family if self.device_info else None,
            "device_browser_family": self.device_info.browser_family if self.device_info else None,
            "device_is_emulator": self.device_info.is_emulator if self.device_info else False,
            "device_is_rooted": self.device_info.is_rooted if self.device_info else False,
            "device_is_known": self.is_known_device,
            "device_age_days": self.device_age_days,
            "device_is_new": self.is_new_device,
            "device_trust_score": self.device_trust_score,
            "device_accounts_count": self.accounts_on_device,
            "device_is_multi_account": self.is_multi_account_device,
            "device_transactions_count": self.transactions_on_device,
            "device_risk_score": self.device_risk_score,
        }


class DeviceStore:
    """Abstract interface for device history storage.

    Production implementations should use Redis or PostgreSQL to persist
    device history, customer-device mappings, and trust scores.
    """

    def get_device_history(self, device_id: str) -> dict[str, Any] | None:
        """Get stored history for a device.

        Returns:
            Dict with keys: first_seen, last_seen, customer_ids, transaction_count, trust_score
        """
        raise NotImplementedError

    def get_customer_devices(self, customer_id: str) -> list[str]:
        """Get all device IDs associated with a customer."""
        raise NotImplementedError

    def record_device_activity(self, device_id: str, customer_id: str, timestamp: datetime) -> None:
        """Record a device being used by a customer at a given time."""
        raise NotImplementedError


class InMemoryDeviceStore(DeviceStore):
    """In-memory device store for testing and development.

    Not suitable for production — use Redis-backed or DB-backed store.
    """

    def __init__(self) -> None:
        self._devices: dict[str, dict[str, Any]] = {}
        self._customer_devices: dict[str, set[str]] = {}

    def get_device_history(self, device_id: str) -> dict[str, Any] | None:
        return self._devices.get(device_id)

    def get_customer_devices(self, customer_id: str) -> list[str]:
        return list(self._customer_devices.get(customer_id, set()))

    def record_device_activity(self, device_id: str, customer_id: str, timestamp: datetime) -> None:
        if device_id not in self._devices:
            self._devices[device_id] = {
                "first_seen": timestamp,
                "last_seen": timestamp,
                "customer_ids": {customer_id},
                "transaction_count": 1,
                "trust_score": 0.5,
            }
        else:
            record = self._devices[device_id]
            record["last_seen"] = timestamp
            record["customer_ids"].add(customer_id)
            record["transaction_count"] += 1
            # Trust increases with usage (capped at 1.0)
            record["trust_score"] = min(
                1.0, record["trust_score"] + 0.01 * record["transaction_count"]
            )

        self._customer_devices.setdefault(customer_id, set()).add(device_id)


class DeviceEnricher:
    """Enriches transactions with device intelligence.

    Features:
    - Device fingerprint hashing and classification
    - Known device detection per customer
    - Multi-account device detection
    - Device risk scoring

    Usage:
        store = InMemoryDeviceStore()
        enricher = DeviceEnricher(device_store=store)
        result = enricher.enrich(transaction, customer_profile)
    """

    def __init__(
        self,
        device_store: DeviceStore | None = None,
        new_device_threshold_days: float = _NEW_DEVICE_THRESHOLD_DAYS,
        multi_account_threshold: int = _SUSPICIOUS_MULTI_ACCOUNT_THRESHOLD,
    ) -> None:
        self._store = device_store or InMemoryDeviceStore()
        self._new_device_threshold_days = new_device_threshold_days
        self._multi_account_threshold = multi_account_threshold

    def enrich(
        self,
        transaction: dict[str, Any],
        customer_profile: dict[str, Any] | None = None,
    ) -> DeviceEnrichmentResult:
        """Enrich a transaction with device intelligence.

        Args:
            transaction: Current transaction record with device_id, device_type fields.
            customer_profile: Customer profile with known_devices list.

        Returns:
            DeviceEnrichmentResult with all device enrichment fields.
        """
        start = time.perf_counter()
        result = DeviceEnrichmentResult()

        try:
            device_id = transaction.get("device_id", "")
            if not device_id:
                result.device_risk_score = 0.8  # Missing device is risky
                result.enrichment_latency_ms = (time.perf_counter() - start) * 1000
                return result

            # Parse device info
            device_info = self._parse_device_info(transaction)
            result.device_info = device_info

            # Look up device history
            history = self._store.get_device_history(device_id)
            profile = customer_profile or {}
            customer_id = transaction.get("customer_id", "")

            if history:
                # Known device checks
                result.is_known_device = customer_id in history.get("customer_ids", set())
                result.transactions_on_device = history.get("transaction_count", 0)
                result.device_trust_score = history.get("trust_score", 0.5)

                # Device age
                first_seen = history.get("first_seen")
                if first_seen:
                    now = datetime.now(timezone.utc)
                    if isinstance(first_seen, datetime):
                        age = (now - first_seen).total_seconds() / 86400.0
                    else:
                        age = 0.0
                    result.device_age_days = age
                    result.is_new_device = age < self._new_device_threshold_days

                # Multi-account detection
                account_ids = history.get("customer_ids", set())
                result.accounts_on_device = len(account_ids)
                result.is_multi_account_device = (
                    result.accounts_on_device >= self._multi_account_threshold
                )
            else:
                # Brand new device
                result.is_known_device = False
                result.is_new_device = True
                result.device_age_days = 0.0
                result.device_trust_score = 0.3

            # Also check against customer's known devices list
            known_devices = profile.get("known_devices", [])
            if known_devices and device_id not in known_devices:
                result.is_known_device = False

            # Compute risk score
            result.device_risk_score = self._compute_risk_score(result)

            # Record this activity
            txn_timestamp = self._parse_timestamp(transaction.get("transaction_timestamp"))
            self._store.record_device_activity(device_id, customer_id, txn_timestamp)

        except Exception as e:
            result.error = str(e)
            logger.error(
                "Device enrichment failed",
                error=str(e),
                transaction_id=transaction.get("external_transaction_id"),
            )

        result.enrichment_latency_ms = (time.perf_counter() - start) * 1000
        return result

    def _parse_device_info(self, transaction: dict[str, Any]) -> DeviceInfo:
        """Extract and normalize device information from transaction."""
        device_id = transaction.get("device_id", "unknown")
        device_type = transaction.get("device_type", "unknown").lower()

        # Normalize device type
        valid_types = {"mobile", "desktop", "tablet", "pos", "atm"}
        if device_type not in valid_types:
            device_type = "unknown"

        # Build fingerprint from available attributes
        fingerprint_parts = [
            device_id,
            device_type,
            transaction.get("user_agent", ""),
            transaction.get("screen_resolution", ""),
            transaction.get("timezone_offset", ""),
        ]
        fingerprint_hash = hashlib.sha256(
            "|".join(str(p) for p in fingerprint_parts).encode()
        ).hexdigest()[:16]

        return DeviceInfo(
            device_id=device_id,
            device_type=device_type,
            fingerprint_hash=fingerprint_hash,
            os_family=transaction.get("os_family", "unknown"),
            browser_family=transaction.get("browser_family", "unknown"),
            is_emulator=transaction.get("is_emulator", False),
            is_rooted=transaction.get("is_rooted", False),
        )

    def _compute_risk_score(self, result: DeviceEnrichmentResult) -> float:
        """Compute device risk score from 0.0 (safe) to 1.0 (highest risk).

        Risk factors:
        - New/unknown device: +0.3
        - Multi-account device: +0.3
        - Emulator/rooted: +0.2
        - Low trust score inverts to risk
        """
        score = 0.0

        # New or unknown device
        if not result.is_known_device:
            score += 0.3
        elif result.is_new_device:
            score += 0.15

        # Multi-account
        if result.is_multi_account_device:
            score += 0.3

        # Emulator/rooted flags
        if result.device_info:
            if result.device_info.is_emulator:
                score += 0.2
            if result.device_info.is_rooted:
                score += 0.1

        # Trust score inversely contributes
        trust_penalty = max(0.0, 0.2 * (1.0 - result.device_trust_score))
        score += trust_penalty

        return min(1.0, score)

    @staticmethod
    def _parse_timestamp(ts: Any) -> datetime:
        """Parse a timestamp from various formats to datetime."""
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        if isinstance(ts, str):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.now(timezone.utc)

    def enrich_batch(
        self,
        transactions: list[dict[str, Any]],
        customer_profiles: dict[str, dict[str, Any]] | None = None,
    ) -> list[DeviceEnrichmentResult]:
        """Enrich a batch of transactions with device intelligence.

        Args:
            transactions: List of transaction records.
            customer_profiles: Map of customer_id → profile.

        Returns:
            List of DeviceEnrichmentResult in same order as input.
        """
        profiles = customer_profiles or {}
        results = []
        for txn in transactions:
            customer_id = txn.get("customer_id", "")
            profile = profiles.get(customer_id)
            results.append(self.enrich(txn, profile))
        return results
