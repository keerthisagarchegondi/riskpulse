"""Merchant enrichment module.

Enriches transactions with merchant-related context:
- Merchant risk categorization by MCC (Merchant Category Code)
- Historical fraud rate for the merchant
- MCC-based risk scoring
- New merchant detection
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__, component="merchant_enricher")

# Default thresholds
_HIGH_FRAUD_RATE_THRESHOLD = 0.05  # 5% fraud rate
_NEW_MERCHANT_THRESHOLD_DAYS = 30


def _load_merchant_risk_categories() -> dict[str, dict[str, Any]]:
    """Load merchant risk category configuration from YAML."""
    config_path = (
        Path(__file__).resolve().parents[2] / "config" / "merchant_risk_categories.yaml"
    )
    if not config_path.exists():
        logger.warning("merchant_risk_categories.yaml not found, using defaults")
        return {}
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("mcc_categories", {})


@dataclass
class MerchantInfo:
    """Resolved merchant data."""

    merchant_id: str
    merchant_name: str
    mcc: str
    category_name: str
    risk_category: str  # low, medium, high, critical
    first_seen: datetime | None = None
    total_transactions: int = 0
    fraud_count: int = 0
    fraud_rate: float = 0.0


@dataclass
class MerchantEnrichmentResult:
    """Result of merchant enrichment for a transaction."""

    merchant_info: MerchantInfo | None = None
    mcc_risk_score: float = 0.0
    merchant_fraud_rate: float = 0.0
    is_high_fraud_merchant: bool = False
    is_new_merchant: bool = True
    merchant_age_days: float | None = None
    merchant_risk_score: float = 0.0
    enrichment_latency_ms: float = 0.0
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "merchant_id": self.merchant_info.merchant_id if self.merchant_info else None,
            "merchant_name": self.merchant_info.merchant_name if self.merchant_info else None,
            "merchant_mcc": self.merchant_info.mcc if self.merchant_info else None,
            "merchant_category_name": (
                self.merchant_info.category_name if self.merchant_info else None
            ),
            "merchant_risk_category": (
                self.merchant_info.risk_category if self.merchant_info else "unknown"
            ),
            "merchant_mcc_risk_score": self.mcc_risk_score,
            "merchant_fraud_rate": self.merchant_fraud_rate,
            "merchant_is_high_fraud": self.is_high_fraud_merchant,
            "merchant_is_new": self.is_new_merchant,
            "merchant_age_days": self.merchant_age_days,
            "merchant_risk_score": self.merchant_risk_score,
        }


class MerchantStore:
    """Abstract interface for merchant history storage.

    Production implementations should use PostgreSQL or Redis to track
    merchant fraud statistics and history.
    """

    def get_merchant_history(self, merchant_id: str) -> dict[str, Any] | None:
        """Get stored history for a merchant.

        Returns:
            Dict with keys: first_seen, total_transactions, fraud_count, fraud_rate
        """
        raise NotImplementedError

    def record_merchant_transaction(
        self, merchant_id: str, merchant_name: str, mcc: str, timestamp: datetime
    ) -> None:
        """Record a transaction for a merchant."""
        raise NotImplementedError

    def record_merchant_fraud(self, merchant_id: str) -> None:
        """Record a fraud incident for a merchant."""
        raise NotImplementedError


class InMemoryMerchantStore(MerchantStore):
    """In-memory merchant store for testing and development."""

    def __init__(self) -> None:
        self._merchants: dict[str, dict[str, Any]] = {}

    def get_merchant_history(self, merchant_id: str) -> dict[str, Any] | None:
        return self._merchants.get(merchant_id)

    def record_merchant_transaction(
        self, merchant_id: str, merchant_name: str, mcc: str, timestamp: datetime
    ) -> None:
        if merchant_id not in self._merchants:
            self._merchants[merchant_id] = {
                "merchant_name": merchant_name,
                "mcc": mcc,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "total_transactions": 1,
                "fraud_count": 0,
                "fraud_rate": 0.0,
            }
        else:
            record = self._merchants[merchant_id]
            record["last_seen"] = timestamp
            record["total_transactions"] += 1
            if record["total_transactions"] > 0:
                record["fraud_rate"] = (
                    record["fraud_count"] / record["total_transactions"]
                )

    def record_merchant_fraud(self, merchant_id: str) -> None:
        if merchant_id in self._merchants:
            record = self._merchants[merchant_id]
            record["fraud_count"] += 1
            if record["total_transactions"] > 0:
                record["fraud_rate"] = (
                    record["fraud_count"] / record["total_transactions"]
                )


class MerchantEnricher:
    """Enriches transactions with merchant intelligence.

    Features:
    - MCC-based risk categorization
    - Historical fraud rate lookup
    - New merchant detection
    - Combined merchant risk scoring

    Usage:
        enricher = MerchantEnricher()
        result = enricher.enrich(transaction)
    """

    def __init__(
        self,
        merchant_store: MerchantStore | None = None,
        high_fraud_rate_threshold: float = _HIGH_FRAUD_RATE_THRESHOLD,
        new_merchant_threshold_days: float = _NEW_MERCHANT_THRESHOLD_DAYS,
    ) -> None:
        self._store = merchant_store or InMemoryMerchantStore()
        self._high_fraud_threshold = high_fraud_rate_threshold
        self._new_merchant_threshold_days = new_merchant_threshold_days
        self._mcc_categories = _load_merchant_risk_categories()

    def enrich(self, transaction: dict[str, Any]) -> MerchantEnrichmentResult:
        """Enrich a transaction with merchant intelligence.

        Args:
            transaction: Current transaction record with merchant fields.

        Returns:
            MerchantEnrichmentResult with all merchant enrichment fields.
        """
        start = time.perf_counter()
        result = MerchantEnrichmentResult()

        try:
            merchant_id = transaction.get("merchant_id", "")
            merchant_name = transaction.get("merchant_name", "")
            mcc = transaction.get("merchant_category_code", "")

            if not merchant_id:
                result.merchant_risk_score = 0.5
                result.enrichment_latency_ms = (time.perf_counter() - start) * 1000
                return result

            # MCC risk scoring
            mcc_info = self._get_mcc_info(mcc)
            result.mcc_risk_score = mcc_info.get("risk_score", 0.3)
            category_name = mcc_info.get("category_name", "Unknown")
            risk_category = mcc_info.get("risk_level", "medium")

            # Merchant history lookup
            history = self._store.get_merchant_history(merchant_id)

            if history:
                result.merchant_fraud_rate = history.get("fraud_rate", 0.0)
                result.is_high_fraud_merchant = (
                    result.merchant_fraud_rate >= self._high_fraud_threshold
                )

                first_seen = history.get("first_seen")
                if first_seen:
                    now = datetime.now(timezone.utc)
                    if isinstance(first_seen, datetime):
                        age = (now - first_seen).total_seconds() / 86400.0
                    else:
                        age = 0.0
                    result.merchant_age_days = age
                    result.is_new_merchant = age < self._new_merchant_threshold_days

                result.merchant_info = MerchantInfo(
                    merchant_id=merchant_id,
                    merchant_name=merchant_name,
                    mcc=mcc,
                    category_name=category_name,
                    risk_category=risk_category,
                    first_seen=first_seen,
                    total_transactions=history.get("total_transactions", 0),
                    fraud_count=history.get("fraud_count", 0),
                    fraud_rate=result.merchant_fraud_rate,
                )
            else:
                # New merchant — never seen before
                result.is_new_merchant = True
                result.merchant_age_days = 0.0
                result.merchant_info = MerchantInfo(
                    merchant_id=merchant_id,
                    merchant_name=merchant_name,
                    mcc=mcc,
                    category_name=category_name,
                    risk_category=risk_category,
                )

            # Compute combined risk score
            result.merchant_risk_score = self._compute_risk_score(result)

            # Record this transaction
            txn_timestamp = self._parse_timestamp(
                transaction.get("transaction_timestamp")
            )
            self._store.record_merchant_transaction(
                merchant_id, merchant_name, mcc, txn_timestamp
            )

        except Exception as e:
            result.error = str(e)
            logger.error(
                "Merchant enrichment failed",
                error=str(e),
                transaction_id=transaction.get("external_transaction_id"),
            )

        result.enrichment_latency_ms = (time.perf_counter() - start) * 1000
        return result

    def _get_mcc_info(self, mcc: str) -> dict[str, Any]:
        """Get risk info for a Merchant Category Code.

        Args:
            mcc: 4-digit MCC string.

        Returns:
            Dict with risk_score, category_name, risk_level.
        """
        if not mcc:
            return {"risk_score": 0.5, "category_name": "Unknown", "risk_level": "medium"}

        # Check direct MCC match
        if mcc in self._mcc_categories:
            return self._mcc_categories[mcc]

        # Check MCC range (first 2 digits)
        mcc_prefix = mcc[:2] + "xx"
        if mcc_prefix in self._mcc_categories:
            return self._mcc_categories[mcc_prefix]

        return {"risk_score": 0.3, "category_name": "General", "risk_level": "low"}

    def _compute_risk_score(self, result: MerchantEnrichmentResult) -> float:
        """Compute combined merchant risk score from 0.0 to 1.0.

        Risk factors:
        - MCC category risk: weighted 0.4
        - Fraud rate: weighted 0.3
        - New merchant: weighted 0.2
        - Missing data: weighted 0.1
        """
        score = 0.0

        # MCC risk (40% weight)
        score += 0.4 * result.mcc_risk_score

        # Fraud rate (30% weight) — scaled to 0-1 range (cap at 20% fraud rate)
        fraud_factor = min(1.0, result.merchant_fraud_rate / 0.20)
        score += 0.3 * fraud_factor

        # New merchant penalty (20% weight)
        if result.is_new_merchant:
            score += 0.2 * 0.6  # New merchants get moderate risk boost

        # Missing info penalty (10% weight)
        if result.merchant_info is None or not result.merchant_info.mcc:
            score += 0.1

        return min(1.0, score)

    @staticmethod
    def _parse_timestamp(ts: Any) -> datetime:
        """Parse a timestamp from various formats."""
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        if isinstance(ts, str):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.now(timezone.utc)

    def enrich_batch(
        self, transactions: list[dict[str, Any]]
    ) -> list[MerchantEnrichmentResult]:
        """Enrich a batch of transactions with merchant intelligence.

        Args:
            transactions: List of transaction records.

        Returns:
            List of MerchantEnrichmentResult in same order as input.
        """
        return [self.enrich(txn) for txn in transactions]
