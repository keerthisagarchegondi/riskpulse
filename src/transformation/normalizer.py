"""Data normalization transformations for transaction records.

Standardizes categorical fields, converts currencies, maps merchant
categories, and normalizes country codes using YAML-configurable
reference data with hot-reload support.

Performance target: < 5ms per record.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__, component="data_normalizer")


# --- Metrics ---


@dataclass
class NormalizationMetrics:
    """Thread-safe metrics for normalization operations."""

    total_records_processed: int = 0
    currencies_converted: int = 0
    countries_normalized: int = 0
    mcc_mapped: int = 0
    transaction_types_normalized: int = 0
    channels_normalized: int = 0
    card_types_normalized: int = 0
    currencies_standardized: int = 0
    amounts_converted: int = 0
    unknown_mappings: int = 0
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
                "currencies_converted": self.currencies_converted,
                "countries_normalized": self.countries_normalized,
                "mcc_mapped": self.mcc_mapped,
                "transaction_types_normalized": self.transaction_types_normalized,
                "channels_normalized": self.channels_normalized,
                "card_types_normalized": self.card_types_normalized,
                "currencies_standardized": self.currencies_standardized,
                "amounts_converted": self.amounts_converted,
                "unknown_mappings": self.unknown_mappings,
            }

    def reset(self) -> None:
        with self._lock:
            self.total_records_processed = 0
            self.currencies_converted = 0
            self.countries_normalized = 0
            self.mcc_mapped = 0
            self.transaction_types_normalized = 0
            self.channels_normalized = 0
            self.card_types_normalized = 0
            self.currencies_standardized = 0
            self.amounts_converted = 0
            self.unknown_mappings = 0


# --- Normalization Result ---


@dataclass
class NormalizationResult:
    """Result of normalizing a single transaction record."""

    record: dict[str, Any]
    changes: list[str] = field(default_factory=list)
    amount_usd: float | None = None
    original_currency: str | None = None
    mcc_category: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record,
            "changes": self.changes,
            "amount_usd": self.amount_usd,
            "original_currency": self.original_currency,
            "mcc_category": self.mcc_category,
            "latency_ms": round(self.latency_ms, 4),
        }


# --- Data Normalizer ---


class DataNormalizer:
    """Production data normalizer for transaction records.

    Normalizes categorical fields using YAML-configurable reference
    mappings with support for hot-reload.

    Normalization steps:
    1. Currency code standardization
    2. Amount conversion to base currency (USD)
    3. Country code normalization (→ ISO 3166-1 alpha-2)
    4. MCC code → readable category mapping
    5. Transaction type standardization
    6. Channel normalization
    7. Card type normalization

    Thread-safe for concurrent processing.
    """

    def __init__(self, mappings_path: str | Path | None = None) -> None:
        self._mappings_path = self._resolve_mappings_path(mappings_path)
        self._lock = Lock()
        self._config_hash: str = ""
        self._last_load_time: float = 0.0
        self._reload_interval_seconds: float = 30.0

        # Mapping tables (populated from YAML)
        self._exchange_rates: dict[str, float] = {}
        self._base_currency: str = "USD"
        self._country_codes: dict[str, str] = {}
        self._mcc_categories: dict[str, str] = {}
        self._transaction_type_aliases: dict[str, str] = {}
        self._channel_aliases: dict[str, str] = {}
        self._card_type_aliases: dict[str, str] = {}
        self._currency_aliases: dict[str, str] = {}
        self._pii_fields: list[str] = []

        self.metrics = NormalizationMetrics()

        self._load_mappings()

    @staticmethod
    def _resolve_mappings_path(mappings_path: str | Path | None) -> Path:
        """Resolve the normalization mappings YAML path."""
        if mappings_path:
            return Path(mappings_path)
        settings = get_settings()
        project_root = Path(settings.get("project_root", "."))
        candidate = project_root / "config" / "normalization_mappings.yaml"
        if candidate.exists():
            return candidate
        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            candidate = parent / "config" / "normalization_mappings.yaml"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "normalization_mappings.yaml not found. Provide explicit path or place in config/"
        )

    def _load_mappings(self) -> None:
        """Load reference mappings from YAML."""
        try:
            with open(self._mappings_path, "r", encoding="utf-8") as f:
                content = f.read()

            config_hash = hashlib.md5(content.encode()).hexdigest()
            if config_hash == self._config_hash:
                return

            data = yaml.safe_load(content)
            if not data:
                logger.warning("mappings_config_empty", path=str(self._mappings_path))
                return

            with self._lock:
                # Exchange rates
                exchange = data.get("exchange_rates", {})
                self._base_currency = exchange.get("base_currency", "USD")
                self._exchange_rates = {
                    k.upper(): float(v) for k, v in exchange.get("rates", {}).items()
                }

                # Country codes — build lowercase lookup
                raw_countries = data.get("country_codes", {})
                self._country_codes = {
                    str(k).lower().strip(): v for k, v in raw_countries.items()
                }

                # MCC categories
                self._mcc_categories = {
                    str(k): v for k, v in data.get("mcc_categories", {}).items()
                }

                # Aliases
                self._transaction_type_aliases = {
                    str(k).lower().strip(): v
                    for k, v in data.get("transaction_type_aliases", {}).items()
                }
                self._channel_aliases = {
                    str(k).lower().strip(): v
                    for k, v in data.get("channel_aliases", {}).items()
                }
                self._card_type_aliases = {
                    str(k).lower().strip(): v
                    for k, v in data.get("card_type_aliases", {}).items()
                }
                self._currency_aliases = {
                    str(k).lower().strip(): v
                    for k, v in data.get("currency_aliases", {}).items()
                }
                self._pii_fields = data.get("pii_fields", [])

                self._config_hash = config_hash
                self._last_load_time = time.time()

            logger.info(
                "mappings_loaded",
                exchange_rates=len(self._exchange_rates),
                country_codes=len(self._country_codes),
                mcc_categories=len(self._mcc_categories),
                path=str(self._mappings_path),
            )

        except FileNotFoundError:
            logger.error("mappings_file_not_found", path=str(self._mappings_path))
            raise
        except yaml.YAMLError as exc:
            logger.error("mappings_yaml_error", error=str(exc))
            raise

    def reload_if_needed(self) -> bool:
        """Hot-reload mappings if the file has changed."""
        if time.time() - self._last_load_time < self._reload_interval_seconds:
            return False
        try:
            self._load_mappings()
            return True
        except Exception as exc:
            logger.error("mappings_reload_failed", error=str(exc))
            return False

    def force_reload(self) -> None:
        """Force immediate reload of mappings."""
        self._config_hash = ""
        self._load_mappings()

    # --- Public API ---

    def normalize(self, record: dict[str, Any]) -> NormalizationResult:
        """Normalize a single transaction record.

        Applies all normalization steps and returns the result with
        change tracking.
        """
        start = time.perf_counter()

        self.reload_if_needed()

        changes: list[str] = []
        normalized = dict(record)

        # 1. Normalize currency code
        currency_changes = self._normalize_currency_code(normalized)
        changes.extend(currency_changes)

        # 2. Convert amount to USD
        original_currency = normalized.get("transaction_currency", "USD")
        amount_usd = self._convert_to_usd(normalized)
        if amount_usd is not None:
            normalized["transaction_amount_usd"] = amount_usd
            if original_currency != self._base_currency:
                changes.append(f"amount_converted:{original_currency}->USD")

        # 3. Normalize country code
        country_changes = self._normalize_country(normalized)
        changes.extend(country_changes)

        # 4. Map MCC to category
        mcc_category = self._map_mcc(normalized)
        if mcc_category:
            normalized["mcc_category"] = mcc_category

        # 5. Normalize transaction type
        type_changes = self._normalize_transaction_type(normalized)
        changes.extend(type_changes)

        # 6. Normalize channel
        channel_changes = self._normalize_channel(normalized)
        changes.extend(channel_changes)

        # 7. Normalize card type
        card_changes = self._normalize_card_type(normalized)
        changes.extend(card_changes)

        elapsed = (time.perf_counter() - start) * 1000
        self.metrics.increment(total_records_processed=1)

        return NormalizationResult(
            record=normalized,
            changes=changes,
            amount_usd=amount_usd,
            original_currency=original_currency,
            mcc_category=mcc_category,
            latency_ms=elapsed,
        )

    def normalize_batch(self, records: list[dict[str, Any]]) -> list[NormalizationResult]:
        """Normalize a batch of records."""
        return [self.normalize(record) for record in records]

    def convert_currency(
        self, amount: float, from_currency: str, to_currency: str = "USD"
    ) -> float | None:
        """Convert an amount between currencies.

        Returns the converted amount rounded to 4 decimal places,
        or None if either currency rate is unknown.
        """
        from_upper = from_currency.upper()
        to_upper = to_currency.upper()

        if from_upper == to_upper:
            return round(amount, 4)

        with self._lock:
            from_rate = self._exchange_rates.get(from_upper)
            to_rate = self._exchange_rates.get(to_upper)

        if from_rate is None or to_rate is None:
            return None

        # Convert: source → USD → target
        usd_amount = amount * from_rate
        result = usd_amount / to_rate
        return round(result, 4)

    def get_mcc_category(self, mcc_code: str) -> str | None:
        """Look up the category name for an MCC code."""
        with self._lock:
            return self._mcc_categories.get(str(mcc_code))

    def get_exchange_rate(self, currency: str) -> float | None:
        """Get the exchange rate for a currency to USD."""
        with self._lock:
            return self._exchange_rates.get(currency.upper())

    @property
    def supported_currencies(self) -> list[str]:
        """List of currencies with known exchange rates."""
        with self._lock:
            return sorted(self._exchange_rates.keys())

    @property
    def pii_field_names(self) -> list[str]:
        """PII field names loaded from config."""
        with self._lock:
            return list(self._pii_fields)

    # --- Normalization Steps ---

    def _normalize_currency_code(self, record: dict[str, Any]) -> list[str]:
        """Standardize currency code to uppercase ISO 4217."""
        changes = []
        raw = record.get("transaction_currency")
        if raw is None:
            return changes

        raw_str = str(raw).strip()

        # Try alias lookup first
        with self._lock:
            alias = self._currency_aliases.get(raw_str.lower())

        if alias:
            if alias != raw_str:
                record["transaction_currency"] = alias
                changes.append(f"currency_alias:{raw_str}->{alias}")
                self.metrics.increment(currencies_standardized=1)
        else:
            upper = raw_str.upper()
            if upper != raw_str:
                record["transaction_currency"] = upper
                changes.append(f"currency_uppercased:{raw_str}->{upper}")
                self.metrics.increment(currencies_standardized=1)

        return changes

    def _convert_to_usd(self, record: dict[str, Any]) -> float | None:
        """Convert transaction amount to USD."""
        amount = record.get("transaction_amount")
        currency = record.get("transaction_currency", "USD")

        if amount is None:
            return None

        try:
            amount_f = float(amount)
        except (ValueError, TypeError):
            return None

        if currency == self._base_currency:
            return round(amount_f, 4)

        with self._lock:
            rate = self._exchange_rates.get(currency.upper())

        if rate is None:
            self.metrics.increment(unknown_mappings=1)
            logger.debug(
                "unknown_currency_rate",
                currency=currency,
            )
            return None

        converted = round(amount_f * rate, 4)
        self.metrics.increment(amounts_converted=1)
        return converted

    def _normalize_country(self, record: dict[str, Any]) -> list[str]:
        """Normalize country code to ISO 3166-1 alpha-2."""
        changes = []
        raw = record.get("geo_country")
        if raw is None:
            return changes

        raw_str = str(raw).strip()

        # Check alias lookup first (handles "uk" -> "GB", etc.)
        with self._lock:
            alias_mapped = self._country_codes.get(raw_str.lower())
        if alias_mapped:
            if alias_mapped != raw_str:
                record["geo_country"] = alias_mapped
                changes.append(f"country_normalized:{raw_str}->{alias_mapped}")
                self.metrics.increment(countries_normalized=1)
            return changes

        # Already a valid 2-letter code? Keep it (uppercase)
        if len(raw_str) == 2 and raw_str.isalpha():
            upper = raw_str.upper()
            if upper != raw_str:
                record["geo_country"] = upper
                changes.append(f"country_uppercased:{raw_str}->{upper}")
                self.metrics.increment(countries_normalized=1)
            return changes

        # Look up alias
        with self._lock:
            mapped = self._country_codes.get(raw_str.lower())

        if mapped:
            record["geo_country"] = mapped
            changes.append(f"country_normalized:{raw_str}->{mapped}")
            self.metrics.increment(countries_normalized=1)
        else:
            # Try 3-letter code directly (case-sensitive lookup for alpha-3)
            with self._lock:
                mapped_upper = self._country_codes.get(raw_str.upper())
            if mapped_upper:
                record["geo_country"] = mapped_upper
                changes.append(f"country_alpha3_to_alpha2:{raw_str}->{mapped_upper}")
                self.metrics.increment(countries_normalized=1)
            elif len(raw_str) > 2:
                self.metrics.increment(unknown_mappings=1)
                logger.debug("unknown_country_code", raw_value=raw_str)

        return changes

    def _map_mcc(self, record: dict[str, Any]) -> str | None:
        """Map MCC code to readable category name."""
        mcc = record.get("merchant_category_code")
        if mcc is None:
            return None

        mcc_str = str(mcc).strip()
        with self._lock:
            category = self._mcc_categories.get(mcc_str)

        if category:
            self.metrics.increment(mcc_mapped=1)
            return category

        # Try prefix matching for airline codes (3000-3999)
        if mcc_str.isdigit() and 3000 <= int(mcc_str) <= 3999:
            with self._lock:
                prefix_category = self._mcc_categories.get("3000")
            if prefix_category:
                self.metrics.increment(mcc_mapped=1)
                return prefix_category

        # Try prefix matching for car rental (3500-3999)
        if mcc_str.isdigit() and 3500 <= int(mcc_str) <= 3999:
            with self._lock:
                prefix_category = self._mcc_categories.get("3500")
            if prefix_category:
                self.metrics.increment(mcc_mapped=1)
                return prefix_category

        return None

    def _normalize_transaction_type(self, record: dict[str, Any]) -> list[str]:
        """Normalize transaction type to canonical values."""
        changes = []
        raw = record.get("transaction_type")
        if raw is None:
            return changes

        raw_str = str(raw).strip().lower()

        with self._lock:
            canonical = self._transaction_type_aliases.get(raw_str)

        if canonical and canonical != raw_str:
            record["transaction_type"] = canonical
            changes.append(f"txn_type_normalized:{raw}->{canonical}")
            self.metrics.increment(transaction_types_normalized=1)
        elif canonical is None and raw_str not in ("purchase", "withdrawal", "transfer", "refund"):
            self.metrics.increment(unknown_mappings=1)
            logger.debug("unknown_transaction_type", raw_value=raw)

        return changes

    def _normalize_channel(self, record: dict[str, Any]) -> list[str]:
        """Normalize channel to canonical values."""
        changes = []
        raw = record.get("channel")
        if raw is None:
            return changes

        raw_str = str(raw).strip().lower()

        with self._lock:
            canonical = self._channel_aliases.get(raw_str)

        if canonical and canonical != raw_str:
            record["channel"] = canonical
            changes.append(f"channel_normalized:{raw}->{canonical}")
            self.metrics.increment(channels_normalized=1)
        elif canonical is None and raw_str not in ("online", "pos", "atm", "mobile"):
            self.metrics.increment(unknown_mappings=1)
            logger.debug("unknown_channel", raw_value=raw)

        return changes

    def _normalize_card_type(self, record: dict[str, Any]) -> list[str]:
        """Normalize card type to canonical values."""
        changes = []
        raw = record.get("card_type")
        if raw is None:
            return changes

        raw_str = str(raw).strip().lower()

        with self._lock:
            canonical = self._card_type_aliases.get(raw_str)

        if canonical and canonical != raw_str:
            record["card_type"] = canonical
            changes.append(f"card_type_normalized:{raw}->{canonical}")
            self.metrics.increment(card_types_normalized=1)
        elif canonical is None and raw_str not in ("credit", "debit", "prepaid"):
            self.metrics.increment(unknown_mappings=1)
            logger.debug("unknown_card_type", raw_value=raw)

        return changes


# --- Module-level singleton ---

_normalizer_instance: DataNormalizer | None = None
_normalizer_lock = Lock()


def get_normalizer(mappings_path: str | Path | None = None) -> DataNormalizer:
    """Get or create the singleton DataNormalizer instance."""
    global _normalizer_instance
    if _normalizer_instance is None:
        with _normalizer_lock:
            if _normalizer_instance is None:
                _normalizer_instance = DataNormalizer(mappings_path=mappings_path)
    return _normalizer_instance


def reset_normalizer() -> None:
    """Reset the singleton instance (for testing)."""
    global _normalizer_instance
    with _normalizer_lock:
        _normalizer_instance = None
