"""Comprehensive unit tests for the RiskPulse Data Cleaning & Normalization pipeline.

Tests cover:
- DataCleaner: deduplication, whitespace, encoding, string standardization,
  date parsing, imputation, PII masking, numeric cleaning, sanitization
- DataNormalizer: currency conversion, country codes, MCC mapping,
  transaction type / channel / card type normalization
- Performance benchmarks
- Edge cases and error handling
"""

from __future__ import annotations

import copy
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from src.transformation.cleaner import DataCleaner, CleaningResult
from src.transformation.normalizer import DataNormalizer, NormalizationResult, reset_normalizer


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mappings_path(tmp_path):
    """Create a temporary normalization mappings YAML."""
    data = {
        "metadata": {"schema_version": "1.0.0"},
        "exchange_rates": {
            "base_currency": "USD",
            "rates": {
                "USD": 1.0,
                "EUR": 1.0840,
                "GBP": 1.2650,
                "CAD": 0.7350,
                "JPY": 0.0064,
                "INR": 0.0120,
            },
        },
        "country_codes": {
            "united states": "US",
            "usa": "US",
            "united kingdom": "GB",
            "uk": "GB",
            "canada": "CA",
            "germany": "DE",
            "japan": "JP",
            "india": "IN",
            "russia": "RU",
            "USA": "US",
            "GBR": "GB",
            "CAN": "CA",
            "DEU": "DE",
            "JPN": "JP",
            "IND": "IN",
            "RUS": "RU",
        },
        "mcc_categories": {
            "5411": "grocery_stores",
            "5812": "restaurants",
            "5541": "gas_stations",
            "7995": "gambling_betting",
            "6051": "crypto_currency_exchange",
            "3000": "airlines",
            "3500": "car_rental",
        },
        "transaction_type_aliases": {
            "purchase": "purchase",
            "buy": "purchase",
            "sale": "purchase",
            "pos": "purchase",
            "withdrawal": "withdrawal",
            "withdraw": "withdrawal",
            "cash_out": "withdrawal",
            "transfer": "transfer",
            "xfer": "transfer",
            "wire": "transfer",
            "refund": "refund",
            "return": "refund",
            "reversal": "refund",
            "chargeback": "refund",
        },
        "channel_aliases": {
            "online": "online",
            "web": "online",
            "ecommerce": "online",
            "e-commerce": "online",
            "pos": "pos",
            "in_store": "pos",
            "in-store": "pos",
            "terminal": "pos",
            "contactless": "pos",
            "atm": "atm",
            "mobile": "mobile",
            "app": "mobile",
            "mobile_app": "mobile",
        },
        "card_type_aliases": {
            "credit": "credit",
            "credit_card": "credit",
            "visa": "credit",
            "mastercard": "credit",
            "debit": "debit",
            "debit_card": "debit",
            "prepaid": "prepaid",
            "gift_card": "prepaid",
        },
        "currency_aliases": {
            "dollar": "USD",
            "dollars": "USD",
            "usd": "USD",
            "euro": "EUR",
            "euros": "EUR",
            "pound": "GBP",
            "yen": "JPY",
        },
        "pii_fields": [
            "card_number",
            "card_last_four",
            "ssn",
            "email",
            "phone",
            "ip_address",
            "device_id",
        ],
    }
    path = tmp_path / "normalization_mappings.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return path


@pytest.fixture
def cleaner():
    """Create a fresh DataCleaner."""
    return DataCleaner()


@pytest.fixture
def normalizer(mappings_path):
    """Create a fresh DataNormalizer with test mappings."""
    reset_normalizer()
    n = DataNormalizer(mappings_path=mappings_path)
    yield n
    reset_normalizer()


@pytest.fixture
def base_txn():
    """Standard valid transaction record."""
    return {
        "external_transaction_id": "TXN-2026-001",
        "account_id": "ACC-12345",
        "customer_id": "CUST-67890",
        "merchant_id": "MERCH-11111",
        "merchant_name": "Amazon Online Store",
        "merchant_category_code": "5411",
        "transaction_amount": 125.50,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": "credit",
        "card_last_four": "4242",
        "ip_address": "192.168.1.100",
        "device_id": "device-abc-123",
        "device_type": "mobile",
        "geo_latitude": 40.7128,
        "geo_longitude": -74.0060,
        "geo_country": "US",
        "geo_city": "New York",
        "is_international": False,
        "transaction_timestamp": "2026-06-15T10:30:00Z",
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLEANER TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestDeduplication:
    """Tests for idempotency-key based deduplication."""

    def test_first_record_not_duplicate(self, cleaner, base_txn):
        result = cleaner.clean(base_txn)
        assert result.is_duplicate is False

    def test_exact_duplicate_detected(self, cleaner, base_txn):
        cleaner.clean(base_txn)
        result = cleaner.clean(base_txn)
        assert result.is_duplicate is True
        assert "duplicate_detected" in result.changes

    def test_different_txn_not_duplicate(self, cleaner, base_txn):
        cleaner.clean(base_txn)
        txn2 = copy.deepcopy(base_txn)
        txn2["external_transaction_id"] = "TXN-2026-002"
        result = cleaner.clean(txn2)
        assert result.is_duplicate is False

    def test_dedup_metrics(self, cleaner, base_txn):
        cleaner.clean(base_txn)
        cleaner.clean(base_txn)
        assert cleaner.metrics.total_records_deduplicated == 1

    def test_dedup_cache_reset(self, cleaner, base_txn):
        cleaner.clean(base_txn)
        cleaner.reset_dedup_cache()
        result = cleaner.clean(base_txn)
        assert result.is_duplicate is False

    def test_dedup_cache_size(self, cleaner, base_txn):
        cleaner.clean(base_txn)
        assert cleaner.dedup_cache_size == 1


class TestWhitespaceCleaning:
    """Tests for whitespace trimming and collapsing."""

    def test_leading_trailing_whitespace(self, cleaner, base_txn):
        base_txn["account_id"] = "  ACC-12345  "
        result = cleaner.clean(base_txn)
        assert result.record["account_id"] == "ACC-12345"
        assert any("whitespace_trimmed" in c for c in result.changes)

    def test_internal_whitespace_collapsed(self, cleaner, base_txn):
        base_txn["merchant_name"] = "  Amazon   Online   Store  "
        result = cleaner.clean(base_txn)
        assert result.record["merchant_name"] == "Amazon Online Store"

    def test_no_change_on_clean_string(self, cleaner, base_txn):
        result = cleaner.clean(base_txn)
        assert not any("whitespace_trimmed" in c for c in result.changes)


class TestEncodingFixes:
    """Tests for Unicode normalization and control character removal."""

    def test_null_bytes_removed(self, cleaner, base_txn):
        base_txn["merchant_name"] = "Ama\x00zon"
        result = cleaner.clean(base_txn)
        assert "\x00" not in result.record["merchant_name"]

    def test_control_characters_removed(self, cleaner, base_txn):
        base_txn["merchant_name"] = "Amazon\x01Store"
        result = cleaner.clean(base_txn)
        assert "\x01" not in result.record["merchant_name"]

    def test_unicode_normalization(self, cleaner, base_txn):
        # é can be composed (NFC) or decomposed (NFD) — should normalize to NFC
        base_txn["merchant_name"] = "Cafe\u0301"  # NFD form
        result = cleaner.clean(base_txn)
        assert result.record["merchant_name"] == "Café"  # NFC form


class TestStringStandardization:
    """Tests for case normalization of specific fields."""

    def test_transaction_type_lowercased(self, cleaner, base_txn):
        base_txn["transaction_type"] = "PURCHASE"
        result = cleaner.clean(base_txn)
        assert result.record["transaction_type"] == "purchase"

    def test_channel_lowercased(self, cleaner, base_txn):
        base_txn["channel"] = "ONLINE"
        result = cleaner.clean(base_txn)
        assert result.record["channel"] == "online"

    def test_currency_uppercased(self, cleaner, base_txn):
        base_txn["transaction_currency"] = "usd"
        result = cleaner.clean(base_txn)
        assert result.record["transaction_currency"] == "USD"

    def test_country_uppercased(self, cleaner, base_txn):
        base_txn["geo_country"] = "us"
        result = cleaner.clean(base_txn)
        assert result.record["geo_country"] == "US"

    def test_card_type_lowercased(self, cleaner, base_txn):
        base_txn["card_type"] = "CREDIT"
        result = cleaner.clean(base_txn)
        assert result.record["card_type"] == "credit"

    def test_device_type_lowercased(self, cleaner, base_txn):
        base_txn["device_type"] = "MOBILE"
        result = cleaner.clean(base_txn)
        assert result.record["device_type"] == "mobile"


class TestDateParsing:
    """Tests for date parsing and normalization to ISO 8601 UTC."""

    def test_iso8601_z_format(self, cleaner, base_txn):
        base_txn["transaction_timestamp"] = "2026-06-15T10:30:00Z"
        result = cleaner.clean(base_txn)
        assert result.record["transaction_timestamp"] == "2026-06-15T10:30:00Z"

    def test_iso8601_with_offset(self, cleaner, base_txn):
        base_txn["transaction_timestamp"] = "2026-06-15T10:30:00+05:30"
        result = cleaner.clean(base_txn)
        assert result.record["transaction_timestamp"] == "2026-06-15T05:00:00Z"

    def test_iso8601_with_microseconds(self, cleaner, base_txn):
        base_txn["transaction_timestamp"] = "2026-06-15T10:30:00.123456Z"
        result = cleaner.clean(base_txn)
        assert result.record["transaction_timestamp"] == "2026-06-15T10:30:00Z"

    def test_space_separated_datetime(self, cleaner, base_txn):
        base_txn["transaction_timestamp"] = "2026-06-15 10:30:00"
        result = cleaner.clean(base_txn)
        assert result.record["transaction_timestamp"] == "2026-06-15T10:30:00Z"

    def test_date_only_format(self, cleaner, base_txn):
        base_txn["transaction_timestamp"] = "2026-06-15"
        result = cleaner.clean(base_txn)
        assert result.record["transaction_timestamp"] == "2026-06-15T00:00:00Z"

    def test_us_date_format(self, cleaner, base_txn):
        base_txn["transaction_timestamp"] = "06/15/2026 10:30:00"
        result = cleaner.clean(base_txn)
        assert "2026" in result.record["transaction_timestamp"]

    def test_epoch_seconds(self, cleaner, base_txn):
        base_txn["transaction_timestamp"] = "1781720400"  # Some future timestamp
        result = cleaner.clean(base_txn)
        assert "T" in result.record["transaction_timestamp"]

    def test_epoch_milliseconds(self, cleaner, base_txn):
        base_txn["transaction_timestamp"] = "1781720400000"
        result = cleaner.clean(base_txn)
        assert "T" in result.record["transaction_timestamp"]

    def test_datetime_object_input(self, cleaner, base_txn):
        base_txn["transaction_timestamp"] = datetime(2026, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = cleaner.clean(base_txn)
        assert result.record["transaction_timestamp"] == "2026-06-15T10:30:00Z"

    def test_none_timestamp_unchanged(self, cleaner, base_txn):
        base_txn["transaction_timestamp"] = None
        result = cleaner.clean(base_txn)
        assert result.record["transaction_timestamp"] is None


class TestMissingValueImputation:
    """Tests for default value imputation."""

    def test_missing_currency_imputed(self, cleaner, base_txn):
        del base_txn["transaction_currency"]
        result = cleaner.clean(base_txn)
        assert result.record["transaction_currency"] == "USD"
        assert any("imputed:transaction_currency" in c for c in result.changes)

    def test_missing_is_international_imputed(self, cleaner, base_txn):
        del base_txn["is_international"]
        result = cleaner.clean(base_txn)
        assert result.record["is_international"] is False

    def test_empty_string_treated_as_missing(self, cleaner, base_txn):
        base_txn["transaction_currency"] = ""
        result = cleaner.clean(base_txn)
        assert result.record["transaction_currency"] == "USD"

    def test_missing_device_type_imputed(self, cleaner, base_txn):
        del base_txn["device_type"]
        result = cleaner.clean(base_txn)
        assert result.record["device_type"] == "unknown"

    def test_present_value_not_overwritten(self, cleaner, base_txn):
        base_txn["transaction_currency"] = "EUR"
        result = cleaner.clean(base_txn)
        assert result.record["transaction_currency"] == "EUR"


class TestNumericCleaning:
    """Tests for numeric field cleaning."""

    def test_amount_with_currency_symbol(self, cleaner, base_txn):
        base_txn["transaction_amount"] = "$1,234.56"
        result = cleaner.clean(base_txn)
        assert result.record["transaction_amount"] == 1234.56

    def test_amount_european_format(self, cleaner, base_txn):
        base_txn["transaction_amount"] = "1.234,56"
        result = cleaner.clean(base_txn)
        assert result.record["transaction_amount"] == 1234.56

    def test_amount_comma_as_decimal(self, cleaner, base_txn):
        base_txn["transaction_amount"] = "123,45"
        result = cleaner.clean(base_txn)
        assert result.record["transaction_amount"] == 123.45

    def test_amount_euro_symbol(self, cleaner, base_txn):
        base_txn["transaction_amount"] = "€500.00"
        result = cleaner.clean(base_txn)
        assert result.record["transaction_amount"] == 500.00

    def test_amount_float_rounding(self, cleaner, base_txn):
        base_txn["transaction_amount"] = 123.456789
        result = cleaner.clean(base_txn)
        assert result.record["transaction_amount"] == 123.4568

    def test_invalid_latitude_cleared(self, cleaner, base_txn):
        base_txn["geo_latitude"] = 200.0
        result = cleaner.clean(base_txn)
        assert result.record["geo_latitude"] is None

    def test_invalid_longitude_cleared(self, cleaner, base_txn):
        base_txn["geo_longitude"] = -250.0
        result = cleaner.clean(base_txn)
        assert result.record["geo_longitude"] is None

    def test_unparseable_latitude_cleared(self, cleaner, base_txn):
        base_txn["geo_latitude"] = "not_a_number"
        result = cleaner.clean(base_txn)
        assert result.record["geo_latitude"] is None


class TestPIIMasking:
    """Tests for PII field masking."""

    def test_email_masked(self, cleaner, base_txn):
        base_txn["email"] = "john.doe@example.com"
        masked = cleaner.mask_pii(base_txn)
        assert masked["email"] == "j***@example.com"
        assert base_txn["email"] == "john.doe@example.com"  # Original unchanged

    def test_phone_masked(self, cleaner, base_txn):
        base_txn["phone"] = "1234567890"
        masked = cleaner.mask_pii(base_txn)
        assert masked["phone"] == "******7890"

    def test_card_number_masked(self, cleaner, base_txn):
        base_txn["card_number"] = "4111111111111111"
        masked = cleaner.mask_pii(base_txn)
        assert masked["card_number"] == "************1111"

    def test_ip_address_masked(self, cleaner, base_txn):
        # ip_address is in PII fields — should be masked
        base_txn["ip_address"] = "192.168.1.100"
        masked = cleaner.mask_pii(base_txn)
        assert "***" in masked["ip_address"]
        assert masked["ip_address"] != "192.168.1.100"

    def test_device_id_masked(self, cleaner, base_txn):
        base_txn["device_id"] = "device-abc-123456"
        masked = cleaner.mask_pii(base_txn)
        assert "***" in masked["device_id"]

    def test_none_field_masked(self, cleaner, base_txn):
        base_txn["email"] = None
        masked = cleaner.mask_pii(base_txn)
        assert masked["email"] == "***"

    def test_non_pii_fields_unchanged(self, cleaner, base_txn):
        masked = cleaner.mask_pii(base_txn)
        assert masked["transaction_amount"] == base_txn["transaction_amount"]
        assert masked["transaction_type"] == base_txn["transaction_type"]

    def test_masking_metrics_incremented(self, cleaner, base_txn):
        cleaner.mask_pii(base_txn)
        assert cleaner.metrics.total_fields_masked > 0


class TestSanitization:
    """Tests for HTML/script tag removal."""

    def test_html_tags_stripped(self, cleaner, base_txn):
        base_txn["merchant_name"] = "<script>alert('xss')</script>Amazon"
        result = cleaner.clean(base_txn)
        assert "<script>" not in result.record["merchant_name"]
        assert "Amazon" in result.record["merchant_name"]

    def test_html_link_stripped(self, cleaner, base_txn):
        base_txn["merchant_name"] = '<a href="http://evil.com">Store</a>'
        result = cleaner.clean(base_txn)
        assert "<a" not in result.record["merchant_name"]
        assert "Store" in result.record["merchant_name"]


class TestBatchCleaning:
    """Tests for batch processing."""

    def test_batch_clean(self, cleaner, base_txn):
        txns = []
        for i in range(5):
            t = copy.deepcopy(base_txn)
            t["external_transaction_id"] = f"TXN-BATCH-{i}"
            txns.append(t)
        results = cleaner.clean_batch(txns)
        assert len(results) == 5
        assert all(isinstance(r, CleaningResult) for r in results)
        assert all(r.is_duplicate is False for r in results)

    def test_batch_with_duplicates(self, cleaner, base_txn):
        txns = [copy.deepcopy(base_txn) for _ in range(3)]
        results = cleaner.clean_batch(txns)
        assert results[0].is_duplicate is False
        assert results[1].is_duplicate is True
        assert results[2].is_duplicate is True


class TestCleaningMetrics:
    """Tests for metrics tracking."""

    def test_metrics_accumulate(self, cleaner, base_txn):
        for i in range(5):
            t = copy.deepcopy(base_txn)
            t["external_transaction_id"] = f"TXN-M-{i}"
            cleaner.clean(t)
        assert cleaner.metrics.total_records_processed == 5

    def test_metrics_reset(self, cleaner, base_txn):
        cleaner.clean(base_txn)
        cleaner.metrics.reset()
        assert cleaner.metrics.total_records_processed == 0

    def test_metrics_to_dict(self, cleaner, base_txn):
        cleaner.clean(base_txn)
        d = cleaner.metrics.to_dict()
        assert "total_records_processed" in d
        assert isinstance(d["total_records_processed"], int)


# ══════════════════════════════════════════════════════════════════════════════
# DATA NORMALIZER TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestCurrencyConversion:
    """Tests for currency conversion to USD."""

    def test_usd_no_conversion(self, normalizer, base_txn):
        result = normalizer.normalize(base_txn)
        assert result.amount_usd == 125.50

    def test_eur_to_usd(self, normalizer, base_txn):
        base_txn["transaction_currency"] = "EUR"
        base_txn["transaction_amount"] = 100.00
        result = normalizer.normalize(base_txn)
        assert result.amount_usd == 108.40
        assert result.original_currency == "EUR"

    def test_gbp_to_usd(self, normalizer, base_txn):
        base_txn["transaction_currency"] = "GBP"
        base_txn["transaction_amount"] = 100.00
        result = normalizer.normalize(base_txn)
        assert result.amount_usd == 126.50

    def test_jpy_to_usd(self, normalizer, base_txn):
        base_txn["transaction_currency"] = "JPY"
        base_txn["transaction_amount"] = 10000
        result = normalizer.normalize(base_txn)
        assert result.amount_usd == 64.0

    def test_unknown_currency_returns_none(self, normalizer, base_txn):
        base_txn["transaction_currency"] = "XYZ"
        result = normalizer.normalize(base_txn)
        assert result.amount_usd is None

    def test_convert_currency_api(self, normalizer):
        result = normalizer.convert_currency(100.0, "EUR", "USD")
        assert result == 108.40

    def test_convert_same_currency(self, normalizer):
        result = normalizer.convert_currency(100.0, "USD", "USD")
        assert result == 100.0

    def test_conversion_accuracy_4_decimals(self, normalizer, base_txn):
        base_txn["transaction_currency"] = "INR"
        base_txn["transaction_amount"] = 8333.33
        result = normalizer.normalize(base_txn)
        assert result.amount_usd is not None
        # Should be rounded to 4 decimal places
        str_amount = str(result.amount_usd)
        if "." in str_amount:
            decimals = len(str_amount.split(".")[1])
            assert decimals <= 4

    def test_amount_usd_field_added(self, normalizer, base_txn):
        base_txn["transaction_currency"] = "EUR"
        result = normalizer.normalize(base_txn)
        assert "transaction_amount_usd" in result.record


class TestCurrencyCodeNormalization:
    """Tests for currency code standardization."""

    def test_lowercase_currency_uppercased(self, normalizer, base_txn):
        base_txn["transaction_currency"] = "eur"
        result = normalizer.normalize(base_txn)
        assert result.record["transaction_currency"] == "EUR"

    def test_alias_dollar(self, normalizer, base_txn):
        base_txn["transaction_currency"] = "dollar"
        result = normalizer.normalize(base_txn)
        assert result.record["transaction_currency"] == "USD"

    def test_alias_euro(self, normalizer, base_txn):
        base_txn["transaction_currency"] = "euro"
        result = normalizer.normalize(base_txn)
        assert result.record["transaction_currency"] == "EUR"

    def test_alias_pound(self, normalizer, base_txn):
        base_txn["transaction_currency"] = "pound"
        result = normalizer.normalize(base_txn)
        assert result.record["transaction_currency"] == "GBP"


class TestCountryNormalization:
    """Tests for ISO 3166-1 alpha-2 country code normalization."""

    def test_two_letter_code_unchanged(self, normalizer, base_txn):
        base_txn["geo_country"] = "US"
        result = normalizer.normalize(base_txn)
        assert result.record["geo_country"] == "US"

    def test_lowercase_two_letter_uppercased(self, normalizer, base_txn):
        base_txn["geo_country"] = "us"
        result = normalizer.normalize(base_txn)
        assert result.record["geo_country"] == "US"

    def test_full_name_to_code(self, normalizer, base_txn):
        base_txn["geo_country"] = "united states"
        result = normalizer.normalize(base_txn)
        assert result.record["geo_country"] == "US"

    def test_full_name_case_insensitive(self, normalizer, base_txn):
        base_txn["geo_country"] = "United Kingdom"
        result = normalizer.normalize(base_txn)
        assert result.record["geo_country"] == "GB"

    def test_alpha3_to_alpha2(self, normalizer, base_txn):
        base_txn["geo_country"] = "CAN"
        result = normalizer.normalize(base_txn)
        assert result.record["geo_country"] == "CA"

    def test_common_alias(self, normalizer, base_txn):
        base_txn["geo_country"] = "uk"
        result = normalizer.normalize(base_txn)
        assert result.record["geo_country"] == "GB"

    def test_none_country_unchanged(self, normalizer, base_txn):
        base_txn["geo_country"] = None
        result = normalizer.normalize(base_txn)
        assert result.record["geo_country"] is None


class TestMCCMapping:
    """Tests for Merchant Category Code to category mapping."""

    def test_grocery_mcc(self, normalizer, base_txn):
        base_txn["merchant_category_code"] = "5411"
        result = normalizer.normalize(base_txn)
        assert result.mcc_category == "grocery_stores"
        assert result.record["mcc_category"] == "grocery_stores"

    def test_restaurant_mcc(self, normalizer, base_txn):
        base_txn["merchant_category_code"] = "5812"
        result = normalizer.normalize(base_txn)
        assert result.mcc_category == "restaurants"

    def test_gambling_mcc(self, normalizer, base_txn):
        base_txn["merchant_category_code"] = "7995"
        result = normalizer.normalize(base_txn)
        assert result.mcc_category == "gambling_betting"

    def test_unknown_mcc_returns_none(self, normalizer, base_txn):
        base_txn["merchant_category_code"] = "9999"
        result = normalizer.normalize(base_txn)
        assert result.mcc_category is None

    def test_none_mcc(self, normalizer, base_txn):
        base_txn["merchant_category_code"] = None
        result = normalizer.normalize(base_txn)
        assert result.mcc_category is None

    def test_airline_prefix_match(self, normalizer, base_txn):
        base_txn["merchant_category_code"] = "3012"
        result = normalizer.normalize(base_txn)
        assert result.mcc_category == "airlines"

    def test_get_mcc_category_api(self, normalizer):
        assert normalizer.get_mcc_category("5411") == "grocery_stores"
        assert normalizer.get_mcc_category("9999") is None


class TestTransactionTypeNormalization:
    """Tests for transaction type standardization."""

    def test_standard_type_unchanged(self, normalizer, base_txn):
        base_txn["transaction_type"] = "purchase"
        result = normalizer.normalize(base_txn)
        assert result.record["transaction_type"] == "purchase"

    def test_alias_buy(self, normalizer, base_txn):
        base_txn["transaction_type"] = "buy"
        result = normalizer.normalize(base_txn)
        assert result.record["transaction_type"] == "purchase"

    def test_alias_wire(self, normalizer, base_txn):
        base_txn["transaction_type"] = "wire"
        result = normalizer.normalize(base_txn)
        assert result.record["transaction_type"] == "transfer"

    def test_alias_chargeback(self, normalizer, base_txn):
        base_txn["transaction_type"] = "chargeback"
        result = normalizer.normalize(base_txn)
        assert result.record["transaction_type"] == "refund"

    def test_alias_cash_out(self, normalizer, base_txn):
        base_txn["transaction_type"] = "cash_out"
        result = normalizer.normalize(base_txn)
        assert result.record["transaction_type"] == "withdrawal"

    def test_none_type_unchanged(self, normalizer, base_txn):
        base_txn["transaction_type"] = None
        result = normalizer.normalize(base_txn)
        assert result.record["transaction_type"] is None


class TestChannelNormalization:
    """Tests for channel standardization."""

    def test_standard_channel_unchanged(self, normalizer, base_txn):
        base_txn["channel"] = "online"
        result = normalizer.normalize(base_txn)
        assert result.record["channel"] == "online"

    def test_alias_web(self, normalizer, base_txn):
        base_txn["channel"] = "web"
        result = normalizer.normalize(base_txn)
        assert result.record["channel"] == "online"

    def test_alias_ecommerce(self, normalizer, base_txn):
        base_txn["channel"] = "ecommerce"
        result = normalizer.normalize(base_txn)
        assert result.record["channel"] == "online"

    def test_alias_in_store(self, normalizer, base_txn):
        base_txn["channel"] = "in_store"
        result = normalizer.normalize(base_txn)
        assert result.record["channel"] == "pos"

    def test_alias_contactless(self, normalizer, base_txn):
        base_txn["channel"] = "contactless"
        result = normalizer.normalize(base_txn)
        assert result.record["channel"] == "pos"

    def test_alias_app(self, normalizer, base_txn):
        base_txn["channel"] = "app"
        result = normalizer.normalize(base_txn)
        assert result.record["channel"] == "mobile"


class TestCardTypeNormalization:
    """Tests for card type standardization."""

    def test_alias_visa(self, normalizer, base_txn):
        base_txn["card_type"] = "visa"
        result = normalizer.normalize(base_txn)
        assert result.record["card_type"] == "credit"

    def test_alias_mastercard(self, normalizer, base_txn):
        base_txn["card_type"] = "mastercard"
        result = normalizer.normalize(base_txn)
        assert result.record["card_type"] == "credit"

    def test_alias_debit_card(self, normalizer, base_txn):
        base_txn["card_type"] = "debit_card"
        result = normalizer.normalize(base_txn)
        assert result.record["card_type"] == "debit"

    def test_alias_gift_card(self, normalizer, base_txn):
        base_txn["card_type"] = "gift_card"
        result = normalizer.normalize(base_txn)
        assert result.record["card_type"] == "prepaid"

    def test_none_card_type(self, normalizer, base_txn):
        base_txn["card_type"] = None
        result = normalizer.normalize(base_txn)
        assert result.record["card_type"] is None


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION: CLEANER + NORMALIZER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


class TestCleanerNormalizerPipeline:
    """Tests for the full cleaning → normalization pipeline."""

    def test_full_pipeline(self, cleaner, normalizer, base_txn):
        # Dirty up the record
        base_txn["transaction_type"] = "  BUY  "
        base_txn["channel"] = "E-COMMERCE"
        base_txn["transaction_currency"] = "eur"
        base_txn["geo_country"] = "united kingdom"
        base_txn["merchant_name"] = "  Test  Store  "

        cleaned = cleaner.clean(base_txn)
        assert cleaned.is_duplicate is False

        normalized = normalizer.normalize(cleaned.record)
        assert normalized.record["transaction_type"] == "purchase"
        assert normalized.record["channel"] == "online"
        assert normalized.record["geo_country"] == "GB"
        assert normalized.amount_usd is not None

    def test_pipeline_preserves_clean_data(self, cleaner, normalizer, base_txn):
        cleaned = cleaner.clean(base_txn)
        normalized = normalizer.normalize(cleaned.record)
        assert normalized.record["transaction_amount"] == 125.50
        assert normalized.record["account_id"] == "ACC-12345"


class TestNormalizerBatch:
    """Tests for batch normalization."""

    def test_batch_normalize(self, normalizer, base_txn):
        txns = []
        for curr in ["USD", "EUR", "GBP"]:
            t = copy.deepcopy(base_txn)
            t["transaction_currency"] = curr
            txns.append(t)
        results = normalizer.normalize_batch(txns)
        assert len(results) == 3
        assert all(isinstance(r, NormalizationResult) for r in results)
        assert all(r.amount_usd is not None for r in results)


class TestNormalizerMetrics:
    """Tests for normalizer metrics."""

    def test_metrics_accumulate(self, normalizer, base_txn):
        for i in range(3):
            normalizer.normalize(base_txn)
        assert normalizer.metrics.total_records_processed == 3

    def test_conversion_metric(self, normalizer, base_txn):
        base_txn["transaction_currency"] = "EUR"
        normalizer.normalize(base_txn)
        assert normalizer.metrics.amounts_converted >= 1

    def test_metrics_reset(self, normalizer, base_txn):
        normalizer.normalize(base_txn)
        normalizer.metrics.reset()
        assert normalizer.metrics.total_records_processed == 0


class TestNormalizerUtilities:
    """Tests for normalizer utility methods."""

    def test_supported_currencies(self, normalizer):
        currencies = normalizer.supported_currencies
        assert "USD" in currencies
        assert "EUR" in currencies

    def test_get_exchange_rate(self, normalizer):
        rate = normalizer.get_exchange_rate("EUR")
        assert rate == 1.084

    def test_get_exchange_rate_unknown(self, normalizer):
        assert normalizer.get_exchange_rate("XYZ") is None

    def test_pii_field_names(self, normalizer):
        pii = normalizer.pii_field_names
        assert "card_number" in pii
        assert "email" in pii

    def test_force_reload(self, normalizer):
        normalizer.force_reload()
        assert normalizer.metrics.total_records_processed == 0  # reload doesn't reset metrics


class TestNormalizerHotReload:
    """Tests for hot-reload of mappings."""

    def test_reload_detects_changes(self, normalizer, mappings_path):
        with open(mappings_path, "r") as f:
            data = yaml.safe_load(f)
        data["exchange_rates"]["rates"]["EUR"] = 1.1000
        with open(mappings_path, "w") as f:
            yaml.dump(data, f)
        normalizer.force_reload()
        assert normalizer.get_exchange_rate("EUR") == 1.1000


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestPerformance:
    """Performance benchmarks for cleaning and normalization."""

    def test_cleaner_under_5ms(self, cleaner, base_txn):
        # Warm up
        for i in range(5):
            t = copy.deepcopy(base_txn)
            t["external_transaction_id"] = f"TXN-WARMUP-{i}"
            cleaner.clean(t)

        times = []
        for i in range(100):
            t = copy.deepcopy(base_txn)
            t["external_transaction_id"] = f"TXN-PERF-{i}"
            start = time.perf_counter()
            cleaner.clean(t)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg = sum(times) / len(times)
        assert avg < 5.0, f"Cleaner avg {avg:.2f}ms exceeds 5ms"

    def test_normalizer_under_5ms(self, normalizer, base_txn):
        # Warm up
        normalizer.normalize(base_txn)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            normalizer.normalize(base_txn)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg = sum(times) / len(times)
        assert avg < 5.0, f"Normalizer avg {avg:.2f}ms exceeds 5ms"

    def test_full_pipeline_under_10ms(self, cleaner, normalizer, base_txn):
        # Warm up
        r = cleaner.clean(base_txn)
        normalizer.normalize(r.record)
        cleaner.reset_dedup_cache()

        times = []
        for i in range(100):
            t = copy.deepcopy(base_txn)
            t["external_transaction_id"] = f"TXN-FULL-{i}"
            start = time.perf_counter()
            cleaned = cleaner.clean(t)
            normalizer.normalize(cleaned.record)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg = sum(times) / len(times)
        assert avg < 10.0, f"Full pipeline avg {avg:.2f}ms exceeds 10ms"


# ══════════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_record(self, cleaner):
        result = cleaner.clean({})
        assert result.is_duplicate is False

    def test_all_none_values(self, cleaner):
        record = {
            "external_transaction_id": None,
            "account_id": None,
            "transaction_amount": None,
        }
        result = cleaner.clean(record)
        assert isinstance(result, CleaningResult)

    def test_normalizer_empty_record(self, normalizer):
        result = normalizer.normalize({})
        assert isinstance(result, NormalizationResult)

    def test_normalizer_missing_file(self):
        with pytest.raises(FileNotFoundError):
            DataNormalizer(mappings_path="/nonexistent/path.yaml")

    def test_concurrent_cleaning(self, cleaner, base_txn):
        import concurrent.futures

        def clean_txn(i):
            t = copy.deepcopy(base_txn)
            t["external_transaction_id"] = f"TXN-CC-{i}"
            return cleaner.clean(t)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(clean_txn, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        assert all(isinstance(r, CleaningResult) for r in results)

    def test_concurrent_normalization(self, normalizer, base_txn):
        import concurrent.futures

        def normalize_txn(_):
            return normalizer.normalize(copy.deepcopy(base_txn))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(normalize_txn, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        assert all(isinstance(r, NormalizationResult) for r in results)

    def test_very_long_string_handled(self, cleaner, base_txn):
        base_txn["merchant_name"] = "A" * 10000
        result = cleaner.clean(base_txn)
        assert len(result.record["merchant_name"]) == 10000

    def test_special_unicode_characters(self, cleaner, base_txn):
        base_txn["merchant_name"] = "Ünïcödé Störé 日本語"
        result = cleaner.clean(base_txn)
        assert "Ünïcödé" in result.record["merchant_name"]
