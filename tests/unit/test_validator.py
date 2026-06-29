"""Comprehensive unit tests for the schema validation engine and quarantine handler.

Covers: required fields, field types, range validation, enum validation,
pattern matching, cross-field rules, custom business rules, batch validation,
metrics, quarantine lifecycle, and re-processing.
"""

from __future__ import annotations

import json
import copy
import time
from pathlib import Path

import pytest

from src.validation.schema_validator import (
    SchemaValidator,
    ValidationError,
    ValidationMetrics,
    ValidationResult,
    ValidationSeverity,
)
from src.validation.quarantine_handler import (
    QuarantineHandler,
    QuarantinedRecord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
VALIDATION_RULES_PATH = CONFIG_DIR / "validation_rules.yaml"


@pytest.fixture
def validator():
    """Create a SchemaValidator with the project validation rules."""
    return SchemaValidator(config_path=VALIDATION_RULES_PATH)


@pytest.fixture
def quarantine_handler():
    """Create a QuarantineHandler with test defaults."""
    return QuarantineHandler(max_retry_attempts=3, retention_days=30)


@pytest.fixture
def valid_transaction():
    """A fully valid transaction record."""
    return {
        "external_transaction_id": "TXN-VALID-001",
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
        "geo_country": "USA",
        "geo_city": "New York",
        "is_international": False,
        "transaction_timestamp": "2026-06-15T10:30:00Z",
    }


@pytest.fixture
def invalid_transactions():
    """Load invalid transactions from fixture file."""
    path = FIXTURES_DIR / "invalid_transactions.json"
    with open(path, "r") as f:
        return json.load(f)


# ===========================================================================
# SchemaValidator — Required Fields
# ===========================================================================


class TestRequiredFields:
    """Tests for required field validation."""

    def test_valid_record_passes(self, validator, valid_transaction):
        result = validator.validate(valid_transaction)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_account_id(self, validator, valid_transaction):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.field == "account_id" and e.rule == "required_field" for e in result.errors)

    def test_missing_customer_id(self, validator, valid_transaction):
        del valid_transaction["customer_id"]
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.field == "customer_id" for e in result.errors)

    def test_missing_transaction_amount(self, validator, valid_transaction):
        del valid_transaction["transaction_amount"]
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.field == "transaction_amount" for e in result.errors)

    def test_missing_transaction_type(self, validator, valid_transaction):
        del valid_transaction["transaction_type"]
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.field == "transaction_type" for e in result.errors)

    def test_missing_channel(self, validator, valid_transaction):
        del valid_transaction["channel"]
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.field == "channel" for e in result.errors)

    def test_missing_transaction_timestamp(self, validator, valid_transaction):
        del valid_transaction["transaction_timestamp"]
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.field == "transaction_timestamp" for e in result.errors)

    def test_missing_external_transaction_id(self, validator, valid_transaction):
        del valid_transaction["external_transaction_id"]
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.field == "external_transaction_id" for e in result.errors)

    def test_empty_string_required_field(self, validator, valid_transaction):
        valid_transaction["account_id"] = ""
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(
            e.field == "account_id" and e.rule == "required_field_empty"
            for e in result.errors
        )

    def test_whitespace_only_required_field(self, validator, valid_transaction):
        valid_transaction["account_id"] = "   "
        result = validator.validate(valid_transaction)
        assert result.is_valid is False

    def test_none_required_field(self, validator, valid_transaction):
        valid_transaction["account_id"] = None
        result = validator.validate(valid_transaction)
        assert result.is_valid is False

    def test_multiple_missing_required_fields(self, validator):
        record = {
            "external_transaction_id": "TXN-001",
            "transaction_amount": 100.00,
        }
        result = validator.validate(record)
        assert result.is_valid is False
        missing_fields = {e.field for e in result.errors if "required" in e.rule}
        assert "account_id" in missing_fields
        assert "customer_id" in missing_fields
        assert "transaction_type" in missing_fields
        assert "channel" in missing_fields
        assert "transaction_timestamp" in missing_fields


# ===========================================================================
# SchemaValidator — Field Type Validation
# ===========================================================================


class TestFieldTypeValidation:
    """Tests for field type checks."""

    def test_amount_as_string(self, validator, valid_transaction):
        valid_transaction["transaction_amount"] = "one hundred"
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.field == "transaction_amount" and e.rule == "type_check" for e in result.errors)

    def test_amount_as_boolean(self, validator, valid_transaction):
        valid_transaction["transaction_amount"] = True
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.field == "transaction_amount" and e.rule == "type_check" for e in result.errors)

    def test_boolean_field_with_string(self, validator, valid_transaction):
        valid_transaction["is_international"] = "yes"
        result = validator.validate(valid_transaction)
        assert any(e.field == "is_international" and e.rule == "type_check" for e in result.errors)

    def test_integer_amount_accepted(self, validator, valid_transaction):
        valid_transaction["transaction_amount"] = 100
        result = validator.validate(valid_transaction)
        assert result.is_valid is True

    def test_float_amount_accepted(self, validator, valid_transaction):
        valid_transaction["transaction_amount"] = 99.99
        result = validator.validate(valid_transaction)
        assert result.is_valid is True


# ===========================================================================
# SchemaValidator — Range Validation
# ===========================================================================


class TestRangeValidation:
    """Tests for numeric range checks."""

    def test_negative_amount(self, validator, valid_transaction):
        valid_transaction["transaction_amount"] = -50.00
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.rule == "min_value" for e in result.errors)

    def test_zero_amount(self, validator, valid_transaction):
        valid_transaction["transaction_amount"] = 0
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.field == "transaction_amount" and e.rule == "min_value" for e in result.errors)

    def test_amount_just_above_minimum(self, validator, valid_transaction):
        valid_transaction["transaction_amount"] = 0.01
        result = validator.validate(valid_transaction)
        assert result.is_valid is True

    def test_amount_exceeds_maximum(self, validator, valid_transaction):
        valid_transaction["transaction_amount"] = 9999999999.99
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.rule == "max_value" for e in result.errors)

    def test_latitude_out_of_range(self, validator, valid_transaction):
        valid_transaction["geo_latitude"] = 95.0
        result = validator.validate(valid_transaction)
        assert any(e.field == "geo_latitude" and e.rule == "max_value" for e in result.errors)

    def test_longitude_out_of_range(self, validator, valid_transaction):
        valid_transaction["geo_longitude"] = -200.0
        result = validator.validate(valid_transaction)
        assert any(e.field == "geo_longitude" and e.rule == "min_value" for e in result.errors)

    def test_valid_geo_boundaries(self, validator, valid_transaction):
        valid_transaction["geo_latitude"] = 90.0
        valid_transaction["geo_longitude"] = 180.0
        result = validator.validate(valid_transaction)
        geo_errors = [
            e for e in result.errors
            if e.field in ("geo_latitude", "geo_longitude")
            and e.rule in ("min_value", "max_value")
        ]
        assert len(geo_errors) == 0

    def test_negative_geo_boundaries(self, validator, valid_transaction):
        valid_transaction["geo_latitude"] = -90.0
        valid_transaction["geo_longitude"] = -180.0
        result = validator.validate(valid_transaction)
        geo_errors = [
            e for e in result.errors
            if e.field in ("geo_latitude", "geo_longitude")
            and e.rule in ("min_value", "max_value")
        ]
        assert len(geo_errors) == 0


# ===========================================================================
# SchemaValidator — Enum / Allowed Values
# ===========================================================================


class TestEnumValidation:
    """Tests for enum/allowed value checks."""

    def test_invalid_transaction_type(self, validator, valid_transaction):
        valid_transaction["transaction_type"] = "loan"
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.rule == "allowed_values" for e in result.errors)

    def test_invalid_channel(self, validator, valid_transaction):
        valid_transaction["channel"] = "mail_order"
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.field == "channel" and e.rule == "allowed_values" for e in result.errors)

    def test_invalid_currency(self, validator, valid_transaction):
        valid_transaction["transaction_currency"] = "XYZ"
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(
            e.field == "transaction_currency" and e.rule == "allowed_values"
            for e in result.errors
        )

    def test_invalid_card_type(self, validator, valid_transaction):
        valid_transaction["card_type"] = "virtual"
        result = validator.validate(valid_transaction)
        assert any(e.field == "card_type" and e.rule == "allowed_values" for e in result.errors)

    def test_valid_transaction_types(self, validator, valid_transaction):
        for txn_type in ("purchase", "withdrawal", "transfer", "refund"):
            valid_transaction["transaction_type"] = txn_type
            result = validator.validate(valid_transaction)
            type_errors = [
                e for e in result.errors
                if e.field == "transaction_type" and e.rule == "allowed_values"
            ]
            assert len(type_errors) == 0, f"Failed for type: {txn_type}"

    def test_valid_channels(self, validator, valid_transaction):
        for channel in ("online", "pos", "atm", "mobile"):
            txn = copy.deepcopy(valid_transaction)
            txn["channel"] = channel
            if channel in ("pos", "atm"):
                txn["merchant_id"] = "MERCH-001"
            if channel == "atm":
                txn["transaction_type"] = "withdrawal"
            result = validator.validate(txn)
            channel_errors = [
                e for e in result.errors
                if e.field == "channel" and e.rule == "allowed_values"
            ]
            assert len(channel_errors) == 0, f"Failed for channel: {channel}"

    def test_valid_currencies(self, validator, valid_transaction):
        for currency in ("USD", "EUR", "GBP", "CAD", "AUD", "JPY"):
            valid_transaction["transaction_currency"] = currency
            result = validator.validate(valid_transaction)
            curr_errors = [
                e for e in result.errors
                if e.field == "transaction_currency" and e.rule == "allowed_values"
            ]
            assert len(curr_errors) == 0, f"Failed for currency: {currency}"


# ===========================================================================
# SchemaValidator — Pattern Matching
# ===========================================================================


class TestPatternValidation:
    """Tests for regex pattern checks."""

    def test_invalid_card_last_four_letters(self, validator, valid_transaction):
        valid_transaction["card_last_four"] = "12A3"
        result = validator.validate(valid_transaction)
        assert any(e.field == "card_last_four" and e.rule == "pattern" for e in result.errors)

    def test_invalid_card_last_four_too_short(self, validator, valid_transaction):
        valid_transaction["card_last_four"] = "12"
        result = validator.validate(valid_transaction)
        assert any(e.field == "card_last_four" and e.rule == "pattern" for e in result.errors)

    def test_invalid_card_last_four_too_long(self, validator, valid_transaction):
        valid_transaction["card_last_four"] = "12345"
        result = validator.validate(valid_transaction)
        assert any(e.field == "card_last_four" and e.rule == "pattern" for e in result.errors)

    def test_valid_card_last_four(self, validator, valid_transaction):
        valid_transaction["card_last_four"] = "9876"
        result = validator.validate(valid_transaction)
        pattern_errors = [
            e for e in result.errors
            if e.field == "card_last_four" and e.rule == "pattern"
        ]
        assert len(pattern_errors) == 0

    def test_nullable_card_last_four(self, validator, valid_transaction):
        del valid_transaction["card_last_four"]
        del valid_transaction["card_type"]
        result = validator.validate(valid_transaction)
        card_errors = [e for e in result.errors if e.field == "card_last_four"]
        assert len(card_errors) == 0


# ===========================================================================
# SchemaValidator — IP Address Validation
# ===========================================================================


class TestIPAddressValidation:
    """Tests for IP address format validation."""

    def test_invalid_ip_address(self, validator, valid_transaction):
        valid_transaction["ip_address"] = "not-an-ip"
        result = validator.validate(valid_transaction)
        assert any(e.field == "ip_address" and e.rule == "ip_address_format" for e in result.errors)

    def test_valid_ipv4(self, validator, valid_transaction):
        valid_transaction["ip_address"] = "10.0.0.1"
        result = validator.validate(valid_transaction)
        ip_errors = [e for e in result.errors if e.field == "ip_address"]
        assert len(ip_errors) == 0

    def test_valid_ipv6(self, validator, valid_transaction):
        valid_transaction["ip_address"] = "::1"
        result = validator.validate(valid_transaction)
        ip_errors = [e for e in result.errors if e.field == "ip_address"]
        assert len(ip_errors) == 0

    def test_nullable_ip_address(self, validator, valid_transaction):
        valid_transaction["channel"] = "pos"
        valid_transaction["merchant_id"] = "MERCH-001"
        del valid_transaction["ip_address"]
        result = validator.validate(valid_transaction)
        ip_errors = [e for e in result.errors if e.field == "ip_address"]
        assert len(ip_errors) == 0


# ===========================================================================
# SchemaValidator — Timestamp Validation
# ===========================================================================


class TestTimestampValidation:
    """Tests for ISO 8601 timestamp validation."""

    def test_invalid_timestamp(self, validator, valid_transaction):
        valid_transaction["transaction_timestamp"] = "not-a-timestamp"
        result = validator.validate(valid_transaction)
        assert result.is_valid is False
        assert any(e.rule == "iso8601_format" for e in result.errors)

    def test_valid_timestamp_with_z(self, validator, valid_transaction):
        valid_transaction["transaction_timestamp"] = "2026-06-15T10:30:00Z"
        result = validator.validate(valid_transaction)
        ts_errors = [e for e in result.errors if e.field == "transaction_timestamp"]
        assert len(ts_errors) == 0

    def test_valid_timestamp_with_offset(self, validator, valid_transaction):
        valid_transaction["transaction_timestamp"] = "2026-06-15T10:30:00+05:30"
        result = validator.validate(valid_transaction)
        ts_errors = [e for e in result.errors if e.field == "transaction_timestamp"]
        assert len(ts_errors) == 0

    def test_valid_timestamp_no_timezone(self, validator, valid_transaction):
        valid_transaction["transaction_timestamp"] = "2026-06-15T10:30:00"
        result = validator.validate(valid_transaction)
        ts_errors = [e for e in result.errors if e.field == "transaction_timestamp" and e.rule == "iso8601_format"]
        assert len(ts_errors) == 0


# ===========================================================================
# SchemaValidator — String Length Validation
# ===========================================================================


class TestStringLengthValidation:
    """Tests for min/max string length checks."""

    def test_external_id_exceeds_max_length(self, validator, valid_transaction):
        valid_transaction["external_transaction_id"] = "X" * 65
        result = validator.validate(valid_transaction)
        assert any(
            e.field == "external_transaction_id" and e.rule == "max_length"
            for e in result.errors
        )

    def test_external_id_at_max_length(self, validator, valid_transaction):
        valid_transaction["external_transaction_id"] = "X" * 64
        result = validator.validate(valid_transaction)
        length_errors = [
            e for e in result.errors
            if e.field == "external_transaction_id" and e.rule == "max_length"
        ]
        assert len(length_errors) == 0

    def test_merchant_name_exceeds_max_length(self, validator, valid_transaction):
        valid_transaction["merchant_name"] = "M" * 256
        result = validator.validate(valid_transaction)
        assert any(
            e.field == "merchant_name" and e.rule == "max_length"
            for e in result.errors
        )


# ===========================================================================
# SchemaValidator — Cross-Field Validation
# ===========================================================================


class TestCrossFieldValidation:
    """Tests for cross-field validation rules."""

    def test_international_flag_mismatch(self, validator, valid_transaction):
        valid_transaction["geo_country"] = "GBR"
        valid_transaction["is_international"] = False
        result = validator.validate(valid_transaction)
        assert any(
            e.rule == "international_flag_matches_country" for e in result.warnings
        )

    def test_international_flag_correct(self, validator, valid_transaction):
        valid_transaction["geo_country"] = "GBR"
        valid_transaction["is_international"] = True
        result = validator.validate(valid_transaction)
        intl_warnings = [
            e for e in result.warnings
            if e.rule == "international_flag_matches_country"
        ]
        assert len(intl_warnings) == 0

    def test_domestic_no_flag_needed(self, validator, valid_transaction):
        valid_transaction["geo_country"] = "USA"
        valid_transaction["is_international"] = False
        result = validator.validate(valid_transaction)
        intl_warnings = [
            e for e in result.warnings
            if e.rule == "international_flag_matches_country"
        ]
        assert len(intl_warnings) == 0

    def test_card_type_without_last_four(self, validator, valid_transaction):
        valid_transaction["card_type"] = "debit"
        del valid_transaction["card_last_four"]
        result = validator.validate(valid_transaction)
        assert any(
            e.rule == "card_last_four_required_for_card_transactions"
            for e in result.warnings
        )

    def test_online_without_ip(self, validator, valid_transaction):
        valid_transaction["channel"] = "online"
        del valid_transaction["ip_address"]
        result = validator.validate(valid_transaction)
        assert any(e.rule == "ip_required_for_online" for e in result.warnings)

    def test_geo_latitude_only(self, validator, valid_transaction):
        valid_transaction["geo_latitude"] = 34.0522
        del valid_transaction["geo_longitude"]
        result = validator.validate(valid_transaction)
        assert any(
            e.rule == "geo_coordinates_paired" for e in result.errors
        )

    def test_geo_longitude_only(self, validator, valid_transaction):
        del valid_transaction["geo_latitude"]
        valid_transaction["geo_longitude"] = -118.2437
        result = validator.validate(valid_transaction)
        assert any(
            e.rule == "geo_coordinates_paired" for e in result.errors
        )

    def test_geo_both_present(self, validator, valid_transaction):
        valid_transaction["geo_latitude"] = 34.0522
        valid_transaction["geo_longitude"] = -118.2437
        result = validator.validate(valid_transaction)
        geo_errors = [e for e in result.errors if e.rule == "geo_coordinates_paired"]
        assert len(geo_errors) == 0

    def test_geo_both_absent(self, validator, valid_transaction):
        del valid_transaction["geo_latitude"]
        del valid_transaction["geo_longitude"]
        result = validator.validate(valid_transaction)
        geo_errors = [e for e in result.errors if e.rule == "geo_coordinates_paired"]
        assert len(geo_errors) == 0

    def test_pos_without_merchant(self, validator, valid_transaction):
        valid_transaction["channel"] = "pos"
        del valid_transaction["merchant_id"]
        result = validator.validate(valid_transaction)
        assert any(e.rule == "pos_requires_merchant" for e in result.warnings)

    def test_atm_non_withdrawal(self, validator, valid_transaction):
        valid_transaction["channel"] = "atm"
        valid_transaction["transaction_type"] = "purchase"
        result = validator.validate(valid_transaction)
        assert any(
            e.rule == "atm_no_merchant_category" for e in result.warnings
        )

    def test_refund_exceeds_limit(self, validator, valid_transaction):
        valid_transaction["transaction_type"] = "refund"
        valid_transaction["transaction_amount"] = 75000.00
        result = validator.validate(valid_transaction)
        assert any(e.rule == "refund_amount_limit" for e in result.errors)

    def test_refund_within_limit(self, validator, valid_transaction):
        valid_transaction["transaction_type"] = "refund"
        valid_transaction["transaction_amount"] = 49999.99
        result = validator.validate(valid_transaction)
        refund_errors = [e for e in result.errors if e.rule == "refund_amount_limit"]
        assert len(refund_errors) == 0


# ===========================================================================
# SchemaValidator — Custom Business Rules
# ===========================================================================


class TestCustomBusinessRules:
    """Tests for custom configurable rules."""

    def test_suspicious_round_amount(self, validator, valid_transaction):
        valid_transaction["transaction_amount"] = 5000.00
        result = validator.validate(valid_transaction)
        assert any(e.rule == "suspicious_round_amount" for e in result.warnings)

    def test_non_round_amount_no_warning(self, validator, valid_transaction):
        valid_transaction["transaction_amount"] = 5000.50
        result = validator.validate(valid_transaction)
        round_warnings = [
            e for e in result.warnings if e.rule == "suspicious_round_amount"
        ]
        assert len(round_warnings) == 0

    def test_round_amount_below_threshold(self, validator, valid_transaction):
        valid_transaction["transaction_amount"] = 500.00
        result = validator.validate(valid_transaction)
        round_warnings = [
            e for e in result.warnings if e.rule == "suspicious_round_amount"
        ]
        assert len(round_warnings) == 0

    def test_high_value_transaction(self, validator, valid_transaction):
        valid_transaction["transaction_amount"] = 60000.00
        result = validator.validate(valid_transaction)
        assert any(e.rule == "high_value_transaction" for e in result.warnings)

    def test_normal_value_no_high_value_warning(self, validator, valid_transaction):
        valid_transaction["transaction_amount"] = 1000.50
        result = validator.validate(valid_transaction)
        hv_warnings = [
            e for e in result.warnings if e.rule == "high_value_transaction"
        ]
        assert len(hv_warnings) == 0


# ===========================================================================
# SchemaValidator — Batch Validation
# ===========================================================================


class TestBatchValidation:
    """Tests for batch validation."""

    def test_batch_validation_returns_results_per_record(
        self, validator, valid_transaction
    ):
        batch = [copy.deepcopy(valid_transaction) for _ in range(5)]
        results = validator.validate_batch(batch)
        assert len(results) == 5
        assert all(r.is_valid for r in results)

    def test_batch_mixed_valid_invalid(self, validator, valid_transaction):
        invalid = copy.deepcopy(valid_transaction)
        del invalid["account_id"]
        batch = [valid_transaction, invalid, valid_transaction]
        results = validator.validate_batch(batch)
        assert results[0].is_valid is True
        assert results[1].is_valid is False
        assert results[2].is_valid is True

    def test_empty_batch(self, validator):
        results = validator.validate_batch([])
        assert results == []


# ===========================================================================
# SchemaValidator — Validation Result
# ===========================================================================


class TestValidationResult:
    """Tests for the ValidationResult data class."""

    def test_to_dict(self, validator, valid_transaction):
        result = validator.validate(valid_transaction)
        d = result.to_dict()
        assert "is_valid" in d
        assert "errors" in d
        assert "warnings" in d
        assert "latency_ms" in d
        assert isinstance(d["latency_ms"], float)

    def test_all_issues_combines(self):
        err = ValidationError("f1", "r1", "msg1", ValidationSeverity.ERROR)
        warn = ValidationError("f2", "r2", "msg2", ValidationSeverity.WARNING)
        result = ValidationResult(is_valid=False, errors=[err], warnings=[warn])
        assert len(result.all_issues) == 2

    def test_latency_recorded(self, validator, valid_transaction):
        result = validator.validate(valid_transaction)
        assert result.latency_ms > 0


# ===========================================================================
# SchemaValidator — Metrics
# ===========================================================================


class TestValidationMetrics:
    """Tests for validation metrics tracking."""

    def test_metrics_track_valid(self, validator, valid_transaction):
        validator.metrics.reset()
        validator.validate(valid_transaction)
        assert validator.metrics.total_processed == 1
        assert validator.metrics.total_passed == 1
        assert validator.metrics.total_failed == 0
        assert validator.metrics.pass_rate == 1.0

    def test_metrics_track_invalid(self, validator, valid_transaction):
        validator.metrics.reset()
        del valid_transaction["account_id"]
        validator.validate(valid_transaction)
        assert validator.metrics.total_processed == 1
        assert validator.metrics.total_passed == 0
        assert validator.metrics.total_failed == 1
        assert validator.metrics.pass_rate == 0.0

    def test_metrics_track_warnings(self, validator, valid_transaction):
        validator.metrics.reset()
        valid_transaction["geo_country"] = "GBR"
        valid_transaction["is_international"] = False
        validator.validate(valid_transaction)
        assert validator.metrics.total_warnings > 0

    def test_metrics_failure_reasons(self, validator, valid_transaction):
        validator.metrics.reset()
        del valid_transaction["account_id"]
        validator.validate(valid_transaction)
        reasons = validator.metrics.failure_reasons
        assert "account_id:required_field" in reasons

    def test_metrics_to_dict(self, validator, valid_transaction):
        validator.metrics.reset()
        validator.validate(valid_transaction)
        d = validator.metrics.to_dict()
        assert "total_processed" in d
        assert "pass_rate" in d
        assert "avg_latency_ms" in d

    def test_metrics_reset(self, validator, valid_transaction):
        validator.validate(valid_transaction)
        validator.metrics.reset()
        assert validator.metrics.total_processed == 0
        assert validator.metrics.total_passed == 0
        assert validator.metrics.avg_latency_ms == 0.0

    def test_metrics_batch(self, validator, valid_transaction):
        validator.metrics.reset()
        invalid = copy.deepcopy(valid_transaction)
        del invalid["account_id"]
        validator.validate_batch([valid_transaction, invalid, valid_transaction])
        assert validator.metrics.total_processed == 3
        assert validator.metrics.total_passed == 2
        assert validator.metrics.total_failed == 1


# ===========================================================================
# SchemaValidator — Config Reload
# ===========================================================================


class TestConfigReload:
    """Tests for hot-reload of validation rules."""

    def test_reload_rules(self, validator):
        initial_rules = validator.rules
        validator.reload_rules(config_path=VALIDATION_RULES_PATH)
        assert validator.rules is not initial_rules  # New dict loaded

    def test_reload_preserves_metrics(self, validator, valid_transaction):
        validator.validate(valid_transaction)
        count_before = validator.metrics.total_processed
        validator.reload_rules(config_path=VALIDATION_RULES_PATH)
        assert validator.metrics.total_processed == count_before


# ===========================================================================
# SchemaValidator — Fixture File Tests
# ===========================================================================


class TestFixtureInvalidTransactions:
    """Test all invalid transactions from the fixtures file."""

    def test_all_invalid_transactions_caught(self, validator, invalid_transactions):
        """Every record in invalid_transactions.json should have at least one
        error or warning."""
        for i, txn in enumerate(invalid_transactions):
            result = validator.validate(txn)
            has_issues = len(result.errors) > 0 or len(result.warnings) > 0
            assert has_issues, (
                f"Transaction at index {i} ({txn.get('_comment', 'no comment')}) "
                f"should have validation issues but passed cleanly"
            )

    def test_hard_errors_on_critical_records(self, validator, invalid_transactions):
        """Records with fundamentally broken data should fail validation."""
        # Index 0: missing account_id
        result = validator.validate(invalid_transactions[0])
        assert result.is_valid is False

        # Index 1: negative amount
        result = validator.validate(invalid_transactions[1])
        assert result.is_valid is False

        # Index 2: zero amount
        result = validator.validate(invalid_transactions[2])
        assert result.is_valid is False

        # Index 3: invalid transaction_type
        result = validator.validate(invalid_transactions[3])
        assert result.is_valid is False

        # Index 4: invalid channel
        result = validator.validate(invalid_transactions[4])
        assert result.is_valid is False


# ===========================================================================
# SchemaValidator — Valid Transaction Variations
# ===========================================================================


class TestValidTransactionVariations:
    """Ensure valid transactions are NOT falsely rejected."""

    def test_minimal_valid_transaction(self, validator):
        """Transaction with only required fields should pass."""
        record = {
            "external_transaction_id": "TXN-MINIMAL-001",
            "account_id": "ACC-001",
            "customer_id": "CUST-001",
            "transaction_amount": 10.00,
            "transaction_type": "purchase",
            "channel": "mobile",
            "transaction_timestamp": "2026-06-15T10:00:00Z",
        }
        result = validator.validate(record)
        assert result.is_valid is True

    def test_full_valid_transaction(self, validator, valid_transaction):
        result = validator.validate(valid_transaction)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_valid_with_all_optional_fields(self, validator, valid_transaction):
        valid_transaction["metadata"] = {"source": "test"}
        result = validator.validate(valid_transaction)
        assert result.is_valid is True

    def test_valid_international_transaction(self, validator, valid_transaction):
        valid_transaction["geo_country"] = "GBR"
        valid_transaction["is_international"] = True
        valid_transaction["transaction_currency"] = "GBP"
        result = validator.validate(valid_transaction)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_valid_pos_transaction(self, validator, valid_transaction):
        valid_transaction["channel"] = "pos"
        valid_transaction["merchant_id"] = "MERCH-POS"
        result = validator.validate(valid_transaction)
        assert result.is_valid is True

    def test_valid_atm_withdrawal(self, validator, valid_transaction):
        valid_transaction["channel"] = "atm"
        valid_transaction["transaction_type"] = "withdrawal"
        result = validator.validate(valid_transaction)
        assert result.is_valid is True


# ===========================================================================
# SchemaValidator — Performance
# ===========================================================================


class TestValidationPerformance:
    """Test that validation meets the < 5ms per record SLA."""

    def test_single_record_latency(self, validator, valid_transaction):
        result = validator.validate(valid_transaction)
        assert result.latency_ms < 5.0, (
            f"Validation took {result.latency_ms:.3f}ms, exceeds 5ms SLA"
        )

    def test_batch_average_latency(self, validator, valid_transaction):
        validator.metrics.reset()
        batch = [copy.deepcopy(valid_transaction) for _ in range(100)]
        validator.validate_batch(batch)
        avg = validator.metrics.avg_latency_ms
        assert avg < 5.0, (
            f"Average batch latency {avg:.3f}ms exceeds 5ms SLA"
        )


# ===========================================================================
# QuarantineHandler — Core Operations
# ===========================================================================


class TestQuarantineHandler:
    """Tests for quarantine handler operations."""

    def test_quarantine_record(self, quarantine_handler, validator, valid_transaction):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        entry = quarantine_handler.quarantine(valid_transaction, result)
        assert entry.status == "quarantined"
        assert entry.retry_count == 0
        assert len(entry.failure_reasons) > 0
        assert quarantine_handler.count == 1

    def test_quarantine_stores_failure_reasons(
        self, quarantine_handler, validator, valid_transaction
    ):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        entry = quarantine_handler.quarantine(valid_transaction, result)
        assert any(r["field"] == "account_id" for r in entry.failure_reasons)

    def test_get_quarantined_record(
        self, quarantine_handler, validator, valid_transaction
    ):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        entry = quarantine_handler.quarantine(valid_transaction, result)
        retrieved = quarantine_handler.get(entry.quarantine_id)
        assert retrieved is not None
        assert retrieved.quarantine_id == entry.quarantine_id

    def test_get_nonexistent_record(self, quarantine_handler):
        assert quarantine_handler.get("nonexistent-id") is None

    def test_list_quarantined(self, quarantine_handler, validator, valid_transaction):
        for i in range(5):
            txn = copy.deepcopy(valid_transaction)
            del txn["account_id"]
            txn["external_transaction_id"] = f"TXN-Q-{i}"
            result = validator.validate(txn)
            quarantine_handler.quarantine(txn, result)
        records = quarantine_handler.list_quarantined()
        assert len(records) == 5

    def test_list_quarantined_with_status_filter(
        self, quarantine_handler, validator, valid_transaction
    ):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        quarantine_handler.quarantine(valid_transaction, result)
        assert len(quarantine_handler.list_quarantined(status="quarantined")) == 1
        assert len(quarantine_handler.list_quarantined(status="reprocessed")) == 0

    def test_list_quarantined_with_limit(
        self, quarantine_handler, validator, valid_transaction
    ):
        for i in range(10):
            txn = copy.deepcopy(valid_transaction)
            del txn["account_id"]
            txn["external_transaction_id"] = f"TXN-QL-{i}"
            result = validator.validate(txn)
            quarantine_handler.quarantine(txn, result)
        records = quarantine_handler.list_quarantined(limit=3)
        assert len(records) == 3

    def test_quarantined_record_to_dict(
        self, quarantine_handler, validator, valid_transaction
    ):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        entry = quarantine_handler.quarantine(valid_transaction, result)
        d = entry.to_dict()
        assert "quarantine_id" in d
        assert "record" in d
        assert "failure_reasons" in d
        assert "quarantined_at" in d
        assert "status" in d


# ===========================================================================
# QuarantineHandler — Re-processing
# ===========================================================================


class TestQuarantineReprocessing:
    """Tests for re-processing quarantined records."""

    def test_reprocess_still_invalid(
        self, quarantine_handler, validator, valid_transaction
    ):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        entry = quarantine_handler.quarantine(valid_transaction, result)
        results = quarantine_handler.reprocess(validator, entry.quarantine_id)
        assert len(results) == 1
        _, revalidation = results[0]
        assert revalidation.is_valid is False
        assert entry.retry_count == 1
        assert entry.status == "quarantined"

    def test_reprocess_now_valid(
        self, quarantine_handler, validator, valid_transaction
    ):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        entry = quarantine_handler.quarantine(valid_transaction, result)
        # Fix the record
        entry.record["account_id"] = "ACC-FIXED"
        results = quarantine_handler.reprocess(validator, entry.quarantine_id)
        _, revalidation = results[0]
        assert revalidation.is_valid is True
        assert entry.status == "reprocessed"

    def test_reprocess_all_eligible(
        self, quarantine_handler, validator, valid_transaction
    ):
        for i in range(3):
            txn = copy.deepcopy(valid_transaction)
            del txn["account_id"]
            txn["external_transaction_id"] = f"TXN-RP-{i}"
            result = validator.validate(txn)
            quarantine_handler.quarantine(txn, result)
        results = quarantine_handler.reprocess(validator)
        assert len(results) == 3

    def test_max_retries_discard(
        self, quarantine_handler, validator, valid_transaction
    ):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        entry = quarantine_handler.quarantine(valid_transaction, result)
        # Exhaust retries
        for _ in range(3):
            quarantine_handler.reprocess(validator, entry.quarantine_id)
        assert entry.status == "discarded"
        assert entry.retry_count == 3

    def test_discarded_not_reprocessed(
        self, quarantine_handler, validator, valid_transaction
    ):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        entry = quarantine_handler.quarantine(valid_transaction, result)
        # Discard
        quarantine_handler.discard(entry.quarantine_id)
        # Try reprocessing all — should not pick up discarded
        results = quarantine_handler.reprocess(validator)
        assert len(results) == 0


# ===========================================================================
# QuarantineHandler — Discard
# ===========================================================================


class TestQuarantineDiscard:
    """Tests for manual discard of quarantined records."""

    def test_discard_record(self, quarantine_handler, validator, valid_transaction):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        entry = quarantine_handler.quarantine(valid_transaction, result)
        success = quarantine_handler.discard(entry.quarantine_id)
        assert success is True
        assert entry.status == "discarded"

    def test_discard_nonexistent(self, quarantine_handler):
        assert quarantine_handler.discard("nonexistent-id") is False

    def test_discard_already_discarded(
        self, quarantine_handler, validator, valid_transaction
    ):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        entry = quarantine_handler.quarantine(valid_transaction, result)
        quarantine_handler.discard(entry.quarantine_id)
        assert quarantine_handler.discard(entry.quarantine_id) is False


# ===========================================================================
# QuarantineHandler — Metrics
# ===========================================================================


class TestQuarantineMetrics:
    """Tests for quarantine metrics tracking."""

    def test_quarantine_metrics(self, quarantine_handler, validator, valid_transaction):
        quarantine_handler.metrics.reset()
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        quarantine_handler.quarantine(valid_transaction, result)
        m = quarantine_handler.metrics.to_dict()
        assert m["total_quarantined"] == 1
        assert m["active_quarantined"] == 1
        assert "required_field" in m["reasons_breakdown"]

    def test_metrics_after_reprocess(
        self, quarantine_handler, validator, valid_transaction
    ):
        quarantine_handler.metrics.reset()
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        entry = quarantine_handler.quarantine(valid_transaction, result)
        entry.record["account_id"] = "ACC-FIXED"
        quarantine_handler.reprocess(validator, entry.quarantine_id)
        m = quarantine_handler.metrics.to_dict()
        assert m["total_reprocessed"] == 1
        assert m["active_quarantined"] == 0

    def test_metrics_after_discard(
        self, quarantine_handler, validator, valid_transaction
    ):
        quarantine_handler.metrics.reset()
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        entry = quarantine_handler.quarantine(valid_transaction, result)
        quarantine_handler.discard(entry.quarantine_id)
        m = quarantine_handler.metrics.to_dict()
        assert m["total_discarded"] == 1
        assert m["active_quarantined"] == 0

    def test_metrics_reset(self, quarantine_handler, validator, valid_transaction):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        quarantine_handler.quarantine(valid_transaction, result)
        quarantine_handler.metrics.reset()
        m = quarantine_handler.metrics.to_dict()
        assert m["total_quarantined"] == 0
        assert m["active_quarantined"] == 0


# ===========================================================================
# QuarantineHandler — Counts
# ===========================================================================


class TestQuarantineCounts:
    """Tests for quarantine count properties."""

    def test_count_active(self, quarantine_handler, validator, valid_transaction):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        quarantine_handler.quarantine(valid_transaction, result)
        assert quarantine_handler.count == 1
        assert quarantine_handler.total_count == 1

    def test_count_after_discard(
        self, quarantine_handler, validator, valid_transaction
    ):
        del valid_transaction["account_id"]
        result = validator.validate(valid_transaction)
        entry = quarantine_handler.quarantine(valid_transaction, result)
        quarantine_handler.discard(entry.quarantine_id)
        assert quarantine_handler.count == 0
        assert quarantine_handler.total_count == 1
