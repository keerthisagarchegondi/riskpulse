"""Unit tests for Schema Registry and Avro schema validation.

Tests cover:
- Schema loading and caching
- Record validation against Avro schemas
- Serialization/deserialization roundtrip
- Type coercion (floats to decimal bytes, enum validation)
- Error handling for malformed schemas and records
- Edge cases (null unions, missing optional fields, defaults)
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ingestion.schema_registry import (
    SchemaRegistry,
    SchemaRegistryError,
    SchemaValidationError,
    get_schema_registry,
)


# ============================================================================
# Fixtures
# ============================================================================

SCHEMAS_DIR = Path(__file__).resolve().parent.parent.parent / "schemas"


@pytest.fixture
def registry():
    """Create a SchemaRegistry pointing to the project schemas directory."""
    return SchemaRegistry(schemas_dir=SCHEMAS_DIR)


@pytest.fixture
def valid_event():
    """A complete valid transaction event matching the Avro schema."""
    return {
        "event_id": "550e8400-e29b-41d4-a716-446655440000",
        "event_timestamp": 1718450000000,
        "event_version": "1.0.0",
        "external_transaction_id": "TXN-ABC123DEF456",
        "account_id": "ACC-12345",
        "customer_id": "CUST-67890",
        "merchant_id": "MERCH-001",
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
        "metadata": None,
    }


@pytest.fixture
def minimal_event():
    """A valid event with only required fields and defaults."""
    return {
        "event_id": "550e8400-e29b-41d4-a716-446655440001",
        "event_timestamp": 1718450000000,
        "event_version": "1.0.0",
        "external_transaction_id": "TXN-MINIMAL001",
        "account_id": "ACC-99999",
        "customer_id": "CUST-99999",
        "merchant_id": "MERCH-001",
        "merchant_name": "Test Merchant",
        "merchant_category_code": "5411",
        "transaction_amount": 10.00,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "pos",
        "card_type": "debit",
        "card_last_four": "1234",
        "ip_address": None,
        "device_id": None,
        "device_type": None,
        "geo_latitude": None,
        "geo_longitude": None,
        "geo_country": None,
        "geo_city": None,
        "is_international": False,
        "transaction_timestamp": "2026-06-20T12:00:00Z",
        "metadata": None,
    }


# ============================================================================
# Schema Loading Tests
# ============================================================================


class TestSchemaLoading:
    """Test schema file loading and parsing."""

    def test_load_transaction_event_schema(self, registry):
        """Should successfully load the transaction_event schema."""
        schema = registry.get_schema("transaction_event")
        assert schema is not None
        assert schema.name == "TransactionEvent"
        assert schema.namespace == "com.riskpulse.events"

    def test_schema_caching(self, registry):
        """Second load should return cached schema."""
        schema1 = registry.get_schema("transaction_event")
        schema2 = registry.get_schema("transaction_event")
        assert schema1 is schema2

    def test_load_nonexistent_schema_raises(self, registry):
        """Should raise SchemaRegistryError for missing schema files."""
        with pytest.raises(SchemaRegistryError, match="not found"):
            registry.get_schema("nonexistent_schema")

    def test_load_invalid_json_raises(self, tmp_path):
        """Should raise SchemaRegistryError for invalid JSON."""
        bad_schema = tmp_path / "bad.avsc"
        bad_schema.write_text("{ invalid json }")

        registry = SchemaRegistry(schemas_dir=tmp_path)
        with pytest.raises(SchemaRegistryError, match="Invalid JSON"):
            registry.get_schema("bad")

    def test_load_invalid_avro_schema_raises(self, tmp_path):
        """Should raise SchemaRegistryError for valid JSON but invalid Avro."""
        bad_schema = tmp_path / "invalid_avro.avsc"
        bad_schema.write_text(json.dumps({
            "type": "record",
            "name": "Bad",
            "fields": [
                {"name": "field1", "type": "nonexistent_type"}
            ]
        }))

        registry = SchemaRegistry(schemas_dir=tmp_path)
        with pytest.raises(SchemaRegistryError, match="Invalid Avro schema"):
            registry.get_schema("invalid_avro")

    def test_list_schemas(self, registry):
        """Should list available schema names."""
        schemas = registry.list_schemas()
        assert "transaction_event" in schemas

    def test_get_schema_json(self, registry):
        """Should return raw JSON dict."""
        schema_json = registry.get_schema_json("transaction_event")
        assert schema_json["type"] == "record"
        assert schema_json["name"] == "TransactionEvent"
        assert "fields" in schema_json


# ============================================================================
# Validation Tests
# ============================================================================


class TestSchemaValidation:
    """Test record validation against schemas."""

    def test_validate_valid_event(self, registry, valid_event):
        """Should pass validation for a complete valid event."""
        result = registry.validate("transaction_event", valid_event)
        assert result["account_id"] == "ACC-12345"
        assert result["transaction_type"] == "purchase"

    def test_validate_minimal_event(self, registry, minimal_event):
        """Should pass validation with null optional fields."""
        result = registry.validate("transaction_event", minimal_event)
        assert result["ip_address"] is None
        assert result["device_id"] is None

    def test_validate_rejects_invalid_enum(self, registry, valid_event):
        """Should reject invalid enum values."""
        valid_event["transaction_type"] = "invalid_type"
        with pytest.raises(SchemaValidationError):
            registry.validate("transaction_event", valid_event)

    def test_validate_rejects_invalid_channel(self, registry, valid_event):
        """Should reject invalid channel enum."""
        valid_event["channel"] = "smoke_signal"
        with pytest.raises(SchemaValidationError):
            registry.validate("transaction_event", valid_event)

    def test_validate_rejects_invalid_card_type(self, registry, valid_event):
        """Should reject invalid card_type enum."""
        valid_event["card_type"] = "bitcoin"
        with pytest.raises(SchemaValidationError):
            registry.validate("transaction_event", valid_event)

    def test_validate_accepts_all_transaction_types(self, registry, valid_event):
        """Should accept all valid transaction type enums."""
        for txn_type in ["purchase", "withdrawal", "transfer", "refund"]:
            event = valid_event.copy()
            event["transaction_type"] = txn_type
            result = registry.validate("transaction_event", event)
            assert result["transaction_type"] == txn_type

    def test_validate_accepts_all_channels(self, registry, valid_event):
        """Should accept all valid channel enums."""
        for channel in ["online", "pos", "atm", "mobile"]:
            event = valid_event.copy()
            event["channel"] = channel
            result = registry.validate("transaction_event", event)
            assert result["channel"] == channel

    def test_validate_handles_metadata_map(self, registry, valid_event):
        """Should accept metadata as a string map."""
        valid_event["metadata"] = {"source": "api", "batch_id": "B-001"}
        result = registry.validate("transaction_event", valid_event)
        assert result["metadata"] == {"source": "api", "batch_id": "B-001"}

    def test_validate_handles_null_metadata(self, registry, valid_event):
        """Should accept null metadata."""
        valid_event["metadata"] = None
        result = registry.validate("transaction_event", valid_event)
        assert result["metadata"] is None


# ============================================================================
# Serialization/Deserialization Tests
# ============================================================================


class TestSerialization:
    """Test Avro binary serialization roundtrip."""

    def test_serialize_produces_bytes(self, registry, valid_event):
        """Serialization should produce non-empty bytes."""
        coerced = registry._coerce_types(
            registry.get_schema("transaction_event"), valid_event
        )
        data = registry.serialize("transaction_event", coerced)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_serialize_deserialize_roundtrip(self, registry, valid_event):
        """Should faithfully roundtrip through serialize/deserialize."""
        coerced = registry._coerce_types(
            registry.get_schema("transaction_event"), valid_event
        )
        data = registry.serialize("transaction_event", coerced)
        result = registry.deserialize("transaction_event", data)

        assert result["event_id"] == valid_event["event_id"]
        assert result["account_id"] == valid_event["account_id"]
        assert result["merchant_name"] == valid_event["merchant_name"]
        assert result["transaction_type"] == valid_event["transaction_type"]
        assert result["channel"] == valid_event["channel"]
        assert result["is_international"] == valid_event["is_international"]
        assert result["transaction_timestamp"] == valid_event["transaction_timestamp"]

    def test_serialize_minimal_event_roundtrip(self, registry, minimal_event):
        """Should roundtrip minimal events with null fields."""
        coerced = registry._coerce_types(
            registry.get_schema("transaction_event"), minimal_event
        )
        data = registry.serialize("transaction_event", coerced)
        result = registry.deserialize("transaction_event", data)

        assert result["ip_address"] is None
        assert result["device_id"] is None
        assert result["geo_latitude"] is None


# ============================================================================
# Type Coercion Tests
# ============================================================================


class TestTypeCoercion:
    """Test type coercion for Avro compatibility."""

    def test_decimal_bytes_from_float(self, registry):
        """Should convert float to Avro decimal bytes."""
        result = registry._to_decimal_bytes(125.50, precision=12, scale=2)
        assert isinstance(result, bytes)
        # 125.50 * 100 = 12550, which in big-endian signed bytes
        value = int.from_bytes(result, byteorder="big", signed=True)
        assert value == 12550

    def test_decimal_bytes_from_int(self, registry):
        """Should convert int to Avro decimal bytes."""
        result = registry._to_decimal_bytes(100, precision=12, scale=2)
        value = int.from_bytes(result, byteorder="big", signed=True)
        assert value == 10000

    def test_decimal_bytes_from_decimal(self, registry):
        """Should handle Decimal input directly."""
        result = registry._to_decimal_bytes(Decimal("99.99"), precision=12, scale=2)
        value = int.from_bytes(result, byteorder="big", signed=True)
        assert value == 9999

    def test_decimal_bytes_zero(self, registry):
        """Zero should serialize to single null byte."""
        result = registry._to_decimal_bytes(0, precision=12, scale=2)
        assert result == b"\x00"

    def test_decimal_bytes_large_amount(self, registry):
        """Should handle large transaction amounts."""
        result = registry._to_decimal_bytes(9999.99, precision=12, scale=2)
        value = int.from_bytes(result, byteorder="big", signed=True)
        assert value == 999999

    def test_coerce_types_handles_union_null(self, registry, valid_event):
        """Should pass through None for nullable union fields."""
        valid_event["ip_address"] = None
        schema = registry.get_schema("transaction_event")
        coerced = registry._coerce_types(schema, valid_event)
        assert coerced["ip_address"] is None

    def test_coerce_types_handles_union_value(self, registry, valid_event):
        """Should pass through non-null values for union fields."""
        valid_event["ip_address"] = "10.0.0.1"
        schema = registry.get_schema("transaction_event")
        coerced = registry._coerce_types(schema, valid_event)
        assert coerced["ip_address"] == "10.0.0.1"


# ============================================================================
# Singleton Tests
# ============================================================================


class TestSingleton:
    """Test the get_schema_registry singleton."""

    def test_get_schema_registry_returns_instance(self):
        """Should return a SchemaRegistry instance."""
        # Clear lru_cache for isolated test
        get_schema_registry.cache_clear()
        registry = get_schema_registry()
        assert isinstance(registry, SchemaRegistry)

    def test_get_schema_registry_is_singleton(self):
        """Should return the same instance on repeated calls."""
        get_schema_registry.cache_clear()
        r1 = get_schema_registry()
        r2 = get_schema_registry()
        assert r1 is r2
