"""Schema Registry client for Avro schema management.

Provides schema validation, registration, and compatibility checking
for Kafka event schemas. Supports both Confluent Schema Registry
and local file-based schemas for development/testing.
"""

from __future__ import annotations

import io
import json
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import avro.io
import avro.schema
import structlog

from src.utils.config import get_settings

logger = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class SchemaRegistryError(Exception):
    """Raised when schema operations fail."""


class SchemaValidationError(SchemaRegistryError):
    """Raised when a record fails schema validation."""


class SchemaRegistry:
    """Manages Avro schemas for Kafka event serialization.

    Supports local file-based schemas and Confluent Schema Registry.
    In production, register schemas with Confluent Schema Registry for
    centralized governance; locally, schemas load from the `schemas/` directory.
    """

    def __init__(
        self,
        schema_registry_url: str | None = None,
        schemas_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self._registry_url = schema_registry_url or settings.get(
            "kafka.schema_registry_url"
        )
        self._schemas_dir = schemas_dir or _PROJECT_ROOT / "schemas"
        self._schema_cache: dict[str, avro.schema.Schema] = {}
        self._parsed_schemas: dict[str, avro.schema.Schema] = {}

    def get_schema(self, schema_name: str) -> avro.schema.Schema:
        """Load and cache an Avro schema by name.

        Args:
            schema_name: Schema filename without extension (e.g., 'transaction_event')

        Returns:
            Parsed Avro schema object.

        Raises:
            SchemaRegistryError: If schema file not found or invalid.
        """
        if schema_name in self._schema_cache:
            return self._schema_cache[schema_name]

        schema_path = self._schemas_dir / f"{schema_name}.avsc"
        if not schema_path.exists():
            raise SchemaRegistryError(
                f"Schema file not found: {schema_path}"
            )

        try:
            with open(schema_path, "r") as f:
                schema_json = json.load(f)

            parsed = avro.schema.parse(json.dumps(schema_json))
            self._schema_cache[schema_name] = parsed
            logger.info(
                "schema_loaded",
                schema_name=schema_name,
                schema_path=str(schema_path),
            )
            return parsed
        except json.JSONDecodeError as e:
            raise SchemaRegistryError(
                f"Invalid JSON in schema file {schema_path}: {e}"
            ) from e
        except avro.schema.SchemaParseException as e:
            raise SchemaRegistryError(
                f"Invalid Avro schema {schema_path}: {e}"
            ) from e

    def validate(self, schema_name: str, record: dict[str, Any]) -> dict[str, Any]:
        """Validate a record against a named schema.

        Performs type coercion for compatible types (e.g., float -> Decimal).

        Args:
            schema_name: Name of the schema to validate against.
            record: Dictionary record to validate.

        Returns:
            The validated (and possibly coerced) record.

        Raises:
            SchemaValidationError: If the record does not conform to the schema.
        """
        schema = self.get_schema(schema_name)
        coerced = self._coerce_types(schema, record)

        # Validate by attempting serialization
        try:
            self.serialize(schema_name, coerced)
        except Exception as e:
            raise SchemaValidationError(
                f"Record failed schema validation for '{schema_name}': {e}"
            ) from e

        return coerced

    def serialize(self, schema_name: str, record: dict[str, Any]) -> bytes:
        """Serialize a record to Avro binary format.

        Args:
            schema_name: Name of the schema for serialization.
            record: Dictionary record to serialize.

        Returns:
            Avro-encoded bytes.

        Raises:
            SchemaValidationError: If serialization fails due to schema mismatch.
        """
        schema = self.get_schema(schema_name)
        try:
            writer = avro.io.DatumWriter(schema)
            buffer = io.BytesIO()
            encoder = avro.io.BinaryEncoder(buffer)
            writer.write(record, encoder)
            return buffer.getvalue()
        except avro.io.AvroTypeException as e:
            raise SchemaValidationError(
                f"Serialization failed for schema '{schema_name}': {e}"
            ) from e

    def deserialize(self, schema_name: str, data: bytes) -> dict[str, Any]:
        """Deserialize Avro binary data back to a dictionary.

        Args:
            schema_name: Name of the schema for deserialization.
            data: Avro-encoded bytes.

        Returns:
            Deserialized dictionary record.
        """
        schema = self.get_schema(schema_name)
        reader = avro.io.DatumReader(schema)
        buffer = io.BytesIO(data)
        decoder = avro.io.BinaryDecoder(buffer)
        return reader.read(decoder)

    def _coerce_types(
        self, schema: avro.schema.Schema, record: dict[str, Any]
    ) -> dict[str, Any]:
        """Coerce Python types to Avro-compatible types.

        Handles:
        - float/int -> Decimal bytes for decimal logical types
        - None handling for union types
        """
        if schema.type != "record":
            return record

        coerced = {}
        for field in schema.fields:
            field_name = field.name
            if field_name not in record:
                if field.has_default:
                    coerced[field_name] = field.default
                continue

            value = record[field_name]
            coerced[field_name] = self._coerce_field(field.type, value)

        return coerced

    def _coerce_field(self, field_schema: avro.schema.Schema, value: Any) -> Any:
        """Coerce a single field value based on its schema type."""
        # Handle union types (e.g., ["null", "string"])
        if field_schema.type == "union":
            if value is None:
                return value
            # Find the non-null schema in the union
            for schema in field_schema.schemas:
                if schema.type == "null":
                    continue
                return self._coerce_field(schema, value)
            return value

        # Handle decimal logical type
        props = getattr(field_schema, "props", {})
        if (
            field_schema.type == "bytes"
            and props.get("logicalType") == "decimal"
        ):
            return self._to_decimal_bytes(
                value,
                props.get("precision", 12),
                props.get("scale", 2),
            )

        # Handle enum types - convert string to proper value
        if field_schema.type == "enum":
            if isinstance(value, str) and value in field_schema.symbols:
                return value
            raise SchemaValidationError(
                f"Invalid enum value '{value}'. Expected one of: {field_schema.symbols}"
            )

        return value

    @staticmethod
    def _to_decimal_bytes(value: float | int | Decimal, precision: int, scale: int) -> bytes:
        """Convert a numeric value to Avro decimal bytes encoding.

        Avro encodes decimals as big-endian two's complement byte arrays
        representing the unscaled integer value.
        """
        if isinstance(value, (int, float)):
            value = Decimal(str(value))

        # Scale to integer: 125.50 with scale=2 -> 12550
        unscaled = int(value * (10**scale))

        # Encode as big-endian signed bytes (two's complement)
        if unscaled == 0:
            return b"\x00"

        # Calculate required byte length
        byte_length = (unscaled.bit_length() + 8) // 8  # +1 for sign bit
        return unscaled.to_bytes(byte_length, byteorder="big", signed=True)

    def get_schema_json(self, schema_name: str) -> dict[str, Any]:
        """Get the raw JSON representation of a schema.

        Args:
            schema_name: Schema filename without extension.

        Returns:
            Schema as a dictionary.
        """
        schema_path = self._schemas_dir / f"{schema_name}.avsc"
        if not schema_path.exists():
            raise SchemaRegistryError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r") as f:
            return json.load(f)

    def list_schemas(self) -> list[str]:
        """List all available schema names in the schemas directory."""
        if not self._schemas_dir.exists():
            return []
        return [
            p.stem for p in self._schemas_dir.glob("*.avsc")
        ]


@lru_cache(maxsize=1)
def get_schema_registry() -> SchemaRegistry:
    """Get the singleton SchemaRegistry instance."""
    return SchemaRegistry()
