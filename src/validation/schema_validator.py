"""Schema validation engine for incoming transaction events.

Enforces data contracts using YAML-configurable validation rules.
Supports field type checks, range validation, cross-field rules,
and custom business logic. Designed for < 5ms per record latency.
"""

from __future__ import annotations

import ipaddress
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__, component="schema_validator")


class ValidationSeverity(str, Enum):
    """Severity level for validation failures."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationError:
    """A single validation failure."""

    field: str
    rule: str
    message: str
    severity: ValidationSeverity
    value: Any = None


@dataclass
class ValidationResult:
    """Result of validating a single transaction record."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def all_issues(self) -> list[ValidationError]:
        return self.errors + self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": [
                {
                    "field": e.field,
                    "rule": e.rule,
                    "message": e.message,
                    "severity": e.severity.value,
                }
                for e in self.errors
            ],
            "warnings": [
                {
                    "field": w.field,
                    "rule": w.rule,
                    "message": w.message,
                    "severity": w.severity.value,
                }
                for w in self.warnings
            ],
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass
class ValidationMetrics:
    """Thread-safe metrics for validation operations."""

    total_processed: int = 0
    total_passed: int = 0
    total_failed: int = 0
    total_warnings: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, result: ValidationResult) -> None:
        with self._lock:
            self.total_processed += 1
            if result.is_valid:
                self.total_passed += 1
            else:
                self.total_failed += 1
                for error in result.errors:
                    key = f"{error.field}:{error.rule}"
                    self.failure_reasons[key] = self.failure_reasons.get(key, 0) + 1
            self.total_warnings += len(result.warnings)
            self.total_latency_ms += result.latency_ms

    @property
    def pass_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.total_passed / self.total_processed

    @property
    def avg_latency_ms(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.total_latency_ms / self.total_processed

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_processed": self.total_processed,
                "total_passed": self.total_passed,
                "total_failed": self.total_failed,
                "total_warnings": self.total_warnings,
                "pass_rate": round(self.pass_rate, 4),
                "avg_latency_ms": round(self.avg_latency_ms, 3),
                "failure_reasons": dict(self.failure_reasons),
            }

    def reset(self) -> None:
        with self._lock:
            self.total_processed = 0
            self.total_passed = 0
            self.total_failed = 0
            self.total_warnings = 0
            self.failure_reasons.clear()
            self.total_latency_ms = 0.0


def _load_validation_rules(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load validation rules from YAML config file."""
    if config_path is None:
        settings = get_settings()
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "config" / "validation_rules.yaml"

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Validation rules config not found: {path}")

    with open(path, "r") as f:
        rules = yaml.safe_load(f)

    if not isinstance(rules, dict):
        raise ValueError("Validation rules config must be a YAML mapping")

    return rules


class SchemaValidator:
    """Production-grade schema validation engine for transaction events.

    Loads validation rules from YAML configuration and applies them to
    incoming transaction records. Supports required field checks, type
    validation, range validation, cross-field rules, and custom business
    rules.

    Usage:
        validator = SchemaValidator()
        result = validator.validate(transaction_dict)
        if not result.is_valid:
            quarantine_handler.quarantine(transaction_dict, result)
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._rules = _load_validation_rules(config_path)
        self._metrics = ValidationMetrics()
        self._compiled_patterns: dict[str, re.Pattern] = {}
        self._precompile_patterns()
        logger.info(
            "Schema validator initialized",
            required_fields=len(self._rules.get("required_fields", [])),
            field_types=len(self._rules.get("field_types", {})),
            cross_field_rules=len(self._rules.get("cross_field_rules", [])),
            custom_rules=len(self._rules.get("custom_rules", [])),
        )

    def _precompile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        for field_name, field_config in self._rules.get("field_types", {}).items():
            if "pattern" in field_config:
                self._compiled_patterns[field_name] = re.compile(field_config["pattern"])

    @property
    def metrics(self) -> ValidationMetrics:
        return self._metrics

    @property
    def rules(self) -> dict[str, Any]:
        return self._rules

    def reload_rules(self, config_path: str | Path | None = None) -> None:
        """Hot-reload validation rules from config without restart."""
        self._rules = _load_validation_rules(config_path)
        self._compiled_patterns.clear()
        self._precompile_patterns()
        logger.info("Validation rules reloaded")

    def validate(self, record: dict[str, Any]) -> ValidationResult:
        """Validate a single transaction record against all rules.

        Args:
            record: Transaction record as a dictionary.

        Returns:
            ValidationResult with errors, warnings, and latency.
        """
        start = time.perf_counter()
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        # 1. Required field checks
        self._check_required_fields(record, errors)

        # 2. Field type validation
        self._check_field_types(record, errors, warnings)

        # 3. Cross-field validation
        self._check_cross_field_rules(record, errors, warnings)

        # 4. Custom business rules
        self._check_custom_rules(record, errors, warnings)

        latency_ms = (time.perf_counter() - start) * 1000
        is_valid = len(errors) == 0

        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            latency_ms=latency_ms,
        )

        self._metrics.record(result)
        return result

    def validate_batch(
        self, records: list[dict[str, Any]]
    ) -> list[ValidationResult]:
        """Validate a batch of transaction records.

        Args:
            records: List of transaction record dictionaries.

        Returns:
            List of ValidationResult, one per record.
        """
        return [self.validate(record) for record in records]

    # -------------------------------------------------------------------------
    # Required fields
    # -------------------------------------------------------------------------

    def _check_required_fields(
        self, record: dict[str, Any], errors: list[ValidationError]
    ) -> None:
        required = self._rules.get("required_fields", [])
        for field_name in required:
            value = record.get(field_name)
            if value is None:
                errors.append(
                    ValidationError(
                        field=field_name,
                        rule="required_field",
                        message=f"Required field '{field_name}' is missing",
                        severity=ValidationSeverity.ERROR,
                    )
                )
            elif isinstance(value, str) and not value.strip():
                errors.append(
                    ValidationError(
                        field=field_name,
                        rule="required_field_empty",
                        message=f"Required field '{field_name}' is empty",
                        severity=ValidationSeverity.ERROR,
                    )
                )

    # -------------------------------------------------------------------------
    # Field type validation
    # -------------------------------------------------------------------------

    def _check_field_types(
        self,
        record: dict[str, Any],
        errors: list[ValidationError],
        warnings: list[ValidationError],
    ) -> None:
        field_types = self._rules.get("field_types", {})
        for field_name, config in field_types.items():
            value = record.get(field_name)

            # Skip nullable fields that are absent
            if value is None:
                if config.get("nullable", False):
                    continue
                # If not nullable and not in required (required handled separately)
                if field_name not in self._rules.get("required_fields", []):
                    continue
                # Already reported by required check
                continue

            expected_type = config.get("type")

            # Type check
            if expected_type == "string":
                self._validate_string_field(field_name, value, config, errors)
            elif expected_type == "number":
                self._validate_number_field(field_name, value, config, errors)
            elif expected_type == "boolean":
                self._validate_boolean_field(field_name, value, config, errors)

    def _validate_string_field(
        self,
        field_name: str,
        value: Any,
        config: dict[str, Any],
        errors: list[ValidationError],
    ) -> None:
        if not isinstance(value, str):
            errors.append(
                ValidationError(
                    field=field_name,
                    rule="type_check",
                    message=f"Expected string for '{field_name}', got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                    value=value,
                )
            )
            return

        # Min length
        min_len = config.get("min_length")
        if min_len is not None and len(value) < min_len:
            errors.append(
                ValidationError(
                    field=field_name,
                    rule="min_length",
                    message=f"'{field_name}' must be at least {min_len} characters",
                    severity=ValidationSeverity.ERROR,
                    value=value,
                )
            )

        # Max length
        max_len = config.get("max_length")
        if max_len is not None and len(value) > max_len:
            errors.append(
                ValidationError(
                    field=field_name,
                    rule="max_length",
                    message=f"'{field_name}' must be at most {max_len} characters",
                    severity=ValidationSeverity.ERROR,
                    value=value,
                )
            )

        # Allowed values (enum check)
        allowed = config.get("allowed_values")
        if allowed is not None and value not in allowed:
            errors.append(
                ValidationError(
                    field=field_name,
                    rule="allowed_values",
                    message=f"'{field_name}' value '{value}' not in allowed values: {allowed}",
                    severity=ValidationSeverity.ERROR,
                    value=value,
                )
            )

        # Regex pattern
        if field_name in self._compiled_patterns:
            if not self._compiled_patterns[field_name].match(value):
                errors.append(
                    ValidationError(
                        field=field_name,
                        rule="pattern",
                        message=f"'{field_name}' value '{value}' does not match required pattern",
                        severity=ValidationSeverity.ERROR,
                        value=value,
                    )
                )

        # Format: ip_address
        if config.get("format") == "ip_address":
            self._validate_ip_address(field_name, value, errors)

        # Format: iso8601
        if config.get("format") == "iso8601":
            self._validate_iso8601(field_name, value, errors)

    def _validate_number_field(
        self,
        field_name: str,
        value: Any,
        config: dict[str, Any],
        errors: list[ValidationError],
    ) -> None:
        if not isinstance(value, (int, float)):
            errors.append(
                ValidationError(
                    field=field_name,
                    rule="type_check",
                    message=f"Expected number for '{field_name}', got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                    value=value,
                )
            )
            return

        # Boolean is a subclass of int in Python — reject it
        if isinstance(value, bool):
            errors.append(
                ValidationError(
                    field=field_name,
                    rule="type_check",
                    message=f"Expected number for '{field_name}', got boolean",
                    severity=ValidationSeverity.ERROR,
                    value=value,
                )
            )
            return

        min_val = config.get("min_value")
        if min_val is not None and value < min_val:
            errors.append(
                ValidationError(
                    field=field_name,
                    rule="min_value",
                    message=f"'{field_name}' value {value} is below minimum {min_val}",
                    severity=ValidationSeverity.ERROR,
                    value=value,
                )
            )

        max_val = config.get("max_value")
        if max_val is not None and value > max_val:
            errors.append(
                ValidationError(
                    field=field_name,
                    rule="max_value",
                    message=f"'{field_name}' value {value} exceeds maximum {max_val}",
                    severity=ValidationSeverity.ERROR,
                    value=value,
                )
            )

    def _validate_boolean_field(
        self,
        field_name: str,
        value: Any,
        config: dict[str, Any],
        errors: list[ValidationError],
    ) -> None:
        if not isinstance(value, bool):
            errors.append(
                ValidationError(
                    field=field_name,
                    rule="type_check",
                    message=f"Expected boolean for '{field_name}', got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                    value=value,
                )
            )

    def _validate_ip_address(
        self,
        field_name: str,
        value: str,
        errors: list[ValidationError],
    ) -> None:
        try:
            ipaddress.ip_address(value)
        except ValueError:
            errors.append(
                ValidationError(
                    field=field_name,
                    rule="ip_address_format",
                    message=f"'{field_name}' value '{value}' is not a valid IP address",
                    severity=ValidationSeverity.ERROR,
                    value=value,
                )
            )

    def _validate_iso8601(
        self,
        field_name: str,
        value: str,
        errors: list[ValidationError],
    ) -> None:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            errors.append(
                ValidationError(
                    field=field_name,
                    rule="iso8601_format",
                    message=f"'{field_name}' value '{value}' is not valid ISO 8601",
                    severity=ValidationSeverity.ERROR,
                    value=value,
                )
            )

    # -------------------------------------------------------------------------
    # Cross-field validation
    # -------------------------------------------------------------------------

    def _check_cross_field_rules(
        self,
        record: dict[str, Any],
        errors: list[ValidationError],
        warnings: list[ValidationError],
    ) -> None:
        cross_rules = self._rules.get("cross_field_rules", [])
        for rule in cross_rules:
            severity = ValidationSeverity(rule.get("severity", "error"))
            target = errors if severity == ValidationSeverity.ERROR else warnings

            condition = rule.get("condition", "")
            name = rule["name"]

            if condition == "geo_country is not None and geo_country != domestic_country and is_international is False":
                self._rule_international_flag(record, rule, name, severity, target)
            elif condition == "card_type is not None and card_last_four is None":
                self._rule_card_last_four_required(record, name, severity, target)
            elif condition == "channel == 'online' and ip_address is None":
                self._rule_ip_required_online(record, name, severity, target)
            elif condition == "(geo_latitude is None) != (geo_longitude is None)":
                self._rule_geo_coordinates_paired(record, name, severity, target)
            elif condition == "channel == 'pos' and merchant_id is None":
                self._rule_pos_requires_merchant(record, name, severity, target)
            elif condition == "channel == 'atm' and transaction_type != 'withdrawal'":
                self._rule_atm_withdrawal(record, name, severity, target)
            elif condition == "transaction_type == 'refund' and transaction_amount > 50000":
                self._rule_refund_limit(record, rule, name, severity, target)
            elif condition == "transaction_timestamp_is_future":
                self._rule_future_timestamp(record, rule, name, severity, target)
            elif condition == "transaction_timestamp_is_stale":
                self._rule_stale_timestamp(record, rule, name, severity, target)

    def _rule_international_flag(
        self,
        record: dict[str, Any],
        rule: dict[str, Any],
        name: str,
        severity: ValidationSeverity,
        target: list[ValidationError],
    ) -> None:
        geo_country = record.get("geo_country")
        is_international = record.get("is_international", False)
        domestic = rule.get("domestic_country", "USA")

        if geo_country is not None and geo_country != domestic and not is_international:
            target.append(
                ValidationError(
                    field="is_international",
                    rule=name,
                    message=f"is_international should be True for country '{geo_country}' (domestic: '{domestic}')",
                    severity=severity,
                    value=is_international,
                )
            )

    def _rule_card_last_four_required(
        self,
        record: dict[str, Any],
        name: str,
        severity: ValidationSeverity,
        target: list[ValidationError],
    ) -> None:
        if record.get("card_type") is not None and record.get("card_last_four") is None:
            target.append(
                ValidationError(
                    field="card_last_four",
                    rule=name,
                    message="card_last_four should be provided when card_type is set",
                    severity=severity,
                )
            )

    def _rule_ip_required_online(
        self,
        record: dict[str, Any],
        name: str,
        severity: ValidationSeverity,
        target: list[ValidationError],
    ) -> None:
        if record.get("channel") == "online" and record.get("ip_address") is None:
            target.append(
                ValidationError(
                    field="ip_address",
                    rule=name,
                    message="ip_address is required for online channel transactions",
                    severity=severity,
                )
            )

    def _rule_geo_coordinates_paired(
        self,
        record: dict[str, Any],
        name: str,
        severity: ValidationSeverity,
        target: list[ValidationError],
    ) -> None:
        has_lat = record.get("geo_latitude") is not None
        has_lon = record.get("geo_longitude") is not None
        if has_lat != has_lon:
            missing = "geo_longitude" if has_lat else "geo_latitude"
            target.append(
                ValidationError(
                    field=missing,
                    rule=name,
                    message="geo_latitude and geo_longitude must both be present or both absent",
                    severity=severity,
                )
            )

    def _rule_pos_requires_merchant(
        self,
        record: dict[str, Any],
        name: str,
        severity: ValidationSeverity,
        target: list[ValidationError],
    ) -> None:
        if record.get("channel") == "pos" and record.get("merchant_id") is None:
            target.append(
                ValidationError(
                    field="merchant_id",
                    rule=name,
                    message="POS transactions must have merchant_id",
                    severity=severity,
                )
            )

    def _rule_atm_withdrawal(
        self,
        record: dict[str, Any],
        name: str,
        severity: ValidationSeverity,
        target: list[ValidationError],
    ) -> None:
        if record.get("channel") == "atm" and record.get("transaction_type") != "withdrawal":
            target.append(
                ValidationError(
                    field="transaction_type",
                    rule=name,
                    message="ATM channel transactions should have 'withdrawal' transaction_type",
                    severity=severity,
                    value=record.get("transaction_type"),
                )
            )

    def _rule_refund_limit(
        self,
        record: dict[str, Any],
        rule: dict[str, Any],
        name: str,
        severity: ValidationSeverity,
        target: list[ValidationError],
    ) -> None:
        amount = record.get("transaction_amount")
        if (
            record.get("transaction_type") == "refund"
            and isinstance(amount, (int, float))
            and amount > 50000
        ):
            target.append(
                ValidationError(
                    field="transaction_amount",
                    rule=name,
                    message=f"Refund amount {amount} exceeds limit of 50000",
                    severity=severity,
                    value=amount,
                )
            )

    def _rule_future_timestamp(
        self,
        record: dict[str, Any],
        rule: dict[str, Any],
        name: str,
        severity: ValidationSeverity,
        target: list[ValidationError],
    ) -> None:
        ts_str = record.get("transaction_timestamp")
        if not isinstance(ts_str, str):
            return
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            tolerance = timedelta(seconds=rule.get("future_tolerance_seconds", 300))
            now = datetime.now(timezone.utc)
            if ts > now + tolerance:
                target.append(
                    ValidationError(
                        field="transaction_timestamp",
                        rule=name,
                        message=f"Transaction timestamp {ts_str} is in the future",
                        severity=severity,
                        value=ts_str,
                    )
                )
        except (ValueError, AttributeError):
            pass  # Already caught by iso8601 format check

    def _rule_stale_timestamp(
        self,
        record: dict[str, Any],
        rule: dict[str, Any],
        name: str,
        severity: ValidationSeverity,
        target: list[ValidationError],
    ) -> None:
        ts_str = record.get("transaction_timestamp")
        if not isinstance(ts_str, str):
            return
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            max_age = timedelta(days=rule.get("max_age_days", 90))
            now = datetime.now(timezone.utc)
            if ts < now - max_age:
                target.append(
                    ValidationError(
                        field="transaction_timestamp",
                        rule=name,
                        message=f"Transaction timestamp {ts_str} is older than {rule.get('max_age_days', 90)} days",
                        severity=severity,
                        value=ts_str,
                    )
                )
        except (ValueError, AttributeError):
            pass

    # -------------------------------------------------------------------------
    # Custom business rules
    # -------------------------------------------------------------------------

    def _check_custom_rules(
        self,
        record: dict[str, Any],
        errors: list[ValidationError],
        warnings: list[ValidationError],
    ) -> None:
        custom_rules = self._rules.get("custom_rules", [])
        for rule in custom_rules:
            severity = ValidationSeverity(rule.get("severity", "warning"))
            target = errors if severity == ValidationSeverity.ERROR else warnings
            condition = rule.get("condition", "")
            name = rule["name"]

            if condition == "amount_is_round and amount > threshold":
                self._rule_round_amount(record, rule, name, severity, target)
            elif condition == "amount > threshold":
                self._rule_high_value(record, rule, name, severity, target)

    def _rule_round_amount(
        self,
        record: dict[str, Any],
        rule: dict[str, Any],
        name: str,
        severity: ValidationSeverity,
        target: list[ValidationError],
    ) -> None:
        amount = record.get("transaction_amount")
        threshold = rule.get("threshold", 1000)
        if isinstance(amount, (int, float)) and not isinstance(amount, bool):
            if amount > threshold and amount == int(amount):
                target.append(
                    ValidationError(
                        field="transaction_amount",
                        rule=name,
                        message=f"Suspicious round amount {amount} above threshold {threshold}",
                        severity=severity,
                        value=amount,
                    )
                )

    def _rule_high_value(
        self,
        record: dict[str, Any],
        rule: dict[str, Any],
        name: str,
        severity: ValidationSeverity,
        target: list[ValidationError],
    ) -> None:
        amount = record.get("transaction_amount")
        threshold = rule.get("threshold", 50000)
        if isinstance(amount, (int, float)) and not isinstance(amount, bool):
            if amount > threshold:
                target.append(
                    ValidationError(
                        field="transaction_amount",
                        rule=name,
                        message=f"High value transaction: {amount} exceeds threshold {threshold}",
                        severity=severity,
                        value=amount,
                    )
                )
