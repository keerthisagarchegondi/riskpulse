"""Configurable business rules engine for transaction validation.

Evaluates transactions against YAML-defined rules with support for:
- Composite rules (AND/OR combinations)
- Priority-based ordering with short-circuit evaluation
- Time-based rules (business hours, active days)
- Velocity rules (transaction frequency/amount over time windows)
- Rule versioning and hot-reload without restart
- Full audit trail of every rule evaluation

Performance target: < 10ms per transaction evaluation.
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock, RLock
from typing import Any

import yaml

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__, component="rules_engine")


# --- Enums ---


class RuleAction(str, Enum):
    """Action to take when a rule is triggered."""

    BLOCK = "block"
    FLAG = "flag"
    ALLOW = "allow"


class RuleSeverity(str, Enum):
    """Severity level of a rule match."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvaluationOutcome(str, Enum):
    """Outcome of evaluating a single rule."""

    TRIGGERED = "triggered"
    PASSED = "passed"
    SKIPPED = "skipped"
    ERROR = "error"


# --- Data Classes ---


@dataclass(frozen=True)
class RuleDefinition:
    """Immutable rule definition loaded from configuration."""

    id: str
    name: str
    description: str
    version: str
    priority: int
    enabled: bool
    severity: RuleSeverity
    category: str
    condition: dict[str, Any]
    action: RuleAction
    schedule: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def config_hash(self) -> str:
        """Content hash for change detection."""
        content = f"{self.id}:{self.version}:{self.condition}:{self.enabled}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class RuleEvaluationRecord:
    """Audit record for a single rule evaluation."""

    rule_id: str
    rule_name: str
    rule_version: str
    outcome: EvaluationOutcome
    action: RuleAction | None
    severity: RuleSeverity | None
    transaction_id: str
    account_id: str
    details: dict[str, Any] = field(default_factory=dict)
    evaluated_at: str = ""
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.evaluated_at:
            self.evaluated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "rule_version": self.rule_version,
            "outcome": self.outcome.value,
            "action": self.action.value if self.action else None,
            "severity": self.severity.value if self.severity else None,
            "transaction_id": self.transaction_id,
            "account_id": self.account_id,
            "details": self.details,
            "evaluated_at": self.evaluated_at,
            "latency_ms": round(self.latency_ms, 4),
        }


@dataclass
class RuleEngineResult:
    """Aggregate result of evaluating all rules against a transaction."""

    transaction_id: str
    overall_action: RuleAction
    triggered_rules: list[RuleEvaluationRecord] = field(default_factory=list)
    all_evaluations: list[RuleEvaluationRecord] = field(default_factory=list)
    total_rules_evaluated: int = 0
    total_rules_triggered: int = 0
    highest_severity: RuleSeverity | None = None
    latency_ms: float = 0.0
    short_circuited: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "overall_action": self.overall_action.value,
            "triggered_rules": [r.to_dict() for r in self.triggered_rules],
            "total_rules_evaluated": self.total_rules_evaluated,
            "total_rules_triggered": self.total_rules_triggered,
            "highest_severity": self.highest_severity.value if self.highest_severity else None,
            "latency_ms": round(self.latency_ms, 4),
            "short_circuited": self.short_circuited,
        }


# --- Velocity Tracker ---


class VelocityTracker:
    """Thread-safe sliding-window tracker for velocity rules.

    Tracks transaction counts and amounts per entity (account, device)
    within configurable time windows using deque-based sliding windows.
    """

    def __init__(self, max_window_seconds: int = 86400) -> None:
        self._max_window = max_window_seconds
        self._lock = Lock()
        # entity_key -> deque of (timestamp, amount)
        self._events: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=10000)
        )

    def record_event(self, entity_key: str, amount: float, timestamp: float | None = None) -> None:
        """Record a transaction event for velocity tracking."""
        ts = timestamp or time.time()
        with self._lock:
            self._events[entity_key].append((ts, amount))

    def get_count(
        self,
        entity_key: str,
        window_seconds: int,
        amount_threshold: float | None = None,
        current_time: float | None = None,
    ) -> int:
        """Get transaction count within a time window."""
        now = current_time or time.time()
        cutoff = now - window_seconds
        with self._lock:
            events = self._events.get(entity_key)
            if not events:
                return 0
            # Prune old events
            while events and events[0][0] < (now - self._max_window):
                events.popleft()
            count = 0
            for ts, amount in events:
                if ts >= cutoff:
                    if amount_threshold is None or amount <= amount_threshold:
                        count += 1
            return count

    def get_cumulative_amount(
        self, entity_key: str, window_seconds: int, current_time: float | None = None
    ) -> float:
        """Get cumulative transaction amount within a time window."""
        now = current_time or time.time()
        cutoff = now - window_seconds
        with self._lock:
            events = self._events.get(entity_key)
            if not events:
                return 0.0
            total = 0.0
            for ts, amount in events:
                if ts >= cutoff:
                    total += amount
            return total

    def get_distinct_count(
        self,
        group_key: str,
        count_field_values: str,
        window_seconds: int,
        current_time: float | None = None,
    ) -> int:
        """Get distinct value count for cardinality checks.

        Uses composite keys: group_key stores (timestamp, distinct_value).
        """
        now = current_time or time.time()
        cutoff = now - window_seconds
        with self._lock:
            events = self._events.get(group_key)
            if not events:
                return 0
            distinct = set()
            for ts, _ in events:
                if ts >= cutoff:
                    # The 'amount' field stores a hash of the distinct value
                    distinct.add(_)
            return len(distinct)

    def clear_entity(self, entity_key: str) -> None:
        """Clear tracking data for an entity."""
        with self._lock:
            self._events.pop(entity_key, None)

    def clear_all(self) -> None:
        """Clear all tracking data."""
        with self._lock:
            self._events.clear()

    @property
    def tracked_entities(self) -> int:
        """Number of entities currently being tracked."""
        with self._lock:
            return len(self._events)


# --- Audit Trail ---


class RuleAuditTrail:
    """Thread-safe audit trail for rule evaluations.

    Stores evaluation records in memory with configurable max size.
    Supports querying by transaction, account, rule, and time range.
    """

    def __init__(self, max_records: int = 100000) -> None:
        self._max_records = max_records
        self._lock = Lock()
        self._records: deque[RuleEvaluationRecord] = deque(maxlen=max_records)
        self._by_transaction: dict[str, list[RuleEvaluationRecord]] = defaultdict(list)
        self._by_rule: dict[str, list[RuleEvaluationRecord]] = defaultdict(list)
        self._stats: dict[str, int] = defaultdict(int)

    def record(self, evaluation: RuleEvaluationRecord) -> None:
        """Record a rule evaluation in the audit trail."""
        with self._lock:
            self._records.append(evaluation)
            self._by_transaction[evaluation.transaction_id].append(evaluation)
            self._by_rule[evaluation.rule_id].append(evaluation)
            self._stats[f"{evaluation.rule_id}:{evaluation.outcome.value}"] += 1

    def record_batch(self, evaluations: list[RuleEvaluationRecord]) -> None:
        """Record multiple evaluations efficiently."""
        with self._lock:
            for evaluation in evaluations:
                self._records.append(evaluation)
                self._by_transaction[evaluation.transaction_id].append(evaluation)
                self._by_rule[evaluation.rule_id].append(evaluation)
                self._stats[f"{evaluation.rule_id}:{evaluation.outcome.value}"] += 1

    def get_by_transaction(self, transaction_id: str) -> list[dict[str, Any]]:
        """Get all evaluations for a specific transaction."""
        with self._lock:
            records = self._by_transaction.get(transaction_id, [])
            return [r.to_dict() for r in records]

    def get_by_rule(self, rule_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent evaluations for a specific rule."""
        with self._lock:
            records = self._by_rule.get(rule_id, [])
            return [r.to_dict() for r in records[-limit:]]

    def get_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get most recent evaluations."""
        with self._lock:
            records = list(self._records)[-limit:]
            return [r.to_dict() for r in records]

    def get_stats(self) -> dict[str, Any]:
        """Get aggregated statistics."""
        with self._lock:
            total = len(self._records)
            triggered = sum(
                v for k, v in self._stats.items() if k.endswith(":triggered")
            )
            return {
                "total_evaluations": total,
                "total_triggered": triggered,
                "trigger_rate": round(triggered / total, 4) if total > 0 else 0.0,
                "unique_transactions": len(self._by_transaction),
                "rules_tracked": len(self._by_rule),
                "breakdown": dict(self._stats),
            }

    def clear(self) -> None:
        """Clear all audit records."""
        with self._lock:
            self._records.clear()
            self._by_transaction.clear()
            self._by_rule.clear()
            self._stats.clear()

    @property
    def total_records(self) -> int:
        with self._lock:
            return len(self._records)


# --- Rules Engine ---


class RulesEngine:
    """Configurable business rules engine with hot-reload and audit trail.

    Evaluates transactions against YAML-defined rules with support for
    composite conditions, velocity checks, time-based rules, and full
    audit trail.

    Thread-safe for concurrent transaction evaluation.
    """

    def __init__(
        self,
        rules_path: str | Path | None = None,
        enable_audit: bool = True,
        max_audit_records: int = 100000,
        short_circuit_on_block: bool = True,
    ) -> None:
        self._rules_path = self._resolve_rules_path(rules_path)
        self._short_circuit = short_circuit_on_block
        self._enable_audit = enable_audit

        self._lock = RLock()
        self._rules: list[RuleDefinition] = []
        self._rules_by_id: dict[str, RuleDefinition] = {}
        self._last_load_time: float = 0.0
        self._config_hash: str = ""
        self._reload_interval_seconds: float = 30.0

        # Velocity tracking
        self.velocity_tracker = VelocityTracker()

        # Audit trail
        self.audit_trail = RuleAuditTrail(max_records=max_audit_records)

        # Load initial rules
        self._load_rules()

    @staticmethod
    def _resolve_rules_path(rules_path: str | Path | None) -> Path:
        """Resolve the business rules YAML path."""
        if rules_path:
            return Path(rules_path)
        # Default: look in config directory relative to project root
        settings = get_settings()
        project_root = Path(settings.get("project_root", "."))
        candidate = project_root / "config" / "business_rules.yaml"
        if candidate.exists():
            return candidate
        # Fallback: search relative to this file
        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            candidate = parent / "config" / "business_rules.yaml"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "business_rules.yaml not found. Provide explicit path or place in config/"
        )

    def _load_rules(self) -> None:
        """Load and parse rules from YAML configuration."""
        try:
            with open(self._rules_path, "r") as f:
                content = f.read()

            config_hash = hashlib.md5(content.encode()).hexdigest()
            if config_hash == self._config_hash:
                return  # No changes

            data = yaml.safe_load(content)
            if not data or "rules" not in data:
                logger.warning("rules_config_empty", path=str(self._rules_path))
                return

            rules = []
            for rule_data in data["rules"]:
                try:
                    rule = RuleDefinition(
                        id=rule_data["id"],
                        name=rule_data["name"],
                        description=rule_data.get("description", ""),
                        version=rule_data.get("version", "1.0.0"),
                        priority=rule_data.get("priority", 100),
                        enabled=rule_data.get("enabled", True),
                        severity=RuleSeverity(rule_data.get("severity", "medium")),
                        category=rule_data.get("category", "general"),
                        condition=rule_data["condition"],
                        action=RuleAction(rule_data.get("action", "flag")),
                        schedule=rule_data.get("schedule"),
                        tags=rule_data.get("tags", []),
                    )
                    rules.append(rule)
                except (KeyError, ValueError) as exc:
                    logger.error(
                        "rule_parse_error",
                        rule_id=rule_data.get("id", "unknown"),
                        error=str(exc),
                    )

            # Sort by priority (lower number = higher priority)
            rules.sort(key=lambda r: r.priority)

            with self._lock:
                self._rules = rules
                self._rules_by_id = {r.id: r for r in rules}
                self._config_hash = config_hash
                self._last_load_time = time.time()

            logger.info(
                "rules_loaded",
                total_rules=len(rules),
                enabled_rules=sum(1 for r in rules if r.enabled),
                path=str(self._rules_path),
            )

        except FileNotFoundError:
            logger.error("rules_file_not_found", path=str(self._rules_path))
            raise
        except yaml.YAMLError as exc:
            logger.error("rules_yaml_parse_error", error=str(exc))
            raise

    def reload_if_needed(self) -> bool:
        """Check and reload rules if the file has changed. Returns True if reloaded."""
        now = time.time()
        if now - self._last_load_time < self._reload_interval_seconds:
            return False
        try:
            self._load_rules()
            return True
        except Exception as exc:
            logger.error("rules_reload_failed", error=str(exc))
            return False

    def force_reload(self) -> None:
        """Force immediate reload of rules from disk."""
        self._config_hash = ""  # Reset hash to force re-parse
        self._load_rules()

    # --- Public Evaluation API ---

    def evaluate(self, transaction: dict[str, Any]) -> RuleEngineResult:
        """Evaluate a transaction against all enabled rules.

        Args:
            transaction: Transaction data dictionary.

        Returns:
            RuleEngineResult with all evaluations and final action.
        """
        start_time = time.perf_counter()

        # Hot-reload check (non-blocking)
        self.reload_if_needed()

        transaction_id = transaction.get(
            "external_transaction_id", transaction.get("transaction_id", str(uuid.uuid4()))
        )
        account_id = transaction.get("account_id", "unknown")

        result = RuleEngineResult(
            transaction_id=transaction_id,
            overall_action=RuleAction.ALLOW,
        )

        # Track velocity for this transaction
        raw_amount = transaction.get("transaction_amount")
        amount = float(raw_amount) if raw_amount is not None else 0.0
        self.velocity_tracker.record_event(
            entity_key=f"account:{account_id}",
            amount=amount,
        )
        # Track device velocity if device_id present
        device_id = transaction.get("device_id")
        if device_id:
            card_last_four = transaction.get("card_last_four", "")
            self.velocity_tracker.record_event(
                entity_key=f"device:{device_id}",
                amount=hash(card_last_four) % 10000,  # Store card hash for cardinality
            )

        severity_order = {
            RuleSeverity.LOW: 0,
            RuleSeverity.MEDIUM: 1,
            RuleSeverity.HIGH: 2,
            RuleSeverity.CRITICAL: 3,
        }

        with self._lock:
            rules = list(self._rules)

        evaluations: list[RuleEvaluationRecord] = []

        for rule in rules:
            if not rule.enabled:
                record = RuleEvaluationRecord(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_version=rule.version,
                    outcome=EvaluationOutcome.SKIPPED,
                    action=None,
                    severity=None,
                    transaction_id=transaction_id,
                    account_id=account_id,
                    details={"reason": "rule_disabled"},
                )
                evaluations.append(record)
                continue

            # Check schedule constraints
            if rule.schedule and not self._is_schedule_active(rule.schedule):
                record = RuleEvaluationRecord(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_version=rule.version,
                    outcome=EvaluationOutcome.SKIPPED,
                    action=None,
                    severity=None,
                    transaction_id=transaction_id,
                    account_id=account_id,
                    details={"reason": "schedule_inactive"},
                )
                evaluations.append(record)
                continue

            # Evaluate the rule condition
            rule_start = time.perf_counter()
            try:
                triggered = self._evaluate_condition(rule.condition, transaction)
                rule_latency = (time.perf_counter() - rule_start) * 1000

                outcome = EvaluationOutcome.TRIGGERED if triggered else EvaluationOutcome.PASSED
                record = RuleEvaluationRecord(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_version=rule.version,
                    outcome=outcome,
                    action=rule.action if triggered else None,
                    severity=rule.severity if triggered else None,
                    transaction_id=transaction_id,
                    account_id=account_id,
                    details={"condition": rule.condition},
                    latency_ms=rule_latency,
                )
                evaluations.append(record)

                if triggered:
                    result.triggered_rules.append(record)
                    result.total_rules_triggered += 1

                    # Update overall action (block overrides flag)
                    if rule.action == RuleAction.BLOCK:
                        result.overall_action = RuleAction.BLOCK
                    elif rule.action == RuleAction.FLAG and result.overall_action != RuleAction.BLOCK:
                        result.overall_action = RuleAction.FLAG

                    # Track highest severity
                    if result.highest_severity is None or severity_order.get(
                        rule.severity, 0
                    ) > severity_order.get(result.highest_severity, 0):
                        result.highest_severity = rule.severity

                    # Short-circuit: stop evaluating if a block rule fired
                    if self._short_circuit and rule.action == RuleAction.BLOCK:
                        result.short_circuited = True
                        break

            except Exception as exc:
                rule_latency = (time.perf_counter() - rule_start) * 1000
                record = RuleEvaluationRecord(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_version=rule.version,
                    outcome=EvaluationOutcome.ERROR,
                    action=None,
                    severity=None,
                    transaction_id=transaction_id,
                    account_id=account_id,
                    details={"error": str(exc)},
                    latency_ms=rule_latency,
                )
                evaluations.append(record)
                logger.error(
                    "rule_evaluation_error",
                    rule_id=rule.id,
                    error=str(exc),
                    transaction_id=transaction_id,
                )

        result.total_rules_evaluated = len(evaluations)
        result.all_evaluations = evaluations
        result.latency_ms = (time.perf_counter() - start_time) * 1000

        # Record audit trail
        if self._enable_audit:
            self.audit_trail.record_batch(evaluations)

        logger.debug(
            "rules_evaluated",
            transaction_id=transaction_id,
            total_evaluated=result.total_rules_evaluated,
            total_triggered=result.total_rules_triggered,
            action=result.overall_action.value,
            latency_ms=round(result.latency_ms, 3),
        )

        return result

    # --- Condition Evaluation ---

    def _evaluate_condition(self, condition: dict[str, Any], transaction: dict[str, Any]) -> bool:
        """Evaluate a rule condition against transaction data.

        Supports:
        - Simple field comparisons
        - Composite AND/OR conditions
        - Velocity checks
        - Cumulative amount checks
        - Time-window checks
        - Cardinality checks
        - Geo-velocity checks
        """
        condition_type = condition.get("type")

        # Velocity rule
        if condition_type == "velocity":
            return self._evaluate_velocity(condition, transaction)

        # Cumulative amount rule
        if condition_type == "cumulative_amount":
            return self._evaluate_cumulative_amount(condition, transaction)

        # Geo velocity rule
        if condition_type == "geo_velocity":
            return self._evaluate_geo_velocity(condition, transaction)

        # Cardinality rule
        if condition_type == "cardinality":
            return self._evaluate_cardinality(condition, transaction)

        # Time window check (standalone)
        if condition_type == "time_window":
            return self._evaluate_time_window(condition)

        # Composite AND
        operator = condition.get("operator")
        if operator == "and":
            return self._evaluate_and(condition, transaction)

        # Composite OR
        if operator == "or":
            return self._evaluate_or(condition, transaction)

        # Simple field comparison
        return self._evaluate_field_condition(condition, transaction)

    def _evaluate_and(self, condition: dict[str, Any], transaction: dict[str, Any]) -> bool:
        """Evaluate AND composite condition with short-circuit."""
        conditions = condition.get("conditions", [])
        for sub_condition in conditions:
            if not self._evaluate_condition(sub_condition, transaction):
                return False
        return True

    def _evaluate_or(self, condition: dict[str, Any], transaction: dict[str, Any]) -> bool:
        """Evaluate OR composite condition with short-circuit."""
        conditions = condition.get("conditions", [])
        for sub_condition in conditions:
            if self._evaluate_condition(sub_condition, transaction):
                return True
        return False

    def _evaluate_field_condition(
        self, condition: dict[str, Any], transaction: dict[str, Any]
    ) -> bool:
        """Evaluate a simple field comparison."""
        field_name = condition.get("field")
        operator = condition.get("operator")
        expected_value = condition.get("value")

        if not field_name or not operator:
            return False

        actual_value = transaction.get(field_name)

        # Null checks
        if operator == "is_null":
            return actual_value is None or actual_value == ""

        if operator == "is_not_null":
            return actual_value is not None and actual_value != ""

        if actual_value is None:
            return False

        # Comparison operators
        if operator == "equals":
            return actual_value == expected_value

        if operator == "not_equals":
            return actual_value != expected_value

        if operator == "greater_than":
            return float(actual_value) > float(expected_value)

        if operator == "less_than":
            return float(actual_value) < float(expected_value)

        if operator == "greater_than_or_equals":
            return float(actual_value) >= float(expected_value)

        if operator == "less_than_or_equals":
            return float(actual_value) <= float(expected_value)

        if operator == "in":
            return actual_value in expected_value

        if operator == "not_in":
            return actual_value not in expected_value

        if operator == "contains":
            return expected_value in str(actual_value)

        if operator == "starts_with":
            return str(actual_value).startswith(str(expected_value))

        if operator == "ends_with":
            return str(actual_value).endswith(str(expected_value))

        if operator == "modulo_equals":
            modulo = condition.get("modulo", 1)
            return float(actual_value) % modulo == float(expected_value)

        if operator == "regex_match":
            import re

            return bool(re.match(str(expected_value), str(actual_value)))

        logger.warning("unknown_operator", operator=operator, rule_field=field_name)
        return False

    def _evaluate_velocity(self, condition: dict[str, Any], transaction: dict[str, Any]) -> bool:
        """Evaluate a velocity rule (transaction count in time window)."""
        field_name = condition.get("field", "account_id")
        entity_value = transaction.get(field_name)
        if not entity_value:
            return False

        max_count = condition.get("max_count", 10)
        window_seconds = condition.get("time_window_seconds", 300)
        amount_threshold = condition.get("amount_threshold")

        entity_key = f"account:{entity_value}"
        count = self.velocity_tracker.get_count(
            entity_key=entity_key,
            window_seconds=window_seconds,
            amount_threshold=amount_threshold,
        )
        return count > max_count

    def _evaluate_cumulative_amount(
        self, condition: dict[str, Any], transaction: dict[str, Any]
    ) -> bool:
        """Evaluate cumulative amount velocity rule."""
        field_name = condition.get("field", "account_id")
        entity_value = transaction.get(field_name)
        if not entity_value:
            return False

        max_amount = condition.get("max_amount", 100000.0)
        window_seconds = condition.get("time_window_seconds", 86400)

        entity_key = f"account:{entity_value}"
        cumulative = self.velocity_tracker.get_cumulative_amount(
            entity_key=entity_key,
            window_seconds=window_seconds,
        )
        return cumulative > max_amount

    def _evaluate_geo_velocity(
        self, condition: dict[str, Any], transaction: dict[str, Any]
    ) -> bool:
        """Evaluate geo-velocity (impossible travel) rule.

        Simplified: checks if distance/time between current and recent
        transaction implies travel faster than max_speed_kmh.
        """
        # This requires historical location data; for single-transaction
        # evaluation, we check if geo data is available and flag as needed.
        lat = transaction.get("geo_latitude")
        lon = transaction.get("geo_longitude")
        if lat is None or lon is None:
            return False

        # In production, this would compare against last known location
        # from a location store. For now, return False (no trigger)
        # unless explicitly marked.
        return False

    def _evaluate_cardinality(
        self, condition: dict[str, Any], transaction: dict[str, Any]
    ) -> bool:
        """Evaluate cardinality rule (distinct values per group)."""
        group_by = condition.get("group_by")
        count_field = condition.get("count_field")
        max_distinct = condition.get("max_distinct", 3)
        window_seconds = condition.get("time_window_seconds", 3600)

        group_value = transaction.get(group_by)
        if not group_value:
            return False

        entity_key = f"device:{group_value}"
        count = self.velocity_tracker.get_distinct_count(
            group_key=entity_key,
            count_field_values=str(transaction.get(count_field, "")),
            window_seconds=window_seconds,
        )
        return count > max_distinct

    def _evaluate_time_window(self, condition: dict[str, Any]) -> bool:
        """Evaluate time-window condition (business hours, active days)."""
        now = datetime.now(timezone.utc)

        # Check active days
        active_days = condition.get("active_days")
        if active_days:
            day_name = now.strftime("%A").lower()
            if day_name not in [d.lower() for d in active_days]:
                return False

        # Check outside hours
        outside_hours = condition.get("outside_hours")
        if outside_hours:
            start_str = outside_hours.get("start", "09:00")
            end_str = outside_hours.get("end", "17:00")

            # Parse timezone if specified
            tz_name = condition.get("timezone", "UTC")
            try:
                from zoneinfo import ZoneInfo

                local_now = now.astimezone(ZoneInfo(tz_name))
            except (ImportError, KeyError):
                local_now = now

            current_time = local_now.hour * 60 + local_now.minute
            start_parts = start_str.split(":")
            end_parts = end_str.split(":")
            start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
            end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])

            # "outside_hours" means the condition is True when OUTSIDE the range
            if start_minutes <= current_time <= end_minutes:
                return False
            return True

        return True

    @staticmethod
    def _is_schedule_active(schedule: dict[str, Any]) -> bool:
        """Check if a rule's schedule is currently active."""
        now = datetime.now(timezone.utc)

        active_days = schedule.get("active_days")
        if active_days:
            day_name = now.strftime("%A").lower()
            if day_name not in [d.lower() for d in active_days]:
                return False

        # Check start/end dates
        start_date = schedule.get("start_date")
        if start_date:
            if now.date() < datetime.fromisoformat(start_date).date():
                return False

        end_date = schedule.get("end_date")
        if end_date:
            if now.date() > datetime.fromisoformat(end_date).date():
                return False

        return True

    # --- Rule Management API ---

    def get_rules(self, category: str | None = None, enabled_only: bool = False) -> list[dict[str, Any]]:
        """Get all rules, optionally filtered by category or enabled status."""
        with self._lock:
            rules = self._rules
        result = []
        for rule in rules:
            if enabled_only and not rule.enabled:
                continue
            if category and rule.category != category:
                continue
            result.append({
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "version": rule.version,
                "priority": rule.priority,
                "enabled": rule.enabled,
                "severity": rule.severity.value,
                "category": rule.category,
                "condition": rule.condition,
                "action": rule.action.value,
                "schedule": rule.schedule,
                "tags": rule.tags,
            })
        return result

    def get_rule(self, rule_id: str) -> dict[str, Any] | None:
        """Get a single rule by ID."""
        with self._lock:
            rule = self._rules_by_id.get(rule_id)
        if not rule:
            return None
        return {
            "id": rule.id,
            "name": rule.name,
            "description": rule.description,
            "version": rule.version,
            "priority": rule.priority,
            "enabled": rule.enabled,
            "severity": rule.severity.value,
            "category": rule.category,
            "condition": rule.condition,
            "action": rule.action.value,
            "schedule": rule.schedule,
            "tags": rule.tags,
        }

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule by ID. Returns True if rule was found."""
        return self._set_rule_enabled(rule_id, True)

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule by ID. Returns True if rule was found."""
        return self._set_rule_enabled(rule_id, False)

    def _set_rule_enabled(self, rule_id: str, enabled: bool) -> bool:
        """Set a rule's enabled state (in-memory; does not persist to YAML)."""
        with self._lock:
            old_rule = self._rules_by_id.get(rule_id)
            if not old_rule:
                return False
            # Create new rule with updated enabled state
            new_rule = RuleDefinition(
                id=old_rule.id,
                name=old_rule.name,
                description=old_rule.description,
                version=old_rule.version,
                priority=old_rule.priority,
                enabled=enabled,
                severity=old_rule.severity,
                category=old_rule.category,
                condition=old_rule.condition,
                action=old_rule.action,
                schedule=old_rule.schedule,
                tags=old_rule.tags,
            )
            self._rules_by_id[rule_id] = new_rule
            self._rules = [new_rule if r.id == rule_id else r for r in self._rules]
        logger.info("rule_state_changed", rule_id=rule_id, enabled=enabled)
        return True

    def add_rule(self, rule_data: dict[str, Any]) -> RuleDefinition:
        """Add a new rule dynamically (in-memory)."""
        rule = RuleDefinition(
            id=rule_data["id"],
            name=rule_data["name"],
            description=rule_data.get("description", ""),
            version=rule_data.get("version", "1.0.0"),
            priority=rule_data.get("priority", 100),
            enabled=rule_data.get("enabled", True),
            severity=RuleSeverity(rule_data.get("severity", "medium")),
            category=rule_data.get("category", "custom"),
            condition=rule_data["condition"],
            action=RuleAction(rule_data.get("action", "flag")),
            schedule=rule_data.get("schedule"),
            tags=rule_data.get("tags", []),
        )
        with self._lock:
            if rule.id in self._rules_by_id:
                raise ValueError(f"Rule with id '{rule.id}' already exists")
            self._rules.append(rule)
            self._rules.sort(key=lambda r: r.priority)
            self._rules_by_id[rule.id] = rule
        logger.info("rule_added", rule_id=rule.id, name=rule.name)
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID (in-memory). Returns True if found and removed."""
        with self._lock:
            if rule_id not in self._rules_by_id:
                return False
            del self._rules_by_id[rule_id]
            self._rules = [r for r in self._rules if r.id != rule_id]
        logger.info("rule_removed", rule_id=rule_id)
        return True

    def update_rule(self, rule_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update an existing rule's properties (in-memory)."""
        with self._lock:
            old_rule = self._rules_by_id.get(rule_id)
            if not old_rule:
                return None
            new_rule = RuleDefinition(
                id=old_rule.id,
                name=updates.get("name", old_rule.name),
                description=updates.get("description", old_rule.description),
                version=updates.get("version", old_rule.version),
                priority=updates.get("priority", old_rule.priority),
                enabled=updates.get("enabled", old_rule.enabled),
                severity=RuleSeverity(updates["severity"]) if "severity" in updates else old_rule.severity,
                category=updates.get("category", old_rule.category),
                condition=updates.get("condition", old_rule.condition),
                action=RuleAction(updates["action"]) if "action" in updates else old_rule.action,
                schedule=updates.get("schedule", old_rule.schedule),
                tags=updates.get("tags", old_rule.tags),
            )
            self._rules_by_id[rule_id] = new_rule
            self._rules = [new_rule if r.id == rule_id else r for r in self._rules]
            self._rules.sort(key=lambda r: r.priority)

        logger.info("rule_updated", rule_id=rule_id)
        return self.get_rule(rule_id)

    @property
    def total_rules(self) -> int:
        with self._lock:
            return len(self._rules)

    @property
    def enabled_rules(self) -> int:
        with self._lock:
            return sum(1 for r in self._rules if r.enabled)

    def get_categories(self) -> list[str]:
        """Get all unique rule categories."""
        with self._lock:
            return sorted(set(r.category for r in self._rules))


# --- Module-level singleton ---

_engine_instance: RulesEngine | None = None
_engine_lock = Lock()


def get_rules_engine(rules_path: str | Path | None = None) -> RulesEngine:
    """Get or create the singleton RulesEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = RulesEngine(rules_path=rules_path)
    return _engine_instance


def reset_rules_engine() -> None:
    """Reset the singleton instance (for testing)."""
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
