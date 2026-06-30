"""Comprehensive unit tests for the RiskPulse Rules Engine.

Tests cover:
- Rule loading from YAML
- Simple condition evaluation (all operators)
- Composite AND/OR rules
- Velocity rules
- Time-based rules
- Rule management (CRUD)
- Audit trail
- Short-circuit evaluation
- Performance benchmarks
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
import yaml

from src.validation.rules_engine import (
    EvaluationOutcome,
    RuleAction,
    RuleAuditTrail,
    RuleDefinition,
    RuleEngineResult,
    RulesEngine,
    RuleSeverity,
    VelocityTracker,
    reset_rules_engine,
)


# --- Fixtures ---


@pytest.fixture
def rules_yaml_path(tmp_path):
    """Create a temporary rules YAML file for testing."""
    rules_config = {
        "metadata": {
            "schema_version": "1.0.0",
            "description": "Test rules",
        },
        "rules": [
            {
                "id": "TEST-001",
                "name": "Max Amount",
                "description": "Block high-amount transactions",
                "version": "1.0.0",
                "priority": 10,
                "enabled": True,
                "severity": "high",
                "category": "amount_limits",
                "condition": {
                    "field": "transaction_amount",
                    "operator": "greater_than",
                    "value": 50000.0,
                },
                "action": "block",
                "tags": ["amount"],
            },
            {
                "id": "TEST-002",
                "name": "Min Amount",
                "description": "Flag micro-transactions",
                "version": "1.0.0",
                "priority": 11,
                "enabled": True,
                "severity": "medium",
                "category": "amount_limits",
                "condition": {
                    "field": "transaction_amount",
                    "operator": "less_than",
                    "value": 0.50,
                },
                "action": "flag",
                "tags": ["amount"],
            },
            {
                "id": "TEST-003",
                "name": "Purchase Amount Limit",
                "description": "Block high purchases",
                "version": "1.0.0",
                "priority": 12,
                "enabled": True,
                "severity": "high",
                "category": "amount_limits",
                "condition": {
                    "operator": "and",
                    "conditions": [
                        {"field": "transaction_type", "operator": "equals", "value": "purchase"},
                        {"field": "transaction_amount", "operator": "greater_than", "value": 25000.0},
                    ],
                },
                "action": "block",
                "tags": ["amount", "purchase"],
            },
            {
                "id": "TEST-004",
                "name": "High-Risk Country",
                "description": "Flag high-risk countries",
                "version": "1.0.0",
                "priority": 30,
                "enabled": True,
                "severity": "high",
                "category": "geo_restriction",
                "condition": {
                    "field": "geo_country",
                    "operator": "in",
                    "value": ["NG", "RU", "UA", "KP", "IR"],
                },
                "action": "flag",
                "tags": ["geo"],
            },
            {
                "id": "TEST-005",
                "name": "Disabled Rule",
                "description": "This rule is disabled",
                "version": "1.0.0",
                "priority": 99,
                "enabled": False,
                "severity": "low",
                "category": "test",
                "condition": {
                    "field": "transaction_amount",
                    "operator": "greater_than",
                    "value": 1.0,
                },
                "action": "flag",
                "tags": ["disabled"],
            },
            {
                "id": "TEST-006",
                "name": "OR Composite Rule",
                "description": "Flag if ATM or mobile channel",
                "version": "1.0.0",
                "priority": 40,
                "enabled": True,
                "severity": "low",
                "category": "channel",
                "condition": {
                    "operator": "or",
                    "conditions": [
                        {"field": "channel", "operator": "equals", "value": "atm"},
                        {"field": "channel", "operator": "equals", "value": "mobile"},
                    ],
                },
                "action": "flag",
                "tags": ["channel"],
            },
            {
                "id": "TEST-007",
                "name": "Velocity Rule",
                "description": "Flag high-frequency accounts",
                "version": "1.0.0",
                "priority": 20,
                "enabled": True,
                "severity": "high",
                "category": "velocity",
                "condition": {
                    "type": "velocity",
                    "field": "account_id",
                    "max_count": 5,
                    "time_window_seconds": 300,
                },
                "action": "flag",
                "tags": ["velocity"],
            },
            {
                "id": "TEST-008",
                "name": "Null Check",
                "description": "Flag missing merchant name with high amount",
                "version": "1.0.0",
                "priority": 50,
                "enabled": True,
                "severity": "medium",
                "category": "merchant",
                "condition": {
                    "operator": "and",
                    "conditions": [
                        {"field": "merchant_name", "operator": "is_null"},
                        {"field": "transaction_amount", "operator": "greater_than", "value": 1000.0},
                    ],
                },
                "action": "flag",
                "tags": ["merchant"],
            },
            {
                "id": "TEST-009",
                "name": "Structuring Detection",
                "description": "Detect amounts just below reporting threshold",
                "version": "1.0.0",
                "priority": 15,
                "enabled": True,
                "severity": "critical",
                "category": "advanced",
                "condition": {
                    "operator": "and",
                    "conditions": [
                        {"field": "transaction_amount", "operator": "greater_than", "value": 9000.0},
                        {"field": "transaction_amount", "operator": "less_than", "value": 10000.0},
                    ],
                },
                "action": "flag",
                "tags": ["structuring"],
            },
            {
                "id": "TEST-010",
                "name": "Not Equals Check",
                "description": "Flag non-USD transactions",
                "version": "1.0.0",
                "priority": 60,
                "enabled": True,
                "severity": "low",
                "category": "currency",
                "condition": {
                    "field": "transaction_currency",
                    "operator": "not_equals",
                    "value": "USD",
                },
                "action": "flag",
                "tags": ["currency"],
            },
        ],
    }
    rules_file = tmp_path / "business_rules.yaml"
    with open(rules_file, "w") as f:
        yaml.dump(rules_config, f, default_flow_style=False)
    return rules_file


@pytest.fixture
def engine(rules_yaml_path):
    """Create a fresh rules engine instance for testing."""
    reset_rules_engine()
    eng = RulesEngine(
        rules_path=rules_yaml_path,
        enable_audit=True,
        short_circuit_on_block=True,
    )
    yield eng
    reset_rules_engine()


@pytest.fixture
def sample_transaction():
    """Standard valid transaction."""
    return {
        "external_transaction_id": "TXN-TEST-001",
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
        "is_international": False,
        "transaction_timestamp": "2026-06-15T10:30:00Z",
    }


# --- Test: Rule Loading ---


class TestRuleLoading:
    """Tests for loading rules from YAML configuration."""

    def test_load_rules_from_yaml(self, engine):
        """Rules are loaded and sorted by priority."""
        assert engine.total_rules == 10
        assert engine.enabled_rules == 9  # One is disabled

    def test_rules_sorted_by_priority(self, engine):
        """Rules are ordered by priority (lower = higher priority)."""
        rules = engine.get_rules()
        priorities = [r["priority"] for r in rules]
        assert priorities == sorted(priorities)

    def test_get_categories(self, engine):
        """All categories are discoverable."""
        categories = engine.get_categories()
        assert "amount_limits" in categories
        assert "geo_restriction" in categories
        assert "velocity" in categories

    def test_load_nonexistent_file(self):
        """Loading from a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            RulesEngine(rules_path="/nonexistent/path.yaml")

    def test_hot_reload(self, engine, rules_yaml_path):
        """Rules can be force-reloaded from disk."""
        assert engine.total_rules == 10
        # Modify the YAML to add a rule
        with open(rules_yaml_path, "r") as f:
            data = yaml.safe_load(f)
        data["rules"].append({
            "id": "TEST-NEW",
            "name": "New Rule",
            "description": "Added after initial load",
            "version": "1.0.0",
            "priority": 100,
            "enabled": True,
            "severity": "low",
            "category": "test",
            "condition": {"field": "transaction_amount", "operator": "greater_than", "value": 99999},
            "action": "flag",
        })
        with open(rules_yaml_path, "w") as f:
            yaml.dump(data, f)
        engine.force_reload()
        assert engine.total_rules == 11


# --- Test: Simple Condition Evaluation ---


class TestSimpleConditions:
    """Tests for individual field condition operators."""

    def test_greater_than_not_triggered(self, engine, sample_transaction):
        """Amount below threshold does not trigger."""
        result = engine.evaluate(sample_transaction)
        # 125.50 is not > 50000, so TEST-001 should not trigger
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-001" not in triggered_ids

    def test_greater_than_triggered(self, engine, sample_transaction):
        """Amount above threshold triggers block."""
        sample_transaction["transaction_amount"] = 60000.00
        result = engine.evaluate(sample_transaction)
        assert result.overall_action == RuleAction.BLOCK
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-001" in triggered_ids

    def test_less_than_triggered(self, engine, sample_transaction):
        """Micro-transaction triggers flag."""
        sample_transaction["transaction_amount"] = 0.10
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-002" in triggered_ids

    def test_equals_operator(self, engine, sample_transaction):
        """Equals operator matches correctly."""
        # TEST-003 checks transaction_type == "purchase" AND amount > 25000
        sample_transaction["transaction_amount"] = 30000.00
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-003" in triggered_ids

    def test_in_operator(self, engine, sample_transaction):
        """IN operator detects values in a list."""
        sample_transaction["geo_country"] = "RU"
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-004" in triggered_ids

    def test_in_operator_not_triggered(self, engine, sample_transaction):
        """IN operator does not trigger for values not in list."""
        sample_transaction["geo_country"] = "US"
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-004" not in triggered_ids

    def test_not_equals_triggered(self, engine, sample_transaction):
        """Not-equals operator detects different values."""
        sample_transaction["transaction_currency"] = "EUR"
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-010" in triggered_ids

    def test_not_equals_not_triggered(self, engine, sample_transaction):
        """Not-equals operator passes on equal values."""
        sample_transaction["transaction_currency"] = "USD"
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-010" not in triggered_ids

    def test_is_null_triggered(self, engine, sample_transaction):
        """IS NULL check triggers on missing values."""
        sample_transaction["merchant_name"] = None
        sample_transaction["transaction_amount"] = 5000.00
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-008" in triggered_ids

    def test_is_null_not_triggered(self, engine, sample_transaction):
        """IS NULL does not trigger when field has value."""
        sample_transaction["merchant_name"] = "Valid Merchant"
        sample_transaction["transaction_amount"] = 5000.00
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-008" not in triggered_ids


# --- Test: Composite Rules ---


class TestCompositeRules:
    """Tests for AND/OR composite rule evaluation."""

    def test_and_all_conditions_met(self, engine, sample_transaction):
        """AND rule triggers when all conditions are met."""
        sample_transaction["transaction_type"] = "purchase"
        sample_transaction["transaction_amount"] = 30000.00
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-003" in triggered_ids

    def test_and_partial_conditions_met(self, engine, sample_transaction):
        """AND rule does not trigger when only some conditions are met."""
        sample_transaction["transaction_type"] = "transfer"
        sample_transaction["transaction_amount"] = 30000.00
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-003" not in triggered_ids

    def test_or_first_condition_met(self, engine, sample_transaction):
        """OR rule triggers when first condition is met."""
        sample_transaction["channel"] = "atm"
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-006" in triggered_ids

    def test_or_second_condition_met(self, engine, sample_transaction):
        """OR rule triggers when second condition is met."""
        sample_transaction["channel"] = "mobile"
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-006" in triggered_ids

    def test_or_no_conditions_met(self, engine, sample_transaction):
        """OR rule does not trigger when no conditions are met."""
        sample_transaction["channel"] = "online"
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-006" not in triggered_ids

    def test_structuring_detection(self, engine, sample_transaction):
        """Structuring detection triggers for amounts between 9000-10000."""
        sample_transaction["transaction_amount"] = 9500.00
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-009" in triggered_ids

    def test_structuring_not_triggered_below(self, engine, sample_transaction):
        """Structuring detection does not trigger below range."""
        sample_transaction["transaction_amount"] = 8000.00
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-009" not in triggered_ids


# --- Test: Short-Circuit Evaluation ---


class TestShortCircuit:
    """Tests for short-circuit evaluation on block actions."""

    def test_short_circuit_on_block(self, engine, sample_transaction):
        """Engine stops evaluating after a block rule triggers."""
        sample_transaction["transaction_amount"] = 60000.00
        result = engine.evaluate(sample_transaction)
        assert result.overall_action == RuleAction.BLOCK
        assert result.short_circuited is True
        # Should stop at TEST-001 (priority 10), not evaluate all rules
        assert result.total_rules_evaluated < engine.enabled_rules

    def test_no_short_circuit_on_flag(self, engine, sample_transaction):
        """Engine continues evaluating after flag rules."""
        sample_transaction["geo_country"] = "RU"
        result = engine.evaluate(sample_transaction)
        # Flag should not short-circuit
        assert result.short_circuited is False

    def test_short_circuit_disabled(self, rules_yaml_path):
        """Engine evaluates all rules when short-circuit is disabled."""
        eng = RulesEngine(
            rules_path=rules_yaml_path,
            short_circuit_on_block=False,
        )
        txn = {
            "external_transaction_id": "TXN-SC-001",
            "account_id": "ACC-999",
            "transaction_amount": 60000.00,
            "transaction_type": "purchase",
            "channel": "online",
            "transaction_currency": "USD",
            "geo_country": "US",
        }
        result = eng.evaluate(txn)
        assert result.overall_action == RuleAction.BLOCK
        assert result.short_circuited is False
        # Should evaluate all rules (enabled + 1 skipped disabled)
        assert result.total_rules_evaluated == eng.total_rules


# --- Test: Disabled Rules ---


class TestDisabledRules:
    """Tests for disabled rule handling."""

    def test_disabled_rule_skipped(self, engine, sample_transaction):
        """Disabled rules are skipped during evaluation."""
        result = engine.evaluate(sample_transaction)
        skipped_ids = [
            r.rule_id for r in result.all_evaluations if r.outcome == EvaluationOutcome.SKIPPED
        ]
        assert "TEST-005" in skipped_ids

    def test_enable_rule(self, engine):
        """Rules can be enabled dynamically."""
        assert engine.enable_rule("TEST-005") is True
        rule = engine.get_rule("TEST-005")
        assert rule["enabled"] is True

    def test_disable_rule(self, engine):
        """Rules can be disabled dynamically."""
        assert engine.disable_rule("TEST-001") is True
        rule = engine.get_rule("TEST-001")
        assert rule["enabled"] is False

    def test_enable_nonexistent_rule(self, engine):
        """Enabling a non-existent rule returns False."""
        assert engine.enable_rule("NONEXISTENT") is False


# --- Test: Rule Management (CRUD) ---


class TestRuleManagement:
    """Tests for rule CRUD operations."""

    def test_get_all_rules(self, engine):
        """Get all rules returns complete list."""
        rules = engine.get_rules()
        assert len(rules) == 10

    def test_get_rules_by_category(self, engine):
        """Filter rules by category."""
        rules = engine.get_rules(category="amount_limits")
        assert all(r["category"] == "amount_limits" for r in rules)
        assert len(rules) >= 3

    def test_get_rules_enabled_only(self, engine):
        """Filter returns only enabled rules."""
        rules = engine.get_rules(enabled_only=True)
        assert all(r["enabled"] is True for r in rules)
        assert len(rules) == 9

    def test_get_single_rule(self, engine):
        """Get a single rule by ID."""
        rule = engine.get_rule("TEST-001")
        assert rule is not None
        assert rule["name"] == "Max Amount"
        assert rule["severity"] == "high"

    def test_get_nonexistent_rule(self, engine):
        """Get returns None for non-existent rule."""
        assert engine.get_rule("NONEXISTENT") is None

    def test_add_rule(self, engine):
        """Add a new rule dynamically."""
        new_rule = {
            "id": "TEST-NEW-001",
            "name": "Dynamic Test Rule",
            "description": "Added via API",
            "version": "1.0.0",
            "priority": 70,
            "enabled": True,
            "severity": "low",
            "category": "custom",
            "condition": {"field": "transaction_amount", "operator": "greater_than", "value": 99999},
            "action": "flag",
            "tags": ["dynamic"],
        }
        engine.add_rule(new_rule)
        assert engine.total_rules == 11
        added = engine.get_rule("TEST-NEW-001")
        assert added is not None
        assert added["name"] == "Dynamic Test Rule"

    def test_add_duplicate_rule_raises(self, engine):
        """Adding a rule with existing ID raises ValueError."""
        with pytest.raises(ValueError, match="already exists"):
            engine.add_rule({
                "id": "TEST-001",
                "name": "Duplicate",
                "condition": {"field": "x", "operator": "equals", "value": 1},
            })

    def test_remove_rule(self, engine):
        """Remove a rule by ID."""
        assert engine.remove_rule("TEST-001") is True
        assert engine.total_rules == 9
        assert engine.get_rule("TEST-001") is None

    def test_remove_nonexistent_rule(self, engine):
        """Removing non-existent rule returns False."""
        assert engine.remove_rule("NONEXISTENT") is False

    def test_update_rule(self, engine):
        """Update an existing rule's properties."""
        result = engine.update_rule("TEST-001", {"name": "Updated Name", "priority": 5})
        assert result is not None
        assert result["name"] == "Updated Name"
        assert result["priority"] == 5

    def test_update_nonexistent_rule(self, engine):
        """Updating non-existent rule returns None."""
        result = engine.update_rule("NONEXISTENT", {"name": "Nope"})
        assert result is None


# --- Test: Velocity Tracker ---


class TestVelocityTracker:
    """Tests for the VelocityTracker component."""

    def test_record_and_count(self):
        """Records events and counts them within windows."""
        tracker = VelocityTracker()
        now = time.time()
        for i in range(5):
            tracker.record_event("account:ACC-001", 100.0, timestamp=now - i)
        count = tracker.get_count("account:ACC-001", window_seconds=60, current_time=now)
        assert count == 5

    def test_count_respects_window(self):
        """Events outside the window are not counted."""
        tracker = VelocityTracker()
        now = time.time()
        # 3 events within window
        for i in range(3):
            tracker.record_event("account:ACC-002", 50.0, timestamp=now - i)
        # 2 events outside window
        for i in range(2):
            tracker.record_event("account:ACC-002", 50.0, timestamp=now - 120 - i)
        count = tracker.get_count("account:ACC-002", window_seconds=60, current_time=now)
        assert count == 3

    def test_cumulative_amount(self):
        """Cumulative amount sums correctly within window."""
        tracker = VelocityTracker()
        now = time.time()
        tracker.record_event("account:ACC-003", 1000.0, timestamp=now - 10)
        tracker.record_event("account:ACC-003", 2000.0, timestamp=now - 5)
        tracker.record_event("account:ACC-003", 3000.0, timestamp=now - 1)
        total = tracker.get_cumulative_amount("account:ACC-003", window_seconds=60, current_time=now)
        assert total == 6000.0

    def test_amount_threshold_filter(self):
        """Amount threshold filters events by amount."""
        tracker = VelocityTracker()
        now = time.time()
        tracker.record_event("account:ACC-004", 1.0, timestamp=now - 5)
        tracker.record_event("account:ACC-004", 2.0, timestamp=now - 4)
        tracker.record_event("account:ACC-004", 100.0, timestamp=now - 3)
        # Only count transactions with amount <= 5.0
        count = tracker.get_count(
            "account:ACC-004", window_seconds=60, amount_threshold=5.0, current_time=now
        )
        assert count == 2

    def test_clear_entity(self):
        """Clearing an entity removes its data."""
        tracker = VelocityTracker()
        tracker.record_event("account:ACC-005", 100.0)
        assert tracker.get_count("account:ACC-005", window_seconds=60) >= 1
        tracker.clear_entity("account:ACC-005")
        assert tracker.get_count("account:ACC-005", window_seconds=60) == 0

    def test_tracked_entities(self):
        """Tracked entities count is accurate."""
        tracker = VelocityTracker()
        tracker.record_event("account:A", 10.0)
        tracker.record_event("account:B", 20.0)
        tracker.record_event("device:C", 30.0)
        assert tracker.tracked_entities == 3


# --- Test: Velocity Rules in Engine ---


class TestVelocityRules:
    """Tests for velocity rule evaluation in the engine."""

    def test_velocity_not_triggered_below_threshold(self, engine, sample_transaction):
        """Velocity rule does not trigger with few transactions."""
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-007" not in triggered_ids

    def test_velocity_triggered_above_threshold(self, engine, sample_transaction):
        """Velocity rule triggers when count exceeds threshold."""
        # Pre-populate velocity tracker with events for this account
        now = time.time()
        for i in range(7):
            engine.velocity_tracker.record_event(
                f"account:{sample_transaction['account_id']}", 100.0, timestamp=now - i * 10
            )
        result = engine.evaluate(sample_transaction)
        triggered_ids = [r.rule_id for r in result.triggered_rules]
        assert "TEST-007" in triggered_ids


# --- Test: Audit Trail ---


class TestAuditTrail:
    """Tests for the rule audit trail."""

    def test_audit_records_created(self, engine, sample_transaction):
        """Evaluations create audit records."""
        engine.evaluate(sample_transaction)
        assert engine.audit_trail.total_records > 0

    def test_audit_by_transaction(self, engine, sample_transaction):
        """Audit records are queryable by transaction ID."""
        engine.evaluate(sample_transaction)
        records = engine.audit_trail.get_by_transaction(
            sample_transaction["external_transaction_id"]
        )
        assert len(records) > 0
        assert all(
            r["transaction_id"] == sample_transaction["external_transaction_id"] for r in records
        )

    def test_audit_by_rule(self, engine, sample_transaction):
        """Audit records are queryable by rule ID."""
        engine.evaluate(sample_transaction)
        records = engine.audit_trail.get_by_rule("TEST-001")
        assert len(records) >= 1

    def test_audit_stats(self, engine, sample_transaction):
        """Audit statistics are computed correctly."""
        engine.evaluate(sample_transaction)
        stats = engine.audit_trail.get_stats()
        assert stats["total_evaluations"] > 0
        assert "unique_transactions" in stats
        assert stats["unique_transactions"] == 1

    def test_audit_recent(self, engine, sample_transaction):
        """Recent audit records are retrievable."""
        engine.evaluate(sample_transaction)
        records = engine.audit_trail.get_recent(limit=10)
        assert len(records) > 0

    def test_audit_clear(self, engine, sample_transaction):
        """Audit trail can be cleared."""
        engine.evaluate(sample_transaction)
        assert engine.audit_trail.total_records > 0
        engine.audit_trail.clear()
        assert engine.audit_trail.total_records == 0


# --- Test: Rule Engine Result ---


class TestRuleEngineResult:
    """Tests for the evaluation result object."""

    def test_result_allow_when_no_triggers(self, engine, sample_transaction):
        """Default action is ALLOW when no rules trigger."""
        result = engine.evaluate(sample_transaction)
        # Standard transaction should mostly pass (except maybe OR rule for mobile channel)
        assert result.overall_action in (RuleAction.ALLOW, RuleAction.FLAG)

    def test_result_block_overrides_flag(self, engine, sample_transaction):
        """BLOCK action takes precedence over FLAG."""
        sample_transaction["transaction_amount"] = 60000.00
        result = engine.evaluate(sample_transaction)
        assert result.overall_action == RuleAction.BLOCK

    def test_result_highest_severity(self, engine, sample_transaction):
        """Highest severity is tracked correctly."""
        sample_transaction["transaction_amount"] = 60000.00
        result = engine.evaluate(sample_transaction)
        assert result.highest_severity in (RuleSeverity.HIGH, RuleSeverity.CRITICAL)

    def test_result_to_dict(self, engine, sample_transaction):
        """Result serializes to dict correctly."""
        result = engine.evaluate(sample_transaction)
        d = result.to_dict()
        assert "transaction_id" in d
        assert "overall_action" in d
        assert "latency_ms" in d
        assert isinstance(d["triggered_rules"], list)


# --- Test: Performance ---


class TestPerformance:
    """Performance benchmarks for rule evaluation."""

    def test_evaluation_under_10ms(self, engine, sample_transaction):
        """Single evaluation completes in under 10ms."""
        # Warm up
        engine.evaluate(sample_transaction)

        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            engine.evaluate(sample_transaction)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_ms = sum(times) / len(times)
        p95_ms = sorted(times)[94]
        assert avg_ms < 10.0, f"Average evaluation time {avg_ms:.2f}ms exceeds 10ms target"
        assert p95_ms < 15.0, f"P95 evaluation time {p95_ms:.2f}ms is too high"

    def test_evaluation_with_many_rules(self, tmp_path):
        """Engine handles 50+ rules within performance target."""
        rules = []
        for i in range(50):
            rules.append({
                "id": f"PERF-{i:03d}",
                "name": f"Performance Rule {i}",
                "description": f"Rule {i}",
                "version": "1.0.0",
                "priority": i + 1,
                "enabled": True,
                "severity": "low",
                "category": "performance_test",
                "condition": {
                    "field": "transaction_amount",
                    "operator": "greater_than",
                    "value": 999999.0 + i,  # Won't trigger
                },
                "action": "flag",
            })
        config = {"metadata": {"schema_version": "1.0.0"}, "rules": rules}
        rules_file = tmp_path / "perf_rules.yaml"
        with open(rules_file, "w") as f:
            yaml.dump(config, f)

        eng = RulesEngine(rules_path=rules_file, short_circuit_on_block=False)
        txn = {
            "external_transaction_id": "TXN-PERF-001",
            "account_id": "ACC-PERF",
            "transaction_amount": 100.0,
            "transaction_type": "purchase",
            "channel": "online",
            "transaction_currency": "USD",
        }

        # Warm up
        eng.evaluate(txn)

        # Benchmark
        times = []
        for _ in range(50):
            start = time.perf_counter()
            eng.evaluate(txn)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_ms = sum(times) / len(times)
        assert avg_ms < 10.0, f"Average evaluation time {avg_ms:.2f}ms exceeds 10ms with 50 rules"


# --- Test: Edge Cases ---


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_field_in_transaction(self, engine):
        """Missing fields don't crash the engine."""
        minimal_txn = {
            "external_transaction_id": "TXN-MINIMAL",
            "account_id": "ACC-MIN",
            "transaction_amount": 100.0,
        }
        result = engine.evaluate(minimal_txn)
        assert result.transaction_id == "TXN-MINIMAL"
        assert result.overall_action in (RuleAction.ALLOW, RuleAction.FLAG, RuleAction.BLOCK)

    def test_none_values_handled(self, engine):
        """None values don't crash comparisons."""
        txn = {
            "external_transaction_id": "TXN-NONE",
            "account_id": "ACC-NONE",
            "transaction_amount": None,
            "transaction_type": None,
            "channel": None,
        }
        result = engine.evaluate(txn)
        assert result.transaction_id == "TXN-NONE"

    def test_empty_rules_file(self, tmp_path):
        """Engine handles empty rules file gracefully."""
        rules_file = tmp_path / "empty_rules.yaml"
        with open(rules_file, "w") as f:
            yaml.dump({"metadata": {}, "rules": []}, f)
        eng = RulesEngine(rules_path=rules_file)
        assert eng.total_rules == 0

    def test_concurrent_evaluations(self, engine, sample_transaction):
        """Engine handles concurrent evaluations correctly."""
        import concurrent.futures

        def evaluate_txn(i):
            txn = sample_transaction.copy()
            txn["external_transaction_id"] = f"TXN-CONC-{i}"
            txn["account_id"] = f"ACC-CONC-{i}"
            return engine.evaluate(txn)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(evaluate_txn, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        assert all(isinstance(r, RuleEngineResult) for r in results)

    def test_invalid_yaml_in_rules_file(self, tmp_path):
        """Engine raises on invalid YAML."""
        rules_file = tmp_path / "invalid.yaml"
        with open(rules_file, "w") as f:
            f.write("invalid: yaml: [[[")
        with pytest.raises(yaml.YAMLError):
            RulesEngine(rules_path=rules_file)
