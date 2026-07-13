"""Unit tests for the rule-based fraud detection engine.

Covers every production rule, severity escalation, confidence scoring,
backtesting, and engine metrics.
"""

from __future__ import annotations

import copy
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.fraud_detection.rule_engine import (
    FraudRuleEngine,
    RuleEvaluationResult,
    RuleMatch,
    RulePerformanceMetrics,
    _haversine_miles,
    _parse_timestamp,
    _safe_float,
    _safe_int,
)

# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    """Return a fresh FraudRuleEngine using the default fraud_rules.yaml."""
    return FraudRuleEngine()


@pytest.fixture
def base_transaction():
    """Minimal legitimate transaction that should trigger zero rules."""
    return {
        "external_transaction_id": "TXN-TEST-001",
        "account_id": "ACC-12345",
        "customer_id": "CUST-67890",
        "merchant_category_code": "5411",
        "transaction_amount": 50.00,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": "credit",
        "device_id": "device-known-1",
        "geo_latitude": 40.7128,
        "geo_longitude": -74.0060,
        "geo_country": "US",
        "is_international": False,
        "transaction_timestamp": "2026-07-01T14:30:00Z",
    }


@pytest.fixture
def base_context():
    """Context that should not trigger any rule."""
    return {
        "customer_avg_amount": 100.0,
        "customer_history_count": 50,
        "recent_transactions": [],
        "last_transaction": None,
        "customer_mcc_distribution": {"5411": 0.5, "5812": 0.3, "5541": 0.2},
        "customer_channels": {"online", "pos", "mobile"},
        "is_new_device": False,
        "device_age_days": 60.0,
        "accounts_on_device": 1,
        "is_domestic_only": False,
        "days_since_last_transaction": 2.0,
    }


# ── Helper Tests ─────────────────────────────────────────────────────

class TestHelpers:
    def test_safe_float_valid(self):
        assert _safe_float("123.45") == 123.45
        assert _safe_float(42) == 42.0

    def test_safe_float_invalid(self):
        assert _safe_float(None) == 0.0
        assert _safe_float("abc", 5.0) == 5.0

    def test_safe_int_valid(self):
        assert _safe_int("10") == 10
        assert _safe_int(7.9) == 7

    def test_safe_int_invalid(self):
        assert _safe_int(None) == 0
        assert _safe_int("xyz", 3) == 3

    def test_parse_timestamp_iso(self):
        dt = _parse_timestamp("2026-07-01T12:00:00Z")
        assert dt is not None
        assert dt.year == 2026
        assert dt.tzinfo is not None

    def test_parse_timestamp_with_tz(self):
        dt = _parse_timestamp("2026-07-01T12:00:00+00:00")
        assert dt is not None

    def test_parse_timestamp_none(self):
        assert _parse_timestamp(None) is None
        assert _parse_timestamp("not-a-date") is None

    def test_parse_timestamp_datetime_obj(self):
        now = datetime.now(timezone.utc)
        assert _parse_timestamp(now) == now

    def test_haversine_same_point(self):
        assert _haversine_miles(40.0, -74.0, 40.0, -74.0) == 0.0

    def test_haversine_known_distance(self):
        # NYC to London ≈ 3,459 miles
        dist = _haversine_miles(40.7128, -74.006, 51.5074, -0.1278)
        assert 3400 < dist < 3600


# ── Engine Initialization ────────────────────────────────────────────

class TestEngineInit:
    def test_loads_rules(self, engine):
        assert len(engine.rules) >= 17

    def test_rules_sorted_by_priority(self, engine):
        priorities = [r.priority for r in engine.rules]
        assert priorities == sorted(priorities)

    def test_all_rules_have_id(self, engine):
        for rule in engine.rules:
            assert rule.id.startswith("FRAUD-")

    def test_all_categories_present(self, engine):
        categories = {r.category for r in engine.rules}
        assert categories >= {"amount", "velocity", "geo", "pattern", "temporal"}

    def test_reload_rules(self, engine):
        original_count = len(engine.rules)
        engine.reload_rules()
        assert len(engine.rules) == original_count


# ── Clean Transaction (Baseline) ────────────────────────────────────

class TestCleanTransaction:
    def test_no_rules_triggered(self, engine, base_transaction, base_context):
        result = engine.evaluate(base_transaction, base_context)
        assert result.triggered_count == 0
        assert result.combined_severity == "low"
        assert result.rule_score == 0.0
        assert result.is_fraud_suspected is False

    def test_evaluation_time_recorded(self, engine, base_transaction, base_context):
        result = engine.evaluate(base_transaction, base_context)
        assert result.evaluation_time_ms >= 0


# ── FRAUD-AMT-001: High Amount vs Customer Average ──────────────────

class TestHighAmountVsAvg:
    def test_triggers_above_3x_avg(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 500.0
        base_context["customer_avg_amount"] = 100.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-AMT-001"]
        assert len(matched) == 1
        assert matched[0].severity == "high"

    def test_does_not_trigger_below_3x(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 250.0
        base_context["customer_avg_amount"] = 100.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-AMT-001"]
        assert len(matched) == 0

    def test_uses_fallback_when_low_history(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 2000.0
        base_context["customer_avg_amount"] = 1000.0
        base_context["customer_history_count"] = 2  # below min_history_count=5
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-AMT-001"]
        # fallback avg is 500, 2000 > 3*500 = 1500 → should trigger
        assert len(matched) == 1


# ── FRAUD-AMT-002: Amount Below Reporting Threshold ─────────────────

class TestStructuring:
    def test_triggers_just_below_10k(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 9500.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-AMT-002"]
        assert len(matched) == 1
        assert matched[0].severity == "critical"

    def test_does_not_trigger_at_10k(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 10000.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-AMT-002"]
        assert len(matched) == 0

    def test_does_not_trigger_below_lower_bound(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 8999.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-AMT-002"]
        assert len(matched) == 0


# ── FRAUD-AMT-003: Round Amount ─────────────────────────────────────

class TestRoundAmount:
    def test_triggers_on_round_amount(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 7000.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-AMT-003"]
        assert len(matched) == 1

    def test_does_not_trigger_non_round(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 7123.45
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-AMT-003"]
        assert len(matched) == 0

    def test_does_not_trigger_small_round(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 1000.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-AMT-003"]
        assert len(matched) == 0  # below min_amount=5000


# ── FRAUD-VEL-001: Rapid Successive Transactions ───────────────────

class TestRapidTransactions:
    def test_triggers_with_5_in_10_min(self, engine, base_transaction, base_context):
        base_ts = datetime(2026, 7, 1, 14, 30, 0, tzinfo=timezone.utc)
        recent = []
        for i in range(5):
            ts = base_ts - timedelta(minutes=i + 1)
            recent.append({
                "transaction_timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "transaction_amount": 25.0,
            })
        base_context["recent_transactions"] = recent
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-VEL-001"]
        assert len(matched) == 1

    def test_does_not_trigger_with_few_recent(self, engine, base_transaction, base_context):
        base_ts = datetime(2026, 7, 1, 14, 30, 0, tzinfo=timezone.utc)
        recent = [
            {"transaction_timestamp": (base_ts - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"), "transaction_amount": 10.0},
            {"transaction_timestamp": (base_ts - timedelta(minutes=8)).strftime("%Y-%m-%dT%H:%M:%SZ"), "transaction_amount": 10.0},
        ]
        base_context["recent_transactions"] = recent
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-VEL-001"]
        assert len(matched) == 0


# ── FRAUD-VEL-002: Declined Then Approved ───────────────────────────

class TestDeclinedThenApproved:
    def test_triggers_3_declines_then_approved(self, engine, base_transaction, base_context):
        base_transaction["status"] = "approved"
        base_ts = datetime(2026, 7, 1, 14, 30, 0, tzinfo=timezone.utc)
        recent = []
        for i in range(3):
            ts = base_ts - timedelta(minutes=i + 1)
            recent.append({
                "transaction_timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "status": "declined",
                "transaction_amount": 100.0,
            })
        base_context["recent_transactions"] = recent
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-VEL-002"]
        assert len(matched) == 1

    def test_does_not_trigger_if_not_approved(self, engine, base_transaction, base_context):
        base_transaction["status"] = "declined"
        base_context["recent_transactions"] = [
            {"transaction_timestamp": "2026-07-01T14:28:00Z", "status": "declined"},
            {"transaction_timestamp": "2026-07-01T14:26:00Z", "status": "declined"},
            {"transaction_timestamp": "2026-07-01T14:24:00Z", "status": "declined"},
        ]
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-VEL-002"]
        assert len(matched) == 0


# ── FRAUD-VEL-003: Escalating Amounts ──────────────────────────────

class TestEscalatingAmounts:
    def test_triggers_on_doubling_pattern(self, engine, base_transaction, base_context):
        base_ts = datetime(2026, 7, 1, 14, 30, 0, tzinfo=timezone.utc)
        base_transaction["transaction_amount"] = 800.0
        recent = [
            {"transaction_timestamp": (base_ts - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"), "transaction_amount": 100.0},
            {"transaction_timestamp": (base_ts - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"), "transaction_amount": 200.0},
        ]
        base_context["recent_transactions"] = recent
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-VEL-003"]
        assert len(matched) == 1

    def test_does_not_trigger_flat(self, engine, base_transaction, base_context):
        base_ts = datetime(2026, 7, 1, 14, 30, 0, tzinfo=timezone.utc)
        base_transaction["transaction_amount"] = 100.0
        recent = [
            {"transaction_timestamp": (base_ts - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"), "transaction_amount": 100.0},
            {"transaction_timestamp": (base_ts - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"), "transaction_amount": 100.0},
        ]
        base_context["recent_transactions"] = recent
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-VEL-003"]
        assert len(matched) == 0


# ── FRAUD-GEO-001: Impossible Travel ───────────────────────────────

class TestImpossibleTravel:
    def test_triggers_nyc_to_london_in_1h(self, engine, base_transaction, base_context):
        base_transaction["geo_latitude"] = 51.5074   # London
        base_transaction["geo_longitude"] = -0.1278
        base_transaction["transaction_timestamp"] = "2026-07-01T15:00:00Z"
        base_context["last_transaction"] = {
            "geo_latitude": 40.7128,   # NYC
            "geo_longitude": -74.006,
            "transaction_timestamp": "2026-07-01T14:00:00Z",
        }
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-GEO-001"]
        assert len(matched) == 1
        assert matched[0].severity == "critical"

    def test_does_not_trigger_reasonable_travel(self, engine, base_transaction, base_context):
        # NYC to Boston (~200 miles) in 4 hours → 50 mph
        base_transaction["geo_latitude"] = 42.3601
        base_transaction["geo_longitude"] = -71.0589
        base_transaction["transaction_timestamp"] = "2026-07-01T18:00:00Z"
        base_context["last_transaction"] = {
            "geo_latitude": 40.7128,
            "geo_longitude": -74.006,
            "transaction_timestamp": "2026-07-01T14:00:00Z",
        }
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-GEO-001"]
        assert len(matched) == 0

    def test_does_not_trigger_no_last_txn(self, engine, base_transaction, base_context):
        base_context["last_transaction"] = None
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-GEO-001"]
        assert len(matched) == 0


# ── FRAUD-GEO-002: International From Domestic-Only ────────────────

class TestInternationalDomestic:
    def test_triggers_international_on_domestic_account(self, engine, base_transaction, base_context):
        base_transaction["is_international"] = True
        base_transaction["geo_country"] = "GB"
        base_context["is_domestic_only"] = True
        base_context["customer_history_count"] = 50
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-GEO-002"]
        assert len(matched) == 1

    def test_does_not_trigger_domestic_txn(self, engine, base_transaction, base_context):
        base_transaction["is_international"] = False
        base_context["is_domestic_only"] = True
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-GEO-002"]
        assert len(matched) == 0


# ── FRAUD-GEO-003: High-Risk Country ──────────────────────────────

class TestHighRiskCountry:
    def test_triggers_on_russia(self, engine, base_transaction, base_context):
        base_transaction["geo_country"] = "RU"
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-GEO-003"]
        assert len(matched) == 1

    def test_triggers_on_north_korea(self, engine, base_transaction, base_context):
        base_transaction["geo_country"] = "KP"
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-GEO-003"]
        assert len(matched) == 1

    def test_does_not_trigger_on_us(self, engine, base_transaction, base_context):
        base_transaction["geo_country"] = "US"
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-GEO-003"]
        assert len(matched) == 0


# ── FRAUD-PAT-001: New Device + High Amount ────────────────────────

class TestNewDeviceHighAmount:
    def test_triggers_new_device_high_amount(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 2000.0
        base_context["is_new_device"] = True
        base_context["device_age_days"] = 0.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-PAT-001"]
        assert len(matched) == 1

    def test_does_not_trigger_known_device(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 2000.0
        base_context["is_new_device"] = False
        base_context["device_age_days"] = 30.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-PAT-001"]
        assert len(matched) == 0

    def test_does_not_trigger_new_device_low_amount(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 50.0
        base_context["is_new_device"] = True
        base_context["device_age_days"] = 0.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-PAT-001"]
        assert len(matched) == 0


# ── FRAUD-PAT-002: Merchant Category Mismatch ─────────────────────

class TestMerchantMismatch:
    def test_triggers_unusual_mcc(self, engine, base_transaction, base_context):
        base_transaction["merchant_category_code"] = "7995"  # gambling
        base_context["customer_mcc_distribution"] = {"5411": 0.5, "5812": 0.3, "5541": 0.2}
        base_context["customer_history_count"] = 50
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-PAT-002"]
        assert len(matched) == 1

    def test_does_not_trigger_common_mcc(self, engine, base_transaction, base_context):
        base_transaction["merchant_category_code"] = "5411"
        base_context["customer_mcc_distribution"] = {"5411": 0.5}
        base_context["customer_history_count"] = 50
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-PAT-002"]
        assert len(matched) == 0


# ── FRAUD-PAT-003: Card Testing ────────────────────────────────────

class TestCardTesting:
    def test_triggers_small_then_large(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 1500.0
        base_ts = datetime(2026, 7, 1, 14, 30, 0, tzinfo=timezone.utc)
        recent = []
        for i in range(4):
            ts = base_ts - timedelta(minutes=i + 1)
            recent.append({
                "transaction_timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "transaction_amount": round(1.0 + i * 0.5, 2),  # 1.0, 1.5, 2.0, 2.5
            })
        base_context["recent_transactions"] = recent
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-PAT-003"]
        assert len(matched) == 1
        assert matched[0].severity == "critical"

    def test_does_not_trigger_no_small_txns(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 1500.0
        base_context["recent_transactions"] = []
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-PAT-003"]
        assert len(matched) == 0


# ── FRAUD-PAT-004: Multi-Account Device ────────────────────────────

class TestMultiAccountDevice:
    def test_triggers_3_accounts(self, engine, base_transaction, base_context):
        base_context["accounts_on_device"] = 5
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-PAT-004"]
        assert len(matched) == 1

    def test_does_not_trigger_1_account(self, engine, base_transaction, base_context):
        base_context["accounts_on_device"] = 1
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-PAT-004"]
        assert len(matched) == 0


# ── FRAUD-PAT-005: Channel Anomaly ─────────────────────────────────

class TestChannelAnomaly:
    def test_triggers_unknown_channel(self, engine, base_transaction, base_context):
        base_transaction["channel"] = "atm"
        base_context["customer_channels"] = {"online", "pos"}
        base_context["customer_history_count"] = 50
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-PAT-005"]
        assert len(matched) == 1

    def test_does_not_trigger_known_channel(self, engine, base_transaction, base_context):
        base_transaction["channel"] = "online"
        base_context["customer_channels"] = {"online", "pos"}
        base_context["customer_history_count"] = 50
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-PAT-005"]
        assert len(matched) == 0


# ── FRAUD-TMP-001: Late Night High Value ──────────────────────────

class TestLateNightHighValue:
    def test_triggers_2am_high_value(self, engine, base_transaction, base_context):
        base_transaction["transaction_timestamp"] = "2026-07-01T02:30:00Z"
        base_transaction["transaction_amount"] = 5000.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-TMP-001"]
        assert len(matched) == 1

    def test_does_not_trigger_daytime(self, engine, base_transaction, base_context):
        base_transaction["transaction_timestamp"] = "2026-07-01T14:30:00Z"
        base_transaction["transaction_amount"] = 5000.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-TMP-001"]
        assert len(matched) == 0

    def test_does_not_trigger_late_night_low_amount(self, engine, base_transaction, base_context):
        base_transaction["transaction_timestamp"] = "2026-07-01T02:30:00Z"
        base_transaction["transaction_amount"] = 50.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-TMP-001"]
        assert len(matched) == 0


# ── FRAUD-TMP-002: Dormant Account ─────────────────────────────────

class TestDormantAccount:
    def test_triggers_90_day_dormant(self, engine, base_transaction, base_context):
        base_context["days_since_last_transaction"] = 120.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-TMP-002"]
        assert len(matched) == 1

    def test_does_not_trigger_active_account(self, engine, base_transaction, base_context):
        base_context["days_since_last_transaction"] = 5.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-TMP-002"]
        assert len(matched) == 0


# ── FRAUD-TMP-003: Burst After Silence ─────────────────────────────

class TestBurstAfterSilence:
    def test_triggers_burst_after_silence(self, engine, base_transaction, base_context):
        base_ts = datetime(2026, 7, 1, 14, 30, 0, tzinfo=timezone.utc)
        recent = []
        for i in range(5):
            ts = base_ts - timedelta(minutes=i + 1)
            recent.append({
                "transaction_timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "transaction_amount": 50.0,
            })
        base_context["recent_transactions"] = recent
        base_context["days_since_last_transaction"] = 5.0  # 120h > 48h silence
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-TMP-003"]
        assert len(matched) == 1

    def test_does_not_trigger_regular_activity(self, engine, base_transaction, base_context):
        base_context["recent_transactions"] = []
        base_context["days_since_last_transaction"] = 1.0
        result = engine.evaluate(base_transaction, base_context)
        matched = [r for r in result.triggered_rules if r.rule_id == "FRAUD-TMP-003"]
        assert len(matched) == 0


# ── Severity Escalation ─────────────────────────────────────────────

class TestSeverityEscalation:
    def test_single_rule_keeps_base_severity(self, engine, base_transaction, base_context):
        base_transaction["geo_country"] = "RU"
        result = engine.evaluate(base_transaction, base_context)
        # FRAUD-GEO-003 is severity "high"
        assert result.combined_severity in ("high", "critical")

    def test_multiple_rules_escalate(self, engine, base_transaction, base_context):
        # Trigger: high amount + high risk country + late night + structuring
        base_transaction["transaction_amount"] = 9500.0
        base_transaction["geo_country"] = "RU"
        base_transaction["transaction_timestamp"] = "2026-07-01T02:30:00Z"
        base_context["customer_avg_amount"] = 100.0
        result = engine.evaluate(base_transaction, base_context)
        assert result.triggered_count >= 3
        assert result.combined_severity == "critical"

    def test_escalation_caps_at_critical(self, engine, base_transaction, base_context):
        # Trigger as many rules as possible
        base_transaction["transaction_amount"] = 9500.0
        base_transaction["geo_country"] = "RU"
        base_transaction["is_international"] = True
        base_transaction["transaction_timestamp"] = "2026-07-01T02:30:00Z"
        base_context["customer_avg_amount"] = 50.0
        base_context["is_domestic_only"] = True
        base_context["is_new_device"] = True
        base_context["device_age_days"] = 0.0
        base_context["days_since_last_transaction"] = 180.0
        result = engine.evaluate(base_transaction, base_context)
        assert result.combined_severity == "critical"


# ── Confidence Scoring ──────────────────────────────────────────────

class TestConfidenceScoring:
    def test_zero_confidence_no_rules(self, engine, base_transaction, base_context):
        result = engine.evaluate(base_transaction, base_context)
        assert result.combined_confidence == 0.0

    def test_confidence_increases_with_rules(self, engine, base_transaction, base_context):
        # Single rule
        base_transaction["geo_country"] = "RU"
        single_result = engine.evaluate(base_transaction, base_context)

        # Multiple rules
        multi_txn = copy.deepcopy(base_transaction)
        multi_ctx = copy.deepcopy(base_context)
        multi_txn["transaction_amount"] = 9500.0
        multi_ctx["customer_avg_amount"] = 50.0
        multi_result = engine.evaluate(multi_txn, multi_ctx)

        assert multi_result.combined_confidence > single_result.combined_confidence

    def test_confidence_bounded_0_to_1(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 9500.0
        base_transaction["geo_country"] = "RU"
        base_transaction["transaction_timestamp"] = "2026-07-01T02:30:00Z"
        base_context["customer_avg_amount"] = 50.0
        base_context["is_new_device"] = True
        base_context["device_age_days"] = 0.0
        base_context["days_since_last_transaction"] = 180.0
        result = engine.evaluate(base_transaction, base_context)
        assert 0.0 <= result.combined_confidence <= 1.0


# ── Rule Score ──────────────────────────────────────────────────────

class TestRuleScore:
    def test_score_zero_no_rules(self, engine, base_transaction, base_context):
        result = engine.evaluate(base_transaction, base_context)
        assert result.rule_score == 0.0

    def test_score_bounded(self, engine, base_transaction, base_context):
        base_transaction["transaction_amount"] = 9500.0
        base_transaction["geo_country"] = "RU"
        base_context["customer_avg_amount"] = 50.0
        result = engine.evaluate(base_transaction, base_context)
        assert 0.0 <= result.rule_score <= 1.0

    def test_score_higher_with_diverse_categories(self, engine, base_transaction, base_context):
        # Single category
        single = copy.deepcopy(base_transaction)
        single_ctx = copy.deepcopy(base_context)
        single["transaction_amount"] = 9500.0
        single_ctx["customer_avg_amount"] = 50.0
        single_result = engine.evaluate(single, single_ctx)

        # Multiple categories
        multi = copy.deepcopy(base_transaction)
        multi_ctx = copy.deepcopy(base_context)
        multi["transaction_amount"] = 9500.0
        multi["geo_country"] = "RU"
        multi["transaction_timestamp"] = "2026-07-01T02:30:00Z"
        multi_ctx["customer_avg_amount"] = 50.0
        multi_ctx["is_new_device"] = True
        multi_ctx["device_age_days"] = 0.0
        multi_result = engine.evaluate(multi, multi_ctx)

        assert multi_result.rule_score >= single_result.rule_score


# ── Batch Evaluation ────────────────────────────────────────────────

class TestBatchEvaluation:
    def test_batch_returns_correct_count(self, engine, base_transaction, base_context):
        txns = [copy.deepcopy(base_transaction) for _ in range(5)]
        ctxs = [copy.deepcopy(base_context) for _ in range(5)]
        results = engine.evaluate_batch(txns, ctxs)
        assert len(results) == 5

    def test_batch_without_contexts(self, engine, base_transaction):
        txns = [copy.deepcopy(base_transaction) for _ in range(3)]
        results = engine.evaluate_batch(txns)
        assert len(results) == 3


# ── Result Serialization ────────────────────────────────────────────

class TestResultSerialization:
    def test_to_dict(self, engine, base_transaction, base_context):
        base_transaction["geo_country"] = "RU"
        result = engine.evaluate(base_transaction, base_context)
        d = result.to_dict()
        assert "transaction_id" in d
        assert "triggered_rules" in d
        assert isinstance(d["triggered_rules"], list)
        assert "combined_severity" in d
        assert "rule_score" in d


# ── Engine Metrics ──────────────────────────────────────────────────

class TestEngineMetrics:
    def test_metrics_increment(self, engine, base_transaction, base_context):
        engine.evaluate(base_transaction, base_context)
        engine.evaluate(base_transaction, base_context)
        m = engine.metrics
        assert m.transactions_evaluated >= 2

    def test_metrics_to_dict(self, engine, base_transaction, base_context):
        engine.evaluate(base_transaction, base_context)
        d = engine.metrics.to_dict()
        assert "transactions_evaluated" in d
        assert "avg_evaluation_time_ms" in d


# ── Backtesting ─────────────────────────────────────────────────────

class TestBacktesting:
    def test_backtest_returns_metrics(self, engine, base_transaction, base_context):
        legit = copy.deepcopy(base_transaction)
        fraud = copy.deepcopy(base_transaction)
        fraud["transaction_amount"] = 9500.0
        fraud["geo_country"] = "RU"

        fraud_ctx = copy.deepcopy(base_context)
        fraud_ctx["customer_avg_amount"] = 50.0

        metrics = engine.backtest(
            [legit, fraud],
            [False, True],
            [base_context, fraud_ctx],
        )
        assert len(metrics) > 0
        for perf in metrics.values():
            assert isinstance(perf, RulePerformanceMetrics)

    def test_backtest_precision_recall_range(self, engine, base_transaction, base_context):
        legit = copy.deepcopy(base_transaction)
        fraud = copy.deepcopy(base_transaction)
        fraud["transaction_amount"] = 9500.0

        fraud_ctx = copy.deepcopy(base_context)
        fraud_ctx["customer_avg_amount"] = 50.0

        metrics = engine.backtest(
            [legit, fraud],
            [False, True],
            [base_context, fraud_ctx],
        )
        for perf in metrics.values():
            assert 0.0 <= perf.precision <= 1.0
            assert 0.0 <= perf.recall <= 1.0
            assert 0.0 <= perf.f1_score <= 1.0

    def test_backtest_summary(self, engine, base_transaction, base_context):
        fraud = copy.deepcopy(base_transaction)
        fraud["transaction_amount"] = 9500.0
        fraud_ctx = copy.deepcopy(base_context)
        fraud_ctx["customer_avg_amount"] = 50.0

        engine.backtest([fraud], [True], [fraud_ctx])
        summary = engine.get_backtest_summary()
        assert isinstance(summary, dict)


# ── Performance ─────────────────────────────────────────────────────

class TestPerformance:
    def test_evaluation_under_20ms(self, engine, base_transaction, base_context):
        """Each evaluation should complete in < 20 ms."""
        base_transaction["transaction_amount"] = 9500.0
        base_transaction["geo_country"] = "RU"
        base_transaction["transaction_timestamp"] = "2026-07-01T02:30:00Z"
        base_context["customer_avg_amount"] = 50.0
        base_context["is_new_device"] = True
        base_context["device_age_days"] = 0.0
        base_context["days_since_last_transaction"] = 180.0

        # Warm up
        engine.evaluate(base_transaction, base_context)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            engine.evaluate(base_transaction, base_context)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_ms = sum(times) / len(times)
        p99_ms = sorted(times)[98]
        assert avg_ms < 20, f"Average evaluation time {avg_ms:.2f} ms exceeds 20 ms"
        assert p99_ms < 50, f"P99 evaluation time {p99_ms:.2f} ms exceeds 50 ms"


# ── RulePerformanceMetrics ──────────────────────────────────────────

class TestRulePerformanceMetrics:
    def test_precision_no_positives(self):
        m = RulePerformanceMetrics(rule_id="test")
        assert m.precision == 0.0

    def test_recall_no_positives(self):
        m = RulePerformanceMetrics(rule_id="test")
        assert m.recall == 0.0

    def test_f1_zero(self):
        m = RulePerformanceMetrics(rule_id="test")
        assert m.f1_score == 0.0

    def test_perfect_precision(self):
        m = RulePerformanceMetrics(rule_id="test", true_positives=10, false_positives=0)
        assert m.precision == 1.0

    def test_perfect_recall(self):
        m = RulePerformanceMetrics(rule_id="test", true_positives=10, false_negatives=0)
        assert m.recall == 1.0

    def test_to_dict(self):
        m = RulePerformanceMetrics(
            rule_id="test", true_positives=8, false_positives=2,
            true_negatives=90, false_negatives=0,
        )
        d = m.to_dict()
        assert d["rule_id"] == "test"
        assert d["precision"] == 0.8
        assert d["recall"] == 1.0
        assert d["total_evaluated"] == 100
