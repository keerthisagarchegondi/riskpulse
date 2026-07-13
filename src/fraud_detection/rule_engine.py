"""Production rule-based fraud detection engine.

Evaluates configurable YAML-driven rules against enriched transactions,
produces per-rule confidence scores, and escalates severity when multiple
rules fire.  Designed for < 20 ms per-transaction latency.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

import yaml

from src.utils.constants import (
    IMPOSSIBLE_TRAVEL_SPEED_MPH,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, component="rule_engine")

_SEVERITY_ORDER: Dict[str, int] = {
    SEVERITY_LOW: 0,
    SEVERITY_MEDIUM: 1,
    SEVERITY_HIGH: 2,
    SEVERITY_CRITICAL: 3,
}

_SEVERITY_BY_INDEX: Dict[int, str] = {v: k for k, v in _SEVERITY_ORDER.items()}

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_DEFAULT_RULES_PATH = _CONFIG_DIR / "fraud_rules.yaml"


# ── Data Classes ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class RuleMatch:
    """A single rule that fired against a transaction."""

    rule_id: str
    rule_name: str
    category: str
    severity: str
    confidence: float
    details: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleEvaluationResult:
    """Aggregated result of evaluating all rules against one transaction."""

    transaction_id: str
    triggered_rules: List[RuleMatch] = field(default_factory=list)
    combined_severity: str = SEVERITY_LOW
    combined_confidence: float = 0.0
    rule_score: float = 0.0  # 0.0 – 1.0
    evaluation_time_ms: float = 0.0
    total_rules_evaluated: int = 0

    @property
    def is_fraud_suspected(self) -> bool:
        return len(self.triggered_rules) > 0

    @property
    def triggered_count(self) -> int:
        return len(self.triggered_rules)

    @property
    def triggered_categories(self) -> set[str]:
        return {r.category for r in self.triggered_rules}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "triggered_rules": [
                {
                    "rule_id": r.rule_id,
                    "rule_name": r.rule_name,
                    "category": r.category,
                    "severity": r.severity,
                    "confidence": r.confidence,
                    "details": r.details,
                }
                for r in self.triggered_rules
            ],
            "combined_severity": self.combined_severity,
            "combined_confidence": self.combined_confidence,
            "rule_score": self.rule_score,
            "evaluation_time_ms": self.evaluation_time_ms,
            "total_rules_evaluated": self.total_rules_evaluated,
        }


@dataclass
class RulePerformanceMetrics:
    """Precision / recall tracking per rule (updated via backtest)."""

    rule_id: str
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def total_evaluated(self) -> int:
        return (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "total_evaluated": self.total_evaluated,
        }


@dataclass
class EngineMetrics:
    """Aggregate metrics for the engine itself."""

    transactions_evaluated: int = 0
    total_rule_matches: int = 0
    total_evaluation_time_ms: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def avg_evaluation_time_ms(self) -> float:
        if self.transactions_evaluated == 0:
            return 0.0
        return self.total_evaluation_time_ms / self.transactions_evaluated

    def record(self, result: RuleEvaluationResult) -> None:
        with self._lock:
            self.transactions_evaluated += 1
            self.total_rule_matches += result.triggered_count
            self.total_evaluation_time_ms += result.evaluation_time_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transactions_evaluated": self.transactions_evaluated,
            "total_rule_matches": self.total_rule_matches,
            "avg_evaluation_time_ms": round(self.avg_evaluation_time_ms, 3),
        }


# ── Rule Definitions (loaded from YAML) ─────────────────────────────

@dataclass
class FraudRule:
    """Internal parsed representation of a single fraud rule."""

    id: str
    name: str
    description: str
    category: str
    priority: int
    enabled: bool
    severity: str
    confidence: float
    parameters: Dict[str, Any]
    tags: List[str]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FraudRule":
        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            category=d["category"],
            priority=d.get("priority", 100),
            enabled=d.get("enabled", True),
            severity=d.get("severity", SEVERITY_MEDIUM),
            confidence=d.get("confidence", 0.5),
            parameters=d.get("parameters", {}),
            tags=d.get("tags", []),
        )


# ── Helpers ──────────────────────────────────────────────────────────

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_timestamp(ts: Any) -> Optional[datetime]:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%fZ"):
            try:
                dt = datetime.strptime(ts, fmt)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Rule Engine ──────────────────────────────────────────────────────

class FraudRuleEngine:
    """Configurable, priority-ordered fraud rule evaluation engine.

    Usage::

        engine = FraudRuleEngine()                    # loads config/fraud_rules.yaml
        result = engine.evaluate(transaction, context)
        results = engine.evaluate_batch(transactions, contexts)

    ``context`` is an optional dict carrying enrichment data such as:
        - ``customer_avg_amount``  – historical average spend
        - ``customer_history_count`` – total past transactions
        - ``recent_transactions``  – list of recent txn dicts for velocity
        - ``last_transaction``     – previous txn dict (for geo checks)
        - ``customer_mcc_distribution`` – dict of MCC → pct
        - ``customer_channels``    – set of previously used channels
        - ``is_new_device``        – bool from device enricher
        - ``device_age_days``      – float from device enricher
        - ``accounts_on_device``   – int from device enricher
        - ``is_domestic_only``     – bool (never transacted internationally)
        - ``days_since_last_transaction`` – float
    """

    def __init__(self, rules_path: Optional[str | Path] = None) -> None:
        self._rules_path = Path(rules_path) if rules_path else _DEFAULT_RULES_PATH
        self._rules: List[FraudRule] = []
        self._severity_escalation: Dict[str, Any] = {}
        self._metrics = EngineMetrics()
        self._rule_metrics: Dict[str, RulePerformanceMetrics] = {}
        self._load_rules()

    # ── Rule Loading ─────────────────────────────────────────────────

    def _load_rules(self) -> None:
        with open(self._rules_path, "r") as f:
            config = yaml.safe_load(f)

        self._severity_escalation = config.get("severity_escalation", {})
        raw_rules = config.get("rules", [])

        self._rules = sorted(
            [FraudRule.from_dict(r) for r in raw_rules],
            key=lambda r: r.priority,
        )
        for rule in self._rules:
            if rule.id not in self._rule_metrics:
                self._rule_metrics[rule.id] = RulePerformanceMetrics(rule_id=rule.id)

        logger.info(
            "fraud_rules_loaded",
            total=len(self._rules),
            enabled=sum(1 for r in self._rules if r.enabled),
            path=str(self._rules_path),
        )

    def reload_rules(self) -> None:
        self._load_rules()

    # ── Public API ───────────────────────────────────────────────────

    @property
    def rules(self) -> List[FraudRule]:
        return list(self._rules)

    @property
    def metrics(self) -> EngineMetrics:
        return self._metrics

    @property
    def rule_metrics(self) -> Dict[str, RulePerformanceMetrics]:
        return dict(self._rule_metrics)

    def evaluate(
        self,
        transaction: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> RuleEvaluationResult:
        ctx = context or {}
        txn_id = transaction.get("external_transaction_id", "unknown")
        start = time.perf_counter()

        matches: List[RuleMatch] = []
        evaluated = 0

        for rule in self._rules:
            if not rule.enabled:
                continue
            evaluated += 1
            match = self._evaluate_rule(rule, transaction, ctx)
            if match is not None:
                matches.append(match)

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = RuleEvaluationResult(
            transaction_id=txn_id,
            triggered_rules=matches,
            combined_severity=self._compute_combined_severity(matches),
            combined_confidence=self._compute_combined_confidence(matches),
            rule_score=self._compute_rule_score(matches),
            evaluation_time_ms=round(elapsed_ms, 3),
            total_rules_evaluated=evaluated,
        )

        self._metrics.record(result)
        return result

    def evaluate_batch(
        self,
        transactions: Sequence[Dict[str, Any]],
        contexts: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[RuleEvaluationResult]:
        ctxs = contexts or [None] * len(transactions)
        return [self.evaluate(txn, ctx) for txn, ctx in zip(transactions, ctxs)]

    # ── Per-Rule Evaluation Dispatch ─────────────────────────────────

    def _evaluate_rule(
        self,
        rule: FraudRule,
        txn: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Optional[RuleMatch]:
        evaluator = _RULE_EVALUATORS.get(rule.id)
        if evaluator is not None:
            return evaluator(self, rule, txn, ctx)
        return None

    # ── Amount Rules ─────────────────────────────────────────────────

    def _eval_high_amount_vs_avg(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        amount = _safe_float(txn.get("transaction_amount"))
        multiplier = rule.parameters.get("multiplier", 3.0)
        min_history = rule.parameters.get("min_history_count", 5)
        fallback = rule.parameters.get("fallback_average", 500.0)

        history_count = _safe_int(ctx.get("customer_history_count", 0))
        avg = _safe_float(ctx.get("customer_avg_amount", fallback))
        if history_count < min_history:
            avg = fallback

        threshold = avg * multiplier
        if amount > threshold:
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"Amount ${amount:,.2f} exceeds {multiplier}x avg ${avg:,.2f} (threshold ${threshold:,.2f})",
                parameters={"amount": amount, "average": avg, "multiplier": multiplier},
            )
        return None

    def _eval_amount_below_threshold(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        amount = _safe_float(txn.get("transaction_amount"))
        upper = rule.parameters.get("threshold", 10000.0)
        lower = rule.parameters.get("lower_bound", 9000.0)

        if lower <= amount < upper:
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"Amount ${amount:,.2f} is just below reporting threshold ${upper:,.2f}",
                parameters={"amount": amount, "threshold": upper},
            )
        return None

    def _eval_round_amount(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        amount = _safe_float(txn.get("transaction_amount"))
        min_amount = rule.parameters.get("min_amount", 5000.0)
        modulo = rule.parameters.get("modulo", 1000)

        if amount >= min_amount and amount % modulo == 0:
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"Round amount ${amount:,.2f} (multiple of {modulo})",
                parameters={"amount": amount, "modulo": modulo},
            )
        return None

    # ── Velocity Rules ───────────────────────────────────────────────

    def _eval_rapid_transactions(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        max_count = rule.parameters.get("max_count", 5)
        window_minutes = rule.parameters.get("window_minutes", 10)

        recent: List[Dict[str, Any]] = ctx.get("recent_transactions", [])
        txn_ts = _parse_timestamp(txn.get("transaction_timestamp"))
        if txn_ts is None or not recent:
            return None

        window_seconds = window_minutes * 60
        count = 0
        for rt in recent:
            rt_ts = _parse_timestamp(rt.get("transaction_timestamp"))
            if rt_ts is None:
                continue
            diff = abs((txn_ts - rt_ts).total_seconds())
            if diff <= window_seconds:
                count += 1

        if count >= max_count:
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"{count} transactions in {window_minutes} min (limit {max_count})",
                parameters={"count": count, "window_minutes": window_minutes},
            )
        return None

    def _eval_declined_then_approved(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        min_declines = rule.parameters.get("min_declines", 3)
        window_minutes = rule.parameters.get("window_minutes", 30)

        status = txn.get("status", txn.get("transaction_status", ""))
        if status != "approved":
            return None

        recent: List[Dict[str, Any]] = ctx.get("recent_transactions", [])
        txn_ts = _parse_timestamp(txn.get("transaction_timestamp"))
        if txn_ts is None:
            return None

        window_seconds = window_minutes * 60
        decline_count = 0
        for rt in recent:
            rt_ts = _parse_timestamp(rt.get("transaction_timestamp"))
            if rt_ts is None:
                continue
            diff = abs((txn_ts - rt_ts).total_seconds())
            rt_status = rt.get("status", rt.get("transaction_status", ""))
            if diff <= window_seconds and rt_status == "declined":
                decline_count += 1

        if decline_count >= min_declines:
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"{decline_count} declines before approval in {window_minutes} min",
                parameters={"decline_count": decline_count},
            )
        return None

    def _eval_escalating_amounts(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        min_txns = rule.parameters.get("min_transactions", 3)
        factor = rule.parameters.get("escalation_factor", 2.0)
        window_minutes = rule.parameters.get("window_minutes", 60)

        recent: List[Dict[str, Any]] = ctx.get("recent_transactions", [])
        txn_ts = _parse_timestamp(txn.get("transaction_timestamp"))
        if txn_ts is None or len(recent) < min_txns - 1:
            return None

        window_seconds = window_minutes * 60
        amounts: List[float] = []
        for rt in recent:
            rt_ts = _parse_timestamp(rt.get("transaction_timestamp"))
            if rt_ts and abs((txn_ts - rt_ts).total_seconds()) <= window_seconds:
                amounts.append(_safe_float(rt.get("transaction_amount")))

        amounts.append(_safe_float(txn.get("transaction_amount")))
        amounts.sort()

        if len(amounts) < min_txns:
            return None

        last_few = amounts[-min_txns:]
        is_escalating = all(
            last_few[i + 1] >= last_few[i] * factor
            for i in range(len(last_few) - 1)
            if last_few[i] > 0
        )

        if is_escalating and last_few[0] > 0:
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"Escalating amounts: {['$' + f'{a:,.2f}' for a in last_few]}",
                parameters={"amounts": last_few, "factor": factor},
            )
        return None

    # ── Geo Rules ────────────────────────────────────────────────────

    def _eval_impossible_travel(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        max_speed = rule.parameters.get("max_speed_mph", IMPOSSIBLE_TRAVEL_SPEED_MPH)
        min_dist = rule.parameters.get("min_distance_miles", 100)

        last_txn: Optional[Dict[str, Any]] = ctx.get("last_transaction")
        if not last_txn:
            return None

        lat1 = _safe_float(last_txn.get("geo_latitude"))
        lon1 = _safe_float(last_txn.get("geo_longitude"))
        lat2 = _safe_float(txn.get("geo_latitude"))
        lon2 = _safe_float(txn.get("geo_longitude"))

        if not all([lat1, lon1, lat2, lon2]):
            return None

        distance = _haversine_miles(lat1, lon1, lat2, lon2)
        if distance < min_dist:
            return None

        ts1 = _parse_timestamp(last_txn.get("transaction_timestamp"))
        ts2 = _parse_timestamp(txn.get("transaction_timestamp"))
        if ts1 is None or ts2 is None:
            return None

        hours = abs((ts2 - ts1).total_seconds()) / 3600
        if hours <= 0:
            return None

        speed = distance / hours
        if speed > max_speed:
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"Travel speed {speed:,.0f} mph over {distance:,.0f} miles in {hours:.1f} hrs",
                parameters={"speed_mph": round(speed, 1), "distance_miles": round(distance, 1)},
            )
        return None

    def _eval_international_domestic_only(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        is_intl = txn.get("is_international", False)
        if not is_intl:
            return None

        is_domestic_only = ctx.get("is_domestic_only", False)
        min_domestic = rule.parameters.get("min_domestic_transactions", 10)
        history_count = _safe_int(ctx.get("customer_history_count", 0))

        if is_domestic_only and history_count >= min_domestic:
            country = txn.get("geo_country", "unknown")
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"International txn (country={country}) on domestic-only account ({history_count} prior domestic txns)",
                parameters={"country": country, "history_count": history_count},
            )
        return None

    def _eval_high_risk_country(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        country = txn.get("geo_country", "")
        high_risk = rule.parameters.get("high_risk_countries", [])
        if country in high_risk:
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"Transaction from high-risk country: {country}",
                parameters={"country": country},
            )
        return None

    # ── Pattern Rules ────────────────────────────────────────────────

    def _eval_new_device_high_amount(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        amount = _safe_float(txn.get("transaction_amount"))
        amount_threshold = rule.parameters.get("amount_threshold", 1000.0)
        max_device_age = rule.parameters.get("device_age_days", 1)

        is_new = ctx.get("is_new_device", False)
        device_age = _safe_float(ctx.get("device_age_days", 999))

        if (is_new or device_age <= max_device_age) and amount >= amount_threshold:
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"New device (age {device_age:.1f}d) with high amount ${amount:,.2f}",
                parameters={"amount": amount, "device_age_days": device_age},
            )
        return None

    def _eval_merchant_category_mismatch(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        mcc = txn.get("merchant_category_code", "")
        min_history = rule.parameters.get("min_history_count", 10)
        max_pct = rule.parameters.get("max_category_pct", 0.05)

        history_count = _safe_int(ctx.get("customer_history_count", 0))
        if history_count < min_history:
            return None

        mcc_dist: Dict[str, float] = ctx.get("customer_mcc_distribution", {})
        if not mcc_dist:
            return None

        pct = mcc_dist.get(mcc, 0.0)
        if pct <= max_pct:
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"MCC {mcc} represents only {pct:.1%} of customer history",
                parameters={"mcc": mcc, "pct": pct},
            )
        return None

    def _eval_card_testing(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        small_threshold = rule.parameters.get("small_amount_threshold", 5.0)
        min_small = rule.parameters.get("min_small_count", 3)
        large_threshold = rule.parameters.get("large_amount_threshold", 500.0)
        window_minutes = rule.parameters.get("window_minutes", 30)

        amount = _safe_float(txn.get("transaction_amount"))
        if amount < large_threshold:
            return None

        recent: List[Dict[str, Any]] = ctx.get("recent_transactions", [])
        txn_ts = _parse_timestamp(txn.get("transaction_timestamp"))
        if txn_ts is None:
            return None

        window_seconds = window_minutes * 60
        small_count = 0
        for rt in recent:
            rt_ts = _parse_timestamp(rt.get("transaction_timestamp"))
            if rt_ts is None:
                continue
            diff = abs((txn_ts - rt_ts).total_seconds())
            rt_amount = _safe_float(rt.get("transaction_amount"))
            if diff <= window_seconds and rt_amount <= small_threshold:
                small_count += 1

        if small_count >= min_small:
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"{small_count} small txns (≤${small_threshold}) before large ${amount:,.2f}",
                parameters={"small_count": small_count, "large_amount": amount},
            )
        return None

    def _eval_multi_account_device(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        max_accounts = rule.parameters.get("max_accounts", 2)
        accounts_on_device = _safe_int(ctx.get("accounts_on_device", 1))

        if accounts_on_device > max_accounts:
            device_id = txn.get("device_id", "unknown")
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"Device {device_id} used by {accounts_on_device} accounts (max {max_accounts})",
                parameters={"accounts_on_device": accounts_on_device, "device_id": device_id},
            )
        return None

    def _eval_channel_anomaly(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        min_history = rule.parameters.get("min_history_count", 10)
        history_count = _safe_int(ctx.get("customer_history_count", 0))
        if history_count < min_history:
            return None

        channel = txn.get("channel", "")
        known_channels: set = ctx.get("customer_channels", set())
        if not known_channels:
            return None

        if channel not in known_channels:
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"Channel '{channel}' never used before (known: {sorted(known_channels)})",
                parameters={"channel": channel, "known_channels": sorted(known_channels)},
            )
        return None

    # ── Temporal Rules ───────────────────────────────────────────────

    def _eval_late_night_high_value(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        start_hour = rule.parameters.get("start_hour", 0)
        end_hour = rule.parameters.get("end_hour", 5)
        amount_threshold = rule.parameters.get("amount_threshold", 2000.0)

        amount = _safe_float(txn.get("transaction_amount"))
        ts = _parse_timestamp(txn.get("transaction_timestamp"))
        if ts is None or amount < amount_threshold:
            return None

        hour = ts.hour
        if start_hour <= hour < end_hour:
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"${amount:,.2f} transaction at {hour:02d}:00 (late-night window {start_hour:02d}:00-{end_hour:02d}:00)",
                parameters={"amount": amount, "hour": hour},
            )
        return None

    def _eval_dormant_account(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        dormant_days = rule.parameters.get("dormant_days", 90)
        days_since = _safe_float(ctx.get("days_since_last_transaction", 0))

        if days_since >= dormant_days:
            return RuleMatch(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                confidence=rule.confidence,
                details=f"Account dormant for {days_since:.0f} days (threshold {dormant_days})",
                parameters={"days_since_last": days_since, "dormant_days": dormant_days},
            )
        return None

    def _eval_burst_after_silence(
        self, rule: FraudRule, txn: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Optional[RuleMatch]:
        silence_hours = rule.parameters.get("silence_hours", 48)
        burst_count = rule.parameters.get("burst_count", 5)
        burst_window_minutes = rule.parameters.get("burst_window_minutes", 30)

        recent: List[Dict[str, Any]] = ctx.get("recent_transactions", [])
        txn_ts = _parse_timestamp(txn.get("transaction_timestamp"))
        if txn_ts is None or not recent:
            return None

        burst_window_sec = burst_window_minutes * 60
        silence_sec = silence_hours * 3600

        timestamps = []
        for rt in recent:
            rt_ts = _parse_timestamp(rt.get("transaction_timestamp"))
            if rt_ts:
                timestamps.append(rt_ts)

        if not timestamps:
            return None

        timestamps.sort()

        in_burst = sum(
            1 for ts in timestamps if abs((txn_ts - ts).total_seconds()) <= burst_window_sec
        )

        if in_burst < burst_count - 1:
            return None

        pre_burst = [
            ts for ts in timestamps if (txn_ts - ts).total_seconds() > burst_window_sec
        ]
        if pre_burst:
            gap_seconds = (txn_ts - max(pre_burst)).total_seconds() - burst_window_sec
            if gap_seconds < silence_sec:
                return None
        else:
            days_since = _safe_float(ctx.get("days_since_last_transaction", 0))
            if days_since * 86400 < silence_sec:
                return None

        return RuleMatch(
            rule_id=rule.id,
            rule_name=rule.name,
            category=rule.category,
            severity=rule.severity,
            confidence=rule.confidence,
            details=f"{in_burst + 1} txns in {burst_window_minutes} min after {silence_hours}h silence",
            parameters={"burst_count": in_burst + 1, "silence_hours": silence_hours},
        )

    # ── Scoring & Severity ───────────────────────────────────────────

    def _compute_combined_severity(self, matches: List[RuleMatch]) -> str:
        if not matches:
            return SEVERITY_LOW

        max_sev_idx = max(_SEVERITY_ORDER.get(m.severity, 0) for m in matches)

        escalation = self._severity_escalation
        if escalation.get("enabled", False) and len(matches) > 1:
            rules_per_tier = escalation.get("rules_per_tier", 2)
            extra_tiers = (len(matches) - 1) // rules_per_tier
            max_sev_idx = min(max_sev_idx + extra_tiers, _SEVERITY_ORDER[SEVERITY_CRITICAL])

        return _SEVERITY_BY_INDEX.get(max_sev_idx, SEVERITY_LOW)

    def _compute_combined_confidence(self, matches: List[RuleMatch]) -> float:
        if not matches:
            return 0.0
        confidences = [m.confidence for m in matches]
        # Noisy-OR combination: P(fraud) = 1 - ∏(1 - p_i)
        combined = 1.0
        for c in confidences:
            combined *= (1.0 - c)
        return round(1.0 - combined, 4)

    def _compute_rule_score(self, matches: List[RuleMatch]) -> float:
        if not matches:
            return 0.0
        combined_conf = self._compute_combined_confidence(matches)
        severity_weight = max(_SEVERITY_ORDER.get(m.severity, 0) for m in matches) / 3.0
        category_diversity = len({m.category for m in matches}) / 5.0
        score = (
            0.50 * combined_conf
            + 0.30 * severity_weight
            + 0.20 * min(category_diversity, 1.0)
        )
        return round(min(score, 1.0), 4)

    # ── Backtesting ──────────────────────────────────────────────────

    def backtest(
        self,
        transactions: Sequence[Dict[str, Any]],
        labels: Sequence[bool],
        contexts: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> Dict[str, RulePerformanceMetrics]:
        """Run all rules against labelled data and compute per-rule metrics.

        Args:
            transactions: historical transactions
            labels: True = actually fraudulent, False = legitimate
            contexts: optional enrichment contexts per transaction

        Returns:
            dict mapping rule_id → RulePerformanceMetrics
        """
        ctxs = contexts or [None] * len(transactions)
        metrics: Dict[str, RulePerformanceMetrics] = {
            r.id: RulePerformanceMetrics(rule_id=r.id)
            for r in self._rules
            if r.enabled
        }

        for txn, label, ctx in zip(transactions, labels, ctxs):
            result = self.evaluate(txn, ctx)
            triggered_ids = {m.rule_id for m in result.triggered_rules}

            for rule_id, perf in metrics.items():
                fired = rule_id in triggered_ids
                if fired and label:
                    perf.true_positives += 1
                elif fired and not label:
                    perf.false_positives += 1
                elif not fired and label:
                    perf.false_negatives += 1
                else:
                    perf.true_negatives += 1

        self._rule_metrics.update(metrics)

        logger.info(
            "backtest_complete",
            total_transactions=len(transactions),
            total_fraud=sum(labels),
            rules_evaluated=len(metrics),
        )
        return metrics

    def get_backtest_summary(self) -> Dict[str, Any]:
        return {
            rule_id: perf.to_dict()
            for rule_id, perf in self._rule_metrics.items()
            if perf.total_evaluated > 0
        }


# ── Rule Evaluator Registry ─────────────────────────────────────────
# Maps rule IDs from fraud_rules.yaml to their evaluator methods.

_RULE_EVALUATORS: Dict[str, Any] = {}


def _register_evaluators() -> None:
    global _RULE_EVALUATORS
    _RULE_EVALUATORS = {
        "FRAUD-AMT-001": FraudRuleEngine._eval_high_amount_vs_avg,
        "FRAUD-AMT-002": FraudRuleEngine._eval_amount_below_threshold,
        "FRAUD-AMT-003": FraudRuleEngine._eval_round_amount,
        "FRAUD-VEL-001": FraudRuleEngine._eval_rapid_transactions,
        "FRAUD-VEL-002": FraudRuleEngine._eval_declined_then_approved,
        "FRAUD-VEL-003": FraudRuleEngine._eval_escalating_amounts,
        "FRAUD-GEO-001": FraudRuleEngine._eval_impossible_travel,
        "FRAUD-GEO-002": FraudRuleEngine._eval_international_domestic_only,
        "FRAUD-GEO-003": FraudRuleEngine._eval_high_risk_country,
        "FRAUD-PAT-001": FraudRuleEngine._eval_new_device_high_amount,
        "FRAUD-PAT-002": FraudRuleEngine._eval_merchant_category_mismatch,
        "FRAUD-PAT-003": FraudRuleEngine._eval_card_testing,
        "FRAUD-PAT-004": FraudRuleEngine._eval_multi_account_device,
        "FRAUD-PAT-005": FraudRuleEngine._eval_channel_anomaly,
        "FRAUD-TMP-001": FraudRuleEngine._eval_late_night_high_value,
        "FRAUD-TMP-002": FraudRuleEngine._eval_dormant_account,
        "FRAUD-TMP-003": FraudRuleEngine._eval_burst_after_silence,
    }


_register_evaluators()


# ── Module-level convenience ─────────────────────────────────────────

_engine_instance: Optional[FraudRuleEngine] = None
_engine_lock = Lock()


def get_rule_engine(rules_path: Optional[str | Path] = None) -> FraudRuleEngine:
    """Return a module-level singleton FraudRuleEngine."""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = FraudRuleEngine(rules_path=rules_path)
        return _engine_instance
