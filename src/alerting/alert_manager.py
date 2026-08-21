"""Alert Generation Framework — Real-time fraud alert management.

Production-grade alert system with:
- Alert creation from unified scoring results
- Deduplication (same account + rule + type within configurable window)
- Severity-based routing (low→email, medium→dashboard, high→SMS, critical→immediate)
- Alert enrichment (customer history, recent transactions, risk profile)
- Lifecycle management (open → investigating → resolved / false_positive)
- Suppression (skip alerts on already-investigated accounts)
- Throttling and storm detection
- Kafka publishing and PostgreSQL persistence
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, cast

import yaml

from src.utils.constants import (
    ALERT_INVESTIGATING,
    ALERT_RESOLVED,
    TOPIC_FRAUD_ALERTS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, component="alert_manager")

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_DEFAULT_ROUTING_PATH = _CONFIG_DIR / "alert_routing.yaml"


# ── Enums ────────────────────────────────────────────────────────────────────


class AlertType(str, Enum):
    RULE_BASED = "rule_based"
    ANOMALY = "anomaly"
    ML_SCORE = "ml_score"
    ENSEMBLE = "ensemble"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class AlertChannel(str, Enum):
    EMAIL = "email"
    DASHBOARD = "dashboard"
    SMS = "sms"
    WEBHOOK = "webhook"


# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class Alert:
    """Represents a single fraud alert with full context."""

    alert_id: str
    transaction_id: str
    account_id: str
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus
    risk_score: float
    rule_id: str | None = None
    description: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    enrichment: dict[str, Any] = field(default_factory=dict)
    channels: list[str] = field(default_factory=list)
    assigned_to: str | None = None
    resolved_at: datetime | None = None
    resolution_notes: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    suppressed: bool = False
    deduplicated: bool = False
    parent_alert_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize alert to dictionary for storage/publishing."""
        return {
            "alert_id": self.alert_id,
            "transaction_id": self.transaction_id,
            "account_id": self.account_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "risk_score": round(self.risk_score, 6),
            "rule_id": self.rule_id,
            "description": self.description,
            "details": self.details,
            "enrichment": self.enrichment,
            "channels": self.channels,
            "assigned_to": self.assigned_to,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "suppressed": self.suppressed,
            "deduplicated": self.deduplicated,
            "parent_alert_id": self.parent_alert_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Alert":
        """Deserialize alert from dictionary."""
        return cls(
            alert_id=data["alert_id"],
            transaction_id=data["transaction_id"],
            account_id=data["account_id"],
            alert_type=AlertType(data["alert_type"]),
            severity=AlertSeverity(data["severity"]),
            status=AlertStatus(data["status"]),
            risk_score=data["risk_score"],
            rule_id=data.get("rule_id"),
            description=data.get("description", ""),
            details=data.get("details", {}),
            enrichment=data.get("enrichment", {}),
            channels=data.get("channels", []),
            assigned_to=data.get("assigned_to"),
            resolved_at=(
                datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None
            ),
            resolution_notes=data.get("resolution_notes"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            suppressed=data.get("suppressed", False),
            deduplicated=data.get("deduplicated", False),
            parent_alert_id=data.get("parent_alert_id"),
        )


@dataclass
class AlertStatistics:
    """Aggregated alert statistics."""

    total_generated: int = 0
    total_suppressed: int = 0
    total_deduplicated: int = 0
    total_published: int = 0
    by_severity: dict[str, int] = field(
        default_factory=lambda: {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0,
        }
    )
    by_status: dict[str, int] = field(
        default_factory=lambda: {
            "open": 0,
            "investigating": 0,
            "resolved": 0,
            "false_positive": 0,
        }
    )
    by_type: dict[str, int] = field(
        default_factory=lambda: {
            "rule_based": 0,
            "anomaly": 0,
            "ml_score": 0,
            "ensemble": 0,
        }
    )
    avg_resolution_time_minutes: float = 0.0
    alerts_per_minute: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record_generated(self, severity: str, alert_type: str) -> None:
        with self._lock:
            self.total_generated += 1
            self.by_severity[severity] = self.by_severity.get(severity, 0) + 1
            self.by_type[alert_type] = self.by_type.get(alert_type, 0) + 1
            self.by_status["open"] = self.by_status.get("open", 0) + 1

    def record_suppressed(self) -> None:
        with self._lock:
            self.total_suppressed += 1

    def record_deduplicated(self) -> None:
        with self._lock:
            self.total_deduplicated += 1

    def record_published(self) -> None:
        with self._lock:
            self.total_published += 1

    def record_status_change(self, old_status: str, new_status: str) -> None:
        with self._lock:
            self.by_status[old_status] = max(0, self.by_status.get(old_status, 0) - 1)
            self.by_status[new_status] = self.by_status.get(new_status, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_generated": self.total_generated,
                "total_suppressed": self.total_suppressed,
                "total_deduplicated": self.total_deduplicated,
                "total_published": self.total_published,
                "by_severity": dict(self.by_severity),
                "by_status": dict(self.by_status),
                "by_type": dict(self.by_type),
                "avg_resolution_time_minutes": round(self.avg_resolution_time_minutes, 2),
                "alerts_per_minute": round(self.alerts_per_minute, 2),
            }


# ── Deduplication Engine ─────────────────────────────────────────────────────


class DeduplicationEngine:
    """Prevents duplicate alerts for the same account/rule within a time window."""

    def __init__(self, window_minutes: int = 60, max_entries: int = 100000) -> None:
        self._window_seconds = window_minutes * 60
        self._max_entries = max_entries
        self._seen: dict[str, list[float]] = {}
        self._lock = Lock()

    def _generate_dedup_key(self, account_id: str, rule_id: str | None, alert_type: str) -> str:
        """Generate a deduplication key from alert attributes."""
        raw = f"{account_id}:{rule_id or 'none'}:{alert_type}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def is_duplicate(self, account_id: str, rule_id: str | None, alert_type: str) -> bool:
        """Check if an alert with same key was seen within the dedup window.

        Returns True if a duplicate exists (alert should be suppressed).
        """
        key = self._generate_dedup_key(account_id, rule_id, alert_type)
        now = time.time()

        with self._lock:
            if key in self._seen:
                # Clean expired entries
                self._seen[key] = [
                    ts for ts in self._seen[key] if (now - ts) < self._window_seconds
                ]
                if self._seen[key]:
                    return True

            # Record this alert
            if key not in self._seen:
                self._seen[key] = []
            self._seen[key].append(now)

            # Evict oldest entries if over capacity
            if len(self._seen) > self._max_entries:
                self._evict_expired(now)

            return False

    def _evict_expired(self, now: float) -> None:
        """Remove entries with all timestamps expired."""
        expired_keys = [
            k
            for k, timestamps in self._seen.items()
            if all((now - ts) >= self._window_seconds for ts in timestamps)
        ]
        for k in expired_keys:
            del self._seen[k]

    def clear(self) -> None:
        """Clear all deduplication state."""
        with self._lock:
            self._seen.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._seen)


# ── Suppression Engine ───────────────────────────────────────────────────────


class SuppressionEngine:
    """Suppresses alerts for accounts that are already under investigation."""

    def __init__(
        self,
        suppress_on_status: list[str] | None = None,
        cooldown_minutes: int = 120,
    ) -> None:
        self._suppress_on_status = suppress_on_status or [ALERT_INVESTIGATING, ALERT_RESOLVED]
        self._cooldown_seconds = cooldown_minutes * 60
        self._suppressed_accounts: dict[str, tuple[str, float]] = {}
        self._lock = Lock()

    def add_suppression(self, account_id: str, reason: str) -> None:
        """Add an account to the suppression list."""
        with self._lock:
            self._suppressed_accounts[account_id] = (reason, time.time())

    def remove_suppression(self, account_id: str) -> None:
        """Remove an account from the suppression list."""
        with self._lock:
            self._suppressed_accounts.pop(account_id, None)

    def is_suppressed(self, account_id: str) -> bool:
        """Check if alerts for an account should be suppressed."""
        with self._lock:
            if account_id not in self._suppressed_accounts:
                return False
            reason, timestamp = self._suppressed_accounts[account_id]
            # Check if cooldown has expired
            if (time.time() - timestamp) > self._cooldown_seconds:
                del self._suppressed_accounts[account_id]
                return False
            return True

    def get_suppression_reason(self, account_id: str) -> str | None:
        """Get the reason an account is suppressed."""
        with self._lock:
            if account_id in self._suppressed_accounts:
                return self._suppressed_accounts[account_id][0]
            return None

    def clear(self) -> None:
        with self._lock:
            self._suppressed_accounts.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._suppressed_accounts)


# ── Throttle Engine ──────────────────────────────────────────────────────────


class ThrottleEngine:
    """Rate-limits alert generation to prevent alert storms."""

    def __init__(
        self,
        max_per_account_per_hour: int = 5,
        max_per_rule_per_hour: int = 50,
        max_total_per_minute: int = 100,
        storm_threshold_per_minute: int = 200,
    ) -> None:
        self._max_per_account_per_hour = max_per_account_per_hour
        self._max_per_rule_per_hour = max_per_rule_per_hour
        self._max_total_per_minute = max_total_per_minute
        self._storm_threshold = storm_threshold_per_minute

        self._account_counts: dict[str, list[float]] = defaultdict(list)
        self._rule_counts: dict[str, list[float]] = defaultdict(list)
        self._total_timestamps: list[float] = []
        self._storm_active = False
        self._storm_cooldown_until: float = 0.0
        self._lock = Lock()

    def should_throttle(self, account_id: str, rule_id: str | None) -> tuple[bool, str]:
        """Check if alert should be throttled.

        Returns:
            Tuple of (should_throttle, reason).
        """
        now = time.time()
        one_hour_ago = now - 3600
        one_minute_ago = now - 60

        with self._lock:
            # Storm detection
            if self._storm_active and now < self._storm_cooldown_until:
                return True, "alert_storm_active"

            # Clean and check total rate
            self._total_timestamps = [ts for ts in self._total_timestamps if ts > one_minute_ago]
            if len(self._total_timestamps) >= self._max_total_per_minute:
                return True, "total_rate_exceeded"

            # Storm detection trigger
            if len(self._total_timestamps) >= self._storm_threshold:
                self._storm_active = True
                self._storm_cooldown_until = now + 300  # 5 min cooldown
                logger.warning(
                    "alert_storm_detected",
                    alerts_per_minute=len(self._total_timestamps),
                    cooldown_seconds=300,
                )
                return True, "alert_storm_triggered"

            # Per-account throttle
            self._account_counts[account_id] = [
                ts for ts in self._account_counts[account_id] if ts > one_hour_ago
            ]
            if len(self._account_counts[account_id]) >= self._max_per_account_per_hour:
                return True, f"account_rate_exceeded:{account_id}"

            # Per-rule throttle
            if rule_id:
                self._rule_counts[rule_id] = [
                    ts for ts in self._rule_counts[rule_id] if ts > one_hour_ago
                ]
                if len(self._rule_counts[rule_id]) >= self._max_per_rule_per_hour:
                    return True, f"rule_rate_exceeded:{rule_id}"

            # Record this alert
            self._total_timestamps.append(now)
            self._account_counts[account_id].append(now)
            if rule_id:
                self._rule_counts[rule_id].append(now)

            return False, ""

    @property
    def is_storm_active(self) -> bool:
        with self._lock:
            if self._storm_active and time.time() >= self._storm_cooldown_until:
                self._storm_active = False
            return self._storm_active

    def reset_storm(self) -> None:
        with self._lock:
            self._storm_active = False
            self._storm_cooldown_until = 0.0

    def clear(self) -> None:
        with self._lock:
            self._account_counts.clear()
            self._rule_counts.clear()
            self._total_timestamps.clear()
            self._storm_active = False
            self._storm_cooldown_until = 0.0


# ── Alert Manager ────────────────────────────────────────────────────────────


class AlertManager:
    """Central alert management system orchestrating generation, routing, and lifecycle.

    Coordinates deduplication, suppression, throttling, enrichment,
    severity-based routing, and alert lifecycle transitions.

    Usage::

        manager = AlertManager()
        alert = manager.generate_alert(scoring_result, transaction)
        manager.transition_alert(alert.alert_id, AlertStatus.INVESTIGATING)
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        kafka_producer: Any | None = None,
        db_session: Any | None = None,
    ) -> None:
        self._config = self._load_config(config_path or _DEFAULT_ROUTING_PATH)
        self._kafka_producer = kafka_producer
        self._db_session = db_session

        # Initialize sub-engines from config
        dedup_config = self._config.get("deduplication", {})
        suppression_config = self._config.get("suppression", {})
        throttle_config = self._config.get("throttling", {})

        self._dedup_engine = DeduplicationEngine(
            window_minutes=dedup_config.get("window_minutes", 60),
            max_entries=dedup_config.get("max_suppressed_per_window", 100000),
        )
        self._suppression_engine = SuppressionEngine(
            suppress_on_status=suppression_config.get("suppress_on_status", []),
            cooldown_minutes=suppression_config.get("cooldown_after_resolution_minutes", 120),
        )
        self._throttle_engine = ThrottleEngine(
            max_per_account_per_hour=throttle_config.get("max_alerts_per_account_per_hour", 5),
            max_per_rule_per_hour=throttle_config.get("max_alerts_per_rule_per_hour", 50),
            max_total_per_minute=throttle_config.get("max_total_alerts_per_minute", 100),
            storm_threshold_per_minute=throttle_config.get("storm_detection", {}).get(
                "threshold_per_minute", 200
            ),
        )

        # Alert storage (in-memory index for lifecycle management)
        self._alerts: dict[str, Alert] = {}
        self._alerts_lock = Lock()

        # Statistics
        self._statistics = AlertStatistics()

        # Routing config
        self._routing = self._config.get("routing", {}).get("severity_channels", {})

        # Lifecycle transitions
        self._transitions = self._config.get("lifecycle", {}).get("transitions", {})

        logger.info(
            "alert_manager_initialized",
            dedup_window_minutes=dedup_config.get("window_minutes", 60),
            throttle_max_per_minute=throttle_config.get("max_total_alerts_per_minute", 100),
        )

    @staticmethod
    def _load_config(config_path: str | Path) -> dict[str, Any]:
        """Load alert routing configuration from YAML."""
        path = Path(config_path)
        if not path.exists():
            logger.warning("alert_config_not_found", path=str(path))
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    # ── Alert Generation ─────────────────────────────────────────────────

    def generate_alert(
        self,
        scoring_result: dict[str, Any],
        transaction: dict[str, Any],
        enrichment_context: dict[str, Any] | None = None,
    ) -> Alert | None:
        """Generate an alert from a scoring result.

        Applies deduplication, suppression, and throttling before creating.
        Returns None if the alert was filtered out.

        Args:
            scoring_result: Output from ScoringPipeline (UnifiedScore.to_dict()).
            transaction: Original transaction data.
            enrichment_context: Optional pre-fetched enrichment data.

        Returns:
            Alert instance or None if suppressed/deduplicated/throttled.
        """
        account_id = transaction.get("account_id", "unknown")
        transaction_id = transaction.get("external_transaction_id") or transaction.get(
            "transaction_id", "unknown"
        )
        risk_score = scoring_result.get("final_score", 0.0)
        risk_classification = scoring_result.get("risk_classification", "low")

        # Determine alert type from scoring method
        alert_type = self._determine_alert_type(scoring_result)
        severity = self._classify_severity(risk_score, risk_classification)

        # Determine triggered rule (if any)
        rule_id = self._extract_rule_id(scoring_result)

        # ── Suppression check ────────────────────────────────────────────
        if self._suppression_engine.is_suppressed(account_id):
            reason = self._suppression_engine.get_suppression_reason(account_id)
            logger.debug(
                "alert_suppressed",
                account_id=account_id,
                reason=reason,
            )
            self._statistics.record_suppressed()
            return None

        # ── Deduplication check ──────────────────────────────────────────
        if self._dedup_engine.is_duplicate(account_id, rule_id, alert_type.value):
            logger.debug(
                "alert_deduplicated",
                account_id=account_id,
                rule_id=rule_id,
                alert_type=alert_type.value,
            )
            self._statistics.record_deduplicated()
            return None

        # ── Throttle check ───────────────────────────────────────────────
        throttled, throttle_reason = self._throttle_engine.should_throttle(account_id, rule_id)
        if throttled:
            logger.debug(
                "alert_throttled",
                account_id=account_id,
                reason=throttle_reason,
            )
            self._statistics.record_suppressed()
            return None

        # ── Create alert ─────────────────────────────────────────────────
        channels = self._resolve_channels(severity)
        description = self._build_description(alert_type, severity, risk_score, transaction)

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            transaction_id=transaction_id,
            account_id=account_id,
            alert_type=alert_type,
            severity=severity,
            status=AlertStatus.OPEN,
            risk_score=risk_score,
            rule_id=rule_id,
            description=description,
            details=self._build_details(scoring_result, transaction),
            enrichment=enrichment_context or {},
            channels=[ch.value if isinstance(ch, AlertChannel) else ch for ch in channels],
        )

        # Store alert
        with self._alerts_lock:
            self._alerts[alert.alert_id] = alert

        # Record statistics
        self._statistics.record_generated(severity.value, alert_type.value)

        # Publish to Kafka
        self._publish_alert(alert)

        logger.info(
            "alert_generated",
            alert_id=alert.alert_id,
            account_id=account_id,
            severity=severity.value,
            risk_score=round(risk_score, 4),
            channels=[ch for ch in alert.channels],
        )

        return alert

    def generate_alerts_from_batch(
        self,
        scored_transactions: list[tuple[dict[str, Any], dict[str, Any]]],
        enrichment_contexts: dict[str, dict[str, Any]] | None = None,
    ) -> list[Alert]:
        """Generate alerts from a batch of scored transactions.

        Args:
            scored_transactions: List of (scoring_result, transaction) tuples.
            enrichment_contexts: Map of account_id → enrichment data.

        Returns:
            List of generated alerts (excludes suppressed/deduplicated).
        """
        alerts = []
        enrichment_contexts = enrichment_contexts or {}

        for scoring_result, transaction in scored_transactions:
            # Only alert if recommended by scoring pipeline
            if not scoring_result.get("alert_recommended", False):
                continue

            account_id = transaction.get("account_id", "unknown")
            enrichment = enrichment_contexts.get(account_id, {})
            alert = self.generate_alert(scoring_result, transaction, enrichment)
            if alert is not None:
                alerts.append(alert)

        return alerts

    # ── Alert Lifecycle ──────────────────────────────────────────────────

    def transition_alert(
        self,
        alert_id: str,
        new_status: AlertStatus,
        assigned_to: str | None = None,
        resolution_notes: str | None = None,
    ) -> Alert | None:
        """Transition an alert to a new lifecycle status.

        Validates the transition against allowed state transitions.

        Args:
            alert_id: The alert to transition.
            new_status: Target status.
            assigned_to: Optional assignee for investigating status.
            resolution_notes: Notes when resolving.

        Returns:
            Updated Alert or None if transition is invalid.
        """
        with self._alerts_lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                logger.warning("alert_not_found", alert_id=alert_id)
                return None

            old_status = alert.status

            # Validate transition
            if not self._is_valid_transition(old_status, new_status):
                logger.warning(
                    "invalid_alert_transition",
                    alert_id=alert_id,
                    from_status=old_status.value,
                    to_status=new_status.value,
                )
                return None

            # Apply transition
            alert.status = new_status
            alert.updated_at = datetime.now(timezone.utc)

            if assigned_to:
                alert.assigned_to = assigned_to

            if new_status in (AlertStatus.RESOLVED, AlertStatus.FALSE_POSITIVE):
                alert.resolved_at = datetime.now(timezone.utc)
                alert.resolution_notes = resolution_notes

                # Add account to suppression list
                self._suppression_engine.add_suppression(
                    alert.account_id,
                    f"Alert {alert_id} {new_status.value}",
                )

            if new_status == AlertStatus.INVESTIGATING:
                # Suppress further alerts for this account during investigation
                self._suppression_engine.add_suppression(
                    alert.account_id,
                    f"Alert {alert_id} under investigation",
                )

        # Record status change
        self._statistics.record_status_change(old_status.value, new_status.value)

        logger.info(
            "alert_status_changed",
            alert_id=alert_id,
            from_status=old_status.value,
            to_status=new_status.value,
            assigned_to=assigned_to,
        )

        return alert

    def get_alert(self, alert_id: str) -> Alert | None:
        """Retrieve an alert by ID."""
        with self._alerts_lock:
            return self._alerts.get(alert_id)

    def get_alerts_by_account(self, account_id: str) -> list[Alert]:
        """Retrieve all alerts for an account."""
        with self._alerts_lock:
            return [a for a in self._alerts.values() if a.account_id == account_id]

    def get_alerts_by_status(self, status: AlertStatus) -> list[Alert]:
        """Retrieve all alerts with a given status."""
        with self._alerts_lock:
            return [a for a in self._alerts.values() if a.status == status]

    def get_open_alerts(self) -> list[Alert]:
        """Retrieve all open alerts, ordered by severity (critical first)."""
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
        }
        with self._alerts_lock:
            open_alerts = [a for a in self._alerts.values() if a.status == AlertStatus.OPEN]
        return sorted(open_alerts, key=lambda a: severity_order.get(a.severity, 99))

    # ── Alert Enrichment ─────────────────────────────────────────────────

    def enrich_alert(
        self,
        alert: Alert,
        customer_history: dict[str, Any] | None = None,
        recent_transactions: list[dict[str, Any]] | None = None,
        risk_profile: dict[str, Any] | None = None,
    ) -> Alert:
        """Enrich an alert with additional context.

        Args:
            alert: Alert to enrich.
            customer_history: Historical customer data.
            recent_transactions: Recent transactions for the account.
            risk_profile: Account risk profile.

        Returns:
            Enriched alert.
        """
        enrichment: dict[str, Any] = {}

        if customer_history:
            enrichment["customer_history"] = {
                "total_transactions": customer_history.get("total_transactions", 0),
                "account_age_days": customer_history.get("account_age_days", 0),
                "previous_alerts": customer_history.get("previous_alerts", 0),
                "average_transaction_amount": customer_history.get(
                    "average_transaction_amount", 0.0
                ),
            }

        if recent_transactions:
            enrichment["recent_transactions"] = [
                {
                    "transaction_id": txn.get("transaction_id", ""),
                    "amount": txn.get("transaction_amount", 0.0),
                    "timestamp": txn.get("transaction_timestamp", ""),
                    "merchant": txn.get("merchant_name", ""),
                    "channel": txn.get("channel", ""),
                }
                for txn in recent_transactions[:10]  # Limit to 10 most recent
            ]

        if risk_profile:
            enrichment["risk_profile"] = {
                "risk_tier": risk_profile.get("risk_tier", "standard"),
                "lifetime_risk_score": risk_profile.get("lifetime_risk_score", 0.0),
                "flagged_count": risk_profile.get("flagged_count", 0),
            }

        alert.enrichment = enrichment
        alert.updated_at = datetime.now(timezone.utc)
        return alert

    # ── Statistics ───────────────────────────────────────────────────────

    @property
    def statistics(self) -> AlertStatistics:
        return self._statistics

    def get_statistics_snapshot(self) -> dict[str, Any]:
        """Get current alert statistics."""
        return self._statistics.snapshot()

    # ── Internal Methods ─────────────────────────────────────────────────

    def _determine_alert_type(self, scoring_result: dict[str, Any]) -> AlertType:
        """Determine alert type from scoring method results."""
        method_scores = scoring_result.get("method_scores", [])
        if not method_scores:
            return AlertType.ENSEMBLE

        # Find the highest-contributing method
        max_score = 0.0
        dominant_method = "ensemble"
        for ms in method_scores:
            if ms.get("success") and ms.get("weighted_score", 0) > max_score:
                max_score = ms["weighted_score"]
                dominant_method = ms.get("method", "ensemble")

        method_map = {
            "rule_engine": AlertType.RULE_BASED,
            "anomaly_detection": AlertType.ANOMALY,
            "ml_model": AlertType.ML_SCORE,
        }
        return method_map.get(dominant_method, AlertType.ENSEMBLE)

    def _classify_severity(self, risk_score: float, risk_classification: str) -> AlertSeverity:
        """Classify alert severity from risk score."""
        # Use the classification from scoring if available
        classification_map = {
            "critical": AlertSeverity.CRITICAL,
            "high": AlertSeverity.HIGH,
            "medium": AlertSeverity.MEDIUM,
            "low": AlertSeverity.LOW,
        }
        if risk_classification in classification_map:
            return classification_map[risk_classification]

        # Fallback to score-based classification
        if risk_score >= 0.95:
            return AlertSeverity.CRITICAL
        elif risk_score >= 0.8:
            return AlertSeverity.HIGH
        elif risk_score >= 0.5:
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW

    def _extract_rule_id(self, scoring_result: dict[str, Any]) -> str | None:
        """Extract the primary triggered rule ID from scoring result."""
        method_scores = scoring_result.get("method_scores", [])
        for ms in method_scores:
            if ms.get("method") == "rule_engine" and ms.get("success"):
                details = ms.get("details", {})
                triggered_rules = details.get("triggered_rules", [])
                if triggered_rules:
                    return cast(str | None, triggered_rules[0].get("rule_id"))
        return None

    def _resolve_channels(self, severity: AlertSeverity) -> list[str]:
        """Resolve notification channels based on severity."""
        severity_routing = self._routing.get(severity.value, {})
        return cast(list[str], severity_routing.get("channels", ["dashboard"]))

    def _build_description(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        risk_score: float,
        transaction: dict[str, Any],
    ) -> str:
        """Build human-readable alert description."""
        amount = transaction.get("transaction_amount", 0.0)
        currency = transaction.get("transaction_currency", "USD")
        merchant = transaction.get("merchant_name", "Unknown")
        channel = transaction.get("channel", "unknown")

        type_desc = {
            AlertType.RULE_BASED: "Rule-based detection",
            AlertType.ANOMALY: "Anomaly detection",
            AlertType.ML_SCORE: "ML model prediction",
            AlertType.ENSEMBLE: "Ensemble scoring",
        }

        return (
            f"{type_desc.get(alert_type, 'Detection')} triggered "
            f"[{severity.value.upper()}] - "
            f"Risk score: {risk_score:.4f} | "
            f"Amount: {amount:.2f} {currency} | "
            f"Merchant: {merchant} | Channel: {channel}"
        )

    def _build_details(
        self, scoring_result: dict[str, Any], transaction: dict[str, Any]
    ) -> dict[str, Any]:
        """Build structured alert details from scoring and transaction."""
        return {
            "scoring": {
                "final_score": scoring_result.get("final_score"),
                "risk_classification": scoring_result.get("risk_classification"),
                "methods_succeeded": scoring_result.get("methods_succeeded"),
                "scoring_version": scoring_result.get("scoring_version"),
                "method_scores": scoring_result.get("method_scores", []),
            },
            "transaction": {
                "amount": transaction.get("transaction_amount"),
                "currency": transaction.get("transaction_currency"),
                "type": transaction.get("transaction_type"),
                "channel": transaction.get("channel"),
                "merchant": transaction.get("merchant_name"),
                "merchant_category": transaction.get("merchant_category_code"),
                "country": transaction.get("geo_country"),
                "is_international": transaction.get("is_international"),
                "timestamp": transaction.get("transaction_timestamp"),
            },
        }

    def _is_valid_transition(self, current: AlertStatus, target: AlertStatus) -> bool:
        """Check if a status transition is allowed."""
        allowed = self._transitions.get(current.value, [])
        if not allowed:
            # Default transitions if config is missing
            default_transitions = {
                "open": ["investigating", "resolved", "false_positive"],
                "investigating": ["resolved", "false_positive", "open"],
                "resolved": [],
                "false_positive": [],
            }
            allowed = default_transitions.get(current.value, [])
        return target.value in allowed

    def _publish_alert(self, alert: Alert) -> None:
        """Publish alert to Kafka topic for downstream consumers."""
        if self._kafka_producer is None:
            logger.debug("kafka_producer_not_configured", alert_id=alert.alert_id)
            return

        kafka_config = self._config.get("kafka", {})
        topic = kafka_config.get("topic", TOPIC_FRAUD_ALERTS)
        partition_key = alert.account_id

        message = alert.to_dict()
        headers = kafka_config.get("headers", {})
        message["_headers"] = {
            "source": headers.get("source", "riskpulse-alerting"),
            "version": headers.get("version", "1.0.0"),
            "severity": alert.severity.value,
            "alert_type": alert.alert_type.value,
        }

        try:
            self._kafka_producer.produce(
                topic=topic,
                key=(
                    partition_key.encode("utf-8")
                    if isinstance(partition_key, str)
                    else partition_key
                ),
                value=message,
            )
            self._statistics.record_published()
            logger.debug(
                "alert_published_to_kafka",
                alert_id=alert.alert_id,
                topic=topic,
            )
        except Exception as e:
            logger.error(
                "alert_publish_failed",
                alert_id=alert.alert_id,
                error=str(e),
            )

    # ── Persistence ──────────────────────────────────────────────────────

    async def persist_alert(self, alert: Alert) -> bool:
        """Persist alert to PostgreSQL.

        Args:
            alert: Alert to store.

        Returns:
            True if persisted successfully.
        """
        if self._db_session is None:
            logger.debug("db_session_not_configured", alert_id=alert.alert_id)
            return False

        try:
            query = """
                INSERT INTO fraud_alerts (
                    alert_id, transaction_id, alert_type, rule_id,
                    risk_score, severity, status, description, details
                ) VALUES (
                    :alert_id, :transaction_id, :alert_type, :rule_id,
                    :risk_score, :severity, :status, :description, :details
                )
                ON CONFLICT (alert_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    details = EXCLUDED.details,
                    updated_at = NOW()
            """
            import json

            params = {
                "alert_id": alert.alert_id,
                "transaction_id": alert.transaction_id,
                "alert_type": alert.alert_type.value,
                "rule_id": alert.rule_id,
                "risk_score": alert.risk_score,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "description": alert.description,
                "details": json.dumps({**alert.details, "enrichment": alert.enrichment}),
            }
            await self._db_session.execute(query, params)
            logger.debug("alert_persisted", alert_id=alert.alert_id)
            return True
        except Exception as e:
            logger.error(
                "alert_persist_failed",
                alert_id=alert.alert_id,
                error=str(e),
            )
            return False

    async def persist_batch(self, alerts: list[Alert]) -> int:
        """Persist a batch of alerts to PostgreSQL.

        Returns:
            Number of successfully persisted alerts.
        """
        success_count = 0
        for alert in alerts:
            if await self.persist_alert(alert):
                success_count += 1
        return success_count

    # ── Cleanup ──────────────────────────────────────────────────────────

    def clear_all(self) -> None:
        """Clear all internal state. Use for testing only."""
        with self._alerts_lock:
            self._alerts.clear()
        self._dedup_engine.clear()
        self._suppression_engine.clear()
        self._throttle_engine.clear()
