"""Escalation Engine — Time-based alert escalation with on-call routing.

Production-grade escalation system with:
- Time-based escalation (unacknowledged after timeout → escalate)
- Severity-based routing to appropriate team/individual
- On-call schedule integration
- Multi-level escalation chains
- Escalation audit trail
- Configurable policies per severity/team
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import yaml

from src.alerting.alert_manager import Alert, AlertSeverity, AlertStatus
from src.utils.logger import get_logger

logger = get_logger(__name__, component="escalation_engine")

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


# ── Enums ────────────────────────────────────────────────────────────────────


class EscalationLevel(int, Enum):
    L1 = 1  # First responder / analyst
    L2 = 2  # Senior analyst / team lead
    L3 = 3  # Manager / incident commander
    L4 = 4  # VP / executive escalation


class EscalationStatus(str, Enum):
    PENDING = "pending"
    ESCALATED = "escalated"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    TIMED_OUT = "timed_out"


class EscalationAction(str, Enum):
    NOTIFY = "notify"
    REASSIGN = "reassign"
    PAGE = "page"
    CONFERENCE = "conference"


# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class EscalationPolicy:
    """Defines an escalation policy with multiple levels."""

    policy_id: str
    name: str
    description: str = ""
    levels: list[EscalationLevelConfig] = field(default_factory=list)
    applies_to_severities: list[AlertSeverity] = field(default_factory=list)
    enabled: bool = True

    def get_level_config(self, level: EscalationLevel) -> EscalationLevelConfig | None:
        """Get configuration for a specific escalation level."""
        for lc in self.levels:
            if lc.level == level:
                return lc
        return None

    def get_next_level(self, current: EscalationLevel) -> EscalationLevel | None:
        """Get the next escalation level after current."""
        sorted_levels = sorted(self.levels, key=lambda x: x.level.value)
        for i, lc in enumerate(sorted_levels):
            if lc.level == current and i + 1 < len(sorted_levels):
                return sorted_levels[i + 1].level
        return None


@dataclass
class EscalationLevelConfig:
    """Configuration for a single escalation level."""

    level: EscalationLevel
    timeout_minutes: int
    recipients: list[str]
    actions: list[EscalationAction]
    notify_channels: list[str] = field(default_factory=lambda: ["email", "in_app"])
    auto_assign: bool = True


@dataclass
class OnCallSchedule:
    """On-call schedule for a team."""

    team_id: str
    team_name: str
    schedules: list[OnCallRotation] = field(default_factory=list)

    def get_current_oncall(self) -> str | None:
        """Get the currently on-call person."""
        now = datetime.now(timezone.utc)
        for rotation in self.schedules:
            if rotation.is_active(now):
                return rotation.get_oncall_person(now)
        return None


@dataclass
class OnCallRotation:
    """A single on-call rotation within a schedule."""

    rotation_id: str
    members: list[str]
    rotation_interval_hours: int = 168  # Weekly default
    start_time: datetime = field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))
    end_time: datetime | None = None

    def is_active(self, at: datetime) -> bool:
        """Check if this rotation is active at a given time."""
        if at < self.start_time:
            return False
        if self.end_time and at > self.end_time:
            return False
        return True

    def get_oncall_person(self, at: datetime) -> str:
        """Determine who is on-call at a given time based on rotation."""
        if not self.members:
            return ""
        elapsed_hours = (at - self.start_time).total_seconds() / 3600
        rotation_index = int(elapsed_hours / self.rotation_interval_hours) % len(self.members)
        return self.members[rotation_index]


@dataclass
class EscalationRecord:
    """Tracks a single escalation event."""

    escalation_id: str
    alert_id: str
    policy_id: str
    current_level: EscalationLevel
    status: EscalationStatus
    escalated_to: list[str]
    escalated_at: datetime
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    resolved_at: datetime | None = None
    next_escalation_at: datetime | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "escalation_id": self.escalation_id,
            "alert_id": self.alert_id,
            "policy_id": self.policy_id,
            "current_level": self.current_level.value,
            "status": self.status.value,
            "escalated_to": self.escalated_to,
            "escalated_at": self.escalated_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "next_escalation_at": self.next_escalation_at.isoformat() if self.next_escalation_at else None,
            "history": self.history,
            "metadata": self.metadata,
        }


# ── Escalation Engine ────────────────────────────────────────────────────────


class EscalationEngine:
    """Manages alert escalation lifecycle with time-based triggers.

    Monitors unacknowledged alerts and escalates them through configured
    escalation chains based on severity and timeout policies.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        on_escalate: Callable[[EscalationRecord, Alert], Any] | None = None,
    ):
        self._config_path = config_path or _CONFIG_DIR / "escalation_policies.yaml"
        self._on_escalate = on_escalate

        self._policies: dict[str, EscalationPolicy] = {}
        self._oncall_schedules: dict[str, OnCallSchedule] = {}
        self._active_escalations: dict[str, EscalationRecord] = {}
        self._escalation_by_alert: dict[str, str] = {}
        self._audit_log: list[dict[str, Any]] = []
        self._lock = Lock()

        self._load_config()

    def _load_config(self) -> None:
        """Load escalation policies from YAML."""
        if not self._config_path.exists():
            logger.warning("escalation_config_not_found", path=str(self._config_path))
            self._load_defaults()
            return

        with open(self._config_path, "r") as f:
            config = yaml.safe_load(f) or {}

        # Load policies
        for policy_cfg in config.get("policies", []):
            policy = self._parse_policy(policy_cfg)
            self._policies[policy.policy_id] = policy

        # Load on-call schedules
        for schedule_cfg in config.get("oncall_schedules", []):
            schedule = self._parse_oncall_schedule(schedule_cfg)
            self._oncall_schedules[schedule.team_id] = schedule

        logger.info(
            "escalation_config_loaded",
            policies=len(self._policies),
            schedules=len(self._oncall_schedules),
        )

    def _load_defaults(self) -> None:
        """Load default escalation policies when config not found."""
        default_policy = EscalationPolicy(
            policy_id="default",
            name="Default Escalation Policy",
            description="Default time-based escalation",
            applies_to_severities=[
                AlertSeverity.HIGH,
                AlertSeverity.CRITICAL,
            ],
            levels=[
                EscalationLevelConfig(
                    level=EscalationLevel.L1,
                    timeout_minutes=15,
                    recipients=["fraud-analysts"],
                    actions=[EscalationAction.NOTIFY],
                ),
                EscalationLevelConfig(
                    level=EscalationLevel.L2,
                    timeout_minutes=30,
                    recipients=["fraud-team-lead"],
                    actions=[EscalationAction.NOTIFY, EscalationAction.REASSIGN],
                ),
                EscalationLevelConfig(
                    level=EscalationLevel.L3,
                    timeout_minutes=60,
                    recipients=["fraud-manager"],
                    actions=[EscalationAction.PAGE, EscalationAction.REASSIGN],
                ),
            ],
        )
        self._policies["default"] = default_policy

    def _parse_policy(self, cfg: dict[str, Any]) -> EscalationPolicy:
        """Parse a policy configuration dict into an EscalationPolicy."""
        severity_map = {
            "low": AlertSeverity.LOW,
            "medium": AlertSeverity.MEDIUM,
            "high": AlertSeverity.HIGH,
            "critical": AlertSeverity.CRITICAL,
        }
        action_map = {
            "notify": EscalationAction.NOTIFY,
            "reassign": EscalationAction.REASSIGN,
            "page": EscalationAction.PAGE,
            "conference": EscalationAction.CONFERENCE,
        }
        level_map = {1: EscalationLevel.L1, 2: EscalationLevel.L2, 3: EscalationLevel.L3, 4: EscalationLevel.L4}

        levels = []
        for level_cfg in cfg.get("levels", []):
            levels.append(
                EscalationLevelConfig(
                    level=level_map[level_cfg["level"]],
                    timeout_minutes=level_cfg["timeout_minutes"],
                    recipients=level_cfg["recipients"],
                    actions=[action_map[a] for a in level_cfg.get("actions", ["notify"])],
                    notify_channels=level_cfg.get("notify_channels", ["email", "in_app"]),
                    auto_assign=level_cfg.get("auto_assign", True),
                )
            )

        return EscalationPolicy(
            policy_id=cfg["policy_id"],
            name=cfg["name"],
            description=cfg.get("description", ""),
            levels=levels,
            applies_to_severities=[
                severity_map[s] for s in cfg.get("applies_to_severities", ["high", "critical"])
            ],
            enabled=cfg.get("enabled", True),
        )

    def _parse_oncall_schedule(self, cfg: dict[str, Any]) -> OnCallSchedule:
        """Parse an on-call schedule configuration."""
        rotations = []
        for rot_cfg in cfg.get("rotations", []):
            rotations.append(
                OnCallRotation(
                    rotation_id=rot_cfg.get("rotation_id", str(uuid.uuid4())),
                    members=rot_cfg["members"],
                    rotation_interval_hours=rot_cfg.get("rotation_interval_hours", 168),
                )
            )

        return OnCallSchedule(
            team_id=cfg["team_id"],
            team_name=cfg["team_name"],
            schedules=rotations,
        )

    def get_policy_for_alert(self, alert: Alert) -> EscalationPolicy | None:
        """Find the applicable escalation policy for an alert."""
        for policy in self._policies.values():
            if not policy.enabled:
                continue
            if alert.severity in policy.applies_to_severities:
                return policy
        return None

    def start_escalation(self, alert: Alert, policy_id: str | None = None) -> EscalationRecord | None:
        """Start the escalation process for an alert.

        Args:
            alert: The alert to escalate.
            policy_id: Optional specific policy ID. If None, auto-selects based on severity.

        Returns:
            EscalationRecord if escalation started, None if no applicable policy.
        """
        if policy_id:
            policy = self._policies.get(policy_id)
        else:
            policy = self.get_policy_for_alert(alert)

        if not policy or not policy.levels:
            logger.debug("no_escalation_policy", alert_id=alert.alert_id)
            return None

        # Check if already escalating
        with self._lock:
            if alert.alert_id in self._escalation_by_alert:
                existing_id = self._escalation_by_alert[alert.alert_id]
                return self._active_escalations.get(existing_id)

        first_level = policy.levels[0]
        recipients = self._resolve_recipients(first_level.recipients)
        now = datetime.now(timezone.utc)

        record = EscalationRecord(
            escalation_id=str(uuid.uuid4()),
            alert_id=alert.alert_id,
            policy_id=policy.policy_id,
            current_level=first_level.level,
            status=EscalationStatus.PENDING,
            escalated_to=recipients,
            escalated_at=now,
            next_escalation_at=now + timedelta(minutes=first_level.timeout_minutes),
            history=[
                {
                    "timestamp": now.isoformat(),
                    "action": "escalation_started",
                    "level": first_level.level.value,
                    "recipients": recipients,
                    "policy": policy.policy_id,
                }
            ],
        )

        with self._lock:
            self._active_escalations[record.escalation_id] = record
            self._escalation_by_alert[alert.alert_id] = record.escalation_id

        self._record_audit(
            action="escalation_started",
            alert_id=alert.alert_id,
            escalation_id=record.escalation_id,
            level=first_level.level.value,
            recipients=recipients,
        )

        logger.info(
            "escalation_started",
            alert_id=alert.alert_id,
            policy=policy.policy_id,
            level=first_level.level.value,
            recipients=recipients,
            next_escalation_at=record.next_escalation_at.isoformat(),
        )

        return record

    def acknowledge(self, alert_id: str, acknowledged_by: str) -> EscalationRecord | None:
        """Acknowledge an escalation, stopping further escalation.

        Args:
            alert_id: The alert ID being acknowledged.
            acknowledged_by: The user/team acknowledging.

        Returns:
            Updated EscalationRecord or None if not found.
        """
        with self._lock:
            escalation_id = self._escalation_by_alert.get(alert_id)
            if not escalation_id:
                return None

            record = self._active_escalations.get(escalation_id)
            if not record:
                return None

            now = datetime.now(timezone.utc)
            record.status = EscalationStatus.ACKNOWLEDGED
            record.acknowledged_at = now
            record.acknowledged_by = acknowledged_by
            record.next_escalation_at = None

            record.history.append({
                "timestamp": now.isoformat(),
                "action": "acknowledged",
                "by": acknowledged_by,
                "level_at_ack": record.current_level.value,
            })

        self._record_audit(
            action="escalation_acknowledged",
            alert_id=alert_id,
            escalation_id=record.escalation_id,
            acknowledged_by=acknowledged_by,
        )

        logger.info(
            "escalation_acknowledged",
            alert_id=alert_id,
            acknowledged_by=acknowledged_by,
            level=record.current_level.value,
        )

        return record

    def resolve(self, alert_id: str) -> EscalationRecord | None:
        """Mark an escalation as resolved."""
        with self._lock:
            escalation_id = self._escalation_by_alert.get(alert_id)
            if not escalation_id:
                return None

            record = self._active_escalations.get(escalation_id)
            if not record:
                return None

            now = datetime.now(timezone.utc)
            record.status = EscalationStatus.RESOLVED
            record.resolved_at = now
            record.next_escalation_at = None

            record.history.append({
                "timestamp": now.isoformat(),
                "action": "resolved",
            })

            # Move to completed
            del self._escalation_by_alert[alert_id]

        self._record_audit(
            action="escalation_resolved",
            alert_id=alert_id,
            escalation_id=record.escalation_id,
        )

        logger.info("escalation_resolved", alert_id=alert_id)
        return record

    def check_timeouts(self) -> list[EscalationRecord]:
        """Check all active escalations for timeouts and escalate as needed.

        This should be called periodically (e.g., every minute via scheduler).

        Returns:
            List of escalation records that were escalated to next level.
        """
        now = datetime.now(timezone.utc)
        escalated: list[EscalationRecord] = []

        with self._lock:
            for record in list(self._active_escalations.values()):
                if record.status not in (EscalationStatus.PENDING, EscalationStatus.ESCALATED):
                    continue

                if record.next_escalation_at and now >= record.next_escalation_at:
                    upgraded = self._escalate_to_next_level(record, now)
                    if upgraded:
                        escalated.append(record)

        return escalated

    def _escalate_to_next_level(self, record: EscalationRecord, now: datetime) -> bool:
        """Escalate a record to the next level in the policy chain.

        Returns True if escalation occurred, False if max level reached.
        """
        policy = self._policies.get(record.policy_id)
        if not policy:
            return False

        next_level = policy.get_next_level(record.current_level)
        if next_level is None:
            # Max level reached
            record.status = EscalationStatus.TIMED_OUT
            record.next_escalation_at = None
            record.history.append({
                "timestamp": now.isoformat(),
                "action": "max_level_reached",
                "level": record.current_level.value,
            })
            self._record_audit(
                action="escalation_max_level",
                alert_id=record.alert_id,
                escalation_id=record.escalation_id,
                level=record.current_level.value,
            )
            logger.warning(
                "escalation_max_level_reached",
                alert_id=record.alert_id,
                level=record.current_level.value,
            )
            return False

        next_config = policy.get_level_config(next_level)
        if not next_config:
            return False

        recipients = self._resolve_recipients(next_config.recipients)
        record.current_level = next_level
        record.status = EscalationStatus.ESCALATED
        record.escalated_to = recipients
        record.next_escalation_at = now + timedelta(minutes=next_config.timeout_minutes)

        record.history.append({
            "timestamp": now.isoformat(),
            "action": "escalated",
            "from_level": (next_level.value - 1),
            "to_level": next_level.value,
            "recipients": recipients,
            "timeout_minutes": next_config.timeout_minutes,
        })

        self._record_audit(
            action="escalation_level_up",
            alert_id=record.alert_id,
            escalation_id=record.escalation_id,
            from_level=next_level.value - 1,
            to_level=next_level.value,
            recipients=recipients,
        )

        logger.info(
            "alert_escalated",
            alert_id=record.alert_id,
            from_level=next_level.value - 1,
            to_level=next_level.value,
            recipients=recipients,
            next_timeout_minutes=next_config.timeout_minutes,
        )

        # Trigger callback
        if self._on_escalate:
            try:
                self._on_escalate(record, None)
            except Exception as e:
                logger.error("escalation_callback_error", error=str(e))

        return True

    def _resolve_recipients(self, recipient_refs: list[str]) -> list[str]:
        """Resolve recipient references (team IDs → on-call persons)."""
        resolved = []
        for ref in recipient_refs:
            # Check if it's a team reference with on-call schedule
            schedule = self._oncall_schedules.get(ref)
            if schedule:
                oncall = schedule.get_current_oncall()
                if oncall:
                    resolved.append(oncall)
                else:
                    # Fallback to first member of first rotation
                    for rotation in schedule.schedules:
                        if rotation.members:
                            resolved.append(rotation.members[0])
                            break
            else:
                resolved.append(ref)
        return resolved

    def _record_audit(self, **kwargs: Any) -> None:
        """Record an audit trail entry."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._audit_log.append(entry)

    def get_active_escalations(self) -> list[EscalationRecord]:
        """Get all currently active escalations."""
        with self._lock:
            return [
                r
                for r in self._active_escalations.values()
                if r.status in (EscalationStatus.PENDING, EscalationStatus.ESCALATED)
            ]

    def get_escalation_for_alert(self, alert_id: str) -> EscalationRecord | None:
        """Get the escalation record for a specific alert."""
        with self._lock:
            escalation_id = self._escalation_by_alert.get(alert_id)
            if escalation_id:
                return self._active_escalations.get(escalation_id)
        return None

    def get_audit_log(
        self, alert_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get escalation audit trail, optionally filtered by alert."""
        if alert_id:
            entries = [e for e in self._audit_log if e.get("alert_id") == alert_id]
        else:
            entries = self._audit_log
        return entries[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get escalation engine statistics."""
        with self._lock:
            active = [
                r for r in self._active_escalations.values()
                if r.status in (EscalationStatus.PENDING, EscalationStatus.ESCALATED)
            ]
            acknowledged = [
                r for r in self._active_escalations.values()
                if r.status == EscalationStatus.ACKNOWLEDGED
            ]
            resolved = [
                r for r in self._active_escalations.values()
                if r.status == EscalationStatus.RESOLVED
            ]
            timed_out = [
                r for r in self._active_escalations.values()
                if r.status == EscalationStatus.TIMED_OUT
            ]

        return {
            "active_escalations": len(active),
            "acknowledged": len(acknowledged),
            "resolved": len(resolved),
            "timed_out": len(timed_out),
            "total_tracked": len(self._active_escalations),
            "policies_loaded": len(self._policies),
            "oncall_schedules": len(self._oncall_schedules),
            "audit_entries": len(self._audit_log),
        }
