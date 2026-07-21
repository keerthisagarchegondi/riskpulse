"""Multi-Channel Notification Service — Production-grade notification delivery.

Delivers fraud alerts across multiple channels with:
- Email notifications (async, via AWS SES)
- Webhook notifications (Slack, Teams, PagerDuty)
- SMS notifications (critical alerts, via AWS SNS)
- In-app notifications (WebSocket push to dashboard)
- Notification templating and personalization
- Delivery tracking (sent, delivered, failed, read)
- Rate limiting to prevent alert fatigue
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

import yaml

from src.alerting.alert_manager import Alert, AlertChannel, AlertSeverity
from src.alerting.alert_templates import AlertTemplateRenderer, RenderedAlert
from src.utils.logger import get_logger

logger = get_logger(__name__, component="notification_service")

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


# ── Enums ────────────────────────────────────────────────────────────────────


class NotificationChannel(str, Enum):
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    IN_APP = "in_app"


class DeliveryStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    READ = "read"
    BOUNCED = "bounced"
    RATE_LIMITED = "rate_limited"


class WebhookTarget(str, Enum):
    SLACK = "slack"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"
    CUSTOM = "custom"


# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class NotificationRecord:
    """Tracks a single notification delivery attempt."""

    notification_id: str
    alert_id: str
    channel: NotificationChannel
    recipient: str
    status: DeliveryStatus
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sent_at: datetime | None = None
    delivered_at: datetime | None = None
    read_at: datetime | None = None
    failed_at: datetime | None = None
    failure_reason: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "notification_id": self.notification_id,
            "alert_id": self.alert_id,
            "channel": self.channel.value,
            "recipient": self.recipient,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "read_at": self.read_at.isoformat() if self.read_at else None,
            "failed_at": self.failed_at.isoformat() if self.failed_at else None,
            "failure_reason": self.failure_reason,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
        }


@dataclass
class NotificationPreference:
    """Per-user/team notification preferences."""

    user_id: str
    email_enabled: bool = True
    sms_enabled: bool = True
    webhook_enabled: bool = True
    in_app_enabled: bool = True
    quiet_hours_start: int | None = None  # Hour (0-23)
    quiet_hours_end: int | None = None
    min_severity_email: AlertSeverity = AlertSeverity.LOW
    min_severity_sms: AlertSeverity = AlertSeverity.HIGH
    min_severity_webhook: AlertSeverity = AlertSeverity.MEDIUM
    rate_limit_per_hour: int = 10
    digest_mode: bool = False
    digest_interval_minutes: int = 30


# ── Channel Providers (Protocol-based) ──────────────────────────────────────


class EmailProvider(Protocol):
    async def send_email(
        self, to: str, subject: str, body_html: str, body_text: str
    ) -> dict[str, Any]: ...


class SMSProvider(Protocol):
    async def send_sms(self, phone_number: str, message: str) -> dict[str, Any]: ...


class WebhookProvider(Protocol):
    async def send_webhook(
        self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]: ...


class WebSocketProvider(Protocol):
    async def push_notification(
        self, user_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...


# ── Default Provider Implementations ────────────────────────────────────────


class SESEmailProvider:
    """AWS SES email provider."""

    def __init__(self, region: str = "us-east-1", sender: str = "alerts@riskpulse.io"):
        self._region = region
        self._sender = sender
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("ses", region_name=self._region)
        return self._client

    async def send_email(
        self, to: str, subject: str, body_html: str, body_text: str
    ) -> dict[str, Any]:
        """Send email via AWS SES."""
        client = self._get_client()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.send_email(
                Source=self._sender,
                Destination={"ToAddresses": [to]},
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": {
                        "Html": {"Data": body_html, "Charset": "UTF-8"},
                        "Text": {"Data": body_text, "Charset": "UTF-8"},
                    },
                },
            ),
        )
        return {"message_id": response["MessageId"], "status": "sent"}


class SNSSMSProvider:
    """AWS SNS SMS provider."""

    def __init__(self, region: str = "us-east-1"):
        self._region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("sns", region_name=self._region)
        return self._client

    async def send_sms(self, phone_number: str, message: str) -> dict[str, Any]:
        """Send SMS via AWS SNS."""
        client = self._get_client()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.publish(
                PhoneNumber=phone_number,
                Message=message[:160],
                MessageAttributes={
                    "AWS.SNS.SMS.SMSType": {
                        "DataType": "String",
                        "StringValue": "Transactional",
                    }
                },
            ),
        )
        return {"message_id": response["MessageId"], "status": "sent"}


class HTTPWebhookProvider:
    """HTTP webhook provider for Slack, Teams, PagerDuty."""

    def __init__(self, timeout_seconds: float = 10.0):
        self._timeout = timeout_seconds

    async def send_webhook(
        self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Send webhook via HTTP POST."""
        import httpx

        default_headers = {"Content-Type": "application/json"}
        if headers:
            default_headers.update(headers)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload, headers=default_headers)
            response.raise_for_status()
            return {"status_code": response.status_code, "status": "delivered"}


class InAppWebSocketProvider:
    """In-app WebSocket notification provider."""

    def __init__(self):
        self._connections: dict[str, Any] = {}

    def register_connection(self, user_id: str, websocket: Any) -> None:
        self._connections[user_id] = websocket

    def remove_connection(self, user_id: str) -> None:
        self._connections.pop(user_id, None)

    async def push_notification(
        self, user_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Push notification to connected WebSocket client."""
        import json

        ws = self._connections.get(user_id)
        if ws is None:
            return {"status": "user_offline", "queued": True}

        message = json.dumps({"type": "notification", "data": payload})
        await ws.send_text(message)
        return {"status": "delivered"}


# ── Rate Limiter ─────────────────────────────────────────────────────────────


class NotificationRateLimiter:
    """Token-bucket rate limiter for notification delivery.

    Prevents alert fatigue by limiting notifications per recipient per hour.
    """

    def __init__(self, default_limit_per_hour: int = 10):
        self._default_limit = default_limit_per_hour
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def _cleanup_bucket(self, key: str, now: float) -> None:
        """Remove entries older than 1 hour."""
        cutoff = now - 3600.0
        self._buckets[key] = [t for t in self._buckets[key] if t > cutoff]

    def is_allowed(self, recipient: str, limit: int | None = None) -> bool:
        """Check if a notification can be sent to this recipient.

        Returns True if the recipient hasn't exceeded their hourly limit.
        """
        max_allowed = limit if limit is not None else self._default_limit
        now = time.time()

        with self._lock:
            self._cleanup_bucket(recipient, now)
            if len(self._buckets[recipient]) >= max_allowed:
                return False
            self._buckets[recipient].append(now)
            return True

    def remaining(self, recipient: str, limit: int | None = None) -> int:
        """Get remaining notifications allowed for recipient in current window."""
        max_allowed = limit if limit is not None else self._default_limit
        now = time.time()

        with self._lock:
            self._cleanup_bucket(recipient, now)
            return max(0, max_allowed - len(self._buckets[recipient]))

    def reset(self, recipient: str) -> None:
        """Reset rate limit for a recipient (e.g., after escalation override)."""
        with self._lock:
            self._buckets.pop(recipient, None)

    def get_stats(self) -> dict[str, int]:
        """Get current rate limiter stats."""
        now = time.time()
        with self._lock:
            stats = {}
            for key in list(self._buckets.keys()):
                self._cleanup_bucket(key, now)
                stats[key] = len(self._buckets[key])
            return stats


# ── Delivery Tracker ─────────────────────────────────────────────────────────


class DeliveryTracker:
    """Tracks notification delivery status and provides audit trail."""

    def __init__(self):
        self._records: dict[str, NotificationRecord] = {}
        self._by_alert: dict[str, list[str]] = defaultdict(list)
        self._lock = Lock()

    def create_record(
        self,
        alert_id: str,
        channel: NotificationChannel,
        recipient: str,
        metadata: dict[str, Any] | None = None,
    ) -> NotificationRecord:
        """Create a new notification tracking record."""
        record = NotificationRecord(
            notification_id=str(uuid.uuid4()),
            alert_id=alert_id,
            channel=channel,
            recipient=recipient,
            status=DeliveryStatus.PENDING,
            metadata=metadata or {},
        )
        with self._lock:
            self._records[record.notification_id] = record
            self._by_alert[alert_id].append(record.notification_id)
        return record

    def mark_sent(self, notification_id: str) -> None:
        """Mark notification as sent."""
        with self._lock:
            record = self._records.get(notification_id)
            if record:
                record.status = DeliveryStatus.SENT
                record.sent_at = datetime.now(timezone.utc)

    def mark_delivered(self, notification_id: str) -> None:
        """Mark notification as delivered."""
        with self._lock:
            record = self._records.get(notification_id)
            if record:
                record.status = DeliveryStatus.DELIVERED
                record.delivered_at = datetime.now(timezone.utc)

    def mark_failed(self, notification_id: str, reason: str) -> None:
        """Mark notification as failed."""
        with self._lock:
            record = self._records.get(notification_id)
            if record:
                record.status = DeliveryStatus.FAILED
                record.failed_at = datetime.now(timezone.utc)
                record.failure_reason = reason

    def mark_read(self, notification_id: str) -> None:
        """Mark notification as read by recipient."""
        with self._lock:
            record = self._records.get(notification_id)
            if record:
                record.status = DeliveryStatus.READ
                record.read_at = datetime.now(timezone.utc)

    def mark_rate_limited(self, notification_id: str) -> None:
        """Mark notification as rate-limited."""
        with self._lock:
            record = self._records.get(notification_id)
            if record:
                record.status = DeliveryStatus.RATE_LIMITED

    def get_record(self, notification_id: str) -> NotificationRecord | None:
        with self._lock:
            return self._records.get(notification_id)

    def get_records_for_alert(self, alert_id: str) -> list[NotificationRecord]:
        """Get all notification records for a given alert."""
        with self._lock:
            ids = self._by_alert.get(alert_id, [])
            return [self._records[nid] for nid in ids if nid in self._records]

    def get_delivery_stats(self) -> dict[str, int]:
        """Get aggregate delivery statistics."""
        with self._lock:
            stats: dict[str, int] = defaultdict(int)
            for record in self._records.values():
                stats[record.status.value] += 1
            return dict(stats)

    def get_failed_notifications(
        self, since: datetime | None = None
    ) -> list[NotificationRecord]:
        """Get all failed notifications, optionally since a given time."""
        with self._lock:
            failed = []
            for record in self._records.values():
                if record.status == DeliveryStatus.FAILED:
                    if since is None or (record.failed_at and record.failed_at >= since):
                        failed.append(record)
            return failed


# ── Notification Preferences Manager ────────────────────────────────────────


class PreferencesManager:
    """Manages notification preferences per user/team."""

    def __init__(self, config_path: Path | None = None):
        self._config_path = config_path or _CONFIG_DIR / "notification_preferences.yaml"
        self._preferences: dict[str, NotificationPreference] = {}
        self._defaults: dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load preferences from YAML configuration."""
        if self._config_path.exists():
            with open(self._config_path, "r") as f:
                config = yaml.safe_load(f) or {}

            self._defaults = config.get("defaults", {})

            for user_config in config.get("users", []):
                pref = self._build_preference(user_config)
                self._preferences[pref.user_id] = pref

            for team_config in config.get("teams", []):
                pref = self._build_preference(team_config)
                self._preferences[pref.user_id] = pref

    def _build_preference(self, config: dict[str, Any]) -> NotificationPreference:
        """Build a NotificationPreference from config dict."""
        severity_map = {
            "low": AlertSeverity.LOW,
            "medium": AlertSeverity.MEDIUM,
            "high": AlertSeverity.HIGH,
            "critical": AlertSeverity.CRITICAL,
        }
        return NotificationPreference(
            user_id=config["id"],
            email_enabled=config.get("email_enabled", True),
            sms_enabled=config.get("sms_enabled", True),
            webhook_enabled=config.get("webhook_enabled", True),
            in_app_enabled=config.get("in_app_enabled", True),
            quiet_hours_start=config.get("quiet_hours_start"),
            quiet_hours_end=config.get("quiet_hours_end"),
            min_severity_email=severity_map.get(
                config.get("min_severity_email", "low"), AlertSeverity.LOW
            ),
            min_severity_sms=severity_map.get(
                config.get("min_severity_sms", "high"), AlertSeverity.HIGH
            ),
            min_severity_webhook=severity_map.get(
                config.get("min_severity_webhook", "medium"), AlertSeverity.MEDIUM
            ),
            rate_limit_per_hour=config.get(
                "rate_limit_per_hour",
                self._defaults.get("rate_limit_per_hour", 10),
            ),
            digest_mode=config.get("digest_mode", False),
            digest_interval_minutes=config.get("digest_interval_minutes", 30),
        )

    def get_preference(self, user_id: str) -> NotificationPreference:
        """Get preference for user, returning defaults if not configured."""
        if user_id in self._preferences:
            return self._preferences[user_id]
        return NotificationPreference(
            user_id=user_id,
            rate_limit_per_hour=self._defaults.get("rate_limit_per_hour", 10),
        )

    def update_preference(self, preference: NotificationPreference) -> None:
        """Update preference for a user/team."""
        self._preferences[preference.user_id] = preference

    def is_channel_enabled(
        self, user_id: str, channel: NotificationChannel, severity: AlertSeverity
    ) -> bool:
        """Check if a channel is enabled for user at given severity."""
        pref = self.get_preference(user_id)
        severity_order = {
            AlertSeverity.LOW: 0,
            AlertSeverity.MEDIUM: 1,
            AlertSeverity.HIGH: 2,
            AlertSeverity.CRITICAL: 3,
        }
        alert_level = severity_order[severity]

        if channel == NotificationChannel.EMAIL:
            return pref.email_enabled and alert_level >= severity_order[pref.min_severity_email]
        elif channel == NotificationChannel.SMS:
            return pref.sms_enabled and alert_level >= severity_order[pref.min_severity_sms]
        elif channel == NotificationChannel.WEBHOOK:
            return pref.webhook_enabled and alert_level >= severity_order[pref.min_severity_webhook]
        elif channel == NotificationChannel.IN_APP:
            return pref.in_app_enabled
        return False

    def is_in_quiet_hours(self, user_id: str) -> bool:
        """Check if current time falls in user's quiet hours."""
        pref = self.get_preference(user_id)
        if pref.quiet_hours_start is None or pref.quiet_hours_end is None:
            return False

        current_hour = datetime.now(timezone.utc).hour
        start = pref.quiet_hours_start
        end = pref.quiet_hours_end

        if start <= end:
            return start <= current_hour < end
        else:
            # Wraps midnight (e.g., 22:00 - 06:00)
            return current_hour >= start or current_hour < end


# ── Webhook Formatters ───────────────────────────────────────────────────────


class WebhookFormatter:
    """Formats alert payloads for different webhook targets."""

    @staticmethod
    def format_slack(alert: Alert, rendered: RenderedAlert) -> dict[str, Any]:
        """Format alert as Slack Block Kit message."""
        severity_emoji = {
            AlertSeverity.LOW: ":white_circle:",
            AlertSeverity.MEDIUM: ":large_yellow_circle:",
            AlertSeverity.HIGH: ":large_orange_circle:",
            AlertSeverity.CRITICAL: ":red_circle:",
        }
        emoji = severity_emoji.get(alert.severity, ":warning:")

        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} {rendered.subject}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Severity:* {alert.severity.value.upper()}"},
                        {"type": "mrkdwn", "text": f"*Risk Score:* {alert.risk_score:.2f}"},
                        {"type": "mrkdwn", "text": f"*Account:* {alert.account_id}"},
                        {"type": "mrkdwn", "text": f"*Transaction:* {alert.transaction_id}"},
                    ],
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": rendered.body},
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Investigate"},
                            "style": "danger",
                            "value": alert.alert_id,
                            "action_id": "investigate_alert",
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Acknowledge"},
                            "value": alert.alert_id,
                            "action_id": "acknowledge_alert",
                        },
                    ],
                },
            ],
        }

    @staticmethod
    def format_teams(alert: Alert, rendered: RenderedAlert) -> dict[str, Any]:
        """Format alert as Microsoft Teams Adaptive Card."""
        severity_color = {
            AlertSeverity.LOW: "good",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.HIGH: "attention",
            AlertSeverity.CRITICAL: "attention",
        }

        return {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": rendered.subject,
                                "weight": "bolder",
                                "size": "large",
                                "color": severity_color.get(alert.severity, "default"),
                            },
                            {
                                "type": "FactSet",
                                "facts": [
                                    {"title": "Severity", "value": alert.severity.value.upper()},
                                    {"title": "Risk Score", "value": f"{alert.risk_score:.2f}"},
                                    {"title": "Account", "value": alert.account_id},
                                    {"title": "Transaction", "value": alert.transaction_id},
                                ],
                            },
                            {
                                "type": "TextBlock",
                                "text": rendered.body,
                                "wrap": True,
                            },
                        ],
                        "actions": [
                            {
                                "type": "Action.OpenUrl",
                                "title": "Investigate",
                                "url": f"https://dashboard.riskpulse.io/alerts/{alert.alert_id}",
                            },
                        ],
                    },
                }
            ],
        }

    @staticmethod
    def format_pagerduty(alert: Alert, rendered: RenderedAlert) -> dict[str, Any]:
        """Format alert as PagerDuty Events API v2 payload."""
        severity_map = {
            AlertSeverity.LOW: "info",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.HIGH: "error",
            AlertSeverity.CRITICAL: "critical",
        }

        return {
            "routing_key": "",  # Set by caller from config
            "event_action": "trigger",
            "dedup_key": f"riskpulse-{alert.alert_id}",
            "payload": {
                "summary": rendered.subject,
                "severity": severity_map.get(alert.severity, "warning"),
                "source": "riskpulse-fraud-detection",
                "component": "alert-engine",
                "group": f"account-{alert.account_id}",
                "class": alert.alert_type.value,
                "custom_details": {
                    "alert_id": alert.alert_id,
                    "transaction_id": alert.transaction_id,
                    "account_id": alert.account_id,
                    "risk_score": alert.risk_score,
                    "description": rendered.body,
                },
            },
            "links": [
                {
                    "href": f"https://dashboard.riskpulse.io/alerts/{alert.alert_id}",
                    "text": "View in RiskPulse",
                }
            ],
        }


# ── Main Notification Service ────────────────────────────────────────────────


class NotificationService:
    """Orchestrates multi-channel notification delivery with rate limiting.

    Handles routing, template rendering, rate limiting, preference checking,
    and delivery tracking for all alert notifications.
    """

    def __init__(
        self,
        email_provider: EmailProvider | None = None,
        sms_provider: SMSProvider | None = None,
        webhook_provider: WebhookProvider | None = None,
        websocket_provider: WebSocketProvider | None = None,
        preferences_manager: PreferencesManager | None = None,
        rate_limiter: NotificationRateLimiter | None = None,
        delivery_tracker: DeliveryTracker | None = None,
        template_renderer: AlertTemplateRenderer | None = None,
        config_path: Path | None = None,
    ):
        self._email_provider = email_provider
        self._sms_provider = sms_provider
        self._webhook_provider = webhook_provider or HTTPWebhookProvider()
        self._websocket_provider = websocket_provider
        self._preferences = preferences_manager or PreferencesManager()
        self._rate_limiter = rate_limiter or NotificationRateLimiter()
        self._tracker = delivery_tracker or DeliveryTracker()
        self._renderer = template_renderer or AlertTemplateRenderer()
        self._webhook_formatter = WebhookFormatter()

        self._config = self._load_config(config_path)
        self._webhook_urls: dict[str, str] = self._config.get("webhook_urls", {})
        self._routing: dict[str, list[str]] = self._config.get("routing", {})

    def _load_config(self, config_path: Path | None = None) -> dict[str, Any]:
        """Load notification service configuration."""
        path = config_path or _CONFIG_DIR / "notification_preferences.yaml"
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    async def notify(
        self,
        alert: Alert,
        recipients: list[str],
        channels: list[NotificationChannel] | None = None,
        override_rate_limit: bool = False,
    ) -> list[NotificationRecord]:
        """Send notifications for an alert to specified recipients.

        Args:
            alert: The alert to notify about.
            recipients: List of user/team IDs to notify.
            channels: Optional override of channels (otherwise determined by severity).
            override_rate_limit: Skip rate limiting (for escalations).

        Returns:
            List of NotificationRecord tracking delivery status.
        """
        records: list[NotificationRecord] = []
        target_channels = channels or self._resolve_channels(alert)

        for recipient in recipients:
            for channel in target_channels:
                rendered = self._render_for_channel(alert, channel)
                record = await self._deliver_to_channel(
                    alert=alert,
                    rendered=rendered,
                    recipient=recipient,
                    channel=channel,
                    override_rate_limit=override_rate_limit,
                )
                records.append(record)

        logger.info(
            "notification_batch_complete",
            alert_id=alert.alert_id,
            total_notifications=len(records),
            channels=[c.value for c in target_channels],
            recipients=len(recipients),
        )
        return records

    def _render_for_channel(self, alert: Alert, channel: NotificationChannel) -> RenderedAlert:
        """Render alert content for a specific notification channel."""
        channel_map = {
            NotificationChannel.EMAIL: "email",
            NotificationChannel.SMS: "sms",
            NotificationChannel.WEBHOOK: "webhook",
            NotificationChannel.IN_APP: "dashboard",
        }
        return self._renderer.render(alert, channel_map.get(channel, "dashboard"))

    async def _deliver_to_channel(
        self,
        alert: Alert,
        rendered: RenderedAlert,
        recipient: str,
        channel: NotificationChannel,
        override_rate_limit: bool = False,
    ) -> NotificationRecord:
        """Deliver a single notification to a specific channel."""
        record = self._tracker.create_record(
            alert_id=alert.alert_id,
            channel=channel,
            recipient=recipient,
            metadata={"severity": alert.severity.value},
        )

        # Check preferences
        if not self._preferences.is_channel_enabled(recipient, channel, alert.severity):
            self._tracker.mark_failed(record.notification_id, "channel_disabled_by_preference")
            logger.debug(
                "notification_skipped_preference",
                recipient=recipient,
                channel=channel.value,
            )
            return record

        # Check quiet hours (skip for critical)
        if (
            alert.severity != AlertSeverity.CRITICAL
            and self._preferences.is_in_quiet_hours(recipient)
        ):
            self._tracker.mark_failed(record.notification_id, "quiet_hours")
            logger.debug("notification_skipped_quiet_hours", recipient=recipient)
            return record

        # Check rate limit
        if not override_rate_limit:
            pref = self._preferences.get_preference(recipient)
            if not self._rate_limiter.is_allowed(recipient, pref.rate_limit_per_hour):
                self._tracker.mark_rate_limited(record.notification_id)
                logger.warning(
                    "notification_rate_limited",
                    recipient=recipient,
                    channel=channel.value,
                    alert_id=alert.alert_id,
                )
                return record

        # Dispatch to channel
        try:
            if channel == NotificationChannel.EMAIL:
                await self._send_email(alert, rendered, recipient, record)
            elif channel == NotificationChannel.SMS:
                await self._send_sms(alert, rendered, recipient, record)
            elif channel == NotificationChannel.WEBHOOK:
                await self._send_webhook(alert, rendered, recipient, record)
            elif channel == NotificationChannel.IN_APP:
                await self._send_in_app(alert, rendered, recipient, record)
        except Exception as e:
            self._tracker.mark_failed(record.notification_id, str(e))
            logger.error(
                "notification_delivery_failed",
                channel=channel.value,
                recipient=recipient,
                alert_id=alert.alert_id,
                error=str(e),
            )

        return record

    async def _send_email(
        self,
        alert: Alert,
        rendered: RenderedAlert,
        recipient: str,
        record: NotificationRecord,
    ) -> None:
        """Send email notification."""
        if self._email_provider is None:
            raise RuntimeError("Email provider not configured")

        result = await self._email_provider.send_email(
            to=recipient,
            subject=rendered.subject,
            body_html=rendered.html_body or rendered.body,
            body_text=rendered.body,
        )
        self._tracker.mark_sent(record.notification_id)
        record.metadata["provider_message_id"] = result.get("message_id")
        logger.info(
            "email_sent",
            recipient=recipient,
            alert_id=alert.alert_id,
            message_id=result.get("message_id"),
        )

    async def _send_sms(
        self,
        alert: Alert,
        rendered: RenderedAlert,
        recipient: str,
        record: NotificationRecord,
    ) -> None:
        """Send SMS notification."""
        if self._sms_provider is None:
            raise RuntimeError("SMS provider not configured")

        sms_text = (
            f"[RiskPulse {alert.severity.value.upper()}] "
            f"Alert {alert.alert_id[:8]}: {rendered.subject} "
            f"Score: {alert.risk_score:.2f}"
        )

        result = await self._sms_provider.send_sms(
            phone_number=recipient,
            message=sms_text,
        )
        self._tracker.mark_sent(record.notification_id)
        record.metadata["provider_message_id"] = result.get("message_id")
        logger.info("sms_sent", recipient="***", alert_id=alert.alert_id)

    async def _send_webhook(
        self,
        alert: Alert,
        rendered: RenderedAlert,
        recipient: str,
        record: NotificationRecord,
    ) -> None:
        """Send webhook notification (Slack, Teams, PagerDuty)."""
        webhook_config = self._get_webhook_config(recipient)
        if not webhook_config:
            raise RuntimeError(f"No webhook configuration for recipient: {recipient}")

        target = WebhookTarget(webhook_config.get("type", "custom"))
        url = webhook_config["url"]

        if target == WebhookTarget.SLACK:
            payload = self._webhook_formatter.format_slack(alert, rendered)
        elif target == WebhookTarget.TEAMS:
            payload = self._webhook_formatter.format_teams(alert, rendered)
        elif target == WebhookTarget.PAGERDUTY:
            payload = self._webhook_formatter.format_pagerduty(alert, rendered)
            payload["routing_key"] = webhook_config.get("routing_key", "")
        else:
            payload = {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "title": rendered.subject,
                "body": rendered.body,
                "risk_score": alert.risk_score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        headers = webhook_config.get("headers")
        result = await self._webhook_provider.send_webhook(url, payload, headers)
        self._tracker.mark_delivered(record.notification_id)
        logger.info(
            "webhook_delivered",
            target=target.value,
            alert_id=alert.alert_id,
            status_code=result.get("status_code"),
        )

    async def _send_in_app(
        self,
        alert: Alert,
        rendered: RenderedAlert,
        recipient: str,
        record: NotificationRecord,
    ) -> None:
        """Send in-app WebSocket notification."""
        if self._websocket_provider is None:
            raise RuntimeError("WebSocket provider not configured")

        payload = {
            "notification_id": record.notification_id,
            "alert_id": alert.alert_id,
            "title": rendered.subject,
            "body": rendered.body,
            "severity": alert.severity.value,
            "risk_score": alert.risk_score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions": ["investigate", "acknowledge", "dismiss"],
        }

        result = await self._websocket_provider.push_notification(recipient, payload)
        if result.get("status") == "delivered":
            self._tracker.mark_delivered(record.notification_id)
        else:
            self._tracker.mark_sent(record.notification_id)
            record.metadata["queued"] = True

        logger.info(
            "in_app_notification",
            recipient=recipient,
            alert_id=alert.alert_id,
            status=result.get("status"),
        )

    def _resolve_channels(self, alert: Alert) -> list[NotificationChannel]:
        """Determine channels based on alert severity and routing config."""
        severity_channels = {
            AlertSeverity.LOW: [NotificationChannel.EMAIL],
            AlertSeverity.MEDIUM: [NotificationChannel.EMAIL, NotificationChannel.IN_APP],
            AlertSeverity.HIGH: [
                NotificationChannel.EMAIL,
                NotificationChannel.SMS,
                NotificationChannel.IN_APP,
            ],
            AlertSeverity.CRITICAL: [
                NotificationChannel.EMAIL,
                NotificationChannel.SMS,
                NotificationChannel.WEBHOOK,
                NotificationChannel.IN_APP,
            ],
        }
        return severity_channels.get(alert.severity, [NotificationChannel.EMAIL])

    def _get_webhook_config(self, recipient: str) -> dict[str, Any] | None:
        """Get webhook configuration for a recipient."""
        webhooks = self._config.get("webhooks", {})
        return webhooks.get(recipient)

    def get_delivery_stats(self) -> dict[str, Any]:
        """Get delivery statistics."""
        return {
            "delivery": self._tracker.get_delivery_stats(),
            "rate_limiter": self._rate_limiter.get_stats(),
        }

    def mark_notification_read(self, notification_id: str) -> None:
        """Mark a notification as read (e.g., user opened it)."""
        self._tracker.mark_read(notification_id)

    def get_notifications_for_alert(self, alert_id: str) -> list[NotificationRecord]:
        """Get all notifications sent for a specific alert."""
        return self._tracker.get_records_for_alert(alert_id)

    @property
    def tracker(self) -> DeliveryTracker:
        return self._tracker

    @property
    def rate_limiter(self) -> NotificationRateLimiter:
        return self._rate_limiter

    @property
    def preferences(self) -> PreferencesManager:
        return self._preferences
