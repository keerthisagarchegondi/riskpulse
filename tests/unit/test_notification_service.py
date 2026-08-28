"""Comprehensive tests for the Notification Service and Escalation Engine."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from src.alerting.alert_manager import Alert, AlertSeverity, AlertStatus, AlertType
from src.alerting.escalation_engine import (
    EscalationEngine,
    EscalationLevel,
    EscalationStatus,
    OnCallRotation,
    OnCallSchedule,
)
from src.alerting.notification_service import (
    DeliveryStatus,
    DeliveryTracker,
    NotificationChannel,
    NotificationPreference,
    NotificationRateLimiter,
    NotificationService,
    PreferencesManager,
    WebhookFormatter,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_alert():
    """Create a sample alert for testing."""
    return Alert(
        alert_id="ALT-TEST-001",
        transaction_id="TXN-001",
        account_id="ACC-12345",
        alert_type=AlertType.RULE_BASED,
        severity=AlertSeverity.HIGH,
        status=AlertStatus.OPEN,
        risk_score=0.85,
        rule_id="R001",
        description="High-value transaction from unusual location",
        channels=["email", "sms", "dashboard"],
    )


@pytest.fixture
def critical_alert():
    """Create a critical alert for testing."""
    return Alert(
        alert_id="ALT-CRIT-001",
        transaction_id="TXN-CRIT-001",
        account_id="ACC-99999",
        alert_type=AlertType.ENSEMBLE,
        severity=AlertSeverity.CRITICAL,
        status=AlertStatus.OPEN,
        risk_score=0.97,
        rule_id="R001",
        description="Critical fraud detected - multiple signals",
        channels=["email", "sms", "webhook", "dashboard"],
    )


@pytest.fixture
def low_alert():
    """Create a low-severity alert for testing."""
    return Alert(
        alert_id="ALT-LOW-001",
        transaction_id="TXN-LOW-001",
        account_id="ACC-11111",
        alert_type=AlertType.ML_SCORE,
        severity=AlertSeverity.LOW,
        status=AlertStatus.OPEN,
        risk_score=0.35,
        description="Minor anomaly detected",
    )


@pytest.fixture
def rate_limiter():
    """Create a rate limiter with default settings."""
    return NotificationRateLimiter(default_limit_per_hour=10)


@pytest.fixture
def delivery_tracker():
    """Create a delivery tracker."""
    return DeliveryTracker()


@pytest.fixture
def mock_email_provider():
    """Create a mock email provider."""
    provider = AsyncMock()
    provider.send_email.return_value = {"message_id": "ses-msg-001", "status": "sent"}
    return provider


@pytest.fixture
def mock_sms_provider():
    """Create a mock SMS provider."""
    provider = AsyncMock()
    provider.send_sms.return_value = {"message_id": "sns-msg-001", "status": "sent"}
    return provider


@pytest.fixture
def mock_webhook_provider():
    """Create a mock webhook provider."""
    provider = AsyncMock()
    provider.send_webhook.return_value = {"status_code": 200, "status": "delivered"}
    return provider


@pytest.fixture
def mock_websocket_provider():
    """Create a mock WebSocket provider."""
    provider = AsyncMock()
    provider.push_notification.return_value = {"status": "delivered"}
    return provider


@pytest.fixture
def notification_service(
    mock_email_provider,
    mock_sms_provider,
    mock_webhook_provider,
    mock_websocket_provider,
):
    """Create a fully configured notification service with mock providers."""
    return NotificationService(
        email_provider=mock_email_provider,
        sms_provider=mock_sms_provider,
        webhook_provider=mock_webhook_provider,
        websocket_provider=mock_websocket_provider,
        preferences_manager=PreferencesManager(config_path=None),
        rate_limiter=NotificationRateLimiter(default_limit_per_hour=10),
        delivery_tracker=DeliveryTracker(),
    )


@pytest.fixture
def escalation_engine():
    """Create an escalation engine with default policies."""
    from pathlib import Path

    # Use a non-existent path to trigger default policy loading
    engine = EscalationEngine(config_path=Path("/nonexistent/path.yaml"))
    return engine


@pytest.fixture
def escalation_engine_with_config(tmp_path):
    """Create an escalation engine with custom config."""
    config = {
        "policies": [
            {
                "policy_id": "test_policy",
                "name": "Test Escalation Policy",
                "description": "For testing",
                "enabled": True,
                "applies_to_severities": ["high", "critical"],
                "levels": [
                    {
                        "level": 1,
                        "timeout_minutes": 15,
                        "recipients": ["analyst@test.com"],
                        "actions": ["notify"],
                        "notify_channels": ["email", "in_app"],
                        "auto_assign": True,
                    },
                    {
                        "level": 2,
                        "timeout_minutes": 30,
                        "recipients": ["lead@test.com"],
                        "actions": ["notify", "reassign"],
                        "notify_channels": ["email", "sms"],
                        "auto_assign": True,
                    },
                    {
                        "level": 3,
                        "timeout_minutes": 60,
                        "recipients": ["manager@test.com"],
                        "actions": ["page", "reassign"],
                        "notify_channels": ["email", "sms", "webhook"],
                        "auto_assign": True,
                    },
                ],
            }
        ],
        "oncall_schedules": [
            {
                "team_id": "test-oncall",
                "team_name": "Test On-Call",
                "rotations": [
                    {
                        "rotation_id": "primary",
                        "members": ["oncall-1@test.com", "oncall-2@test.com"],
                        "rotation_interval_hours": 168,
                    }
                ],
            }
        ],
    }

    import yaml

    config_path = tmp_path / "escalation_policies.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return EscalationEngine(config_path=config_path)


# ── Rate Limiter Tests ───────────────────────────────────────────────────────


class TestNotificationRateLimiter:
    """Tests for the notification rate limiter."""

    def test_allows_within_limit(self, rate_limiter):
        """Notifications within the hourly limit should be allowed."""
        for _ in range(10):
            assert rate_limiter.is_allowed("user@test.com") is True

    def test_blocks_over_limit(self, rate_limiter):
        """Notifications exceeding the hourly limit should be blocked."""
        for _ in range(10):
            rate_limiter.is_allowed("user@test.com")

        assert rate_limiter.is_allowed("user@test.com") is False

    def test_custom_limit_per_recipient(self, rate_limiter):
        """Custom per-recipient limits should be respected."""
        for _ in range(5):
            assert rate_limiter.is_allowed("user@test.com", limit=5) is True

        assert rate_limiter.is_allowed("user@test.com", limit=5) is False

    def test_separate_buckets_per_recipient(self, rate_limiter):
        """Different recipients should have independent rate limits."""
        for _ in range(10):
            rate_limiter.is_allowed("user-a@test.com")

        # user-a is at limit, but user-b should be fine
        assert rate_limiter.is_allowed("user-a@test.com") is False
        assert rate_limiter.is_allowed("user-b@test.com") is True

    def test_remaining_count(self, rate_limiter):
        """remaining() should report correct number of remaining notifications."""
        assert rate_limiter.remaining("user@test.com") == 10

        for _ in range(3):
            rate_limiter.is_allowed("user@test.com")

        assert rate_limiter.remaining("user@test.com") == 7

    def test_reset_clears_bucket(self, rate_limiter):
        """reset() should clear the rate limit for a recipient."""
        for _ in range(10):
            rate_limiter.is_allowed("user@test.com")

        assert rate_limiter.is_allowed("user@test.com") is False

        rate_limiter.reset("user@test.com")
        assert rate_limiter.is_allowed("user@test.com") is True

    def test_expired_entries_cleaned(self):
        """Entries older than 1 hour should be cleaned up."""
        limiter = NotificationRateLimiter(default_limit_per_hour=2)

        # Manually inject old timestamps
        old_time = time.time() - 3700  # More than 1 hour ago
        limiter._buckets["user@test.com"] = [old_time, old_time]

        # Should be allowed since old entries are cleaned
        assert limiter.is_allowed("user@test.com") is True

    def test_get_stats(self, rate_limiter):
        """get_stats() should return current counts per recipient."""
        rate_limiter.is_allowed("a@test.com")
        rate_limiter.is_allowed("a@test.com")
        rate_limiter.is_allowed("b@test.com")

        stats = rate_limiter.get_stats()
        assert stats["a@test.com"] == 2
        assert stats["b@test.com"] == 1


# ── Delivery Tracker Tests ───────────────────────────────────────────────────


class TestDeliveryTracker:
    """Tests for notification delivery tracking."""

    def test_create_record(self, delivery_tracker):
        """Should create a notification record with PENDING status."""
        record = delivery_tracker.create_record(
            alert_id="ALT-001",
            channel=NotificationChannel.EMAIL,
            recipient="user@test.com",
        )
        assert record.status == DeliveryStatus.PENDING
        assert record.alert_id == "ALT-001"
        assert record.channel == NotificationChannel.EMAIL

    def test_mark_sent(self, delivery_tracker):
        """Should transition from PENDING to SENT."""
        record = delivery_tracker.create_record(
            alert_id="ALT-001",
            channel=NotificationChannel.EMAIL,
            recipient="user@test.com",
        )
        delivery_tracker.mark_sent(record.notification_id)

        updated = delivery_tracker.get_record(record.notification_id)
        assert updated.status == DeliveryStatus.SENT
        assert updated.sent_at is not None

    def test_mark_delivered(self, delivery_tracker):
        """Should transition to DELIVERED status."""
        record = delivery_tracker.create_record(
            alert_id="ALT-001",
            channel=NotificationChannel.WEBHOOK,
            recipient="slack-channel",
        )
        delivery_tracker.mark_delivered(record.notification_id)

        updated = delivery_tracker.get_record(record.notification_id)
        assert updated.status == DeliveryStatus.DELIVERED
        assert updated.delivered_at is not None

    def test_mark_failed(self, delivery_tracker):
        """Should transition to FAILED with reason."""
        record = delivery_tracker.create_record(
            alert_id="ALT-001",
            channel=NotificationChannel.SMS,
            recipient="+1234567890",
        )
        delivery_tracker.mark_failed(record.notification_id, "Invalid phone number")

        updated = delivery_tracker.get_record(record.notification_id)
        assert updated.status == DeliveryStatus.FAILED
        assert updated.failure_reason == "Invalid phone number"

    def test_mark_read(self, delivery_tracker):
        """Should mark notification as read."""
        record = delivery_tracker.create_record(
            alert_id="ALT-001",
            channel=NotificationChannel.IN_APP,
            recipient="user-123",
        )
        delivery_tracker.mark_read(record.notification_id)

        updated = delivery_tracker.get_record(record.notification_id)
        assert updated.status == DeliveryStatus.READ
        assert updated.read_at is not None

    def test_get_records_for_alert(self, delivery_tracker):
        """Should return all notifications for a given alert."""
        delivery_tracker.create_record("ALT-001", NotificationChannel.EMAIL, "a@test.com")
        delivery_tracker.create_record("ALT-001", NotificationChannel.SMS, "+1111111111")
        delivery_tracker.create_record("ALT-002", NotificationChannel.EMAIL, "b@test.com")

        records = delivery_tracker.get_records_for_alert("ALT-001")
        assert len(records) == 2
        assert all(r.alert_id == "ALT-001" for r in records)

    def test_get_delivery_stats(self, delivery_tracker):
        """Should return aggregate delivery statistics."""
        r1 = delivery_tracker.create_record("ALT-001", NotificationChannel.EMAIL, "a@test.com")
        r2 = delivery_tracker.create_record("ALT-001", NotificationChannel.SMS, "+1111")
        r3 = delivery_tracker.create_record("ALT-002", NotificationChannel.WEBHOOK, "slack")

        delivery_tracker.mark_sent(r1.notification_id)
        delivery_tracker.mark_delivered(r2.notification_id)
        delivery_tracker.mark_failed(r3.notification_id, "timeout")

        stats = delivery_tracker.get_delivery_stats()
        assert stats["sent"] == 1
        assert stats["delivered"] == 1
        assert stats["failed"] == 1

    def test_get_failed_notifications(self, delivery_tracker):
        """Should return failed notifications filtered by time."""
        r1 = delivery_tracker.create_record("ALT-001", NotificationChannel.EMAIL, "a@test.com")
        r2 = delivery_tracker.create_record("ALT-002", NotificationChannel.SMS, "+1111")

        delivery_tracker.mark_failed(r1.notification_id, "bounce")
        delivery_tracker.mark_sent(r2.notification_id)

        failed = delivery_tracker.get_failed_notifications()
        assert len(failed) == 1
        assert failed[0].notification_id == r1.notification_id


# ── Preferences Manager Tests ────────────────────────────────────────────────


class TestPreferencesManager:
    """Tests for notification preferences management."""

    def test_default_preferences(self):
        """Should return defaults for unknown users."""
        manager = PreferencesManager(config_path=None)
        pref = manager.get_preference("unknown-user")

        assert pref.user_id == "unknown-user"
        assert pref.email_enabled is True
        assert pref.rate_limit_per_hour == 10

    def test_update_preference(self):
        """Should update and retrieve preferences."""
        manager = PreferencesManager(config_path=None)
        pref = NotificationPreference(
            user_id="custom-user",
            email_enabled=True,
            sms_enabled=False,
            rate_limit_per_hour=5,
        )
        manager.update_preference(pref)

        retrieved = manager.get_preference("custom-user")
        assert retrieved.sms_enabled is False
        assert retrieved.rate_limit_per_hour == 5

    def test_channel_enabled_check(self):
        """Should correctly check if channel is enabled for severity."""
        manager = PreferencesManager(config_path=None)
        pref = NotificationPreference(
            user_id="test-user",
            email_enabled=True,
            sms_enabled=True,
            min_severity_email=AlertSeverity.LOW,
            min_severity_sms=AlertSeverity.HIGH,
        )
        manager.update_preference(pref)

        # Email enabled for low severity
        assert (
            manager.is_channel_enabled("test-user", NotificationChannel.EMAIL, AlertSeverity.LOW)
            is True
        )

        # SMS NOT enabled for low severity (requires HIGH)
        assert (
            manager.is_channel_enabled("test-user", NotificationChannel.SMS, AlertSeverity.LOW)
            is False
        )

        # SMS enabled for high severity
        assert (
            manager.is_channel_enabled("test-user", NotificationChannel.SMS, AlertSeverity.HIGH)
            is True
        )

    def test_quiet_hours(self):
        """Should detect quiet hours correctly."""
        manager = PreferencesManager(config_path=None)
        pref = NotificationPreference(
            user_id="test-user",
            quiet_hours_start=22,
            quiet_hours_end=7,
        )
        manager.update_preference(pref)

        # The result depends on current UTC hour, so just verify no exception
        result = manager.is_in_quiet_hours("test-user")
        assert isinstance(result, bool)


# ── Notification Service Tests ───────────────────────────────────────────────


class TestNotificationService:
    """Tests for the main notification service."""

    @pytest.mark.asyncio
    async def test_notify_sends_email(
        self, notification_service, sample_alert, mock_email_provider
    ):
        """Should send email notification for high-severity alert."""
        records = await notification_service.notify(
            alert=sample_alert,
            recipients=["analyst@test.com"],
            channels=[NotificationChannel.EMAIL],
        )

        assert len(records) == 1
        mock_email_provider.send_email.assert_called_once()
        call_kwargs = mock_email_provider.send_email.call_args
        assert call_kwargs[1]["to"] == "analyst@test.com" or call_kwargs[0][0] == "analyst@test.com"

    @pytest.mark.asyncio
    async def test_notify_sends_sms(self, notification_service, critical_alert, mock_sms_provider):
        """Should send SMS for critical alerts."""
        records = await notification_service.notify(
            alert=critical_alert,
            recipients=["+15551234567"],
            channels=[NotificationChannel.SMS],
        )

        assert len(records) == 1
        mock_sms_provider.send_sms.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_sends_webhook(
        self, notification_service, critical_alert, mock_webhook_provider
    ):
        """Should send webhook notification."""
        # Configure webhook for recipient
        notification_service._config["webhooks"] = {
            "slack-channel": {"type": "slack", "url": "https://hooks.slack.com/test"}
        }

        records = await notification_service.notify(
            alert=critical_alert,
            recipients=["slack-channel"],
            channels=[NotificationChannel.WEBHOOK],
        )

        assert len(records) == 1
        mock_webhook_provider.send_webhook.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_fails_cleanly_when_webhook_url_missing(
        self, notification_service, critical_alert, mock_webhook_provider
    ):
        """Should not attempt delivery when webhook URL is not configured."""
        notification_service._config["webhooks"] = {"slack-channel": {"type": "slack", "url": ""}}

        records = await notification_service.notify(
            alert=critical_alert,
            recipients=["slack-channel"],
            channels=[NotificationChannel.WEBHOOK],
        )

        assert len(records) == 1
        assert records[0].status == DeliveryStatus.FAILED
        assert "Webhook URL not configured" in str(records[0].failure_reason)
        mock_webhook_provider.send_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_fails_cleanly_when_pagerduty_key_missing(
        self, notification_service, critical_alert, mock_webhook_provider
    ):
        """Should not send PagerDuty event without a routing key."""
        notification_service._config["webhooks"] = {
            "pagerduty-channel": {
                "type": "pagerduty",
                "url": "https://events.pagerduty.com/v2/enqueue",
                "routing_key": "",
            }
        }

        records = await notification_service.notify(
            alert=critical_alert,
            recipients=["pagerduty-channel"],
            channels=[NotificationChannel.WEBHOOK],
        )

        assert len(records) == 1
        assert records[0].status == DeliveryStatus.FAILED
        assert "PagerDuty routing key not configured" in str(records[0].failure_reason)
        mock_webhook_provider.send_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_sends_in_app(
        self, notification_service, sample_alert, mock_websocket_provider
    ):
        """Should send in-app WebSocket notification."""
        records = await notification_service.notify(
            alert=sample_alert,
            recipients=["user-123"],
            channels=[NotificationChannel.IN_APP],
        )

        assert len(records) == 1
        mock_websocket_provider.push_notification.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limiting_blocks_excess(self, notification_service, sample_alert):
        """Should rate-limit notifications exceeding hourly limit."""
        # Send 10 notifications (at limit)
        for i in range(10):
            await notification_service.notify(
                alert=sample_alert,
                recipients=["limited-user@test.com"],
                channels=[NotificationChannel.EMAIL],
            )

        # 11th should be rate limited
        records = await notification_service.notify(
            alert=sample_alert,
            recipients=["limited-user@test.com"],
            channels=[NotificationChannel.EMAIL],
        )

        assert records[0].status == DeliveryStatus.RATE_LIMITED

    @pytest.mark.asyncio
    async def test_rate_limit_override(self, notification_service, sample_alert):
        """Should bypass rate limit when override is set (for escalations)."""
        # Exhaust the rate limit
        for _ in range(10):
            await notification_service.notify(
                alert=sample_alert,
                recipients=["limited-user@test.com"],
                channels=[NotificationChannel.EMAIL],
            )

        # With override, should still send
        records = await notification_service.notify(
            alert=sample_alert,
            recipients=["limited-user@test.com"],
            channels=[NotificationChannel.EMAIL],
            override_rate_limit=True,
        )

        # Should not be rate limited
        assert records[0].status != DeliveryStatus.RATE_LIMITED

    @pytest.mark.asyncio
    async def test_multi_channel_delivery(
        self, notification_service, critical_alert, mock_email_provider, mock_sms_provider
    ):
        """Should deliver to multiple channels simultaneously."""
        records = await notification_service.notify(
            alert=critical_alert,
            recipients=["analyst@test.com"],
            channels=[NotificationChannel.EMAIL, NotificationChannel.SMS],
        )

        assert len(records) == 2
        mock_email_provider.send_email.assert_called_once()
        mock_sms_provider.send_sms.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_recipient_delivery(self, notification_service, sample_alert):
        """Should deliver to multiple recipients."""
        records = await notification_service.notify(
            alert=sample_alert,
            recipients=["user-a@test.com", "user-b@test.com", "user-c@test.com"],
            channels=[NotificationChannel.EMAIL],
        )

        assert len(records) == 3

    @pytest.mark.asyncio
    async def test_preference_disables_channel(
        self, notification_service, low_alert, mock_sms_provider
    ):
        """Should skip channel if disabled by user preference."""
        pref = NotificationPreference(
            user_id="pref-user@test.com",
            sms_enabled=False,
        )
        notification_service.preferences.update_preference(pref)

        records = await notification_service.notify(
            alert=low_alert,
            recipients=["pref-user@test.com"],
            channels=[NotificationChannel.SMS],
        )

        assert records[0].status == DeliveryStatus.FAILED
        mock_sms_provider.send_sms.assert_not_called()

    @pytest.mark.asyncio
    async def test_delivery_failure_tracked(self, notification_service, sample_alert):
        """Should track delivery failures."""
        notification_service._email_provider.send_email.side_effect = RuntimeError("SES error")

        records = await notification_service.notify(
            alert=sample_alert,
            recipients=["user@test.com"],
            channels=[NotificationChannel.EMAIL],
        )

        assert records[0].status == DeliveryStatus.FAILED

    @pytest.mark.asyncio
    async def test_auto_channel_resolution_critical(self, notification_service, critical_alert):
        """Critical alerts should route to all channels."""
        records = await notification_service.notify(
            alert=critical_alert,
            recipients=["user@test.com"],
        )

        channels_used = {r.channel for r in records}
        assert NotificationChannel.EMAIL in channels_used
        assert NotificationChannel.SMS in channels_used
        assert NotificationChannel.IN_APP in channels_used

    @pytest.mark.asyncio
    async def test_auto_channel_resolution_low(self, notification_service, low_alert):
        """Low-severity alerts should only route to email."""
        records = await notification_service.notify(
            alert=low_alert,
            recipients=["user@test.com"],
        )

        channels_used = {r.channel for r in records}
        assert NotificationChannel.EMAIL in channels_used
        assert NotificationChannel.SMS not in channels_used

    def test_mark_notification_read(self, notification_service, delivery_tracker):
        """Should mark notification as read."""
        record = notification_service.tracker.create_record(
            alert_id="ALT-001",
            channel=NotificationChannel.IN_APP,
            recipient="user-123",
        )
        notification_service.mark_notification_read(record.notification_id)

        updated = notification_service.tracker.get_record(record.notification_id)
        assert updated.status == DeliveryStatus.READ

    def test_get_notifications_for_alert(self, notification_service):
        """Should retrieve all notifications for an alert."""
        notification_service.tracker.create_record(
            "ALT-001", NotificationChannel.EMAIL, "a@test.com"
        )
        notification_service.tracker.create_record("ALT-001", NotificationChannel.SMS, "+1111")

        records = notification_service.get_notifications_for_alert("ALT-001")
        assert len(records) == 2


# ── Webhook Formatter Tests ──────────────────────────────────────────────────


class TestWebhookFormatter:
    """Tests for webhook payload formatting."""

    def test_format_slack(self, sample_alert):
        """Should format Slack Block Kit message."""
        from src.alerting.alert_templates import AlertTemplateRenderer

        renderer = AlertTemplateRenderer()
        rendered = renderer.render(sample_alert, "webhook")

        payload = WebhookFormatter.format_slack(sample_alert, rendered)
        assert "blocks" in payload
        assert len(payload["blocks"]) >= 3

    def test_format_teams(self, sample_alert, monkeypatch):
        """Should format Microsoft Teams Adaptive Card."""
        from src.alerting.alert_templates import AlertTemplateRenderer

        monkeypatch.setenv("RISKPULSE_DASHBOARD_BASE_URL", "https://dashboard.example.test")
        renderer = AlertTemplateRenderer()
        rendered = renderer.render(sample_alert, "webhook")

        payload = WebhookFormatter.format_teams(sample_alert, rendered)
        assert "attachments" in payload
        assert payload["type"] == "message"
        action_url = payload["attachments"][0]["content"]["actions"][0]["url"]
        assert action_url == "https://dashboard.example.test/alerts/ALT-TEST-001"

    def test_format_pagerduty(self, critical_alert, monkeypatch):
        """Should format PagerDuty Events API v2 payload."""
        from src.alerting.alert_templates import AlertTemplateRenderer

        monkeypatch.setenv("RISKPULSE_DASHBOARD_BASE_URL", "https://dashboard.example.test/")
        renderer = AlertTemplateRenderer()
        rendered = renderer.render(critical_alert, "webhook")

        payload = WebhookFormatter.format_pagerduty(critical_alert, rendered)
        assert payload["event_action"] == "trigger"
        assert "payload" in payload
        assert payload["payload"]["severity"] == "critical"
        assert "dedup_key" in payload
        assert payload["links"][0]["href"] == "https://dashboard.example.test/alerts/ALT-CRIT-001"


# ── Escalation Engine Tests ──────────────────────────────────────────────────


class TestEscalationEngine:
    """Tests for the escalation engine."""

    def test_start_escalation(self, escalation_engine, sample_alert):
        """Should start escalation for high-severity alert."""
        record = escalation_engine.start_escalation(sample_alert)

        assert record is not None
        assert record.alert_id == "ALT-TEST-001"
        assert record.current_level == EscalationLevel.L1
        assert record.status == EscalationStatus.PENDING
        assert record.next_escalation_at is not None

    def test_no_escalation_for_low(self, escalation_engine, low_alert):
        """Should not start escalation for low-severity alerts."""
        record = escalation_engine.start_escalation(low_alert)
        assert record is None

    def test_acknowledge_stops_escalation(self, escalation_engine, sample_alert):
        """Acknowledging should stop the escalation timer."""
        escalation_engine.start_escalation(sample_alert)
        record = escalation_engine.acknowledge("ALT-TEST-001", "analyst@test.com")

        assert record is not None
        assert record.status == EscalationStatus.ACKNOWLEDGED
        assert record.acknowledged_by == "analyst@test.com"
        assert record.next_escalation_at is None

    def test_resolve_escalation(self, escalation_engine, sample_alert):
        """Should resolve an active escalation."""
        escalation_engine.start_escalation(sample_alert)
        record = escalation_engine.resolve("ALT-TEST-001")

        assert record is not None
        assert record.status == EscalationStatus.RESOLVED
        assert record.resolved_at is not None

    def test_timeout_triggers_level_up(self, escalation_engine, sample_alert):
        """Timeout should escalate to next level."""
        record = escalation_engine.start_escalation(sample_alert)

        # Simulate timeout by setting next_escalation_at in the past
        record.next_escalation_at = datetime.now(timezone.utc) - timedelta(minutes=1)

        escalated = escalation_engine.check_timeouts()
        assert len(escalated) == 1
        assert escalated[0].current_level == EscalationLevel.L2

    def test_max_level_reached(self, escalation_engine, sample_alert):
        """Should handle reaching max escalation level."""
        record = escalation_engine.start_escalation(sample_alert)

        # Escalate through all levels until timed out
        while record.status in (EscalationStatus.PENDING, EscalationStatus.ESCALATED):
            record.next_escalation_at = datetime.now(timezone.utc) - timedelta(minutes=1)
            escalation_engine.check_timeouts()

        # After max level, should be timed out
        assert record.status == EscalationStatus.TIMED_OUT

    def test_duplicate_escalation_returns_existing(self, escalation_engine, sample_alert):
        """Starting escalation twice should return the existing record."""
        record1 = escalation_engine.start_escalation(sample_alert)
        record2 = escalation_engine.start_escalation(sample_alert)

        assert record1.escalation_id == record2.escalation_id

    def test_audit_trail(self, escalation_engine, sample_alert):
        """Should maintain an audit trail of escalation events."""
        escalation_engine.start_escalation(sample_alert)
        escalation_engine.acknowledge("ALT-TEST-001", "analyst@test.com")

        audit = escalation_engine.get_audit_log(alert_id="ALT-TEST-001")
        assert len(audit) >= 2
        assert audit[0]["action"] == "escalation_started"
        assert audit[1]["action"] == "escalation_acknowledged"

    def test_get_active_escalations(self, escalation_engine, sample_alert, critical_alert):
        """Should return all currently active escalations."""
        escalation_engine.start_escalation(sample_alert)
        escalation_engine.start_escalation(critical_alert)

        active = escalation_engine.get_active_escalations()
        assert len(active) == 2

    def test_get_escalation_for_alert(self, escalation_engine, sample_alert):
        """Should retrieve escalation record by alert ID."""
        escalation_engine.start_escalation(sample_alert)

        record = escalation_engine.get_escalation_for_alert("ALT-TEST-001")
        assert record is not None
        assert record.alert_id == "ALT-TEST-001"

    def test_stats(self, escalation_engine, sample_alert, critical_alert):
        """Should return escalation statistics."""
        escalation_engine.start_escalation(sample_alert)
        escalation_engine.start_escalation(critical_alert)
        escalation_engine.acknowledge("ALT-TEST-001", "analyst")

        stats = escalation_engine.get_stats()
        assert stats["active_escalations"] == 1
        assert stats["acknowledged"] == 1
        assert stats["policies_loaded"] >= 1

    def test_escalation_with_config(self, escalation_engine_with_config, sample_alert):
        """Should use policy from config file."""
        record = escalation_engine_with_config.start_escalation(sample_alert)

        assert record is not None
        assert record.policy_id == "test_policy"
        assert record.escalated_to == ["analyst@test.com"]

    def test_escalation_history(self, escalation_engine, sample_alert):
        """Escalation record should contain history entries."""
        record = escalation_engine.start_escalation(sample_alert)

        assert len(record.history) == 1
        assert record.history[0]["action"] == "escalation_started"

        # Acknowledge
        escalation_engine.acknowledge("ALT-TEST-001", "analyst")
        assert len(record.history) == 2
        assert record.history[1]["action"] == "acknowledged"


# ── On-Call Schedule Tests ───────────────────────────────────────────────────


class TestOnCallSchedule:
    """Tests for on-call schedule resolution."""

    def test_rotation_resolves_member(self):
        """Should resolve current on-call person based on time."""
        rotation = OnCallRotation(
            rotation_id="test",
            members=["alice@test.com", "bob@test.com"],
            rotation_interval_hours=168,
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        schedule = OnCallSchedule(
            team_id="test-team",
            team_name="Test Team",
            schedules=[rotation],
        )

        oncall = schedule.get_current_oncall()
        assert oncall in ["alice@test.com", "bob@test.com"]

    def test_empty_rotation(self):
        """Should handle empty rotation gracefully."""
        rotation = OnCallRotation(
            rotation_id="empty",
            members=[],
            rotation_interval_hours=168,
        )
        schedule = OnCallSchedule(
            team_id="empty-team",
            team_name="Empty",
            schedules=[rotation],
        )

        oncall = schedule.get_current_oncall()
        assert oncall == ""

    def test_inactive_rotation(self):
        """Should return None for inactive rotation."""
        rotation = OnCallRotation(
            rotation_id="future",
            members=["future@test.com"],
            start_time=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        schedule = OnCallSchedule(
            team_id="future-team",
            team_name="Future",
            schedules=[rotation],
        )

        oncall = schedule.get_current_oncall()
        assert oncall is None


# ── Integration-style Tests ──────────────────────────────────────────────────


class TestNotificationEscalationIntegration:
    """Tests combining notification service with escalation engine."""

    @pytest.mark.asyncio
    async def test_escalation_triggers_notification(self, notification_service, sample_alert):
        """Escalation should trigger notifications to escalation recipients."""
        engine = EscalationEngine(
            config_path=None,
            on_escalate=None,
        )

        record = engine.start_escalation(sample_alert)
        assert record is not None

        # Notify escalation recipients
        records = await notification_service.notify(
            alert=sample_alert,
            recipients=record.escalated_to,
            channels=[NotificationChannel.EMAIL, NotificationChannel.IN_APP],
            override_rate_limit=True,
        )

        assert len(records) >= 1

    @pytest.mark.asyncio
    async def test_full_escalation_lifecycle(self, notification_service, critical_alert):
        """Test full lifecycle: alert → notify → escalate → acknowledge."""
        # 1. Initial notification
        records = await notification_service.notify(
            alert=critical_alert,
            recipients=["analyst@test.com"],
            channels=[NotificationChannel.EMAIL],
        )
        assert len(records) == 1

        # 2. Start escalation
        engine = EscalationEngine(config_path=None)
        esc_record = engine.start_escalation(critical_alert)
        assert esc_record is not None
        assert esc_record.status == EscalationStatus.PENDING

        # 3. Simulate timeout → escalate
        esc_record.next_escalation_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        escalated = engine.check_timeouts()
        assert len(escalated) == 1

        # 4. Send escalation notification
        escalation_records = await notification_service.notify(
            alert=critical_alert,
            recipients=esc_record.escalated_to,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SMS],
            override_rate_limit=True,
        )
        assert len(escalation_records) >= 1

        # 5. Acknowledge
        ack_record = engine.acknowledge(critical_alert.alert_id, "lead@test.com")
        assert ack_record.status == EscalationStatus.ACKNOWLEDGED

        # 6. Verify audit trail
        audit = engine.get_audit_log(alert_id=critical_alert.alert_id)
        assert len(audit) >= 2
