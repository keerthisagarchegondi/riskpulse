"""Alerting module - Alert management, notifications, and escalation."""

from src.alerting.alert_manager import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertStatistics,
    AlertStatus,
    AlertType,
    DeduplicationEngine,
    SuppressionEngine,
    ThrottleEngine,
)
from src.alerting.alert_templates import AlertTemplateRenderer, RenderedAlert
from src.alerting.escalation_engine import (
    EscalationEngine,
    EscalationLevel,
    EscalationPolicy,
    EscalationRecord,
    EscalationStatus,
)
from src.alerting.notification_service import (
    DeliveryStatus,
    DeliveryTracker,
    LocalEmailProvider,
    LocalSMSProvider,
    NotificationChannel,
    NotificationRateLimiter,
    NotificationRecord,
    NotificationService,
    PreferencesManager,
)

__all__ = [
    "Alert",
    "AlertManager",
    "AlertSeverity",
    "AlertStatus",
    "AlertStatistics",
    "AlertTemplateRenderer",
    "AlertType",
    "DeduplicationEngine",
    "DeliveryStatus",
    "DeliveryTracker",
    "EscalationEngine",
    "EscalationLevel",
    "EscalationPolicy",
    "EscalationRecord",
    "EscalationStatus",
    "LocalEmailProvider",
    "LocalSMSProvider",
    "NotificationChannel",
    "NotificationRateLimiter",
    "NotificationRecord",
    "NotificationService",
    "PreferencesManager",
    "RenderedAlert",
    "SuppressionEngine",
    "ThrottleEngine",
]
