"""Alert Template Rendering — Multi-channel alert formatting.

Provides template-based rendering for alert notifications across
different channels (email, SMS, dashboard, webhook). Supports
dynamic content based on alert type and severity, with localization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.alerting.alert_manager import Alert, AlertChannel, AlertSeverity, AlertType
from src.utils.logger import get_logger

logger = get_logger(__name__, component="alert_templates")


# ── Locale Definitions ───────────────────────────────────────────────────────


class Locale(str, Enum):
    EN_US = "en_US"
    EN_GB = "en_GB"
    ES_ES = "es_ES"
    FR_FR = "fr_FR"
    DE_DE = "de_DE"
    PT_BR = "pt_BR"


_LOCALE_STRINGS: dict[str, dict[str, str]] = {
    "en_US": {
        "alert_title": "Fraud Alert",
        "severity_label": "Severity",
        "risk_score_label": "Risk Score",
        "account_label": "Account",
        "transaction_label": "Transaction",
        "amount_label": "Amount",
        "merchant_label": "Merchant",
        "channel_label": "Channel",
        "timestamp_label": "Timestamp",
        "description_label": "Description",
        "action_required": "Action Required",
        "investigate_prompt": "Please investigate this alert immediately.",
        "low_severity_msg": "This is an informational alert for review.",
        "medium_severity_msg": "This alert requires attention during business hours.",
        "high_severity_msg": "This alert requires prompt investigation.",
        "critical_severity_msg": "URGENT: Immediate action required.",
        "rule_based_title": "Rule-Based Fraud Alert",
        "anomaly_title": "Anomaly Detection Alert",
        "ml_score_title": "ML Model Fraud Alert",
        "ensemble_title": "Ensemble Fraud Alert",
    },
    "es_ES": {
        "alert_title": "Alerta de Fraude",
        "severity_label": "Severidad",
        "risk_score_label": "Puntuación de Riesgo",
        "account_label": "Cuenta",
        "transaction_label": "Transacción",
        "amount_label": "Monto",
        "merchant_label": "Comercio",
        "channel_label": "Canal",
        "timestamp_label": "Marca de Tiempo",
        "description_label": "Descripción",
        "action_required": "Acción Requerida",
        "investigate_prompt": "Por favor investigue esta alerta inmediatamente.",
        "low_severity_msg": "Esta es una alerta informativa para revisión.",
        "medium_severity_msg": "Esta alerta requiere atención en horario laboral.",
        "high_severity_msg": "Esta alerta requiere investigación inmediata.",
        "critical_severity_msg": "URGENTE: Se requiere acción inmediata.",
        "rule_based_title": "Alerta de Fraude Basada en Reglas",
        "anomaly_title": "Alerta de Detección de Anomalías",
        "ml_score_title": "Alerta de Modelo ML",
        "ensemble_title": "Alerta de Ensemble",
    },
    "fr_FR": {
        "alert_title": "Alerte de Fraude",
        "severity_label": "Sévérité",
        "risk_score_label": "Score de Risque",
        "account_label": "Compte",
        "transaction_label": "Transaction",
        "amount_label": "Montant",
        "merchant_label": "Commerçant",
        "channel_label": "Canal",
        "timestamp_label": "Horodatage",
        "description_label": "Description",
        "action_required": "Action Requise",
        "investigate_prompt": "Veuillez enquêter sur cette alerte immédiatement.",
        "low_severity_msg": "Ceci est une alerte informative à examiner.",
        "medium_severity_msg": "Cette alerte nécessite une attention pendant les heures ouvrables.",
        "high_severity_msg": "Cette alerte nécessite une enquête rapide.",
        "critical_severity_msg": "URGENT: Action immédiate requise.",
        "rule_based_title": "Alerte Fraude Basée sur les Règles",
        "anomaly_title": "Alerte Détection d'Anomalies",
        "ml_score_title": "Alerte Modèle ML",
        "ensemble_title": "Alerte Ensemble",
    },
    "de_DE": {
        "alert_title": "Betrugswarnung",
        "severity_label": "Schweregrad",
        "risk_score_label": "Risikobewertung",
        "account_label": "Konto",
        "transaction_label": "Transaktion",
        "amount_label": "Betrag",
        "merchant_label": "Händler",
        "channel_label": "Kanal",
        "timestamp_label": "Zeitstempel",
        "description_label": "Beschreibung",
        "action_required": "Handlung Erforderlich",
        "investigate_prompt": "Bitte untersuchen Sie diese Warnung sofort.",
        "low_severity_msg": "Dies ist eine informative Warnung zur Überprüfung.",
        "medium_severity_msg": "Diese Warnung erfordert Aufmerksamkeit während der Geschäftszeiten.",
        "high_severity_msg": "Diese Warnung erfordert eine sofortige Untersuchung.",
        "critical_severity_msg": "DRINGEND: Sofortige Maßnahmen erforderlich.",
        "rule_based_title": "Regelbasierte Betrugswarnung",
        "anomaly_title": "Anomalieerkennung Warnung",
        "ml_score_title": "ML-Modell Betrugswarnung",
        "ensemble_title": "Ensemble Betrugswarnung",
    },
    "pt_BR": {
        "alert_title": "Alerta de Fraude",
        "severity_label": "Severidade",
        "risk_score_label": "Score de Risco",
        "account_label": "Conta",
        "transaction_label": "Transação",
        "amount_label": "Valor",
        "merchant_label": "Comerciante",
        "channel_label": "Canal",
        "timestamp_label": "Data/Hora",
        "description_label": "Descrição",
        "action_required": "Ação Necessária",
        "investigate_prompt": "Por favor, investigue este alerta imediatamente.",
        "low_severity_msg": "Este é um alerta informativo para revisão.",
        "medium_severity_msg": "Este alerta requer atenção durante o horário comercial.",
        "high_severity_msg": "Este alerta requer investigação imediata.",
        "critical_severity_msg": "URGENTE: Ação imediata necessária.",
        "rule_based_title": "Alerta de Fraude Baseado em Regras",
        "anomaly_title": "Alerta de Detecção de Anomalias",
        "ml_score_title": "Alerta de Modelo ML",
        "ensemble_title": "Alerta de Ensemble",
    },
}

# en_GB falls back to en_US
_LOCALE_STRINGS["en_GB"] = _LOCALE_STRINGS["en_US"]


def _get_string(locale: str, key: str) -> str:
    """Get a localized string, falling back to en_US."""
    strings = _LOCALE_STRINGS.get(locale, _LOCALE_STRINGS["en_US"])
    return strings.get(key, _LOCALE_STRINGS["en_US"].get(key, key))


# ── Rendered Output ──────────────────────────────────────────────────────────


@dataclass
class RenderedAlert:
    """A rendered alert ready for delivery to a specific channel."""

    alert_id: str
    channel: str
    subject: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)
    html_body: str | None = None
    priority: str = "normal"
    rendered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "channel": self.channel,
            "subject": self.subject,
            "body": self.body,
            "html_body": self.html_body,
            "metadata": self.metadata,
            "priority": self.priority,
            "rendered_at": self.rendered_at.isoformat(),
        }


# ── Template Renderer ────────────────────────────────────────────────────────


class AlertTemplateRenderer:
    """Renders alerts into channel-specific formatted messages.

    Supports email (HTML + plain text), SMS (short text), dashboard
    (structured JSON), and webhook (full payload) formats.

    Usage::

        renderer = AlertTemplateRenderer()
        rendered = renderer.render(alert, channel="email")
        rendered_all = renderer.render_all_channels(alert)
    """

    def __init__(self, locale: str = "en_US") -> None:
        self._locale = locale
        if locale not in _LOCALE_STRINGS:
            logger.warning("unsupported_locale_fallback", locale=locale)
            self._locale = "en_US"

    @property
    def locale(self) -> str:
        return self._locale

    @locale.setter
    def locale(self, value: str) -> None:
        if value in _LOCALE_STRINGS:
            self._locale = value
        else:
            logger.warning("unsupported_locale", locale=value)

    def render(self, alert: Alert, channel: str) -> RenderedAlert:
        """Render an alert for a specific channel.

        Args:
            alert: The alert to render.
            channel: Target channel (email, sms, dashboard, webhook).

        Returns:
            RenderedAlert with formatted content.
        """
        channel_lower = channel.lower()
        render_method = {
            "email": self._render_email,
            "sms": self._render_sms,
            "dashboard": self._render_dashboard,
            "webhook": self._render_webhook,
        }.get(channel_lower, self._render_dashboard)

        return render_method(alert)

    def render_all_channels(self, alert: Alert) -> list[RenderedAlert]:
        """Render an alert for all its configured channels.

        Args:
            alert: The alert to render.

        Returns:
            List of RenderedAlert instances, one per channel.
        """
        rendered = []
        for channel in alert.channels:
            rendered.append(self.render(alert, channel))
        return rendered

    # ── Channel-Specific Renderers ───────────────────────────────────────

    def _render_email(self, alert: Alert) -> RenderedAlert:
        """Render alert as email (subject + HTML body + plain text body)."""
        s = lambda key: _get_string(self._locale, key)  # noqa: E731
        subject = self._get_email_subject(alert)
        severity_msg = self._get_severity_message(alert.severity)

        # Plain text body
        body_lines = [
            f"{s('alert_title')} - {alert.alert_id}",
            "=" * 60,
            "",
            f"{s('severity_label')}: {alert.severity.value.upper()}",
            f"{s('risk_score_label')}: {alert.risk_score:.4f}",
            f"{s('account_label')}: {alert.account_id}",
            f"{s('transaction_label')}: {alert.transaction_id}",
            "",
        ]

        # Add transaction details
        txn_details = alert.details.get("transaction", {})
        if txn_details:
            body_lines.extend([
                f"{s('amount_label')}: {txn_details.get('amount', 'N/A')} "
                f"{txn_details.get('currency', 'USD')}",
                f"{s('merchant_label')}: {txn_details.get('merchant', 'N/A')}",
                f"{s('channel_label')}: {txn_details.get('channel', 'N/A')}",
                f"{s('timestamp_label')}: {txn_details.get('timestamp', 'N/A')}",
                "",
            ])

        body_lines.extend([
            f"{s('description_label')}: {alert.description}",
            "",
            f"--- {s('action_required')} ---",
            severity_msg,
        ])

        body = "\n".join(body_lines)

        # HTML body
        html_body = self._render_email_html(alert, s)

        priority = self._get_priority(alert.severity)

        return RenderedAlert(
            alert_id=alert.alert_id,
            channel="email",
            subject=subject,
            body=body,
            html_body=html_body,
            priority=priority,
            metadata={
                "severity": alert.severity.value,
                "account_id": alert.account_id,
            },
        )

    def _render_sms(self, alert: Alert) -> RenderedAlert:
        """Render alert as SMS (short, 160 char target)."""
        s = lambda key: _get_string(self._locale, key)  # noqa: E731
        severity = alert.severity.value.upper()
        score = f"{alert.risk_score:.2f}"

        # Keep SMS concise
        txn_details = alert.details.get("transaction", {})
        amount = txn_details.get("amount", "N/A")
        currency = txn_details.get("currency", "USD")

        body = (
            f"[{severity}] {s('alert_title')} | "
            f"Acct: {alert.account_id[-8:]} | "
            f"Score: {score} | "
            f"Amt: {amount} {currency} | "
            f"ID: {alert.alert_id[:8]}"
        )

        # Truncate to SMS limit if needed
        if len(body) > 160:
            body = body[:157] + "..."

        return RenderedAlert(
            alert_id=alert.alert_id,
            channel="sms",
            subject=f"{severity} Alert",
            body=body,
            priority=self._get_priority(alert.severity),
            metadata={
                "severity": alert.severity.value,
                "truncated": len(body) > 160,
            },
        )

    def _render_dashboard(self, alert: Alert) -> RenderedAlert:
        """Render alert as structured dashboard notification."""
        s = lambda key: _get_string(self._locale, key)  # noqa: E731

        body_data = {
            "alert_id": alert.alert_id,
            "title": self._get_type_title(alert.alert_type),
            "severity": alert.severity.value,
            "status": alert.status.value,
            "risk_score": round(alert.risk_score, 4),
            "account_id": alert.account_id,
            "transaction_id": alert.transaction_id,
            "description": alert.description,
            "details": alert.details,
            "enrichment": alert.enrichment,
            "created_at": alert.created_at.isoformat(),
            "channels": alert.channels,
            "action_message": self._get_severity_message(alert.severity),
        }

        import json
        body = json.dumps(body_data, indent=2, default=str)

        return RenderedAlert(
            alert_id=alert.alert_id,
            channel="dashboard",
            subject=self._get_type_title(alert.alert_type),
            body=body,
            priority=self._get_priority(alert.severity),
            metadata={
                "severity": alert.severity.value,
                "widget_type": "alert_card",
                "dismissible": alert.severity.value in ("low", "medium"),
            },
        )

    def _render_webhook(self, alert: Alert) -> RenderedAlert:
        """Render alert as webhook payload (full structured data)."""
        import json

        payload = {
            "event_type": "fraud.alert.created",
            "event_version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert": alert.to_dict(),
            "routing": {
                "severity": alert.severity.value,
                "channels": alert.channels,
                "priority": self._get_priority(alert.severity),
            },
        }

        body = json.dumps(payload, indent=2, default=str)

        return RenderedAlert(
            alert_id=alert.alert_id,
            channel="webhook",
            subject=f"fraud.alert.{alert.severity.value}",
            body=body,
            priority=self._get_priority(alert.severity),
            metadata={
                "content_type": "application/json",
                "event_type": "fraud.alert.created",
                "severity": alert.severity.value,
            },
        )

    # ── Helper Methods ───────────────────────────────────────────────────

    def _get_email_subject(self, alert: Alert) -> str:
        """Generate email subject line based on alert severity and type."""
        s = lambda key: _get_string(self._locale, key)  # noqa: E731
        severity_prefix = {
            AlertSeverity.LOW: "[LOW]",
            AlertSeverity.MEDIUM: "[MEDIUM]",
            AlertSeverity.HIGH: "[HIGH]",
            AlertSeverity.CRITICAL: "[CRITICAL]",
        }
        prefix = severity_prefix.get(alert.severity, "[ALERT]")
        title = self._get_type_title(alert.alert_type)
        return f"{prefix} {title} - Account {alert.account_id}"

    def _get_type_title(self, alert_type: AlertType) -> str:
        """Get localized title for alert type."""
        type_key_map = {
            AlertType.RULE_BASED: "rule_based_title",
            AlertType.ANOMALY: "anomaly_title",
            AlertType.ML_SCORE: "ml_score_title",
            AlertType.ENSEMBLE: "ensemble_title",
        }
        key = type_key_map.get(alert_type, "alert_title")
        return _get_string(self._locale, key)

    def _get_severity_message(self, severity: AlertSeverity) -> str:
        """Get localized severity-specific action message."""
        msg_key_map = {
            AlertSeverity.LOW: "low_severity_msg",
            AlertSeverity.MEDIUM: "medium_severity_msg",
            AlertSeverity.HIGH: "high_severity_msg",
            AlertSeverity.CRITICAL: "critical_severity_msg",
        }
        key = msg_key_map.get(severity, "low_severity_msg")
        return _get_string(self._locale, key)

    @staticmethod
    def _get_priority(severity: AlertSeverity) -> str:
        """Map severity to delivery priority."""
        return {
            AlertSeverity.LOW: "low",
            AlertSeverity.MEDIUM: "normal",
            AlertSeverity.HIGH: "high",
            AlertSeverity.CRITICAL: "urgent",
        }.get(severity, "normal")

    def _render_email_html(self, alert: Alert, s: Any) -> str:
        """Render HTML email body for an alert."""
        severity_colors = {
            AlertSeverity.LOW: "#4CAF50",
            AlertSeverity.MEDIUM: "#FF9800",
            AlertSeverity.HIGH: "#F44336",
            AlertSeverity.CRITICAL: "#9C27B0",
        }
        color = severity_colors.get(alert.severity, "#757575")

        txn_details = alert.details.get("transaction", {})
        amount = txn_details.get("amount", "N/A")
        currency = txn_details.get("currency", "USD")
        merchant = txn_details.get("merchant", "N/A")
        channel = txn_details.get("channel", "N/A")
        timestamp = txn_details.get("timestamp", "N/A")

        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
  <div style="background-color: {color}; color: white; padding: 16px; border-radius: 4px 4px 0 0;">
    <h2 style="margin: 0;">{s('alert_title')} - {alert.severity.value.upper()}</h2>
    <p style="margin: 4px 0 0 0; opacity: 0.9;">{self._get_type_title(alert.alert_type)}</p>
  </div>
  <div style="border: 1px solid #ddd; border-top: none; padding: 16px; border-radius: 0 0 4px 4px;">
    <table style="width: 100%; border-collapse: collapse;">
      <tr>
        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold;">{s('risk_score_label')}</td>
        <td style="padding: 8px; border-bottom: 1px solid #eee;">{alert.risk_score:.4f}</td>
      </tr>
      <tr>
        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold;">{s('account_label')}</td>
        <td style="padding: 8px; border-bottom: 1px solid #eee;">{alert.account_id}</td>
      </tr>
      <tr>
        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold;">{s('amount_label')}</td>
        <td style="padding: 8px; border-bottom: 1px solid #eee;">{amount} {currency}</td>
      </tr>
      <tr>
        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold;">{s('merchant_label')}</td>
        <td style="padding: 8px; border-bottom: 1px solid #eee;">{merchant}</td>
      </tr>
      <tr>
        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold;">{s('channel_label')}</td>
        <td style="padding: 8px; border-bottom: 1px solid #eee;">{channel}</td>
      </tr>
      <tr>
        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold;">{s('timestamp_label')}</td>
        <td style="padding: 8px; border-bottom: 1px solid #eee;">{timestamp}</td>
      </tr>
    </table>
    <div style="margin-top: 16px; padding: 12px; background-color: #f5f5f5; border-radius: 4px;">
      <strong>{s('description_label')}:</strong><br>
      {alert.description}
    </div>
    <div style="margin-top: 16px; padding: 12px; background-color: #fff3e0; border-left: 4px solid {color}; border-radius: 4px;">
      <strong>{s('action_required')}:</strong><br>
      {self._get_severity_message(alert.severity)}
    </div>
    <p style="margin-top: 16px; font-size: 12px; color: #999;">
      Alert ID: {alert.alert_id} | Generated: {alert.created_at.isoformat()}
    </p>
  </div>
</body>
</html>"""
        return html
