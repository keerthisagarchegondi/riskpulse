"""CloudWatch Logs integration for RiskPulse services.

The module keeps CloudWatch shipping optional and injectable so local tests,
developer machines, and CI can use the same logging code without AWS
credentials. Production services attach :class:`CloudWatchLogHandler` during
startup and continue writing JSON logs through the existing structlog pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

import structlog

from src.utils.config import get_settings
from src.utils.logger import bind_correlation_id

logger = structlog.get_logger(__name__)

DEFAULT_SERVICES = ("api", "worker", "fraud-engine", "airflow")
DEFAULT_LOG_LEVELS = {
    "dev": "DEBUG",
    "staging": "INFO",
    "prod": "INFO",
}

SENSITIVE_FIELD_NAMES = frozenset(
    {
        "authorization",
        "api_key",
        "card_last_four",
        "card_number",
        "cvv",
        "device_id",
        "email",
        "ip_address",
        "password",
        "phone",
        "secret",
        "ssn",
        "token",
    }
)

EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
PHONE_RE = re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b")

_correlation_id: ContextVar[str | None] = ContextVar(
    "riskpulse_monitoring_correlation_id", default=None
)


class CloudWatchLoggerError(Exception):
    """Raised when CloudWatch log shipping cannot be configured."""


def get_log_group_name(service: str, environment: str | None = None) -> str:
    """Return the CloudWatch Logs group name for a service."""
    env = environment or get_settings().environment
    service_name = service.strip().lower()
    if service_name not in DEFAULT_SERVICES:
        service_name = service_name.replace("_", "-")
    return f"/riskpulse/{env}/{service_name}"


def get_log_stream_name(instance_id: str | None = None) -> str:
    """Return a stable stream name for the current process."""
    process_id = os.getpid()
    host = os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "local"
    suffix = instance_id or os.environ.get("RISKPULSE_INSTANCE_ID") or str(process_id)
    return f"{host}/{suffix}"


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Bind a correlation ID for standard logging and structlog processors."""
    value = correlation_id or str(uuid.uuid4())
    _correlation_id.set(value)
    bind_correlation_id(value)
    return value


def get_correlation_id() -> str | None:
    """Return the current correlation ID, if one has been bound."""
    return _correlation_id.get()


def scrub_pii(value: Any) -> Any:
    """Recursively redact common PII fields and inline sensitive values."""
    if isinstance(value, Mapping):
        scrubbed: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in SENSITIVE_FIELD_NAMES:
                scrubbed[key_text] = "***REDACTED***"
            else:
                scrubbed[key_text] = scrub_pii(item)
        return scrubbed

    if isinstance(value, list):
        return [scrub_pii(item) for item in value]

    if isinstance(value, tuple):
        return tuple(scrub_pii(item) for item in value)

    if isinstance(value, str):
        redacted = EMAIL_RE.sub("***REDACTED_EMAIL***", value)
        redacted = SSN_RE.sub("***REDACTED_SSN***", redacted)
        redacted = CARD_RE.sub("***REDACTED_CARD***", redacted)
        redacted = PHONE_RE.sub("***REDACTED_PHONE***", redacted)
        return redacted

    return value


class CloudWatchJSONFormatter(logging.Formatter):
    """Format stdlib records as compact JSON with RiskPulse context."""

    def __init__(self, *, service: str, environment: str) -> None:
        super().__init__()
        self.service = service
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        event: dict[str, Any]
        try:
            parsed = json.loads(message)
            event = parsed if isinstance(parsed, dict) else {"event": parsed}
        except (TypeError, json.JSONDecodeError):
            event = {"event": message}

        event.setdefault("service", self.service)
        event.setdefault("environment", self.environment)
        event.setdefault("logger", record.name)
        event.setdefault("level", record.levelname.lower())
        event.setdefault(
            "timestamp", datetime.fromtimestamp(record.created, timezone.utc).isoformat()
        )

        correlation_id = get_correlation_id()
        if correlation_id:
            event.setdefault("correlation_id", correlation_id)

        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)

        return json.dumps(scrub_pii(event), separators=(",", ":"), default=str)


class CloudWatchLogHandler(logging.Handler):
    """Batching CloudWatch Logs handler with sequence-token management."""

    def __init__(
        self,
        *,
        service: str,
        environment: str | None = None,
        client: Any | None = None,
        log_group_name: str | None = None,
        log_stream_name: str | None = None,
        retention_days: int = 30,
        max_batch_size: int = 100,
        max_batch_bytes: int = 900_000,
        flush_interval_seconds: float = 5.0,
        create_resources: bool = True,
    ) -> None:
        settings = get_settings()
        env = environment or settings.environment
        configured_level = settings.get(
            f"monitoring.log_levels.{env}", DEFAULT_LOG_LEVELS.get(env, "INFO")
        )

        super().__init__(getattr(logging, str(configured_level).upper(), logging.INFO))
        self.service = service
        self.environment = env
        self.log_group_name = log_group_name or get_log_group_name(service, env)
        self.log_stream_name = log_stream_name or get_log_stream_name()
        self.retention_days = retention_days
        self.max_batch_size = max_batch_size
        self.max_batch_bytes = max_batch_bytes
        self.flush_interval_seconds = flush_interval_seconds
        self.create_resources = create_resources
        self._client = client or self._create_client()
        self._sequence_token: str | None = None
        self._buffer: list[dict[str, Any]] = []
        self._buffer_bytes = 0
        self._last_flush = time.monotonic()
        self._lock = threading.RLock()

        self.setFormatter(CloudWatchJSONFormatter(service=service, environment=env))
        if self.create_resources:
            self._ensure_resources()

    @staticmethod
    def _create_client() -> Any:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
            raise CloudWatchLoggerError("boto3 is required for CloudWatch log shipping") from exc
        return boto3.client("logs")

    def emit(self, record: logging.LogRecord) -> None:
        """Add a log record to the buffer and flush when thresholds are met."""
        try:
            message = self.format(record)
            event = {
                "timestamp": int(record.created * 1000),
                "message": message,
            }
            event_size = len(message.encode("utf-8")) + 26

            with self._lock:
                self._buffer.append(event)
                self._buffer_bytes += event_size
                should_flush = (
                    len(self._buffer) >= self.max_batch_size
                    or self._buffer_bytes >= self.max_batch_bytes
                    or (time.monotonic() - self._last_flush) >= self.flush_interval_seconds
                )

            if should_flush:
                self.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        """Ship all buffered events to CloudWatch Logs."""
        with self._lock:
            if not self._buffer:
                return
            events = sorted(self._buffer, key=lambda item: item["timestamp"])
            self._buffer = []
            self._buffer_bytes = 0
            self._last_flush = time.monotonic()

        request = {
            "logGroupName": self.log_group_name,
            "logStreamName": self.log_stream_name,
            "logEvents": events,
        }
        if self._sequence_token:
            request["sequenceToken"] = self._sequence_token

        try:
            response = self._client.put_log_events(**request)
            self._sequence_token = response.get("nextSequenceToken", self._sequence_token)
        except Exception as exc:  # pragma: no cover - AWS SDK exceptions differ by version
            self._sequence_token = self._extract_sequence_token(exc) or self._sequence_token
            if self._sequence_token:
                request["sequenceToken"] = self._sequence_token
                response = self._client.put_log_events(**request)
                self._sequence_token = response.get("nextSequenceToken", self._sequence_token)
            else:
                raise

    def close(self) -> None:
        try:
            self.flush()
        finally:
            super().close()

    def _ensure_resources(self) -> None:
        """Create the log group and stream if they do not already exist."""
        self._call_ignoring_exists(
            "create_log_group",
            logGroupName=self.log_group_name,
        )
        self._call_ignoring_exists(
            "put_retention_policy",
            logGroupName=self.log_group_name,
            retentionInDays=self.retention_days,
        )
        self._call_ignoring_exists(
            "create_log_stream",
            logGroupName=self.log_group_name,
            logStreamName=self.log_stream_name,
        )
        self._sequence_token = self._describe_sequence_token()

    def _call_ignoring_exists(self, method_name: str, **kwargs: Any) -> None:
        method = getattr(self._client, method_name)
        try:
            method(**kwargs)
        except Exception as exc:  # pragma: no cover - AWS SDK exceptions differ by version
            if (
                "AlreadyExists" not in exc.__class__.__name__
                and "already exists" not in str(exc).lower()
            ):
                raise

    def _describe_sequence_token(self) -> str | None:
        try:
            response = self._client.describe_log_streams(
                logGroupName=self.log_group_name,
                logStreamNamePrefix=self.log_stream_name,
                limit=1,
            )
        except Exception:  # pragma: no cover - defensive AWS fallback
            return None

        streams = response.get("logStreams", [])
        for stream in streams:
            if stream.get("logStreamName") == self.log_stream_name:
                return stream.get("uploadSequenceToken")
        return None

    @staticmethod
    def _extract_sequence_token(exc: Exception) -> str | None:
        text = str(exc)
        for marker in ("sequenceToken is: ", "sequence token is: "):
            if marker in text:
                return text.split(marker, 1)[1].split()[0].strip()
        return None


def configure_cloudwatch_logging(
    *,
    service: str,
    environment: str | None = None,
    client: Any | None = None,
    enabled: bool | None = None,
    retention_days: int = 30,
) -> CloudWatchLogHandler | None:
    """Attach CloudWatch Logs shipping to the root logger when enabled."""
    settings = get_settings()
    env = environment or settings.environment
    should_enable = (
        enabled
        if enabled is not None
        else bool(settings.get("monitoring.cloudwatch.enabled", env in {"staging", "prod"}))
    )
    if not should_enable:
        return None

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, CloudWatchLogHandler) and handler.service == service:
            return handler

    handler = CloudWatchLogHandler(
        service=service,
        environment=env,
        client=client,
        retention_days=retention_days,
    )
    root_logger.addHandler(handler)
    root_logger.setLevel(min(root_logger.level or logging.INFO, handler.level))
    logger.info(
        "cloudwatch_logging_configured",
        service=service,
        environment=env,
        log_group=handler.log_group_name,
        log_stream=handler.log_stream_name,
    )
    return handler
