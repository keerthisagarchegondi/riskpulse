"""Structured logging with JSON output for RiskPulse platform.

Provides a configured structlog logger that outputs JSON-formatted logs
suitable for CloudWatch ingestion. Includes correlation ID tracking,
PII scrubbing, and performance timing.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from src.utils.config import get_settings


# Fields that should never appear in logs (PII protection)
_SENSITIVE_FIELDS = frozenset({
    "card_number",
    "card_last_four",
    "ssn",
    "password",
    "secret",
    "token",
    "api_key",
    "authorization",
    "ip_address",
    "email",
    "phone",
    "device_id",
})


def _scrub_pii(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Remove or mask sensitive fields from log output."""
    for key in list(event_dict.keys()):
        if key.lower() in _SENSITIVE_FIELDS:
            event_dict[key] = "***REDACTED***"
        elif isinstance(event_dict[key], dict):
            for subkey in list(event_dict[key].keys()):
                if subkey.lower() in _SENSITIVE_FIELDS:
                    event_dict[key][subkey] = "***REDACTED***"
    return event_dict


def _add_app_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add application context to every log entry."""
    event_dict.setdefault("service", "riskpulse")
    event_dict.setdefault("environment", get_settings().environment)
    return event_dict


def configure_logging(log_level: str | None = None) -> None:
    """Configure structured logging for the application.

    Args:
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                   If None, reads from config.
    """
    settings = get_settings()
    level = log_level or settings.get("monitoring.log_level", "INFO")
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Shared processors for all log entries
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _add_app_context,
        _scrub_pii,
    ]

    # Choose renderer based on environment
    if settings.is_debug:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure formatter for stdlib handler
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    # Apply to root handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)
        **initial_context: Additional key-value pairs bound to all log entries

    Returns:
        Configured structlog BoundLogger instance

    Example:
        logger = get_logger(__name__, component="kafka_producer")
        logger.info("Message published", topic="txn.raw.events", partition=3)
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


def bind_correlation_id(correlation_id: str) -> None:
    """Bind a correlation ID to the current context (thread/async task).

    All subsequent log entries in this context will include the correlation_id.

    Args:
        correlation_id: Unique identifier for request tracing
    """
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)


def clear_context() -> None:
    """Clear all context variables (call at end of request/task)."""
    structlog.contextvars.clear_contextvars()
