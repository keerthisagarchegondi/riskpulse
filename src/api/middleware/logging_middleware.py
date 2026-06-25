"""Request/response logging middleware for RiskPulse API.

Captures request metadata, response status, and latency for all API calls.
Integrates with structlog for JSON-structured output and correlation ID tracking.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.utils.logger import bind_correlation_id, clear_context

logger = structlog.get_logger(__name__)

# Headers that should never be logged
SENSITIVE_HEADERS = frozenset({"authorization", "x-api-key", "cookie", "set-cookie"})

# Paths for which request body should not be logged
BODY_LOG_EXCLUDE_PATHS = frozenset({"/health", "/health/ready", "/health/live"})


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs all HTTP requests and responses with timing.

    For each request, logs:
    - Method, path, query params
    - Client IP and user agent
    - Correlation ID (generated or propagated)
    - Response status and latency
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with logging."""
        # Generate or propagate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        bind_correlation_id(correlation_id)

        # Store on request state for downstream use
        request.state.correlation_id = correlation_id

        # Extract request metadata
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        method = request.method
        path = request.url.path
        query = str(request.query_params) if request.query_params else None

        # Log request (skip health checks at debug level)
        log_level = "debug" if path.startswith("/health") else "info"
        getattr(logger, log_level)(
            "request_started",
            method=method,
            path=path,
            query=query,
            client_ip=client_ip,
            user_agent=user_agent[:100] if user_agent else None,
            correlation_id=correlation_id,
        )

        # Time the request
        start_time = time.perf_counter()
        status_code = 500  # Default in case of unhandled exception

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log response
            log_data = {
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
                "correlation_id": correlation_id,
            }

            if hasattr(request.state, "api_key_name"):
                log_data["api_key_name"] = request.state.api_key_name

            if status_code >= 500:
                logger.error("request_completed", **log_data)
            elif status_code >= 400:
                logger.warning("request_completed", **log_data)
            else:
                getattr(logger, log_level)("request_completed", **log_data)

            # Clean up context vars
            clear_context()

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Request-Duration-Ms"] = str(round(duration_ms, 2))

        return response
