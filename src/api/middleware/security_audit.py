"""Security audit logging middleware."""

from __future__ import annotations

import time
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.monitoring.cloudwatch_logger import scrub_pii

audit_logger = structlog.get_logger("riskpulse.security_audit")


class SecurityAuditMiddleware(BaseHTTPMiddleware):
    """Emit one structured security audit event for every HTTP request."""

    EXEMPT_PATHS = frozenset({"/health/live"})

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        status_code = 500
        response: Response | None = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            if request.url.path not in self.EXEMPT_PATHS:
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                actor = getattr(request.state, "api_key_name", None) or getattr(
                    request.state, "jwt_subject", None
                )
                event: dict[str, Any] = {
                    "event_type": "security_access",
                    "action": f"{request.method} {request.url.path}",
                    "actor": actor or "anonymous",
                    "auth_type": getattr(request.state, "auth_type", "unknown"),
                    "status_code": status_code,
                    "client_ip": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent"),
                    "correlation_id": getattr(request.state, "correlation_id", None),
                    "duration_ms": duration_ms,
                }
                audit_logger.info("security_audit_event", **scrub_pii(event))
