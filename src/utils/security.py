"""Application security helpers for sanitization and token handling."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from jwt import PyJWTError

from src.utils.config import get_settings
from src.utils.secrets_manager import get_secrets_manager

JWT_ALGORITHM = "HS256"
JWT_ISSUER = "riskpulse-api"

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SQLI_PATTERNS = (
    re.compile(
        r"(?i)(?:--|/\*|\*/|;|\bunion\b|\bselect\b|\binsert\b|\bupdate\b|\bdelete\b|\bdrop\b|\balter\b)"
    ),
    re.compile(r"(?i)\bor\s+1\s*=\s*1\b"),
    re.compile(r"(?i)\band\s+1\s*=\s*1\b"),
)
SCRIPT_PATTERNS = (
    re.compile(r"(?i)<\s*/?\s*script\b"),
    re.compile(r"(?i)javascript\s*:"),
    re.compile(r"(?i)on(?:error|load|click)\s*="),
)


class SecurityValidationError(ValueError):
    """Raised when input contains unsafe content."""


def sanitize_string(
    value: str, *, max_length: int | None = None, reject_sql_tokens: bool = True
) -> str:
    """Normalize and validate user-controlled text."""
    cleaned = CONTROL_CHARS_RE.sub("", value).strip()

    if max_length is not None and len(cleaned) > max_length:
        raise SecurityValidationError(f"Value exceeds maximum length {max_length}")

    if any(pattern.search(cleaned) for pattern in SCRIPT_PATTERNS):
        raise SecurityValidationError("Unsafe script pattern detected")

    if reject_sql_tokens and any(pattern.search(cleaned) for pattern in SQLI_PATTERNS):
        raise SecurityValidationError("Unsafe input pattern detected")

    return cleaned


def sanitize_mapping(
    value: dict[str, Any], *, depth: int = 0, max_depth: int = 5
) -> dict[str, Any]:
    """Recursively sanitize metadata-style dictionaries."""
    if depth > max_depth:
        raise SecurityValidationError("Metadata exceeds maximum nesting depth")

    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        safe_key = sanitize_string(str(key), max_length=64, reject_sql_tokens=True)
        if isinstance(item, str):
            sanitized[safe_key] = sanitize_string(item, max_length=2048, reject_sql_tokens=False)
        elif isinstance(item, dict):
            sanitized[safe_key] = sanitize_mapping(item, depth=depth + 1, max_depth=max_depth)
        elif isinstance(item, list):
            sanitized[safe_key] = [
                (
                    sanitize_string(element, max_length=2048, reject_sql_tokens=False)
                    if isinstance(element, str)
                    else element
                )
                for element in item[:100]
            ]
        else:
            sanitized[safe_key] = item
    return sanitized


def create_jwt_token(
    *,
    subject: str,
    permissions: list[str],
    expires_minutes: int | None = None,
    secret: str | None = None,
) -> str:
    """Create a signed JWT for service or analyst API access."""
    settings = get_settings()
    expires = expires_minutes or int(settings.get("security.jwt.expiration_minutes", 60))
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "permissions": permissions,
        "iss": JWT_ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires)).timestamp()),
    }
    signing_secret = secret or get_secrets_manager().get_jwt_secret()
    return jwt.encode(payload, signing_secret, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str, *, secret: str | None = None) -> dict[str, Any]:
    """Verify a RiskPulse JWT and return normalized auth metadata."""
    signing_secret = secret or get_secrets_manager().get_jwt_secret()
    try:
        payload = jwt.decode(
            token,
            signing_secret,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            options={"verify_aud": False},
        )
    except PyJWTError as exc:
        raise SecurityValidationError("Invalid JWT token") from exc

    permissions = payload.get("permissions", [])
    if not isinstance(permissions, list):
        permissions = []

    return {
        "name": str(payload.get("sub", "jwt-subject")),
        "permissions": [str(permission) for permission in permissions],
        "rate_limit": payload.get("rate_limit"),
        "auth_type": "jwt",
    }
