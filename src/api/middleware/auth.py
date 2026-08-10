"""API authentication middleware for RiskPulse API.

Validates API keys passed via X-API-Key and JWT bearer tokens passed via
Authorization. Supports multiple identities with different permission levels.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Any

import structlog
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from src.utils.config import get_settings
from src.utils.secrets_manager import SecretsManagerError, get_secrets_manager
from src.utils.security import SecurityValidationError, verify_jwt_token

logger = structlog.get_logger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
BEARER_TOKEN = HTTPBearer(auto_error=False)


class APIKeyManager:
    """Manages API key validation and metadata lookup."""

    def __init__(self) -> None:
        self._keys: dict[str, dict[str, Any]] = {}
        self._load_keys()

    def _load_keys(self) -> None:
        """Load API keys from configuration.

        Production deployments should configure AWS Secrets Manager. Development
        and tests can still load local keys from settings.
        """
        settings = get_settings()
        try:
            configured_keys = get_secrets_manager().get_api_keys()
        except SecretsManagerError as exc:
            logger.warning("api_keys_secret_unavailable", error=str(exc))
            configured_keys = settings.get("api.api_keys", [])

        if isinstance(configured_keys, list):
            for entry in configured_keys:
                if isinstance(entry, dict) and "key" in entry:
                    key_hash = self._hash_key(entry["key"])
                    self._keys[key_hash] = {
                        "name": entry.get("name", "unknown"),
                        "permissions": entry.get("permissions", ["read"]),
                        "rate_limit": entry.get("rate_limit"),
                        "auth_type": "api_key",
                    }

        # Always include a development key in non-production environments
        env = settings.environment
        if env in ("dev", "test"):
            dev_key_hash = self._hash_key("dev-api-key-riskpulse-2024")
            self._keys[dev_key_hash] = {
                "name": "development",
                "permissions": ["read", "write", "admin"],
                "rate_limit": None,
                "auth_type": "api_key",
            }

    @staticmethod
    def _hash_key(key: str) -> str:
        """Create a constant-time comparable hash of an API key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def validate_key(self, api_key: str) -> dict[str, Any] | None:
        """Validate an API key and return its metadata.

        Uses constant-time comparison to prevent timing attacks.
        """
        key_hash = self._hash_key(api_key)
        for stored_hash, metadata in self._keys.items():
            if hmac.compare_digest(key_hash, stored_hash):
                return metadata
        return None

    @staticmethod
    def generate_key() -> str:
        """Generate a new cryptographically secure API key."""
        return f"rp_{secrets.token_urlsafe(32)}"


# Module-level singleton
_key_manager: APIKeyManager | None = None


def get_key_manager() -> APIKeyManager:
    """Get or create the API key manager singleton."""
    global _key_manager
    if _key_manager is None:
        _key_manager = APIKeyManager()
    return _key_manager


def reset_key_manager() -> None:
    """Reset the key manager (useful for testing)."""
    global _key_manager
    _key_manager = None


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(API_KEY_HEADER),
    bearer: HTTPAuthorizationCredentials | None = Security(BEARER_TOKEN),
) -> dict[str, Any]:
    """Dependency that validates API keys or JWT bearer tokens.

    Returns the key metadata (name, permissions) if valid.
    Raises 401 if missing or invalid.
    """
    if bearer is not None:
        try:
            metadata = verify_jwt_token(bearer.credentials)
        except SecurityValidationError:
            logger.warning("jwt_invalid", path=request.url.path)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token.",
            )

        _bind_auth_metadata(request, metadata)
        request.state.jwt_subject = metadata["name"]
        return metadata

    if api_key is not None:
        manager = get_key_manager()
        metadata = manager.validate_key(api_key)

        if metadata is None:
            logger.warning("api_key_invalid", path=request.url.path)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )

        _bind_auth_metadata(request, metadata)
        return metadata

    logger.warning("auth_missing", path=request.url.path)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing API key or bearer token. Provide X-API-Key or Authorization bearer token.",
    )


def _bind_auth_metadata(request: Request, metadata: dict[str, Any]) -> None:
    """Attach auth identity to request state for logging and rate limiting."""
    request.state.api_key_name = metadata["name"]
    request.state.api_key_permissions = metadata.get("permissions", [])
    request.state.api_key_rate_limit = metadata.get("rate_limit")
    request.state.auth_type = metadata.get("auth_type", "api_key")


async def verify_api_key_only(
    request: Request,
    api_key: str | None = Security(API_KEY_HEADER),
) -> dict[str, Any]:
    """Strict API-key-only dependency for routes that require shared keys."""
    if api_key is None:
        logger.warning("api_key_missing", path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    metadata = get_key_manager().validate_key(api_key)
    if metadata is None:
        logger.warning("api_key_invalid", path=request.url.path)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    _bind_auth_metadata(request, metadata)
    return metadata


def require_permission(permission: str):
    """Factory for creating permission-checking dependencies.

    Usage:
        @router.post("/admin/action", dependencies=[Depends(require_permission("admin"))])
    """

    async def _check_permission(
        request: Request,
        key_meta: dict[str, Any] = Security(verify_api_key),
    ) -> dict[str, Any]:
        if permission not in key_meta.get("permissions", []):
            logger.warning(
                "permission_denied",
                path=request.url.path,
                required=permission,
                key_name=key_meta.get("name"),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}",
            )
        return key_meta

    return _check_permission
