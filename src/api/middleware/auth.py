"""API key authentication middleware for RiskPulse API.

Validates API keys passed via X-API-Key header against configured keys.
Supports multiple API keys with different permission levels.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Any

import structlog
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from src.utils.config import get_settings

logger = structlog.get_logger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyManager:
    """Manages API key validation and metadata lookup."""

    def __init__(self) -> None:
        self._keys: dict[str, dict[str, Any]] = {}
        self._load_keys()

    def _load_keys(self) -> None:
        """Load API keys from configuration.

        In production, these would come from AWS Secrets Manager or a secure vault.
        For development, they are loaded from settings.
        """
        settings = get_settings()
        configured_keys = settings.get("api.api_keys", [])

        if isinstance(configured_keys, list):
            for entry in configured_keys:
                if isinstance(entry, dict) and "key" in entry:
                    key_hash = self._hash_key(entry["key"])
                    self._keys[key_hash] = {
                        "name": entry.get("name", "unknown"),
                        "permissions": entry.get("permissions", ["read"]),
                        "rate_limit": entry.get("rate_limit"),
                    }

        # Always include a development key in non-production environments
        env = settings.environment
        if env in ("dev", "test"):
            dev_key_hash = self._hash_key("dev-api-key-riskpulse-2024")
            self._keys[dev_key_hash] = {
                "name": "development",
                "permissions": ["read", "write", "admin"],
                "rate_limit": None,
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
) -> dict[str, Any]:
    """Dependency that validates the API key from request headers.

    Returns the key metadata (name, permissions) if valid.
    Raises 401 if missing or invalid.
    """
    if api_key is None:
        logger.warning("api_key_missing", path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    manager = get_key_manager()
    metadata = manager.validate_key(api_key)

    if metadata is None:
        logger.warning("api_key_invalid", path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    # Attach API key identity to request state for downstream use
    request.state.api_key_name = metadata["name"]
    request.state.api_key_permissions = metadata["permissions"]

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
