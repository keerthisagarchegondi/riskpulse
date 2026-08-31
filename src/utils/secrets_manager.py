"""AWS Secrets Manager integration for RiskPulse credentials.

Secrets access is optional and injectable so application code can use the same
interface in local tests, CI, and AWS workloads. Production deployments should
grant each service role access only to the specific secret ARNs it needs.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import structlog

from src.utils.config import get_settings

logger = structlog.get_logger(__name__)

_DEVELOPMENT_PLACEHOLDERS = {
    "",
    "riskpulse",
    "riskpulse_dev_password",
    "change-me",
    "dev-api-key-riskpulse-2024",
    "dev-jwt-secret",
}


class SecretsManagerError(Exception):
    """Raised when a secret cannot be loaded or parsed."""


@dataclass(frozen=True)
class SecretValue:
    """A parsed secret value with retrieval metadata."""

    secret_id: str
    value: dict[str, Any]
    version_id: str | None = None
    created_at: float = 0.0


class SecretsManager:
    """Small cached wrapper around AWS Secrets Manager."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        region_name: str | None = None,
        cache_ttl_seconds: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.enabled = (
            enabled
            if enabled is not None
            else bool(settings.get("security.secrets_manager.enabled", False))
        )
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self.cache_ttl_seconds = cache_ttl_seconds or int(
            settings.get("security.secrets_manager.cache_ttl_seconds", 300)
        )
        self._client = (
            client if client is not None else (self._create_client() if self.enabled else None)
        )
        self._cache: dict[str, SecretValue] = {}

    @staticmethod
    def _is_managed_environment(environment: str) -> bool:
        return environment.lower() in {"prod", "production", "staging"}

    @staticmethod
    def _coerce_api_key_entries(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
            return []
        return [item for item in value if isinstance(item, dict) and item.get("key")]

    def _get_api_keys_from_environment(self, *, managed_environment: bool) -> list[dict[str, Any]]:
        raw_json = os.environ.get("RISKPULSE_API_KEYS")
        if raw_json:
            try:
                parsed = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                raise SecretsManagerError("RISKPULSE_API_KEYS must be valid JSON") from exc
            keys = self._coerce_api_key_entries(parsed)
            if not keys:
                raise SecretsManagerError("RISKPULSE_API_KEYS must contain API key entries")
            return keys

        single_key = os.environ.get("RISKPULSE_API_KEY", "")
        if single_key:
            if managed_environment and single_key in _DEVELOPMENT_PLACEHOLDERS:
                raise SecretsManagerError(
                    "Production API key cannot use a development placeholder value"
                )
            return [
                {
                    "name": os.environ.get("RISKPULSE_API_KEY_NAME", "environment"),
                    "key": single_key,
                    "permissions": [
                        item.strip()
                        for item in os.environ.get(
                            "RISKPULSE_API_KEY_PERMISSIONS", "read,write"
                        ).split(",")
                        if item.strip()
                    ],
                    "rate_limit": os.environ.get("RISKPULSE_API_KEY_RATE_LIMIT"),
                }
            ]

        return []

    def get_secret(self, secret_id: str, *, force_refresh: bool = False) -> dict[str, Any]:
        """Return a secret as a dictionary.

        JSON secrets are parsed directly. Plain string secrets are returned under
        the ``value`` key so callers can use the same shape consistently.
        """
        if not secret_id:
            raise SecretsManagerError("secret_id is required")

        cached = self._cache.get(secret_id)
        if (
            cached
            and not force_refresh
            and (time.time() - cached.created_at) < self.cache_ttl_seconds
        ):
            return cached.value.copy()

        if self._client is None:
            raise SecretsManagerError("Secrets Manager is disabled and no client was provided")

        try:
            response = self._client.get_secret_value(SecretId=secret_id)
        except Exception as exc:  # pragma: no cover - AWS SDK exceptions vary by version
            logger.error("secret_load_failed", secret_id=secret_id, error=str(exc))
            raise SecretsManagerError(f"Failed to load secret {secret_id}") from exc

        raw_value = response.get("SecretString")
        if raw_value is None and response.get("SecretBinary") is not None:
            raw_value = response["SecretBinary"].decode("utf-8")
        if raw_value is None:
            raise SecretsManagerError(f"Secret {secret_id} contains no usable value")

        try:
            parsed = json.loads(raw_value)
            value = parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            value = {"value": raw_value}

        self._cache[secret_id] = SecretValue(
            secret_id=secret_id,
            value=value,
            version_id=response.get("VersionId"),
            created_at=time.time(),
        )
        return value.copy()

    def get_database_credentials(self, secret_id: str | None = None) -> dict[str, Any]:
        settings = get_settings()
        configured_id = secret_id or settings.get("security.secrets_manager.database_secret_id")
        if configured_id:
            return self.get_secret(configured_id)
        credentials = {
            "host": os.environ.get("RISKPULSE_DB_HOST", "localhost"),
            "port": os.environ.get("RISKPULSE_DB_PORT", "5432"),
            "database": os.environ.get("RISKPULSE_DB_NAME", "riskpulse"),
            "username": os.environ.get("RISKPULSE_DB_USER", "riskpulse"),
            "password": os.environ.get("RISKPULSE_DB_PASSWORD", "riskpulse"),
        }
        if (
            self._is_managed_environment(settings.environment)
            and credentials["password"] in _DEVELOPMENT_PLACEHOLDERS
        ):
            raise SecretsManagerError(
                "Production database credentials must come from Secrets Manager "
                "or a non-default RISKPULSE_DB_PASSWORD value"
            )
        return credentials

    def get_api_keys(self, secret_id: str | None = None) -> list[dict[str, Any]]:
        settings = get_settings()
        configured_id = secret_id or settings.get("security.secrets_manager.api_keys_secret_id")
        managed_environment = self._is_managed_environment(settings.environment)
        if configured_id:
            secret = self.get_secret(configured_id)
            keys = secret.get("api_keys", secret.get("keys", []))
            if not isinstance(keys, list):
                raise SecretsManagerError("API key secret must contain an api_keys list")
            return self._coerce_api_key_entries(keys)

        env_keys = self._get_api_keys_from_environment(managed_environment=managed_environment)
        if env_keys:
            return env_keys

        configured = settings.get("api.api_keys", [])
        configured_keys = self._coerce_api_key_entries(configured)
        if configured_keys:
            return configured_keys

        if managed_environment:
            raise SecretsManagerError(
                "Production API keys must come from Secrets Manager, RISKPULSE_API_KEYS, "
                "or a non-placeholder RISKPULSE_API_KEY value"
            )

        return []

    def get_jwt_secret(self, secret_id: str | None = None) -> str:
        settings = get_settings()
        configured_id = secret_id or settings.get("security.secrets_manager.jwt_secret_id")
        if configured_id:
            secret = self.get_secret(configured_id)
            value = secret.get("jwt_secret") or secret.get("secret") or secret.get("value")
            if value:
                return str(value)

        env_secret = os.environ.get("RISKPULSE_JWT_SECRET")
        if env_secret and env_secret not in _DEVELOPMENT_PLACEHOLDERS:
            return env_secret

        if self._is_managed_environment(settings.environment):
            raise SecretsManagerError(
                "Production JWT secret must be configured through Secrets Manager "
                "or RISKPULSE_JWT_SECRET; refusing to use development fallback"
            )

        return env_secret or settings.get("security.jwt.dev_secret", "dev-jwt-secret")

    def _create_client(self) -> Any:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SecretsManagerError(
                "boto3 is required for AWS Secrets Manager integration"
            ) from exc
        return boto3.client("secretsmanager", region_name=self.region_name)


_manager: SecretsManager | None = None


def get_secrets_manager() -> SecretsManager:
    """Return the module-level Secrets Manager wrapper."""
    global _manager
    if _manager is None:
        _manager = SecretsManager()
    return _manager


def reset_secrets_manager() -> None:
    """Reset the singleton, mostly for tests."""
    global _manager
    _manager = None
