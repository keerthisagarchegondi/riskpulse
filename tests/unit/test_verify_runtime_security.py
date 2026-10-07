"""Read-only production runtime verification tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from scripts import verify_runtime_security


def _restricted_role() -> dict[str, bool]:
    return {
        "rolsuper": False,
        "rolcreaterole": False,
        "rolcreatedb": False,
        "rolreplication": False,
        "rolbypassrls": False,
        "database_create": False,
        "schema_create": False,
    }


def test_role_check_rejects_superuser_and_schema_creation() -> None:
    role = _restricted_role()
    role["rolsuper"] = True
    role["schema_create"] = True
    assert verify_runtime_security._role_issues(role) == ["rolsuper", "schema_create"]


@pytest.mark.asyncio
async def test_runtime_check_verifies_secrets_connectivity_and_table_privileges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RISKPULSE_MODEL_SIGNING_KEY", "m" * 32)
    monkeypatch.setattr(
        verify_runtime_security, "get_settings", lambda: SimpleNamespace(environment="prod")
    )
    secrets = Mock()
    secrets.get_database_credentials.return_value = {
        "host": "postgres",
        "port": "5432",
        "username": "app",
        "password": "test-secret",
        "database": "riskpulse",
    }
    secrets.get_api_keys.return_value = [{"key": "k" * 32}]
    secrets.get_jwt_secret.return_value = "j" * 32
    monkeypatch.setattr(verify_runtime_security, "get_secrets_manager", lambda: secrets)
    connection = Mock()
    connection.fetchrow = AsyncMock(return_value=_restricted_role())
    connection.fetchval = AsyncMock(return_value=True)
    connection.close = AsyncMock()
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(verify_runtime_security.asyncpg, "connect", connect)

    result = await verify_runtime_security.verify()

    assert result["status"] == "pass"
    assert connection.fetchval.await_count == 11
    connection.close.assert_awaited_once()
    assert connect.await_args.kwargs["host"] == "postgres"


@pytest.mark.asyncio
async def test_runtime_check_rejects_missing_signing_key_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RISKPULSE_MODEL_SIGNING_KEY", raising=False)
    monkeypatch.setattr(
        verify_runtime_security, "get_settings", lambda: SimpleNamespace(environment="prod")
    )
    secrets = Mock()
    secrets.get_database_credentials.return_value = {}
    secrets.get_api_keys.return_value = [{"key": "k" * 32}]
    secrets.get_jwt_secret.return_value = "j" * 32
    monkeypatch.setattr(verify_runtime_security, "get_secrets_manager", lambda: secrets)
    connect = AsyncMock()
    monkeypatch.setattr(verify_runtime_security.asyncpg, "connect", connect)

    with pytest.raises(RuntimeError, match="Managed signing secrets"):
        await verify_runtime_security.verify()
    connect.assert_not_awaited()
