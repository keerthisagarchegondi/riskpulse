"""Read-only production secret and PostgreSQL privilege verification."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from typing import Any

import asyncpg

from src.utils.config import get_settings
from src.utils.secrets_manager import get_secrets_manager

_TABLE_PRIVILEGES = {
    "transactions": ("SELECT", "INSERT", "UPDATE"),
    "risk_scores": ("SELECT", "INSERT"),
    "fraud_alerts": ("SELECT", "INSERT", "UPDATE"),
}


def _role_issues(role: Mapping[str, Any]) -> list[str]:
    restricted = (
        "rolsuper",
        "rolcreaterole",
        "rolcreatedb",
        "rolreplication",
        "rolbypassrls",
        "database_create",
        "schema_create",
    )
    return [name for name in restricted if role.get(name) is not False]


async def verify() -> dict[str, Any]:
    settings = get_settings()
    if settings.environment.lower() not in {"prod", "production", "staging"}:
        raise RuntimeError("Runtime security verification requires a managed environment")

    secrets = get_secrets_manager()
    credentials = secrets.get_database_credentials()
    api_keys = secrets.get_api_keys()
    jwt_secret = secrets.get_jwt_secret()
    signing_key = os.environ.get("RISKPULSE_MODEL_SIGNING_KEY", "")
    if not api_keys or any(len(str(item.get("key", "")).encode()) < 32 for item in api_keys):
        raise RuntimeError("Managed API keys are missing or shorter than 32 bytes")
    if len(jwt_secret.encode()) < 32 or len(signing_key.encode()) < 32:
        raise RuntimeError("Managed signing secrets are missing or shorter than 32 bytes")

    connection = await asyncpg.connect(
        host=str(credentials["host"]),
        port=int(credentials["port"]),
        user=str(credentials["username"]),
        password=str(credentials["password"]),
        database=str(credentials["database"]),
        timeout=5,
    )
    try:
        role = await connection.fetchrow("""SELECT r.rolsuper, r.rolcreaterole, r.rolcreatedb,
                      r.rolreplication, r.rolbypassrls,
                      has_database_privilege(current_user, current_database(), 'CREATE') AS database_create,
                      has_schema_privilege(current_user, 'public', 'CREATE') AS schema_create
               FROM pg_roles AS r WHERE r.rolname = current_user""")
        if role is None:
            raise RuntimeError("Application database role could not be inspected")
        issues = _role_issues(dict(role))
        for table, privileges in _TABLE_PRIVILEGES.items():
            relation = f"public.{table}"
            if not await connection.fetchval("SELECT to_regclass($1) IS NOT NULL", relation):
                issues.append(f"{table}_missing")
                continue
            for privilege in privileges:
                allowed = await connection.fetchval(
                    "SELECT has_table_privilege(current_user, $1, $2)", relation, privilege
                )
                if not allowed:
                    issues.append(f"{table}_{privilege.lower()}_missing")
        if issues:
            raise RuntimeError("Database role checks failed: " + ", ".join(issues))
    finally:
        await connection.close()

    return {"status": "pass", "secrets": "validated", "database_role": "least_privilege"}


def main() -> int:
    try:
        print(json.dumps(asyncio.run(verify())))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "error": str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__,
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
