"""Tests for Day 35 IAM, secrets, and application security controls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.middleware.auth import reset_key_manager
from src.utils.config import get_settings
from src.utils.secrets_manager import SecretsManager, SecretsManagerError
from src.utils.security import (
    SecurityValidationError,
    create_jwt_token,
    sanitize_string,
    verify_jwt_token,
)
from src.utils.sql_security import SqlFilter, UnsafeQueryError, build_where_clause

UNIT_SECRET = "unit-secret-for-hs256-tests-32-bytes"
PROD_SECRET = "prod-unit-secret-for-hs256-tests-32-bytes"
DEV_JWT_SECRET = "dev-jwt-secret-for-hs256-tests-32-bytes"


class FakeSecretsClient:
    def __init__(self) -> None:
        self.calls = 0

    def get_secret_value(self, SecretId: str):
        self.calls += 1
        return {
            "SecretString": json.dumps(
                {
                    "api_keys": [
                        {
                            "name": "ci",
                            "key": "rp-ci-key",
                            "permissions": ["read", "write"],
                            "rate_limit": 500,
                        }
                    ],
                    "jwt_secret": UNIT_SECRET,
                }
            ),
            "VersionId": "v1",
        }


class FakeBoto3:
    def __init__(self) -> None:
        self.region_name = None

    def client(self, service_name: str, *, region_name: str | None = None):
        self.region_name = region_name
        assert service_name == "secretsmanager"
        return FakeSecretsClient()


@pytest.fixture(autouse=True)
def _reset_auth_state():
    reset_key_manager()
    yield
    reset_key_manager()


@pytest.fixture
def client():
    with patch("src.api.app.configure_logging"):
        return TestClient(create_app())


@pytest.fixture
def valid_transaction():
    return {
        "external_transaction_id": "TXN-SEC-001",
        "account_id": "ACC-SEC",
        "customer_id": "CUST-SEC",
        "transaction_amount": "125.50",
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "transaction_timestamp": "2026-08-10T12:00:00Z",
    }


def test_secrets_manager_parses_and_caches_json_secret() -> None:
    fake = FakeSecretsClient()
    manager = SecretsManager(client=fake, enabled=True, cache_ttl_seconds=60)

    first = manager.get_secret("riskpulse/dev/api")
    second = manager.get_secret("riskpulse/dev/api")

    assert first["jwt_secret"] == UNIT_SECRET
    assert second["api_keys"][0]["name"] == "ci"
    assert fake.calls == 1


def test_secrets_manager_passes_configured_region(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_boto3 = FakeBoto3()

    with patch.dict("sys.modules", {"boto3": fake_boto3}):
        manager = SecretsManager(enabled=True, region_name="us-west-2")

    assert fake_boto3.region_name == "us-west-2"
    assert manager.get_secret("riskpulse/dev/api")["jwt_secret"] == UNIT_SECRET


def test_prod_jwt_secret_refuses_development_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RISKPULSE_ENV", "prod")
    monkeypatch.delenv("RISKPULSE_JWT_SECRET", raising=False)
    get_settings.cache_clear()

    try:
        manager = SecretsManager(enabled=False)

        with pytest.raises(SecretsManagerError, match="Production JWT secret"):
            manager.get_jwt_secret()
    finally:
        get_settings.cache_clear()


def test_prod_jwt_secret_accepts_explicit_environment_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RISKPULSE_ENV", "prod")
    monkeypatch.setenv("RISKPULSE_JWT_SECRET", PROD_SECRET)
    get_settings.cache_clear()

    try:
        manager = SecretsManager(enabled=False)

        assert manager.get_jwt_secret() == PROD_SECRET
    finally:
        get_settings.cache_clear()


def test_jwt_token_round_trip_uses_permissions() -> None:
    token = create_jwt_token(
        subject="service:worker",
        permissions=["read", "write"],
        secret=UNIT_SECRET,
    )

    metadata = verify_jwt_token(token, secret=UNIT_SECRET)

    assert metadata["name"] == "service:worker"
    assert metadata["permissions"] == ["read", "write"]
    assert metadata["auth_type"] == "jwt"


def test_api_accepts_jwt_bearer_token(
    client,
    valid_transaction,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RISKPULSE_JWT_SECRET", DEV_JWT_SECRET)
    token = create_jwt_token(
        subject="analyst@example.com",
        permissions=["read", "write"],
        secret=DEV_JWT_SECRET,
    )

    with patch("src.api.routes.transactions._get_kafka_producer", return_value=None):
        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == status.HTTP_202_ACCEPTED


def test_unsafe_transaction_payload_is_rejected(client, valid_transaction) -> None:
    payload = {
        **valid_transaction,
        "external_transaction_id": "TXN-1'; DROP TABLE transactions; --",
    }

    response = client.post(
        "/api/v1/transactions",
        json=payload,
        headers={"X-API-Key": "dev-api-key-riskpulse-2024"},
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_query_filter_injection_is_rejected_before_storage(client) -> None:
    storage = MagicMock()

    with patch("src.api.routes.transactions._get_storage", return_value=storage):
        response = client.get(
            "/api/v1/transactions?account_id=ACC-1%27%20OR%201%3D1%20--",
            headers={"X-API-Key": "dev-api-key-riskpulse-2024"},
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    storage.list_transactions.assert_not_called()


def test_sanitize_string_rejects_script_payloads() -> None:
    with pytest.raises(SecurityValidationError):
        sanitize_string("<script>alert(1)</script>")


def test_sql_builder_uses_positional_params_and_allowlisted_columns() -> None:
    clause, params, next_idx = build_where_clause(
        [
            SqlFilter("account_id", "=", "ACC-1"),
            SqlFilter("transaction_amount", ">=", 100.0),
        ]
    )

    assert clause == "account_id = $1 AND transaction_amount >= $2"
    assert params == ["ACC-1", 100.0]
    assert next_idx == 3


def test_sql_builder_rejects_unsafe_column() -> None:
    with pytest.raises(UnsafeQueryError):
        build_where_clause([SqlFilter("account_id; DROP TABLE transactions", "=", "ACC-1")])


def test_iam_policy_artifacts_are_valid_json_and_least_privilege() -> None:
    policy_dir = Path("infrastructure/aws/iam_policies")
    policies = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in policy_dir.glob("*.json")
    }

    assert {"s3_access_policy.json", "cloudwatch_policy.json", "snowflake_policy.json"}.issubset(
        policies
    )
    cloudwatch_actions = policies["cloudwatch_policy.json"]["Statement"][0]["Action"]
    assert "logs:PutLogEvents" in cloudwatch_actions
    assert "*" not in policies["s3_access_policy.json"]["Statement"][0]["Action"]


def test_iam_terraform_module_declares_required_roles_and_secrets() -> None:
    module_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("infrastructure/terraform/modules/iam").glob("*.tf")
    )

    for role_name in (
        "api",
        "worker",
        "airflow",
        "dashboard",
        "admin",
    ):
        assert f'resource "aws_iam_role" "{role_name}"' in module_text

    assert 'resource "aws_secretsmanager_secret_rotation" "database"' in module_text
    assert "secretsmanager:GetSecretValue" in module_text


def test_security_architecture_uses_txt_not_markdown() -> None:
    assert Path("docs/security_architecture.txt").exists()
    assert not Path("docs/security_architecture.md").exists()
