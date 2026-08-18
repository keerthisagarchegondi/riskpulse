"""Security tests for injection and unsafe-content handling."""

from __future__ import annotations

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.app import create_app
from src.api.middleware.auth import reset_key_manager
from src.api.schemas.transaction_schema import TransactionCreate
from src.utils.security import SecurityValidationError, sanitize_mapping, sanitize_string
from src.utils.sql_security import SqlFilter, UnsafeQueryError, build_where_clause

AUTH_HEADERS = {"X-API-Key": "dev-api-key-riskpulse-2024"}


@pytest.fixture(autouse=True)
def _reset_auth_state():
    reset_key_manager()
    yield
    reset_key_manager()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def transaction_payload() -> dict[str, object]:
    return {
        "external_transaction_id": "TXN-SECURITY-001",
        "account_id": "ACC-SECURITY",
        "customer_id": "CUST-SECURITY",
        "merchant_id": "MERCH-001",
        "merchant_name": "Known Merchant",
        "transaction_amount": "42.10",
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "transaction_timestamp": "2026-08-13T12:00:00Z",
    }


@pytest.mark.security
@pytest.mark.parametrize(
    "field,payload",
    [
        ("external_transaction_id", "TXN-1'; DROP TABLE transactions; --"),
        ("account_id", "ACC-1 OR 1=1 --"),
        ("customer_id", "CUST-1 UNION SELECT password FROM users"),
        ("merchant_name", "merchant */ UPDATE transactions SET amount = 0"),
    ],
)
def test_transaction_schema_rejects_sql_injection_payloads(
    transaction_payload: dict[str, object],
    field: str,
    payload: str,
) -> None:
    transaction_payload[field] = payload

    with pytest.raises(ValidationError):
        TransactionCreate.model_validate(transaction_payload)


@pytest.mark.security
@pytest.mark.parametrize(
    "payload",
    [
        "<script>alert('xss')</script>",
        "javascript:alert(document.cookie)",
        "<img src=x onerror=alert(1)>",
        "Safe text\x00 with control chars",
    ],
)
def test_sanitizer_blocks_xss_and_strips_control_chars(payload: str) -> None:
    if "\x00" in payload:
        assert sanitize_string(payload, reject_sql_tokens=False) == "Safe text with control chars"
    else:
        with pytest.raises(SecurityValidationError):
            sanitize_string(payload, reject_sql_tokens=False)


@pytest.mark.security
def test_nested_metadata_rejects_script_values(transaction_payload: dict[str, object]) -> None:
    transaction_payload["metadata"] = {
        "checkout": {
            "browser": "Chrome",
            "notes": "<script>steal()</script>",
        }
    }

    with pytest.raises(ValidationError):
        TransactionCreate.model_validate(transaction_payload)


@pytest.mark.security
def test_metadata_rejects_unsafe_keys() -> None:
    with pytest.raises(SecurityValidationError):
        sanitize_mapping({"safe_key; DROP TABLE audit_log": "value"})


@pytest.mark.security
def test_query_filter_injection_is_rejected_before_storage(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StorageShouldNotRun:
        async def list_transactions(self, filters):  # pragma: no cover
            raise AssertionError("Unsafe query reached storage")

    monkeypatch.setattr(
        "src.api.routes.transactions._get_storage",
        lambda: StorageShouldNotRun(),
    )

    response = client.get(
        "/api/v1/transactions?account_id=ACC-1%27%20OR%201%3D1%20--",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "Unsafe input pattern" in response.text


@pytest.mark.security
def test_where_clause_is_parameterized_for_allowed_filters() -> None:
    clause, params, next_index = build_where_clause(
        [
            SqlFilter("account_id", "=", "ACC-1"),
            SqlFilter("transaction_amount", ">=", 100.0),
            SqlFilter("status", "=", "approved"),
        ]
    )

    assert clause == "account_id = $1 AND transaction_amount >= $2 AND status = $3"
    assert params == ["ACC-1", 100.0, "approved"]
    assert next_index == 4
    assert "ACC-1" not in clause


@pytest.mark.security
@pytest.mark.parametrize(
    "unsafe_filter",
    [
        SqlFilter("account_id; DROP TABLE transactions", "=", "ACC-1"),
        SqlFilter("account_id", "LIKE", "%ACC%"),
        SqlFilter("password_hash", "=", "secret"),
    ],
)
def test_where_clause_rejects_unsafe_filter_shapes(unsafe_filter: SqlFilter) -> None:
    with pytest.raises(UnsafeQueryError):
        build_where_clause([unsafe_filter])
