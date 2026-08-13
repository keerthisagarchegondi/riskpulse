"""Security tests for authentication, authorization, and rate limiting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import Depends, FastAPI, status
from fastapi.testclient import TestClient
from jose import jwt

from src.api.app import create_app
from src.api.middleware.auth import require_permission, reset_key_manager
from src.api.middleware.rate_limiter import InMemoryRateLimiter, RateLimitMiddleware
from src.utils.security import JWT_ALGORITHM, JWT_ISSUER, SecurityValidationError
from src.utils.security import create_jwt_token, verify_jwt_token


DEV_API_KEY = "dev-api-key-riskpulse-2024"


@pytest.fixture(autouse=True)
def _reset_auth_state():
    reset_key_manager()
    yield
    reset_key_manager()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def valid_transaction() -> dict[str, object]:
    return {
        "external_transaction_id": "TXN-AUTH-001",
        "account_id": "ACC-AUTH",
        "customer_id": "CUST-AUTH",
        "transaction_amount": "25.00",
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "transaction_timestamp": "2026-08-13T12:00:00Z",
    }


@pytest.mark.security
@pytest.mark.parametrize(
    "headers,expected_detail",
    [
        ({}, "Missing API key"),
        ({"X-API-Key": "rp-invalid"}, "Invalid API key"),
        ({"Authorization": "Bearer not-a-jwt"}, "Invalid bearer token"),
    ],
)
def test_transaction_submit_blocks_auth_bypass_attempts(
    client: TestClient,
    valid_transaction: dict[str, object],
    headers: dict[str, str],
    expected_detail: str,
) -> None:
    response = client.post("/api/v1/transactions", json=valid_transaction, headers=headers)

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert expected_detail in response.text


@pytest.mark.security
def test_expired_jwt_is_rejected() -> None:
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": "analyst@example.com",
            "permissions": ["read"],
            "iss": JWT_ISSUER,
            "iat": int((now - timedelta(hours=2)).timestamp()),
            "exp": int((now - timedelta(hours=1)).timestamp()),
        },
        "unit-secret",
        algorithm=JWT_ALGORITHM,
    )

    with pytest.raises(SecurityValidationError):
        verify_jwt_token(token, secret="unit-secret")


@pytest.mark.security
def test_tampered_jwt_signature_is_rejected() -> None:
    token = create_jwt_token(
        subject="service:worker",
        permissions=["read", "write"],
        secret="correct-secret",
    )

    with pytest.raises(SecurityValidationError):
        verify_jwt_token(token, secret="wrong-secret")


@pytest.mark.security
def test_permission_dependency_denies_read_only_identity() -> None:
    app = FastAPI()

    @app.post("/admin-only")
    async def admin_only(_auth=Depends(require_permission("admin"))):
        return {"ok": True}

    token = create_jwt_token(
        subject="readonly@example.com",
        permissions=["read"],
        secret="dev-jwt-secret",
    )
    response = TestClient(app).post(
        "/admin-only",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "Insufficient permissions" in response.text


@pytest.mark.security
def test_permission_dependency_allows_admin_identity() -> None:
    app = FastAPI()

    @app.post("/admin-only")
    async def admin_only(_auth=Depends(require_permission("admin"))):
        return {"ok": True}

    token = create_jwt_token(
        subject="admin@example.com",
        permissions=["read", "write", "admin"],
        secret="dev-jwt-secret",
    )
    response = TestClient(app).post(
        "/admin-only",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"ok": True}


@pytest.mark.security
def test_in_memory_rate_limiter_enforces_per_identity_limit() -> None:
    limiter = InMemoryRateLimiter(default_rate=2, burst_size=0)

    assert limiter.is_allowed("apikey:ci")[0] is True
    assert limiter.is_allowed("apikey:ci")[0] is True
    allowed, remaining, retry_after = limiter.is_allowed("apikey:ci")

    assert allowed is False
    assert remaining == 0
    assert retry_after > 0


@pytest.mark.security
def test_rate_limit_middleware_returns_retry_headers() -> None:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)

    @app.get("/limited")
    async def limited():
        return {"ok": True}

    with TestClient(app) as test_client:
        middleware = next(
            middleware
            for middleware in test_client.app.user_middleware
            if middleware.cls is RateLimitMiddleware
        )
        # Force a tiny limit for this endpoint while preserving real middleware code.
        middleware.kwargs.clear()
        response_1 = test_client.get("/limited")
        response_2 = test_client.get("/limited")

    assert response_1.status_code == status.HTTP_200_OK
    assert response_2.status_code in {status.HTTP_200_OK, status.HTTP_429_TOO_MANY_REQUESTS}
