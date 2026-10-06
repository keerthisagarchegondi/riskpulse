"""Security tests for authentication, authorization, and rate limiting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from time import time
from unittest.mock import AsyncMock, Mock

import jwt
import pytest
from fastapi import Depends, FastAPI, status
from fastapi.testclient import TestClient

from src.api.app import _cors_options, create_app
from src.api.middleware.auth import get_key_manager, require_permission, reset_key_manager
from src.api.middleware.rate_limiter import (
    InMemoryRateLimiter,
    RateLimitMiddleware,
    RedisRateLimiter,
)
from src.api.routes.scoring import get_scoring_pipeline
from src.utils.config import get_settings
from src.utils.security import (
    JWT_ALGORITHM,
    JWT_ISSUER,
    SecurityValidationError,
    create_jwt_token,
    verify_jwt_token,
)

DEV_API_KEY = "dev-api-key-riskpulse-2024"
UNIT_SECRET = "unit-secret-for-hs256-tests-32-bytes"
CORRECT_SECRET = "correct-secret-for-hs256-tests-32-bytes"
WRONG_SECRET = "wrong-secret-for-hs256-tests-32-bytes"
DEV_JWT_SECRET = "dev-jwt-secret-for-hs256-tests-32-bytes"


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
def test_read_only_key_cannot_submit_transactions_or_change_weights(
    client: TestClient, valid_transaction: dict[str, object]
) -> None:
    manager = get_key_manager()
    manager._keys[manager._hash_key("rp-read-only")] = {
        "name": "readonly",
        "permissions": ["read"],
        "rate_limit": None,
        "auth_type": "api_key",
    }
    pipeline = Mock()
    client.app.dependency_overrides[get_scoring_pipeline] = lambda: pipeline
    headers = {"X-API-Key": "rp-read-only"}
    weights = {"rule_score": 0.4, "anomaly_score": 0.3, "ml_score": 0.3}

    assert (
        client.post("/api/v1/transactions", json=valid_transaction, headers=headers).status_code
        == 403
    )
    assert (
        client.post(
            "/api/v1/transactions/batch",
            json={"transactions": [valid_transaction]},
            headers=headers,
        ).status_code
        == 403
    )
    assert client.put("/api/v1/score/weights", json=weights, headers=headers).status_code == 403
    pipeline.update_weights.assert_not_called()

    pipeline.weights = weights
    admin_response = client.put(
        "/api/v1/score/weights", json=weights, headers={"X-API-Key": DEV_API_KEY}
    )
    assert admin_response.status_code == 200
    pipeline.update_weights.assert_called_once_with(weights)


@pytest.mark.security
def test_api_routes_enforce_read_and_write_scopes(client: TestClient) -> None:
    manager = get_key_manager()
    for key, permissions in (("rp-no-scope", []), ("rp-read-only", ["read"])):
        manager._keys[manager._hash_key(key)] = {
            "name": key,
            "permissions": permissions,
            "rate_limit": None,
            "auth_type": "api_key",
        }

    client.app.dependency_overrides[get_scoring_pipeline] = Mock
    no_scope = {"X-API-Key": "rp-no-scope"}
    for path in (
        "/api/v1/transactions",
        "/api/v1/rules/status",
        "/api/v1/risk-scores/monitoring/alerts",
        "/api/v1/score/metrics/summary",
    ):
        assert client.get(path, headers=no_scope).status_code == 403

    read_only = {"X-API-Key": "rp-read-only"}
    assert client.get("/api/v1/risk-scores/monitoring/alerts", headers=read_only).status_code == 200
    score = {
        "transaction_id": "txn-1",
        "customer_id": "customer-1",
        "transaction_amount": 25,
        "transaction_type": "purchase",
        "channel": "online",
    }
    assert client.post("/api/v1/score", json=score, headers=read_only).status_code == 403
    assert (
        client.post(
            "/api/v1/score/batch", json={"transactions": [score]}, headers=read_only
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/v1/rules/evaluate", json={"transaction": score}, headers=read_only
        ).status_code
        == 403
    )
    for path in ("/api/v1/risk-scores/predict", "/api/v1/risk-scores/predict/batch"):
        assert client.post(path, json={}, headers=read_only).status_code == 403


@pytest.mark.security
def test_api_security_headers_cover_success_and_auth_failure(client: TestClient) -> None:
    for response in (
        client.get("/health/live"),
        client.post("/api/v1/transactions", json={}),
    ):
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Referrer-Policy"] == "no-referrer"


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
        UNIT_SECRET,
        algorithm=JWT_ALGORITHM,
    )

    with pytest.raises(SecurityValidationError):
        verify_jwt_token(token, secret=UNIT_SECRET)


@pytest.mark.security
def test_tampered_jwt_signature_is_rejected() -> None:
    token = create_jwt_token(
        subject="service:worker",
        permissions=["read", "write"],
        secret=CORRECT_SECRET,
    )

    with pytest.raises(SecurityValidationError):
        verify_jwt_token(token, secret=WRONG_SECRET)


@pytest.mark.security
def test_wildcard_cors_disables_credentials_in_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RISKPULSE_ENV", "dev")
    monkeypatch.setenv("RISKPULSE_API__CORS_ORIGINS", "*")
    get_settings.cache_clear()

    try:
        origins, allow_credentials = _cors_options()
    finally:
        get_settings.cache_clear()

    assert origins == ["*"]
    assert allow_credentials is False


@pytest.mark.security
def test_wildcard_cors_is_rejected_in_managed_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RISKPULSE_ENV", "prod")
    monkeypatch.setenv("RISKPULSE_API__CORS_ORIGINS", "*")
    get_settings.cache_clear()

    try:
        with pytest.raises(ValueError, match="Wildcard CORS"):
            _cors_options()
    finally:
        get_settings.cache_clear()


@pytest.mark.security
def test_permission_dependency_denies_read_only_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RISKPULSE_JWT_SECRET", DEV_JWT_SECRET)
    app = FastAPI()

    @app.post("/admin-only")
    async def admin_only(_auth=Depends(require_permission("admin"))):
        return {"ok": True}

    token = create_jwt_token(
        subject="readonly@example.com",
        permissions=["read"],
        secret=DEV_JWT_SECRET,
    )
    response = TestClient(app).post(
        "/admin-only",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "Insufficient permissions" in response.text


@pytest.mark.security
def test_permission_dependency_allows_admin_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RISKPULSE_JWT_SECRET", DEV_JWT_SECRET)
    app = FastAPI()

    @app.post("/admin-only")
    async def admin_only(_auth=Depends(require_permission("admin"))):
        return {"ok": True}

    token = create_jwt_token(
        subject="admin@example.com",
        permissions=["read", "write", "admin"],
        secret=DEV_JWT_SECRET,
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
def test_rate_limit_middleware_returns_retry_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RISKPULSE_API__RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setenv("RISKPULSE_API__RATE_LIMIT__BURST_SIZE", "0")
    get_settings.cache_clear()
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)

    @app.get("/limited")
    async def limited():
        return {"ok": True}

    try:
        with TestClient(app) as test_client:
            response_1 = test_client.get("/limited")
            response_2 = test_client.get("/limited")
    finally:
        get_settings.cache_clear()

    assert response_1.status_code == status.HTTP_200_OK
    assert response_2.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert int(response_2.headers["Retry-After"]) > 0
    assert response_2.headers["X-RateLimit-Remaining"] == "0"


@pytest.mark.security
def test_rate_limit_middleware_honors_api_key_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RISKPULSE_API__RATE_LIMIT__BURST_SIZE", "0")
    get_settings.cache_clear()
    manager = get_key_manager()
    manager._keys[manager._hash_key("rp-custom-limit")] = {
        "name": "custom-limit",
        "permissions": ["read"],
        "rate_limit": 1,
        "auth_type": "api_key",
    }
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)

    @app.get("/limited")
    async def limited():
        return {"ok": True}

    try:
        with TestClient(app) as test_client:
            first = test_client.get("/limited", headers={"X-API-Key": "rp-custom-limit"})
            second = test_client.get("/limited", headers={"X-API-Key": "rp-custom-limit"})
    finally:
        get_settings.cache_clear()

    assert first.status_code == status.HTTP_200_OK
    assert second.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert second.headers["X-RateLimit-Limit"] == "1"


@pytest.mark.security
async def test_redis_rate_limiter_reports_remaining_window() -> None:
    redis_client = Mock()
    pipeline = redis_client.pipeline.return_value
    pipeline.execute = AsyncMock(return_value=[0, 1, 1, True, [(b"old", time() - 10)]])
    redis_client.zrem = AsyncMock()

    allowed, remaining, retry_after = await RedisRateLimiter(
        redis_client, default_rate=1, window_seconds=60
    ).is_allowed("custom")

    assert (allowed, remaining) == (False, 0)
    assert 49 <= retry_after <= 51
    member = next(iter(pipeline.zadd.call_args.args[1]))
    redis_client.zrem.assert_awaited_once_with("ratelimit:custom", member)


@pytest.mark.security
def test_rate_limit_key_uses_hashed_api_key_header() -> None:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)

    @app.get("/limited")
    async def limited():
        return {"ok": True}

    with TestClient(app) as test_client:
        response = test_client.get("/limited", headers={"X-API-Key": "rp-secret"})

    assert response.status_code == status.HTTP_200_OK
