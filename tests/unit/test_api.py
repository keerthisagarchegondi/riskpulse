"""Comprehensive tests for the RiskPulse REST API.

Tests cover:
- Transaction submission (single and batch)
- Transaction retrieval and listing
- Authentication (valid/invalid/missing API keys)
- Rate limiting (token bucket enforcement)
- Input validation (Pydantic schema enforcement)
- Health check endpoints
- Error handling
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.middleware.auth import reset_key_manager
from src.api.middleware.rate_limiter import InMemoryRateLimiter


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _reset_auth():
    """Reset the API key manager before each test."""
    reset_key_manager()
    yield
    reset_key_manager()


@pytest.fixture
def app():
    """Create a fresh FastAPI application for testing."""
    with patch("src.api.app.configure_logging"):
        application = create_app()
    return application


@pytest.fixture
def client(app):
    """Create a test client with authentication."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Valid authentication headers for testing."""
    return {"X-API-Key": "dev-api-key-riskpulse-2024"}


@pytest.fixture
def valid_transaction():
    """A valid transaction payload."""
    return {
        "external_transaction_id": "TXN-TEST-001",
        "account_id": "ACC-12345",
        "customer_id": "CUST-67890",
        "merchant_id": "MERCH-11111",
        "merchant_name": "Test Merchant",
        "merchant_category_code": "5411",
        "transaction_amount": "125.50",
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": "credit",
        "card_last_four": "4242",
        "ip_address": "192.168.1.100",
        "device_id": "device-abc-123",
        "device_type": "mobile",
        "geo_latitude": "40.7128",
        "geo_longitude": "-74.0060",
        "geo_country": "USA",
        "geo_city": "New York",
        "is_international": False,
        "transaction_timestamp": "2026-06-15T10:30:00Z",
    }


@pytest.fixture
def minimal_transaction():
    """A transaction with only required fields."""
    return {
        "external_transaction_id": "TXN-MINIMAL-001",
        "account_id": "ACC-001",
        "customer_id": "CUST-001",
        "transaction_amount": "50.00",
        "transaction_type": "purchase",
        "channel": "pos",
        "transaction_timestamp": "2026-06-15T12:00:00Z",
    }


# --- Authentication Tests ---


class TestAuthentication:
    """Test API key authentication middleware."""

    def test_missing_api_key_returns_401(self, client):
        """Request without API key should be rejected."""
        response = client.post("/api/v1/transactions", json={})
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Missing API key" in response.json()["detail"]

    def test_invalid_api_key_returns_401(self, client):
        """Request with invalid API key should be rejected."""
        response = client.post(
            "/api/v1/transactions",
            json={},
            headers={"X-API-Key": "invalid-key-12345"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid API key" in response.json()["detail"]

    def test_valid_api_key_passes_auth(self, client, auth_headers, valid_transaction):
        """Request with valid API key should pass authentication."""
        with patch("src.api.routes.transactions._get_kafka_producer", return_value=None):
            response = client.post(
                "/api/v1/transactions",
                json=valid_transaction,
                headers=auth_headers,
            )
        # Should not be 401 — either 202 or another valid status
        assert response.status_code != status.HTTP_401_UNAUTHORIZED

    def test_health_endpoint_no_auth_required(self, client):
        """Health endpoints should not require authentication."""
        response = client.get("/health/live")
        assert response.status_code == status.HTTP_200_OK

    def test_api_key_in_wrong_header_returns_401(self, client):
        """API key in wrong header should be rejected."""
        response = client.post(
            "/api/v1/transactions",
            json={},
            headers={"Authorization": "Bearer dev-api-key-riskpulse-2024"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


# --- Transaction Submission Tests ---


class TestTransactionSubmission:
    """Test single transaction submission endpoint."""

    @patch("src.api.routes.transactions._get_kafka_producer")
    def test_submit_valid_transaction(self, mock_producer, client, auth_headers, valid_transaction):
        """Valid transaction should be accepted (202)."""
        mock_prod_instance = MagicMock()
        mock_producer.return_value = mock_prod_instance

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert data["status"] == "accepted"
        assert data["external_transaction_id"] == "TXN-TEST-001"
        assert "transaction_id" in data
        # Verify UUID format
        uuid.UUID(data["transaction_id"])

    @patch("src.api.routes.transactions._get_kafka_producer")
    def test_submit_minimal_transaction(self, mock_producer, client, auth_headers, minimal_transaction):
        """Transaction with only required fields should be accepted."""
        mock_producer.return_value = MagicMock()

        response = client.post(
            "/api/v1/transactions",
            json=minimal_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_202_ACCEPTED

    @patch("src.api.routes.transactions._get_kafka_producer")
    def test_submit_publishes_to_kafka(self, mock_producer, client, auth_headers, valid_transaction):
        """Submitted transaction should be published to Kafka."""
        mock_prod_instance = MagicMock()
        mock_producer.return_value = mock_prod_instance

        client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        mock_prod_instance.produce.assert_called_once()
        call_args = mock_prod_instance.produce.call_args
        event = call_args[0][0]
        assert event["account_id"] == "ACC-12345"
        assert event["transaction_amount"] == 125.50
        assert call_args[1]["topic"] == "txn.raw.events"

    @patch("src.api.routes.transactions._get_kafka_producer")
    def test_kafka_failure_returns_503(self, mock_producer, client, auth_headers, valid_transaction):
        """Kafka publish failure should return 503."""
        mock_prod_instance = MagicMock()
        mock_prod_instance.produce.side_effect = Exception("Kafka broker unavailable")
        mock_producer.return_value = mock_prod_instance

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "retry" in response.json()["detail"].lower()

    @patch("src.api.routes.transactions._get_kafka_producer")
    def test_kafka_unavailable_graceful(self, mock_producer, client, auth_headers, valid_transaction):
        """When Kafka producer returns None, should still log warning and succeed."""
        mock_producer.return_value = None

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        # Graceful degradation: accepted but won't be processed immediately
        assert response.status_code == status.HTTP_202_ACCEPTED


# --- Batch Submission Tests ---


class TestBatchSubmission:
    """Test batch transaction submission endpoint."""

    @patch("src.api.routes.transactions._get_kafka_producer")
    def test_submit_batch_valid(self, mock_producer, client, auth_headers, valid_transaction):
        """Batch of valid transactions should be accepted."""
        mock_prod_instance = MagicMock()
        mock_prod_instance.produce_batch.return_value = []
        mock_producer.return_value = mock_prod_instance

        batch = {"transactions": [valid_transaction, valid_transaction]}
        # Need unique external IDs
        batch["transactions"][1] = {**valid_transaction, "external_transaction_id": "TXN-TEST-002"}

        response = client.post(
            "/api/v1/transactions/batch",
            json=batch,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert data["accepted"] == 2
        assert data["rejected"] == 0
        assert len(data["transactions"]) == 2

    @patch("src.api.routes.transactions._get_kafka_producer")
    def test_submit_batch_max_size(self, mock_producer, client, auth_headers, minimal_transaction):
        """Batch at max size (1000) should be accepted."""
        mock_prod_instance = MagicMock()
        mock_prod_instance.produce_batch.return_value = []
        mock_producer.return_value = mock_prod_instance

        transactions = []
        for i in range(1000):
            txn = {**minimal_transaction, "external_transaction_id": f"TXN-BATCH-{i:04d}"}
            transactions.append(txn)

        response = client.post(
            "/api/v1/transactions/batch",
            json={"transactions": transactions},
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        assert response.json()["accepted"] == 1000

    def test_submit_batch_exceeds_max_size(self, client, auth_headers, minimal_transaction):
        """Batch exceeding 1000 should return 422."""
        transactions = []
        for i in range(1001):
            txn = {**minimal_transaction, "external_transaction_id": f"TXN-OVER-{i:04d}"}
            transactions.append(txn)

        response = client.post(
            "/api/v1/transactions/batch",
            json={"transactions": transactions},
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_submit_empty_batch(self, client, auth_headers):
        """Empty batch should return 422."""
        response = client.post(
            "/api/v1/transactions/batch",
            json={"transactions": []},
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# --- Validation Tests ---


class TestInputValidation:
    """Test Pydantic schema validation for transactions."""

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_missing_required_field(self, _mock, client, auth_headers):
        """Missing required fields should return 422."""
        response = client.post(
            "/api/v1/transactions",
            json={"external_transaction_id": "TXN-001"},
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert data["error"] == "Validation Error"
        assert len(data["detail"]) > 0

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_invalid_transaction_type(self, _mock, client, auth_headers, valid_transaction):
        """Invalid transaction type should return 422."""
        valid_transaction["transaction_type"] = "invalid_type"

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_invalid_channel(self, _mock, client, auth_headers, valid_transaction):
        """Invalid channel should return 422."""
        valid_transaction["channel"] = "carrier_pigeon"

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_negative_amount(self, _mock, client, auth_headers, valid_transaction):
        """Negative transaction amount should return 422."""
        valid_transaction["transaction_amount"] = "-100.00"

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_zero_amount(self, _mock, client, auth_headers, valid_transaction):
        """Zero transaction amount should return 422."""
        valid_transaction["transaction_amount"] = "0.00"

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_invalid_currency(self, _mock, client, auth_headers, valid_transaction):
        """Unsupported currency should return 422."""
        valid_transaction["transaction_currency"] = "XYZ"

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_invalid_ip_address(self, _mock, client, auth_headers, valid_transaction):
        """Invalid IP address should return 422."""
        valid_transaction["ip_address"] = "999.999.999.999"

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_invalid_card_last_four(self, _mock, client, auth_headers, valid_transaction):
        """Card last four must be exactly 4 digits."""
        valid_transaction["card_last_four"] = "12"

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_card_last_four_non_numeric(self, _mock, client, auth_headers, valid_transaction):
        """Card last four must be numeric."""
        valid_transaction["card_last_four"] = "abcd"

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_latitude_out_of_range(self, _mock, client, auth_headers, valid_transaction):
        """Latitude must be between -90 and 90."""
        valid_transaction["geo_latitude"] = "91.0"
        valid_transaction["geo_longitude"] = "0.0"

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_latitude_without_longitude(self, _mock, client, auth_headers, valid_transaction):
        """Latitude without longitude should return 422."""
        valid_transaction["geo_latitude"] = "40.0"
        valid_transaction.pop("geo_longitude", None)

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_invalid_timestamp(self, _mock, client, auth_headers, valid_transaction):
        """Invalid timestamp format should return 422."""
        valid_transaction["transaction_timestamp"] = "not-a-date"

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_empty_external_id(self, _mock, client, auth_headers, valid_transaction):
        """Empty external_transaction_id should return 422."""
        valid_transaction["external_transaction_id"] = ""

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_valid_card_types(self, _mock, client, auth_headers, valid_transaction):
        """All valid card types should be accepted."""
        for card_type in ("credit", "debit", "prepaid"):
            valid_transaction["card_type"] = card_type
            response = client.post(
                "/api/v1/transactions",
                json=valid_transaction,
                headers=auth_headers,
            )
            assert response.status_code == status.HTTP_202_ACCEPTED


# --- Transaction Retrieval Tests ---


class TestTransactionRetrieval:
    """Test transaction GET endpoints."""

    @patch("src.api.routes.transactions._get_storage")
    def test_get_transaction_found(self, mock_storage, client, auth_headers):
        """Existing transaction should be returned."""
        txn_id = uuid.uuid4()
        mock_store = MagicMock()
        mock_store.get_transaction = AsyncMock(
            return_value={
                "transaction_id": txn_id,
                "external_transaction_id": "TXN-001",
                "account_id": "ACC-001",
                "customer_id": "CUST-001",
                "transaction_amount": Decimal("100.00"),
                "transaction_currency": "USD",
                "transaction_type": "purchase",
                "channel": "online",
                "is_international": False,
                "transaction_timestamp": datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
                "status": "pending",
                "created_at": datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
            }
        )
        mock_storage.return_value = mock_store

        response = client.get(
            f"/api/v1/transactions/{txn_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["transaction_id"] == str(txn_id)

    @patch("src.api.routes.transactions._get_storage")
    def test_get_transaction_not_found(self, mock_storage, client, auth_headers):
        """Non-existent transaction should return 404."""
        mock_store = MagicMock()
        mock_store.get_transaction = AsyncMock(return_value=None)
        mock_storage.return_value = mock_store

        txn_id = uuid.uuid4()
        response = client.get(
            f"/api/v1/transactions/{txn_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    def test_get_transaction_invalid_uuid(self, client, auth_headers):
        """Invalid UUID format should return 422."""
        response = client.get(
            "/api/v1/transactions/not-a-uuid",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.api.routes.transactions._get_storage")
    def test_list_transactions(self, mock_storage, client, auth_headers):
        """List endpoint should return paginated results."""
        mock_store = MagicMock()
        mock_store.list_transactions = AsyncMock(return_value=([], 0))
        mock_storage.return_value = mock_store

        response = client.get(
            "/api/v1/transactions",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "pages" in data

    @patch("src.api.routes.transactions._get_storage")
    def test_list_transactions_with_filters(self, mock_storage, client, auth_headers):
        """List with filter parameters should pass them correctly."""
        mock_store = MagicMock()
        mock_store.list_transactions = AsyncMock(return_value=([], 0))
        mock_storage.return_value = mock_store

        response = client.get(
            "/api/v1/transactions?account_id=ACC-001&status=pending&page=2&page_size=10",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK

    def test_list_transactions_invalid_page_size(self, client, auth_headers):
        """Page size exceeding max should return 422."""
        response = client.get(
            "/api/v1/transactions?page_size=500",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_list_transactions_page_zero(self, client, auth_headers):
        """Page number 0 should return 422."""
        response = client.get(
            "/api/v1/transactions?page=0",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# --- Rate Limiting Tests ---


class TestRateLimiting:
    """Test token bucket rate limiting."""

    def test_rate_limit_headers_present(self, client, auth_headers, valid_transaction):
        """Rate limit headers should be in responses."""
        with patch("src.api.routes.transactions._get_kafka_producer", return_value=None):
            response = client.post(
                "/api/v1/transactions",
                json=valid_transaction,
                headers=auth_headers,
            )

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

    def test_rate_limit_not_applied_to_health(self, client):
        """Health endpoints should be exempt from rate limiting."""
        # Make many requests to health — should never get 429
        for _ in range(150):
            response = client.get("/health/live")
            assert response.status_code == status.HTTP_200_OK

    def test_token_bucket_allows_burst(self):
        """Token bucket should allow burst within capacity."""
        limiter = InMemoryRateLimiter(default_rate=10, burst_size=5)

        # Should allow burst
        for _ in range(10):
            allowed, _, _ = limiter.is_allowed("test-key")
            assert allowed is True

    def test_token_bucket_blocks_after_exhaustion(self):
        """Token bucket should block after tokens exhausted."""
        limiter = InMemoryRateLimiter(default_rate=5, burst_size=0)

        # Exhaust tokens
        for _ in range(5):
            limiter.is_allowed("test-key")

        # Next should be blocked
        allowed, remaining, retry_after = limiter.is_allowed("test-key")
        assert allowed is False
        assert remaining == 0
        assert retry_after > 0

    def test_token_bucket_different_keys_independent(self):
        """Different API keys should have independent rate limits."""
        limiter = InMemoryRateLimiter(default_rate=2, burst_size=0)

        # Exhaust key1
        limiter.is_allowed("key1")
        limiter.is_allowed("key1")
        allowed_key1, _, _ = limiter.is_allowed("key1")

        # key2 should still be allowed
        allowed_key2, _, _ = limiter.is_allowed("key2")

        assert allowed_key1 is False
        assert allowed_key2 is True


# --- Health Check Tests ---


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_liveness_probe(self, client):
        """Liveness probe should always return 200."""
        response = client.get("/health/live")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "alive"

    def test_readiness_probe(self, client):
        """Readiness probe should return service ready status."""
        with patch("src.api.routes.health._check_kafka") as mock_kafka:
            mock_kafka.return_value = MagicMock(status="healthy")
            response = client.get("/health/ready")

        assert response.status_code == status.HTTP_200_OK

    def test_full_health_check(self, client):
        """Full health check should return comprehensive status."""
        from src.api.routes.health import DependencyStatus

        with (
            patch("src.api.routes.health._check_kafka") as mock_kafka,
            patch("src.api.routes.health._check_postgres") as mock_pg,
            patch("src.api.routes.health._check_redis") as mock_redis,
        ):
            mock_kafka.return_value = DependencyStatus(name="kafka", status="healthy", latency_ms=5.0)
            mock_pg.return_value = DependencyStatus(name="postgresql", status="healthy", latency_ms=3.0)
            mock_redis.return_value = DependencyStatus(name="redis", status="healthy", latency_ms=1.0)

            response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "RiskPulse"
        assert "version" in data
        assert "uptime_seconds" in data
        assert "environment" in data
        assert "timestamp" in data

    def test_health_degraded_when_kafka_down(self, client):
        """Health should be degraded when Kafka is unhealthy."""
        from src.api.routes.health import DependencyStatus

        with (
            patch("src.api.routes.health._check_kafka") as mock_kafka,
            patch("src.api.routes.health._check_postgres") as mock_pg,
            patch("src.api.routes.health._check_redis") as mock_redis,
        ):
            mock_kafka.return_value = DependencyStatus(name="kafka", status="unhealthy", latency_ms=5000.0, detail="timeout")
            mock_pg.return_value = DependencyStatus(name="postgresql", status="healthy", latency_ms=3.0)
            mock_redis.return_value = DependencyStatus(name="redis", status="healthy", latency_ms=1.0)

            response = client.get("/health")

        data = response.json()
        assert data["status"] == "degraded"


# --- Correlation ID Tests ---


class TestCorrelationID:
    """Test correlation ID propagation."""

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_correlation_id_generated(self, _mock, client, auth_headers, valid_transaction):
        """Correlation ID should be generated and returned in response."""
        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=auth_headers,
        )

        assert "X-Correlation-ID" in response.headers
        # Validate UUID format
        uuid.UUID(response.headers["X-Correlation-ID"])

    @patch("src.api.routes.transactions._get_kafka_producer", return_value=None)
    def test_correlation_id_propagated(self, _mock, client, auth_headers, valid_transaction):
        """Provided correlation ID should be propagated."""
        custom_id = str(uuid.uuid4())
        headers = {**auth_headers, "X-Correlation-ID": custom_id}

        response = client.post(
            "/api/v1/transactions",
            json=valid_transaction,
            headers=headers,
        )

        assert response.headers["X-Correlation-ID"] == custom_id

    def test_request_duration_header(self, client):
        """Request duration should be returned in header."""
        response = client.get("/health/live")
        assert "X-Request-Duration-Ms" in response.headers
        duration = float(response.headers["X-Request-Duration-Ms"])
        assert duration >= 0


# --- Root Endpoint Tests ---


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_returns_service_info(self, client):
        """Root endpoint should return service information."""
        response = client.get("/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["service"] == "RiskPulse"
        assert "version" in data
        assert data["docs"] == "/docs"


# --- OpenAPI Documentation Tests ---


class TestDocumentation:
    """Test that API documentation is accessible."""

    def test_openapi_schema_accessible(self, client):
        """OpenAPI schema should be generated and accessible."""
        response = client.get("/openapi.json")

        assert response.status_code == status.HTTP_200_OK
        schema = response.json()
        assert schema["info"]["title"] == "RiskPulse API"
        assert "/api/v1/transactions" in schema["paths"]

    def test_swagger_docs_accessible(self, client):
        """Swagger UI should be accessible."""
        response = client.get("/docs")
        assert response.status_code == status.HTTP_200_OK

    def test_redoc_accessible(self, client):
        """ReDoc should be accessible."""
        response = client.get("/redoc")
        assert response.status_code == status.HTTP_200_OK
