"""OpenAPI schema customization for RiskPulse API documentation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from src.utils.constants import APP_NAME, APP_VERSION, DEFAULT_RATE_LIMIT


AUTHENTICATION_DESCRIPTION = (
    "RiskPulse uses API key authentication. Send the key in the `X-API-Key` header. "
    "Development and test environments include `dev-api-key-riskpulse-2024` by default. "
    "Production keys should be provisioned through the configured secret store and scoped with "
    "`read`, `write`, or `admin` permissions."
)

RATE_LIMIT_DESCRIPTION = (
    f"Default limit is {DEFAULT_RATE_LIMIT} requests per minute per API key or source IP. "
    "Responses include `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `Retry-After` "
    "headers when applicable."
)

ERROR_CODES: dict[str, str] = {
    "400": "Malformed request or invalid operation state.",
    "401": "Missing or invalid `X-API-Key` header.",
    "403": "Authenticated key lacks required permission.",
    "404": "Requested entity was not found.",
    "409": "Requested write conflicts with an existing resource.",
    "422": "Request payload or query parameter validation failed.",
    "429": "Rate limit exceeded. Retry after the `Retry-After` header value.",
    "500": "Unexpected server error with a correlation request id when available.",
    "503": "Dependency unavailable, such as Kafka, PostgreSQL, or model serving.",
}

TRANSACTION_EXAMPLE: dict[str, Any] = {
    "external_transaction_id": "EXT-20260806-0001",
    "account_id": "ACC-12345",
    "customer_id": "CUST-67890",
    "merchant_id": "MERCH-11111",
    "merchant_name": "Example Electronics",
    "merchant_category_code": "5732",
    "transaction_amount": "249.99",
    "transaction_currency": "USD",
    "transaction_type": "purchase",
    "channel": "online",
    "card_type": "credit",
    "card_last_four": "4242",
    "ip_address": "203.0.113.10",
    "device_id": "device-web-123",
    "device_type": "desktop",
    "geo_latitude": "40.7128",
    "geo_longitude": "-74.0060",
    "geo_country": "USA",
    "geo_city": "New York",
    "is_international": False,
    "transaction_timestamp": "2026-08-06T14:30:00Z",
    "metadata": {"checkout_session_id": "sess_abc123"},
}

SCORE_EXAMPLE: dict[str, Any] = {
    "transaction_id": "EXT-20260806-0001",
    "customer_id": "CUST-67890",
    "transaction_amount": 249.99,
    "transaction_currency": "USD",
    "transaction_type": "purchase",
    "channel": "online",
    "merchant_id": "MERCH-11111",
    "merchant_name": "Example Electronics",
    "merchant_category_code": "5732",
    "geo_country": "USA",
    "is_international": False,
    "features": {
        "amount_zscore": 0.7,
        "velocity_1h": 1.0,
        "merchant_risk": 0.2,
    },
    "use_cache": True,
}


def install_custom_openapi(app: FastAPI) -> None:
    """Install a custom OpenAPI builder on a FastAPI app."""

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=f"{APP_NAME} API",
            version=APP_VERSION,
            description=_build_description(),
            routes=app.routes,
            tags=[
                {"name": "Health", "description": "Service liveness, readiness, and dependency status."},
                {"name": "Transactions", "description": "Transaction ingestion and transaction lookup APIs."},
                {"name": "Scoring", "description": "Unified rule, anomaly, and ML fraud scoring APIs."},
                {"name": "Risk Scores", "description": "Model serving, model health, and monitoring APIs."},
                {"name": "Rules Engine", "description": "Rule management, rule evaluation, and rule audit APIs."},
            ],
        )

        schema.setdefault("components", {}).setdefault("securitySchemes", {})["ApiKeyAuth"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": AUTHENTICATION_DESCRIPTION,
        }
        schema["security"] = [{"ApiKeyAuth": []}]
        schema["x-riskpulse-rate-limits"] = {
            "defaultRequestsPerMinute": DEFAULT_RATE_LIMIT,
            "headers": ["X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
        }
        schema["x-riskpulse-error-codes"] = deepcopy(ERROR_CODES)

        _apply_common_responses(schema)
        _apply_examples(schema)

        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]


def _build_description() -> str:
    """Return rich API documentation used by Swagger UI."""
    error_lines = "\n".join(f"- `{code}`: {description}" for code, description in ERROR_CODES.items())
    return (
        "Fraud Analytics & Risk Intelligence Platform API.\n\n"
        "## Authentication\n"
        f"{AUTHENTICATION_DESCRIPTION}\n\n"
        "## Rate Limits\n"
        f"{RATE_LIMIT_DESCRIPTION}\n\n"
        "## Error Codes\n"
        f"{error_lines}\n\n"
        "## Correlation\n"
        "Every request is logged with a correlation id when available. Error responses include "
        "`request_id` for support and incident triage."
    )


def _apply_common_responses(schema: dict[str, Any]) -> None:
    """Add shared response documentation to protected operations."""
    paths = schema.get("paths", {})
    common_responses = {
        "401": {"description": ERROR_CODES["401"]},
        "429": {
            "description": ERROR_CODES["429"],
            "headers": {
                "Retry-After": {
                    "description": "Seconds to wait before retrying.",
                    "schema": {"type": "integer"},
                },
                "X-RateLimit-Limit": {
                    "description": "Maximum requests allowed in the current window.",
                    "schema": {"type": "integer"},
                },
                "X-RateLimit-Remaining": {
                    "description": "Approximate remaining requests in the current window.",
                    "schema": {"type": "integer"},
                },
            },
        },
        "500": {"description": ERROR_CODES["500"]},
        "503": {"description": ERROR_CODES["503"]},
    }

    for path, methods in paths.items():
        for method, operation in methods.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            operation.setdefault("responses", {})
            if path not in {"/health", "/health/live", "/health/ready"}:
                operation.setdefault("security", [{"ApiKeyAuth": []}])
                for status_code, response_doc in common_responses.items():
                    operation["responses"].setdefault(status_code, response_doc)


def _apply_examples(schema: dict[str, Any]) -> None:
    """Inject representative request examples into the OpenAPI schema."""
    _set_json_example(
        schema,
        "/api/v1/transactions",
        "post",
        "singleTransaction",
        "Single online purchase",
        TRANSACTION_EXAMPLE,
    )
    _set_json_example(
        schema,
        "/api/v1/transactions/batch",
        "post",
        "batchTransactions",
        "Two transaction batch",
        {"transactions": [TRANSACTION_EXAMPLE, {**TRANSACTION_EXAMPLE, "external_transaction_id": "EXT-20260806-0002"}]},
    )
    _set_json_example(
        schema,
        "/api/v1/score",
        "post",
        "scoreTransaction",
        "Score a transaction with feature hints",
        SCORE_EXAMPLE,
    )
    _set_json_example(
        schema,
        "/api/v1/score/batch",
        "post",
        "scoreBatch",
        "Batch scoring request",
        {"transactions": [SCORE_EXAMPLE]},
    )
    _set_json_example(
        schema,
        "/api/v1/risk-scores/predict",
        "post",
        "riskScorePredict",
        "Model serving request",
        {
            "transaction_id": "EXT-20260806-0001",
            "customer_id": "CUST-67890",
            "transaction_amount": 249.99,
            "transaction_currency": "USD",
            "transaction_type": "purchase",
            "channel": "online",
            "features": {"amount_zscore": 0.7, "velocity_1h": 1.0},
        },
    )
    _set_json_example(
        schema,
        "/api/v1/rules/evaluate",
        "post",
        "ruleEvaluation",
        "Evaluate rules before deployment",
        {"transaction": TRANSACTION_EXAMPLE},
    )


def _set_json_example(
    schema: dict[str, Any],
    path: str,
    method: str,
    example_name: str,
    summary: str,
    value: dict[str, Any],
) -> None:
    operation = schema.get("paths", {}).get(path, {}).get(method)
    if not operation:
        return
    content = (
        operation.setdefault("requestBody", {})
        .setdefault("content", {})
        .setdefault("application/json", {})
    )
    content.setdefault("examples", {})[example_name] = {
        "summary": summary,
        "value": value,
    }
