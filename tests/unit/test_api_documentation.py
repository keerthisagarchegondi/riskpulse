"""Tests for API documentation and OpenAPI production metadata."""

from __future__ import annotations

import json
from pathlib import Path

from src.api.app import create_app


def test_openapi_documents_api_key_authentication() -> None:
    schema = create_app().openapi()

    security_schemes = schema["components"]["securitySchemes"]
    assert security_schemes["ApiKeyAuth"]["name"] == "X-API-Key"
    assert security_schemes["ApiKeyAuth"]["in"] == "header"
    assert {"ApiKeyAuth": []} in schema["security"]


def test_openapi_includes_rate_limits_and_error_codes() -> None:
    schema = create_app().openapi()

    assert schema["x-riskpulse-rate-limits"]["defaultRequestsPerMinute"] == 100
    assert schema["x-riskpulse-error-codes"]["429"].startswith("Rate limit exceeded")
    transaction_post = schema["paths"]["/api/v1/transactions"]["post"]
    assert "429" in transaction_post["responses"]
    assert "401" in transaction_post["responses"]


def test_openapi_includes_request_examples() -> None:
    schema = create_app().openapi()

    examples = schema["paths"]["/api/v1/score"]["post"]["requestBody"]["content"][
        "application/json"
    ]["examples"]
    assert "scoreTransaction" in examples
    assert examples["scoreTransaction"]["value"]["transaction_id"] == "EXT-20260806-0001"


def test_generated_openapi_artifact_is_valid_json() -> None:
    artifact = Path("docs/openapi.json")
    schema = json.loads(artifact.read_text(encoding="utf-8"))

    assert schema["info"]["title"] == "RiskPulse API"
    assert "/docs" not in schema.get("paths", {})
    assert "/api/v1/rules/evaluate" in schema["paths"]
