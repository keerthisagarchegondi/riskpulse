"""Regression tests for the deployment smoke gates."""

from __future__ import annotations

import base64
from io import BytesIO
from urllib.error import HTTPError

import pytest

from scripts import smoke_test


def test_expected_auth_rejection_is_a_passing_check(monkeypatch: pytest.MonkeyPatch) -> None:
    def unauthorized(*args, **kwargs):
        raise HTTPError("https://api.example.invalid", 401, "Unauthorized", {}, BytesIO(b"{}"))

    monkeypatch.setattr(smoke_test, "_request_json", unauthorized)
    result = smoke_test._run_check(
        "api_rejects_unauthenticated_read",
        "https://api.example.invalid/api/v1/transactions",
        expected_status={401},
        timeout=1,
    )
    assert result.passed
    assert result.response_status == 401


def test_streamlit_basic_auth_checks_both_denial_and_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RISKPULSE_INGRESS_BASIC_PASSWORD", "test-password")
    args = smoke_test.parse_args(
        [
            "--streamlit-url",
            "https://dashboard.example.invalid",
            "--streamlit-basic-user",
            "auditor",
            "--verify-auth",
        ]
    )
    calls = []

    def check(name, url, **kwargs):
        calls.append((name, kwargs))
        return smoke_test.CheckResult(name=name, status="pass")

    monkeypatch.setattr(smoke_test, "_run_check", check)
    smoke_test._build_optional_service_checks(args)
    denied, allowed = calls[:2]
    assert denied[0] == "streamlit_rejects_unauthenticated_request"
    assert denied[1]["expected_status"] == {401}
    assert "headers" not in denied[1]
    assert allowed[0] == "streamlit_health"
    expected = base64.b64encode(b"auditor:test-password").decode("ascii")
    assert allowed[1]["headers"] == {"Authorization": f"Basic {expected}"}


@pytest.mark.parametrize("url", ["http://dashboard.example.invalid", "http://192.0.2.10"])
def test_streamlit_basic_auth_rejects_cleartext_remote_url(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setenv("RISKPULSE_INGRESS_BASIC_PASSWORD", "test-password")
    with pytest.raises(SystemExit):
        smoke_test.parse_args(["--streamlit-url", url, "--streamlit-basic-user", "auditor"])


def test_processed_transaction_must_match_id_and_final_status() -> None:
    transaction_id = "85b33a25-97aa-47a2-9ac6-89be38495a9c"
    assert smoke_test._validate_processed_transaction(
        {"transaction_id": transaction_id, "status": "pending"}, transaction_id
    )
    assert smoke_test._validate_processed_transaction(
        {"transaction_id": "other", "status": "flagged"}, transaction_id
    )
    assert not smoke_test._validate_processed_transaction(
        {"transaction_id": transaction_id, "status": "flagged"}, transaction_id
    )


def test_processed_transaction_check_retries_until_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    transaction_id = "85b33a25-97aa-47a2-9ac6-89be38495a9c"
    attempts = iter(["fail", "pass"])
    monkeypatch.setattr(smoke_test.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        smoke_test,
        "_run_check",
        lambda name, url, **kwargs: smoke_test.CheckResult(name=name, status=next(attempts)),
    )
    args = smoke_test.parse_args(["--processing-timeout", "1"])
    result = smoke_test._check_processed_transaction(
        "https://api.example.invalid/", transaction_id, {"X-API-Key": "test"}, args
    )
    assert result.passed


def test_dependency_health_requires_all_local_backends() -> None:
    dependencies = [
        {"name": "kafka", "status": "healthy"},
        {"name": "postgresql", "status": "healthy"},
        {"name": "redis", "status": "unhealthy"},
    ]
    assert "redis" in smoke_test._validate_dependencies({"dependencies": dependencies})
    dependencies[2]["status"] = "healthy"
    assert smoke_test._validate_dependencies({"dependencies": dependencies}) == ""


def test_api_auth_checks_both_denial_and_authorized_read(monkeypatch: pytest.MonkeyPatch) -> None:
    args = smoke_test.parse_args(["--verify-auth", "--api-key", "test-key"])
    calls = []

    def check(name, url, **kwargs):
        calls.append((name, kwargs))
        return smoke_test.CheckResult(name=name, status="pass")

    monkeypatch.setattr(smoke_test, "_run_check", check)
    smoke_test._build_api_checks(args)
    denied = next(kwargs for name, kwargs in calls if name == "api_rejects_unauthenticated_read")
    allowed = next(kwargs for name, kwargs in calls if name == "api_accepts_authenticated_read")
    assert "headers" not in denied
    assert allowed["headers"]["X-API-Key"] == "test-key"


@pytest.mark.parametrize(
    "options",
    [
        ["--submit-test-transaction", "--retries", "2"],
        ["--submit-test-transaction", "--monitor-seconds", "30"],
    ],
)
def test_synthetic_write_cannot_repeat_automatically(options: list[str]) -> None:
    with pytest.raises(SystemExit):
        smoke_test.parse_args(options)
