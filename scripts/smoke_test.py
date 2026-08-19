"""Production smoke tests for RiskPulse deployments.

This script intentionally uses only Python's standard library so it can run in
deployment jobs before project dependencies are installed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

DEFAULT_API_KEY = "dev-api-key-riskpulse-2024"


@dataclass
class CheckResult:
    name: str
    status: str
    url: str | None = None
    detail: str = ""
    latency_ms: float = 0.0
    response_status: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "url": self.url,
            "detail": self.detail,
            "latency_ms": round(self.latency_ms, 3),
            "response_status": self.response_status,
            "metadata": self.metadata,
        }


def _normalize_base_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme for smoke check: {parsed.scheme!r}")
    return url.rstrip("/") + "/"


def _build_headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Correlation-ID": f"smoke-{uuid.uuid4()}",
    }
    if args.bearer_token:
        headers["Authorization"] = f"Bearer {args.bearer_token}"
    elif args.api_key:
        headers["X-API-Key"] = args.api_key
    return headers


def _request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, method=method, headers=headers or {})
    with urlopen(request, timeout=timeout) as response:  # nosec B310
        body = response.read().decode("utf-8")
        try:
            parsed: dict[str, Any] | list[Any] | str = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = body
        return response.status, parsed


def _extract_metadata(body: dict[str, Any] | list[Any] | str) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    metadata: dict[str, Any] = {}
    for key in ("status", "service", "version", "environment"):
        if key in body:
            metadata[key] = body[key]
    if "transaction_id" in body:
        metadata["transaction_id"] = body["transaction_id"]
    if "external_transaction_id" in body:
        metadata["external_transaction_id"] = body["external_transaction_id"]
    if "paths" in body and isinstance(body["paths"], dict):
        metadata["path_count"] = len(body["paths"])
    return metadata


def _run_check(
    name: str,
    url: str,
    *,
    expected_status: set[int],
    timeout: float,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    method: str = "GET",
    validator: Any | None = None,
) -> CheckResult:
    started = time.perf_counter()
    try:
        status_code, body = _request_json(
            method,
            url,
            headers=headers,
            payload=payload,
            timeout=timeout,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if status_code not in expected_status:
            return CheckResult(
                name=name,
                status="fail",
                url=url,
                detail=f"Unexpected status {status_code}; expected {sorted(expected_status)}",
                latency_ms=elapsed_ms,
                response_status=status_code,
            )
        if validator is not None:
            validation_error = validator(body)
            if validation_error:
                return CheckResult(
                    name=name,
                    status="fail",
                    url=url,
                    detail=validation_error,
                    latency_ms=elapsed_ms,
                    response_status=status_code,
                )
        return CheckResult(
            name=name,
            status="pass",
            url=url,
            detail="ok",
            latency_ms=elapsed_ms,
            response_status=status_code,
            metadata=_extract_metadata(body),
        )
    except HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        detail = exc.read().decode("utf-8", errors="replace")
        return CheckResult(
            name=name,
            status="fail",
            url=url,
            detail=f"HTTP {exc.code}: {detail[:500]}",
            latency_ms=elapsed_ms,
            response_status=exc.code,
        )
    except (TimeoutError, URLError, OSError) as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return CheckResult(
            name=name,
            status="fail",
            url=url,
            detail=str(exc),
            latency_ms=elapsed_ms,
        )


def _validate_ready(body: Any) -> str:
    if not isinstance(body, dict):
        return "Readiness response was not a JSON object"
    if body.get("status") != "ready":
        return f"Readiness status is {body.get('status')!r}; expected 'ready'"
    return ""


def _validate_openapi(body: Any) -> str:
    if not isinstance(body, dict):
        return "OpenAPI response was not a JSON object"
    paths = body.get("paths")
    if not isinstance(paths, dict):
        return "OpenAPI schema missing paths object"
    for expected_path in ("/health/live", "/health/ready", "/api/v1/transactions"):
        if expected_path not in paths:
            return f"OpenAPI schema missing {expected_path}"
    return ""


def _fraud_transaction_payload() -> dict[str, Any]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    suffix = uuid.uuid4().hex[:12].upper()
    return {
        "external_transaction_id": f"TXN-SMOKE-FRAUD-{suffix}",
        "account_id": "SMOKE-ACC-FRAUD",
        "customer_id": "SMOKE-CUST-FRAUD",
        "merchant_id": "MERCH-090",
        "merchant_name": "Smoke Test High Risk Merchant",
        "merchant_category_code": "7995",
        "transaction_amount": "9876.54",
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": "credit",
        "card_last_four": "4242",
        "ip_address": "203.0.113.10",
        "device_id": f"smoke-device-{suffix.lower()}",
        "device_type": "desktop",
        "geo_latitude": "55.7558",
        "geo_longitude": "37.6173",
        "geo_country": "RUS",
        "geo_city": "Moscow",
        "is_international": True,
        "transaction_timestamp": now.isoformat(),
        "metadata": {
            "source": "production_smoke_test",
            "safe_synthetic": True,
            "expected_pattern": "high_value_international_gambling",
        },
    }


def _build_api_checks(args: argparse.Namespace) -> list[CheckResult]:
    base_url = _normalize_base_url(args.base_url)
    headers = _build_headers(args)
    checks = [
        _run_check(
            "api_liveness",
            urljoin(base_url, "health/live"),
            expected_status={200},
            timeout=args.timeout,
        ),
        _run_check(
            "api_readiness",
            urljoin(base_url, "health/ready"),
            expected_status={200},
            timeout=args.timeout,
            validator=_validate_ready,
        ),
    ]
    if not args.skip_openapi:
        checks.append(
            _run_check(
                "api_openapi_schema",
                urljoin(base_url, "openapi.json"),
                expected_status={200},
                timeout=args.timeout,
                validator=_validate_openapi,
            )
        )
    if args.submit_test_transaction:
        checks.append(
            _run_check(
                "api_synthetic_fraud_transaction",
                urljoin(base_url, "api/v1/transactions"),
                expected_status={202},
                headers=headers,
                payload=_fraud_transaction_payload(),
                method="POST",
                timeout=args.timeout,
            )
        )
    return checks


def _build_optional_service_checks(args: argparse.Namespace) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if args.streamlit_url:
        checks.append(
            _run_check(
                "streamlit_health",
                urljoin(_normalize_base_url(args.streamlit_url), "_stcore/health"),
                expected_status={200},
                timeout=args.timeout,
            )
        )
    if args.airflow_url:
        checks.append(
            _run_check(
                "airflow_health",
                urljoin(_normalize_base_url(args.airflow_url), "health"),
                expected_status={200},
                timeout=args.timeout,
            )
        )
    return checks


def _run_once(args: argparse.Namespace) -> list[CheckResult]:
    if args.dry_run:
        payload = _fraud_transaction_payload()
        return [
            CheckResult(
                name="dry_run",
                status="pass",
                detail="Smoke configuration and synthetic payload generation succeeded.",
                metadata={
                    "base_url": args.base_url,
                    "streamlit_url": args.streamlit_url,
                    "airflow_url": args.airflow_url,
                    "submit_test_transaction": args.submit_test_transaction,
                    "payload_external_transaction_id": payload["external_transaction_id"],
                },
            )
        ]
    return _build_api_checks(args) + _build_optional_service_checks(args)


def _summarize(results: list[CheckResult], *, started_at: str) -> dict[str, Any]:
    failed = [item for item in results if not item.passed]
    return {
        "suite": "riskpulse_production_smoke",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "results": [item.to_dict() for item in results],
    }


def _write_report(path: str | None, summary: dict[str, Any]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RiskPulse production smoke checks.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("PRODUCTION_BASE_URL")
        or os.environ.get("RISKPULSE_BASE_URL")
        or "http://127.0.0.1:8000",
        help="Base URL for the RiskPulse API.",
    )
    parser.add_argument(
        "--streamlit-url",
        default=os.environ.get("PRODUCTION_STREAMLIT_URL")
        or os.environ.get("RISKPULSE_STREAMLIT_URL"),
        help="Optional Streamlit dashboard URL.",
    )
    parser.add_argument(
        "--airflow-url",
        default=os.environ.get("PRODUCTION_AIRFLOW_URL") or os.environ.get("RISKPULSE_AIRFLOW_URL"),
        help="Optional Airflow webserver URL.",
    )
    parser.add_argument("--api-key", default=os.environ.get("RISKPULSE_API_KEY"))
    parser.add_argument(
        "--bearer-token",
        default=os.environ.get("RISKPULSE_BEARER_TOKEN"),
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--monitor-seconds", type=int, default=0)
    parser.add_argument("--monitor-interval", type=float, default=15.0)
    parser.add_argument("--skip-openapi", action="store_true")
    parser.add_argument("--submit-test-transaction", action="store_true")
    parser.add_argument(
        "--use-dev-key",
        action="store_true",
        help="Use the development API key when no API key or bearer token is set.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-file", default=os.environ.get("SMOKE_TEST_REPORT"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.use_dev_key and not args.api_key and not args.bearer_token:
        args.api_key = DEFAULT_API_KEY

    started_at = datetime.now(timezone.utc).isoformat()
    deadline = time.monotonic() + args.monitor_seconds if args.monitor_seconds > 0 else None
    all_results: list[CheckResult] = []

    while True:
        attempt_results: list[CheckResult] = []
        for attempt in range(1, args.retries + 1):
            attempt_results = _run_once(args)
            if all(result.passed for result in attempt_results):
                break
            if attempt < args.retries:
                time.sleep(args.retry_delay)
        all_results.extend(attempt_results)

        if deadline is None or time.monotonic() >= deadline:
            break
        time.sleep(args.monitor_interval)

    summary = _summarize(all_results, started_at=started_at)
    print(json.dumps(summary, indent=2))
    _write_report(args.report_file, summary)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
