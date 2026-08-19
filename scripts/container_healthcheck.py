"""Container health checks for RiskPulse services."""

from __future__ import annotations

import os
import socket
import sys
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


def _check_http(url: str, timeout: float = 5.0) -> int:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return 1
    try:
        with urlopen(url, timeout=timeout) as response:  # nosec B310
            return 0 if 200 <= response.status < 500 else 1
    except URLError:
        return 1


def _check_tcp(host: str, port: int, timeout: float = 5.0) -> int:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return 0
    except OSError:
        return 1


def main() -> int:
    service = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RISKPULSE_SERVICE", "api")

    if service == "api":
        return _check_http(os.environ.get("HEALTHCHECK_URL", "http://127.0.0.1:8000/health/live"))
    if service == "streamlit":
        return _check_http(
            os.environ.get("HEALTHCHECK_URL", "http://127.0.0.1:8501/_stcore/health")
        )
    if service == "worker":
        host = os.environ.get("RISKPULSE_KAFKA_HOST", "kafka")
        port = int(os.environ.get("RISKPULSE_KAFKA_PORT", "29092"))
        return _check_tcp(host, port)
    if service == "airflow":
        return _check_http(os.environ.get("HEALTHCHECK_URL", "http://127.0.0.1:8080/health"))

    print(f"Unknown healthcheck service: {service}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
