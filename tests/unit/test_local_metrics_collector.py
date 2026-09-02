from __future__ import annotations

import json

from src.monitoring.metrics_collector import (
    METRIC_TRANSACTIONS_PER_MINUTE,
    CloudWatchMetricsCollector,
)


def test_metrics_collector_writes_local_jsonl_when_cloudwatch_disabled(
    monkeypatch, tmp_path
) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("RISKPULSE_METRICS_BACKEND", "local")
    monkeypatch.setenv("RISKPULSE_LOCAL_METRICS_PATH", str(metrics_path))

    collector = CloudWatchMetricsCollector(service="api", environment="dev", enabled=False)
    collector.record_transactions_processed(12, window_seconds=60)

    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]

    assert [payload["MetricName"] for payload in payloads] == [
        METRIC_TRANSACTIONS_PER_MINUTE,
        "PipelineThroughputRecords",
    ]
    assert payloads[0]["Value"] == 12
    assert payloads[0]["Namespace"] == "RiskPulse/Platform"
