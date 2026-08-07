"""Tests for CloudWatch monitoring, metrics, alarms, and health models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.monitoring.cloudwatch_logger import CloudWatchLogHandler, scrub_pii, set_correlation_id
from src.monitoring.health_checker import DependencyCheckResult, HealthChecker
from src.monitoring.metrics_collector import (
    METRIC_ALERT_GENERATION_RATE,
    METRIC_ERROR_RATE,
    METRIC_FRAUD_DETECTION_RATE,
    METRIC_KAFKA_CONSUMER_LAG,
    METRIC_PIPELINE_LATENCY,
    CloudWatchMetricsCollector,
    build_platform_dashboard_body,
)


class FakeLogsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.put_batches: list[list[dict]] = []

    def create_log_group(self, **kwargs):
        self.calls.append(("create_log_group", kwargs))

    def put_retention_policy(self, **kwargs):
        self.calls.append(("put_retention_policy", kwargs))

    def create_log_stream(self, **kwargs):
        self.calls.append(("create_log_stream", kwargs))

    def describe_log_streams(self, **kwargs):
        self.calls.append(("describe_log_streams", kwargs))
        return {"logStreams": [{"logStreamName": kwargs["logStreamNamePrefix"]}]}

    def put_log_events(self, **kwargs):
        self.calls.append(("put_log_events", kwargs))
        self.put_batches.append(kwargs["logEvents"])
        return {"nextSequenceToken": "token-2"}


class FakeMetricClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def put_metric_data(self, **kwargs):
        self.calls.append(kwargs)


def test_scrub_pii_recurses_and_masks_inline_values() -> None:
    payload = {
        "email": "analyst@example.com",
        "nested": {
            "message": "customer analyst@example.com used 4111 1111 1111 1111",
            "device_id": "device-123",
        },
        "notes": ["call +1 212-555-0199", "ssn 123-45-6789"],
    }

    scrubbed = scrub_pii(payload)

    assert scrubbed["email"] == "***REDACTED***"
    assert scrubbed["nested"]["device_id"] == "***REDACTED***"
    assert "***REDACTED_EMAIL***" in scrubbed["nested"]["message"]
    assert "***REDACTED_CARD***" in scrubbed["nested"]["message"]
    assert "***REDACTED_PHONE***" in scrubbed["notes"][0]
    assert "***REDACTED_SSN***" in scrubbed["notes"][1]


def test_cloudwatch_log_handler_ships_redacted_json_with_correlation_id() -> None:
    client = FakeLogsClient()
    handler = CloudWatchLogHandler(
        service="api",
        environment="prod",
        client=client,
        log_stream_name="unit-test",
        max_batch_size=1,
    )
    logger = logging.getLogger("riskpulse.test.cloudwatch")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    set_correlation_id("corr-123")
    logger.info(
        json.dumps(
            {
                "event": "payment_received",
                "email": "customer@example.com",
                "card_number": "4111111111111111",
            }
        )
    )
    handler.close()

    assert any(name == "create_log_group" for name, _ in client.calls)
    assert client.put_batches
    shipped = json.loads(client.put_batches[0][0]["message"])
    assert shipped["correlation_id"] == "corr-123"
    assert shipped["email"] == "***REDACTED***"
    assert shipped["card_number"] == "***REDACTED***"


def test_metrics_collector_publishes_required_custom_metrics() -> None:
    client = FakeMetricClient()
    collector = CloudWatchMetricsCollector(
        service="api",
        environment="prod",
        client=client,
        batch_size=20,
        enabled=True,
    )

    collector.record_transactions_processed(600, window_seconds=60)
    collector.record_fraud_detection_rate(fraud_count=12, total_count=600)
    collector.record_alert_generation_rate(18, window_seconds=60)
    collector.record_pipeline_latency(120.0, stage="ingestion")
    collector.record_latency_percentiles([100, 140, 200, 420, 900], stage="scoring")
    collector.record_model_prediction_latency(42.0, model_name="risk_scorer")
    collector.record_error(error_count=3, total_count=600)
    collector.record_kafka_consumer_lag(128, topic="txn.scored", consumer_group="alerts")
    collector.record_dependency_health("kafka", True)
    collector.record_sla_compliance(99.2, workflow="alert-resolution")
    collector.record_false_positive_rate(false_positive_count=4, resolved_count=80)
    collector.flush()

    metric_data = [datum for call in client.calls for datum in call["MetricData"]]
    metric_names = {datum["MetricName"] for datum in metric_data}

    assert len(metric_data) >= 12
    assert METRIC_FRAUD_DETECTION_RATE in metric_names
    assert METRIC_ALERT_GENERATION_RATE in metric_names
    assert METRIC_PIPELINE_LATENCY in metric_names
    assert METRIC_ERROR_RATE in metric_names
    assert METRIC_KAFKA_CONSUMER_LAG in metric_names
    assert all(
        {"Environment", "Service", "Severity"}.issubset(
            {dimension["Name"] for dimension in datum["Dimensions"]}
        )
        for datum in metric_data
    )


def test_health_checker_marks_critical_failure_degraded() -> None:
    checker = HealthChecker(environment="prod", start_time=1000.0)
    health = checker.build_health(
        [
            DependencyCheckResult("kafka", "healthy", 2.0, critical=True),
            DependencyCheckResult("postgresql", "unhealthy", 500.0, critical=True, detail="timeout"),
            DependencyCheckResult("redis", "healthy", 1.0, critical=False),
        ]
    )

    assert health.status == "degraded"
    assert health.environment == "prod"
    assert health.dependencies[1].detail == "timeout"


def test_cloudwatch_dashboard_and_alarm_artifacts_are_valid() -> None:
    dashboard_path = Path("infrastructure/aws/cloudwatch/dashboards/platform_dashboard.json")
    high_fraud_path = Path("infrastructure/aws/cloudwatch/alarms/high_fraud_rate.json")
    pipeline_path = Path("infrastructure/aws/cloudwatch/alarms/pipeline_failure.json")

    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    high_fraud = json.loads(high_fraud_path.read_text(encoding="utf-8"))
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))

    assert len(dashboard["widgets"]) == 4
    assert high_fraud["MetricName"] == METRIC_FRAUD_DETECTION_RATE
    assert high_fraud["Threshold"] == 2.0
    assert len(pipeline["alarms"]) >= 5
    assert {alarm["MetricName"] for alarm in pipeline["alarms"]} >= {
        "ErrorRate",
        "PipelineLatency",
        "KafkaConsumerLag",
        "DependencyHealth",
    }


def test_dashboard_builder_matches_required_panel_count() -> None:
    body = build_platform_dashboard_body(environment="prod")

    titles = {widget["properties"]["title"] for widget in body["widgets"]}
    assert titles == {
        "Pipeline Health Overview",
        "Fraud Metrics",
        "Error Rate",
        "Latency and Consumer Lag",
    }


def test_terraform_cloudwatch_module_contains_core_resources() -> None:
    module = Path("infrastructure/terraform/modules/cloudwatch/main.tf").read_text(encoding="utf-8")

    assert 'resource "aws_cloudwatch_log_group" "service"' in module
    assert 'resource "aws_cloudwatch_dashboard" "platform"' in module
    assert module.count('resource "aws_cloudwatch_metric_alarm"') >= 5
