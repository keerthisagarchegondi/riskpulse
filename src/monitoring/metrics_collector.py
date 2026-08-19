"""CloudWatch custom metrics for RiskPulse platform monitoring."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generator

from src.utils.config import get_settings

NAMESPACE = "RiskPulse/Platform"

METRIC_TRANSACTIONS_PER_MINUTE = "TransactionsProcessedPerMinute"
METRIC_FRAUD_DETECTION_RATE = "FraudDetectionRate"
METRIC_ALERT_GENERATION_RATE = "AlertGenerationRate"
METRIC_PIPELINE_LATENCY = "PipelineLatency"
METRIC_MODEL_PREDICTION_LATENCY = "ModelPredictionLatency"
METRIC_ERROR_RATE = "ErrorRate"
METRIC_ERROR_COUNT = "ErrorCount"
METRIC_KAFKA_CONSUMER_LAG = "KafkaConsumerLag"
METRIC_DEPENDENCY_HEALTH = "DependencyHealth"
METRIC_SLA_COMPLIANCE = "SlaCompliance"
METRIC_FALSE_POSITIVE_RATE = "FalsePositiveRate"
METRIC_THROUGHPUT_RECORDS = "PipelineThroughputRecords"

DEFAULT_DASHBOARD_NAME = "riskpulse-platform-health"


@dataclass(frozen=True)
class MetricDimensions:
    """Common dimensions attached to RiskPulse platform metrics."""

    environment: str
    service: str
    severity: str = "none"

    def to_cloudwatch(self, extra: dict[str, str] | None = None) -> list[dict[str, str]]:
        values = {
            "Environment": self.environment,
            "Service": self.service,
            "Severity": self.severity,
        }
        if extra:
            values.update(extra)
        return [{"Name": key, "Value": str(value)} for key, value in values.items()]


class CloudWatchMetricsCollector:
    """Publishes custom RiskPulse metrics to Amazon CloudWatch."""

    def __init__(
        self,
        *,
        service: str,
        environment: str | None = None,
        namespace: str = NAMESPACE,
        client: Any | None = None,
        batch_size: int = 20,
        enabled: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.service = service
        self.environment = environment or settings.environment
        self.namespace = namespace
        self.batch_size = batch_size
        self.enabled = (
            enabled
            if enabled is not None
            else bool(settings.get("monitoring.cloudwatch.metrics_enabled", True))
        )
        self._client = client or (self._create_client() if self.enabled else None)
        self._buffer: list[dict[str, Any]] = []

    @staticmethod
    def _create_client() -> Any:
        import boto3

        return boto3.client("cloudwatch")

    def put_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "Count",
        severity: str = "none",
        timestamp: datetime | None = None,
        dimensions: dict[str, str] | None = None,
        storage_resolution: int = 60,
    ) -> None:
        """Buffer one metric datum and flush when the batch is full."""
        if not self.enabled:
            return

        datum = {
            "MetricName": name,
            "Dimensions": MetricDimensions(
                environment=self.environment,
                service=self.service,
                severity=severity,
            ).to_cloudwatch(dimensions),
            "Timestamp": timestamp or datetime.now(timezone.utc),
            "Value": float(value),
            "Unit": unit,
            "StorageResolution": storage_resolution,
        }
        self._buffer.append(datum)
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Publish all buffered metrics to CloudWatch."""
        if not self.enabled or not self._client or not self._buffer:
            return

        for start in range(0, len(self._buffer), self.batch_size):
            batch = self._buffer[start : start + self.batch_size]
            self._client.put_metric_data(Namespace=self.namespace, MetricData=batch)
        self._buffer = []

    def record_transactions_processed(self, count: int, *, window_seconds: int = 60) -> None:
        per_minute = count * (60 / max(window_seconds, 1))
        self.put_metric(METRIC_TRANSACTIONS_PER_MINUTE, per_minute, unit="Count")
        self.put_metric(METRIC_THROUGHPUT_RECORDS, count, unit="Count")

    def record_fraud_detection_rate(self, fraud_count: int, total_count: int) -> None:
        rate = (fraud_count / total_count) * 100 if total_count else 0.0
        self.put_metric(METRIC_FRAUD_DETECTION_RATE, rate, unit="Percent", severity="high")

    def record_alert_generation_rate(self, alert_count: int, *, window_seconds: int = 60) -> None:
        rate = alert_count * (60 / max(window_seconds, 1))
        self.put_metric(METRIC_ALERT_GENERATION_RATE, rate, unit="Count", severity="medium")

    def record_pipeline_latency(self, latency_ms: float, *, stage: str) -> None:
        self.put_metric(
            METRIC_PIPELINE_LATENCY,
            latency_ms,
            unit="Milliseconds",
            dimensions={"Stage": stage},
        )

    def record_latency_percentiles(self, latencies_ms: Iterable[float], *, stage: str) -> None:
        values = sorted(float(value) for value in latencies_ms)
        if not values:
            return

        for percentile in (50, 95, 99):
            value = _percentile(values, percentile)
            self.put_metric(
                METRIC_PIPELINE_LATENCY,
                value,
                unit="Milliseconds",
                severity="high" if percentile == 99 else "none",
                dimensions={"Stage": stage, "Percentile": f"P{percentile}"},
            )

    def record_model_prediction_latency(self, latency_ms: float, *, model_name: str) -> None:
        self.put_metric(
            METRIC_MODEL_PREDICTION_LATENCY,
            latency_ms,
            unit="Milliseconds",
            dimensions={"ModelName": model_name},
        )

    def record_error(
        self, *, error_count: int = 1, total_count: int | None = None, severity: str = "high"
    ) -> None:
        self.put_metric(METRIC_ERROR_COUNT, error_count, unit="Count", severity=severity)
        if total_count is not None:
            rate = (error_count / total_count) * 100 if total_count else 0.0
            self.put_metric(METRIC_ERROR_RATE, rate, unit="Percent", severity=severity)

    def record_kafka_consumer_lag(
        self, lag_messages: int, *, topic: str, consumer_group: str
    ) -> None:
        self.put_metric(
            METRIC_KAFKA_CONSUMER_LAG,
            lag_messages,
            unit="Count",
            severity="high",
            dimensions={"Topic": topic, "ConsumerGroup": consumer_group},
        )

    def record_dependency_health(self, dependency: str, healthy: bool) -> None:
        self.put_metric(
            METRIC_DEPENDENCY_HEALTH,
            1 if healthy else 0,
            unit="None",
            severity="critical" if not healthy else "none",
            dimensions={"Dependency": dependency},
        )

    def record_sla_compliance(self, compliance_percent: float, *, workflow: str) -> None:
        self.put_metric(
            METRIC_SLA_COMPLIANCE,
            compliance_percent,
            unit="Percent",
            dimensions={"Workflow": workflow},
        )

    def record_false_positive_rate(self, false_positive_count: int, resolved_count: int) -> None:
        rate = (false_positive_count / resolved_count) * 100 if resolved_count else 0.0
        self.put_metric(METRIC_FALSE_POSITIVE_RATE, rate, unit="Percent", severity="medium")

    @contextmanager
    def time_model_prediction(self, *, model_name: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record_model_prediction_latency(
                (time.perf_counter() - start) * 1000,
                model_name=model_name,
            )


def _percentile(sorted_values: list[float], percentile: int) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (rank - lower)


def build_platform_dashboard_body(
    *,
    environment: str,
    region: str = "us-east-1",
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """Build a CloudWatch dashboard body with four operational panels."""
    return {
        "start": "-PT3H",
        "periodOverride": "auto",
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "Pipeline Health Overview",
                    "region": region,
                    "view": "timeSeries",
                    "stacked": False,
                    "metrics": [
                        [namespace, METRIC_TRANSACTIONS_PER_MINUTE, "Environment", environment],
                        [".", METRIC_THROUGHPUT_RECORDS, ".", "."],
                        [".", METRIC_DEPENDENCY_HEALTH, ".", ".", "Dependency", "kafka"],
                        [".", METRIC_DEPENDENCY_HEALTH, ".", ".", "Dependency", "postgresql"],
                    ],
                },
            },
            {
                "type": "metric",
                "x": 12,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "Fraud Metrics",
                    "region": region,
                    "view": "timeSeries",
                    "metrics": [
                        [namespace, METRIC_FRAUD_DETECTION_RATE, "Environment", environment],
                        [".", METRIC_ALERT_GENERATION_RATE, ".", "."],
                        [".", METRIC_FALSE_POSITIVE_RATE, ".", "."],
                    ],
                    "yAxis": {"left": {"min": 0}},
                },
            },
            {
                "type": "metric",
                "x": 0,
                "y": 6,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "Error Rate",
                    "region": region,
                    "view": "timeSeries",
                    "metrics": [
                        [namespace, METRIC_ERROR_RATE, "Environment", environment],
                        [".", METRIC_ERROR_COUNT, ".", "."],
                    ],
                    "annotations": {"horizontal": [{"label": "1% error rate", "value": 1}]},
                },
            },
            {
                "type": "metric",
                "x": 12,
                "y": 6,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "Latency and Consumer Lag",
                    "region": region,
                    "view": "timeSeries",
                    "metrics": [
                        [
                            namespace,
                            METRIC_PIPELINE_LATENCY,
                            "Environment",
                            environment,
                            "Percentile",
                            "P99",
                        ],
                        [".", METRIC_MODEL_PREDICTION_LATENCY, ".", "."],
                        [".", METRIC_KAFKA_CONSUMER_LAG, ".", "."],
                    ],
                    "annotations": {
                        "horizontal": [
                            {"label": "P99 latency 5s", "value": 5000},
                            {"label": "Lag 10K", "value": 10000},
                        ]
                    },
                },
            },
        ],
    }
