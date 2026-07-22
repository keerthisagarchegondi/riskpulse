"""Data Quality DAG — scheduled daily quality checks.

Runs completeness, freshness, and volume anomaly detection against the
operational database.  Generates a quality report and alerts the team
when any critical degradation is detected.

Schedule: daily at 06:00 UTC
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

from operators.data_quality_operator import (
    CheckResult,
    CheckSeverity,
    DataQualityOperator,
    QualityReport,
)
from src.utils.config import get_settings
from src.utils.constants import TOPIC_METRICS
from src.utils.logger import get_logger

logger = get_logger(__name__, component="data_quality_dag")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

default_args = {
    "owner": "riskpulse",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

_COMPLETENESS_CHECKS = [
    {"table": "transactions", "column": "transaction_id", "severity": CheckSeverity.CRITICAL},
    {"table": "transactions", "column": "amount", "severity": CheckSeverity.CRITICAL},
    {"table": "transactions", "column": "account_id", "severity": CheckSeverity.CRITICAL},
    {"table": "transactions", "column": "currency", "severity": CheckSeverity.CRITICAL},
    {"table": "transactions", "column": "transaction_type", "severity": CheckSeverity.WARNING},
    {"table": "transactions", "column": "merchant_name", "severity": CheckSeverity.WARNING},
    {"table": "transactions", "column": "country", "severity": CheckSeverity.WARNING},
    {"table": "fraud_alerts", "column": "alert_id", "severity": CheckSeverity.CRITICAL},
    {"table": "fraud_alerts", "column": "transaction_id", "severity": CheckSeverity.CRITICAL},
    {"table": "fraud_alerts", "column": "severity", "severity": CheckSeverity.CRITICAL},
    {"table": "risk_scores", "column": "score_id", "severity": CheckSeverity.CRITICAL},
    {"table": "risk_scores", "column": "risk_score", "severity": CheckSeverity.CRITICAL},
]

_FRESHNESS_CHECKS = [
    {
        "table": "transactions",
        "timestamp_column": "created_at",
        "max_delay_minutes": 30,
        "severity": CheckSeverity.CRITICAL,
    },
    {
        "table": "fraud_alerts",
        "timestamp_column": "created_at",
        "max_delay_minutes": 60,
        "severity": CheckSeverity.WARNING,
    },
    {
        "table": "risk_scores",
        "timestamp_column": "created_at",
        "max_delay_minutes": 60,
        "severity": CheckSeverity.WARNING,
    },
    {
        "table": "audit_logs",
        "timestamp_column": "created_at",
        "max_delay_minutes": 120,
        "severity": CheckSeverity.WARNING,
    },
]

_VOLUME_TABLES = ["transactions", "fraud_alerts", "risk_scores"]


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _run_completeness_checks(**context: Any) -> dict[str, Any]:
    """Execute completeness checks via the DataQualityOperator."""
    op = DataQualityOperator(
        task_id="_internal_completeness",
        conn_id="riskpulse_postgres",
        completeness_checks=_COMPLETENESS_CHECKS,
        freshness_checks=[],
        volume_tables=[],
    )
    return op.execute(context)


def _run_freshness_checks(**context: Any) -> dict[str, Any]:
    """Execute freshness checks via the DataQualityOperator."""
    op = DataQualityOperator(
        task_id="_internal_freshness",
        conn_id="riskpulse_postgres",
        completeness_checks=[],
        freshness_checks=_FRESHNESS_CHECKS,
        volume_tables=[],
    )
    return op.execute(context)


def _run_volume_anomaly_detection(**context: Any) -> dict[str, Any]:
    """Execute volume anomaly detection via the DataQualityOperator."""
    op = DataQualityOperator(
        task_id="_internal_volume",
        conn_id="riskpulse_postgres",
        completeness_checks=[],
        freshness_checks=[],
        volume_tables=_VOLUME_TABLES,
        volume_lookback_days=7,
        volume_stddev_threshold=2.0,
    )
    return op.execute(context)


def _generate_quality_report(**context: Any) -> dict[str, Any]:
    """Aggregate results from all check tasks into a unified report."""
    ti = context["ti"]

    completeness = ti.xcom_pull(task_ids="completeness_checks") or {}
    freshness = ti.xcom_pull(task_ids="freshness_checks") or {}
    volume = ti.xcom_pull(task_ids="volume_anomaly_detection") or {}

    report = {
        "dag_id": "data_quality_pipeline",
        "run_id": context["run_id"],
        "execution_date": context["execution_date"].isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "completeness": {
                "total": completeness.get("total_checks", 0),
                "passed": completeness.get("passed", 0),
                "warnings": completeness.get("warnings", 0),
                "critical": completeness.get("critical_failures", 0),
            },
            "freshness": {
                "total": freshness.get("total_checks", 0),
                "passed": freshness.get("passed", 0),
                "warnings": freshness.get("warnings", 0),
                "critical": freshness.get("critical_failures", 0),
            },
            "volume": {
                "total": volume.get("total_checks", 0),
                "passed": volume.get("passed", 0),
                "warnings": volume.get("warnings", 0),
                "critical": volume.get("critical_failures", 0),
            },
        },
        "all_checks": (
            completeness.get("checks", [])
            + freshness.get("checks", [])
            + volume.get("checks", [])
        ),
    }

    total_critical = (
        completeness.get("critical_failures", 0)
        + freshness.get("critical_failures", 0)
        + volume.get("critical_failures", 0)
    )
    total_warnings = (
        completeness.get("warnings", 0)
        + freshness.get("warnings", 0)
        + volume.get("warnings", 0)
    )
    total_passed = (
        completeness.get("passed", 0)
        + freshness.get("passed", 0)
        + volume.get("passed", 0)
    )

    report["overall"] = {
        "total_checks": total_passed + total_warnings + total_critical,
        "passed": total_passed,
        "warnings": total_warnings,
        "critical_failures": total_critical,
        "status": "HEALTHY" if total_critical == 0 else "DEGRADED",
    }

    logger.info(
        "Quality report generated",
        status=report["overall"]["status"],
        total=report["overall"]["total_checks"],
        passed=total_passed,
        warnings=total_warnings,
        critical=total_critical,
    )

    # Publish report as metrics
    _publish_quality_metrics(report)

    return report


def _publish_quality_metrics(report: dict[str, Any]) -> None:
    """Publish quality report to the metrics Kafka topic."""
    from confluent_kafka import Producer as KafkaProducer

    try:
        settings = get_settings()
        producer = KafkaProducer(
            {"bootstrap.servers": settings.kafka_bootstrap_servers}
        )
        producer.produce(
            topic=TOPIC_METRICS,
            key="data_quality_report",
            value=json.dumps(report, default=str).encode("utf-8"),
        )
        producer.flush(timeout=10)
        logger.info("Quality metrics published")
    except Exception as exc:
        logger.warning("Failed to publish quality metrics", error=str(exc))


def _notify_on_degradation(**context: Any) -> None:
    """Send alert if the quality report shows degradation."""
    ti = context["ti"]
    report = ti.xcom_pull(task_ids="generate_quality_report")

    if not report:
        logger.warning("No quality report available for alerting")
        return

    status = report.get("overall", {}).get("status", "UNKNOWN")
    if status == "HEALTHY":
        logger.info("Data quality healthy, no alert needed")
        return

    critical = report.get("overall", {}).get("critical_failures", 0)
    warnings = report.get("overall", {}).get("warnings", 0)

    from src.alerting.alert_manager import AlertManager

    try:
        alert_manager = AlertManager()

        failed_checks = [
            c for c in report.get("all_checks", []) if not c.get("passed")
        ]
        check_summary = "; ".join(
            f"{c['check_name']}: {c.get('details', 'failed')}"
            for c in failed_checks[:10]
        )

        alert_manager.generate_alert(
            scoring_result={
                "score": 0.0,
                "risk_level": "high" if critical > 0 else "medium",
                "triggered_rules": [],
            },
            transaction={
                "transaction_id": f"dq_alert_{context.get('run_id', 'unknown')}",
                "amount": 0,
                "type": "system",
                "channel": "airflow",
                "metadata": {
                    "alert_type": "data_quality_degradation",
                    "critical_failures": critical,
                    "warnings": warnings,
                    "failed_checks": check_summary,
                },
            },
        )
        logger.info(
            "Data quality degradation alert sent",
            critical=critical,
            warnings=warnings,
        )
    except Exception as exc:
        logger.error("Failed to send quality degradation alert", error=str(exc))


def _on_failure_callback(context: dict[str, Any]) -> None:
    """Alert on task failure."""
    from src.alerting.alert_manager import AlertManager

    dag_id = context.get("dag", {})
    task_id = context.get("task_instance", {})
    exception = context.get("exception")

    logger.error(
        "Data quality DAG task failed",
        dag_id=str(dag_id),
        task_id=str(task_id),
        error=str(exception),
    )

    try:
        alert_manager = AlertManager()
        alert_manager.generate_alert(
            scoring_result={
                "score": 0.0,
                "risk_level": "critical",
                "triggered_rules": [],
            },
            transaction={
                "transaction_id": f"dag_failure_{context.get('run_id', 'unknown')}",
                "amount": 0,
                "type": "system",
                "channel": "airflow",
                "metadata": {
                    "dag_id": str(dag_id),
                    "task_id": str(task_id),
                    "error": str(exception)[:500],
                },
            },
        )
    except Exception as alert_exc:
        logger.error("Failed to send failure alert", error=str(alert_exc))


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="data_quality_pipeline",
    description="Daily data quality checks: completeness, freshness, volume anomalies",
    schedule_interval="0 6 * * *",  # Daily at 06:00 UTC
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["data-quality", "monitoring", "riskpulse"],
    default_args={**default_args, "on_failure_callback": _on_failure_callback},
) as dag:

    completeness = PythonOperator(
        task_id="completeness_checks",
        python_callable=_run_completeness_checks,
    )

    freshness = PythonOperator(
        task_id="freshness_checks",
        python_callable=_run_freshness_checks,
    )

    volume = PythonOperator(
        task_id="volume_anomaly_detection",
        python_callable=_run_volume_anomaly_detection,
    )

    report = PythonOperator(
        task_id="generate_quality_report",
        python_callable=_generate_quality_report,
        trigger_rule="all_done",  # run even if some checks fail
    )

    alert = PythonOperator(
        task_id="notify_on_degradation",
        python_callable=_notify_on_degradation,
        trigger_rule="all_done",
    )

    # Completeness, freshness, and volume run in parallel; report aggregates
    [completeness, freshness, volume] >> report >> alert
