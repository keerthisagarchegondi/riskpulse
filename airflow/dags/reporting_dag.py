"""Reporting DAG — daily fraud summary, reporting schema, and digest email.

Generates daily fraud analytics summaries, populates the reporting
schema in Snowflake, and sends a digest email to the fraud operations
team.

Schedule: daily at 06:00 UTC
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

from airflow import DAG
from src.alerting.notification_service import NotificationService
from src.utils.config import get_settings
from src.utils.constants import TOPIC_METRICS
from src.utils.logger import get_logger

logger = get_logger(__name__, component="reporting_dag")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SNOWFLAKE_CONN_ID = "riskpulse_snowflake"
_SNOWFLAKE_DATABASE = "RISKPULSE"
_REPORT_RECIPIENTS = [
    "fraud-ops@riskpulse.io",
    "risk-management@riskpulse.io",
]
_DIGEST_HOUR = 6  # 06:00 UTC

default_args = {
    "owner": "riskpulse",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=20),
    "execution_timeout": timedelta(minutes=45),
}


# ---------------------------------------------------------------------------
# SQL templates for reporting
# ---------------------------------------------------------------------------

_DAILY_SUMMARY_SQL = """
SELECT
    DATE_TRUNC('day', transaction_date) AS report_date,
    COUNT(*) AS total_transactions,
    SUM(amount) AS total_volume,
    AVG(amount) AS avg_transaction_amount,
    SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) AS fraud_transactions,
    SUM(CASE WHEN is_fraudulent THEN amount ELSE 0 END) AS fraud_volume,
    ROUND(SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END)::FLOAT /
          NULLIF(COUNT(*), 0) * 100, 4) AS fraud_rate_pct,
    AVG(ensemble_score) AS avg_risk_score,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ensemble_score) AS median_score,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ensemble_score) AS p95_score,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ensemble_score) AS p99_score,
    SUM(CASE WHEN ensemble_risk_level = 'low' THEN 1 ELSE 0 END) AS low_risk_count,
    SUM(CASE WHEN ensemble_risk_level = 'medium' THEN 1 ELSE 0 END) AS medium_risk_count,
    SUM(CASE WHEN ensemble_risk_level = 'high' THEN 1 ELSE 0 END) AS high_risk_count,
    SUM(CASE WHEN ensemble_risk_level = 'critical' THEN 1 ELSE 0 END) AS critical_risk_count,
    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) AS anomaly_count,
    COUNT(DISTINCT account_id) AS unique_accounts,
    COUNT(DISTINCT merchant_name) AS unique_merchants,
    COUNT(DISTINCT country) AS unique_countries
FROM {database}.ANALYTICS.FACT_TRANSACTIONS
WHERE transaction_date = DATEADD(day, -1, CURRENT_DATE())
GROUP BY 1;
"""

_TOP_FRAUD_MERCHANTS_SQL = """
SELECT
    merchant_name,
    merchant_category,
    COUNT(*) AS fraud_transactions,
    SUM(amount) AS fraud_volume,
    AVG(ensemble_score) AS avg_score
FROM {database}.ANALYTICS.FACT_TRANSACTIONS
WHERE transaction_date = DATEADD(day, -1, CURRENT_DATE())
  AND is_fraudulent = TRUE
GROUP BY 1, 2
ORDER BY fraud_volume DESC
LIMIT 20;
"""

_TOP_FRAUD_COUNTRIES_SQL = """
SELECT
    country,
    COUNT(*) AS fraud_transactions,
    SUM(amount) AS fraud_volume,
    AVG(ensemble_score) AS avg_score,
    ROUND(SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END)::FLOAT /
          NULLIF(COUNT(*), 0) * 100, 2) AS fraud_rate_pct
FROM {database}.ANALYTICS.FACT_TRANSACTIONS
WHERE transaction_date = DATEADD(day, -1, CURRENT_DATE())
GROUP BY 1
HAVING COUNT(*) >= 10
ORDER BY fraud_rate_pct DESC
LIMIT 20;
"""

_ALERT_SUMMARY_SQL = """
SELECT
    severity,
    status,
    COUNT(*) AS alert_count,
    AVG(risk_score) AS avg_risk_score
FROM {database}.STAGING.FRAUD_ALERTS
WHERE DATE_TRUNC('day', created_at) = DATEADD(day, -1, CURRENT_DATE())
GROUP BY 1, 2
ORDER BY 1, 2;
"""

_POPULATE_REPORTING_SCHEMA = """
INSERT INTO {database}.REPORTING.DAILY_FRAUD_SUMMARY (
    report_date, total_transactions, total_volume,
    avg_transaction_amount, fraud_transactions, fraud_volume,
    fraud_rate_pct, avg_risk_score, median_score,
    p95_score, p99_score, low_risk_count, medium_risk_count,
    high_risk_count, critical_risk_count, anomaly_count,
    unique_accounts, unique_merchants, unique_countries,
    generated_at
)
SELECT
    DATE_TRUNC('day', transaction_date) AS report_date,
    COUNT(*) AS total_transactions,
    SUM(amount) AS total_volume,
    AVG(amount) AS avg_transaction_amount,
    SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) AS fraud_transactions,
    SUM(CASE WHEN is_fraudulent THEN amount ELSE 0 END) AS fraud_volume,
    ROUND(SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END)::FLOAT /
          NULLIF(COUNT(*), 0) * 100, 4) AS fraud_rate_pct,
    AVG(ensemble_score) AS avg_risk_score,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ensemble_score) AS median_score,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ensemble_score) AS p95_score,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ensemble_score) AS p99_score,
    SUM(CASE WHEN ensemble_risk_level = 'low' THEN 1 ELSE 0 END),
    SUM(CASE WHEN ensemble_risk_level = 'medium' THEN 1 ELSE 0 END),
    SUM(CASE WHEN ensemble_risk_level = 'high' THEN 1 ELSE 0 END),
    SUM(CASE WHEN ensemble_risk_level = 'critical' THEN 1 ELSE 0 END),
    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END),
    COUNT(DISTINCT account_id),
    COUNT(DISTINCT merchant_name),
    COUNT(DISTINCT country),
    CURRENT_TIMESTAMP()
FROM {database}.ANALYTICS.FACT_TRANSACTIONS
WHERE transaction_date = DATEADD(day, -1, CURRENT_DATE())
GROUP BY 1;
"""

_POPULATE_MERCHANT_REPORT = """
INSERT INTO {database}.REPORTING.DAILY_MERCHANT_RISK (
    report_date, merchant_name, merchant_category,
    transaction_count, fraud_count, fraud_volume,
    avg_risk_score, fraud_rate_pct, generated_at
)
SELECT
    DATEADD(day, -1, CURRENT_DATE()) AS report_date,
    merchant_name,
    merchant_category,
    COUNT(*) AS transaction_count,
    SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) AS fraud_count,
    SUM(CASE WHEN is_fraudulent THEN amount ELSE 0 END) AS fraud_volume,
    AVG(ensemble_score) AS avg_risk_score,
    ROUND(SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END)::FLOAT /
          NULLIF(COUNT(*), 0) * 100, 2) AS fraud_rate_pct,
    CURRENT_TIMESTAMP()
FROM {database}.ANALYTICS.FACT_TRANSACTIONS
WHERE transaction_date = DATEADD(day, -1, CURRENT_DATE())
GROUP BY 1, 2, 3
HAVING transaction_count >= 5;
"""


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _generate_daily_summary(**context: Any) -> dict[str, Any]:
    """Query Snowflake for daily fraud summary statistics."""
    from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook

    hook = SnowflakeHook(snowflake_conn_id=_SNOWFLAKE_CONN_ID)
    execution_date: datetime = context["execution_date"]
    report_date = (execution_date - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("Generating daily fraud summary", report_date=report_date)

    try:
        # Daily summary
        sql = _DAILY_SUMMARY_SQL.format(database=_SNOWFLAKE_DATABASE)
        summary_rows = hook.get_records(sql)

        if not summary_rows or not summary_rows[0]:
            logger.warning("No data for daily summary", report_date=report_date)
            return {"report_date": report_date, "has_data": False}

        row = summary_rows[0]
        summary = {
            "report_date": report_date,
            "has_data": True,
            "total_transactions": row[1],
            "total_volume": float(row[2]) if row[2] else 0,
            "avg_transaction_amount": float(row[3]) if row[3] else 0,
            "fraud_transactions": row[4],
            "fraud_volume": float(row[5]) if row[5] else 0,
            "fraud_rate_pct": float(row[6]) if row[6] else 0,
            "avg_risk_score": float(row[7]) if row[7] else 0,
            "median_score": float(row[8]) if row[8] else 0,
            "p95_score": float(row[9]) if row[9] else 0,
            "p99_score": float(row[10]) if row[10] else 0,
            "low_risk_count": row[11],
            "medium_risk_count": row[12],
            "high_risk_count": row[13],
            "critical_risk_count": row[14],
            "anomaly_count": row[15],
            "unique_accounts": row[16],
            "unique_merchants": row[17],
            "unique_countries": row[18],
        }

        # Top fraud merchants
        merchants_sql = _TOP_FRAUD_MERCHANTS_SQL.format(database=_SNOWFLAKE_DATABASE)
        merchant_rows = hook.get_records(merchants_sql)
        summary["top_fraud_merchants"] = [
            {
                "merchant_name": r[0],
                "merchant_category": r[1],
                "fraud_transactions": r[2],
                "fraud_volume": float(r[3]) if r[3] else 0,
                "avg_score": float(r[4]) if r[4] else 0,
            }
            for r in (merchant_rows or [])
        ]

        # Top fraud countries
        countries_sql = _TOP_FRAUD_COUNTRIES_SQL.format(database=_SNOWFLAKE_DATABASE)
        country_rows = hook.get_records(countries_sql)
        summary["top_fraud_countries"] = [
            {
                "country": r[0],
                "fraud_transactions": r[1],
                "fraud_volume": float(r[2]) if r[2] else 0,
                "avg_score": float(r[3]) if r[3] else 0,
                "fraud_rate_pct": float(r[4]) if r[4] else 0,
            }
            for r in (country_rows or [])
        ]

        # Alert summary
        alert_sql = _ALERT_SUMMARY_SQL.format(database=_SNOWFLAKE_DATABASE)
        alert_rows = hook.get_records(alert_sql)
        summary["alert_summary"] = [
            {
                "severity": r[0],
                "status": r[1],
                "alert_count": r[2],
                "avg_risk_score": float(r[3]) if r[3] else 0,
            }
            for r in (alert_rows or [])
        ]

        context["ti"].xcom_push(key="daily_summary", value=summary)
        logger.info(
            "Daily summary generated",
            **{
                k: v
                for k, v in summary.items()
                if k != "top_fraud_merchants"
                and k != "top_fraud_countries"
                and k != "alert_summary"
            },
        )
        return summary

    except Exception as exc:
        logger.error("Failed to generate daily summary", error=str(exc))
        raise


def _populate_reporting_schema(**context: Any) -> dict[str, Any]:
    """Insert daily data into Snowflake reporting schema."""
    from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook

    hook = SnowflakeHook(snowflake_conn_id=_SNOWFLAKE_CONN_ID)

    logger.info("Populating reporting schema")

    try:
        # Remove existing data for the report date (idempotent)
        cleanup_sql = f"""
            DELETE FROM {_SNOWFLAKE_DATABASE}.REPORTING.DAILY_FRAUD_SUMMARY
            WHERE report_date = DATEADD(day, -1, CURRENT_DATE());
        """  # nosec B608
        hook.run(cleanup_sql, autocommit=True)

        cleanup_merchant_sql = f"""
            DELETE FROM {_SNOWFLAKE_DATABASE}.REPORTING.DAILY_MERCHANT_RISK
            WHERE report_date = DATEADD(day, -1, CURRENT_DATE());
        """  # nosec B608
        hook.run(cleanup_merchant_sql, autocommit=True)

        # Populate daily summary
        summary_sql = _POPULATE_REPORTING_SCHEMA.format(database=_SNOWFLAKE_DATABASE)
        hook.run(summary_sql, autocommit=True)

        # Populate merchant risk report
        merchant_sql = _POPULATE_MERCHANT_REPORT.format(database=_SNOWFLAKE_DATABASE)
        hook.run(merchant_sql, autocommit=True)

        summary = {"reporting_populated": True}
        logger.info("Reporting schema populated")
        return summary

    except Exception as exc:
        logger.error("Failed to populate reporting schema", error=str(exc))
        raise


def _send_daily_digest(**context: Any) -> dict[str, Any]:
    """Send daily fraud digest email to operations team."""
    ti = context["ti"]
    summary = ti.xcom_pull(task_ids="generate_daily_summary", key="daily_summary")
    settings = get_settings()

    if not summary or not summary.get("has_data"):
        logger.info("No data for digest email, skipping")
        return {"emails_sent": 0, "reason": "no_data"}

    report_date = summary["report_date"]

    # Build email body
    subject = f"[RiskPulse] Daily Fraud Summary — {report_date}"

    body_lines = [
        f"Daily Fraud Summary for {report_date}",
        "=" * 50,
        "",
        "TRANSACTION OVERVIEW",
        f"  Total Transactions:    {summary['total_transactions']:,}",
        f"  Total Volume:          ${summary['total_volume']:,.2f}",
        f"  Avg Amount:            ${summary['avg_transaction_amount']:,.2f}",
        f"  Unique Accounts:       {summary['unique_accounts']:,}",
        f"  Unique Merchants:      {summary['unique_merchants']:,}",
        f"  Countries:             {summary['unique_countries']:,}",
        "",
        "FRAUD METRICS",
        f"  Fraud Transactions:    {summary['fraud_transactions']:,}",
        f"  Fraud Volume:          ${summary['fraud_volume']:,.2f}",
        f"  Fraud Rate:            {summary['fraud_rate_pct']:.4f}%",
        f"  Anomalies Detected:    {summary['anomaly_count']:,}",
        "",
        "RISK DISTRIBUTION",
        f"  Low Risk:              {summary['low_risk_count']:,}",
        f"  Medium Risk:           {summary['medium_risk_count']:,}",
        f"  High Risk:             {summary['high_risk_count']:,}",
        f"  Critical Risk:         {summary['critical_risk_count']:,}",
        "",
        "SCORING METRICS",
        f"  Avg Risk Score:        {summary['avg_risk_score']:.4f}",
        f"  Median Score:          {summary['median_score']:.4f}",
        f"  P95 Score:             {summary['p95_score']:.4f}",
        f"  P99 Score:             {summary['p99_score']:.4f}",
        "",
    ]

    # Top merchants
    if summary.get("top_fraud_merchants"):
        body_lines.append("TOP FRAUD MERCHANTS (by volume)")
        body_lines.append("-" * 40)
        for m in summary["top_fraud_merchants"][:10]:
            body_lines.append(
                f"  {m['merchant_name'][:30]:<30} "
                f"Txns: {m['fraud_transactions']:>5}  "
                f"Vol: ${m['fraud_volume']:>12,.2f}"
            )
        body_lines.append("")

    # Top countries
    if summary.get("top_fraud_countries"):
        body_lines.append("TOP FRAUD COUNTRIES (by rate)")
        body_lines.append("-" * 40)
        for c in summary["top_fraud_countries"][:10]:
            body_lines.append(
                f"  {c['country']:<20} "
                f"Rate: {c['fraud_rate_pct']:>6.2f}%  "
                f"Txns: {c['fraud_transactions']:>5}"
            )
        body_lines.append("")

    # Alert summary
    if summary.get("alert_summary"):
        body_lines.append("ALERT SUMMARY")
        body_lines.append("-" * 40)
        for a in summary["alert_summary"]:
            body_lines.append(
                f"  {a['severity']:<10} | {a['status']:<15} | " f"Count: {a['alert_count']:>5}"
            )
        body_lines.append("")

    body_lines.extend(
        [
            "",
            "---",
            "Generated by RiskPulse Reporting Pipeline",
            f"Timestamp: {datetime.now(timezone.utc).isoformat()}",
        ]
    )

    body = "\n".join(body_lines)

    # Send via notification service
    try:
        notification_service = NotificationService()
        emails_sent = 0

        for recipient in _REPORT_RECIPIENTS:
            notification_service.send(
                channel="email",
                recipient=recipient,
                subject=subject,
                body=body,
            )
            emails_sent += 1

        result = {"emails_sent": emails_sent, "subject": subject}
        logger.info("Daily digest sent", **result)
        return result

    except Exception as exc:
        logger.error("Failed to send digest email", error=str(exc))
        # Don't fail the DAG for email issues
        return {"emails_sent": 0, "error": str(exc)}


def _publish_report_metrics(**context: Any) -> dict[str, Any]:
    """Publish reporting pipeline metrics to Kafka."""
    from confluent_kafka import Producer as KafkaProducer

    ti = context["ti"]
    settings = get_settings()

    summary = ti.xcom_pull(task_ids="generate_daily_summary", key="daily_summary") or {}
    digest_result = ti.xcom_pull(task_ids="send_daily_digest") or {}

    metrics = {
        "dag_id": "reporting_pipeline",
        "run_id": context["run_id"],
        "execution_date": context["execution_date"].isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "report_date": summary.get("report_date"),
        "total_transactions": summary.get("total_transactions", 0),
        "fraud_transactions": summary.get("fraud_transactions", 0),
        "fraud_rate_pct": summary.get("fraud_rate_pct", 0),
        "emails_sent": digest_result.get("emails_sent", 0),
    }

    producer = KafkaProducer({"bootstrap.servers": settings.kafka_bootstrap_servers})
    producer.produce(
        TOPIC_METRICS,
        key="reporting_pipeline",
        value=json.dumps(metrics).encode("utf-8"),
    )
    producer.flush(timeout=10)

    logger.info("Reporting metrics published")
    return metrics


def _on_failure_callback(context: dict[str, Any]) -> None:
    """Alert on task failure in the reporting DAG."""
    from src.alerting.alert_manager import AlertManager

    dag_id = context.get("dag", {})
    task_id = context.get("task_instance", {})
    exception = context.get("exception")

    logger.error(
        "Reporting DAG task failed",
        dag_id=str(dag_id),
        task_id=str(task_id),
        error=str(exception),
    )

    try:
        alert_manager = AlertManager()
        alert_manager.generate_alert(
            scoring_result={
                "score": 0.0,
                "risk_level": "high",
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
    dag_id="reporting_pipeline",
    description="Daily fraud summary reporting, Snowflake reporting schema, and digest email",
    schedule_interval="0 6 * * *",  # Daily at 06:00 UTC
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["reporting", "daily", "summary", "email", "riskpulse"],
    default_args={**default_args, "on_failure_callback": _on_failure_callback},
) as dag:

    wait_for_snowflake = ExternalTaskSensor(
        task_id="wait_for_snowflake_load",
        external_dag_id="snowflake_load",
        external_task_id="publish_load_metrics",
        mode="reschedule",
        poke_interval=300,
        timeout=3600,
        allowed_states=["success"],
        # Look at the most recent successful snowflake_load run
        execution_delta=timedelta(hours=1),
    )

    generate_summary = PythonOperator(
        task_id="generate_daily_summary",
        python_callable=_generate_daily_summary,
    )

    populate_reporting = PythonOperator(
        task_id="populate_reporting_schema",
        python_callable=_populate_reporting_schema,
    )

    send_digest = PythonOperator(
        task_id="send_daily_digest",
        python_callable=_send_daily_digest,
    )

    publish_metrics = PythonOperator(
        task_id="publish_report_metrics",
        python_callable=_publish_report_metrics,
    )

    (wait_for_snowflake >> generate_summary >> populate_reporting >> send_digest >> publish_metrics)
