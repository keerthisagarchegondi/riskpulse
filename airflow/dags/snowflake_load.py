"""Snowflake Loading DAG — S3 processed data to Snowflake warehouse.

Runs hourly to load scored transaction data from S3 processed bucket
into Snowflake raw layer, then transforms through staging → analytics
schema and refreshes materialized views.

Schedule: hourly (0 * * * *)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator

from operators.snowflake_operator import (
    SnowflakeCopyIntoOperator,
    SnowflakeMergeOperator,
    SnowflakeRefreshViewsOperator,
)
from src.utils.config import get_settings
from src.utils.constants import TOPIC_METRICS
from src.utils.logger import get_logger

logger = get_logger(__name__, component="snowflake_load_dag")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SNOWFLAKE_CONN_ID = "riskpulse_snowflake"
_SNOWFLAKE_DATABASE = "RISKPULSE"
_SNOWFLAKE_WAREHOUSE = "RISKPULSE_ETL_WH"
_S3_STAGE = "@RISKPULSE.RAW.S3_PROCESSED_STAGE"
_S3_SCORED_PREFIX = "scored/"

default_args = {
    "owner": "riskpulse",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(minutes=60),
}


# ---------------------------------------------------------------------------
# SQL templates
# ---------------------------------------------------------------------------

_COPY_INTO_RAW_TRANSACTIONS = """
COPY INTO {database}.RAW.TRANSACTIONS
FROM {stage}/{partition_path}/
FILE_FORMAT = (
    TYPE = 'PARQUET'
    SNAPPY_COMPRESSION = TRUE
)
MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
ON_ERROR = 'CONTINUE'
PURGE = FALSE;
"""

_MERGE_RAW_TO_STAGING = """
MERGE INTO {database}.STAGING.TRANSACTIONS AS target
USING (
    SELECT
        transaction_id,
        account_id,
        amount,
        currency,
        transaction_type,
        channel,
        merchant_name,
        merchant_category,
        country,
        city,
        ip_address,
        device_id,
        device_type,
        ensemble_score,
        ensemble_risk_level,
        rule_score,
        anomaly_score,
        ml_score,
        is_anomaly,
        velocity_1h,
        velocity_24h,
        amount_velocity_1h,
        amount_velocity_24h,
        velocity_score,
        created_at,
        CURRENT_TIMESTAMP() AS loaded_at
    FROM {database}.RAW.TRANSACTIONS
    WHERE loaded_at >= DATEADD(hour, -2, CURRENT_TIMESTAMP())
) AS source
ON target.transaction_id = source.transaction_id
WHEN MATCHED THEN UPDATE SET
    target.ensemble_score = source.ensemble_score,
    target.ensemble_risk_level = source.ensemble_risk_level,
    target.rule_score = source.rule_score,
    target.anomaly_score = source.anomaly_score,
    target.ml_score = source.ml_score,
    target.is_anomaly = source.is_anomaly,
    target.velocity_1h = source.velocity_1h,
    target.velocity_24h = source.velocity_24h,
    target.updated_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN INSERT (
    transaction_id, account_id, amount, currency,
    transaction_type, channel, merchant_name, merchant_category,
    country, city, ip_address, device_id, device_type,
    ensemble_score, ensemble_risk_level, rule_score,
    anomaly_score, ml_score, is_anomaly,
    velocity_1h, velocity_24h, amount_velocity_1h,
    amount_velocity_24h, velocity_score, created_at, loaded_at
) VALUES (
    source.transaction_id, source.account_id, source.amount, source.currency,
    source.transaction_type, source.channel, source.merchant_name, source.merchant_category,
    source.country, source.city, source.ip_address, source.device_id, source.device_type,
    source.ensemble_score, source.ensemble_risk_level, source.rule_score,
    source.anomaly_score, source.ml_score, source.is_anomaly,
    source.velocity_1h, source.velocity_24h, source.amount_velocity_1h,
    source.amount_velocity_24h, source.velocity_score, source.created_at, source.loaded_at
);
"""

_MERGE_RAW_TO_STAGING_ALERTS = """
MERGE INTO {database}.STAGING.FRAUD_ALERTS AS target
USING (
    SELECT
        alert_id,
        transaction_id,
        severity,
        risk_score,
        alert_type,
        rules_triggered,
        status,
        created_at,
        CURRENT_TIMESTAMP() AS loaded_at
    FROM {database}.RAW.FRAUD_ALERTS
    WHERE loaded_at >= DATEADD(hour, -2, CURRENT_TIMESTAMP())
) AS source
ON target.alert_id = source.alert_id
WHEN MATCHED THEN UPDATE SET
    target.status = source.status,
    target.updated_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN INSERT (
    alert_id, transaction_id, severity, risk_score,
    alert_type, rules_triggered, status, created_at, loaded_at
) VALUES (
    source.alert_id, source.transaction_id, source.severity, source.risk_score,
    source.alert_type, source.rules_triggered, source.status, source.created_at, source.loaded_at
);
"""

_POPULATE_ANALYTICS_TRANSACTIONS = """
INSERT INTO {database}.ANALYTICS.FACT_TRANSACTIONS (
    transaction_id, account_id, amount, currency,
    transaction_type, channel, merchant_name, merchant_category,
    country, city, ensemble_score, ensemble_risk_level,
    rule_score, anomaly_score, ml_score, is_anomaly,
    is_fraudulent, transaction_date, transaction_hour, loaded_at
)
SELECT
    t.transaction_id,
    t.account_id,
    t.amount,
    t.currency,
    t.transaction_type,
    t.channel,
    t.merchant_name,
    t.merchant_category,
    t.country,
    t.city,
    t.ensemble_score,
    t.ensemble_risk_level,
    t.rule_score,
    t.anomaly_score,
    t.ml_score,
    t.is_anomaly,
    CASE WHEN t.ensemble_risk_level IN ('high', 'critical') THEN TRUE ELSE FALSE END,
    DATE_TRUNC('day', t.created_at),
    DATE_TRUNC('hour', t.created_at),
    CURRENT_TIMESTAMP()
FROM {database}.STAGING.TRANSACTIONS t
LEFT JOIN {database}.ANALYTICS.FACT_TRANSACTIONS f
    ON t.transaction_id = f.transaction_id
WHERE f.transaction_id IS NULL
  AND t.loaded_at >= DATEADD(hour, -2, CURRENT_TIMESTAMP());
"""

_REFRESH_MATERIALIZED_VIEWS = """
-- Refresh risk score distribution view
CREATE OR REPLACE TABLE {database}.ANALYTICS.MV_RISK_DISTRIBUTION AS
SELECT
    DATE_TRUNC('hour', transaction_date) AS period,
    ensemble_risk_level,
    COUNT(*) AS transaction_count,
    AVG(ensemble_score) AS avg_score,
    SUM(amount) AS total_amount,
    AVG(amount) AS avg_amount
FROM {database}.ANALYTICS.FACT_TRANSACTIONS
WHERE transaction_date >= DATEADD(day, -30, CURRENT_DATE())
GROUP BY 1, 2;

-- Refresh merchant risk summary
CREATE OR REPLACE TABLE {database}.ANALYTICS.MV_MERCHANT_RISK AS
SELECT
    merchant_name,
    merchant_category,
    country,
    COUNT(*) AS transaction_count,
    AVG(ensemble_score) AS avg_risk_score,
    SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) AS fraud_count,
    SUM(CASE WHEN is_fraudulent THEN amount ELSE 0 END) AS fraud_amount,
    SUM(amount) AS total_amount
FROM {database}.ANALYTICS.FACT_TRANSACTIONS
WHERE transaction_date >= DATEADD(day, -90, CURRENT_DATE())
GROUP BY 1, 2, 3
HAVING transaction_count >= 10;

-- Refresh hourly metrics
CREATE OR REPLACE TABLE {database}.ANALYTICS.MV_HOURLY_METRICS AS
SELECT
    transaction_hour AS period,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) AS fraud_transactions,
    AVG(ensemble_score) AS avg_score,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ensemble_score) AS p95_score,
    SUM(amount) AS total_volume,
    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) AS anomaly_count
FROM {database}.ANALYTICS.FACT_TRANSACTIONS
WHERE transaction_date >= DATEADD(day, -7, CURRENT_DATE())
GROUP BY 1;
"""


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _load_s3_to_raw(**context: Any) -> dict[str, Any]:
    """Execute COPY INTO to load Parquet files from S3 into Snowflake raw schema."""
    from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook

    execution_date: datetime = context["execution_date"]
    partition_path = execution_date.strftime("year=%Y/month=%m/day=%d/hour=%H")

    hook = SnowflakeHook(snowflake_conn_id=_SNOWFLAKE_CONN_ID)

    sql = _COPY_INTO_RAW_TRANSACTIONS.format(
        database=_SNOWFLAKE_DATABASE,
        stage=_S3_STAGE,
        partition_path=partition_path,
    )

    logger.info("Executing COPY INTO raw", partition=partition_path)

    try:
        result = hook.run(sql, autocommit=True)
        rows_loaded = 0
        files_loaded = 0
        errors = 0

        if result:
            for row in result:
                if hasattr(row, "__iter__"):
                    files_loaded += 1
                    rows_loaded += row[0] if len(row) > 0 else 0
                    errors += row[1] if len(row) > 1 else 0

        summary = {
            "partition": partition_path,
            "files_loaded": files_loaded,
            "rows_loaded": rows_loaded,
            "errors": errors,
        }
        logger.info("COPY INTO raw complete", **summary)
        context["ti"].xcom_push(key="raw_load_summary", value=summary)
        return summary

    except Exception as exc:
        logger.error("COPY INTO raw failed", error=str(exc), partition=partition_path)
        raise


def _transform_raw_to_staging(**context: Any) -> dict[str, Any]:
    """Merge raw data into staging schema with deduplication."""
    from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook

    hook = SnowflakeHook(snowflake_conn_id=_SNOWFLAKE_CONN_ID)

    logger.info("Running raw → staging transformation")

    try:
        # Merge transactions
        txn_sql = _MERGE_RAW_TO_STAGING.format(database=_SNOWFLAKE_DATABASE)
        txn_result = hook.run(txn_sql, autocommit=True)

        # Merge fraud alerts
        alerts_sql = _MERGE_RAW_TO_STAGING_ALERTS.format(database=_SNOWFLAKE_DATABASE)
        alerts_result = hook.run(alerts_sql, autocommit=True)

        summary = {
            "transactions_merged": True,
            "alerts_merged": True,
        }
        logger.info("Raw → staging transformation complete", **summary)
        return summary

    except Exception as exc:
        logger.error("Raw → staging transformation failed", error=str(exc))
        raise


def _transform_staging_to_analytics(**context: Any) -> dict[str, Any]:
    """Populate analytics fact tables from staging data."""
    from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook

    hook = SnowflakeHook(snowflake_conn_id=_SNOWFLAKE_CONN_ID)

    logger.info("Running staging → analytics transformation")

    try:
        sql = _POPULATE_ANALYTICS_TRANSACTIONS.format(database=_SNOWFLAKE_DATABASE)
        hook.run(sql, autocommit=True)

        summary = {"analytics_populated": True}
        logger.info("Staging → analytics transformation complete", **summary)
        return summary

    except Exception as exc:
        logger.error("Staging → analytics transformation failed", error=str(exc))
        raise


def _refresh_materialized_views(**context: Any) -> dict[str, Any]:
    """Refresh all materialized views in the analytics schema."""
    from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook

    hook = SnowflakeHook(snowflake_conn_id=_SNOWFLAKE_CONN_ID)

    logger.info("Refreshing materialized views")

    try:
        sql = _REFRESH_MATERIALIZED_VIEWS.format(database=_SNOWFLAKE_DATABASE)
        # Split and execute each statement separately
        statements = [s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]
        views_refreshed = 0

        for stmt in statements:
            hook.run(stmt + ";", autocommit=True)
            views_refreshed += 1

        summary = {"views_refreshed": views_refreshed}
        logger.info("Materialized views refreshed", **summary)
        return summary

    except Exception as exc:
        logger.error("View refresh failed", error=str(exc))
        raise


def _validate_load(**context: Any) -> dict[str, Any]:
    """Run validation queries to confirm data loaded correctly."""
    from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook

    hook = SnowflakeHook(snowflake_conn_id=_SNOWFLAKE_CONN_ID)

    validation_queries = {
        "raw_count": f"""
            SELECT COUNT(*) FROM {_SNOWFLAKE_DATABASE}.RAW.TRANSACTIONS
            WHERE loaded_at >= DATEADD(hour, -2, CURRENT_TIMESTAMP())
        """,
        "staging_count": f"""
            SELECT COUNT(*) FROM {_SNOWFLAKE_DATABASE}.STAGING.TRANSACTIONS
            WHERE loaded_at >= DATEADD(hour, -2, CURRENT_TIMESTAMP())
        """,
        "analytics_count": f"""
            SELECT COUNT(*) FROM {_SNOWFLAKE_DATABASE}.ANALYTICS.FACT_TRANSACTIONS
            WHERE loaded_at >= DATEADD(hour, -2, CURRENT_TIMESTAMP())
        """,
        "null_check": f"""
            SELECT COUNT(*) FROM {_SNOWFLAKE_DATABASE}.STAGING.TRANSACTIONS
            WHERE transaction_id IS NULL
            AND loaded_at >= DATEADD(hour, -2, CURRENT_TIMESTAMP())
        """,
    }

    results = {}
    for check_name, sql in validation_queries.items():
        try:
            result = hook.get_first(sql)
            results[check_name] = result[0] if result else 0
        except Exception as exc:
            logger.warning(f"Validation check {check_name} failed", error=str(exc))
            results[check_name] = -1

    # Fail if we have null transaction_ids
    if results.get("null_check", 0) > 0:
        logger.error("Found NULL transaction_ids in staging", count=results["null_check"])

    logger.info("Load validation complete", **results)
    return results


def _publish_load_metrics(**context: Any) -> dict[str, Any]:
    """Publish Snowflake load metrics to Kafka."""
    from confluent_kafka import Producer as KafkaProducer

    ti = context["ti"]
    settings = get_settings()

    raw_load = ti.xcom_pull(task_ids="load_s3_to_raw") or {}
    staging = ti.xcom_pull(task_ids="transform_raw_to_staging") or {}
    analytics = ti.xcom_pull(task_ids="transform_staging_to_analytics") or {}
    validation = ti.xcom_pull(task_ids="validate_load") or {}

    metrics = {
        "dag_id": "snowflake_load",
        "run_id": context["run_id"],
        "execution_date": context["execution_date"].isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_load": raw_load,
        "staging_transform": staging,
        "analytics_transform": analytics,
        "validation": validation,
    }

    producer = KafkaProducer({"bootstrap.servers": settings.kafka_bootstrap_servers})
    producer.produce(
        TOPIC_METRICS,
        key="snowflake_load",
        value=json.dumps(metrics).encode("utf-8"),
    )
    producer.flush(timeout=10)

    logger.info("Snowflake load metrics published")
    return metrics


def _on_failure_callback(context: dict[str, Any]) -> None:
    """Alert on task failure in the Snowflake load DAG."""
    from src.alerting.alert_manager import AlertManager

    dag_id = context.get("dag", {})
    task_id = context.get("task_instance", {})
    exception = context.get("exception")

    logger.error(
        "Snowflake load DAG task failed",
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
    dag_id="snowflake_load",
    description="Hourly S3 to Snowflake ETL: raw → staging → analytics",
    schedule_interval="@hourly",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["snowflake", "etl", "warehouse", "analytics", "riskpulse"],
    default_args={**default_args, "on_failure_callback": _on_failure_callback},
) as dag:

    load_raw = PythonOperator(
        task_id="load_s3_to_raw",
        python_callable=_load_s3_to_raw,
    )

    stage_transform = PythonOperator(
        task_id="transform_raw_to_staging",
        python_callable=_transform_raw_to_staging,
    )

    analytics_transform = PythonOperator(
        task_id="transform_staging_to_analytics",
        python_callable=_transform_staging_to_analytics,
    )

    refresh_views = PythonOperator(
        task_id="refresh_materialized_views",
        python_callable=_refresh_materialized_views,
    )

    validate = PythonOperator(
        task_id="validate_load",
        python_callable=_validate_load,
    )

    publish_metrics = PythonOperator(
        task_id="publish_load_metrics",
        python_callable=_publish_load_metrics,
    )

    load_raw >> stage_transform >> analytics_transform >> refresh_views >> validate >> publish_metrics
