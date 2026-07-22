"""Transformation DAG — cleaning, normalization, and feature engineering.

Triggered after validated data is available.  Runs the RiskPulse
transformation pipeline stages (clean → normalize → features) and
writes processed output to S3 / the data warehouse.

Schedule: every 30 minutes (runs after ingestion lands validated data)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor

from src.storage.s3_handler import S3Handler, StorageLayer
from src.transformation.cleaner import DataCleaner
from src.transformation.normalizer import DataNormalizer, get_normalizer
from src.transformation.feature_engineer import FeatureEngineer
from src.utils.config import get_settings
from src.utils.constants import TOPIC_VALIDATED
from src.utils.logger import get_logger

logger = get_logger(__name__, component="transformation_dag")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SENSOR_POKE_INTERVAL = 120
_SENSOR_TIMEOUT = 900

default_args = {
    "owner": "riskpulse",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=15),
    "execution_timeout": timedelta(minutes=45),
}


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _check_validated_data(**context: Any) -> bool:
    """Return True when new validated data is available for processing.

    Checks S3 for validated-data partitions created since the last
    successful DAG run.
    """
    import boto3

    settings = get_settings()
    s3 = boto3.client("s3")
    bucket = "riskpulse-raw"
    prefix = "validated/"

    execution_date: datetime = context["execution_date"]
    partition = execution_date.strftime("year=%Y/month=%m/day=%d/")

    try:
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{prefix}{partition}",
            MaxKeys=1,
        )
        has_data = response.get("KeyCount", 0) > 0
        logger.info(
            "Validated data check",
            partition=partition,
            has_data=has_data,
        )
        return has_data
    except Exception as exc:
        logger.warning("Failed to check validated data, proceeding", error=str(exc))
        return True


def _run_cleaning(**context: Any) -> dict[str, Any]:
    """Run the data cleaning pipeline on validated records."""
    ti = context["ti"]
    handler = S3Handler()

    execution_date: datetime = context["execution_date"]
    partition = execution_date.strftime("year=%Y/month=%m/day=%d/hour=%H")

    records = handler.read_batch(
        storage_layer=StorageLayer.RAW,
        partition_path=f"validated/{partition}",
    )

    if not records:
        logger.info("No validated records to clean")
        return {"records_in": 0, "records_out": 0, "duplicates": 0, "changes": 0}

    cleaner = DataCleaner()
    cleaned: list[dict[str, Any]] = []
    duplicates = 0
    total_changes = 0

    for record in records:
        result = cleaner.clean(record)
        if result.is_duplicate:
            duplicates += 1
            continue
        cleaned.append(result.record)
        total_changes += len(result.changes)

    ti.xcom_push(key="cleaned_records", value=len(cleaned))

    # Persist intermediate cleaned data
    handler.write_batch(
        records=cleaned,
        storage_layer=StorageLayer.RAW,
        partition_path=f"cleaned/{partition}",
    )

    summary = {
        "records_in": len(records),
        "records_out": len(cleaned),
        "duplicates": duplicates,
        "total_changes": total_changes,
    }
    logger.info("Cleaning stage complete", **summary)
    return summary


def _run_normalization(**context: Any) -> dict[str, Any]:
    """Normalize cleaned records (currency conversion, field standardisation)."""
    ti = context["ti"]
    handler = S3Handler()

    execution_date: datetime = context["execution_date"]
    partition = execution_date.strftime("year=%Y/month=%m/day=%d/hour=%H")

    records = handler.read_batch(
        storage_layer=StorageLayer.RAW,
        partition_path=f"cleaned/{partition}",
    )

    if not records:
        logger.info("No cleaned records to normalize")
        return {"records_in": 0, "records_out": 0, "conversions": 0}

    normalizer = get_normalizer()
    normalized: list[dict[str, Any]] = []
    conversions = 0

    for record in records:
        result = normalizer.normalize(record)
        normalized.append(result.record)
        conversions += len(result.changes)

    handler.write_batch(
        records=normalized,
        storage_layer=StorageLayer.RAW,
        partition_path=f"normalized/{partition}",
    )

    summary = {
        "records_in": len(records),
        "records_out": len(normalized),
        "conversions": conversions,
    }
    logger.info("Normalization stage complete", **summary)
    return summary


def _run_feature_engineering(**context: Any) -> dict[str, Any]:
    """Compute derived features for each normalised transaction."""
    ti = context["ti"]
    handler = S3Handler()

    execution_date: datetime = context["execution_date"]
    partition = execution_date.strftime("year=%Y/month=%m/day=%d/hour=%H")

    records = handler.read_batch(
        storage_layer=StorageLayer.RAW,
        partition_path=f"normalized/{partition}",
    )

    if not records:
        logger.info("No normalized records for feature engineering")
        return {"records_in": 0, "records_out": 0, "features_computed": 0}

    engineer = FeatureEngineer()
    enriched: list[dict[str, Any]] = []
    features_computed = 0

    for record in records:
        result = engineer.compute_features(record)
        if result.is_success:
            combined = {**record, **result.features}
            enriched.append(combined)
            features_computed += len(result.features)
        else:
            logger.warning(
                "Feature engineering failed for record",
                error=result.error,
                transaction_id=record.get("transaction_id"),
            )
            # Still include record without extra features
            enriched.append(record)

    handler.write_batch(
        records=enriched,
        storage_layer=StorageLayer.RAW,
        partition_path=f"features/{partition}",
    )

    summary = {
        "records_in": len(records),
        "records_out": len(enriched),
        "features_computed": features_computed,
    }
    logger.info("Feature engineering complete", **summary)
    return summary


def _write_processed_data(**context: Any) -> dict[str, Any]:
    """Write final processed data to the processed S3 bucket."""
    ti = context["ti"]
    handler = S3Handler()

    execution_date: datetime = context["execution_date"]
    partition = execution_date.strftime("year=%Y/month=%m/day=%d/hour=%H")

    # Read feature-enriched records
    records = handler.read_batch(
        storage_layer=StorageLayer.RAW,
        partition_path=f"features/{partition}",
    )

    if not records:
        logger.info("No processed records to write")
        return {"files_written": 0, "record_count": 0}

    file_key = handler.write_batch(
        records=records,
        storage_layer=StorageLayer.PROCESSED,
        partition_path=partition,
        metadata={
            "dag_id": context["dag"].dag_id,
            "run_id": context["run_id"],
            "record_count": len(records),
        },
    )

    result = {
        "files_written": 1,
        "s3_key": file_key,
        "partition": partition,
        "record_count": len(records),
    }
    logger.info("Processed data written", **result)
    return result


def _on_failure_callback(context: dict[str, Any]) -> None:
    """Alert on task failure in the transformation DAG."""
    from src.alerting.alert_manager import AlertManager

    dag_id = context.get("dag", {})
    task_id = context.get("task_instance", {})
    exception = context.get("exception")

    logger.error(
        "Transformation DAG task failed",
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
    dag_id="transformation_pipeline",
    description="Clean, normalize, and feature-engineer validated transaction data",
    schedule_interval=timedelta(minutes=30),
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["transformation", "cleaning", "normalization", "features", "riskpulse"],
    default_args={**default_args, "on_failure_callback": _on_failure_callback},
) as dag:

    sensor = PythonSensor(
        task_id="check_validated_data",
        python_callable=_check_validated_data,
        poke_interval=_SENSOR_POKE_INTERVAL,
        timeout=_SENSOR_TIMEOUT,
        mode="reschedule",
        soft_fail=True,
    )

    clean = PythonOperator(
        task_id="run_cleaning",
        python_callable=_run_cleaning,
    )

    normalize = PythonOperator(
        task_id="run_normalization",
        python_callable=_run_normalization,
    )

    features = PythonOperator(
        task_id="run_feature_engineering",
        python_callable=_run_feature_engineering,
    )

    write = PythonOperator(
        task_id="write_processed_data",
        python_callable=_write_processed_data,
    )

    sensor >> clean >> normalize >> features >> write
