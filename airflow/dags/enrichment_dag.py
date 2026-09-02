"""Enrichment DAG — geo, device, merchant enrichment and profile updates.

Triggered by transformation pipeline completion via TriggerDagRunOperator.
Runs the full enrichment pipeline (geo → device → merchant → velocity)
and updates customer profiles before writing enriched data to processed S3.

Schedule: None (externally triggered)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor

from airflow import DAG
from src.enrichment import (
    DeviceEnricher,
    GeoEnricher,
    MerchantEnricher,
    VelocityCalculator,
)
from src.storage.s3_handler import StorageLayer, get_s3_handler
from src.utils.config import get_settings
from src.utils.constants import TOPIC_METRICS
from src.utils.logger import get_logger

logger = get_logger(__name__, component="enrichment_dag")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ENRICHMENT_BATCH_SIZE = 1000
_PROFILE_UPDATE_BATCH_SIZE = 500

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


def _load_processed_data(**context: Any) -> dict[str, Any]:
    """Load feature-engineered data from S3 processed layer."""
    handler = get_s3_handler()

    execution_date: datetime = context["execution_date"]
    partition = execution_date.strftime("year=%Y/month=%m/day=%d/hour=%H")

    records = handler.read_batch(
        storage_layer=StorageLayer.PROCESSED,
        partition_path=partition,
    )

    if not records:
        logger.info("No processed records available for enrichment")
        context["ti"].xcom_push(key="record_count", value=0)
        return {"records_loaded": 0}

    context["ti"].xcom_push(key="record_count", value=len(records))
    context["ti"].xcom_push(key="partition", value=partition)

    logger.info("Loaded processed records for enrichment", count=len(records))
    return {"records_loaded": len(records), "partition": partition}


def _run_geo_enrichment(**context: Any) -> dict[str, Any]:
    """Enrich transactions with geolocation data — country risk, distance, IP geo."""
    ti = context["ti"]
    record_count = ti.xcom_pull(task_ids="load_processed_data", key="record_count")

    if not record_count:
        return {"records_enriched": 0, "geo_hits": 0}

    handler = get_s3_handler()
    partition = ti.xcom_pull(task_ids="load_processed_data", key="partition")

    records = handler.read_batch(
        storage_layer=StorageLayer.PROCESSED,
        partition_path=partition,
    )

    geo_enricher = GeoEnricher()
    enriched: list[dict[str, Any]] = []
    geo_hits = 0
    geo_failures = 0

    for record in records:
        try:
            result = geo_enricher.enrich(record)
            enriched_record = {**record, **result.to_dict()}
            enriched.append(enriched_record)
            if result.country_code:
                geo_hits += 1
        except Exception as exc:
            logger.warning(
                "Geo enrichment failed for record",
                transaction_id=record.get("transaction_id"),
                error=str(exc),
            )
            enriched.append(record)
            geo_failures += 1

    handler.write_batch(
        records=enriched,
        storage_layer=StorageLayer.RAW,
        partition_path=f"enriched/geo/{partition}",
    )

    summary = {
        "records_enriched": len(enriched),
        "geo_hits": geo_hits,
        "geo_failures": geo_failures,
    }
    logger.info("Geo enrichment complete", **summary)
    return summary


def _run_device_enrichment(**context: Any) -> dict[str, Any]:
    """Enrich transactions with device fingerprinting and trust scores."""
    ti = context["ti"]
    record_count = ti.xcom_pull(task_ids="load_processed_data", key="record_count")

    if not record_count:
        return {"records_enriched": 0, "device_matches": 0}

    handler = get_s3_handler()
    partition = ti.xcom_pull(task_ids="load_processed_data", key="partition")

    records = handler.read_batch(
        storage_layer=StorageLayer.RAW,
        partition_path=f"enriched/geo/{partition}",
    )

    device_enricher = DeviceEnricher()
    enriched: list[dict[str, Any]] = []
    device_matches = 0
    new_devices = 0

    for record in records:
        try:
            result = device_enricher.enrich(record)
            enriched_record = {**record, **result}
            enriched.append(enriched_record)
            if result.get("device_known", False):
                device_matches += 1
            else:
                new_devices += 1
        except Exception as exc:
            logger.warning(
                "Device enrichment failed",
                transaction_id=record.get("transaction_id"),
                error=str(exc),
            )
            enriched.append(record)

    handler.write_batch(
        records=enriched,
        storage_layer=StorageLayer.RAW,
        partition_path=f"enriched/device/{partition}",
    )

    summary = {
        "records_enriched": len(enriched),
        "device_matches": device_matches,
        "new_devices": new_devices,
    }
    logger.info("Device enrichment complete", **summary)
    return summary


def _run_merchant_enrichment(**context: Any) -> dict[str, Any]:
    """Enrich transactions with merchant risk category and trust data."""
    ti = context["ti"]
    record_count = ti.xcom_pull(task_ids="load_processed_data", key="record_count")

    if not record_count:
        return {"records_enriched": 0, "merchant_matches": 0}

    handler = get_s3_handler()
    partition = ti.xcom_pull(task_ids="load_processed_data", key="partition")

    records = handler.read_batch(
        storage_layer=StorageLayer.RAW,
        partition_path=f"enriched/device/{partition}",
    )

    merchant_enricher = MerchantEnricher()
    enriched: list[dict[str, Any]] = []
    merchant_matches = 0

    for record in records:
        try:
            result = merchant_enricher.enrich(record)
            enriched_record = {**record, **result}
            enriched.append(enriched_record)
            if result.get("merchant_category"):
                merchant_matches += 1
        except Exception as exc:
            logger.warning(
                "Merchant enrichment failed",
                transaction_id=record.get("transaction_id"),
                error=str(exc),
            )
            enriched.append(record)

    handler.write_batch(
        records=enriched,
        storage_layer=StorageLayer.RAW,
        partition_path=f"enriched/merchant/{partition}",
    )

    summary = {
        "records_enriched": len(enriched),
        "merchant_matches": merchant_matches,
    }
    logger.info("Merchant enrichment complete", **summary)
    return summary


def _compute_velocity(**context: Any) -> dict[str, Any]:
    """Compute velocity features — transaction frequency, amount patterns."""
    ti = context["ti"]
    record_count = ti.xcom_pull(task_ids="load_processed_data", key="record_count")

    if not record_count:
        return {"records_processed": 0, "velocity_alerts": 0}

    handler = get_s3_handler()
    partition = ti.xcom_pull(task_ids="load_processed_data", key="partition")

    records = handler.read_batch(
        storage_layer=StorageLayer.RAW,
        partition_path=f"enriched/merchant/{partition}",
    )

    velocity_calc = VelocityCalculator()
    enriched: list[dict[str, Any]] = []
    velocity_alerts = 0

    for record in records:
        try:
            result = velocity_calc.calculate(record)
            enriched_record = {
                **record,
                "velocity_1h": result.count_1h,
                "velocity_24h": result.count_24h,
                "amount_velocity_1h": result.amount_1h,
                "amount_velocity_24h": result.amount_24h,
                "velocity_score": result.risk_score,
            }
            enriched.append(enriched_record)
            if result.risk_score > 0.7:
                velocity_alerts += 1
        except Exception as exc:
            logger.warning(
                "Velocity calculation failed",
                transaction_id=record.get("transaction_id"),
                error=str(exc),
            )
            enriched.append(record)

    handler.write_batch(
        records=enriched,
        storage_layer=StorageLayer.RAW,
        partition_path=f"enriched/velocity/{partition}",
    )

    summary = {
        "records_processed": len(enriched),
        "velocity_alerts": velocity_alerts,
    }
    logger.info("Velocity computation complete", **summary)
    return summary


def _update_customer_profiles(**context: Any) -> dict[str, Any]:
    """Update customer profile aggregates from enriched transaction data."""
    ti = context["ti"]
    record_count = ti.xcom_pull(task_ids="load_processed_data", key="record_count")

    if not record_count:
        return {"profiles_updated": 0}

    handler = get_s3_handler()
    partition = ti.xcom_pull(task_ids="load_processed_data", key="partition")
    get_settings()

    records = handler.read_batch(
        storage_layer=StorageLayer.RAW,
        partition_path=f"enriched/velocity/{partition}",
    )

    # Group records by account_id for profile updates
    profiles: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        account_id = record.get("account_id")
        if account_id:
            profiles.setdefault(account_id, []).append(record)

    profiles_updated = 0
    for account_id, account_records in profiles.items():
        try:
            # Compute profile aggregates
            total_amount = sum(r.get("amount", 0) for r in account_records)
            avg_amount = total_amount / len(account_records) if account_records else 0
            max_velocity = max((r.get("velocity_score", 0) for r in account_records), default=0)
            unique_merchants = len(
                {r.get("merchant_name") for r in account_records if r.get("merchant_name")}
            )
            unique_countries = len({r.get("country") for r in account_records if r.get("country")})

            profile_update = {
                "account_id": account_id,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "batch_transaction_count": len(account_records),
                "batch_total_amount": total_amount,
                "batch_avg_amount": avg_amount,
                "batch_max_velocity_score": max_velocity,
                "batch_unique_merchants": unique_merchants,
                "batch_unique_countries": unique_countries,
            }

            handler.write_batch(
                records=[profile_update],
                storage_layer=StorageLayer.RAW,
                partition_path=f"profiles/{account_id}/{partition}",
            )
            profiles_updated += 1

        except Exception as exc:
            logger.warning(
                "Profile update failed",
                account_id=account_id,
                error=str(exc),
            )

    summary = {"profiles_updated": profiles_updated, "total_accounts": len(profiles)}
    logger.info("Customer profiles updated", **summary)
    return summary


def _write_enriched_data(**context: Any) -> dict[str, Any]:
    """Write final enriched data to processed S3 for downstream consumption."""
    ti = context["ti"]
    record_count = ti.xcom_pull(task_ids="load_processed_data", key="record_count")

    if not record_count:
        return {"files_written": 0, "record_count": 0}

    handler = get_s3_handler()
    partition = ti.xcom_pull(task_ids="load_processed_data", key="partition")

    records = handler.read_batch(
        storage_layer=StorageLayer.RAW,
        partition_path=f"enriched/velocity/{partition}",
    )

    file_key = handler.write_batch(
        records=records,
        storage_layer=StorageLayer.ENRICHED,
        partition_path=partition,
        metadata={
            "dag_id": context["dag"].dag_id,
            "run_id": context["run_id"],
            "record_count": len(records),
            "enrichment_stages": ["geo", "device", "merchant", "velocity"],
        },
    )

    result = {
        "files_written": 1,
        "s3_key": file_key,
        "partition": partition,
        "record_count": len(records),
    }
    logger.info("Enriched data written to S3", **result)
    return result


def _publish_enrichment_metrics(**context: Any) -> dict[str, Any]:
    """Publish enrichment pipeline metrics."""
    from confluent_kafka import Producer as KafkaProducer

    ti = context["ti"]
    settings = get_settings()

    geo_result = ti.xcom_pull(task_ids="run_geo_enrichment") or {}
    device_result = ti.xcom_pull(task_ids="run_device_enrichment") or {}
    merchant_result = ti.xcom_pull(task_ids="run_merchant_enrichment") or {}
    velocity_result = ti.xcom_pull(task_ids="compute_velocity") or {}
    profile_result = ti.xcom_pull(task_ids="update_customer_profiles") or {}

    metrics = {
        "dag_id": "enrichment_pipeline",
        "run_id": context["run_id"],
        "execution_date": context["execution_date"].isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "geo_enrichment": geo_result,
        "device_enrichment": device_result,
        "merchant_enrichment": merchant_result,
        "velocity_computation": velocity_result,
        "profile_updates": profile_result,
    }

    producer = KafkaProducer({"bootstrap.servers": settings.kafka_bootstrap_servers})
    producer.produce(
        TOPIC_METRICS,
        key="enrichment_pipeline",
        value=json.dumps(metrics).encode("utf-8"),
    )
    producer.flush(timeout=10)

    logger.info("Enrichment metrics published")
    return metrics


def _on_failure_callback(context: dict[str, Any]) -> None:
    """Alert on task failure in the enrichment DAG."""
    from src.alerting.alert_manager import AlertManager

    dag_id = context.get("dag", {})
    task_id = context.get("task_instance", {})
    exception = context.get("exception")

    logger.error(
        "Enrichment DAG task failed",
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
    dag_id="enrichment_pipeline",
    description="Geo, device, merchant enrichment and customer profile updates",
    schedule_interval=None,  # Triggered by transformation_pipeline
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=2,
    tags=["enrichment", "geo", "device", "merchant", "velocity", "riskpulse"],
    default_args={**default_args, "on_failure_callback": _on_failure_callback},
) as dag:

    wait_for_transformation = ExternalTaskSensor(
        task_id="wait_for_transformation",
        external_dag_id="transformation_pipeline",
        external_task_id="write_processed_data",
        mode="reschedule",
        poke_interval=120,
        timeout=1800,
        allowed_states=["success"],
    )

    load_data = PythonOperator(
        task_id="load_processed_data",
        python_callable=_load_processed_data,
    )

    geo_enrich = PythonOperator(
        task_id="run_geo_enrichment",
        python_callable=_run_geo_enrichment,
    )

    device_enrich = PythonOperator(
        task_id="run_device_enrichment",
        python_callable=_run_device_enrichment,
    )

    merchant_enrich = PythonOperator(
        task_id="run_merchant_enrichment",
        python_callable=_run_merchant_enrichment,
    )

    velocity = PythonOperator(
        task_id="compute_velocity",
        python_callable=_compute_velocity,
    )

    update_profiles = PythonOperator(
        task_id="update_customer_profiles",
        python_callable=_update_customer_profiles,
    )

    write_enriched = PythonOperator(
        task_id="write_enriched_data",
        python_callable=_write_enriched_data,
    )

    publish_metrics = PythonOperator(
        task_id="publish_enrichment_metrics",
        python_callable=_publish_enrichment_metrics,
    )

    trigger_fraud_detection = TriggerDagRunOperator(
        task_id="trigger_fraud_detection",
        trigger_dag_id="fraud_detection_pipeline",
        conf={"triggered_by": "enrichment_pipeline"},
        wait_for_completion=False,
    )

    # DAG dependency chain
    (
        wait_for_transformation
        >> load_data
        >> geo_enrich
        >> device_enrich
        >> merchant_enrich
        >> velocity
        >> update_profiles
        >> write_enriched
        >> publish_metrics
        >> trigger_fraud_detection
    )
