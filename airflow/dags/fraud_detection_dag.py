"""Fraud Detection DAG — rule-based, anomaly, and ML scoring with alert generation.

Triggered by enrichment pipeline completion. Runs the full fraud detection
pipeline (rule engine → anomaly detection → ML scoring → ensemble) and
generates alerts for high-risk transactions.

Schedule: None (externally triggered by enrichment_pipeline)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

from airflow import DAG
from src.alerting.alert_manager import AlertManager
from src.fraud_detection.anomaly_detector import AnomalyDetector
from src.fraud_detection.risk_scorer import RiskScorer
from src.fraud_detection.rule_engine import FraudRuleEngine
from src.fraud_detection.scoring_pipeline import RiskClassification
from src.storage.s3_handler import StorageLayer, get_s3_handler
from src.utils.config import get_settings
from src.utils.constants import (
    SCORE_THRESHOLD_CRITICAL,
    SCORE_THRESHOLD_HIGH,
    TOPIC_FRAUD_ALERTS,
    TOPIC_METRICS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, component="fraud_detection_dag")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ALERT_THRESHOLD = SCORE_THRESHOLD_HIGH  # 0.8
_CRITICAL_THRESHOLD = SCORE_THRESHOLD_CRITICAL  # 0.95
_SCORING_BATCH_SIZE = 500

default_args = {
    "owner": "riskpulse",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(minutes=30),
}


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _load_enriched_data(**context: Any) -> dict[str, Any]:
    """Load enriched transaction data from S3 for fraud scoring."""
    handler = get_s3_handler()

    execution_date: datetime = context["execution_date"]
    partition = execution_date.strftime("year=%Y/month=%m/day=%d/hour=%H")

    records = handler.read_batch(
        storage_layer=StorageLayer.ENRICHED,
        partition_path=partition,
    )

    if not records:
        logger.info("No enriched records available for fraud detection")
        context["ti"].xcom_push(key="record_count", value=0)
        return {"records_loaded": 0}

    context["ti"].xcom_push(key="record_count", value=len(records))
    context["ti"].xcom_push(key="partition", value=partition)

    logger.info("Loaded enriched records for scoring", count=len(records))
    return {"records_loaded": len(records), "partition": partition}


def _run_rule_based_scoring(**context: Any) -> dict[str, Any]:
    """Execute rule-based fraud detection on enriched transactions."""
    ti = context["ti"]
    record_count = ti.xcom_pull(task_ids="load_enriched_data", key="record_count")

    if not record_count:
        return {"records_scored": 0, "rules_triggered": 0, "high_risk": 0}

    handler = get_s3_handler()
    partition = ti.xcom_pull(task_ids="load_enriched_data", key="partition")

    records = handler.read_batch(
        storage_layer=StorageLayer.ENRICHED,
        partition_path=partition,
    )

    rule_engine = FraudRuleEngine()
    scored: list[dict[str, Any]] = []
    rules_triggered = 0
    high_risk_count = 0

    for record in records:
        try:
            result = rule_engine.evaluate(record)
            scored_record = {
                **record,
                "rule_score": result.risk_score,
                "rules_triggered": [r.rule_id for r in result.triggered_rules],
                "rule_risk_level": result.risk_level,
            }
            scored.append(scored_record)
            rules_triggered += len(result.triggered_rules)
            if result.risk_score >= _ALERT_THRESHOLD:
                high_risk_count += 1
        except Exception as exc:
            logger.warning(
                "Rule evaluation failed",
                transaction_id=record.get("transaction_id"),
                error=str(exc),
            )
            scored.append(
                {**record, "rule_score": 0.0, "rules_triggered": [], "rule_risk_level": "low"}
            )

    handler.write_batch(
        records=scored,
        storage_layer=StorageLayer.RAW,
        partition_path=f"scored/rules/{partition}",
    )

    summary = {
        "records_scored": len(scored),
        "rules_triggered": rules_triggered,
        "high_risk": high_risk_count,
    }
    logger.info("Rule-based scoring complete", **summary)
    return summary


def _run_anomaly_detection(**context: Any) -> dict[str, Any]:
    """Run Isolation Forest anomaly detection on enriched transactions."""
    ti = context["ti"]
    record_count = ti.xcom_pull(task_ids="load_enriched_data", key="record_count")

    if not record_count:
        return {"records_scored": 0, "anomalies_detected": 0}

    handler = get_s3_handler()
    partition = ti.xcom_pull(task_ids="load_enriched_data", key="partition")

    records = handler.read_batch(
        storage_layer=StorageLayer.RAW,
        partition_path=f"scored/rules/{partition}",
    )

    detector = AnomalyDetector()
    scored: list[dict[str, Any]] = []
    anomalies_detected = 0

    for record in records:
        try:
            result = detector.detect(record)
            scored_record = {
                **record,
                "anomaly_score": result.anomaly_score,
                "is_anomaly": result.is_anomaly,
                "anomaly_features": result.contributing_features,
            }
            scored.append(scored_record)
            if result.is_anomaly:
                anomalies_detected += 1
        except Exception as exc:
            logger.warning(
                "Anomaly detection failed",
                transaction_id=record.get("transaction_id"),
                error=str(exc),
            )
            scored.append(
                {**record, "anomaly_score": 0.0, "is_anomaly": False, "anomaly_features": []}
            )

    handler.write_batch(
        records=scored,
        storage_layer=StorageLayer.RAW,
        partition_path=f"scored/anomaly/{partition}",
    )

    summary = {
        "records_scored": len(scored),
        "anomalies_detected": anomalies_detected,
        "anomaly_rate": anomalies_detected / len(scored) if scored else 0,
    }
    logger.info("Anomaly detection complete", **summary)
    return summary


def _run_ml_scoring(**context: Any) -> dict[str, Any]:
    """Run gradient boosted tree ML model scoring."""
    ti = context["ti"]
    record_count = ti.xcom_pull(task_ids="load_enriched_data", key="record_count")

    if not record_count:
        return {"records_scored": 0, "high_risk": 0, "critical": 0}

    handler = get_s3_handler()
    partition = ti.xcom_pull(task_ids="load_enriched_data", key="partition")

    records = handler.read_batch(
        storage_layer=StorageLayer.RAW,
        partition_path=f"scored/anomaly/{partition}",
    )

    scorer = RiskScorer()
    scored: list[dict[str, Any]] = []
    high_risk_count = 0
    critical_count = 0

    for record in records:
        try:
            result = scorer.score(record)
            scored_record = {
                **record,
                "ml_score": result.risk_score,
                "ml_risk_level": result.risk_level,
                "ml_confidence": result.confidence,
                "ml_top_features": result.top_features,
                "model_version": result.model_version,
            }
            scored.append(scored_record)
            if result.risk_score >= _ALERT_THRESHOLD:
                high_risk_count += 1
            if result.risk_score >= _CRITICAL_THRESHOLD:
                critical_count += 1
        except Exception as exc:
            logger.warning(
                "ML scoring failed",
                transaction_id=record.get("transaction_id"),
                error=str(exc),
            )
            scored.append(
                {
                    **record,
                    "ml_score": 0.0,
                    "ml_risk_level": "low",
                    "ml_confidence": 0.0,
                    "ml_top_features": [],
                    "model_version": "fallback",
                }
            )

    handler.write_batch(
        records=scored,
        storage_layer=StorageLayer.RAW,
        partition_path=f"scored/ml/{partition}",
    )

    summary = {
        "records_scored": len(scored),
        "high_risk": high_risk_count,
        "critical": critical_count,
    }
    logger.info("ML scoring complete", **summary)
    return summary


def _compute_ensemble_score(**context: Any) -> dict[str, Any]:
    """Compute weighted ensemble score from rule, anomaly, and ML results."""
    ti = context["ti"]
    record_count = ti.xcom_pull(task_ids="load_enriched_data", key="record_count")

    if not record_count:
        return {"records_scored": 0, "high_risk": 0, "critical": 0}

    handler = get_s3_handler()
    partition = ti.xcom_pull(task_ids="load_enriched_data", key="partition")
    get_settings()

    records = handler.read_batch(
        storage_layer=StorageLayer.RAW,
        partition_path=f"scored/ml/{partition}",
    )

    # Ensemble weights (configurable)
    rule_weight = 0.30
    anomaly_weight = 0.25
    ml_weight = 0.45

    scored: list[dict[str, Any]] = []
    high_risk_count = 0
    critical_count = 0

    for record in records:
        rule_score = record.get("rule_score", 0.0)
        anomaly_score = record.get("anomaly_score", 0.0)
        ml_score = record.get("ml_score", 0.0)

        ensemble_score = (
            rule_weight * rule_score + anomaly_weight * anomaly_score + ml_weight * ml_score
        )

        # Classify risk level
        if ensemble_score >= _CRITICAL_THRESHOLD:
            risk_level = RiskClassification.CRITICAL.value
            critical_count += 1
            high_risk_count += 1
        elif ensemble_score >= _ALERT_THRESHOLD:
            risk_level = RiskClassification.HIGH.value
            high_risk_count += 1
        elif ensemble_score >= 0.5:
            risk_level = RiskClassification.MEDIUM.value
        else:
            risk_level = RiskClassification.LOW.value

        scored_record = {
            **record,
            "ensemble_score": round(ensemble_score, 6),
            "ensemble_risk_level": risk_level,
            "scoring_weights": {
                "rule": rule_weight,
                "anomaly": anomaly_weight,
                "ml": ml_weight,
            },
        }
        scored.append(scored_record)

    handler.write_batch(
        records=scored,
        storage_layer=StorageLayer.RAW,
        partition_path=f"scored/ensemble/{partition}",
    )

    summary = {
        "records_scored": len(scored),
        "high_risk": high_risk_count,
        "critical": critical_count,
        "avg_score": sum(r["ensemble_score"] for r in scored) / len(scored) if scored else 0,
    }
    logger.info("Ensemble scoring complete", **summary)
    ti.xcom_push(key="ensemble_summary", value=summary)
    return summary


def _generate_alerts(**context: Any) -> dict[str, Any]:
    """Generate fraud alerts for high-risk transactions and publish to Kafka."""
    from confluent_kafka import Producer as KafkaProducer

    ti = context["ti"]
    record_count = ti.xcom_pull(task_ids="load_enriched_data", key="record_count")

    if not record_count:
        return {"alerts_generated": 0, "critical_alerts": 0}

    handler = get_s3_handler()
    partition = ti.xcom_pull(task_ids="load_enriched_data", key="partition")
    settings = get_settings()

    records = handler.read_batch(
        storage_layer=StorageLayer.RAW,
        partition_path=f"scored/ensemble/{partition}",
    )

    alert_manager = AlertManager()
    producer = KafkaProducer({"bootstrap.servers": settings.kafka_bootstrap_servers})

    alerts_generated = 0
    critical_alerts = 0

    for record in records:
        ensemble_score = record.get("ensemble_score", 0.0)
        risk_level = record.get("ensemble_risk_level", "low")

        if ensemble_score < _ALERT_THRESHOLD:
            continue

        try:
            alert = alert_manager.generate_alert(
                scoring_result={
                    "score": ensemble_score,
                    "risk_level": risk_level,
                    "triggered_rules": record.get("rules_triggered", []),
                    "anomaly_score": record.get("anomaly_score", 0.0),
                    "ml_score": record.get("ml_score", 0.0),
                },
                transaction=record,
            )

            # Publish alert to Kafka for real-time consumers
            alert_payload = {
                "alert_id": str(alert.alert_id) if hasattr(alert, "alert_id") else None,
                "transaction_id": record.get("transaction_id"),
                "ensemble_score": ensemble_score,
                "risk_level": risk_level,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "rules_triggered": record.get("rules_triggered", []),
            }

            producer.produce(
                TOPIC_FRAUD_ALERTS,
                key=record.get("transaction_id", "").encode("utf-8"),
                value=json.dumps(alert_payload).encode("utf-8"),
            )

            alerts_generated += 1
            if ensemble_score >= _CRITICAL_THRESHOLD:
                critical_alerts += 1

        except Exception as exc:
            logger.error(
                "Alert generation failed",
                transaction_id=record.get("transaction_id"),
                error=str(exc),
            )

    producer.flush(timeout=30)

    summary = {
        "alerts_generated": alerts_generated,
        "critical_alerts": critical_alerts,
        "total_records": len(records),
    }
    logger.info("Alert generation complete", **summary)
    return summary


def _write_scored_data(**context: Any) -> dict[str, Any]:
    """Write final scored data to S3 for Snowflake ingestion."""
    ti = context["ti"]
    record_count = ti.xcom_pull(task_ids="load_enriched_data", key="record_count")

    if not record_count:
        return {"files_written": 0, "record_count": 0}

    handler = get_s3_handler()
    partition = ti.xcom_pull(task_ids="load_enriched_data", key="partition")

    records = handler.read_batch(
        storage_layer=StorageLayer.RAW,
        partition_path=f"scored/ensemble/{partition}",
    )

    file_key = handler.write_batch(
        records=records,
        storage_layer=StorageLayer.PROCESSED,
        partition_path=f"scored/{partition}",
        metadata={
            "dag_id": context["dag"].dag_id,
            "run_id": context["run_id"],
            "record_count": len(records),
            "scoring_methods": ["rules", "anomaly", "ml", "ensemble"],
        },
    )

    result = {
        "files_written": 1,
        "s3_key": file_key,
        "partition": partition,
        "record_count": len(records),
    }
    logger.info("Scored data written to S3", **result)
    return result


def _publish_scoring_metrics(**context: Any) -> dict[str, Any]:
    """Publish fraud detection pipeline metrics."""
    from confluent_kafka import Producer as KafkaProducer

    ti = context["ti"]
    settings = get_settings()

    rule_result = ti.xcom_pull(task_ids="run_rule_based_scoring") or {}
    anomaly_result = ti.xcom_pull(task_ids="run_anomaly_detection") or {}
    ml_result = ti.xcom_pull(task_ids="run_ml_scoring") or {}
    ensemble_result = ti.xcom_pull(task_ids="compute_ensemble_score") or {}
    alert_result = ti.xcom_pull(task_ids="generate_alerts") or {}

    metrics = {
        "dag_id": "fraud_detection_pipeline",
        "run_id": context["run_id"],
        "execution_date": context["execution_date"].isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rule_scoring": rule_result,
        "anomaly_detection": anomaly_result,
        "ml_scoring": ml_result,
        "ensemble": ensemble_result,
        "alerts": alert_result,
    }

    producer = KafkaProducer({"bootstrap.servers": settings.kafka_bootstrap_servers})
    producer.produce(
        TOPIC_METRICS,
        key="fraud_detection_pipeline",
        value=json.dumps(metrics).encode("utf-8"),
    )
    producer.flush(timeout=10)

    logger.info("Fraud detection metrics published")
    return metrics


def _on_failure_callback(context: dict[str, Any]) -> None:
    """Alert on task failure in the fraud detection DAG."""
    from src.alerting.alert_manager import AlertManager

    dag_id = context.get("dag", {})
    task_id = context.get("task_instance", {})
    exception = context.get("exception")

    logger.error(
        "Fraud detection DAG task failed",
        dag_id=str(dag_id),
        task_id=str(task_id),
        error=str(exception),
    )

    try:
        manager = AlertManager()
        manager.generate_alert(
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
    dag_id="fraud_detection_pipeline",
    description="Rule-based, anomaly, and ML fraud scoring with alert generation",
    schedule_interval=None,  # Triggered by enrichment_pipeline
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=2,
    tags=["fraud", "scoring", "rules", "anomaly", "ml", "alerts", "riskpulse"],
    default_args={**default_args, "on_failure_callback": _on_failure_callback},
) as dag:

    wait_for_enrichment = ExternalTaskSensor(
        task_id="wait_for_enrichment",
        external_dag_id="enrichment_pipeline",
        external_task_id="write_enriched_data",
        mode="reschedule",
        poke_interval=120,
        timeout=1800,
        allowed_states=["success"],
    )

    load_data = PythonOperator(
        task_id="load_enriched_data",
        python_callable=_load_enriched_data,
    )

    rule_scoring = PythonOperator(
        task_id="run_rule_based_scoring",
        python_callable=_run_rule_based_scoring,
    )

    anomaly_detection = PythonOperator(
        task_id="run_anomaly_detection",
        python_callable=_run_anomaly_detection,
    )

    ml_scoring = PythonOperator(
        task_id="run_ml_scoring",
        python_callable=_run_ml_scoring,
    )

    ensemble = PythonOperator(
        task_id="compute_ensemble_score",
        python_callable=_compute_ensemble_score,
    )

    alerts = PythonOperator(
        task_id="generate_alerts",
        python_callable=_generate_alerts,
    )

    write_scored = PythonOperator(
        task_id="write_scored_data",
        python_callable=_write_scored_data,
    )

    publish_metrics = PythonOperator(
        task_id="publish_scoring_metrics",
        python_callable=_publish_scoring_metrics,
    )

    # DAG dependency chain — sequential scoring pipeline
    (
        wait_for_enrichment
        >> load_data
        >> rule_scoring
        >> anomaly_detection
        >> ml_scoring
        >> ensemble
        >> alerts
        >> write_scored
        >> publish_metrics
    )
