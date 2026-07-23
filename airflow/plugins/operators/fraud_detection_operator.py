"""Custom Fraud Detection operator for Airflow.

Orchestrates the full fraud detection scoring pipeline (rule engine,
anomaly detection, ML scoring, ensemble) as a single reusable operator
with configurable thresholds, alert generation, and metrics collection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from airflow.exceptions import AirflowException
from airflow.models import BaseOperator
from airflow.utils.context import Context

from src.fraud_detection.anomaly_detector import AnomalyDetector
from src.fraud_detection.risk_scorer import RiskScorer
from src.fraud_detection.rule_engine import FraudRuleEngine
from src.fraud_detection.scoring_pipeline import RiskClassification
from src.utils.config import get_settings
from src.utils.constants import (
    SCORE_THRESHOLD_CRITICAL,
    SCORE_THRESHOLD_HIGH,
    SCORE_THRESHOLD_MEDIUM,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, component="fraud_detection_operator")


@dataclass
class FraudDetectionResult:
    """Aggregated result from a batch fraud detection run."""

    total_records: int = 0
    records_scored: int = 0
    scoring_failures: int = 0
    low_risk: int = 0
    medium_risk: int = 0
    high_risk: int = 0
    critical_risk: int = 0
    alerts_generated: int = 0
    anomalies_detected: int = 0
    rules_triggered: int = 0
    avg_ensemble_score: float = 0.0
    max_ensemble_score: float = 0.0
    elapsed_ms: float = 0.0
    scored_records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "records_scored": self.records_scored,
            "scoring_failures": self.scoring_failures,
            "low_risk": self.low_risk,
            "medium_risk": self.medium_risk,
            "high_risk": self.high_risk,
            "critical_risk": self.critical_risk,
            "alerts_generated": self.alerts_generated,
            "anomalies_detected": self.anomalies_detected,
            "rules_triggered": self.rules_triggered,
            "avg_ensemble_score": round(self.avg_ensemble_score, 6),
            "max_ensemble_score": round(self.max_ensemble_score, 6),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class FraudDetectionOperator(BaseOperator):
    """Run the full fraud detection pipeline on a batch of transactions.

    Executes rule-based scoring, anomaly detection, and ML scoring in
    sequence, computes a weighted ensemble score, classifies risk, and
    optionally generates alerts for high-risk transactions.

    Parameters
    ----------
    records_xcom_task : str | None
        Task ID to pull input records from via XCom. If None, expects
        records passed directly in operator_params at runtime.
    records_xcom_key : str
        XCom key to pull records from.
    rule_weight : float
        Ensemble weight for rule-based scoring (0.0–1.0).
    anomaly_weight : float
        Ensemble weight for anomaly detection (0.0–1.0).
    ml_weight : float
        Ensemble weight for ML scoring (0.0–1.0).
    alert_threshold : float
        Ensemble score threshold for generating alerts.
    critical_threshold : float
        Ensemble score threshold for critical alerts.
    generate_alerts : bool
        Whether to generate alerts for high-risk transactions.
    fail_on_high_failure_rate : bool
        If True, fail the task when scoring failure rate exceeds max_failure_rate.
    max_failure_rate : float
        Maximum allowed scoring failure rate (0.0–1.0).
    """

    template_fields: Sequence[str] = ("records_xcom_task", "records_xcom_key")

    def __init__(
        self,
        *,
        records_xcom_task: str | None = None,
        records_xcom_key: str = "records",
        rule_weight: float = 0.30,
        anomaly_weight: float = 0.25,
        ml_weight: float = 0.45,
        alert_threshold: float = SCORE_THRESHOLD_HIGH,
        critical_threshold: float = SCORE_THRESHOLD_CRITICAL,
        generate_alerts: bool = True,
        fail_on_high_failure_rate: bool = True,
        max_failure_rate: float = 0.10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.records_xcom_task = records_xcom_task
        self.records_xcom_key = records_xcom_key
        self.rule_weight = rule_weight
        self.anomaly_weight = anomaly_weight
        self.ml_weight = ml_weight
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        self.generate_alerts = generate_alerts
        self.fail_on_high_failure_rate = fail_on_high_failure_rate
        self.max_failure_rate = max_failure_rate

        # Validate weights sum to ~1.0
        total_weight = rule_weight + anomaly_weight + ml_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(
                f"Ensemble weights must sum to 1.0, got {total_weight:.3f}"
            )

    def execute(self, context: Context) -> dict[str, Any]:
        start_time = time.monotonic()

        # Load records
        records = self._load_records(context)
        if not records:
            logger.info("No records to score")
            return FraudDetectionResult().to_dict()

        # Initialize scoring components
        rule_engine = FraudRuleEngine()
        anomaly_detector = AnomalyDetector()
        risk_scorer = RiskScorer()

        result = FraudDetectionResult(total_records=len(records))

        # Score each record
        scored_records: list[dict[str, Any]] = []
        total_score = 0.0

        for record in records:
            try:
                scored = self._score_single_record(
                    record, rule_engine, anomaly_detector, risk_scorer
                )
                scored_records.append(scored)
                result.records_scored += 1

                # Track metrics
                ensemble_score = scored["ensemble_score"]
                total_score += ensemble_score
                result.max_ensemble_score = max(result.max_ensemble_score, ensemble_score)

                risk_level = scored["ensemble_risk_level"]
                if risk_level == RiskClassification.LOW.value:
                    result.low_risk += 1
                elif risk_level == RiskClassification.MEDIUM.value:
                    result.medium_risk += 1
                elif risk_level == RiskClassification.HIGH.value:
                    result.high_risk += 1
                elif risk_level == RiskClassification.CRITICAL.value:
                    result.critical_risk += 1

                if scored.get("is_anomaly"):
                    result.anomalies_detected += 1
                result.rules_triggered += len(scored.get("rules_triggered", []))

            except Exception as exc:
                result.scoring_failures += 1
                logger.warning(
                    "Scoring failed for record",
                    transaction_id=record.get("transaction_id"),
                    error=str(exc),
                )

        # Compute averages
        if result.records_scored > 0:
            result.avg_ensemble_score = total_score / result.records_scored

        # Check failure rate
        if result.total_records > 0:
            failure_rate = result.scoring_failures / result.total_records
            if self.fail_on_high_failure_rate and failure_rate > self.max_failure_rate:
                raise AirflowException(
                    f"Scoring failure rate {failure_rate:.2%} exceeds "
                    f"threshold {self.max_failure_rate:.2%}"
                )

        # Generate alerts
        if self.generate_alerts:
            result.alerts_generated = self._generate_alerts(scored_records)

        result.elapsed_ms = (time.monotonic() - start_time) * 1000
        result.scored_records = scored_records

        logger.info("Fraud detection batch complete", **result.to_dict())

        # Push results to XCom
        context["ti"].xcom_push(key="scoring_result", value=result.to_dict())
        context["ti"].xcom_push(key="scored_records", value=scored_records)

        return result.to_dict()

    def _load_records(self, context: Context) -> list[dict[str, Any]]:
        """Load records from XCom or operator params."""
        if self.records_xcom_task:
            ti = context["ti"]
            records = ti.xcom_pull(
                task_ids=self.records_xcom_task,
                key=self.records_xcom_key,
            )
            return records or []

        # Check params for direct record passing
        params = context.get("params", {})
        return params.get("records", [])

    def _score_single_record(
        self,
        record: dict[str, Any],
        rule_engine: FraudRuleEngine,
        anomaly_detector: AnomalyDetector,
        risk_scorer: RiskScorer,
    ) -> dict[str, Any]:
        """Score a single record through all three methods and compute ensemble."""
        # Rule-based scoring
        rule_result = rule_engine.evaluate(record)
        rule_score = rule_result.risk_score
        rules_triggered = [r.rule_id for r in rule_result.triggered_rules]

        # Anomaly detection
        anomaly_result = anomaly_detector.detect(record)
        anomaly_score = anomaly_result.anomaly_score
        is_anomaly = anomaly_result.is_anomaly

        # ML scoring
        ml_result = risk_scorer.score(record)
        ml_score = ml_result.risk_score

        # Ensemble
        ensemble_score = (
            self.rule_weight * rule_score
            + self.anomaly_weight * anomaly_score
            + self.ml_weight * ml_score
        )

        # Classify
        if ensemble_score >= self.critical_threshold:
            risk_level = RiskClassification.CRITICAL.value
        elif ensemble_score >= self.alert_threshold:
            risk_level = RiskClassification.HIGH.value
        elif ensemble_score >= SCORE_THRESHOLD_MEDIUM:
            risk_level = RiskClassification.MEDIUM.value
        else:
            risk_level = RiskClassification.LOW.value

        return {
            **record,
            "rule_score": rule_score,
            "rules_triggered": rules_triggered,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "ml_score": ml_score,
            "ml_confidence": ml_result.confidence,
            "ensemble_score": round(ensemble_score, 6),
            "ensemble_risk_level": risk_level,
        }

    def _generate_alerts(self, scored_records: list[dict[str, Any]]) -> int:
        """Generate alerts for high-risk transactions."""
        from src.alerting.alert_manager import AlertManager

        alert_manager = AlertManager()
        alerts_generated = 0

        for record in scored_records:
            ensemble_score = record.get("ensemble_score", 0.0)
            if ensemble_score < self.alert_threshold:
                continue

            try:
                alert_manager.generate_alert(
                    scoring_result={
                        "score": ensemble_score,
                        "risk_level": record.get("ensemble_risk_level", "high"),
                        "triggered_rules": record.get("rules_triggered", []),
                        "anomaly_score": record.get("anomaly_score", 0.0),
                        "ml_score": record.get("ml_score", 0.0),
                    },
                    transaction=record,
                )
                alerts_generated += 1
            except Exception as exc:
                logger.warning(
                    "Alert generation failed",
                    transaction_id=record.get("transaction_id"),
                    error=str(exc),
                )

        return alerts_generated
