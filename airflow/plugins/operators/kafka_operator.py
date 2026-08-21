"""Custom Kafka consume operator for Airflow.

Consumes a batch of messages from Kafka topics and processes them through
the RiskPulse pipeline orchestrator. Supports configurable batch sizes,
timeouts, and dead-letter queue routing.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from airflow.exceptions import AirflowException
from airflow.models import BaseOperator
from airflow.utils.context import Context
from confluent_kafka import Consumer, KafkaError, KafkaException

from src.pipeline_orchestrator import (
    BatchResult,
    PipelineOrchestrator,
)
from src.utils.config import get_settings
from src.utils.constants import TOPIC_RAW_EVENTS
from src.utils.logger import get_logger

logger = get_logger(__name__, component="kafka_operator")

_DEFAULT_BATCH_SIZE = 500
_DEFAULT_POLL_TIMEOUT_MS = 5000
_MAX_EMPTY_POLLS = 10
_CONSUMER_SESSION_TIMEOUT_MS = 30000
_CONSUMER_HEARTBEAT_INTERVAL_MS = 10000


class KafkaConsumeOperator(BaseOperator):
    """Consume and process a batch of messages from a Kafka topic.

    Pulls up to ``batch_size`` messages from the configured topic, deserialises
    them, and feeds them through ``PipelineOrchestrator.process_batch()``.

    The operator pushes the ``BatchResult`` summary dict to XCom so downstream
    tasks can inspect success/failure counts.

    Parameters
    ----------
    topics : list[str] | None
        Topics to subscribe to. Defaults to ``[txn.raw.events]``.
    batch_size : int
        Maximum number of messages to consume per execution.
    poll_timeout_ms : int
        Per-poll timeout in milliseconds.
    group_id : str | None
        Kafka consumer group id. Falls back to config/settings.
    bootstrap_servers : str | None
        Kafka bootstrap servers. Falls back to config/settings.
    fail_on_empty : bool
        If ``True``, fail the task when no messages are available.
    min_success_rate : float
        Minimum ratio of successfully processed records to consider the
        task a success (0.0–1.0). Defaults to 0.95.
    """

    template_fields: Sequence[str] = ("topics", "batch_size")

    def __init__(
        self,
        *,
        topics: list[str] | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        poll_timeout_ms: int = _DEFAULT_POLL_TIMEOUT_MS,
        group_id: str | None = None,
        bootstrap_servers: str | None = None,
        fail_on_empty: bool = False,
        min_success_rate: float = 0.95,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.topics = topics or [TOPIC_RAW_EVENTS]
        self.batch_size = batch_size
        self.poll_timeout_ms = poll_timeout_ms
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
        self.fail_on_empty = fail_on_empty
        self.min_success_rate = min_success_rate

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, context: Context) -> dict[str, Any]:
        settings = get_settings()
        bootstrap = self.bootstrap_servers or settings.kafka_bootstrap_servers
        group = self.group_id or "riskpulse-airflow-consumer"

        consumer = self._create_consumer(bootstrap, group)
        try:
            records = self._poll_records(consumer)

            if not records:
                logger.info("No messages available from Kafka", topics=self.topics)
                if self.fail_on_empty:
                    raise AirflowException(
                        f"No messages consumed from {self.topics} and fail_on_empty=True"
                    )
                return {"total": 0, "succeeded": 0, "failed": 0, "dlq": 0}

            result = self._process_records(records)
            consumer.commit(asynchronous=False)

            summary = {
                "batch_id": result.batch_id,
                "total": result.total,
                "succeeded": result.succeeded,
                "failed": result.failed,
                "dlq": result.dlq_count,
                "success_rate": round(result.success_rate, 4),
            }

            logger.info("Kafka batch processed", **summary)

            if result.success_rate < self.min_success_rate:
                raise AirflowException(
                    f"Batch success rate {result.success_rate:.2%} below "
                    f"threshold {self.min_success_rate:.2%}"
                )

            return summary

        finally:
            consumer.close()
            logger.info("Kafka consumer closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_consumer(self, bootstrap_servers: str, group_id: str) -> Consumer:
        config = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": 300000,
            "session.timeout.ms": _CONSUMER_SESSION_TIMEOUT_MS,
            "heartbeat.interval.ms": _CONSUMER_HEARTBEAT_INTERVAL_MS,
            "fetch.min.bytes": 1,
            "fetch.max.wait.ms": 500,
        }
        consumer = Consumer(config)
        consumer.subscribe(self.topics)
        logger.info(
            "Kafka consumer created for Airflow task",
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            topics=self.topics,
        )
        return consumer

    def _poll_records(self, consumer: Consumer) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        empty_polls = 0

        while len(records) < self.batch_size and empty_polls < _MAX_EMPTY_POLLS:
            msg = consumer.poll(timeout=self.poll_timeout_ms / 1000.0)

            if msg is None:
                empty_polls += 1
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug("Reached end of partition", partition=msg.partition())
                    empty_polls += 1
                    continue
                raise KafkaException(msg.error())

            empty_polls = 0
            try:
                value = json.loads(msg.value().decode("utf-8"))
                records.append(value)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning(
                    "Failed to deserialize message",
                    partition=msg.partition(),
                    offset=msg.offset(),
                    error=str(exc),
                )

        return records

    def _process_records(self, records: list[dict[str, Any]]) -> BatchResult:
        pipeline = PipelineOrchestrator(batch_size=self.batch_size)
        return pipeline.process_batch(records)
