"""Kafka consumer with end-to-end pipeline integration.

Consumes raw transaction events from Kafka and processes them through
the full pipeline: Validate → Clean → Normalize → Features → Enrich.

Features:
- Configurable batch processing with poll-based consumption
- Stage-level error handling (skip vs. halt per stage)
- Dead-letter queue routing for failed records
- Graceful shutdown with offset commit
- Per-batch and aggregate pipeline metrics
- Consumer group rebalance handling
"""

from __future__ import annotations

import json
import signal
import time
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Any, Callable

from confluent_kafka import Consumer, KafkaError, KafkaException, TopicPartition

from src.monitoring.cloudwatch_logger import configure_cloudwatch_logging, set_correlation_id
from src.monitoring.metrics_collector import CloudWatchMetricsCollector
from src.pipeline_orchestrator import (
    BatchResult,
    PipelineOrchestrator,
    StageErrorPolicy,
)
from src.utils.config import get_settings
from src.utils.constants import (
    BATCH_SIZE_DEFAULT,
    MAX_BATCH_SIZE,
    POLL_TIMEOUT_MS,
    TOPIC_RAW_EVENTS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, component="kafka_consumer")


@dataclass
class ConsumerMetrics:
    """Aggregate consumer metrics."""

    total_messages_consumed: int = 0
    total_batches_processed: int = 0
    total_records_succeeded: int = 0
    total_records_failed: int = 0
    total_records_dlq: int = 0
    total_deserialization_errors: int = 0
    total_poll_empty: int = 0
    uptime_seconds: float = 0.0
    _start_time: float = field(default_factory=time.time, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def throughput_per_second(self) -> float:
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self.total_messages_consumed / elapsed

    @property
    def success_rate(self) -> float:
        total = self.total_records_succeeded + self.total_records_failed
        if total == 0:
            return 1.0
        return self.total_records_succeeded / total

    def record_batch(self, batch_result: BatchResult) -> None:
        with self._lock:
            self.total_batches_processed += 1
            self.total_messages_consumed += batch_result.total
            self.total_records_succeeded += batch_result.succeeded
            self.total_records_failed += batch_result.failed
            self.total_records_dlq += batch_result.dlq_count

    def record_deserialization_error(self) -> None:
        with self._lock:
            self.total_deserialization_errors += 1

    def record_empty_poll(self) -> None:
        with self._lock:
            self.total_poll_empty += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self.uptime_seconds = time.time() - self._start_time
            return {
                "total_messages_consumed": self.total_messages_consumed,
                "total_batches_processed": self.total_batches_processed,
                "total_records_succeeded": self.total_records_succeeded,
                "total_records_failed": self.total_records_failed,
                "total_records_dlq": self.total_records_dlq,
                "total_deserialization_errors": self.total_deserialization_errors,
                "throughput_per_second": round(self.throughput_per_second, 2),
                "success_rate": round(self.success_rate, 4),
                "uptime_seconds": round(self.uptime_seconds, 2),
            }


class TransactionConsumer:
    """Kafka consumer that feeds the end-to-end processing pipeline.

    Consumes from the raw events topic, batches messages, and processes
    them through the pipeline orchestrator. Supports graceful shutdown,
    manual offset commits, and configurable error handling.

    Usage:
        consumer = TransactionConsumer(
            bootstrap_servers="localhost:9092",
            group_id="riskpulse-pipeline",
        )
        consumer.start()  # Blocks until shutdown signal
    """

    def __init__(
        self,
        *,
        bootstrap_servers: str | None = None,
        group_id: str | None = None,
        topics: list[str] | None = None,
        batch_size: int | None = None,
        poll_timeout_ms: int | None = None,
        max_poll_records: int | None = None,
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = False,
        pipeline: PipelineOrchestrator | None = None,
        error_policy: dict[str, StageErrorPolicy] | None = None,
        on_batch_complete: Callable[[BatchResult], None] | None = None,
        on_dlq: Callable[[dict[str, Any], str, str], None] | None = None,
    ) -> None:
        settings = get_settings()
        configure_cloudwatch_logging(
            service="worker",
            environment=settings.environment,
            retention_days=settings.get("monitoring.cloudwatch.log_retention_days", 30),
        )

        self._bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self._group_id = group_id or "riskpulse-pipeline-consumer"
        self._topics = topics or [TOPIC_RAW_EVENTS]
        self._batch_size = min(
            batch_size or settings.get("kafka.consumer.batch_size", BATCH_SIZE_DEFAULT),
            MAX_BATCH_SIZE,
        )
        self._poll_timeout_ms = poll_timeout_ms or settings.get(
            "kafka.consumer.poll_timeout_ms", POLL_TIMEOUT_MS
        )
        self._max_poll_records = max_poll_records or settings.get(
            "kafka.consumer.max_poll_records", 500
        )
        self._auto_offset_reset = auto_offset_reset
        self._enable_auto_commit = enable_auto_commit

        # Pipeline orchestrator
        self._pipeline = pipeline or PipelineOrchestrator(
            batch_size=self._batch_size,
            error_policy=error_policy,
            on_dlq=on_dlq,
        )

        # Callbacks
        self._on_batch_complete = on_batch_complete
        self._on_dlq = on_dlq

        # State
        self._consumer: Consumer | None = None
        self._running = Event()
        self._shutdown = Event()
        self._metrics = ConsumerMetrics()
        self._cloudwatch_metrics = CloudWatchMetricsCollector(
            service="worker",
            environment=settings.environment,
            enabled=bool(settings.get("monitoring.cloudwatch.enabled", False)),
        )
        self._lock = Lock()

        logger.info(
            "Transaction consumer configured",
            bootstrap_servers=self._bootstrap_servers,
            group_id=self._group_id,
            topics=self._topics,
            batch_size=self._batch_size,
            poll_timeout_ms=self._poll_timeout_ms,
        )

    @property
    def metrics(self) -> ConsumerMetrics:
        return self._metrics

    @property
    def pipeline(self) -> PipelineOrchestrator:
        return self._pipeline

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        """Start consuming messages. Blocks until shutdown is signaled."""
        self._create_consumer()
        self._running.set()
        self._register_signal_handlers()

        logger.info("Consumer started", topics=self._topics, group_id=self._group_id)

        try:
            self._consume_loop()
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by keyboard")
        finally:
            self._shutdown_consumer()

    def stop(self) -> None:
        """Signal the consumer to stop gracefully."""
        logger.info("Consumer stop requested")
        self._shutdown.set()

    def consume_batch(self) -> BatchResult | None:
        """Consume and process a single batch (non-blocking API for testing).

        Returns:
            BatchResult if messages were processed, None if no messages available.
        """
        if self._consumer is None:
            self._create_consumer()

        messages = self._poll_batch()
        if not messages:
            self._metrics.record_empty_poll()
            return None

        records = self._deserialize_messages(messages)
        if not records:
            return None

        batch_result = self._pipeline.process_batch(records)
        self._metrics.record_batch(batch_result)
        self._publish_batch_metrics(batch_result)

        # Commit offsets after successful batch processing
        self._commit_offsets()

        if self._on_batch_complete:
            try:
                self._on_batch_complete(batch_result)
            except Exception as e:
                logger.error("Batch complete callback failed", error=str(e))

        return batch_result

    def process_records(self, records: list[dict[str, Any]]) -> BatchResult:
        """Process records directly through the pipeline (bypass Kafka for testing).

        Args:
            records: List of transaction records.

        Returns:
            BatchResult with processing outcomes.
        """
        batch_result = self._pipeline.process_batch(records)
        self._metrics.record_batch(batch_result)
        self._publish_batch_metrics(batch_result)
        return batch_result

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _create_consumer(self) -> None:
        """Create and configure the Kafka consumer."""
        config = {
            "bootstrap.servers": self._bootstrap_servers,
            "group.id": self._group_id,
            "auto.offset.reset": self._auto_offset_reset,
            "enable.auto.commit": self._enable_auto_commit,
            "max.poll.interval.ms": 300000,
            "session.timeout.ms": 30000,
            "heartbeat.interval.ms": 10000,
            "fetch.min.bytes": 1,
            "fetch.max.wait.ms": 500,
        }

        self._consumer = Consumer(config)
        self._consumer.subscribe(self._topics, on_assign=self._on_assign, on_revoke=self._on_revoke)

        logger.info(
            "Kafka consumer created",
            config={k: v for k, v in config.items() if "password" not in k},
        )

    def _consume_loop(self) -> None:
        """Main consumption loop."""
        while not self._shutdown.is_set():
            try:
                batch_result = self.consume_batch()
                if batch_result is None:
                    continue

                logger.debug(
                    "Batch processed",
                    succeeded=batch_result.succeeded,
                    failed=batch_result.failed,
                    throughput=round(batch_result.metrics.throughput_per_second, 2),
                )

            except KafkaException as e:
                logger.error("Kafka consumption error", error=str(e))
                if self._is_fatal_error(e):
                    logger.critical("Fatal Kafka error, shutting down", error=str(e))
                    break
                time.sleep(1.0)

            except Exception as e:
                logger.error("Unexpected error in consume loop", error=str(e))
                time.sleep(0.5)

    def _poll_batch(self) -> list[Any]:
        """Poll Kafka for a batch of messages."""
        if self._consumer is None:
            return []

        messages: list[Any] = []
        poll_timeout = self._poll_timeout_ms / 1000.0

        # Collect messages up to batch_size
        while len(messages) < self._batch_size:
            msg = self._consumer.poll(timeout=poll_timeout)

            if msg is None:
                break

            if msg.error():
                error = msg.error()
                if error is None:
                    continue
                if error.code() == KafkaError._PARTITION_EOF:
                    continue
                elif error.code() == KafkaError._ALL_BROKERS_DOWN:
                    logger.error("All brokers down")
                    break
                else:
                    logger.warning("Consumer poll error", error=str(error))
                    continue

            messages.append(msg)

            # After first message, use shorter timeout for batching
            poll_timeout = 0.05

        return messages

    def _deserialize_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Deserialize Kafka messages to transaction records."""
        records: list[dict[str, Any]] = []

        for msg in messages:
            try:
                value = msg.value()
                if value is None:
                    continue

                if isinstance(value, bytes):
                    record = json.loads(value.decode("utf-8"))
                elif isinstance(value, str):
                    record = json.loads(value)
                else:
                    record = value

                # Attach Kafka metadata
                record["_kafka_topic"] = msg.topic()
                record["_kafka_partition"] = msg.partition()
                record["_kafka_offset"] = msg.offset()
                record["_kafka_timestamp"] = msg.timestamp()[1] if msg.timestamp() else None

                records.append(record)

            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._metrics.record_deserialization_error()
                self._cloudwatch_metrics.record_error(error_count=1, severity="medium")
                logger.warning(
                    "Message deserialization failed",
                    error=str(e),
                    topic=msg.topic(),
                    partition=msg.partition(),
                    offset=msg.offset(),
                )

        return records

    def _publish_batch_metrics(self, batch_result: BatchResult) -> None:
        """Publish worker batch metrics to CloudWatch when enabled."""
        set_correlation_id(batch_result.batch_id)
        self._cloudwatch_metrics.record_transactions_processed(batch_result.total)
        self._cloudwatch_metrics.record_error(
            error_count=batch_result.failed,
            total_count=batch_result.total,
            severity="high" if batch_result.failed else "none",
        )
        self._cloudwatch_metrics.record_pipeline_latency(
            batch_result.latency_ms or batch_result.metrics.total_pipeline_latency_ms,
            stage="batch",
        )
        for stage_name, stage_metrics in batch_result.metrics.stage_metrics.items():
            self._cloudwatch_metrics.record_pipeline_latency(
                stage_metrics.avg_latency_ms,
                stage=stage_name,
            )
        self._cloudwatch_metrics.flush()

    def _commit_offsets(self) -> None:
        """Commit consumer offsets."""
        if self._consumer and not self._enable_auto_commit:
            try:
                self._consumer.commit(asynchronous=False)
            except KafkaException as e:
                logger.error("Offset commit failed", error=str(e))

    def _shutdown_consumer(self) -> None:
        """Gracefully shut down the consumer."""
        self._running.clear()

        if self._consumer:
            try:
                self._commit_offsets()
                self._consumer.close()
            except Exception as e:
                logger.error("Error during consumer shutdown", error=str(e))

        logger.info(
            "Consumer shut down",
            metrics=self._metrics.snapshot(),
        )

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (OSError, ValueError):
            # Cannot set signal handlers in non-main thread
            pass

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info("Received shutdown signal", signal=signum)
        self.stop()

    def _on_assign(self, consumer: Consumer, partitions: list[TopicPartition]) -> None:
        """Handle partition assignment during rebalance."""
        logger.info(
            "Partitions assigned",
            partitions=[f"{p.topic}[{p.partition}]" for p in partitions],
        )

    def _on_revoke(self, consumer: Consumer, partitions: list[TopicPartition]) -> None:
        """Handle partition revocation during rebalance."""
        logger.info(
            "Partitions revoked",
            partitions=[f"{p.topic}[{p.partition}]" for p in partitions],
        )
        # Commit offsets before revocation
        self._commit_offsets()

    @staticmethod
    def _is_fatal_error(error: KafkaException) -> bool:
        """Determine if a Kafka error is fatal (unrecoverable)."""
        fatal_codes = {
            KafkaError._FATAL,
            KafkaError._ALL_BROKERS_DOWN,
            KafkaError.BROKER_NOT_AVAILABLE,
        }
        kafka_error = error.args[0] if error.args else None
        if kafka_error is None:
            return False
        code_method = getattr(kafka_error, "code", None)
        if callable(code_method):
            code = code_method() if callable(code_method) else None
            return code in fatal_codes if code is not None else False
        return False


def main() -> None:
    """Run the transaction consumer as a container-friendly worker."""
    consumer = TransactionConsumer()
    consumer.start()


if __name__ == "__main__":
    main()
