"""Unit tests for Kafka producer with comprehensive coverage.

Tests cover:
- Producer initialization and configuration
- Message production with schema validation
- Partition key strategy (account_id based)
- Delivery callbacks and error handling
- Batch production
- Metrics collection
- Graceful shutdown
- Edge cases and error scenarios
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, call
from typing import Any

import pytest

from src.ingestion.kafka_producer import (
    ProducerDeliveryError,
    ProducerError,
    ProducerMetrics,
    TransactionProducer,
)
from src.ingestion.schema_registry import SchemaValidationError


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_confluent_producer():
    """Mock the confluent_kafka.Producer class."""
    with patch("src.ingestion.kafka_producer.Producer") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.__len__ = MagicMock(return_value=0)
        mock_instance.flush.return_value = 0
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_schema_registry():
    """Mock SchemaRegistry that passes validation and returns bytes."""
    mock = MagicMock()
    mock.validate.side_effect = lambda schema_name, record: record
    mock.serialize.return_value = b"\x00\x01\x02\x03"
    return mock


@pytest.fixture
def valid_transaction():
    """A valid transaction matching the schema."""
    return {
        "external_transaction_id": "TXN-ABC123DEF456",
        "account_id": "ACC-12345",
        "customer_id": "CUST-67890",
        "merchant_id": "MERCH-001",
        "merchant_name": "Amazon",
        "merchant_category_code": "5411",
        "transaction_amount": 125.50,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "card_type": "credit",
        "card_last_four": "4242",
        "ip_address": "192.168.1.100",
        "device_id": "device-abc-123",
        "device_type": "mobile",
        "geo_latitude": 40.7128,
        "geo_longitude": -74.0060,
        "geo_country": "US",
        "geo_city": "New York",
        "is_international": False,
        "transaction_timestamp": "2026-06-15T10:30:00Z",
    }


@pytest.fixture
def producer(mock_confluent_producer, mock_schema_registry):
    """Create a TransactionProducer with mocked dependencies."""
    with patch("src.ingestion.kafka_producer.get_schema_registry", return_value=mock_schema_registry):
        with patch("src.ingestion.kafka_producer.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                get=MagicMock(side_effect=lambda key, default=None: {
                    "kafka.bootstrap_servers": "localhost:9092",
                    "kafka.producer.acks": "all",
                    "kafka.producer.retries": 3,
                    "kafka.producer.batch_size": 16384,
                    "kafka.producer.linger_ms": 10,
                }.get(key, default))
            )
            p = TransactionProducer(
                bootstrap_servers="localhost:9092",
                schema_registry=mock_schema_registry,
            )
            yield p
            # Prevent double close
            p._closed = True


# ============================================================================
# Producer Initialization Tests
# ============================================================================


class TestProducerInitialization:
    """Test producer creation and configuration."""

    def test_producer_initializes_with_defaults(self, producer, mock_confluent_producer):
        """Producer should initialize with sensible defaults."""
        assert producer._topic == "txn.raw.events"
        assert producer._bootstrap_servers == "localhost:9092"
        assert not producer.is_closed

    def test_producer_config_includes_idempotence(self, producer):
        """Producer should enable idempotent mode for exactly-once semantics."""
        # Verify config was built with idempotence enabled
        config = producer._build_config(
            MagicMock(get=MagicMock(return_value=None)), None
        )
        assert config["enable.idempotence"] is True
        assert config["compression.type"] == "snappy"

    def test_producer_custom_config_overrides(self, producer):
        """Custom config should override defaults."""
        config = producer._build_config(
            MagicMock(get=MagicMock(return_value=None)),
            {"batch.size": 32768, "bootstrap.servers": "kafka:29092"},
        )
        assert config["batch.size"] == 32768
        assert config["bootstrap.servers"] == "kafka:29092"


# ============================================================================
# Message Production Tests
# ============================================================================


class TestMessageProduction:
    """Test producing individual messages."""

    def test_produce_valid_transaction(
        self, producer, mock_confluent_producer, mock_schema_registry, valid_transaction
    ):
        """Should produce a valid transaction without errors."""
        producer.produce(valid_transaction)

        mock_schema_registry.validate.assert_called_once()
        mock_schema_registry.serialize.assert_called_once()
        mock_confluent_producer.produce.assert_called_once()

    def test_produce_uses_account_id_as_partition_key(
        self, producer, mock_confluent_producer, valid_transaction
    ):
        """Partition key should be the account_id for ordering guarantees."""
        producer.produce(valid_transaction)

        call_kwargs = mock_confluent_producer.produce.call_args[1]
        assert call_kwargs["key"] == b"ACC-12345"

    def test_produce_adds_event_metadata(
        self, producer, mock_schema_registry, valid_transaction
    ):
        """Should enrich transaction with event_id and event_timestamp."""
        producer.produce(valid_transaction)

        validated_record = mock_schema_registry.validate.call_args[0][1]
        assert "event_id" in validated_record
        assert "event_timestamp" in validated_record
        assert "event_version" in validated_record
        assert validated_record["event_version"] == "1.0.0"

    def test_produce_preserves_existing_event_id(
        self, producer, mock_schema_registry, valid_transaction
    ):
        """Should not override pre-existing event_id."""
        valid_transaction["event_id"] = "my-custom-event-id"
        producer.produce(valid_transaction)

        validated_record = mock_schema_registry.validate.call_args[0][1]
        assert validated_record["event_id"] == "my-custom-event-id"

    def test_produce_includes_headers(
        self, producer, mock_confluent_producer, valid_transaction
    ):
        """Should include event metadata in Kafka headers."""
        producer.produce(valid_transaction)

        call_kwargs = mock_confluent_producer.produce.call_args[1]
        headers = dict(call_kwargs["headers"])
        assert b"event_id" in {h[0].encode() if isinstance(h[0], str) else h[0] for h in call_kwargs["headers"]}

    def test_produce_custom_headers(
        self, producer, mock_confluent_producer, valid_transaction
    ):
        """Should merge custom headers with default headers."""
        producer.produce(valid_transaction, headers={"source": "test"})

        call_kwargs = mock_confluent_producer.produce.call_args[1]
        header_keys = [h[0] for h in call_kwargs["headers"]]
        assert "source" in header_keys

    def test_produce_to_custom_topic(
        self, producer, mock_confluent_producer, valid_transaction
    ):
        """Should allow overriding the target topic."""
        producer.produce(valid_transaction, topic="custom.topic")

        call_kwargs = mock_confluent_producer.produce.call_args[1]
        assert call_kwargs["topic"] == "custom.topic"

    def test_produce_raises_on_closed_producer(self, producer, valid_transaction):
        """Should raise ProducerError when producing after close."""
        producer._closed = True

        with pytest.raises(ProducerError, match="Producer is closed"):
            producer.produce(valid_transaction)

    def test_produce_raises_on_schema_validation_failure(
        self, producer, mock_schema_registry, valid_transaction
    ):
        """Should propagate SchemaValidationError for invalid records."""
        mock_schema_registry.validate.side_effect = SchemaValidationError(
            "Invalid field type"
        )

        with pytest.raises(SchemaValidationError):
            producer.produce(valid_transaction)

    def test_produce_raises_on_buffer_full(
        self, producer, mock_confluent_producer, valid_transaction
    ):
        """Should raise ProducerDeliveryError when buffer is full."""
        mock_confluent_producer.produce.side_effect = BufferError("Queue full")
        mock_confluent_producer.__len__.return_value = 100000

        with pytest.raises(ProducerDeliveryError, match="buffer full"):
            producer.produce(valid_transaction)


# ============================================================================
# Batch Production Tests
# ============================================================================


class TestBatchProduction:
    """Test batch message production."""

    def test_batch_produce_all_valid(
        self, producer, mock_confluent_producer, valid_transaction
    ):
        """Should produce all valid transactions in a batch."""
        transactions = [valid_transaction.copy() for _ in range(50)]
        result = producer.produce_batch(transactions)

        assert result["produced"] == 50
        assert result["failed"] == 0
        assert mock_confluent_producer.produce.call_count == 50

    def test_batch_produce_handles_partial_failures(
        self, producer, mock_schema_registry, valid_transaction
    ):
        """Should continue processing after individual failures."""
        transactions = [valid_transaction.copy() for _ in range(10)]

        # Make every 3rd validation fail
        call_count = [0]

        def side_effect(schema_name, record):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                raise SchemaValidationError("Bad record")
            return record

        mock_schema_registry.validate.side_effect = side_effect

        result = producer.produce_batch(transactions)

        assert result["failed"] == 3  # items 3, 6, 9
        assert result["produced"] == 7

    def test_batch_produce_polls_periodically(
        self, producer, mock_confluent_producer, valid_transaction
    ):
        """Should poll every 100 messages to handle callbacks."""
        transactions = [valid_transaction.copy() for _ in range(250)]
        producer.produce_batch(transactions)

        # Poll called at 100, 200, and final
        poll_calls = mock_confluent_producer.poll.call_count
        assert poll_calls >= 3


# ============================================================================
# Delivery Callback Tests
# ============================================================================


class TestDeliveryCallbacks:
    """Test delivery confirmation callbacks."""

    def test_successful_delivery_records_metrics(self, producer):
        """Successful delivery should update metrics."""
        mock_msg = MagicMock()
        mock_msg.topic.return_value = "txn.raw.events"
        mock_msg.partition.return_value = 3
        mock_msg.offset.return_value = 42
        mock_msg.value.return_value = b"x" * 100

        start_time = time.monotonic() - 0.005  # 5ms ago
        producer._default_delivery_callback(None, mock_msg, start_time)

        assert producer.metrics.messages_produced == 1
        assert producer.metrics.bytes_produced == 100

    def test_failed_delivery_records_error(self, producer):
        """Failed delivery should record error metrics."""
        mock_err = MagicMock()
        mock_err.__str__ = lambda self: "MSG_TIMED_OUT"
        mock_msg = MagicMock()
        mock_msg.topic.return_value = "txn.raw.events"
        mock_msg.partition.return_value = -1

        producer._default_delivery_callback(mock_err, mock_msg, time.monotonic())

        assert producer.metrics.messages_failed == 1
        assert producer.metrics.last_error is not None


# ============================================================================
# Metrics Tests
# ============================================================================


class TestProducerMetrics:
    """Test metrics collection and reporting."""

    def test_metrics_initial_state(self):
        """Metrics should start at zero."""
        metrics = ProducerMetrics()
        assert metrics.messages_produced == 0
        assert metrics.messages_failed == 0
        assert metrics.bytes_produced == 0
        assert metrics.average_latency_ms == 0.0
        assert metrics.error_rate == 0.0

    def test_metrics_record_success(self):
        """Should accumulate success metrics."""
        metrics = ProducerMetrics()
        metrics.record_success(byte_size=256, latency_ms=5.0)
        metrics.record_success(byte_size=512, latency_ms=10.0)

        assert metrics.messages_produced == 2
        assert metrics.bytes_produced == 768
        assert metrics.average_latency_ms == 7.5

    def test_metrics_record_failure(self):
        """Should track failures and error details."""
        metrics = ProducerMetrics()
        metrics.record_failure("Connection timeout")

        assert metrics.messages_failed == 1
        assert metrics.last_error == "Connection timeout"
        assert metrics.last_error_timestamp is not None

    def test_metrics_error_rate(self):
        """Error rate should be failures / total."""
        metrics = ProducerMetrics()
        metrics.record_success(100, 5.0)
        metrics.record_success(100, 5.0)
        metrics.record_success(100, 5.0)
        metrics.record_failure("error")

        assert metrics.error_rate == 0.25  # 1 fail / 4 total

    def test_metrics_snapshot(self):
        """Snapshot should return a serializable dict."""
        metrics = ProducerMetrics()
        metrics.record_success(256, 5.0)
        metrics.record_failure("test error")

        snapshot = metrics.snapshot()
        assert isinstance(snapshot, dict)
        assert snapshot["messages_produced"] == 1
        assert snapshot["messages_failed"] == 1
        assert snapshot["bytes_produced"] == 256
        assert snapshot["average_latency_ms"] == 5.0
        assert snapshot["error_rate"] == 0.5
        assert snapshot["last_error"] == "test error"


# ============================================================================
# Lifecycle Tests
# ============================================================================


class TestProducerLifecycle:
    """Test producer lifecycle (flush, close, context manager)."""

    def test_flush_returns_remaining_count(
        self, producer, mock_confluent_producer
    ):
        """Flush should return number of remaining messages."""
        mock_confluent_producer.flush.return_value = 0
        remaining = producer.flush()
        assert remaining == 0

    def test_flush_with_pending_messages(
        self, producer, mock_confluent_producer
    ):
        """Should warn when messages remain after flush timeout."""
        mock_confluent_producer.flush.return_value = 5
        remaining = producer.flush(timeout=1.0)
        assert remaining == 5

    def test_close_flushes_and_marks_closed(
        self, producer, mock_confluent_producer
    ):
        """Close should flush remaining messages and mark producer as closed."""
        producer.close()
        assert producer.is_closed
        mock_confluent_producer.flush.assert_called_once()

    def test_close_idempotent(self, producer, mock_confluent_producer):
        """Calling close multiple times should be safe."""
        producer.close()
        producer.close()
        # flush called only once
        assert mock_confluent_producer.flush.call_count == 1

    def test_context_manager(self, producer, mock_confluent_producer):
        """Should support context manager protocol with automatic cleanup."""
        # Test __enter__ returns self
        assert producer.__enter__() is producer
        assert not producer.is_closed

        # Test __exit__ calls close
        producer.__exit__(None, None, None)
        assert producer.is_closed
        mock_confluent_producer.flush.assert_called()


# ============================================================================
# Partition Key Strategy Tests
# ============================================================================


class TestPartitionKeyStrategy:
    """Test that partitioning correctly uses account_id."""

    def test_same_account_same_key(
        self, producer, mock_confluent_producer, valid_transaction
    ):
        """Transactions from same account should use same partition key."""
        txn1 = valid_transaction.copy()
        txn2 = valid_transaction.copy()
        txn2["external_transaction_id"] = "TXN-DIFFERENT"

        producer.produce(txn1)
        producer.produce(txn2)

        calls = mock_confluent_producer.produce.call_args_list
        key1 = calls[0][1]["key"]
        key2 = calls[1][1]["key"]
        assert key1 == key2 == b"ACC-12345"

    def test_different_accounts_different_keys(
        self, producer, mock_confluent_producer, valid_transaction
    ):
        """Transactions from different accounts should have different keys."""
        txn1 = valid_transaction.copy()
        txn1["account_id"] = "ACC-11111"

        txn2 = valid_transaction.copy()
        txn2["account_id"] = "ACC-22222"

        producer.produce(txn1)
        producer.produce(txn2)

        calls = mock_confluent_producer.produce.call_args_list
        key1 = calls[0][1]["key"]
        key2 = calls[1][1]["key"]
        assert key1 == b"ACC-11111"
        assert key2 == b"ACC-22222"
        assert key1 != key2
