"""Data ingestion module - Kafka producers, consumers, and API ingestion."""

from src.ingestion.api_ingestion import (
    BatchIngestionHandler,
    DetectedSchema,
    FileFormat,
    IngestionError,
    IngestionResult,
    IngestionStatus,
)
from src.ingestion.kafka_admin import KafkaAdminError, KafkaTopicManager
from src.ingestion.kafka_consumer import ConsumerMetrics, TransactionConsumer
from src.ingestion.kafka_producer import ProducerDeliveryError, ProducerError, TransactionProducer
from src.ingestion.schema_registry import SchemaRegistry, SchemaRegistryError, SchemaValidationError

__all__ = [
    "ConsumerMetrics",
    "TransactionConsumer",
    "TransactionProducer",
    "ProducerError",
    "ProducerDeliveryError",
    "SchemaRegistry",
    "SchemaRegistryError",
    "SchemaValidationError",
    "KafkaTopicManager",
    "KafkaAdminError",
    "BatchIngestionHandler",
    "IngestionResult",
    "IngestionStatus",
    "IngestionError",
    "FileFormat",
    "DetectedSchema",
]
