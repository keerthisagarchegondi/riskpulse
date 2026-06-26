"""Data ingestion module - Kafka producers, consumers, and API ingestion."""

from src.ingestion.kafka_producer import TransactionProducer, ProducerError, ProducerDeliveryError
from src.ingestion.schema_registry import SchemaRegistry, SchemaRegistryError, SchemaValidationError
from src.ingestion.kafka_admin import KafkaTopicManager, KafkaAdminError
from src.ingestion.api_ingestion import (
    BatchIngestionHandler,
    IngestionResult,
    IngestionStatus,
    IngestionError,
    FileFormat,
    DetectedSchema,
)

__all__ = [
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
