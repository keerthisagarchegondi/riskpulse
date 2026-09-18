"""Storage module with local and cloud storage backends."""

from importlib import import_module
from typing import Any

from src.storage.cache_handler import (
    CacheConnectionError,
    CacheHandler,
    CacheHandlerError,
    CacheMetrics,
    CacheOperationError,
    CacheStrategy,
    create_cache_handler,
)
from src.storage.local_object_storage import (
    LocalObjectNotFoundError,
    LocalObjectStorageClient,
    LocalObjectStorageError,
    LocalObjectStorageHandler,
)
from src.storage.models import (
    AuditLog,
    Base,
    CustomerProfile,
    FraudAlert,
    RiskScore,
    Transaction,
)
from src.storage.postgres_handler import (
    BulkOperationError,
    ConnectionError,
    PoolStats,
    PostgresHandler,
    PostgresHandlerError,
    QueryError,
    QueryMetrics,
    create_postgres_handler,
)
from src.storage.s3_handler import (
    S3_BUCKET_ARCHIVE,
    S3_BUCKET_MODELS,
    S3_BUCKET_PROCESSED,
    S3_BUCKET_RAW,
    S3ConfigError,
    S3DownloadError,
    S3Handler,
    S3HandlerError,
    S3Metrics,
    S3UploadError,
    StorageLayer,
    get_s3_handler,
)

_LAZY_EXPORTS = {
    "LocalWarehouseHandler": ("src.storage.local_warehouse_handler", "LocalWarehouseHandler"),
    "SnowflakeHandler": ("src.storage.snowflake_handler", "SnowflakeHandler"),
    "SnowflakeHandlerError": ("src.storage.snowflake_handler", "SnowflakeHandlerError"),
    "SnowflakeConnectionError": ("src.storage.snowflake_handler", "SnowflakeConnectionError"),
    "SnowflakeQueryError": ("src.storage.snowflake_handler", "SnowflakeQueryError"),
    "SnowflakeLoadError": ("src.storage.snowflake_handler", "SnowflakeLoadError"),
    "SnowflakeSchemaError": ("src.storage.snowflake_handler", "SnowflakeSchemaError"),
    "SnowflakeMetrics": ("src.storage.snowflake_handler", "SnowflakeMetrics"),
    "LoadMetrics": ("src.storage.snowflake_handler", "LoadMetrics"),
    "LoadStrategy": ("src.storage.snowflake_handler", "LoadStrategy"),
    "FileFormat": ("src.storage.snowflake_handler", "FileFormat"),
    "QueryResult": ("src.storage.snowflake_handler", "QueryResult"),
    "WatermarkState": ("src.storage.snowflake_handler", "WatermarkState"),
    "create_snowflake_handler": ("src.storage.snowflake_handler", "create_snowflake_handler"),
    "SCHEMA_RAW": ("src.storage.snowflake_handler", "SCHEMA_RAW"),
    "SCHEMA_STAGING": ("src.storage.snowflake_handler", "SCHEMA_STAGING"),
    "SCHEMA_ANALYTICS": ("src.storage.snowflake_handler", "SCHEMA_ANALYTICS"),
    "SCHEMA_REPORTING": ("src.storage.snowflake_handler", "SCHEMA_REPORTING"),
    "StorageOrchestrator": ("src.storage.storage_orchestrator", "StorageOrchestrator"),
    "StorageBackend": ("src.storage.storage_orchestrator", "StorageBackend"),
    "StorageState": ("src.storage.storage_orchestrator", "StorageState"),
    "BackendHealth": ("src.storage.storage_orchestrator", "BackendHealth"),
    "WriteResult": ("src.storage.storage_orchestrator", "WriteResult"),
    "OrchestratedWriteResult": (
        "src.storage.storage_orchestrator",
        "OrchestratedWriteResult",
    ),
    "create_storage_orchestrator": (
        "src.storage.storage_orchestrator",
        "create_storage_orchestrator",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    # PostgreSQL
    "PostgresHandler",
    "PostgresHandlerError",
    "ConnectionError",
    "QueryError",
    "BulkOperationError",
    "PoolStats",
    "QueryMetrics",
    "create_postgres_handler",
    # ORM Models
    "Base",
    "Transaction",
    "FraudAlert",
    "RiskScore",
    "CustomerProfile",
    "AuditLog",
    # S3
    "S3Handler",
    "S3HandlerError",
    "S3UploadError",
    "S3DownloadError",
    "S3ConfigError",
    "S3Metrics",
    "StorageLayer",
    "get_s3_handler",
    "S3_BUCKET_RAW",
    "S3_BUCKET_PROCESSED",
    "S3_BUCKET_MODELS",
    "S3_BUCKET_ARCHIVE",
    "LocalObjectStorageHandler",
    "LocalObjectStorageClient",
    "LocalObjectStorageError",
    "LocalObjectNotFoundError",
    "LocalWarehouseHandler",
    # Snowflake
    "SnowflakeHandler",
    "SnowflakeHandlerError",
    "SnowflakeConnectionError",
    "SnowflakeQueryError",
    "SnowflakeLoadError",
    "SnowflakeSchemaError",
    "SnowflakeMetrics",
    "LoadMetrics",
    "LoadStrategy",
    "FileFormat",
    "QueryResult",
    "WatermarkState",
    "create_snowflake_handler",
    "SCHEMA_RAW",
    "SCHEMA_STAGING",
    "SCHEMA_ANALYTICS",
    "SCHEMA_REPORTING",
    # Redis Cache
    "CacheHandler",
    "CacheHandlerError",
    "CacheConnectionError",
    "CacheOperationError",
    "CacheMetrics",
    "CacheStrategy",
    "create_cache_handler",
    # Storage Orchestrator
    "StorageOrchestrator",
    "StorageBackend",
    "StorageState",
    "BackendHealth",
    "WriteResult",
    "OrchestratedWriteResult",
    "create_storage_orchestrator",
]
