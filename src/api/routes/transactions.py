"""Transaction ingestion API routes.

Provides endpoints for submitting, retrieving, and listing transactions.
Validated transactions are published to Kafka for downstream processing.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from src.api.middleware.auth import require_permission, verify_api_key
from src.api.schemas.transaction_schema import (
    BatchSubmitResponse,
    ErrorResponse,
    TransactionBatchCreate,
    TransactionCreate,
    TransactionFilter,
    TransactionListResponse,
    TransactionResponse,
    TransactionSubmitResponse,
)
from src.utils.constants import API_PREFIX, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, TOPIC_RAW_EVENTS

logger = structlog.get_logger(__name__)

router = APIRouter(prefix=f"{API_PREFIX}/transactions", tags=["Transactions"])


@router.post(
    "",
    response_model=TransactionSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a single transaction",
    description="Submit a single transaction for fraud analysis. The transaction is validated and published to Kafka.",
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        422: {"description": "Validation error"},
        503: {"description": "Kafka unavailable"},
    },
)
async def submit_transaction(
    transaction: TransactionCreate,
    request: Request,
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> TransactionSubmitResponse:
    """Submit a single transaction for processing.

    The transaction is validated via Pydantic, assigned an internal ID,
    and published to the Kafka raw events topic for downstream processing.
    """
    transaction_id = uuid.uuid4()
    correlation_id = getattr(request.state, "correlation_id", None)

    logger.info(
        "transaction_received",
        transaction_id=str(transaction_id),
        external_id=transaction.external_transaction_id,
        account_id=transaction.account_id,
        amount=float(transaction.transaction_amount),
        correlation_id=correlation_id,
    )

    # Prepare event for Kafka
    event = _build_kafka_event(transaction_id, transaction)

    # Publish to Kafka
    try:
        producer = _get_kafka_producer()
        if producer is not None:
            producer.produce(event, topic=TOPIC_RAW_EVENTS)
            logger.info(
                "transaction_published",
                transaction_id=str(transaction_id),
                topic=TOPIC_RAW_EVENTS,
            )
        else:
            logger.warning("kafka_producer_unavailable", transaction_id=str(transaction_id))
    except Exception as exc:
        logger.error(
            "transaction_publish_failed",
            transaction_id=str(transaction_id),
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to publish transaction to processing pipeline. Please retry.",
        )

    return TransactionSubmitResponse(
        transaction_id=transaction_id,
        external_transaction_id=transaction.external_transaction_id,
        status="accepted",
        message="Transaction accepted for processing",
    )


@router.post(
    "/batch",
    response_model=BatchSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a batch of transactions",
    description="Submit up to 1000 transactions in a single request for bulk processing.",
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        422: {"description": "Validation error"},
    },
)
async def submit_batch(
    batch: TransactionBatchCreate,
    request: Request,
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> BatchSubmitResponse:
    """Submit a batch of transactions for processing.

    Each transaction is validated individually. Valid transactions are published
    to Kafka; invalid ones are reported in the error list.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    accepted: list[TransactionSubmitResponse] = []
    errors: list[dict[str, Any]] = []

    logger.info(
        "batch_received",
        batch_size=len(batch.transactions),
        correlation_id=correlation_id,
    )

    producer = _get_kafka_producer()
    events: list[dict[str, Any]] = []

    for idx, transaction in enumerate(batch.transactions):
        transaction_id = uuid.uuid4()
        event = _build_kafka_event(transaction_id, transaction)
        events.append(event)
        accepted.append(
            TransactionSubmitResponse(
                transaction_id=transaction_id,
                external_transaction_id=transaction.external_transaction_id,
                status="accepted",
                message="Transaction accepted for processing",
            )
        )

    # Publish batch to Kafka
    if producer is not None and events:
        try:
            results = producer.produce_batch(events, topic=TOPIC_RAW_EVENTS)
            logger.info(
                "batch_published",
                total=len(events),
                correlation_id=correlation_id,
            )
        except Exception as exc:
            logger.error("batch_publish_partial_failure", error=str(exc))

    return BatchSubmitResponse(
        accepted=len(accepted),
        rejected=len(errors),
        transactions=accepted,
        errors=errors,
    )


@router.get(
    "/{transaction_id}",
    response_model=TransactionResponse,
    summary="Retrieve a transaction by ID",
    description="Get details of a specific transaction by its internal UUID.",
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        404: {"model": ErrorResponse, "description": "Transaction not found"},
    },
)
async def get_transaction(
    transaction_id: uuid.UUID,
    request: Request,
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> TransactionResponse:
    """Retrieve a single transaction by its internal ID."""
    # In production, this queries PostgreSQL
    storage = _get_storage()
    transaction = await storage.get_transaction(transaction_id)

    if transaction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {transaction_id} not found",
        )

    return transaction


@router.get(
    "",
    response_model=TransactionListResponse,
    summary="List transactions with filters",
    description="Retrieve a paginated list of transactions with optional filtering.",
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
    },
)
async def list_transactions(
    request: Request,
    account_id: str | None = Query(None, description="Filter by account ID"),
    customer_id: str | None = Query(None, description="Filter by customer ID"),
    status_filter: str | None = Query(None, alias="status", description="Filter by status"),
    transaction_type: str | None = Query(None, description="Filter by transaction type"),
    channel: str | None = Query(None, description="Filter by channel"),
    min_amount: Decimal | None = Query(None, ge=0, description="Minimum amount filter"),
    max_amount: Decimal | None = Query(None, ge=0, description="Maximum amount filter"),
    start_date: datetime | None = Query(None, description="Start date filter (ISO 8601)"),
    end_date: datetime | None = Query(None, description="End date filter (ISO 8601)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Items per page"),
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> TransactionListResponse:
    """List transactions with pagination and filtering."""
    filters = TransactionFilter(
        account_id=account_id,
        customer_id=customer_id,
        status=status_filter,
        transaction_type=transaction_type,
        channel=channel,
        min_amount=min_amount,
        max_amount=max_amount,
        start_date=start_date,
        end_date=end_date,
        page=page,
        page_size=page_size,
    )

    storage = _get_storage()
    items, total = await storage.list_transactions(filters)

    pages = (total + page_size - 1) // page_size if total > 0 else 0

    return TransactionListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


# --- Internal helpers ---


def _build_kafka_event(transaction_id: uuid.UUID, transaction: TransactionCreate) -> dict[str, Any]:
    """Convert a validated transaction to a Kafka event payload."""
    return {
        "transaction_id": str(transaction_id),
        "external_transaction_id": transaction.external_transaction_id,
        "account_id": transaction.account_id,
        "customer_id": transaction.customer_id,
        "merchant_id": transaction.merchant_id,
        "merchant_name": transaction.merchant_name,
        "merchant_category_code": transaction.merchant_category_code,
        "transaction_amount": float(transaction.transaction_amount),
        "transaction_currency": transaction.transaction_currency,
        "transaction_type": transaction.transaction_type,
        "channel": transaction.channel,
        "card_type": transaction.card_type,
        "card_last_four": transaction.card_last_four,
        "ip_address": transaction.ip_address,
        "device_id": transaction.device_id,
        "device_type": transaction.device_type,
        "geo_latitude": float(transaction.geo_latitude) if transaction.geo_latitude else None,
        "geo_longitude": float(transaction.geo_longitude) if transaction.geo_longitude else None,
        "geo_country": transaction.geo_country,
        "geo_city": transaction.geo_city,
        "is_international": transaction.is_international,
        "transaction_timestamp": transaction.transaction_timestamp.isoformat(),
        "metadata": transaction.metadata,
    }


def _get_kafka_producer():
    """Get the Kafka producer instance from the application state.

    Returns None if the producer is not available (graceful degradation).
    """
    try:
        from src.ingestion.kafka_producer import TransactionProducer

        return TransactionProducer.from_settings()
    except Exception as exc:
        logger.warning("kafka_producer_init_failed", error=str(exc))
        return None


class _TransactionStorage:
    """Abstraction for transaction storage operations.

    In production, this interfaces with PostgreSQL via SQLAlchemy.
    Designed to be replaceable with mock implementations for testing.
    """

    async def get_transaction(self, transaction_id: uuid.UUID) -> TransactionResponse | None:
        """Retrieve a transaction by ID from the database."""
        try:
            import asyncpg

            from src.utils.config import get_settings

            settings = get_settings()
            conn = await asyncpg.connect(dsn=settings.database_url.replace("+asyncpg", ""))
            row = await conn.fetchrow(
                "SELECT * FROM transactions WHERE transaction_id = $1",
                transaction_id,
            )
            await conn.close()

            if row is None:
                return None

            return TransactionResponse(**dict(row))
        except Exception as exc:
            logger.error("storage_get_failed", transaction_id=str(transaction_id), error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Storage service temporarily unavailable",
            )

    async def list_transactions(self, filters: TransactionFilter) -> tuple[list[TransactionResponse], int]:
        """List transactions with filtering and pagination."""
        try:
            import asyncpg

            from src.utils.config import get_settings

            settings = get_settings()
            conn = await asyncpg.connect(dsn=settings.database_url.replace("+asyncpg", ""))

            # Build dynamic query
            conditions: list[str] = []
            params: list[Any] = []
            param_idx = 1

            if filters.account_id:
                conditions.append(f"account_id = ${param_idx}")
                params.append(filters.account_id)
                param_idx += 1
            if filters.customer_id:
                conditions.append(f"customer_id = ${param_idx}")
                params.append(filters.customer_id)
                param_idx += 1
            if filters.status:
                conditions.append(f"status = ${param_idx}")
                params.append(filters.status)
                param_idx += 1
            if filters.transaction_type:
                conditions.append(f"transaction_type = ${param_idx}")
                params.append(filters.transaction_type)
                param_idx += 1
            if filters.channel:
                conditions.append(f"channel = ${param_idx}")
                params.append(filters.channel)
                param_idx += 1
            if filters.min_amount is not None:
                conditions.append(f"transaction_amount >= ${param_idx}")
                params.append(float(filters.min_amount))
                param_idx += 1
            if filters.max_amount is not None:
                conditions.append(f"transaction_amount <= ${param_idx}")
                params.append(float(filters.max_amount))
                param_idx += 1
            if filters.start_date:
                conditions.append(f"transaction_timestamp >= ${param_idx}")
                params.append(filters.start_date)
                param_idx += 1
            if filters.end_date:
                conditions.append(f"transaction_timestamp <= ${param_idx}")
                params.append(filters.end_date)
                param_idx += 1

            where_clause = " AND ".join(conditions) if conditions else "TRUE"
            offset = (filters.page - 1) * filters.page_size

            # Get total count
            count_query = f"SELECT COUNT(*) FROM transactions WHERE {where_clause}"
            total = await conn.fetchval(count_query, *params)

            # Get paginated results
            data_query = (
                f"SELECT * FROM transactions WHERE {where_clause} "
                f"ORDER BY created_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
            )
            params.extend([filters.page_size, offset])
            rows = await conn.fetch(data_query, *params)
            await conn.close()

            items = [TransactionResponse(**dict(row)) for row in rows]
            return items, total or 0
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("storage_list_failed", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Storage service temporarily unavailable",
            )


# Module-level storage instance
_storage: _TransactionStorage | None = None


def _get_storage() -> _TransactionStorage:
    """Get or create the storage instance."""
    global _storage
    if _storage is None:
        _storage = _TransactionStorage()
    return _storage


def set_storage(storage: Any) -> None:
    """Replace the storage instance (for testing)."""
    global _storage
    _storage = storage
