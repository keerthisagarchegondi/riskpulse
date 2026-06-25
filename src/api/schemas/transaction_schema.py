"""Pydantic models for transaction API request/response validation."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.utils.constants import (
    CARD_TYPES,
    CHANNELS,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    SUPPORTED_CURRENCIES,
    TRANSACTION_TYPES,
)


class TransactionCreate(BaseModel):
    """Schema for creating a new transaction."""

    model_config = ConfigDict(str_strip_whitespace=True)

    external_transaction_id: str = Field(
        ..., min_length=1, max_length=64, description="Unique external identifier for the transaction"
    )
    account_id: str = Field(..., min_length=1, max_length=64, description="Account identifier")
    customer_id: str = Field(..., min_length=1, max_length=64, description="Customer identifier")
    merchant_id: str | None = Field(None, max_length=64, description="Merchant identifier")
    merchant_name: str | None = Field(None, max_length=255, description="Merchant name")
    merchant_category_code: str | None = Field(None, max_length=10, description="MCC code")
    transaction_amount: Decimal = Field(..., gt=0, max_digits=15, decimal_places=2, description="Transaction amount")
    transaction_currency: str = Field("USD", max_length=3, description="ISO 4217 currency code")
    transaction_type: str = Field(..., description="Type of transaction")
    channel: str = Field(..., description="Transaction channel")
    card_type: str | None = Field(None, description="Card type")
    card_last_four: str | None = Field(None, min_length=4, max_length=4, pattern=r"^\d{4}$")
    ip_address: str | None = Field(None, max_length=45, description="Client IP address")
    device_id: str | None = Field(None, max_length=128, description="Device fingerprint")
    device_type: str | None = Field(None, max_length=50, description="Device type")
    geo_latitude: Decimal | None = Field(None, ge=-90, le=90, description="Transaction latitude")
    geo_longitude: Decimal | None = Field(None, ge=-180, le=180, description="Transaction longitude")
    geo_country: str | None = Field(None, max_length=3, description="ISO 3166-1 alpha-3 country code")
    geo_city: str | None = Field(None, max_length=100, description="City name")
    is_international: bool = Field(False, description="Whether this is an international transaction")
    transaction_timestamp: datetime = Field(..., description="When the transaction occurred")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")

    @field_validator("transaction_currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        v_upper = v.upper()
        if v_upper not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency: {v}. Supported: {', '.join(SUPPORTED_CURRENCIES)}")
        return v_upper

    @field_validator("transaction_type")
    @classmethod
    def validate_transaction_type(cls, v: str) -> str:
        v_lower = v.lower()
        if v_lower not in TRANSACTION_TYPES:
            raise ValueError(f"Invalid transaction type: {v}. Allowed: {', '.join(TRANSACTION_TYPES)}")
        return v_lower

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v: str) -> str:
        v_lower = v.lower()
        if v_lower not in CHANNELS:
            raise ValueError(f"Invalid channel: {v}. Allowed: {', '.join(CHANNELS)}")
        return v_lower

    @field_validator("card_type")
    @classmethod
    def validate_card_type(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v_lower = v.lower()
        if v_lower not in CARD_TYPES:
            raise ValueError(f"Invalid card type: {v}. Allowed: {', '.join(CARD_TYPES)}")
        return v_lower

    @field_validator("ip_address")
    @classmethod
    def validate_ip_address(cls, v: str | None) -> str | None:
        if v is None:
            return v
        import ipaddress

        try:
            ipaddress.ip_address(v)
        except ValueError:
            raise ValueError(f"Invalid IP address: {v}")
        return v

    @model_validator(mode="after")
    def validate_geo_fields(self) -> "TransactionCreate":
        lat = self.geo_latitude
        lon = self.geo_longitude
        if (lat is not None) != (lon is not None):
            raise ValueError("Both geo_latitude and geo_longitude must be provided together")
        return self


class TransactionBatchCreate(BaseModel):
    """Schema for batch transaction submission."""

    transactions: list[TransactionCreate] = Field(
        ..., min_length=1, max_length=1000, description="List of transactions (max 1000)"
    )


class TransactionResponse(BaseModel):
    """Schema for transaction response."""

    model_config = ConfigDict(from_attributes=True)

    transaction_id: uuid.UUID
    external_transaction_id: str
    account_id: str
    customer_id: str
    merchant_id: str | None = None
    merchant_name: str | None = None
    merchant_category_code: str | None = None
    transaction_amount: Decimal
    transaction_currency: str
    transaction_type: str
    channel: str
    card_type: str | None = None
    card_last_four: str | None = None
    ip_address: str | None = None
    device_id: str | None = None
    device_type: str | None = None
    geo_latitude: Decimal | None = None
    geo_longitude: Decimal | None = None
    geo_country: str | None = None
    geo_city: str | None = None
    is_international: bool
    transaction_timestamp: datetime
    processed_timestamp: datetime | None = None
    status: str
    created_at: datetime
    updated_at: datetime


class TransactionSubmitResponse(BaseModel):
    """Response after submitting a transaction for processing."""

    transaction_id: uuid.UUID
    external_transaction_id: str
    status: str = "accepted"
    message: str = "Transaction accepted for processing"


class BatchSubmitResponse(BaseModel):
    """Response after submitting a batch of transactions."""

    accepted: int
    rejected: int
    transactions: list[TransactionSubmitResponse]
    errors: list[dict[str, Any]] = Field(default_factory=list)


class TransactionListResponse(BaseModel):
    """Paginated list of transactions."""

    items: list[TransactionResponse]
    total: int
    page: int
    page_size: int
    pages: int


class TransactionFilter(BaseModel):
    """Query parameters for filtering transactions."""

    account_id: str | None = None
    customer_id: str | None = None
    status: str | None = None
    transaction_type: str | None = None
    channel: str | None = None
    min_amount: Decimal | None = Field(None, ge=0)
    max_amount: Decimal | None = Field(None, ge=0)
    start_date: datetime | None = None
    end_date: datetime | None = None
    page: int = Field(1, ge=1)
    page_size: int = Field(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    request_id: str | None = None
