"""SQLAlchemy ORM models for RiskPulse PostgreSQL operational layer."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    type_annotation_map = {
        dict[str, Any]: JSONB,
    }


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    external_transaction_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    account_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    merchant_id: Mapped[str | None] = mapped_column(String(64))
    merchant_name: Mapped[str | None] = mapped_column(String(255))
    merchant_category_code: Mapped[str | None] = mapped_column(String(10))
    transaction_amount: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    transaction_currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    transaction_type: Mapped[str] = mapped_column(String(20), nullable=False)
    channel: Mapped[str] = mapped_column(String(20), nullable=False)
    card_type: Mapped[str | None] = mapped_column(String(20))
    card_last_four: Mapped[str | None] = mapped_column(String(4))
    ip_address: Mapped[str | None] = mapped_column(INET)
    device_id: Mapped[str | None] = mapped_column(String(128))
    device_type: Mapped[str | None] = mapped_column(String(50))
    geo_latitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 8))
    geo_longitude: Mapped[Decimal | None] = mapped_column(Numeric(11, 8))
    geo_country: Mapped[str | None] = mapped_column(String(3))
    geo_city: Mapped[str | None] = mapped_column(String(100))
    is_international: Mapped[bool] = mapped_column(Boolean, default=False)
    transaction_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    processed_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    alerts: Mapped[list["FraudAlert"]] = relationship(back_populates="transaction", lazy="selectin")
    risk_scores: Mapped[list["RiskScore"]] = relationship(
        back_populates="transaction", lazy="selectin"
    )

    __table_args__ = (
        CheckConstraint("transaction_amount > 0", name="ck_transactions_amount_positive"),
        CheckConstraint(
            "transaction_type IN ('purchase', 'withdrawal', 'transfer', 'refund')",
            name="ck_transactions_type",
        ),
        CheckConstraint(
            "channel IN ('online', 'pos', 'atm', 'mobile')",
            name="ck_transactions_channel",
        ),
        CheckConstraint(
            "status IN ('pending', 'approved', 'declined', 'flagged')",
            name="ck_transactions_status",
        ),
        Index("idx_transactions_timestamp", "transaction_timestamp"),
        Index("idx_transactions_status_timestamp", "status", "transaction_timestamp"),
    )


class FraudAlert(Base):
    __tablename__ = "fraud_alerts"

    alert_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    rule_id: Mapped[str | None] = mapped_column(String(50))
    risk_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open")
    description: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    assigned_to: Mapped[str | None] = mapped_column(String(100))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    transaction: Mapped["Transaction"] = relationship(back_populates="alerts")

    __table_args__ = (
        CheckConstraint(
            "alert_type IN ('rule_based', 'anomaly', 'ml_score', 'ensemble')",
            name="ck_fraud_alerts_type",
        ),
        CheckConstraint(
            "severity IN ('low', 'medium', 'high', 'critical')",
            name="ck_fraud_alerts_severity",
        ),
        CheckConstraint(
            "status IN ('open', 'investigating', 'resolved', 'false_positive')",
            name="ck_fraud_alerts_status",
        ),
        CheckConstraint("risk_score BETWEEN 0 AND 1", name="ck_fraud_alerts_score_range"),
        Index("idx_fraud_alerts_status_severity", "status", "severity", "created_at"),
    )


class RiskScore(Base):
    __tablename__ = "risk_scores"

    score_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    overall_score: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    rule_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    anomaly_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    ml_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    feature_contributions: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    scoring_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer)

    # Relationships
    transaction: Mapped["Transaction"] = relationship(back_populates="risk_scores")

    __table_args__ = (
        CheckConstraint("overall_score BETWEEN 0 AND 1", name="ck_risk_scores_overall_range"),
        CheckConstraint("rule_score BETWEEN 0 AND 1", name="ck_risk_scores_rule_range"),
        CheckConstraint("anomaly_score BETWEEN 0 AND 1", name="ck_risk_scores_anomaly_range"),
        CheckConstraint("ml_score BETWEEN 0 AND 1", name="ck_risk_scores_ml_range"),
        CheckConstraint("latency_ms >= 0", name="ck_risk_scores_latency_positive"),
        Index("idx_risk_scores_overall", "overall_score"),
        Index("idx_risk_scores_timestamp", "scoring_timestamp"),
    )


class CustomerProfile(Base):
    __tablename__ = "customer_profiles"

    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    total_transactions_24h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_amount_24h: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False, default=0)
    total_transactions_7d: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_amount_7d: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False, default=0)
    avg_transaction_amount: Mapped[Decimal | None] = mapped_column(Numeric(15, 2))
    max_transaction_amount: Mapped[Decimal | None] = mapped_column(Numeric(15, 2))
    unique_merchants_7d: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unique_countries_7d: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_transaction_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    risk_tier: Mapped[str] = mapped_column(String(20), nullable=False, default="standard")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "risk_tier IN ('low', 'standard', 'elevated', 'high')",
            name="ck_customer_profiles_risk_tier",
        ),
        Index("idx_customer_profiles_risk_tier", "risk_tier"),
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    log_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    actor: Mapped[str | None] = mapped_column(String(100))
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    ip_address: Mapped[str | None] = mapped_column(INET)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_audit_logs_entity", "entity_type", "entity_id"),
        Index("idx_audit_logs_created", "created_at"),
        Index("idx_audit_logs_event_type", "event_type"),
    )
