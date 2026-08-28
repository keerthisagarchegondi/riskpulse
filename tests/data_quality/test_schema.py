"""Schema, relationship, and business-rule data quality checks."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest

from src.utils.constants import (
    CARD_TYPES,
    CHANNELS,
    SUPPORTED_CURRENCIES,
    TRANSACTION_TYPES,
)

REQUIRED_COLUMNS = {
    "customers": {"customer_id", "risk_tier", "home_country", "created_at"},
    "accounts": {"account_id", "customer_id", "account_status", "opened_at"},
    "transactions": {
        "transaction_id",
        "external_transaction_id",
        "account_id",
        "customer_id",
        "merchant_id",
        "transaction_amount",
        "transaction_currency",
        "transaction_type",
        "channel",
        "card_type",
        "device_id",
        "geo_country",
        "is_international",
        "transaction_timestamp",
        "ingested_at",
        "status",
        "is_fraud",
    },
    "fraud_alerts": {"alert_id", "transaction_id", "severity", "status", "created_at"},
    "model_scores": {
        "score_id",
        "transaction_id",
        "model_version",
        "risk_score",
        "score_timestamp",
    },
}


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.mark.data_quality
def test_all_required_tables_are_present(data_quality_dataset: dict[str, Any]) -> None:
    assert REQUIRED_COLUMNS.keys() <= data_quality_dataset.keys()


@pytest.mark.data_quality
@pytest.mark.parametrize("table_name,required", REQUIRED_COLUMNS.items())
def test_table_schema_conformance(
    data_quality_dataset: dict[str, Any],
    table_name: str,
    required: set[str],
) -> None:
    rows = data_quality_dataset[table_name]
    assert rows, f"{table_name} should not be empty"
    for row in rows:
        assert required <= row.keys(), f"{table_name} missing columns: {required - row.keys()}"


@pytest.mark.data_quality
def test_table_column_types_are_parseable(data_quality_dataset: dict[str, Any]) -> None:
    transaction = data_quality_dataset["transactions"][0]
    score = data_quality_dataset["model_scores"][0]
    account = data_quality_dataset["accounts"][0]

    assert Decimal(str(transaction["transaction_amount"])) > 0
    assert isinstance(transaction["is_international"], bool)
    assert isinstance(transaction["is_fraud"], bool)
    assert _parse_dt(transaction["transaction_timestamp"]).tzinfo is not None
    assert _parse_dt(transaction["ingested_at"]).tzinfo is not None
    assert 0.0 <= float(score["risk_score"]) <= 1.0
    assert account["account_status"] == "active"


@pytest.mark.data_quality
def test_primary_keys_are_unique(data_quality_dataset: dict[str, Any]) -> None:
    for table_name, key in (
        ("customers", "customer_id"),
        ("accounts", "account_id"),
        ("transactions", "transaction_id"),
        ("fraud_alerts", "alert_id"),
        ("model_scores", "score_id"),
    ):
        values = [row[key] for row in data_quality_dataset[table_name]]
        assert len(values) == len(set(values)), f"{table_name}.{key} contains duplicates"


@pytest.mark.data_quality
def test_referential_integrity_across_tables(data_quality_dataset: dict[str, Any]) -> None:
    customers = {row["customer_id"] for row in data_quality_dataset["customers"]}
    accounts = {row["account_id"] for row in data_quality_dataset["accounts"]}
    transactions = {row["transaction_id"] for row in data_quality_dataset["transactions"]}

    assert all(row["customer_id"] in customers for row in data_quality_dataset["accounts"])
    assert all(row["customer_id"] in customers for row in data_quality_dataset["transactions"])
    assert all(row["account_id"] in accounts for row in data_quality_dataset["transactions"])
    assert all(
        row["transaction_id"] in transactions for row in data_quality_dataset["fraud_alerts"]
    )
    assert all(
        row["transaction_id"] in transactions for row in data_quality_dataset["model_scores"]
    )


@pytest.mark.data_quality
def test_transaction_business_rules(data_quality_dataset: dict[str, Any]) -> None:
    for row in data_quality_dataset["transactions"]:
        assert float(row["transaction_amount"]) > 0.0
        assert row["transaction_currency"] in SUPPORTED_CURRENCIES
        assert row["transaction_type"] in TRANSACTION_TYPES
        assert row["channel"] in CHANNELS
        assert row["card_type"] in CARD_TYPES
        assert -90.0 <= float(row["geo_latitude"]) <= 90.0
        assert -180.0 <= float(row["geo_longitude"]) <= 180.0


@pytest.mark.data_quality
def test_alert_and_score_business_rules(data_quality_dataset: dict[str, Any]) -> None:
    valid_severities = {"low", "medium", "high", "critical"}
    valid_alert_statuses = {"open", "investigating", "resolved", "false_positive"}

    for alert in data_quality_dataset["fraud_alerts"]:
        assert alert["severity"] in valid_severities
        assert alert["status"] in valid_alert_statuses
    for score in data_quality_dataset["model_scores"]:
        assert 0.0 <= float(score["risk_score"]) <= 1.0
        assert score["model_version"].startswith("prod-")


@pytest.mark.data_quality
def test_test_fixtures_are_anonymized(data_quality_dataset: dict[str, Any]) -> None:
    for row in data_quality_dataset["transactions"]:
        assert row["customer_id"].startswith("ANON-")
        assert row["account_id"].startswith("ANON-")
        assert row["device_id"].startswith("ANON-")
        assert row["ip_address"] == "192.0.2.0"
