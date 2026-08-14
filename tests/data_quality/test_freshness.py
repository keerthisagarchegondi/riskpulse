"""Freshness SLA and volume anomaly data quality checks."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pytest


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.mark.data_quality
def test_transactions_meet_15_minute_freshness_sla(data_quality_dataset: dict[str, Any]) -> None:
    now = datetime.now(timezone.utc)
    latest_ingest = max(
        _parse_dt(row["ingested_at"]) for row in data_quality_dataset["transactions"]
    )
    max_age_minutes = (now - latest_ingest).total_seconds() / 60.0

    assert max_age_minutes < data_quality_dataset["expected_profiles"]["freshness_sla_minutes"]


@pytest.mark.data_quality
def test_model_scores_are_not_staler_than_transactions(
    data_quality_dataset: dict[str, Any],
) -> None:
    score_by_txn = {
        row["transaction_id"]: _parse_dt(row["score_timestamp"])
        for row in data_quality_dataset["model_scores"]
    }

    for txn in data_quality_dataset["transactions"]:
        assert score_by_txn[txn["transaction_id"]] >= _parse_dt(txn["ingested_at"])


@pytest.mark.data_quality
def test_alerts_are_created_after_ingest(data_quality_dataset: dict[str, Any]) -> None:
    txn_ingested = {
        row["transaction_id"]: _parse_dt(row["ingested_at"])
        for row in data_quality_dataset["transactions"]
    }

    for alert in data_quality_dataset["fraud_alerts"]:
        assert _parse_dt(alert["created_at"]) >= txn_ingested[alert["transaction_id"]]


@pytest.mark.data_quality
def test_volume_anomaly_within_three_standard_deviations(
    data_quality_dataset: dict[str, Any],
) -> None:
    per_minute = Counter(
        _parse_dt(row["ingested_at"]).replace(second=0, microsecond=0)
        for row in data_quality_dataset["transactions"]
    )
    expected = data_quality_dataset["expected_profiles"]
    lower = expected["transactions_per_minute_mean"] - 3 * expected["transactions_per_minute_std"]
    upper = expected["transactions_per_minute_mean"] + 3 * expected["transactions_per_minute_std"]

    assert per_minute
    assert all(lower <= count <= upper for count in per_minute.values())


@pytest.mark.data_quality
def test_fixture_version_metadata_is_present(data_quality_dataset: dict[str, Any]) -> None:
    metadata = data_quality_dataset["metadata"]

    assert metadata["fixture_version"] == "2026.08.day39"
    assert len(metadata["record_hash"]) == 64
    assert _parse_dt(metadata["generated_at"]).tzinfo is not None
