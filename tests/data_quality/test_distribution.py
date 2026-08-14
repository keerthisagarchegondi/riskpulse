"""Distribution, skew, and aggregate-quality checks."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

import pytest


@pytest.mark.data_quality
def test_channel_distribution_has_no_unexpected_skew(data_quality_dataset: dict[str, Any]) -> None:
    channels = Counter(row["channel"] for row in data_quality_dataset["transactions"])
    total = sum(channels.values())

    assert len(channels) >= 3
    assert max(channels.values()) / total < 0.7


@pytest.mark.data_quality
def test_country_distribution_has_expected_coverage(data_quality_dataset: dict[str, Any]) -> None:
    countries = Counter(row["geo_country"] for row in data_quality_dataset["transactions"])

    assert "US" in countries
    assert len(countries) >= 4
    assert countries["US"] / sum(countries.values()) < 0.9


@pytest.mark.data_quality
def test_amount_distribution_is_not_pathologically_skewed(
    data_quality_dataset: dict[str, Any],
) -> None:
    amounts = sorted(
        float(row["transaction_amount"]) for row in data_quality_dataset["transactions"]
    )
    p50 = median(amounts)
    p95 = amounts[int(len(amounts) * 0.95)]

    assert p50 > 0.0
    assert p95 / p50 < 80.0


@pytest.mark.data_quality
def test_fraud_rate_matches_expected_profile(data_quality_dataset: dict[str, Any]) -> None:
    rows = data_quality_dataset["transactions"]
    observed = sum(1 for row in rows if row["is_fraud"]) / len(rows)
    expected = data_quality_dataset["expected_profiles"]["fraud_rate"]

    assert abs(observed - expected) <= 0.01


@pytest.mark.data_quality
def test_model_score_distribution_separates_fraud_from_legit(
    data_quality_dataset: dict[str, Any],
) -> None:
    txns = {row["transaction_id"]: row for row in data_quality_dataset["transactions"]}
    fraud_scores = []
    legit_scores = []
    for score in data_quality_dataset["model_scores"]:
        target = fraud_scores if txns[score["transaction_id"]]["is_fraud"] else legit_scores
        target.append(float(score["risk_score"]))

    assert sum(fraud_scores) / len(fraud_scores) > 0.75
    assert sum(legit_scores) / len(legit_scores) < 0.35


@pytest.mark.data_quality
def test_no_nulls_in_required_columns(data_quality_dataset: dict[str, Any]) -> None:
    required_by_table = {
        "customers": ("customer_id", "risk_tier"),
        "accounts": ("account_id", "customer_id"),
        "transactions": ("transaction_id", "customer_id", "account_id", "transaction_amount"),
        "fraud_alerts": ("alert_id", "transaction_id", "severity"),
        "model_scores": ("score_id", "transaction_id", "risk_score"),
    }

    for table_name, fields in required_by_table.items():
        for row in data_quality_dataset[table_name]:
            for field in fields:
                assert row[field] not in (None, ""), f"{table_name}.{field} is null-like"
