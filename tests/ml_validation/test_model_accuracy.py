"""Holdout accuracy, calibration, latency, and model-comparison checks."""

from __future__ import annotations

import math
import time
from statistics import quantiles
from typing import Any

import pytest


def _labels_and_scores(dataset: dict[str, Any], score_field: str) -> tuple[list[int], list[float]]:
    labels = [int(row["label"]) for row in dataset["records"]]
    scores = [float(row[score_field]) for row in dataset["records"]]
    return labels, scores


def _predictions(scores: list[float], threshold: float = 0.5) -> list[int]:
    return [1 if score >= threshold else 0 for score in scores]


def _accuracy(labels: list[int], predictions: list[int]) -> float:
    return sum(int(label == pred) for label, pred in zip(labels, predictions)) / len(labels)


def _precision_recall(labels: list[int], predictions: list[int]) -> tuple[float, float]:
    tp = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
    fp = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall


def _brier_score(labels: list[int], scores: list[float]) -> float:
    return sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)


def _score_record(features: dict[str, float | int]) -> float:
    logit = (
        -3.0
        + min(float(features["transaction_amount"]), 6000.0) / 1600.0
        + 0.9 * int(features["is_international"])
        + 0.18 * int(features["velocity_1h"])
        + 0.75 * int(features["new_device"])
        + 1.3 * float(features["merchant_risk"])
        + 0.28 * int(features["prior_declines_24h"])
        + (0.35 if int(features["hour_of_day"]) < 6 else 0.0)
        - min(int(features["customer_tenure_days"]), 2500) / 7000.0
    )
    return 1.0 / (1.0 + math.exp(-logit))


@pytest.mark.ml_validation
def test_production_model_accuracy_on_holdout_set(ml_validation_dataset: dict[str, Any]) -> None:
    labels, scores = _labels_and_scores(ml_validation_dataset, "production_score")
    preds = _predictions(scores)
    precision, recall = _precision_recall(labels, preds)

    assert _accuracy(labels, preds) >= 0.92
    assert precision >= 0.88
    assert recall >= 0.88


@pytest.mark.ml_validation
def test_score_calibration_brier_score(ml_validation_dataset: dict[str, Any]) -> None:
    labels, scores = _labels_and_scores(ml_validation_dataset, "production_score")

    assert _brier_score(labels, scores) <= 0.12


@pytest.mark.ml_validation
def test_prediction_latency_p99_under_budget(ml_validation_dataset: dict[str, Any]) -> None:
    latencies_ms: list[float] = []

    for row in ml_validation_dataset["records"][:400]:
        started = time.perf_counter()
        score = _score_record(row["features"])
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        assert 0.0 <= score <= 1.0

    assert quantiles(latencies_ms, n=100)[98] < 5.0


@pytest.mark.ml_validation
def test_challenger_model_is_not_worse_than_current_production(
    ml_validation_dataset: dict[str, Any],
) -> None:
    labels, production_scores = _labels_and_scores(ml_validation_dataset, "production_score")
    _, challenger_scores = _labels_and_scores(ml_validation_dataset, "challenger_score")
    production_accuracy = _accuracy(labels, _predictions(production_scores))
    challenger_accuracy = _accuracy(labels, _predictions(challenger_scores))
    production_brier = _brier_score(labels, production_scores)
    challenger_brier = _brier_score(labels, challenger_scores)

    assert challenger_accuracy >= production_accuracy - 0.02
    assert challenger_brier <= production_brier + 0.01


@pytest.mark.ml_validation
def test_holdout_fixture_is_versioned(ml_validation_dataset: dict[str, Any]) -> None:
    assert ml_validation_dataset["metadata"]["fixture_version"] == "2026.08.day39.ml"
    assert len(ml_validation_dataset["metadata"]["record_hash"]) == 64
