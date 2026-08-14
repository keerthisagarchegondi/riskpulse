"""Segment fairness and feature-stability model validation checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytest


def _segment_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["segment"]].append(row)

    metrics: dict[str, dict[str, float]] = {}
    for segment, segment_rows in grouped.items():
        labels = [int(row["label"]) for row in segment_rows]
        preds = [1 if float(row["production_score"]) >= 0.5 else 0 for row in segment_rows]
        tp = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 1)
        tn = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 0)
        fp = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 1)
        fn = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 0)
        metrics[segment] = {
            "accuracy": (tp + tn) / len(segment_rows),
            "true_positive_rate": tp / max(tp + fn, 1),
            "false_positive_rate": fp / max(fp + tn, 1),
            "positive_rate": sum(preds) / len(preds),
        }
    return metrics


def _ranked_features(importance: dict[str, float]) -> list[str]:
    return [name for name, _ in sorted(importance.items(), key=lambda item: item[1], reverse=True)]


@pytest.mark.ml_validation
def test_model_performance_across_customer_segments(
    ml_validation_dataset: dict[str, Any],
) -> None:
    metrics = _segment_metrics(ml_validation_dataset["records"])

    assert set(metrics) == {"standard", "new_to_credit", "cross_border", "high_velocity"}
    assert all(segment["accuracy"] >= 0.86 for segment in metrics.values())


@pytest.mark.ml_validation
def test_no_material_fairness_gap_across_segments(ml_validation_dataset: dict[str, Any]) -> None:
    metrics = _segment_metrics(ml_validation_dataset["records"])
    tpr_values = [segment["true_positive_rate"] for segment in metrics.values()]
    fpr_values = [segment["false_positive_rate"] for segment in metrics.values()]

    assert max(tpr_values) - min(tpr_values) <= 0.18
    assert max(fpr_values) - min(fpr_values) <= 0.12


@pytest.mark.ml_validation
def test_segment_positive_rates_are_monitored(ml_validation_dataset: dict[str, Any]) -> None:
    metrics = _segment_metrics(ml_validation_dataset["records"])
    rates = {segment: values["positive_rate"] for segment, values in metrics.items()}

    assert rates["high_velocity"] >= rates["standard"]
    assert rates["cross_border"] >= rates["standard"]
    assert all(0.0 <= rate <= 1.0 for rate in rates.values())


@pytest.mark.ml_validation
def test_feature_importance_stability_between_model_versions(
    ml_validation_dataset: dict[str, Any],
) -> None:
    production = ml_validation_dataset["models"]["production"]["feature_importance"]
    challenger = ml_validation_dataset["models"]["challenger"]["feature_importance"]
    production_top = set(_ranked_features(production)[:5])
    challenger_top = set(_ranked_features(challenger)[:5])

    overlap = len(production_top & challenger_top) / len(production_top)
    total_delta = sum(abs(production[name] - challenger[name]) for name in production)

    assert overlap >= 0.8
    assert total_delta <= 0.12


@pytest.mark.ml_validation
def test_feature_vector_schema_is_stable(ml_validation_dataset: dict[str, Any]) -> None:
    expected_features = ml_validation_dataset["feature_names"]

    for row in ml_validation_dataset["records"][:25]:
        assert list(row["features"].keys()) == expected_features
