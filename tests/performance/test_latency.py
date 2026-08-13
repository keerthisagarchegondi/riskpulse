"""Latency benchmarks for fraud scoring and API validation paths."""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Any

import pytest

from src.fraud_detection.anomaly_detector import AnomalyResult
from src.fraud_detection.risk_scorer import RiskScore
from src.fraud_detection.rule_engine import RuleEvaluationResult
from src.fraud_detection.scoring_pipeline import ScoringPipeline
from src.utils.security import sanitize_string


P99_SCORING_BUDGET_MS = 500.0
P99_VALIDATION_BUDGET_MS = 20.0


@dataclass
class StaticRuleEngine:
    rule_score: float = 0.72

    def evaluate(self, transaction: dict[str, Any], context: dict[str, Any] | None = None):
        return RuleEvaluationResult(
            transaction_id=transaction["external_transaction_id"],
            triggered_rules=[],
            combined_severity="medium",
            combined_confidence=0.7,
            rule_score=self.rule_score,
            evaluation_time_ms=0.2,
            total_rules_evaluated=12,
        )


@dataclass
class StaticAnomalyDetector:
    anomaly_score: float = -0.45

    def predict(self, transaction: dict[str, Any]) -> AnomalyResult:
        return AnomalyResult(
            transaction_id=transaction["external_transaction_id"],
            anomaly_score=self.anomaly_score,
            is_anomaly=True,
            confidence=0.82,
            model_version="latency-test",
        )


@dataclass
class StaticRiskScorer:
    risk_score: float = 0.78

    def predict(self, transaction: dict[str, Any]) -> RiskScore:
        return RiskScore(
            transaction_id=transaction["external_transaction_id"],
            risk_score=self.risk_score,
            risk_level="high",
            raw_score=self.risk_score,
            confidence=0.91,
            top_features=[{"feature": "transaction_amount", "importance": 0.4}],
            model_version="latency-test",
        )


def _transaction(index: int) -> dict[str, Any]:
    return {
        "external_transaction_id": f"TXN-LATENCY-{index:05d}",
        "account_id": f"ACC-{index % 100:03d}",
        "customer_id": f"CUST-{index % 100:03d}",
        "transaction_amount": 150.0 + (index % 50),
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "merchant_category_code": "5411",
        "transaction_timestamp": "2026-08-13T12:00:00Z",
    }


@pytest.fixture
def scoring_pipeline() -> ScoringPipeline:
    pipeline = ScoringPipeline(
        rule_engine=StaticRuleEngine(),
        anomaly_detector=StaticAnomalyDetector(),
        risk_scorer=StaticRiskScorer(),
        weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4},
    )
    pipeline._cache = None
    pipeline._timeout_ms = int(P99_SCORING_BUDGET_MS)
    return pipeline


@pytest.mark.performance
def test_scoring_p99_latency_under_500ms(scoring_pipeline: ScoringPipeline) -> None:
    latencies_ms: list[float] = []

    for index in range(250):
        started = time.perf_counter()
        result = scoring_pipeline.score_transaction_sync(_transaction(index), use_cache=False)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        assert result.methods_succeeded == 3

    p99_latency = statistics.quantiles(latencies_ms, n=100)[98]
    assert p99_latency < P99_SCORING_BUDGET_MS


@pytest.mark.performance
def test_async_batch_scoring_preserves_latency_budget(
    scoring_pipeline: ScoringPipeline,
) -> None:
    batch = [_transaction(index) for index in range(300)]

    started = time.perf_counter()
    results = asyncio.run(scoring_pipeline.score_batch(batch, use_cache=False))
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert len(results) == len(batch)
    assert max(result.total_latency_ms for result in results) < P99_SCORING_BUDGET_MS
    assert elapsed_ms / len(batch) < P99_SCORING_BUDGET_MS


@pytest.mark.performance
def test_validation_hot_path_p99_latency_under_20ms() -> None:
    latencies_ms: list[float] = []
    payloads = [
        f"TXN-{index:05d}-ACC-{index % 100:03d}-merchant-note"
        for index in range(500)
    ]

    for payload in payloads:
        started = time.perf_counter()
        assert sanitize_string(payload) == payload
        latencies_ms.append((time.perf_counter() - started) * 1000)

    assert statistics.quantiles(latencies_ms, n=100)[98] < P99_VALIDATION_BUDGET_MS
