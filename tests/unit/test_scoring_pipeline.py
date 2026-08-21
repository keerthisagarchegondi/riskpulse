"""Comprehensive tests for the Unified Fraud Scoring Pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.fraud_detection.anomaly_detector import AnomalyResult
from src.fraud_detection.risk_scorer import RiskScore
from src.fraud_detection.rule_engine import RuleEvaluationResult, RuleMatch
from src.fraud_detection.scoring_pipeline import (
    RiskClassification,
    ScoringMethodResult,
    ScoringPipeline,
    UnifiedScore,
    _LRUCache,
)

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_rule_engine():
    """Mock rule engine that returns configurable results."""
    engine = MagicMock()
    engine.evaluate.return_value = RuleEvaluationResult(
        transaction_id="TXN-001",
        triggered_rules=[
            RuleMatch(
                rule_id="R001",
                rule_name="High Amount",
                category="amount",
                severity="high",
                confidence=0.8,
                details="Amount exceeds threshold",
            )
        ],
        combined_severity="high",
        combined_confidence=0.8,
        rule_score=0.7,
        evaluation_time_ms=5.0,
        total_rules_evaluated=10,
    )
    return engine


@pytest.fixture
def mock_anomaly_detector():
    """Mock anomaly detector that returns configurable results."""
    detector = MagicMock()
    detector.predict.return_value = AnomalyResult(
        transaction_id="TXN-001",
        anomaly_score=-0.5,  # anomalous
        is_anomaly=True,
        confidence=0.85,
        contributing_features=[],
        prediction_latency_ms=8.0,
        model_version="v1.0",
    )
    return detector


@pytest.fixture
def mock_risk_scorer():
    """Mock ML risk scorer that returns configurable results."""
    scorer = MagicMock()
    scorer.predict.return_value = RiskScore(
        transaction_id="TXN-001",
        risk_score=0.75,
        risk_level="high",
        raw_score=0.82,
        confidence=0.9,
        top_features=[{"feature": "amount", "importance": 0.4}],
        prediction_latency_ms=12.0,
        model_version="v2.1",
    )
    return scorer


@pytest.fixture
def sample_transaction():
    """Sample transaction for testing."""
    return {
        "external_transaction_id": "TXN-TEST-001",
        "customer_id": "CUST-001",
        "transaction_amount": 5000.0,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "merchant_category_code": "5411",
        "ip_address": "192.168.1.1",
        "device_id": "device-123",
        "geo_country": "US",
        "is_international": False,
        "transaction_timestamp": "2026-07-15T10:30:00Z",
    }


@pytest.fixture
def scoring_pipeline(mock_rule_engine, mock_anomaly_detector, mock_risk_scorer):
    """Fully configured scoring pipeline with mocked methods."""
    return ScoringPipeline(
        rule_engine=mock_rule_engine,
        anomaly_detector=mock_anomaly_detector,
        risk_scorer=mock_risk_scorer,
        weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4},
    )


@pytest.fixture
def scoring_pipeline_no_cache(mock_rule_engine, mock_anomaly_detector, mock_risk_scorer):
    """Pipeline with cache disabled."""
    pipeline = ScoringPipeline(
        rule_engine=mock_rule_engine,
        anomaly_detector=mock_anomaly_detector,
        risk_scorer=mock_risk_scorer,
        weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4},
    )
    pipeline._cache = None
    return pipeline


# ── Test: Weight Validation ──────────────────────────────────────────


class TestWeightValidation:
    def test_valid_weights(self):
        pipeline = ScoringPipeline(
            weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4}
        )
        assert pipeline.weights == {"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4}

    def test_invalid_weights_not_sum_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ScoringPipeline(weights={"rule_score": 0.5, "anomaly_score": 0.5, "ml_score": 0.5})

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            ScoringPipeline(weights={"rule_score": -0.1, "anomaly_score": 0.5, "ml_score": 0.6})

    def test_update_weights(self, scoring_pipeline):
        scoring_pipeline.update_weights({"rule_score": 0.2, "anomaly_score": 0.3, "ml_score": 0.5})
        assert scoring_pipeline.weights["ml_score"] == 0.5

    def test_update_weights_invalid_sum(self, scoring_pipeline):
        with pytest.raises(ValueError):
            scoring_pipeline.update_weights(
                {"rule_score": 0.5, "anomaly_score": 0.5, "ml_score": 0.5}
            )


# ── Test: Risk Classification ────────────────────────────────────────


class TestRiskClassification:
    def test_low_classification(self, scoring_pipeline):
        assert scoring_pipeline._classify_risk(0.0) == RiskClassification.LOW
        assert scoring_pipeline._classify_risk(0.1) == RiskClassification.LOW
        assert scoring_pipeline._classify_risk(0.29) == RiskClassification.LOW

    def test_medium_classification(self, scoring_pipeline):
        assert scoring_pipeline._classify_risk(0.3) == RiskClassification.MEDIUM
        assert scoring_pipeline._classify_risk(0.5) == RiskClassification.MEDIUM
        assert scoring_pipeline._classify_risk(0.59) == RiskClassification.MEDIUM

    def test_high_classification(self, scoring_pipeline):
        assert scoring_pipeline._classify_risk(0.6) == RiskClassification.HIGH
        assert scoring_pipeline._classify_risk(0.7) == RiskClassification.HIGH
        assert scoring_pipeline._classify_risk(0.84) == RiskClassification.HIGH

    def test_critical_classification(self, scoring_pipeline):
        assert scoring_pipeline._classify_risk(0.85) == RiskClassification.CRITICAL
        assert scoring_pipeline._classify_risk(0.95) == RiskClassification.CRITICAL
        assert scoring_pipeline._classify_risk(1.0) == RiskClassification.CRITICAL


# ── Test: Ensemble Score Computation ─────────────────────────────────


class TestEnsembleScore:
    def test_all_methods_succeed(self, scoring_pipeline):
        results = [
            ScoringMethodResult(
                method="rule_engine",
                raw_score=0.7,
                normalized_score=0.7,
                weight=0.3,
                weighted_score=0.21,
                latency_ms=5.0,
                success=True,
            ),
            ScoringMethodResult(
                method="anomaly_detector",
                raw_score=-0.5,
                normalized_score=0.75,
                weight=0.3,
                weighted_score=0.225,
                latency_ms=8.0,
                success=True,
            ),
            ScoringMethodResult(
                method="ml_model",
                raw_score=0.8,
                normalized_score=0.8,
                weight=0.4,
                weighted_score=0.32,
                latency_ms=12.0,
                success=True,
            ),
        ]
        score = scoring_pipeline._compute_ensemble_score(results)
        # Expected: (0.7*0.3 + 0.75*0.3 + 0.8*0.4) / 1.0 = 0.755
        assert 0.74 < score < 0.76

    def test_one_method_fails_renormalize(self, scoring_pipeline):
        results = [
            ScoringMethodResult(
                method="rule_engine",
                raw_score=0.7,
                normalized_score=0.7,
                weight=0.3,
                weighted_score=0.21,
                latency_ms=5.0,
                success=True,
            ),
            ScoringMethodResult(
                method="anomaly_detector",
                raw_score=0.0,
                normalized_score=0.0,
                weight=0.3,
                weighted_score=0.0,
                latency_ms=0.0,
                success=False,
                error="Not initialized",
            ),
            ScoringMethodResult(
                method="ml_model",
                raw_score=0.8,
                normalized_score=0.8,
                weight=0.4,
                weighted_score=0.32,
                latency_ms=12.0,
                success=True,
            ),
        ]
        score = scoring_pipeline._compute_ensemble_score(results)
        # Re-normalized: rule=0.3/(0.3+0.4)=0.4286, ml=0.4/(0.3+0.4)=0.5714
        # Score: 0.7*0.4286 + 0.8*0.5714 ≈ 0.757
        assert 0.75 < score < 0.77

    def test_all_methods_fail(self, scoring_pipeline):
        results = [
            ScoringMethodResult(
                method="rule_engine",
                raw_score=0.0,
                normalized_score=0.0,
                weight=0.3,
                weighted_score=0.0,
                latency_ms=0.0,
                success=False,
            ),
            ScoringMethodResult(
                method="anomaly_detector",
                raw_score=0.0,
                normalized_score=0.0,
                weight=0.3,
                weighted_score=0.0,
                latency_ms=0.0,
                success=False,
            ),
            ScoringMethodResult(
                method="ml_model",
                raw_score=0.0,
                normalized_score=0.0,
                weight=0.4,
                weighted_score=0.0,
                latency_ms=0.0,
                success=False,
            ),
        ]
        score = scoring_pipeline._compute_ensemble_score(results)
        assert score == 0.0

    def test_score_bounded_zero_to_one(self, scoring_pipeline):
        results = [
            ScoringMethodResult(
                method="rule_engine",
                raw_score=1.0,
                normalized_score=1.0,
                weight=0.3,
                weighted_score=0.3,
                latency_ms=5.0,
                success=True,
            ),
            ScoringMethodResult(
                method="anomaly_detector",
                raw_score=1.0,
                normalized_score=1.0,
                weight=0.3,
                weighted_score=0.3,
                latency_ms=8.0,
                success=True,
            ),
            ScoringMethodResult(
                method="ml_model",
                raw_score=1.0,
                normalized_score=1.0,
                weight=0.4,
                weighted_score=0.4,
                latency_ms=12.0,
                success=True,
            ),
        ]
        score = scoring_pipeline._compute_ensemble_score(results)
        assert 0.0 <= score <= 1.0
        assert score == 1.0


# ── Test: Synchronous Scoring ────────────────────────────────────────


class TestSyncScoring:
    def test_score_transaction_sync(self, scoring_pipeline, sample_transaction):
        result = scoring_pipeline.score_transaction_sync(sample_transaction)

        assert isinstance(result, UnifiedScore)
        assert result.transaction_id == "TXN-TEST-001"
        assert 0.0 <= result.final_score <= 1.0
        assert result.risk_classification in RiskClassification
        assert result.methods_succeeded == 3
        assert result.methods_failed == 0
        assert result.total_latency_ms > 0
        assert len(result.method_scores) == 3

    def test_score_produces_expected_classification(
        self, mock_rule_engine, mock_anomaly_detector, mock_risk_scorer, sample_transaction
    ):
        # All high scores → should classify as high or critical
        mock_rule_engine.evaluate.return_value = RuleEvaluationResult(
            transaction_id="TXN-TEST-001",
            rule_score=0.9,
            combined_severity="critical",
            combined_confidence=0.95,
            evaluation_time_ms=3.0,
            total_rules_evaluated=10,
        )
        mock_anomaly_detector.predict.return_value = AnomalyResult(
            transaction_id="TXN-TEST-001",
            anomaly_score=-0.8,
            is_anomaly=True,
            confidence=0.95,
            prediction_latency_ms=5.0,
            model_version="v1",
        )
        mock_risk_scorer.predict.return_value = RiskScore(
            transaction_id="TXN-TEST-001",
            risk_score=0.92,
            risk_level="critical",
            raw_score=0.95,
            confidence=0.98,
            prediction_latency_ms=8.0,
            model_version="v2",
        )

        pipeline = ScoringPipeline(
            rule_engine=mock_rule_engine,
            anomaly_detector=mock_anomaly_detector,
            risk_scorer=mock_risk_scorer,
            weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4},
        )
        result = pipeline.score_transaction_sync(sample_transaction)
        assert result.risk_classification in (
            RiskClassification.HIGH,
            RiskClassification.CRITICAL,
        )
        assert result.final_score > 0.6

    def test_score_low_risk_transaction(self, sample_transaction):
        rule_engine = MagicMock()
        rule_engine.evaluate.return_value = RuleEvaluationResult(
            transaction_id="TXN-TEST-001",
            rule_score=0.0,
            combined_severity="low",
            combined_confidence=0.0,
            evaluation_time_ms=2.0,
            total_rules_evaluated=10,
        )
        anomaly = MagicMock()
        anomaly.predict.return_value = AnomalyResult(
            transaction_id="TXN-TEST-001",
            anomaly_score=0.8,  # normal
            is_anomaly=False,
            confidence=0.1,
            prediction_latency_ms=4.0,
            model_version="v1",
        )
        scorer = MagicMock()
        scorer.predict.return_value = RiskScore(
            transaction_id="TXN-TEST-001",
            risk_score=0.05,
            risk_level="low",
            raw_score=0.03,
            confidence=0.95,
            prediction_latency_ms=6.0,
            model_version="v2",
        )

        pipeline = ScoringPipeline(
            rule_engine=rule_engine,
            anomaly_detector=anomaly,
            risk_scorer=scorer,
            weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4},
        )
        result = pipeline.score_transaction_sync(sample_transaction)
        assert result.risk_classification == RiskClassification.LOW
        assert result.final_score < 0.3

    def test_partial_failure_uses_available_methods(self, sample_transaction):
        rule_engine = MagicMock()
        rule_engine.evaluate.return_value = RuleEvaluationResult(
            transaction_id="TXN-TEST-001",
            rule_score=0.5,
            combined_severity="medium",
            combined_confidence=0.6,
            evaluation_time_ms=3.0,
            total_rules_evaluated=10,
        )
        # Anomaly detector raises exception
        anomaly = MagicMock()
        anomaly.predict.side_effect = RuntimeError("Model not fitted")
        scorer = MagicMock()
        scorer.predict.return_value = RiskScore(
            transaction_id="TXN-TEST-001",
            risk_score=0.6,
            risk_level="medium",
            raw_score=0.55,
            confidence=0.8,
            prediction_latency_ms=7.0,
            model_version="v2",
        )

        pipeline = ScoringPipeline(
            rule_engine=rule_engine,
            anomaly_detector=anomaly,
            risk_scorer=scorer,
            weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4},
        )
        result = pipeline.score_transaction_sync(sample_transaction)
        assert result.methods_succeeded == 2
        assert result.methods_failed == 1
        # Score should still be computed from the two working methods
        assert result.final_score > 0.0

    def test_batch_sync(self, scoring_pipeline, sample_transaction):
        txns = [sample_transaction.copy() for _ in range(5)]
        for i, t in enumerate(txns):
            t["external_transaction_id"] = f"TXN-BATCH-{i:03d}"

        results = scoring_pipeline.score_batch_sync(txns)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, UnifiedScore)
            assert 0.0 <= r.final_score <= 1.0

    def test_batch_exceeds_max(self, scoring_pipeline, sample_transaction):
        txns = [sample_transaction.copy() for _ in range(1001)]
        with pytest.raises(ValueError, match="exceeds maximum"):
            scoring_pipeline.score_batch_sync(txns)


# ── Test: Async Scoring ──────────────────────────────────────────────


class TestAsyncScoring:
    @pytest.mark.asyncio
    async def test_score_transaction_async(self, scoring_pipeline, sample_transaction):
        result = await scoring_pipeline.score_transaction(sample_transaction)

        assert isinstance(result, UnifiedScore)
        assert result.transaction_id == "TXN-TEST-001"
        assert 0.0 <= result.final_score <= 1.0
        assert result.methods_succeeded == 3

    @pytest.mark.asyncio
    async def test_score_batch_async(self, scoring_pipeline, sample_transaction):
        txns = [sample_transaction.copy() for _ in range(3)]
        for i, t in enumerate(txns):
            t["external_transaction_id"] = f"TXN-ASYNC-{i:03d}"

        results = await scoring_pipeline.score_batch(txns)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_score_batch_empty(self, scoring_pipeline):
        results = await scoring_pipeline.score_batch([])
        assert results == []


# ── Test: Caching ────────────────────────────────────────────────────


class TestCaching:
    def test_cache_hit(self, scoring_pipeline, sample_transaction):
        # First call populates cache
        result1 = scoring_pipeline.score_transaction_sync(sample_transaction)
        assert not result1.cached

        # Second call should be cached
        result2 = scoring_pipeline.score_transaction_sync(sample_transaction)
        assert result2.cached
        assert result2.final_score == result1.final_score

    def test_cache_bypass(self, scoring_pipeline, sample_transaction):
        # First call
        scoring_pipeline.score_transaction_sync(sample_transaction)
        # Second call with cache disabled
        result = scoring_pipeline.score_transaction_sync(sample_transaction, use_cache=False)
        assert not result.cached

    def test_cache_invalidation(self, scoring_pipeline, sample_transaction):
        scoring_pipeline.score_transaction_sync(sample_transaction)
        scoring_pipeline.invalidate_cache()
        # After clearing, should not be cached
        result = scoring_pipeline.score_transaction_sync(sample_transaction)
        assert not result.cached

    def test_lru_cache_ttl_expiry(self):
        cache = _LRUCache(max_entries=10, ttl_seconds=0)
        score = UnifiedScore(
            transaction_id="TXN-001",
            final_score=0.5,
            risk_classification=RiskClassification.MEDIUM,
        )
        cache.put("key1", score)
        # TTL=0 means immediately expired
        import time

        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_lru_cache_eviction(self):
        cache = _LRUCache(max_entries=2, ttl_seconds=300)
        s1 = UnifiedScore(
            transaction_id="T1",
            final_score=0.1,
            risk_classification=RiskClassification.LOW,
        )
        s2 = UnifiedScore(
            transaction_id="T2",
            final_score=0.2,
            risk_classification=RiskClassification.LOW,
        )
        s3 = UnifiedScore(
            transaction_id="T3",
            final_score=0.3,
            risk_classification=RiskClassification.LOW,
        )
        cache.put("k1", s1)
        cache.put("k2", s2)
        cache.put("k3", s3)  # Should evict k1
        assert cache.get("k1") is None
        assert cache.get("k2") is not None
        assert cache.get("k3") is not None

    def test_cache_stats(self, scoring_pipeline, sample_transaction):
        scoring_pipeline.score_transaction_sync(sample_transaction)
        stats = scoring_pipeline.cache_stats
        assert stats is not None
        assert stats["size"] == 1
        assert "hits" in stats
        assert "misses" in stats


# ── Test: Anomaly Score Normalization ────────────────────────────────


class TestAnomalyNormalization:
    def test_anomalous_score_maps_high(self, sample_transaction):
        """Isolation Forest score -1 (most anomalous) → normalized 1.0."""
        anomaly = MagicMock()
        anomaly.predict.return_value = AnomalyResult(
            transaction_id="TXN-001",
            anomaly_score=-1.0,
            is_anomaly=True,
            confidence=0.99,
            prediction_latency_ms=5.0,
            model_version="v1",
        )
        pipeline = ScoringPipeline(
            anomaly_detector=anomaly,
            weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4},
        )
        result = pipeline._score_anomaly(sample_transaction)
        assert result.success
        assert result.normalized_score == 1.0

    def test_normal_score_maps_low(self, sample_transaction):
        """Isolation Forest score +1 (most normal) → normalized 0.0."""
        anomaly = MagicMock()
        anomaly.predict.return_value = AnomalyResult(
            transaction_id="TXN-001",
            anomaly_score=1.0,
            is_anomaly=False,
            confidence=0.1,
            prediction_latency_ms=5.0,
            model_version="v1",
        )
        pipeline = ScoringPipeline(
            anomaly_detector=anomaly,
            weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4},
        )
        result = pipeline._score_anomaly(sample_transaction)
        assert result.success
        assert result.normalized_score == 0.0

    def test_neutral_score_maps_mid(self, sample_transaction):
        """Isolation Forest score 0 → normalized 0.5."""
        anomaly = MagicMock()
        anomaly.predict.return_value = AnomalyResult(
            transaction_id="TXN-001",
            anomaly_score=0.0,
            is_anomaly=False,
            confidence=0.5,
            prediction_latency_ms=5.0,
            model_version="v1",
        )
        pipeline = ScoringPipeline(
            anomaly_detector=anomaly,
            weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4},
        )
        result = pipeline._score_anomaly(sample_transaction)
        assert result.success
        assert result.normalized_score == 0.5


# ── Test: ML Confidence Penalty ──────────────────────────────────────


class TestMLConfidencePenalty:
    def test_low_confidence_reduces_weight(self, sample_transaction):
        """When ML confidence is below threshold, weight is penalized."""
        scorer = MagicMock()
        scorer.predict.return_value = RiskScore(
            transaction_id="TXN-001",
            risk_score=0.8,
            risk_level="high",
            raw_score=0.85,
            confidence=0.3,  # Below default 0.5 threshold
            prediction_latency_ms=7.0,
            model_version="v2",
        )
        pipeline = ScoringPipeline(
            risk_scorer=scorer,
            weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4},
        )
        result = pipeline._score_ml(sample_transaction)
        assert result.success
        # Weight should be penalized: 0.4 * 0.5 = 0.2
        assert result.weight == pytest.approx(0.2)

    def test_high_confidence_no_penalty(self, sample_transaction):
        """When ML confidence is above threshold, full weight is used."""
        scorer = MagicMock()
        scorer.predict.return_value = RiskScore(
            transaction_id="TXN-001",
            risk_score=0.8,
            risk_level="high",
            raw_score=0.85,
            confidence=0.9,
            prediction_latency_ms=7.0,
            model_version="v2",
        )
        pipeline = ScoringPipeline(
            risk_scorer=scorer,
            weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4},
        )
        result = pipeline._score_ml(sample_transaction)
        assert result.success
        assert result.weight == 0.4


# ── Test: Alert/Block Recommendations ───────────────────────────────


class TestAlertRecommendations:
    def test_alert_recommended_above_threshold(self, sample_transaction):
        rule_engine = MagicMock()
        rule_engine.evaluate.return_value = RuleEvaluationResult(
            transaction_id="TXN-001",
            rule_score=0.9,
            combined_severity="critical",
            combined_confidence=0.95,
            evaluation_time_ms=3.0,
            total_rules_evaluated=10,
        )
        anomaly = MagicMock()
        anomaly.predict.return_value = AnomalyResult(
            transaction_id="TXN-001",
            anomaly_score=-0.9,
            is_anomaly=True,
            confidence=0.95,
            prediction_latency_ms=5.0,
            model_version="v1",
        )
        scorer = MagicMock()
        scorer.predict.return_value = RiskScore(
            transaction_id="TXN-001",
            risk_score=0.95,
            risk_level="critical",
            raw_score=0.97,
            confidence=0.98,
            prediction_latency_ms=8.0,
            model_version="v2",
        )
        pipeline = ScoringPipeline(
            rule_engine=rule_engine,
            anomaly_detector=anomaly,
            risk_scorer=scorer,
            weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4},
        )
        result = pipeline.score_transaction_sync(sample_transaction)
        assert result.alert_recommended
        assert result.auto_block_recommended

    def test_no_alert_for_low_risk(self, sample_transaction):
        rule_engine = MagicMock()
        rule_engine.evaluate.return_value = RuleEvaluationResult(
            transaction_id="TXN-001",
            rule_score=0.0,
            combined_severity="low",
            combined_confidence=0.0,
            evaluation_time_ms=2.0,
            total_rules_evaluated=10,
        )
        anomaly = MagicMock()
        anomaly.predict.return_value = AnomalyResult(
            transaction_id="TXN-001",
            anomaly_score=0.9,
            is_anomaly=False,
            confidence=0.1,
            prediction_latency_ms=4.0,
            model_version="v1",
        )
        scorer = MagicMock()
        scorer.predict.return_value = RiskScore(
            transaction_id="TXN-001",
            risk_score=0.02,
            risk_level="low",
            raw_score=0.01,
            confidence=0.95,
            prediction_latency_ms=6.0,
            model_version="v2",
        )
        pipeline = ScoringPipeline(
            rule_engine=rule_engine,
            anomaly_detector=anomaly,
            risk_scorer=scorer,
            weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4},
        )
        result = pipeline.score_transaction_sync(sample_transaction)
        assert not result.alert_recommended
        assert not result.auto_block_recommended


# ── Test: Metrics ────────────────────────────────────────────────────


class TestMetrics:
    def test_metrics_updated_after_scoring(self, scoring_pipeline, sample_transaction):
        scoring_pipeline.score_transaction_sync(sample_transaction)
        metrics = scoring_pipeline.metrics
        assert metrics["total_scored"] == 1
        assert metrics["avg_latency_ms"] > 0

    def test_classification_distribution(self, scoring_pipeline, sample_transaction):
        for i in range(3):
            txn = sample_transaction.copy()
            txn["external_transaction_id"] = f"TXN-METRIC-{i}"
            txn["transaction_timestamp"] = f"2026-07-15T10:3{i}:00Z"
            scoring_pipeline.score_transaction_sync(txn, use_cache=False)
        metrics = scoring_pipeline.metrics
        total = sum(metrics["classification_distribution"].values())
        assert total == 3


# ── Test: to_dict Serialization ──────────────────────────────────────


class TestSerialization:
    def test_unified_score_to_dict(self, scoring_pipeline, sample_transaction):
        result = scoring_pipeline.score_transaction_sync(sample_transaction)
        d = result.to_dict()

        assert "transaction_id" in d
        assert "final_score" in d
        assert "risk_classification" in d
        assert "method_scores" in d
        assert "total_latency_ms" in d
        assert isinstance(d["method_scores"], list)
        assert len(d["method_scores"]) == 3

    def test_score_values_in_range(self, scoring_pipeline, sample_transaction):
        result = scoring_pipeline.score_transaction_sync(sample_transaction)
        d = result.to_dict()
        assert 0.0 <= d["final_score"] <= 1.0
        for ms in d["method_scores"]:
            assert 0.0 <= ms["normalized_score"] <= 1.0


# ── Test: No Methods Initialized ─────────────────────────────────────


class TestNoMethods:
    def test_all_methods_uninitialized(self, sample_transaction):
        pipeline = ScoringPipeline(
            weights={"rule_score": 0.3, "anomaly_score": 0.3, "ml_score": 0.4}
        )
        result = pipeline.score_transaction_sync(sample_transaction)
        assert result.methods_succeeded == 0
        assert result.methods_failed == 3
        assert result.final_score == 0.0
        assert result.risk_classification == RiskClassification.LOW
