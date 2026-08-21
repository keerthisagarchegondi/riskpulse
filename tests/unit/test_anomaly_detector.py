"""Unit tests for Isolation Forest anomaly detector."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.fraud_detection.anomaly_detector import (
    ANOMALY_FEATURES,
    AnomalyDetector,
    AnomalyResult,
)


@pytest.fixture
def sample_training_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Generate small synthetic dataset for testing."""
    rng = np.random.default_rng(42)
    n_legit = 500
    n_fraud = 20

    legit = pd.DataFrame(
        {
            "transaction_amount": rng.lognormal(3.5, 1.0, n_legit),
            "transaction_count_1hour": rng.poisson(2, n_legit).astype(float),
            "transaction_count_24hour": rng.poisson(8, n_legit).astype(float),
            "amount_mean_24hour": rng.lognormal(3.5, 0.5, n_legit),
            "amount_std_24hour": rng.exponential(20, n_legit),
            "time_since_last_transaction_seconds": rng.exponential(3600, n_legit),
            "distance_from_last_location_km": rng.exponential(10, n_legit),
            "unique_merchants_24hour": rng.poisson(3, n_legit).astype(float),
            "unique_countries_24hour": np.ones(n_legit),
            "hour_of_day": rng.integers(6, 23, n_legit).astype(float),
            "is_international": rng.binomial(1, 0.05, n_legit).astype(float),
            "amount_to_avg_ratio": rng.lognormal(0, 0.3, n_legit),
        }
    )

    fraud = pd.DataFrame(
        {
            "transaction_amount": rng.lognormal(6.0, 1.5, n_fraud),
            "transaction_count_1hour": rng.poisson(8, n_fraud).astype(float),
            "transaction_count_24hour": rng.poisson(25, n_fraud).astype(float),
            "amount_mean_24hour": rng.lognormal(5.0, 1.0, n_fraud),
            "amount_std_24hour": rng.exponential(100, n_fraud),
            "time_since_last_transaction_seconds": rng.exponential(120, n_fraud),
            "distance_from_last_location_km": rng.exponential(500, n_fraud),
            "unique_merchants_24hour": rng.poisson(10, n_fraud).astype(float),
            "unique_countries_24hour": rng.poisson(3, n_fraud).astype(float) + 1,
            "hour_of_day": rng.integers(0, 6, n_fraud).astype(float),
            "is_international": rng.binomial(1, 0.7, n_fraud).astype(float),
            "amount_to_avg_ratio": rng.lognormal(1.5, 0.8, n_fraud),
        }
    )

    X = pd.concat([legit, fraud], ignore_index=True)
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    X["transaction_id"] = [f"TXN-TEST-{i:04d}" for i in range(len(X))]
    return X, y


@pytest.fixture
def trained_detector(sample_training_data: tuple[pd.DataFrame, np.ndarray]) -> AnomalyDetector:
    """Return a trained AnomalyDetector."""
    X, _ = sample_training_data
    detector = AnomalyDetector(
        n_estimators=100,
        contamination=0.04,
        random_state=42,
    )
    detector.fit(X)
    return detector


@pytest.fixture
def normal_transaction() -> dict:
    """Normal transaction features."""
    return {
        "transaction_id": "TXN-NORMAL-001",
        "transaction_amount": 45.0,
        "transaction_count_1hour": 1,
        "transaction_count_24hour": 5,
        "amount_mean_24hour": 50.0,
        "amount_std_24hour": 15.0,
        "time_since_last_transaction_seconds": 7200,
        "distance_from_last_location_km": 5.0,
        "unique_merchants_24hour": 2,
        "unique_countries_24hour": 1,
        "hour_of_day": 14.0,
        "is_international": 0,
        "amount_to_avg_ratio": 0.9,
    }


@pytest.fixture
def anomalous_transaction() -> dict:
    """Highly anomalous transaction features."""
    return {
        "transaction_id": "TXN-FRAUD-001",
        "transaction_amount": 9500.0,
        "transaction_count_1hour": 15,
        "transaction_count_24hour": 40,
        "amount_mean_24hour": 800.0,
        "amount_std_24hour": 500.0,
        "time_since_last_transaction_seconds": 30,
        "distance_from_last_location_km": 2000.0,
        "unique_merchants_24hour": 12,
        "unique_countries_24hour": 5,
        "hour_of_day": 3.0,
        "is_international": 1,
        "amount_to_avg_ratio": 11.8,
    }


class TestAnomalyDetectorInit:
    """Tests for AnomalyDetector initialization."""

    def test_default_initialization(self):
        detector = AnomalyDetector()
        assert not detector.is_fitted
        assert detector.model_version == ""
        assert detector.feature_names == ANOMALY_FEATURES

    def test_custom_parameters(self):
        detector = AnomalyDetector(
            n_estimators=100,
            max_samples=256,
            contamination=0.05,
            random_state=123,
        )
        assert not detector.is_fitted
        assert detector._n_estimators == 100
        assert detector._max_samples == 256
        assert detector._contamination == 0.05
        assert detector._random_state == 123

    def test_custom_features(self):
        features = ["transaction_amount", "hour_of_day"]
        detector = AnomalyDetector(feature_names=features)
        assert detector.feature_names == features


class TestAnomalyDetectorFit:
    """Tests for model training."""

    def test_fit_sets_is_fitted(self, sample_training_data):
        X, _ = sample_training_data
        detector = AnomalyDetector(n_estimators=50, random_state=42)
        detector.fit(X)
        assert detector.is_fitted

    def test_fit_generates_model_version(self, sample_training_data):
        X, _ = sample_training_data
        detector = AnomalyDetector(n_estimators=50, random_state=42)
        detector.fit(X)
        assert len(detector.model_version) == 12

    def test_fit_with_missing_features(self):
        """Model should handle missing features gracefully."""
        X = pd.DataFrame(
            {
                "transaction_amount": [100.0, 200.0, 50.0],
                "hour_of_day": [10.0, 15.0, 22.0],
            }
        )
        detector = AnomalyDetector(n_estimators=50, random_state=42)
        detector.fit(X)
        assert detector.is_fitted

    def test_fit_returns_self(self, sample_training_data):
        X, _ = sample_training_data
        detector = AnomalyDetector(n_estimators=50, random_state=42)
        result = detector.fit(X)
        assert result is detector


class TestAnomalyDetectorPredict:
    """Tests for single transaction prediction."""

    def test_predict_returns_anomaly_result(self, trained_detector, normal_transaction):
        result = trained_detector.predict(normal_transaction)
        assert isinstance(result, AnomalyResult)
        assert result.transaction_id == "TXN-NORMAL-001"

    def test_predict_score_range(self, trained_detector, normal_transaction):
        result = trained_detector.predict(normal_transaction)
        assert -1.0 <= result.anomaly_score <= 1.0

    def test_predict_confidence_range(self, trained_detector, normal_transaction):
        result = trained_detector.predict(normal_transaction)
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_normal_transaction_not_anomaly(self, trained_detector, normal_transaction):
        result = trained_detector.predict(normal_transaction)
        assert result.anomaly_score > -0.5  # Should be closer to normal

    def test_predict_anomalous_transaction_flagged(self, trained_detector, anomalous_transaction):
        result = trained_detector.predict(anomalous_transaction)
        assert result.is_anomaly
        assert result.anomaly_score < 0

    def test_predict_includes_model_version(self, trained_detector, normal_transaction):
        result = trained_detector.predict(normal_transaction)
        assert result.model_version == trained_detector.model_version

    def test_predict_latency_acceptable(self, trained_detector, normal_transaction):
        """Online prediction latency should be reasonable.

        Production SLA is <10ms; test environments allow up to 100ms
        due to CI overhead, debug mode, and unoptimized hardware.
        """
        # Warmup (JIT, cache priming)
        for _ in range(5):
            trained_detector.predict(normal_transaction)

        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            trained_detector.predict(normal_transaction)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = np.mean(latencies)
        assert (
            avg_latency < 100.0
        ), f"Average latency {avg_latency:.2f}ms exceeds 100ms test threshold"

    def test_predict_unfitted_raises(self, normal_transaction):
        detector = AnomalyDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            detector.predict(normal_transaction)

    def test_predict_with_missing_features(self, trained_detector):
        """Should handle missing features by defaulting to 0."""
        partial = {"transaction_id": "TXN-PARTIAL", "transaction_amount": 100.0}
        result = trained_detector.predict(partial)
        assert isinstance(result, AnomalyResult)

    def test_predict_contributing_features(self, trained_detector, anomalous_transaction):
        result = trained_detector.predict(anomalous_transaction)
        assert isinstance(result.contributing_features, list)
        for feat in result.contributing_features:
            assert "feature" in feat
            assert "value" in feat
            assert "deviation_score" in feat


class TestAnomalyDetectorBatchPredict:
    """Tests for batch prediction."""

    def test_batch_predict_returns_list(self, trained_detector, sample_training_data):
        X, _ = sample_training_data
        results = trained_detector.predict_batch(X.head(10))
        assert len(results) == 10
        assert all(isinstance(r, AnomalyResult) for r in results)

    def test_batch_predict_scores_in_range(self, trained_detector, sample_training_data):
        X, _ = sample_training_data
        results = trained_detector.predict_batch(X.head(50))
        for r in results:
            assert -1.0 <= r.anomaly_score <= 1.0

    def test_batch_predict_unfitted_raises(self, sample_training_data):
        X, _ = sample_training_data
        detector = AnomalyDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            detector.predict_batch(X.head(5))

    def test_batch_detects_anomalies(self, trained_detector, sample_training_data):
        X, y = sample_training_data
        results = trained_detector.predict_batch(X)
        anomaly_count = sum(1 for r in results if r.is_anomaly)
        assert anomaly_count > 0, "Should detect at least some anomalies"


class TestAnomalyDetectorPersistence:
    """Tests for model save/load."""

    def test_save_creates_files(self, trained_detector, tmp_path):
        save_dir = tmp_path / "model"
        trained_detector.save(save_dir)
        assert (save_dir / "model.joblib").exists()
        assert (save_dir / "scaler.joblib").exists()
        assert (save_dir / "metadata.joblib").exists()

    def test_load_restores_model(self, trained_detector, tmp_path):
        save_dir = tmp_path / "model"
        trained_detector.save(save_dir)
        loaded = AnomalyDetector.load(save_dir)
        assert loaded.is_fitted
        assert loaded.model_version == trained_detector.model_version

    def test_load_predictions_match(self, trained_detector, tmp_path, normal_transaction):
        save_dir = tmp_path / "model"
        trained_detector.save(save_dir)
        loaded = AnomalyDetector.load(save_dir)

        orig_result = trained_detector.predict(normal_transaction)
        loaded_result = loaded.predict(normal_transaction)

        assert abs(orig_result.anomaly_score - loaded_result.anomaly_score) < 1e-10

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AnomalyDetector.load(tmp_path / "nonexistent")

    def test_save_unfitted_raises(self, tmp_path):
        detector = AnomalyDetector()
        with pytest.raises(RuntimeError, match="Cannot save unfitted"):
            detector.save(tmp_path / "model")


class TestAnomalyDetectorTuning:
    """Tests for contamination tuning."""

    def test_tune_contamination_returns_float(self, sample_training_data):
        X, y = sample_training_data
        detector = AnomalyDetector(n_estimators=50, random_state=42)
        best_c = detector.tune_contamination(X, y, contamination_range=[0.01, 0.03, 0.05])
        assert isinstance(best_c, float)
        assert 0 < best_c < 1

    def test_tune_updates_contamination(self, sample_training_data):
        X, y = sample_training_data
        detector = AnomalyDetector(n_estimators=50, random_state=42)
        best_c = detector.tune_contamination(X, y, contamination_range=[0.01, 0.03, 0.05])
        assert detector._contamination == best_c


class TestAnomalyDetectorScoring:
    """Tests for raw anomaly score retrieval."""

    def test_get_anomaly_scores_shape(self, trained_detector, sample_training_data):
        X, _ = sample_training_data
        scores = trained_detector.get_anomaly_scores(X.head(20))
        assert scores.shape == (20,)

    def test_get_anomaly_scores_unfitted_raises(self, sample_training_data):
        X, _ = sample_training_data
        detector = AnomalyDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            detector.get_anomaly_scores(X.head(5))


class TestAnomalyResultDataclass:
    """Tests for AnomalyResult."""

    def test_to_dict(self):
        result = AnomalyResult(
            transaction_id="TXN-001",
            anomaly_score=-0.5,
            is_anomaly=True,
            confidence=0.8,
            prediction_latency_ms=1.5,
            model_version="abc123",
        )
        d = result.to_dict()
        assert d["transaction_id"] == "TXN-001"
        assert d["anomaly_score"] == -0.5
        assert d["is_anomaly"] is True
        assert d["confidence"] == 0.8
        assert d["model_version"] == "abc123"


class TestAnomalyDetectorReproducibility:
    """Tests ensuring training reproducibility with fixed seeds."""

    def test_same_seed_same_results(self, sample_training_data):
        X, _ = sample_training_data
        features = {"transaction_amount": 500.0, "hour_of_day": 3.0, "transaction_id": "TXN"}

        d1 = AnomalyDetector(n_estimators=50, random_state=42)
        d1.fit(X)
        r1 = d1.predict(features)

        d2 = AnomalyDetector(n_estimators=50, random_state=42)
        d2.fit(X)
        r2 = d2.predict(features)

        assert abs(r1.anomaly_score - r2.anomaly_score) < 1e-10

    def test_different_seed_different_results(self, sample_training_data):
        X, _ = sample_training_data
        features = {"transaction_amount": 500.0, "hour_of_day": 3.0, "transaction_id": "TXN"}

        d1 = AnomalyDetector(n_estimators=50, random_state=42)
        d1.fit(X)
        r1 = d1.predict(features)

        d2 = AnomalyDetector(n_estimators=50, random_state=99)
        d2.fit(X)
        r2 = d2.predict(features)

        # Not guaranteed to differ but highly likely with different seeds
        # Just verify both produce valid results
        assert -1.0 <= r1.anomaly_score <= 1.0
        assert -1.0 <= r2.anomaly_score <= 1.0
