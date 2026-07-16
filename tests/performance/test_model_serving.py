"""Load tests for model serving — targeting 1000 predictions/second.

Tests:
- Single prediction latency (P50, P95, P99)
- Batch prediction throughput
- Concurrent request handling
- Hot-reload under load (no dropped requests)
- A/B test routing consistency
- Fallback on model failure
"""

from __future__ import annotations

import hashlib
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.fraud_detection.model_registry import (
    ABTestConfig,
    ModelMetadata,
    ModelRegistry,
    ModelServer,
    ModelStage,
    ModelVersion,
)
from src.fraud_detection.model_monitor import (
    DataQualityChecker,
    FeatureDriftDetector,
    ModelMonitor,
    PredictionDistributionTracker,
)


# --- Fixtures ---


@pytest.fixture
def feature_names() -> list[str]:
    return [
        "transaction_amount",
        "transaction_count_1hour",
        "transaction_count_24hour",
        "amount_mean_24hour",
        "amount_std_24hour",
        "time_since_last_transaction_seconds",
        "distance_from_last_location_km",
        "unique_merchants_24hour",
        "unique_countries_24hour",
        "hour_of_day",
        "is_international",
        "amount_to_avg_ratio",
    ]


@pytest.fixture
def mock_model():
    """Create a mock model with predict_proba."""
    model = MagicMock()

    def _predict_proba(X):
        n = X.shape[0] if X.ndim > 1 else 1
        scores = np.random.uniform(0, 1, size=(n, 2))
        scores[:, 0] = 1 - scores[:, 1]
        return scores

    model.predict_proba = MagicMock(side_effect=_predict_proba)
    return model


@pytest.fixture
def registry_with_model(tmp_path, mock_model):
    """Set up a registry with a registered and promoted production model."""
    registry_path = tmp_path / "registry"
    model_path = tmp_path / "models" / "v1"
    model_path.mkdir(parents=True)

    # Create a placeholder file (mock models can't be serialized with joblib)
    (model_path / "model.joblib").write_bytes(b"placeholder")

    registry = ModelRegistry(registry_path)
    registry.register_model(
        name="risk_scorer",
        version="1.0.0",
        artifact_path=str(model_path),
        model_type="xgboost",
        metrics={"auc": 0.95, "precision": 0.92},
        feature_names=[
            "transaction_amount",
            "transaction_count_1hour",
            "transaction_count_24hour",
            "amount_mean_24hour",
            "amount_std_24hour",
            "time_since_last_transaction_seconds",
            "distance_from_last_location_km",
            "unique_merchants_24hour",
            "unique_countries_24hour",
            "hour_of_day",
            "is_international",
            "amount_to_avg_ratio",
        ],
    )
    registry.promote_model("risk_scorer", "1.0.0", ModelStage.PRODUCTION)

    return registry, model_path


@pytest.fixture
def model_server(registry_with_model, mock_model):
    """Create a model server with a loaded production model."""
    registry, model_path = registry_with_model

    def loader(path):
        return mock_model

    server = ModelServer(registry=registry, model_name="risk_scorer", model_loader=loader)
    server.load_production_model()
    return server


# --- Model Registry Tests ---


class TestModelRegistry:
    """Test model registry versioning and promotion."""

    def test_register_model(self, tmp_path):
        registry = ModelRegistry(tmp_path / "reg")
        model_path = tmp_path / "artifacts" / "v1"
        model_path.mkdir(parents=True)
        (model_path / "model.joblib").touch()

        meta = registry.register_model(
            name="test_model",
            version="1.0.0",
            artifact_path=str(model_path),
            model_type="xgboost",
            metrics={"auc": 0.95},
        )

        assert meta.name == "test_model"
        assert meta.version == "1.0.0"
        assert meta.stage == ModelStage.DEVELOPMENT
        assert meta.metrics["auc"] == 0.95

    def test_duplicate_version_rejected(self, tmp_path):
        registry = ModelRegistry(tmp_path / "reg")
        model_path = tmp_path / "artifacts"
        model_path.mkdir(parents=True)
        (model_path / "model.joblib").touch()

        registry.register_model("m", "1.0.0", str(model_path), "xgboost")
        with pytest.raises(ValueError, match="already exists"):
            registry.register_model("m", "1.0.0", str(model_path), "xgboost")

    def test_promote_to_production(self, tmp_path):
        registry = ModelRegistry(tmp_path / "reg")
        model_path = tmp_path / "artifacts"
        model_path.mkdir(parents=True)
        (model_path / "model.joblib").touch()

        registry.register_model("m", "1.0.0", str(model_path), "xgboost")
        meta = registry.promote_model("m", "1.0.0", ModelStage.PRODUCTION)

        assert meta.stage == ModelStage.PRODUCTION
        prod = registry.get_production_model("m")
        assert prod is not None
        assert prod.version == "1.0.0"

    def test_promotion_archives_previous(self, tmp_path):
        registry = ModelRegistry(tmp_path / "reg")
        model_path = tmp_path / "artifacts"
        model_path.mkdir(parents=True)
        (model_path / "model.joblib").touch()

        registry.register_model("m", "1.0.0", str(model_path), "xgboost")
        registry.register_model("m", "2.0.0", str(model_path), "xgboost")
        registry.promote_model("m", "1.0.0", ModelStage.PRODUCTION)
        registry.promote_model("m", "2.0.0", ModelStage.PRODUCTION)

        meta_v1 = registry.get_model_metadata("m", "1.0.0")
        assert meta_v1.stage == ModelStage.ARCHIVED

    def test_rollback(self, tmp_path):
        registry = ModelRegistry(tmp_path / "reg")
        model_path = tmp_path / "artifacts"
        model_path.mkdir(parents=True)
        (model_path / "model.joblib").touch()

        registry.register_model("m", "1.0.0", str(model_path), "xgboost")
        registry.register_model("m", "2.0.0", str(model_path), "xgboost")
        registry.promote_model("m", "1.0.0", ModelStage.PRODUCTION)
        registry.promote_model("m", "2.0.0", ModelStage.PRODUCTION)

        restored = registry.rollback_model("m")
        assert restored is not None
        assert restored.version == "1.0.0"
        assert restored.stage == ModelStage.PRODUCTION

        meta_v2 = registry.get_model_metadata("m", "2.0.0")
        assert meta_v2.stage == ModelStage.ARCHIVED

    def test_ab_test_creation(self, tmp_path):
        registry = ModelRegistry(tmp_path / "reg")
        model_path = tmp_path / "artifacts"
        model_path.mkdir(parents=True)
        (model_path / "model.joblib").touch()

        registry.register_model("m", "1.0.0", str(model_path), "xgboost")
        registry.register_model("m", "2.0.0", str(model_path), "xgboost")

        config = registry.create_ab_test(
            test_name="test_1",
            model_name="m",
            version_a="1.0.0",
            version_b="2.0.0",
            traffic_split=0.3,
        )
        assert config.traffic_split == 0.3
        assert config.is_active

    def test_ab_test_sticky_assignment(self, tmp_path):
        registry = ModelRegistry(tmp_path / "reg")
        model_path = tmp_path / "artifacts"
        model_path.mkdir(parents=True)
        (model_path / "model.joblib").touch()

        registry.register_model("m", "1.0.0", str(model_path), "xgboost")
        registry.register_model("m", "2.0.0", str(model_path), "xgboost")
        registry.create_ab_test("test_1", "m", "1.0.0", "2.0.0", traffic_split=0.5)

        # Same user should always get same assignment
        assignments = set()
        for _ in range(100):
            version = registry.resolve_ab_assignment("test_1", "user_123")
            assignments.add(version)

        assert len(assignments) == 1  # consistent assignment

    def test_ab_test_traffic_distribution(self, tmp_path):
        registry = ModelRegistry(tmp_path / "reg")
        model_path = tmp_path / "artifacts"
        model_path.mkdir(parents=True)
        (model_path / "model.joblib").touch()

        registry.register_model("m", "1.0.0", str(model_path), "xgboost")
        registry.register_model("m", "2.0.0", str(model_path), "xgboost")
        registry.create_ab_test("test_split", "m", "1.0.0", "2.0.0", traffic_split=0.5)

        # Check distribution across many users
        version_counts: dict[str, int] = {"1.0.0": 0, "2.0.0": 0}
        for i in range(1000):
            version = registry.resolve_ab_assignment("test_split", f"user_{i}")
            version_counts[version] += 1

        # With 50% split and 1000 users, expect roughly equal (within 10%)
        ratio = version_counts["2.0.0"] / 1000
        assert 0.4 <= ratio <= 0.6, f"Traffic split is {ratio}, expected ~0.5"

    def test_semantic_version_ordering(self):
        v1 = ModelVersion.parse("1.0.0")
        v2 = ModelVersion.parse("1.1.0")
        v3 = ModelVersion.parse("2.0.0")

        assert v1 < v2 < v3
        assert v1.bump_minor() == v2
        assert v1.bump_major() == ModelVersion(2, 0, 0)


# --- Model Server Tests ---


class TestModelServer:
    """Test model server prediction and hot-reload."""

    def test_single_prediction(self, model_server, feature_names):
        features = np.random.rand(1, len(feature_names))
        result = model_server.predict(features)
        assert result.shape == (1,)
        assert 0.0 <= result[0] <= 1.0

    def test_batch_prediction(self, model_server, feature_names):
        batch = [np.random.rand(1, len(feature_names)) for _ in range(10)]
        results = model_server.predict_batch(batch)
        assert len(results) == 10

    def test_hot_reload_swaps_model(self, registry_with_model, mock_model, tmp_path):
        registry, model_path = registry_with_model

        def loader(path):
            return mock_model

        server = ModelServer(registry=registry, model_name="risk_scorer", model_loader=loader)
        server.load_production_model()
        assert server.active_version == "1.0.0"

        # Register and promote new version
        v2_path = tmp_path / "models" / "v2"
        v2_path.mkdir(parents=True)
        (v2_path / "model.joblib").write_bytes(b"placeholder")

        registry.register_model("risk_scorer", "2.0.0", str(v2_path), "xgboost")
        registry.promote_model("risk_scorer", "2.0.0", ModelStage.PRODUCTION)

        reloaded = server.hot_reload()
        assert reloaded is True
        assert server.active_version == "2.0.0"

    def test_fallback_on_failure(self, registry_with_model, tmp_path):
        registry, model_path = registry_with_model
        call_count = [0]

        def failing_model_predict_proba(X):
            raise RuntimeError("Model crashed")

        good_model = MagicMock()
        good_model.predict_proba = MagicMock(
            return_value=np.array([[0.3, 0.7]])
        )

        bad_model = MagicMock()
        bad_model.predict_proba = MagicMock(side_effect=failing_model_predict_proba)

        models = [good_model, bad_model]

        def loader(path):
            call_count[0] += 1
            return models[call_count[0] - 1]

        server = ModelServer(registry=registry, model_name="risk_scorer", model_loader=loader)
        server.load_production_model()

        # Register and load v2 (which will fail during prediction)
        v2_path = tmp_path / "models" / "v2"
        v2_path.mkdir(parents=True)
        (v2_path / "model.joblib").write_bytes(b"placeholder")
        registry.register_model("risk_scorer", "2.0.0", str(v2_path), "xgboost")
        registry.promote_model("risk_scorer", "2.0.0", ModelStage.PRODUCTION)
        server.hot_reload()

        # Prediction should fall back to good_model
        features = np.random.rand(1, 12)
        result = server.predict(features)
        assert result[0] == pytest.approx(0.7, abs=0.01)

    def test_stats_tracking(self, model_server, feature_names):
        features = np.random.rand(5, len(feature_names))
        model_server.predict(features)

        stats = model_server.stats
        assert stats["prediction_count"] == 5
        assert stats["error_count"] == 0
        assert stats["is_ready"] is True


# --- Load Tests ---


class TestModelServingPerformance:
    """Performance tests targeting 1000 predictions/second."""

    @pytest.mark.performance
    def test_single_prediction_latency(self, model_server, feature_names):
        """Single prediction should complete under 20ms P95."""
        latencies: list[float] = []
        n_requests = 500

        for _ in range(n_requests):
            features = np.random.rand(1, len(feature_names))
            start = time.perf_counter()
            model_server.predict(features)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        p99 = sorted(latencies)[int(0.99 * len(latencies))]

        assert p50 < 10.0, f"P50 latency {p50:.2f}ms exceeds 10ms"
        assert p95 < 20.0, f"P95 latency {p95:.2f}ms exceeds 20ms"
        # Log results
        print(f"\nSingle prediction latency: P50={p50:.2f}ms, P95={p95:.2f}ms, P99={p99:.2f}ms")

    @pytest.mark.performance
    def test_batch_throughput(self, model_server, feature_names):
        """Batch prediction should handle 1000+ predictions/second."""
        batch_size = 100
        n_batches = 20
        total_predictions = batch_size * n_batches

        start = time.perf_counter()
        for _ in range(n_batches):
            features = np.random.rand(batch_size, len(feature_names))
            model_server.predict(features)
        elapsed = time.perf_counter() - start

        throughput = total_predictions / elapsed
        assert throughput >= 1000, f"Throughput {throughput:.0f} predictions/s below 1000 target"
        print(f"\nBatch throughput: {throughput:.0f} predictions/second ({total_predictions} in {elapsed:.2f}s)")

    @pytest.mark.performance
    def test_concurrent_predictions(self, model_server, feature_names):
        """Concurrent predictions should maintain throughput under parallel load."""
        n_threads = 8
        predictions_per_thread = 125  # total = 1000
        latencies: list[float] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def _predict_worker():
            local_latencies: list[float] = []
            for _ in range(predictions_per_thread):
                features = np.random.rand(1, len(feature_names))
                start = time.perf_counter()
                try:
                    model_server.predict(features)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    local_latencies.append(elapsed_ms)
                except Exception as e:
                    with lock:
                        errors.append(e)
            with lock:
                latencies.extend(local_latencies)

        overall_start = time.perf_counter()
        threads = []
        for _ in range(n_threads):
            t = threading.Thread(target=_predict_worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        overall_elapsed = time.perf_counter() - overall_start
        total_preds = n_threads * predictions_per_thread
        throughput = total_preds / overall_elapsed

        assert len(errors) == 0, f"Got {len(errors)} errors during concurrent predictions"
        assert throughput >= 500, f"Concurrent throughput {throughput:.0f}/s below 500 target"
        print(
            f"\nConcurrent throughput: {throughput:.0f} predictions/second "
            f"({n_threads} threads, {total_preds} total, {overall_elapsed:.2f}s)"
        )

    @pytest.mark.performance
    def test_hot_reload_no_dropped_requests(self, registry_with_model, mock_model, tmp_path, feature_names):
        """Hot-reload should not drop any requests."""
        registry, model_path = registry_with_model

        def loader(path):
            return mock_model

        server = ModelServer(registry=registry, model_name="risk_scorer", model_loader=loader)
        server.load_production_model()

        success_count = [0]
        error_count = [0]
        stop_event = threading.Event()
        lock = threading.Lock()

        def _continuous_predict():
            while not stop_event.is_set():
                features = np.random.rand(1, len(feature_names))
                try:
                    server.predict(features)
                    with lock:
                        success_count[0] += 1
                except Exception:
                    with lock:
                        error_count[0] += 1

        # Start prediction threads
        threads = []
        for _ in range(4):
            t = threading.Thread(target=_continuous_predict)
            threads.append(t)
            t.start()

        # Let predictions run, then trigger reload
        time.sleep(0.1)

        v2_path = tmp_path / "models" / "v2"
        v2_path.mkdir(parents=True)
        (v2_path / "model.joblib").write_bytes(b"placeholder")
        registry.register_model("risk_scorer", "2.0.0", str(v2_path), "xgboost")
        registry.promote_model("risk_scorer", "2.0.0", ModelStage.PRODUCTION)
        server.hot_reload()

        # Let predictions continue after reload
        time.sleep(0.1)
        stop_event.set()

        for t in threads:
            t.join()

        assert error_count[0] == 0, f"Dropped {error_count[0]} requests during hot-reload"
        assert success_count[0] > 0
        print(f"\nHot-reload test: {success_count[0]} predictions, 0 dropped")


# --- Model Monitor Tests ---


class TestModelMonitor:
    """Test model monitoring and drift detection."""

    def test_prediction_distribution_tracking(self):
        tracker = PredictionDistributionTracker(window_size=10000)

        # Set baseline with large sample
        baseline = np.random.beta(2, 5, size=5000)
        tracker.set_baseline(baseline)

        # Record similar distribution (no drift) - large sample for stability
        current = np.random.beta(2, 5, size=5000)
        tracker.record_batch(current)

        psi = tracker.compute_psi()
        assert psi is not None
        assert psi < 0.25  # No significant drift (PSI < 0.25)

    def test_prediction_drift_detection(self):
        tracker = PredictionDistributionTracker(window_size=1000)

        # Set baseline with low scores
        baseline = np.random.beta(2, 8, size=1000)
        tracker.set_baseline(baseline)

        # Record shifted distribution (high scores)
        shifted = np.random.beta(8, 2, size=500)
        tracker.record_batch(shifted)

        psi = tracker.compute_psi()
        assert psi is not None
        assert psi > 0.25  # Significant drift

    def test_feature_drift_detection(self, feature_names):
        detector = FeatureDriftDetector(
            feature_names=feature_names[:3],
            psi_threshold=0.25,
        )

        # Set baseline
        baseline_data = {
            feature_names[0]: np.random.normal(100, 20, 1000),
            feature_names[1]: np.random.normal(5, 2, 1000),
            feature_names[2]: np.random.normal(10, 3, 1000),
        }
        detector.set_baseline(baseline_data)

        # Record shifted data for first feature
        for _ in range(500):
            detector.record({
                feature_names[0]: np.random.normal(200, 20),  # shifted!
                feature_names[1]: np.random.normal(5, 2),
                feature_names[2]: np.random.normal(10, 3),
            })

        drifted = detector.get_drifted_features(method="psi")
        drifted_names = [r.feature_name for r in drifted]
        assert feature_names[0] in drifted_names

    def test_data_quality_checker(self, feature_names):
        checker = DataQualityChecker(
            feature_names=feature_names,
            feature_ranges={"transaction_amount": (0.01, 100000.0)},
        )

        # Good data
        good_features = {name: 1.0 for name in feature_names}
        good_features["transaction_amount"] = 50.0
        report = checker.check(good_features)
        assert report.is_healthy

        # Bad data - missing required feature
        bad_features = {"transaction_amount": 50.0}  # missing others
        report = checker.check(bad_features)
        assert not report.is_healthy
        assert report.missing_count > 0

    def test_model_monitor_full_workflow(self, feature_names):
        alerts_received: list = []

        monitor = ModelMonitor(
            model_version="1.0.0",
            feature_names=feature_names,
            prediction_window_size=1000,
            latency_threshold_ms=50.0,
            alert_callback=lambda a: alerts_received.append(a),
        )

        # Set baselines
        baseline_scores = np.random.beta(2, 5, size=1000)
        monitor.set_baselines(prediction_scores=baseline_scores)

        # Record normal predictions
        for _ in range(200):
            score = np.random.beta(2, 5)
            monitor.record_prediction(score=score, latency_ms=5.0)

        # Run checks - should be clean
        alerts = monitor.run_all_checks()
        initial_alert_count = len(alerts)

        # Now simulate high latency
        monitor.record_prediction(score=0.5, latency_ms=100.0)

        # The single recording above triggers inline alert
        health = monitor.get_health_status()
        assert health["status"] in ("healthy", "degraded")
        assert health["total_predictions"] > 0

    def test_monitor_performance_alerts(self, feature_names):
        monitor = ModelMonitor(
            model_version="1.0.0",
            feature_names=feature_names,
            latency_threshold_ms=20.0,
            error_rate_threshold=0.05,
        )

        # Simulate high error rate
        for i in range(200):
            monitor.record_prediction(score=0.5, latency_ms=5.0, is_error=(i % 10 == 0))

        alerts = monitor.check_performance()
        error_alerts = [a for a in alerts if "Error rate" in a.message]
        assert len(error_alerts) > 0
