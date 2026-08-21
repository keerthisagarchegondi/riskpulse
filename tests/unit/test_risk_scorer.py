"""Unit tests for the ML risk scoring model components.

Tests cover:
- FeatureStore: feature retrieval, freshness checks, missing feature handling
- RiskScorer: model loading, prediction, calibration, SHAP explanations
- Training pipeline: data generation, training, evaluation
"""

from __future__ import annotations

from unittest.mock import MagicMock

import joblib
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.fraud_detection.feature_store import (
    FEATURE_CATALOG,
    FeatureStore,
    FeatureVector,
)
from src.fraud_detection.risk_scorer import RiskScore, RiskScorer

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_transaction():
    """Valid transaction dict for scoring."""
    return {
        "transaction_id": "TXN-TEST-001",
        "external_transaction_id": "TXN-TEST-001",
        "customer_id": "CUST-001",
        "account_id": "ACC-001",
        "transaction_amount": 250.00,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "is_international": False,
        "transaction_timestamp": "2026-07-15T14:30:00Z",
        "geo_country": "US",
        "merchant_id": "MERCH-001",
    }


@pytest.fixture
def fraud_transaction():
    """High-risk fraudulent transaction."""
    return {
        "transaction_id": "TXN-FRAUD-001",
        "external_transaction_id": "TXN-FRAUD-001",
        "customer_id": "CUST-002",
        "account_id": "ACC-002",
        "transaction_amount": 9500.00,
        "transaction_currency": "USD",
        "transaction_type": "purchase",
        "channel": "online",
        "is_international": True,
        "transaction_timestamp": "2026-07-15T03:15:00Z",
        "geo_country": "RU",
        "merchant_id": "MERCH-999",
        "last_channel": "pos",
    }


@pytest.fixture
def mock_model_dir(tmp_path):
    """Create a mock model directory with all required artifacts."""
    model_dir = tmp_path / "risk_scorer"
    model_dir.mkdir()

    # Create a simple sklearn model
    rng = np.random.default_rng(42)
    n_features = len(FEATURE_CATALOG)
    X_train = rng.standard_normal((500, n_features))
    y_train = rng.integers(0, 2, size=500)

    model = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    scaler = StandardScaler()
    scaler.fit(X_train)

    # Save artifacts
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(
        {
            "model_version": "1.0.0-test",
            "model_type": "gradient_boosting",
            "feature_names": FEATURE_CATALOG,
            "thresholds": {"low": 0.3, "medium": 0.5, "high": 0.8, "critical": 0.95},
            "n_features": n_features,
        },
        model_dir / "metadata.joblib",
    )

    return model_dir


@pytest.fixture
def feature_store():
    """Feature store with no external dependencies."""
    return FeatureStore(
        db_handler=None,
        cache_handler=None,
        freshness_check_enabled=False,
    )


@pytest.fixture
def risk_scorer(mock_model_dir):
    """Risk scorer with loaded mock model."""
    scorer = RiskScorer(
        model_path=mock_model_dir,
        enable_shap=False,  # Disable SHAP for faster tests
    )
    return scorer


# =============================================================================
# FeatureStore Tests
# =============================================================================


class TestFeatureStore:
    """Tests for FeatureStore."""

    def test_get_features_returns_feature_vector(self, feature_store, sample_transaction):
        """Feature retrieval returns a valid FeatureVector."""
        vector = feature_store.get_features(
            transaction_id="TXN-TEST-001",
            customer_id="CUST-001",
            transaction_data=sample_transaction,
        )
        assert isinstance(vector, FeatureVector)
        assert vector.transaction_id == "TXN-TEST-001"
        assert vector.customer_id == "CUST-001"
        assert vector.feature_count > 0

    def test_get_features_computes_transaction_features(self, feature_store, sample_transaction):
        """Transaction-level features are computed from raw data."""
        vector = feature_store.get_features(
            transaction_id="TXN-TEST-001",
            customer_id="CUST-001",
            transaction_data=sample_transaction,
        )
        assert vector.features["transaction_amount"] == 250.00
        assert vector.features["amount_log"] == pytest.approx(np.log1p(250.0), rel=1e-5)
        assert vector.features["hour_of_day"] == 14.0
        assert vector.features["day_of_week"] == 2.0  # Wednesday
        assert vector.features["is_weekend"] == 0.0

    def test_get_features_computes_cross_features(self, feature_store, sample_transaction):
        """Cross features are computed from existing features."""
        vector = feature_store.get_features(
            transaction_id="TXN-TEST-001",
            customer_id="CUST-001",
            transaction_data=sample_transaction,
        )
        assert "amount_x_velocity" in vector.features
        assert "international_x_new_merchant" in vector.features

    def test_missing_features_filled_with_defaults(self, feature_store, sample_transaction):
        """Missing features are populated with defaults and flagged."""
        vector = feature_store.get_features(
            transaction_id="TXN-TEST-001",
            customer_id="CUST-001",
            transaction_data=sample_transaction,
        )
        # Some features will be missing since no DB/cache
        # But they should be filled with defaults
        for feat_name in FEATURE_CATALOG:
            assert feat_name in vector.features
            assert vector.features[feat_name] is not None

    def test_feature_vector_to_array(self, feature_store, sample_transaction):
        """FeatureVector.to_array returns correct ordered numpy array."""
        vector = feature_store.get_features(
            transaction_id="TXN-TEST-001",
            customer_id="CUST-001",
            transaction_data=sample_transaction,
        )
        arr = vector.to_array(FEATURE_CATALOG)
        assert isinstance(arr, np.ndarray)
        assert len(arr) == len(FEATURE_CATALOG)
        assert not np.any(np.isnan(arr))

    def test_feature_vector_to_dict(self, feature_store, sample_transaction):
        """FeatureVector.to_dict returns serializable dictionary."""
        vector = feature_store.get_features(
            transaction_id="TXN-TEST-001",
            customer_id="CUST-001",
            transaction_data=sample_transaction,
        )
        d = vector.to_dict()
        assert "transaction_id" in d
        assert "features" in d
        assert "missing_features" in d
        assert "computation_time_ms" in d

    def test_get_features_batch(self, feature_store, sample_transaction, fraud_transaction):
        """Batch feature retrieval returns correct number of vectors."""
        transactions = [sample_transaction, fraud_transaction]
        results = feature_store.get_features_batch(transactions)
        assert len(results) == 2
        assert all(isinstance(r, FeatureVector) for r in results)

    def test_is_feature_set_valid_good_quality(self, sample_transaction):
        """Feature set with low missing ratio is valid."""
        # Use a higher threshold to account for no DB/cache
        store = FeatureStore(
            freshness_check_enabled=False,
            max_missing_feature_ratio=0.8,
        )
        vector = store.get_features(
            transaction_id="TXN-TEST-001",
            customer_id="CUST-001",
            transaction_data=sample_transaction,
        )
        assert store.is_feature_set_valid(vector)

    def test_feature_store_with_cache(self, sample_transaction):
        """Feature store retrieves from cache when available."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = {"txn_count_24h": 10.0, "txn_amount_sum_24h": 500.0}

        store = FeatureStore(cache_handler=mock_cache, freshness_check_enabled=False)
        vector = store.get_features(
            transaction_id="TXN-TEST-001",
            customer_id="CUST-001",
            transaction_data=sample_transaction,
        )
        assert vector.features["txn_count_24h"] == 10.0
        assert vector.features["txn_amount_sum_24h"] == 500.0

    def test_feature_store_stats(self, feature_store, sample_transaction):
        """Stats are tracked correctly."""
        feature_store.get_features("TXN-1", "CUST-1", sample_transaction)
        feature_store.get_features("TXN-2", "CUST-1", sample_transaction)
        stats = feature_store.get_stats()
        assert stats["total_retrievals"] == 2
        assert "cache_hit_rate" in stats

    def test_international_flag_computed(self, feature_store, fraud_transaction):
        """International flag correctly computed from transaction data."""
        vector = feature_store.get_features(
            transaction_id="TXN-FRAUD-001",
            customer_id="CUST-002",
            transaction_data=fraud_transaction,
        )
        assert vector.features["is_international"] == 1.0

    def test_channel_switch_detected(self, feature_store, fraud_transaction):
        """Channel switch flag detected when channels differ."""
        vector = feature_store.get_features(
            transaction_id="TXN-FRAUD-001",
            customer_id="CUST-002",
            transaction_data=fraud_transaction,
        )
        assert vector.features["channel_switch_flag"] == 1.0

    def test_unusual_hour_flag(self, feature_store, fraud_transaction):
        """Unusual hour flag set for early morning transactions."""
        vector = feature_store.get_features(
            transaction_id="TXN-FRAUD-001",
            customer_id="CUST-002",
            transaction_data=fraud_transaction,
        )
        assert vector.features["unusual_hour_flag"] == 1.0


# =============================================================================
# RiskScorer Tests
# =============================================================================


class TestRiskScorer:
    """Tests for RiskScorer."""

    def test_load_model(self, mock_model_dir):
        """Model loads successfully from disk."""
        scorer = RiskScorer(model_path=mock_model_dir, enable_shap=False)
        assert scorer.is_loaded
        assert scorer.model_version == "1.0.0-test"

    def test_load_model_missing_path(self):
        """Raises FileNotFoundError for non-existent path."""
        scorer = RiskScorer(enable_shap=False)
        with pytest.raises(FileNotFoundError):
            scorer.load_model("/nonexistent/path")

    def test_predict_returns_risk_score(self, risk_scorer, sample_transaction):
        """Prediction returns valid RiskScore object."""
        result = risk_scorer.predict(sample_transaction)
        assert isinstance(result, RiskScore)
        assert result.transaction_id == "TXN-TEST-001"
        assert 0.0 <= result.risk_score <= 1.0
        assert result.risk_level in ("low", "medium", "high", "critical")
        assert result.model_version == "1.0.0-test"
        assert result.prediction_latency_ms > 0

    def test_predict_not_loaded_raises(self):
        """Prediction without loaded model raises RuntimeError."""
        scorer = RiskScorer(enable_shap=False)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            scorer.predict({"transaction_id": "X"})

    def test_predict_score_range(self, risk_scorer, sample_transaction):
        """All scores are within valid ranges."""
        result = risk_scorer.predict(sample_transaction)
        assert 0.0 <= result.risk_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.prediction_latency_ms >= 0

    def test_predict_batch(self, risk_scorer, sample_transaction, fraud_transaction):
        """Batch prediction returns correct number of results."""
        transactions = [sample_transaction, fraud_transaction]
        results = risk_scorer.predict_batch(transactions)
        assert len(results) == 2
        assert all(isinstance(r, RiskScore) for r in results)
        assert all(0.0 <= r.risk_score <= 1.0 for r in results)

    def test_risk_classification_thresholds(self, risk_scorer):
        """Risk classification respects threshold boundaries."""
        assert risk_scorer._classify_risk(0.1) == "low"
        assert risk_scorer._classify_risk(0.29) == "low"
        assert risk_scorer._classify_risk(0.6) == "medium"
        assert risk_scorer._classify_risk(0.85) == "high"
        assert risk_scorer._classify_risk(0.96) == "critical"

    def test_feature_quality_assessment(self, risk_scorer):
        """Feature quality correctly assessed based on missing/stale ratios."""
        good_vector = FeatureVector(
            transaction_id="X",
            customer_id="C",
            features={f: 1.0 for f in FEATURE_CATALOG},
            missing_features=[],
            stale_features=[],
        )
        assert risk_scorer._assess_feature_quality(good_vector) == "good"

        degraded_vector = FeatureVector(
            transaction_id="X",
            customer_id="C",
            features={f: 1.0 for f in FEATURE_CATALOG},
            missing_features=FEATURE_CATALOG[:8],  # ~15% missing
            stale_features=[],
        )
        assert risk_scorer._assess_feature_quality(degraded_vector) == "degraded"

        poor_vector = FeatureVector(
            transaction_id="X",
            customer_id="C",
            features={f: 1.0 for f in FEATURE_CATALOG},
            missing_features=FEATURE_CATALOG[:20],  # >30% missing
            stale_features=[],
        )
        assert risk_scorer._assess_feature_quality(poor_vector) == "poor"

    def test_confidence_computation(self, risk_scorer):
        """Confidence is high when score is far from 0.5."""
        conf_high = risk_scorer._compute_confidence(0.95, "good")
        conf_low = risk_scorer._compute_confidence(0.52, "good")
        assert conf_high > conf_low

    def test_confidence_degraded_quality(self, risk_scorer):
        """Confidence is reduced for degraded feature quality."""
        conf_good = risk_scorer._compute_confidence(0.9, "good")
        conf_degraded = risk_scorer._compute_confidence(0.9, "degraded")
        assert conf_good > conf_degraded

    def test_save_and_reload_model(self, risk_scorer, tmp_path):
        """Model can be saved and reloaded."""
        rng = np.random.default_rng(42)
        n_features = len(FEATURE_CATALOG)
        X = rng.standard_normal((100, n_features))
        y = rng.integers(0, 2, size=100)

        model = GradientBoostingClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        scaler = StandardScaler().fit(X)

        save_path = tmp_path / "saved_model"
        risk_scorer.save_model(
            output_path=save_path,
            model=model,
            scaler=scaler,
            feature_names=FEATURE_CATALOG,
            model_type="gradient_boosting",
            model_version="2.0.0",
        )

        # Reload
        new_scorer = RiskScorer(model_path=save_path, enable_shap=False)
        assert new_scorer.is_loaded
        assert new_scorer.model_version == "2.0.0"

    def test_risk_score_to_dict(self, risk_scorer, sample_transaction):
        """RiskScore.to_dict returns complete serializable dict."""
        result = risk_scorer.predict(sample_transaction)
        d = result.to_dict()
        assert "transaction_id" in d
        assert "risk_score" in d
        assert "risk_level" in d
        assert "confidence" in d
        assert "model_version" in d
        assert "prediction_latency_ms" in d

    def test_predict_latency_under_sla(self, risk_scorer, sample_transaction):
        """Single prediction completes under 20ms SLA (generous for test env)."""
        import time

        start = time.perf_counter()
        risk_scorer.predict(sample_transaction)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Use 200ms limit for test environments (CI may be slow)
        assert elapsed_ms < 200.0

    def test_feature_names_property(self, risk_scorer):
        """Feature names accessible from loaded model."""
        assert risk_scorer.feature_names == FEATURE_CATALOG
        assert len(risk_scorer.feature_names) > 50


# =============================================================================
# Training Pipeline Tests
# =============================================================================


class TestTrainingPipeline:
    """Tests for the training pipeline functions."""

    def test_generate_synthetic_data(self):
        """Synthetic data has correct shape and fraud ratio."""
        from ml.training.train_risk_scorer import generate_synthetic_data

        X, y = generate_synthetic_data(n_samples=1000, fraud_ratio=0.05, random_state=42)
        assert len(X) == 1000
        assert len(y) == 1000
        assert abs(y.mean() - 0.05) < 0.02  # Approximate ratio
        # Should have 50+ feature columns (plus timestamp and transaction_id)
        feature_cols = [c for c in X.columns if c not in {"timestamp", "transaction_id"}]
        assert len(feature_cols) >= 50

    def test_time_based_split(self):
        """Time-based split preserves temporal order."""
        from ml.training.train_risk_scorer import generate_synthetic_data, time_based_split

        X, y = generate_synthetic_data(n_samples=1000, fraud_ratio=0.02, random_state=42)
        X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(X, y)

        # Check sizes
        assert len(X_train) == 700
        assert len(X_val) == 150
        assert len(X_test) == 150

        # Check temporal order preserved
        if "timestamp" in X_train.columns:
            assert X_train["timestamp"].max() <= X_val["timestamp"].min()
            assert X_val["timestamp"].max() <= X_test["timestamp"].min()

    def test_get_feature_columns(self):
        """Feature columns exclude metadata columns."""
        from ml.training.train_risk_scorer import generate_synthetic_data, get_feature_columns

        X, _ = generate_synthetic_data(n_samples=100, fraud_ratio=0.02, random_state=42)
        cols = get_feature_columns(X)
        assert "timestamp" not in cols
        assert "transaction_id" not in cols
        assert len(cols) >= 50

    def test_compute_metrics(self):
        """Metrics computation returns expected keys."""
        from ml.training.train_risk_scorer import _compute_metrics

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.15, 0.25, 0.85, 0.4])

        metrics = _compute_metrics(y_true, y_pred_proba, threshold=0.5)
        assert "auc_roc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["auc_roc"] <= 1.0
        assert metrics["n_samples"] == 10

    def test_feature_importance_extraction(self):
        """Feature importance extraction returns ranked list."""
        from ml.training.train_risk_scorer import compute_feature_importance

        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 10))
        y = rng.integers(0, 2, size=200)
        model = GradientBoostingClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        feature_names = [f"feat_{i}" for i in range(10)]
        importance = compute_feature_importance(model, feature_names)
        assert len(importance) == 10
        assert importance[0]["rank"] == 1
        assert all(item["importance"] >= 0 for item in importance)
        # Check sorted descending
        for i in range(len(importance) - 1):
            assert importance[i]["importance"] >= importance[i + 1]["importance"]

    def test_train_model_xgboost(self):
        """XGBoost model trains and produces valid metrics."""
        pytest.importorskip("xgboost")
        from ml.training.train_risk_scorer import (
            generate_synthetic_data,
            get_feature_columns,
            time_based_split,
            train_model,
        )

        X, y = generate_synthetic_data(n_samples=2000, fraud_ratio=0.05, random_state=42)
        X_train, X_val, _, y_train, y_val, _ = time_based_split(X, y)
        feature_cols = get_feature_columns(X_train)

        model, scaler, metrics = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            feature_cols=feature_cols,
            model_type="xgboost",
            class_weight_strategy="balanced",
            random_state=42,
        )

        assert model is not None
        assert scaler is not None
        assert metrics["auc_roc"] > 0.5  # Better than random
        assert hasattr(model, "predict_proba")

    def test_train_model_lightgbm(self):
        """LightGBM model trains and produces valid metrics."""
        pytest.importorskip("lightgbm")
        from ml.training.train_risk_scorer import (
            generate_synthetic_data,
            get_feature_columns,
            time_based_split,
            train_model,
        )

        X, y = generate_synthetic_data(n_samples=2000, fraud_ratio=0.05, random_state=42)
        X_train, X_val, _, y_train, y_val, _ = time_based_split(X, y)
        feature_cols = get_feature_columns(X_train)

        model, scaler, metrics = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            feature_cols=feature_cols,
            model_type="lightgbm",
            class_weight_strategy="balanced",
            random_state=42,
        )

        assert model is not None
        assert metrics["auc_roc"] > 0.5

    def test_calibrate_model(self):
        """Score calibration produces valid probabilities."""
        from ml.training.train_risk_scorer import calibrate_model

        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 10))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Deterministic labels for stable calibration

        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X[:350], y[:350])

        calibrator = calibrate_model(model, X[350:], y[350:], method="isotonic")
        assert hasattr(calibrator, "predict_proba")

        proba = calibrator.predict_proba(X[350:])
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_benchmark_latency(self):
        """Latency benchmark returns expected metrics."""
        from ml.training.train_risk_scorer import benchmark_latency

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        y = rng.integers(0, 2, size=100)

        model = GradientBoostingClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        scaler = StandardScaler().fit(X)

        result = benchmark_latency(model, scaler, X, n_iterations=100, warmup=10)
        assert "mean_ms" in result
        assert "p95_ms" in result
        assert "p99_ms" in result
        assert "meets_sla" in result
        assert result["mean_ms"] > 0
        assert result["p95_ms"] >= result["mean_ms"]

    def test_full_training_pipeline(self, tmp_path):
        """End-to-end training pipeline completes successfully."""
        pytest.importorskip("xgboost")
        from ml.training.train_risk_scorer import run_training_pipeline

        results = run_training_pipeline(
            n_samples=2000,
            fraud_ratio=0.05,
            model_type="xgboost",
            class_weight_strategy="balanced",
            output_dir=tmp_path / "model_output",
            random_state=42,
            params={"n_estimators": 50, "max_depth": 3},  # Small for speed
        )

        assert results["production_ready"] is not None
        assert results["test_metrics"]["auc_roc"] > 0.5
        assert results["n_features"] >= 50
        assert (tmp_path / "model_output" / "model.joblib").exists()
        assert (tmp_path / "model_output" / "scaler.joblib").exists()
        assert (tmp_path / "model_output" / "metadata.joblib").exists()
        assert (tmp_path / "model_output" / "evaluation_report.json").exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestRiskScorerIntegration:
    """Integration tests combining FeatureStore and RiskScorer."""

    def test_end_to_end_scoring(self, mock_model_dir, sample_transaction):
        """Complete scoring flow from raw transaction to risk score."""
        feature_store = FeatureStore(freshness_check_enabled=False)
        scorer = RiskScorer(
            model_path=mock_model_dir,
            feature_store=feature_store,
            enable_shap=False,
        )

        result = scorer.predict(sample_transaction)
        assert isinstance(result, RiskScore)
        assert result.transaction_id == "TXN-TEST-001"
        assert 0.0 <= result.risk_score <= 1.0
        assert result.risk_level in ("low", "medium", "high", "critical")

    def test_batch_scoring_consistency(self, mock_model_dir, sample_transaction):
        """Batch scoring produces consistent results with single scoring."""
        scorer = RiskScorer(model_path=mock_model_dir, enable_shap=False)

        single_result = scorer.predict(sample_transaction)
        batch_results = scorer.predict_batch([sample_transaction])

        assert len(batch_results) == 1
        # Scores should be identical (same input)
        assert abs(single_result.risk_score - batch_results[0].risk_score) < 1e-6

    def test_different_transactions_produce_results(
        self, mock_model_dir, sample_transaction, fraud_transaction
    ):
        """Different transactions produce valid risk scores."""
        scorer = RiskScorer(model_path=mock_model_dir, enable_shap=False)

        legit_result = scorer.predict(sample_transaction)
        fraud_result = scorer.predict(fraud_transaction)

        # Both produce valid scores
        assert 0.0 <= legit_result.risk_score <= 1.0
        assert 0.0 <= fraud_result.risk_score <= 1.0
        # Feature quality reflects that many features come from defaults
        assert legit_result.feature_quality in ("good", "degraded", "poor")
        assert fraud_result.feature_quality in ("good", "degraded", "poor")
