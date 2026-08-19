"""Gradient boosted tree risk scoring model for fraud detection.

Provides model loading, real-time prediction, score calibration
(raw score → probability 0-1), SHAP-based explanations, and
model versioning support.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.fraud_detection.feature_store import (
    FEATURE_CATALOG,
    FEATURE_DEFAULTS,
    FeatureStore,
    FeatureVector,
)

logger = logging.getLogger(__name__)


@dataclass
class RiskScore:
    """Result of risk scoring for a single transaction."""

    transaction_id: str
    risk_score: float  # calibrated probability 0.0 to 1.0
    risk_level: str  # low, medium, high, critical
    raw_score: float  # uncalibrated model output
    confidence: float  # model confidence 0.0 to 1.0
    top_features: list[dict[str, Any]] = field(default_factory=list)
    prediction_latency_ms: float = 0.0
    model_version: str = ""
    feature_quality: str = "good"  # good, degraded, poor

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "raw_score": self.raw_score,
            "confidence": self.confidence,
            "top_features": self.top_features,
            "prediction_latency_ms": self.prediction_latency_ms,
            "model_version": self.model_version,
            "feature_quality": self.feature_quality,
        }


@dataclass
class ModelArtifact:
    """Container for a loaded model artifact and its metadata."""

    model: Any
    scaler: StandardScaler | None
    calibrator: Any | None
    feature_names: list[str]
    model_version: str
    model_type: str
    thresholds: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


class RiskScorer:
    """Gradient boosted tree risk scorer for real-time fraud detection.

    Loads a trained XGBoost/LightGBM model and provides:
    - Low-latency prediction (< 20ms per transaction)
    - Calibrated probability scores
    - SHAP-based feature explanations
    - Model versioning with hot-reload support
    """

    DEFAULT_THRESHOLDS = {
        "low": 0.3,
        "medium": 0.5,
        "high": 0.8,
        "critical": 0.95,
    }

    def __init__(
        self,
        model_path: str | Path | None = None,
        feature_store: FeatureStore | None = None,
        thresholds: dict[str, float] | None = None,
        enable_shap: bool = True,
        shap_max_features: int = 10,
    ) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._feature_store = feature_store or FeatureStore()
        self._thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self._enable_shap = enable_shap
        self._shap_max_features = shap_max_features

        self._artifact: ModelArtifact | None = None
        self._shap_explainer: Any | None = None
        self._is_loaded = False

        if self._model_path and self._model_path.exists():
            self.load_model(self._model_path)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_version(self) -> str:
        if self._artifact:
            return self._artifact.model_version
        return "not_loaded"

    @property
    def feature_names(self) -> list[str]:
        if self._artifact:
            return self._artifact.feature_names
        return FEATURE_CATALOG

    def load_model(self, model_path: str | Path) -> None:
        """Load a trained model artifact from disk.

        Expects directory structure:
            model_path/
                model.joblib       - Trained model
                scaler.joblib      - Feature scaler
                calibrator.joblib  - Score calibrator (optional)
                metadata.joblib    - Model metadata
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")

        model_file = path / "model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        model = joblib.load(model_file)

        scaler = None
        scaler_file = path / "scaler.joblib"
        if scaler_file.exists():
            scaler = joblib.load(scaler_file)

        calibrator = None
        calibrator_file = path / "calibrator.joblib"
        if calibrator_file.exists():
            calibrator = joblib.load(calibrator_file)

        metadata_file = path / "metadata.joblib"
        metadata = {}
        if metadata_file.exists():
            metadata = joblib.load(metadata_file)

        feature_names = metadata.get("feature_names", FEATURE_CATALOG)
        model_version = metadata.get("model_version", self._compute_version(model_file))
        model_type = metadata.get("model_type", "unknown")
        thresholds = metadata.get("thresholds", self._thresholds)

        self._artifact = ModelArtifact(
            model=model,
            scaler=scaler,
            calibrator=calibrator,
            feature_names=feature_names,
            model_version=model_version,
            model_type=model_type,
            thresholds=thresholds,
            metadata=metadata,
        )

        if self._enable_shap:
            self._init_shap_explainer()

        self._is_loaded = True
        logger.info(
            "Loaded risk scoring model: version=%s, type=%s, features=%d",
            model_version,
            model_type,
            len(feature_names),
        )

    def predict(self, transaction_data: dict[str, Any]) -> RiskScore:
        """Score a single transaction for fraud risk.

        Args:
            transaction_data: Raw transaction dict with fields needed for feature computation.

        Returns:
            RiskScore with calibrated probability and explanations.
        """
        if not self._is_loaded or self._artifact is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start = time.perf_counter()

        transaction_id = transaction_data.get(
            "transaction_id", transaction_data.get("external_transaction_id", "unknown")
        )
        customer_id = transaction_data.get("customer_id", "unknown")

        # Retrieve features
        feature_vector = self._feature_store.get_features(
            transaction_id=transaction_id,
            customer_id=customer_id,
            transaction_data=transaction_data,
        )

        # Determine feature quality
        feature_quality = self._assess_feature_quality(feature_vector)

        # Prepare feature array
        X = feature_vector.to_array(self._artifact.feature_names).reshape(1, -1)

        # Scale features
        if self._artifact.scaler is not None:
            X = self._artifact.scaler.transform(X)

        # Get raw prediction
        raw_score = self._get_raw_score(X)

        # Calibrate score
        calibrated_score = self._calibrate_score(raw_score, X)

        # Compute confidence
        confidence = self._compute_confidence(calibrated_score, feature_quality)

        # Classify risk level
        risk_level = self._classify_risk(calibrated_score)

        # Get SHAP explanations
        top_features = []
        if self._enable_shap:
            top_features = self._explain_prediction(X, feature_vector)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return RiskScore(
            transaction_id=transaction_id,
            risk_score=float(calibrated_score),
            risk_level=risk_level,
            raw_score=float(raw_score),
            confidence=float(confidence),
            top_features=top_features,
            prediction_latency_ms=elapsed_ms,
            model_version=self._artifact.model_version,
            feature_quality=feature_quality,
        )

    def predict_batch(self, transactions: list[dict[str, Any]]) -> list[RiskScore]:
        """Score a batch of transactions.

        Optimized for batch processing with vectorized feature retrieval.
        """
        if not self._is_loaded or self._artifact is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results: list[RiskScore] = []
        feature_vectors = self._feature_store.get_features_batch(transactions)

        # Build feature matrix
        feature_arrays = []
        for fv in feature_vectors:
            feature_arrays.append(fv.to_array(self._artifact.feature_names))
        X = np.vstack(feature_arrays)

        # Scale
        if self._artifact.scaler is not None:
            X = self._artifact.scaler.transform(X)

        # Batch prediction
        raw_scores = self._get_raw_scores_batch(X)
        calibrated_scores = self._calibrate_scores_batch(raw_scores, X)

        for i, txn in enumerate(transactions):
            transaction_id = txn.get(
                "transaction_id", txn.get("external_transaction_id", f"batch_{i}")
            )
            fv = feature_vectors[i]
            feature_quality = self._assess_feature_quality(fv)
            risk_level = self._classify_risk(calibrated_scores[i])
            confidence = self._compute_confidence(calibrated_scores[i], feature_quality)

            results.append(
                RiskScore(
                    transaction_id=transaction_id,
                    risk_score=float(calibrated_scores[i]),
                    risk_level=risk_level,
                    raw_score=float(raw_scores[i]),
                    confidence=float(confidence),
                    top_features=[],  # Skip SHAP for batch performance
                    prediction_latency_ms=0.0,
                    model_version=self._artifact.model_version,
                    feature_quality=feature_quality,
                )
            )

        return results

    def save_model(
        self,
        output_path: str | Path,
        model: Any,
        scaler: StandardScaler | None = None,
        calibrator: Any | None = None,
        feature_names: list[str] | None = None,
        model_type: str = "xgboost",
        model_version: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save a trained model artifact to disk.

        Args:
            output_path: Directory to save model artifacts.
            model: Trained model object.
            scaler: Feature scaler (fitted).
            calibrator: Score calibrator (fitted).
            feature_names: Ordered list of feature names.
            model_type: Model type identifier.
            model_version: Semantic version string.
            extra_metadata: Additional metadata to store.

        Returns:
            Path to saved model directory.
        """
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)

        model_file = path / "model.joblib"
        joblib.dump(model, model_file)

        if scaler is not None:
            joblib.dump(scaler, path / "scaler.joblib")

        if calibrator is not None:
            joblib.dump(calibrator, path / "calibrator.joblib")

        version = model_version or self._compute_version(model_file)
        metadata = {
            "model_version": version,
            "model_type": model_type,
            "feature_names": feature_names or FEATURE_CATALOG,
            "thresholds": self._thresholds,
            "n_features": len(feature_names or FEATURE_CATALOG),
            **(extra_metadata or {}),
        }
        joblib.dump(metadata, path / "metadata.joblib")

        logger.info("Saved risk scoring model to %s (version=%s)", path, version)
        return path

    def reload_model(self) -> None:
        """Reload model from original path (for hot-reload support)."""
        if self._model_path and self._model_path.exists():
            self.load_model(self._model_path)
            logger.info("Hot-reloaded model version=%s", self.model_version)

    def _get_raw_score(self, X: np.ndarray) -> float:
        """Get raw model prediction score."""
        model = self._artifact.model
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
        elif hasattr(model, "decision_function"):
            return float(model.decision_function(X)[0])
        else:
            return float(model.predict(X)[0])

    def _get_raw_scores_batch(self, X: np.ndarray) -> np.ndarray:
        """Get raw model prediction scores for a batch."""
        model = self._artifact.model
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        elif hasattr(model, "decision_function"):
            return model.decision_function(X)
        else:
            return model.predict(X).astype(float)

    def _calibrate_score(self, raw_score: float, X: np.ndarray) -> float:
        """Calibrate raw score to probability using isotonic regression or Platt scaling."""
        if self._artifact.calibrator is not None:
            try:
                if hasattr(self._artifact.calibrator, "predict_proba"):
                    proba = self._artifact.calibrator.predict_proba(X)
                    return float(np.clip(proba[0, 1], 0.0, 1.0))
                elif hasattr(self._artifact.calibrator, "transform"):
                    return float(
                        np.clip(self._artifact.calibrator.transform([[raw_score]])[0, 0], 0.0, 1.0)
                    )
            except Exception as e:
                logger.debug("Calibration fallback: %s", e)

        # Fallback: sigmoid calibration
        return float(np.clip(raw_score, 0.0, 1.0))

    def _calibrate_scores_batch(self, raw_scores: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Calibrate a batch of raw scores."""
        if self._artifact.calibrator is not None:
            try:
                if hasattr(self._artifact.calibrator, "predict_proba"):
                    proba = self._artifact.calibrator.predict_proba(X)
                    return np.clip(proba[:, 1], 0.0, 1.0)
            except Exception as e:
                logger.debug("Batch calibration fallback: %s", e)
        return np.clip(raw_scores, 0.0, 1.0)

    def _classify_risk(self, score: float) -> str:
        """Classify calibrated score into risk level."""
        if score >= self._thresholds.get("critical", 0.95):
            return "critical"
        elif score >= self._thresholds.get("high", 0.8):
            return "high"
        elif score >= self._thresholds.get("medium", 0.5):
            return "medium"
        return "low"

    def _compute_confidence(self, score: float, feature_quality: str) -> float:
        """Compute prediction confidence based on score certainty and feature quality."""
        # Score certainty: high confidence when far from 0.5 (decision boundary)
        score_certainty = abs(score - 0.5) * 2.0

        # Feature quality discount
        quality_multiplier = {"good": 1.0, "degraded": 0.8, "poor": 0.5}.get(feature_quality, 0.7)

        return float(np.clip(score_certainty * quality_multiplier, 0.0, 1.0))

    def _assess_feature_quality(self, feature_vector: FeatureVector) -> str:
        """Assess the quality of the feature vector."""
        total_features = (
            len(self._artifact.feature_names) if self._artifact else len(FEATURE_CATALOG)
        )
        missing_ratio = len(feature_vector.missing_features) / max(total_features, 1)
        stale_ratio = len(feature_vector.stale_features) / max(total_features, 1)

        combined_degradation = missing_ratio + stale_ratio * 0.5
        if combined_degradation > 0.3:
            return "poor"
        elif combined_degradation > 0.1:
            return "degraded"
        return "good"

    def _init_shap_explainer(self) -> None:
        """Initialize SHAP explainer for the loaded model."""
        try:
            import shap

            model = self._artifact.model
            model_type = self._artifact.model_type

            if model_type in ("xgboost", "lightgbm"):
                self._shap_explainer = shap.TreeExplainer(model)
            else:
                # Fallback to KernelExplainer with a small background
                self._shap_explainer = None
                logger.info("SHAP TreeExplainer not available for model type: %s", model_type)
        except ImportError:
            logger.warning("SHAP library not available. Explanations disabled.")
            self._enable_shap = False
        except Exception as e:
            logger.warning("Failed to initialize SHAP explainer: %s", e)
            self._shap_explainer = None

    def _explain_prediction(
        self, X: np.ndarray, feature_vector: FeatureVector
    ) -> list[dict[str, Any]]:
        """Generate SHAP explanations for a prediction."""
        if self._shap_explainer is None:
            return self._fallback_explanation(X, feature_vector)

        try:
            import shap

            shap_values = self._shap_explainer.shap_values(X)

            # Handle multi-output (binary classification returns list)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class

            feature_names = self._artifact.feature_names
            contributions = shap_values[0] if shap_values.ndim > 1 else shap_values

            # Build feature importance list
            importance_list = []
            for i, (name, shap_val) in enumerate(zip(feature_names, contributions)):
                feat_value = feature_vector.features.get(name, 0.0)
                importance_list.append(
                    {
                        "feature": name,
                        "shap_value": float(shap_val),
                        "feature_value": float(feat_value),
                        "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
                    }
                )

            # Sort by absolute SHAP value and take top N
            importance_list.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            return importance_list[: self._shap_max_features]

        except Exception as e:
            logger.debug("SHAP explanation failed, using fallback: %s", e)
            return self._fallback_explanation(X, feature_vector)

    def _fallback_explanation(
        self, X: np.ndarray, feature_vector: FeatureVector
    ) -> list[dict[str, Any]]:
        """Fallback explanation using feature importances from the model."""
        if not hasattr(self._artifact.model, "feature_importances_"):
            return []

        importances = self._artifact.model.feature_importances_
        feature_names = self._artifact.feature_names

        importance_list = []
        for name, imp in zip(feature_names, importances):
            if imp > 0:
                feat_value = feature_vector.features.get(name, 0.0)
                importance_list.append(
                    {
                        "feature": name,
                        "importance": float(imp),
                        "feature_value": float(feat_value),
                        "direction": "contributes",
                    }
                )

        importance_list.sort(key=lambda x: x["importance"], reverse=True)
        return importance_list[: self._shap_max_features]

    @staticmethod
    def _compute_version(model_file: Path) -> str:
        """Compute a version hash from the model file."""
        h = hashlib.sha256()
        with open(model_file, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
