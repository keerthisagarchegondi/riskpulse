"""Isolation Forest-based anomaly detection for transaction fraud scoring."""

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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ANOMALY_FEATURES = [
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


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single transaction."""

    transaction_id: str
    anomaly_score: float  # -1 (anomaly) to 1 (normal)
    is_anomaly: bool
    confidence: float  # 0.0 to 1.0
    contributing_features: list[dict[str, Any]] = field(default_factory=list)
    prediction_latency_ms: float = 0.0
    model_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "confidence": self.confidence,
            "contributing_features": self.contributing_features,
            "prediction_latency_ms": self.prediction_latency_ms,
            "model_version": self.model_version,
        }


@dataclass
class ModelMetadata:
    """Metadata for a trained model artifact."""

    model_version: str
    trained_at: str
    n_estimators: int
    max_samples: int | str
    contamination: float
    feature_names: list[str]
    training_samples: int
    random_state: int
    checksum: str = ""


class AnomalyDetector:
    """Isolation Forest anomaly detector for real-time transaction scoring.

    Wraps scikit-learn IsolationForest with feature preprocessing,
    model persistence, and low-latency online prediction.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: int | str = "auto",
        contamination: float = 0.02,
        random_state: int = 42,
        max_features: float = 0.8,
        threshold_override: float | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_samples = max_samples
        self._contamination = contamination
        self._random_state = random_state
        self._max_features = max_features
        self._threshold_override = threshold_override
        self._feature_names = feature_names or ANOMALY_FEATURES

        self._model: IsolationForest | None = None
        self._scaler: StandardScaler | None = None
        self._is_fitted = False
        self._model_version = ""
        self._training_samples = 0

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names.copy()

    def fit(self, X: pd.DataFrame) -> "AnomalyDetector":
        """Train the Isolation Forest model on feature data.

        Args:
            X: DataFrame with columns matching self._feature_names.

        Returns:
            self for method chaining.
        """
        features = self._select_features(X)
        logger.info(
            "Training Isolation Forest: samples=%d, features=%d, contamination=%.4f",
            len(features),
            features.shape[1],
            self._contamination,
        )

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(features)

        self._model = IsolationForest(
            n_estimators=self._n_estimators,
            max_samples=self._max_samples,
            contamination=self._contamination,
            random_state=self._random_state,
            max_features=self._max_features,
            n_jobs=-1,
            warm_start=False,
        )
        self._model.fit(X_scaled)

        self._is_fitted = True
        self._training_samples = len(features)
        self._model_version = self._compute_version()

        logger.info("Model trained: version=%s", self._model_version)
        return self

    def predict(self, transaction_features: dict[str, Any]) -> AnomalyResult:
        """Score a single transaction for anomaly detection.

        Args:
            transaction_features: Dict with keys matching feature_names.

        Returns:
            AnomalyResult with score, classification, and metadata.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        start = time.perf_counter()

        feature_vector = self._extract_feature_vector(transaction_features)
        X_input = pd.DataFrame([feature_vector], columns=self._feature_names)
        X_scaled = self._scaler.transform(X_input)

        raw_score = self._model.decision_function(X_scaled)[0]
        prediction = self._model.predict(X_scaled)[0]

        # Normalize score to [-1, 1] range
        anomaly_score = float(np.clip(raw_score, -1.0, 1.0))

        if self._threshold_override is not None:
            is_anomaly = anomaly_score < self._threshold_override
        else:
            is_anomaly = prediction == -1

        confidence = self._compute_confidence(anomaly_score)
        contributing = self._identify_contributing_features(feature_vector, X_scaled[0])

        latency_ms = (time.perf_counter() - start) * 1000

        return AnomalyResult(
            transaction_id=transaction_features.get("transaction_id", "unknown"),
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            confidence=confidence,
            contributing_features=contributing,
            prediction_latency_ms=round(latency_ms, 3),
            model_version=self._model_version,
        )

    def predict_batch(self, X: pd.DataFrame) -> list[AnomalyResult]:
        """Score a batch of transactions.

        Args:
            X: DataFrame with feature columns and optional transaction_id column.

        Returns:
            List of AnomalyResult for each row.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        start = time.perf_counter()
        features = self._select_features(X)
        X_scaled = self._scaler.transform(features)

        raw_scores = self._model.decision_function(X_scaled)
        predictions = self._model.predict(X_scaled)

        results = []
        for i in range(len(X)):
            score = float(np.clip(raw_scores[i], -1.0, 1.0))
            if self._threshold_override is not None:
                is_anomaly = score < self._threshold_override
            else:
                is_anomaly = predictions[i] == -1

            txn_id = (
                str(X.iloc[i].get("transaction_id", f"txn_{i}"))
                if "transaction_id" in X.columns
                else f"txn_{i}"
            )

            results.append(
                AnomalyResult(
                    transaction_id=txn_id,
                    anomaly_score=score,
                    is_anomaly=is_anomaly,
                    confidence=self._compute_confidence(score),
                    prediction_latency_ms=0.0,
                    model_version=self._model_version,
                )
            )

        total_ms = (time.perf_counter() - start) * 1000
        avg_ms = total_ms / len(X) if len(X) > 0 else 0
        logger.info(
            "Batch prediction: n=%d, total_ms=%.1f, avg_ms=%.3f",
            len(X),
            total_ms,
            avg_ms,
        )
        return results

    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw anomaly scores for a DataFrame.

        Returns:
            Array of scores where lower = more anomalous.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        features = self._select_features(X)
        X_scaled = self._scaler.transform(features)
        return self._model.decision_function(X_scaled)

    def save(self, path: str | Path) -> Path:
        """Serialize model, scaler, and metadata to disk.

        Args:
            path: Directory to save model artifacts.

        Returns:
            Path to saved model directory.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "model.joblib"
        scaler_path = save_dir / "scaler.joblib"
        metadata_path = save_dir / "metadata.joblib"

        joblib.dump(self._model, model_path)
        joblib.dump(self._scaler, scaler_path)

        metadata = ModelMetadata(
            model_version=self._model_version,
            trained_at=pd.Timestamp.now(tz="UTC").isoformat(),
            n_estimators=self._n_estimators,
            max_samples=self._max_samples,
            contamination=self._contamination,
            feature_names=self._feature_names,
            training_samples=self._training_samples,
            random_state=self._random_state,
            checksum=self._compute_checksum(model_path),
        )
        joblib.dump(metadata, metadata_path)

        logger.info("Model saved to %s (version=%s)", save_dir, self._model_version)
        return save_dir

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        """Load a trained model from disk.

        Args:
            path: Directory containing model artifacts.

        Returns:
            AnomalyDetector instance ready for prediction.
        """
        load_dir = Path(path)
        model_path = load_dir / "model.joblib"
        scaler_path = load_dir / "scaler.joblib"
        metadata_path = load_dir / "metadata.joblib"

        for p in (model_path, scaler_path, metadata_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing model artifact: {p}")

        metadata: ModelMetadata = joblib.load(metadata_path)

        # Verify model integrity
        actual_checksum = cls._compute_checksum_static(model_path)
        if metadata.checksum and actual_checksum != metadata.checksum:
            raise ValueError(
                f"Model checksum mismatch: expected {metadata.checksum}, "
                f"got {actual_checksum}. Model may be corrupted."
            )

        detector = cls(
            n_estimators=metadata.n_estimators,
            max_samples=metadata.max_samples,
            contamination=metadata.contamination,
            random_state=metadata.random_state,
            feature_names=metadata.feature_names,
        )
        detector._model = joblib.load(model_path)
        detector._scaler = joblib.load(scaler_path)
        detector._is_fitted = True
        detector._model_version = metadata.model_version
        detector._training_samples = metadata.training_samples

        logger.info(
            "Model loaded from %s (version=%s, trained_samples=%d)",
            load_dir,
            metadata.model_version,
            metadata.training_samples,
        )
        return detector

    def tune_contamination(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        contamination_range: list[float] | None = None,
    ) -> float:
        """Find optimal contamination parameter using labeled data.

        Args:
            X: Feature data.
            y_true: Binary labels (1=fraud, 0=legitimate).
            contamination_range: List of contamination values to try.

        Returns:
            Best contamination value maximizing F1 score.
        """
        from sklearn.metrics import f1_score

        if contamination_range is None:
            contamination_range = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1]

        features = self._select_features(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        best_f1 = 0.0
        best_contamination = self._contamination

        for c in contamination_range:
            model = IsolationForest(
                n_estimators=self._n_estimators,
                max_samples=self._max_samples,
                contamination=c,
                random_state=self._random_state,
                max_features=self._max_features,
                n_jobs=-1,
            )
            model.fit(X_scaled)
            preds = model.predict(X_scaled)
            # IsolationForest: -1 = anomaly, 1 = normal
            pred_labels = (preds == -1).astype(int)
            f1 = f1_score(y_true, pred_labels, zero_division=0)

            logger.debug("contamination=%.4f, F1=%.4f", c, f1)
            if f1 > best_f1:
                best_f1 = f1
                best_contamination = c

        logger.info("Best contamination=%.4f (F1=%.4f)", best_contamination, best_f1)
        self._contamination = best_contamination
        return best_contamination

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select and validate features from input DataFrame."""
        available = [f for f in self._feature_names if f in X.columns]
        if not available:
            raise ValueError(
                f"No matching features found. Expected: {self._feature_names}, "
                f"Got columns: {list(X.columns)}"
            )

        missing = set(self._feature_names) - set(X.columns)
        if missing:
            logger.warning("Missing features (will use 0): %s", missing)

        result = pd.DataFrame(index=X.index)
        for feat in self._feature_names:
            if feat in X.columns:
                result[feat] = pd.to_numeric(X[feat], errors="coerce").fillna(0.0)
            else:
                result[feat] = 0.0

        return result

    def _extract_feature_vector(self, features: dict[str, Any]) -> np.ndarray:
        """Extract ordered feature vector from a dictionary."""
        vector = []
        for name in self._feature_names:
            val = features.get(name, 0.0)
            try:
                vector.append(float(val))
            except (TypeError, ValueError):
                vector.append(0.0)
        return np.array(vector, dtype=np.float64)

    def _compute_confidence(self, anomaly_score: float) -> float:
        """Map anomaly score to confidence level [0, 1].

        Scores near 0 are uncertain, scores far from 0 are confident.
        """
        # More negative = more confident anomaly
        # More positive = more confident normal
        abs_score = abs(anomaly_score)
        return float(np.clip(abs_score, 0.0, 1.0))

    def _identify_contributing_features(
        self, raw_vector: np.ndarray, scaled_vector: np.ndarray
    ) -> list[dict[str, Any]]:
        """Identify features contributing most to anomaly score.

        Uses absolute deviation from mean (in scaled space) as proxy.
        """
        deviations = np.abs(scaled_vector)
        top_indices = np.argsort(deviations)[::-1][:5]

        contributing = []
        for idx in top_indices:
            if deviations[idx] > 1.0:  # Only report if > 1 std dev
                contributing.append(
                    {
                        "feature": self._feature_names[idx],
                        "value": float(raw_vector[idx]),
                        "deviation_score": float(deviations[idx]),
                    }
                )
        return contributing

    def _compute_version(self) -> str:
        """Generate a version hash from model parameters and training time."""
        version_str = (
            f"{self._n_estimators}_{self._max_samples}_{self._contamination}_"
            f"{self._random_state}_{self._training_samples}_{pd.Timestamp.now().isoformat()}"
        )
        return hashlib.sha256(version_str.encode()).hexdigest()[:12]

    def _compute_checksum(self, path: Path) -> str:
        return self._compute_checksum_static(path)

    @staticmethod
    def _compute_checksum_static(path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]
