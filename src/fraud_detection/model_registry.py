"""Model registry for versioning, promotion, A/B testing, and serving.

Provides:
- Semantic versioning for model artifacts
- Model metadata storage (metrics, parameters, artifacts)
- Promotion workflow: staging → production
- Model rollback capability
- A/B testing with configurable traffic splitting
- Hot-reload model serving (thread-safe, zero-downtime)
- Prediction batching for throughput
- Fallback to previous model on failure
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """Model lifecycle stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelStatus(str, Enum):
    """Model operational status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    LOADING = "loading"


@dataclass
class ModelVersion:
    """Semantic version representation for models."""

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: ModelVersion) -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    @classmethod
    def parse(cls, version_str: str) -> ModelVersion:
        """Parse a semantic version string like '1.2.3'."""
        parts = version_str.strip().split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}. Expected 'major.minor.patch'")
        return cls(major=int(parts[0]), minor=int(parts[1]), patch=int(parts[2]))

    def bump_major(self) -> ModelVersion:
        return ModelVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> ModelVersion:
        return ModelVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> ModelVersion:
        return ModelVersion(self.major, self.minor, self.patch + 1)


@dataclass
class ModelMetadata:
    """Complete metadata for a registered model version."""

    name: str
    version: str
    stage: ModelStage
    status: ModelStatus
    artifact_path: str
    model_type: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    description: str = ""
    artifact_hash: str = ""
    feature_names: list[str] = field(default_factory=list)
    training_dataset: str = ""
    parent_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "stage": self.stage.value,
            "status": self.status.value,
            "artifact_path": self.artifact_path,
            "model_type": self.model_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "tags": self.tags,
            "description": self.description,
            "artifact_hash": self.artifact_hash,
            "feature_names": self.feature_names,
            "training_dataset": self.training_dataset,
            "parent_version": self.parent_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        return cls(
            name=data["name"],
            version=data["version"],
            stage=ModelStage(data["stage"]),
            status=ModelStatus(data["status"]),
            artifact_path=data["artifact_path"],
            model_type=data["model_type"],
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            metrics=data.get("metrics", {}),
            parameters=data.get("parameters", {}),
            tags=data.get("tags", {}),
            description=data.get("description", ""),
            artifact_hash=data.get("artifact_hash", ""),
            feature_names=data.get("feature_names", []),
            training_dataset=data.get("training_dataset", ""),
            parent_version=data.get("parent_version"),
        )


@dataclass
class ABTestConfig:
    """Configuration for A/B testing between model versions."""

    name: str
    model_a_version: str
    model_b_version: str
    traffic_split: float  # 0.0 to 1.0, fraction of traffic to model B
    sticky_assignment: bool = True  # consistent user assignment
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: str | None = None
    is_active: bool = True
    metrics_a: dict[str, float] = field(default_factory=dict)
    metrics_b: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_a_version": self.model_a_version,
            "model_b_version": self.model_b_version,
            "traffic_split": self.traffic_split,
            "sticky_assignment": self.sticky_assignment,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_active": self.is_active,
            "metrics_a": self.metrics_a,
            "metrics_b": self.metrics_b,
        }


class ModelRegistry:
    """Model registry for versioning, storage, and promotion workflows.

    Stores model metadata and artifacts on disk with JSON-based indexing.
    Supports promotion from staging to production, rollback, and A/B testing.
    """

    REGISTRY_FILE = "registry.json"

    def __init__(self, registry_path: str | Path) -> None:
        self._registry_path = Path(registry_path)
        self._registry_path.mkdir(parents=True, exist_ok=True)
        self._registry_file = self._registry_path / self.REGISTRY_FILE
        self._lock = threading.Lock()
        self._registry: dict[str, Any] = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        """Load registry state from disk."""
        if self._registry_file.exists():
            with open(self._registry_file, "r") as f:
                return cast(dict[str, Any], json.load(f))
        return {"models": {}, "ab_tests": {}, "promotion_history": []}

    def _save_registry(self) -> None:
        """Persist registry state to disk."""
        with open(self._registry_file, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def register_model(
        self,
        name: str,
        version: str,
        artifact_path: str | Path,
        model_type: str,
        metrics: dict[str, float] | None = None,
        parameters: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        description: str = "",
        feature_names: list[str] | None = None,
        training_dataset: str = "",
        parent_version: str | None = None,
    ) -> ModelMetadata:
        """Register a new model version in the registry.

        Args:
            name: Model name (e.g., 'risk_scorer', 'anomaly_detector').
            version: Semantic version string (e.g., '1.0.0').
            artifact_path: Path to model artifact directory.
            model_type: Model type identifier (e.g., 'xgboost', 'isolation_forest').
            metrics: Evaluation metrics (e.g., {'auc': 0.95, 'precision': 0.92}).
            parameters: Training hyperparameters.
            tags: Key-value tags for organization.
            description: Human-readable description of changes.
            feature_names: Ordered list of input feature names.
            training_dataset: Reference to training dataset used.
            parent_version: Previous version this was derived from.

        Returns:
            ModelMetadata for the registered model.

        Raises:
            ValueError: If version already exists for this model name.
        """
        ModelVersion.parse(version)  # validate format

        with self._lock:
            if name not in self._registry["models"]:
                self._registry["models"][name] = {}

            if version in self._registry["models"][name]:
                raise ValueError(
                    f"Model '{name}' version '{version}' already exists. "
                    "Use a new version number."
                )

            artifact_path = Path(artifact_path)
            artifact_hash = self._compute_artifact_hash(artifact_path)

            metadata = ModelMetadata(
                name=name,
                version=version,
                stage=ModelStage.DEVELOPMENT,
                status=ModelStatus.INACTIVE,
                artifact_path=str(artifact_path),
                model_type=model_type,
                metrics=metrics or {},
                parameters=parameters or {},
                tags=tags or {},
                description=description,
                artifact_hash=artifact_hash,
                feature_names=feature_names or [],
                training_dataset=training_dataset,
                parent_version=parent_version,
            )

            self._registry["models"][name][version] = metadata.to_dict()
            self._save_registry()

            logger.info(
                "Model registered: name=%s, version=%s, type=%s",
                name,
                version,
                model_type,
            )
            return metadata

    def get_model_metadata(self, name: str, version: str) -> ModelMetadata:
        """Retrieve metadata for a specific model version."""
        with self._lock:
            if name not in self._registry["models"]:
                raise KeyError(f"Model '{name}' not found in registry")
            if version not in self._registry["models"][name]:
                raise KeyError(f"Version '{version}' not found for model '{name}'")
            return ModelMetadata.from_dict(self._registry["models"][name][version])

    def list_models(
        self, name: str | None = None, stage: ModelStage | None = None
    ) -> list[ModelMetadata]:
        """List all registered models, optionally filtered by name or stage."""
        results: list[ModelMetadata] = []
        with self._lock:
            models = self._registry["models"]
            for model_name, versions in models.items():
                if name and model_name != name:
                    continue
                for _version, data in versions.items():
                    metadata = ModelMetadata.from_dict(data)
                    if stage and metadata.stage != stage:
                        continue
                    results.append(metadata)
        return results

    def get_production_model(self, name: str) -> ModelMetadata | None:
        """Get the current production model for a given name."""
        models = self.list_models(name=name, stage=ModelStage.PRODUCTION)
        if not models:
            return None
        # Return the latest by version
        models.sort(key=lambda m: ModelVersion.parse(m.version))
        return models[-1]

    def get_latest_version(self, name: str) -> ModelMetadata | None:
        """Get the latest version of a model regardless of stage."""
        models = self.list_models(name=name)
        if not models:
            return None
        models.sort(key=lambda m: ModelVersion.parse(m.version))
        return models[-1]

    def promote_model(
        self,
        name: str,
        version: str,
        target_stage: ModelStage,
        promoted_by: str = "system",
    ) -> ModelMetadata:
        """Promote a model to a new stage (e.g., staging → production).

        When promoting to production, the currently active production model
        is automatically archived.

        Args:
            name: Model name.
            version: Version to promote.
            target_stage: Target stage.
            promoted_by: Identifier of who triggered the promotion.

        Returns:
            Updated ModelMetadata.
        """
        with self._lock:
            if name not in self._registry["models"]:
                raise KeyError(f"Model '{name}' not found")
            if version not in self._registry["models"][name]:
                raise KeyError(f"Version '{version}' not found for model '{name}'")

            # If promoting to production, archive current production model
            if target_stage == ModelStage.PRODUCTION:
                for ver, data in self._registry["models"][name].items():
                    if data["stage"] == ModelStage.PRODUCTION.value and ver != version:
                        data["stage"] = ModelStage.ARCHIVED.value
                        data["updated_at"] = datetime.now(timezone.utc).isoformat()

            entry = self._registry["models"][name][version]
            previous_stage = entry["stage"]
            entry["stage"] = target_stage.value
            entry["status"] = ModelStatus.ACTIVE.value
            entry["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Record promotion history
            self._registry["promotion_history"].append(
                {
                    "name": name,
                    "version": version,
                    "from_stage": previous_stage,
                    "to_stage": target_stage.value,
                    "promoted_by": promoted_by,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            self._save_registry()

            logger.info(
                "Model promoted: name=%s, version=%s, %s → %s, by=%s",
                name,
                version,
                previous_stage,
                target_stage.value,
                promoted_by,
            )
            return ModelMetadata.from_dict(entry)

    def rollback_model(self, name: str, promoted_by: str = "system") -> ModelMetadata | None:
        """Rollback to the previous production model.

        Archives the current production model and restores the most recent
        archived version to production.

        Returns:
            The restored model metadata, or None if no archived version available.
        """
        with self._lock:
            if name not in self._registry["models"]:
                raise KeyError(f"Model '{name}' not found")

            # Find current production and most recent archived
            current_production: str | None = None
            archived_versions: list[tuple[str, str]] = []

            for ver, data in self._registry["models"][name].items():
                if data["stage"] == ModelStage.PRODUCTION.value:
                    current_production = ver
                elif data["stage"] == ModelStage.ARCHIVED.value:
                    archived_versions.append((ver, data.get("updated_at", "")))

            if not archived_versions:
                logger.warning("No archived model available for rollback: name=%s", name)
                return None

            # Sort by update time descending to find most recently archived
            archived_versions.sort(key=lambda x: x[1], reverse=True)
            restore_version = archived_versions[0][0]

            # Archive current production
            if current_production:
                self._registry["models"][name][current_production][
                    "stage"
                ] = ModelStage.ARCHIVED.value
                self._registry["models"][name][current_production]["updated_at"] = datetime.now(
                    timezone.utc
                ).isoformat()

            # Restore archived to production
            entry = self._registry["models"][name][restore_version]
            entry["stage"] = ModelStage.PRODUCTION.value
            entry["status"] = ModelStatus.ACTIVE.value
            entry["updated_at"] = datetime.now(timezone.utc).isoformat()

            self._registry["promotion_history"].append(
                {
                    "name": name,
                    "version": restore_version,
                    "from_stage": ModelStage.ARCHIVED.value,
                    "to_stage": ModelStage.PRODUCTION.value,
                    "promoted_by": promoted_by,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": "rollback",
                }
            )

            self._save_registry()

            logger.info(
                "Model rolled back: name=%s, restored=%s, archived=%s",
                name,
                restore_version,
                current_production,
            )
            return ModelMetadata.from_dict(entry)

    def create_ab_test(
        self,
        test_name: str,
        model_name: str,
        version_a: str,
        version_b: str,
        traffic_split: float = 0.5,
        sticky_assignment: bool = True,
    ) -> ABTestConfig:
        """Create an A/B test between two model versions.

        Args:
            test_name: Unique name for the test.
            model_name: The model being tested.
            version_a: Control model version.
            version_b: Treatment model version.
            traffic_split: Fraction of traffic to route to model B (0.0 to 1.0).
            sticky_assignment: Whether to use consistent user-based assignment.

        Returns:
            ABTestConfig for the created test.
        """
        if not 0.0 <= traffic_split <= 1.0:
            raise ValueError("traffic_split must be between 0.0 and 1.0")

        with self._lock:
            # Validate both versions exist
            if model_name not in self._registry["models"]:
                raise KeyError(f"Model '{model_name}' not found")
            if version_a not in self._registry["models"][model_name]:
                raise KeyError(f"Version '{version_a}' not found for model '{model_name}'")
            if version_b not in self._registry["models"][model_name]:
                raise KeyError(f"Version '{version_b}' not found for model '{model_name}'")

            config = ABTestConfig(
                name=test_name,
                model_a_version=version_a,
                model_b_version=version_b,
                traffic_split=traffic_split,
                sticky_assignment=sticky_assignment,
            )

            self._registry["ab_tests"][test_name] = config.to_dict()
            self._save_registry()

            logger.info(
                "A/B test created: name=%s, A=%s, B=%s, split=%.2f",
                test_name,
                version_a,
                version_b,
                traffic_split,
            )
            return config

    def get_ab_test(self, test_name: str) -> ABTestConfig | None:
        """Get A/B test configuration by name."""
        with self._lock:
            data = self._registry["ab_tests"].get(test_name)
            if data is None:
                return None
            return ABTestConfig(**data)

    def end_ab_test(self, test_name: str) -> ABTestConfig | None:
        """End an active A/B test."""
        with self._lock:
            if test_name not in self._registry["ab_tests"]:
                return None
            self._registry["ab_tests"][test_name]["is_active"] = False
            self._registry["ab_tests"][test_name]["end_time"] = datetime.now(
                timezone.utc
            ).isoformat()
            self._save_registry()
            return ABTestConfig(**self._registry["ab_tests"][test_name])

    def resolve_ab_assignment(self, test_name: str, user_id: str) -> str:
        """Determine which model version a user should receive.

        Uses deterministic hashing for sticky assignment so the same user
        always gets the same model variant within a test.

        Args:
            test_name: A/B test name.
            user_id: User/customer identifier.

        Returns:
            Model version string to use for this user.
        """
        config = self.get_ab_test(test_name)
        if config is None or not config.is_active:
            raise ValueError(f"A/B test '{test_name}' not found or inactive")

        if config.sticky_assignment:
            # Deterministic hash for consistent assignment
            hash_input = f"{test_name}:{user_id}".encode()
            hash_val = int(hashlib.sha256(hash_input).hexdigest(), 16)
            # Map to [0, 1) range
            normalized = (hash_val % 10000) / 10000.0
        else:
            import random

            normalized = random.random()

        if normalized < config.traffic_split:
            return config.model_b_version
        return config.model_a_version

    def _compute_artifact_hash(self, artifact_path: Path) -> str:
        """Compute SHA-256 hash of model artifact directory for integrity."""
        if not artifact_path.exists():
            return ""
        hasher = hashlib.sha256()
        for file_path in sorted(artifact_path.rglob("*")):
            if file_path.is_file():
                hasher.update(file_path.name.encode())
                hasher.update(str(file_path.stat().st_size).encode())
        return hasher.hexdigest()[:16]

    def get_promotion_history(self, name: str | None = None) -> list[dict[str, Any]]:
        """Get promotion history, optionally filtered by model name."""
        with self._lock:
            history = self._registry.get("promotion_history", [])
            if name:
                history = [h for h in history if h["name"] == name]
            return cast(list[dict[str, Any]], history)


class ModelServer:
    """Production model server with hot-reload, A/B testing, and fallback.

    Thread-safe model serving with zero-downtime model swaps. Supports
    batched prediction for throughput and automatic fallback to previous
    model on load failure.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        model_name: str,
        model_loader: Any = None,
        batch_size: int = 32,
        batch_timeout_ms: float = 10.0,
    ) -> None:
        self._registry = registry
        self._model_name = model_name
        self._batch_size = batch_size
        self._batch_timeout_ms = batch_timeout_ms

        # Thread-safe model references
        self._lock = threading.RLock()
        self._primary_model: Any | None = None
        self._primary_metadata: ModelMetadata | None = None
        self._fallback_model: Any | None = None
        self._fallback_metadata: ModelMetadata | None = None
        self._ab_models: dict[str, Any] = {}

        # Model loader function: (artifact_path) -> loaded model
        self._model_loader = model_loader or self._default_model_loader

        # Serving stats
        self._prediction_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0
        self._last_reload_time: str | None = None

        # Batch queue
        self._batch_queue: list[dict[str, Any]] = []
        self._batch_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        """Check if server has a loaded model ready to serve."""
        return self._primary_model is not None

    @property
    def active_version(self) -> str:
        """Get the currently active model version."""
        if self._primary_metadata:
            return self._primary_metadata.version
        return "none"

    @property
    def stats(self) -> dict[str, Any]:
        """Get serving statistics."""
        avg_latency = (
            self._total_latency_ms / self._prediction_count if self._prediction_count > 0 else 0.0
        )
        return {
            "model_name": self._model_name,
            "active_version": self.active_version,
            "is_ready": self.is_ready,
            "prediction_count": self._prediction_count,
            "error_count": self._error_count,
            "avg_latency_ms": round(avg_latency, 3),
            "last_reload_time": self._last_reload_time,
            "has_fallback": self._fallback_model is not None,
        }

    def load_production_model(self) -> bool:
        """Load the current production model from registry.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        metadata = self._registry.get_production_model(self._model_name)
        if metadata is None:
            logger.warning("No production model found for: %s", self._model_name)
            return False
        return self._load_model_version(metadata)

    def load_model_version(self, version: str) -> bool:
        """Load a specific model version.

        Args:
            version: Semantic version string to load.

        Returns:
            True if loaded successfully.
        """
        metadata = self._registry.get_model_metadata(self._model_name, version)
        return self._load_model_version(metadata)

    def hot_reload(self) -> bool:
        """Hot-reload the model: check registry for updates and swap if newer.

        Thread-safe: uses lock to swap model reference atomically.
        Current model continues serving during load of new model.

        Returns:
            True if a new model was loaded, False if already up to date.
        """
        current_version = self.active_version
        production_meta = self._registry.get_production_model(self._model_name)

        if production_meta is None:
            return False

        if production_meta.version == current_version:
            return False

        logger.info(
            "Hot-reload triggered: %s → %s",
            current_version,
            production_meta.version,
        )

        # Load new model in background (without lock)
        new_model = self._safe_load_artifact(production_meta.artifact_path)
        if new_model is None:
            logger.error(
                "Hot-reload failed for version %s, keeping current model",
                production_meta.version,
            )
            return False

        # Atomic swap under lock
        with self._lock:
            self._fallback_model = self._primary_model
            self._fallback_metadata = self._primary_metadata
            self._primary_model = new_model
            self._primary_metadata = production_meta
            self._last_reload_time = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Hot-reload complete: now serving version %s",
            production_meta.version,
        )
        return True

    def predict(self, features: np.ndarray, user_id: str | None = None) -> np.ndarray:
        """Make a prediction using the active model.

        If an A/B test is active and user_id is provided, routes to the
        appropriate model variant. Falls back to previous model on failure.

        Args:
            features: Feature array of shape (n_samples, n_features).
            user_id: Optional user ID for A/B test routing.

        Returns:
            Prediction array.

        Raises:
            RuntimeError: If no model is loaded and no fallback available.
        """
        start = time.perf_counter()

        model = self._resolve_model(user_id)
        if model is None:
            raise RuntimeError(
                f"No model available for '{self._model_name}'. "
                "Load a model with load_production_model() first."
            )

        try:
            if hasattr(model, "predict_proba"):
                predictions = model.predict_proba(features)[:, 1]
            elif hasattr(model, "decision_function"):
                predictions = model.decision_function(features)
            else:
                predictions = model.predict(features)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._prediction_count += len(features) if features.ndim > 1 else 1
            self._total_latency_ms += elapsed_ms

            return np.asarray(predictions)

        except Exception as exc:
            self._error_count += 1
            logger.error(
                "Prediction failed with primary model (v%s): %s",
                self.active_version,
                exc,
            )
            # Attempt fallback
            return self._predict_with_fallback(features)

    def predict_batch(
        self, batch: list[np.ndarray], user_id: str | None = None
    ) -> list[np.ndarray]:
        """Batch multiple prediction requests for efficiency.

        Concatenates inputs, runs single model inference, then splits results.

        Args:
            batch: List of feature arrays.
            user_id: Optional user ID for A/B routing.

        Returns:
            List of prediction arrays corresponding to inputs.
        """
        if not batch:
            return []

        # Track sizes for splitting results
        sizes = [arr.shape[0] if arr.ndim > 1 else 1 for arr in batch]

        # Concatenate for single inference call
        if batch[0].ndim == 1:
            combined = np.vstack([arr.reshape(1, -1) for arr in batch])
        else:
            combined = np.vstack(batch)

        predictions = self.predict(combined, user_id=user_id)

        # Split back
        results: list[np.ndarray] = []
        offset = 0
        for size in sizes:
            results.append(predictions[offset : offset + size])
            offset += size

        return results

    def _resolve_model(self, user_id: str | None) -> Any | None:
        """Resolve which model to use, considering A/B tests."""
        if user_id:
            # Check for active A/B tests
            with self._lock:
                for test_name, test_data in self._registry._registry.get("ab_tests", {}).items():
                    if not test_data.get("is_active", False):
                        continue
                    version = self._registry.resolve_ab_assignment(test_name, user_id)
                    if version in self._ab_models:
                        return self._ab_models[version]
                    # Load if not cached
                    ab_model = self._load_ab_model(version)
                    if ab_model is not None:
                        return ab_model

        with self._lock:
            return self._primary_model

    def _load_ab_model(self, version: str) -> Any | None:
        """Load and cache a model for A/B testing."""
        try:
            metadata = self._registry.get_model_metadata(self._model_name, version)
            model = self._safe_load_artifact(metadata.artifact_path)
            if model is not None:
                self._ab_models[version] = model
            return model
        except (KeyError, Exception) as exc:
            logger.warning("Failed to load A/B model version %s: %s", version, exc)
            return None

    def _predict_with_fallback(self, features: np.ndarray) -> np.ndarray:
        """Attempt prediction with fallback model."""
        with self._lock:
            fallback = self._fallback_model

        if fallback is None:
            raise RuntimeError(
                f"Primary model failed and no fallback available for '{self._model_name}'"
            )

        logger.warning(
            "Using fallback model (v%s)",
            self._fallback_metadata.version if self._fallback_metadata else "unknown",
        )

        try:
            if hasattr(fallback, "predict_proba"):
                return cast(np.ndarray, fallback.predict_proba(features)[:, 1])
            elif hasattr(fallback, "decision_function"):
                return cast(np.ndarray, fallback.decision_function(features))
            else:
                return cast(np.ndarray, fallback.predict(features))
        except Exception as exc:
            raise RuntimeError(
                f"Both primary and fallback models failed for '{self._model_name}': {exc}"
            ) from exc

    def _load_model_version(self, metadata: ModelMetadata) -> bool:
        """Load a model version and set as primary."""
        model = self._safe_load_artifact(metadata.artifact_path)
        if model is None:
            return False

        with self._lock:
            self._fallback_model = self._primary_model
            self._fallback_metadata = self._primary_metadata
            self._primary_model = model
            self._primary_metadata = metadata
            self._last_reload_time = datetime.now(timezone.utc).isoformat()

        logger.info("Model loaded: %s v%s", self._model_name, metadata.version)
        return True

    def _safe_load_artifact(self, artifact_path: str) -> Any | None:
        """Safely load a model artifact, returning None on failure."""
        try:
            return self._model_loader(artifact_path)
        except Exception as exc:
            logger.error("Failed to load model artifact from %s: %s", artifact_path, exc)
            return None

    @staticmethod
    def _default_model_loader(artifact_path: str) -> Any:
        """Default model loader using joblib."""
        path = Path(artifact_path)
        model_file = path / "model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        return joblib.load(model_file)

    def load_for_ab_test(self, test_name: str) -> bool:
        """Pre-load models required for an A/B test.

        Args:
            test_name: Name of the A/B test to load models for.

        Returns:
            True if both models loaded successfully.
        """
        config = self._registry.get_ab_test(test_name)
        if config is None:
            logger.warning("A/B test '%s' not found", test_name)
            return False

        success = True
        for version in [config.model_a_version, config.model_b_version]:
            if version not in self._ab_models:
                model = self._load_ab_model(version)
                if model is None:
                    success = False
        return success
