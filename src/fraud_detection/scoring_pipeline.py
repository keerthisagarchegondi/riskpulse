"""Unified Fraud Scoring Pipeline — Ensemble scoring orchestrator.

Combines rule-based, anomaly detection, and ML scoring into a unified
risk assessment with configurable ensemble weights. Supports parallel
execution, score caching, and risk classification.

Latency budget: < 100ms end-to-end for all three methods.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

from src.fraud_detection.anomaly_detector import AnomalyDetector, AnomalyResult
from src.fraud_detection.risk_scorer import RiskScore, RiskScorer
from src.fraud_detection.rule_engine import FraudRuleEngine, RuleEvaluationResult
from src.utils.logger import get_logger

logger = get_logger(__name__, component="scoring_pipeline")

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_DEFAULT_WEIGHTS_PATH = _CONFIG_DIR / "scoring_weights.yaml"


class RiskClassification(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ScoringMethodResult:
    """Result from a single scoring method."""

    method: str
    raw_score: float
    normalized_score: float  # 0.0 to 1.0
    weight: float
    weighted_score: float  # normalized_score * weight
    latency_ms: float
    success: bool
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedScore:
    """Final unified scoring result combining all methods."""

    transaction_id: str
    final_score: float  # 0.0 to 1.0
    risk_classification: RiskClassification
    method_scores: list[ScoringMethodResult] = field(default_factory=list)
    ensemble_weights_used: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    methods_succeeded: int = 0
    methods_failed: int = 0
    alert_recommended: bool = False
    auto_block_recommended: bool = False
    cached: bool = False
    scoring_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "final_score": round(self.final_score, 6),
            "risk_classification": self.risk_classification.value,
            "method_scores": [
                {
                    "method": ms.method,
                    "raw_score": round(ms.raw_score, 6),
                    "normalized_score": round(ms.normalized_score, 6),
                    "weight": ms.weight,
                    "weighted_score": round(ms.weighted_score, 6),
                    "latency_ms": round(ms.latency_ms, 3),
                    "success": ms.success,
                    "error": ms.error,
                }
                for ms in self.method_scores
            ],
            "ensemble_weights_used": self.ensemble_weights_used,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "methods_succeeded": self.methods_succeeded,
            "methods_failed": self.methods_failed,
            "alert_recommended": self.alert_recommended,
            "auto_block_recommended": self.auto_block_recommended,
            "cached": self.cached,
            "scoring_version": self.scoring_version,
        }


class _LRUCache:
    """Thread-safe LRU cache for scored transactions."""

    def __init__(self, max_entries: int = 100000, ttl_seconds: int = 300) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, UnifiedScore]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> UnifiedScore | None:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            timestamp, score = self._cache[key]
            if (time.time() - timestamp) > self._ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            cached_score = score
            cached_score.cached = True
            return cached_score

    def put(self, key: str, score: UnifiedScore) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = (time.time(), score)
            else:
                if len(self._cache) >= self._max_entries:
                    self._cache.popitem(last=False)
                self._cache[key] = (time.time(), score)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


class ScoringPipeline:
    """Unified fraud scoring pipeline with ensemble scoring.

    Orchestrates parallel execution of rule-based, anomaly detection,
    and ML scoring methods, then combines them using configurable
    ensemble weights.

    Usage::

        pipeline = ScoringPipeline()
        score = await pipeline.score_transaction(transaction_data)
        batch_scores = await pipeline.score_batch(transactions)
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        rule_engine: FraudRuleEngine | None = None,
        anomaly_detector: AnomalyDetector | None = None,
        risk_scorer: RiskScorer | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._config = self._load_config(config_path)
        scoring_cfg = self._config.get("scoring", {})

        # Ensemble weights
        configured_weights = scoring_cfg.get("weights", {})
        self._weights = weights or {
            "rule_score": configured_weights.get("rule_score", 0.3),
            "anomaly_score": configured_weights.get("anomaly_score", 0.3),
            "ml_score": configured_weights.get("ml_score", 0.4),
        }
        self._validate_weights()

        # Thresholds for risk classification
        thresholds_cfg = scoring_cfg.get("thresholds", {})
        self._thresholds = {
            RiskClassification.LOW: thresholds_cfg.get("low", 0.0),
            RiskClassification.MEDIUM: thresholds_cfg.get("medium", 0.3),
            RiskClassification.HIGH: thresholds_cfg.get("high", 0.6),
            RiskClassification.CRITICAL: thresholds_cfg.get("critical", 0.85),
        }

        # Alert thresholds
        alerts_cfg = scoring_cfg.get("alerts", {})
        self._alert_threshold = alerts_cfg.get("generate_alert_above", 0.6)
        self._auto_block_threshold = alerts_cfg.get("auto_block_above", 0.9)

        # Pipeline config
        pipeline_cfg = scoring_cfg.get("pipeline", {})
        self._timeout_ms = pipeline_cfg.get("timeout_ms", 100)
        self._fallback_on_timeout = pipeline_cfg.get("fallback_on_timeout", True)
        self._min_methods_required = pipeline_cfg.get("min_methods_required", 2)

        # Method configs
        methods_cfg = scoring_cfg.get("methods", {})
        self._rule_cfg = methods_cfg.get("rule_engine", {})
        self._anomaly_cfg = methods_cfg.get("anomaly_detector", {})
        self._ml_cfg = methods_cfg.get("ml_model", {})

        # Cache
        cache_cfg = scoring_cfg.get("cache", {})
        cache_enabled = cache_cfg.get("enabled", True)
        self._cache: _LRUCache | None = None
        if cache_enabled:
            self._cache = _LRUCache(
                max_entries=cache_cfg.get("max_entries", 100000),
                ttl_seconds=cache_cfg.get("ttl_seconds", 300),
            )

        # Scoring methods (injected or instantiated)
        self._rule_engine = rule_engine
        self._anomaly_detector = anomaly_detector
        self._risk_scorer = risk_scorer

        # Metrics
        self._total_scored = 0
        self._total_latency_ms = 0.0
        self._classification_counts: dict[str, int] = {c.value: 0 for c in RiskClassification}
        self._metrics_lock = Lock()

        logger.info(
            "scoring_pipeline_initialized",
            weights=self._weights,
            thresholds={k.value: v for k, v in self._thresholds.items()},
            cache_enabled=cache_enabled,
        )

    def _load_config(self, config_path: str | Path | None) -> dict[str, Any]:
        path = Path(config_path) if config_path else _DEFAULT_WEIGHTS_PATH
        if not path.exists():
            logger.warning("scoring_config_not_found", path=str(path))
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _validate_weights(self) -> None:
        total = sum(self._weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total}: {self._weights}")
        for name, w in self._weights.items():
            if w < 0.0 or w > 1.0:
                raise ValueError(f"Weight '{name}' must be between 0 and 1, got {w}")

    @property
    def weights(self) -> dict[str, float]:
        return self._weights.copy()

    @property
    def thresholds(self) -> dict[str, float]:
        return {k.value: v for k, v in self._thresholds.items()}

    @property
    def metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            avg_latency = (
                self._total_latency_ms / self._total_scored if self._total_scored > 0 else 0.0
            )
            return {
                "total_scored": self._total_scored,
                "avg_latency_ms": round(avg_latency, 3),
                "classification_distribution": dict(self._classification_counts),
                "cache_stats": self._cache.stats if self._cache else None,
            }

    def update_weights(self, weights: dict[str, float]) -> None:
        """Dynamically update ensemble weights."""
        old_weights = self._weights.copy()
        self._weights = weights
        self._validate_weights()
        logger.info("weights_updated", old=old_weights, new=weights)

    # ── Scoring Methods ──────────────────────────────────────────────

    def _score_rules(
        self, transaction: dict[str, Any], context: dict[str, Any] | None = None
    ) -> ScoringMethodResult:
        """Execute rule-based scoring."""
        start = time.perf_counter()
        try:
            if self._rule_engine is None:
                return ScoringMethodResult(
                    method="rule_engine",
                    raw_score=0.0,
                    normalized_score=0.0,
                    weight=self._weights["rule_score"],
                    weighted_score=0.0,
                    latency_ms=0.0,
                    success=False,
                    error="Rule engine not initialized",
                )

            result: RuleEvaluationResult = self._rule_engine.evaluate(transaction, context)

            # Normalize rule score to [0, 1]
            # Use the rule_score already computed by the engine
            normalized = min(1.0, max(0.0, result.rule_score))

            latency_ms = (time.perf_counter() - start) * 1000
            weight = self._weights["rule_score"]

            return ScoringMethodResult(
                method="rule_engine",
                raw_score=result.rule_score,
                normalized_score=normalized,
                weight=weight,
                weighted_score=normalized * weight,
                latency_ms=latency_ms,
                success=True,
                details={
                    "triggered_rules": result.triggered_count,
                    "combined_severity": result.combined_severity,
                    "rules_evaluated": result.total_rules_evaluated,
                },
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("rule_scoring_failed", error=str(exc))
            return ScoringMethodResult(
                method="rule_engine",
                raw_score=0.0,
                normalized_score=0.0,
                weight=self._weights["rule_score"],
                weighted_score=0.0,
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
            )

    def _score_anomaly(self, transaction: dict[str, Any]) -> ScoringMethodResult:
        """Execute anomaly detection scoring."""
        start = time.perf_counter()
        try:
            if self._anomaly_detector is None:
                return ScoringMethodResult(
                    method="anomaly_detector",
                    raw_score=0.0,
                    normalized_score=0.0,
                    weight=self._weights["anomaly_score"],
                    weighted_score=0.0,
                    latency_ms=0.0,
                    success=False,
                    error="Anomaly detector not initialized",
                )

            result: AnomalyResult = self._anomaly_detector.predict(transaction)

            # Normalize Isolation Forest score to [0, 1]
            # IF score: negative = anomalous, positive = normal
            # Map: -1 → 1.0 (most risky), +1 → 0.0 (least risky)
            raw = result.anomaly_score
            if self._anomaly_cfg.get("score_inversion", True):
                normalized = max(0.0, min(1.0, (1.0 - raw) / 2.0))
            else:
                normalized = max(0.0, min(1.0, raw))

            latency_ms = (time.perf_counter() - start) * 1000
            weight = self._weights["anomaly_score"]

            return ScoringMethodResult(
                method="anomaly_detector",
                raw_score=raw,
                normalized_score=normalized,
                weight=weight,
                weighted_score=normalized * weight,
                latency_ms=latency_ms,
                success=True,
                details={
                    "is_anomaly": result.is_anomaly,
                    "confidence": result.confidence,
                    "model_version": result.model_version,
                },
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("anomaly_scoring_failed", error=str(exc))
            return ScoringMethodResult(
                method="anomaly_detector",
                raw_score=0.0,
                normalized_score=0.0,
                weight=self._weights["anomaly_score"],
                weighted_score=0.0,
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
            )

    def _score_ml(self, transaction: dict[str, Any]) -> ScoringMethodResult:
        """Execute ML model scoring."""
        start = time.perf_counter()
        try:
            if self._risk_scorer is None:
                return ScoringMethodResult(
                    method="ml_model",
                    raw_score=0.0,
                    normalized_score=0.0,
                    weight=self._weights["ml_score"],
                    weighted_score=0.0,
                    latency_ms=0.0,
                    success=False,
                    error="ML risk scorer not initialized",
                )

            result: RiskScore = self._risk_scorer.predict(transaction)

            # ML model output is already a calibrated probability [0, 1]
            normalized = max(0.0, min(1.0, result.risk_score))

            # Apply confidence penalty if below threshold
            weight = self._weights["ml_score"]
            min_confidence = self._ml_cfg.get("min_confidence", 0.5)
            if result.confidence < min_confidence:
                penalty = self._ml_cfg.get("confidence_penalty", 0.5)
                weight *= penalty

            latency_ms = (time.perf_counter() - start) * 1000

            return ScoringMethodResult(
                method="ml_model",
                raw_score=result.raw_score,
                normalized_score=normalized,
                weight=weight,
                weighted_score=normalized * weight,
                latency_ms=latency_ms,
                success=True,
                details={
                    "risk_level": result.risk_level,
                    "confidence": result.confidence,
                    "model_version": result.model_version,
                    "top_features": result.top_features[:3],
                },
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("ml_scoring_failed", error=str(exc))
            return ScoringMethodResult(
                method="ml_model",
                raw_score=0.0,
                normalized_score=0.0,
                weight=self._weights["ml_score"],
                weighted_score=0.0,
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
            )

    # ── Ensemble Scoring ─────────────────────────────────────────────

    def _compute_ensemble_score(self, method_results: list[ScoringMethodResult]) -> float:
        """Compute weighted ensemble score from individual method results.

        If some methods fail, re-normalizes weights across successful methods.
        """
        successful = [r for r in method_results if r.success]
        if not successful:
            return 0.0

        # Re-normalize weights for successful methods only
        total_weight = sum(r.weight for r in successful)
        if total_weight <= 0:
            return 0.0

        ensemble_score = sum(r.normalized_score * (r.weight / total_weight) for r in successful)
        return max(0.0, min(1.0, ensemble_score))

    def _classify_risk(self, score: float) -> RiskClassification:
        """Classify a score into a risk level based on thresholds."""
        if score >= self._thresholds[RiskClassification.CRITICAL]:
            return RiskClassification.CRITICAL
        if score >= self._thresholds[RiskClassification.HIGH]:
            return RiskClassification.HIGH
        if score >= self._thresholds[RiskClassification.MEDIUM]:
            return RiskClassification.MEDIUM
        return RiskClassification.LOW

    def _generate_cache_key(self, transaction: dict[str, Any]) -> str:
        """Generate a deterministic cache key for a transaction."""
        txn_id = transaction.get("external_transaction_id") or transaction.get("transaction_id", "")
        amount = str(transaction.get("transaction_amount", ""))
        ts = str(transaction.get("transaction_timestamp", ""))
        raw = f"{txn_id}:{amount}:{ts}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    # ── Public Synchronous API ───────────────────────────────────────

    def score_transaction_sync(
        self,
        transaction: dict[str, Any],
        context: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> UnifiedScore:
        """Score a single transaction synchronously.

        Args:
            transaction: Transaction data dict.
            context: Optional enrichment context for rule engine.
            use_cache: Whether to check/populate cache.

        Returns:
            UnifiedScore with final ensemble score and classification.
        """
        txn_id = transaction.get("external_transaction_id") or transaction.get(
            "transaction_id", "unknown"
        )
        start = time.perf_counter()

        # Check cache
        if use_cache and self._cache is not None:
            cache_key = self._generate_cache_key(transaction)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Execute all three scoring methods
        rule_result = self._score_rules(transaction, context)
        anomaly_result = self._score_anomaly(transaction)
        ml_result = self._score_ml(transaction)

        method_results = [rule_result, anomaly_result, ml_result]

        # Check minimum methods succeeded
        succeeded = sum(1 for r in method_results if r.success)
        failed = sum(1 for r in method_results if not r.success)

        if succeeded < self._min_methods_required and not self._fallback_on_timeout:
            logger.warning(
                "insufficient_scoring_methods",
                transaction_id=txn_id,
                succeeded=succeeded,
                required=self._min_methods_required,
            )

        # Compute ensemble score
        final_score = self._compute_ensemble_score(method_results)
        classification = self._classify_risk(final_score)

        total_latency = (time.perf_counter() - start) * 1000

        # Determine alert/block recommendations
        alert_recommended = final_score >= self._alert_threshold
        auto_block_recommended = final_score >= self._auto_block_threshold

        unified = UnifiedScore(
            transaction_id=txn_id,
            final_score=final_score,
            risk_classification=classification,
            method_scores=method_results,
            ensemble_weights_used=self._weights.copy(),
            total_latency_ms=total_latency,
            methods_succeeded=succeeded,
            methods_failed=failed,
            alert_recommended=alert_recommended,
            auto_block_recommended=auto_block_recommended,
        )

        # Cache the result
        if use_cache and self._cache is not None:
            cache_key = self._generate_cache_key(transaction)
            self._cache.put(cache_key, unified)

        # Update metrics
        with self._metrics_lock:
            self._total_scored += 1
            self._total_latency_ms += total_latency
            self._classification_counts[classification.value] += 1

        logger.info(
            "transaction_scored",
            transaction_id=txn_id,
            final_score=round(final_score, 4),
            classification=classification.value,
            methods_succeeded=succeeded,
            latency_ms=round(total_latency, 2),
        )

        return unified

    # ── Public Async API ─────────────────────────────────────────────

    async def score_transaction(
        self,
        transaction: dict[str, Any],
        context: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> UnifiedScore:
        """Score a single transaction with parallel method execution.

        Args:
            transaction: Transaction data dict.
            context: Optional enrichment context for rule engine.
            use_cache: Whether to check/populate cache.

        Returns:
            UnifiedScore with final ensemble score and classification.
        """
        txn_id = transaction.get("external_transaction_id") or transaction.get(
            "transaction_id", "unknown"
        )
        start = time.perf_counter()

        # Check cache
        if use_cache and self._cache is not None:
            cache_key = self._generate_cache_key(transaction)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Execute scoring methods in parallel using asyncio
        loop = asyncio.get_event_loop()
        rule_task = loop.run_in_executor(None, self._score_rules, transaction, context)
        anomaly_task = loop.run_in_executor(None, self._score_anomaly, transaction)
        ml_task = loop.run_in_executor(None, self._score_ml, transaction)

        # Apply timeout
        timeout_s = self._timeout_ms / 1000.0
        try:
            results = await asyncio.wait_for(
                asyncio.gather(rule_task, anomaly_task, ml_task, return_exceptions=True),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("scoring_timeout", transaction_id=txn_id)
            # Collect whatever completed
            results = []
            for task in [rule_task, anomaly_task, ml_task]:
                if task.done():
                    results.append(task.result())

        method_results: list[ScoringMethodResult] = []
        for r in results:
            if isinstance(r, ScoringMethodResult):
                method_results.append(r)
            elif isinstance(r, Exception):
                logger.error("scoring_method_exception", error=str(r))

        # Ensure we have entries for all methods (mark missing as failed)
        method_names_present = {mr.method for mr in method_results}
        if "rule_engine" not in method_names_present:
            method_results.append(
                ScoringMethodResult(
                    method="rule_engine",
                    raw_score=0.0,
                    normalized_score=0.0,
                    weight=self._weights["rule_score"],
                    weighted_score=0.0,
                    latency_ms=0.0,
                    success=False,
                    error="Timed out",
                )
            )
        if "anomaly_detector" not in method_names_present:
            method_results.append(
                ScoringMethodResult(
                    method="anomaly_detector",
                    raw_score=0.0,
                    normalized_score=0.0,
                    weight=self._weights["anomaly_score"],
                    weighted_score=0.0,
                    latency_ms=0.0,
                    success=False,
                    error="Timed out",
                )
            )
        if "ml_model" not in method_names_present:
            method_results.append(
                ScoringMethodResult(
                    method="ml_model",
                    raw_score=0.0,
                    normalized_score=0.0,
                    weight=self._weights["ml_score"],
                    weighted_score=0.0,
                    latency_ms=0.0,
                    success=False,
                    error="Timed out",
                )
            )

        succeeded = sum(1 for r in method_results if r.success)
        failed = sum(1 for r in method_results if not r.success)

        # Compute ensemble score
        final_score = self._compute_ensemble_score(method_results)
        classification = self._classify_risk(final_score)

        total_latency = (time.perf_counter() - start) * 1000

        alert_recommended = final_score >= self._alert_threshold
        auto_block_recommended = final_score >= self._auto_block_threshold

        unified = UnifiedScore(
            transaction_id=txn_id,
            final_score=final_score,
            risk_classification=classification,
            method_scores=method_results,
            ensemble_weights_used=self._weights.copy(),
            total_latency_ms=total_latency,
            methods_succeeded=succeeded,
            methods_failed=failed,
            alert_recommended=alert_recommended,
            auto_block_recommended=auto_block_recommended,
        )

        # Cache the result
        if use_cache and self._cache is not None:
            cache_key = self._generate_cache_key(transaction)
            self._cache.put(cache_key, unified)

        # Update metrics
        with self._metrics_lock:
            self._total_scored += 1
            self._total_latency_ms += total_latency
            self._classification_counts[classification.value] += 1

        logger.info(
            "transaction_scored",
            transaction_id=txn_id,
            final_score=round(final_score, 4),
            classification=classification.value,
            methods_succeeded=succeeded,
            latency_ms=round(total_latency, 2),
        )

        return unified

    async def score_batch(
        self,
        transactions: list[dict[str, Any]],
        contexts: list[dict[str, Any] | None] | None = None,
        use_cache: bool = True,
    ) -> list[UnifiedScore]:
        """Score a batch of transactions.

        Args:
            transactions: List of transaction data dicts.
            contexts: Optional list of contexts (one per transaction).
            use_cache: Whether to check/populate cache.

        Returns:
            List of UnifiedScore results.
        """
        if not transactions:
            return []

        batch_cfg = self._config.get("scoring", {}).get("batch", {})
        max_batch = batch_cfg.get("max_batch_size", 1000)
        chunk_size = batch_cfg.get("chunk_size", 100)

        if len(transactions) > max_batch:
            raise ValueError(f"Batch size {len(transactions)} exceeds maximum {max_batch}")

        if contexts is None:
            contexts = [None] * len(transactions)

        results: list[UnifiedScore] = []
        for i in range(0, len(transactions), chunk_size):
            chunk_txns = transactions[i : i + chunk_size]
            chunk_ctxs = contexts[i : i + chunk_size]

            tasks = [
                self.score_transaction(txn, ctx, use_cache=use_cache)
                for txn, ctx in zip(chunk_txns, chunk_ctxs)
            ]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in chunk_results:
                if isinstance(r, UnifiedScore):
                    results.append(r)
                elif isinstance(r, Exception):
                    logger.error("batch_scoring_error", error=str(r))

        return results

    def score_batch_sync(
        self,
        transactions: list[dict[str, Any]],
        contexts: list[dict[str, Any] | None] | None = None,
        use_cache: bool = True,
    ) -> list[UnifiedScore]:
        """Score a batch of transactions synchronously.

        Args:
            transactions: List of transaction data dicts.
            contexts: Optional list of contexts (one per transaction).
            use_cache: Whether to check/populate cache.

        Returns:
            List of UnifiedScore results.
        """
        if not transactions:
            return []

        batch_cfg = self._config.get("scoring", {}).get("batch", {})
        max_batch = batch_cfg.get("max_batch_size", 1000)

        if len(transactions) > max_batch:
            raise ValueError(f"Batch size {len(transactions)} exceeds maximum {max_batch}")

        if contexts is None:
            contexts = [None] * len(transactions)

        return [
            self.score_transaction_sync(txn, ctx, use_cache=use_cache)
            for txn, ctx in zip(transactions, contexts)
        ]

    # ── Cache Management ─────────────────────────────────────────────

    def invalidate_cache(self, transaction_id: str | None = None) -> None:
        """Invalidate cache for a specific transaction or clear entirely."""
        if self._cache is None:
            return
        if transaction_id:
            self._cache.invalidate(transaction_id)
        else:
            self._cache.clear()

    @property
    def cache_stats(self) -> dict[str, Any] | None:
        if self._cache is None:
            return None
        return self._cache.stats
