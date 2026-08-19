"""End-to-end pipeline orchestrator for RiskPulse transaction processing.

Chains all processing stages:
    Ingest → Validate → Transform (Clean → Normalize → Features) → Enrich

Supports configurable batch processing, stage-level error handling,
dead-letter queue routing, and per-stage latency/throughput metrics.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable

from src.enrichment.device_enricher import DeviceEnricher
from src.enrichment.geo_enricher import GeoEnricher
from src.enrichment.merchant_enricher import MerchantEnricher
from src.enrichment.velocity_calculator import VelocityCalculator
from src.transformation.cleaner import DataCleaner
from src.transformation.feature_engineer import FeatureEngineer
from src.transformation.normalizer import DataNormalizer, get_normalizer
from src.utils.logger import get_logger
from src.validation.quarantine_handler import QuarantineHandler
from src.validation.rules_engine import RuleAction, RulesEngine, get_rules_engine
from src.validation.schema_validator import SchemaValidator

logger = get_logger(__name__, component="pipeline_orchestrator")


class StageErrorPolicy(str, Enum):
    """How to handle errors at a pipeline stage."""

    SKIP = "skip"
    HALT = "halt"


class PipelineStage(str, Enum):
    """Pipeline processing stages."""

    VALIDATION = "validation"
    CLEANING = "cleaning"
    NORMALIZATION = "normalization"
    FEATURE_ENGINEERING = "feature_engineering"
    ENRICHMENT = "enrichment"


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""

    stage: str
    records_processed: int = 0
    records_failed: int = 0
    records_skipped: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.records_processed == 0:
            return 0.0
        return self.total_latency_ms / self.records_processed

    @property
    def success_rate(self) -> float:
        total = self.records_processed + self.records_failed
        if total == 0:
            return 1.0
        return self.records_processed / total

    def record_success(self, latency_ms: float) -> None:
        self.records_processed += 1
        self.total_latency_ms += latency_ms
        if latency_ms < self.min_latency_ms:
            self.min_latency_ms = latency_ms
        if latency_ms > self.max_latency_ms:
            self.max_latency_ms = latency_ms

    def record_failure(self) -> None:
        self.records_failed += 1

    def record_skip(self) -> None:
        self.records_skipped += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "records_processed": self.records_processed,
            "records_failed": self.records_failed,
            "records_skipped": self.records_skipped,
            "avg_latency_ms": round(self.avg_latency_ms, 4),
            "min_latency_ms": (
                round(self.min_latency_ms, 4) if self.min_latency_ms != float("inf") else 0.0
            ),
            "max_latency_ms": round(self.max_latency_ms, 4),
            "success_rate": round(self.success_rate, 4),
        }


@dataclass
class PipelineMetrics:
    """Aggregate metrics across all pipeline stages."""

    total_ingested: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_dlq: int = 0
    pipeline_start_time: float = 0.0
    pipeline_end_time: float = 0.0
    stage_metrics: dict[str, StageMetrics] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def throughput_per_second(self) -> float:
        elapsed = self.pipeline_end_time - self.pipeline_start_time
        if elapsed <= 0:
            return 0.0
        return self.total_completed / elapsed

    @property
    def total_pipeline_latency_ms(self) -> float:
        return (self.pipeline_end_time - self.pipeline_start_time) * 1000

    @property
    def avg_per_record_ms(self) -> float:
        if self.total_completed == 0:
            return 0.0
        return self.total_pipeline_latency_ms / self.total_ingested

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ingested": self.total_ingested,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "total_dlq": self.total_dlq,
            "throughput_per_second": round(self.throughput_per_second, 2),
            "avg_per_record_ms": round(self.avg_per_record_ms, 4),
            "total_pipeline_latency_ms": round(self.total_pipeline_latency_ms, 2),
            "stages": {k: v.to_dict() for k, v in self.stage_metrics.items()},
        }


@dataclass
class PipelineResult:
    """Result of processing a single record through the pipeline."""

    transaction_id: str
    success: bool
    record: dict[str, Any] = field(default_factory=dict)
    stage_failed: str | None = None
    error: str | None = None
    dlq: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "success": self.success,
            "stage_failed": self.stage_failed,
            "error": self.error,
            "dlq": self.dlq,
            "latency_ms": round(self.latency_ms, 4),
        }


@dataclass
class BatchResult:
    """Result of processing a batch through the pipeline."""

    batch_id: str
    total: int
    succeeded: int
    failed: int
    dlq_count: int
    results: list[PipelineResult]
    metrics: PipelineMetrics
    latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 1.0
        return self.succeeded / self.total


class PipelineOrchestrator:
    """Orchestrates the end-to-end transaction processing pipeline.

    Chains stages: Validate → Clean → Normalize → Feature Engineer → Enrich.
    Each stage has configurable error handling (skip or halt).
    Failed records are routed to a dead-letter queue with full context.

    Usage:
        orchestrator = PipelineOrchestrator()
        result = orchestrator.process_batch(transactions)
    """

    def __init__(
        self,
        *,
        schema_validator: SchemaValidator | None = None,
        rules_engine: RulesEngine | None = None,
        data_cleaner: DataCleaner | None = None,
        normalizer: DataNormalizer | None = None,
        feature_engineer: FeatureEngineer | None = None,
        geo_enricher: GeoEnricher | None = None,
        device_enricher: DeviceEnricher | None = None,
        merchant_enricher: MerchantEnricher | None = None,
        velocity_calculator: VelocityCalculator | None = None,
        quarantine_handler: QuarantineHandler | None = None,
        error_policy: dict[str, StageErrorPolicy] | None = None,
        batch_size: int = 100,
        on_dlq: Callable[[dict[str, Any], str, str], None] | None = None,
    ) -> None:
        # Stage components
        self._validator = schema_validator or SchemaValidator()
        self._rules_engine = rules_engine or get_rules_engine()
        self._cleaner = data_cleaner or DataCleaner()
        self._normalizer = normalizer or get_normalizer()
        self._feature_engineer = feature_engineer or FeatureEngineer()
        self._geo_enricher = geo_enricher or GeoEnricher()
        self._device_enricher = device_enricher or DeviceEnricher()
        self._merchant_enricher = merchant_enricher or MerchantEnricher()
        self._velocity_calculator = velocity_calculator or VelocityCalculator()
        self._quarantine = quarantine_handler or QuarantineHandler()

        # Error handling policy per stage (default: skip)
        self._error_policy: dict[str, StageErrorPolicy] = {
            PipelineStage.VALIDATION.value: StageErrorPolicy.SKIP,
            PipelineStage.CLEANING.value: StageErrorPolicy.SKIP,
            PipelineStage.NORMALIZATION.value: StageErrorPolicy.SKIP,
            PipelineStage.FEATURE_ENGINEERING.value: StageErrorPolicy.SKIP,
            PipelineStage.ENRICHMENT.value: StageErrorPolicy.SKIP,
        }
        if error_policy:
            self._error_policy.update(error_policy)

        self._batch_size = batch_size
        self._on_dlq = on_dlq
        self._metrics = PipelineMetrics()
        self._dlq: list[dict[str, Any]] = []
        self._lock = Lock()

        logger.info(
            "Pipeline orchestrator initialized",
            batch_size=batch_size,
            error_policies={k: v.value for k, v in self._error_policy.items()},
        )

    @property
    def metrics(self) -> PipelineMetrics:
        return self._metrics

    @property
    def dlq(self) -> list[dict[str, Any]]:
        return list(self._dlq)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def process_record(self, record: dict[str, Any]) -> PipelineResult:
        """Process a single transaction record through all pipeline stages.

        Args:
            record: Raw transaction record.

        Returns:
            PipelineResult with the enriched record or failure context.
        """
        start = time.perf_counter()
        txn_id = record.get(
            "external_transaction_id",
            record.get("transaction_id", f"unknown-{uuid.uuid4().hex[:8]}"),
        )

        try:
            current_record = dict(record)

            # Stage 1: Schema Validation
            current_record = self._run_validation(current_record, txn_id)
            if current_record is None:
                return self._make_failure(
                    txn_id, PipelineStage.VALIDATION, "Schema validation failed", start
                )

            # Stage 2: Business Rules
            current_record = self._run_rules_engine(current_record, txn_id)
            if current_record is None:
                return self._make_failure(
                    txn_id, PipelineStage.VALIDATION, "Blocked by rules engine", start
                )

            # Stage 3: Cleaning
            current_record = self._run_cleaning(current_record, txn_id)
            if current_record is None:
                return self._make_failure(txn_id, PipelineStage.CLEANING, "Cleaning failed", start)

            # Stage 4: Normalization
            current_record = self._run_normalization(current_record, txn_id)
            if current_record is None:
                return self._make_failure(
                    txn_id, PipelineStage.NORMALIZATION, "Normalization failed", start
                )

            # Stage 5: Feature Engineering
            current_record = self._run_feature_engineering(current_record, txn_id)
            if current_record is None:
                return self._make_failure(
                    txn_id, PipelineStage.FEATURE_ENGINEERING, "Feature engineering failed", start
                )

            # Stage 6: Enrichment (Geo + Device + Merchant + Velocity)
            current_record = self._run_enrichment(current_record, txn_id)
            if current_record is None:
                return self._make_failure(
                    txn_id, PipelineStage.ENRICHMENT, "Enrichment failed", start
                )

            # Mark pipeline metadata
            current_record["_pipeline_processed"] = True
            current_record["_pipeline_timestamp"] = time.time()

            elapsed = (time.perf_counter() - start) * 1000
            return PipelineResult(
                transaction_id=txn_id,
                success=True,
                record=current_record,
                latency_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Pipeline processing failed", transaction_id=txn_id, error=str(e))
            self._route_to_dlq(record, "pipeline_exception", str(e))
            return PipelineResult(
                transaction_id=txn_id,
                success=False,
                error=str(e),
                dlq=True,
                latency_ms=elapsed,
            )

    def process_batch(self, records: list[dict[str, Any]]) -> BatchResult:
        """Process a batch of transaction records through the pipeline.

        Args:
            records: List of raw transaction records.

        Returns:
            BatchResult with per-record results and aggregate metrics.
        """
        batch_id = uuid.uuid4().hex[:12]
        batch_start = time.perf_counter()

        self._metrics.pipeline_start_time = batch_start
        self._metrics.total_ingested = len(records)

        # Initialize stage metrics
        for stage in PipelineStage:
            self._metrics.stage_metrics[stage.value] = StageMetrics(stage=stage.value)

        results: list[PipelineResult] = []
        succeeded = 0
        failed = 0
        dlq_count = 0

        for record in records:
            result = self.process_record(record)
            results.append(result)

            if result.success:
                succeeded += 1
            else:
                failed += 1
                if result.dlq:
                    dlq_count += 1

        batch_end = time.perf_counter()
        self._metrics.pipeline_end_time = batch_end
        self._metrics.total_completed = succeeded
        self._metrics.total_failed = failed
        self._metrics.total_dlq = dlq_count

        elapsed = (batch_end - batch_start) * 1000

        logger.info(
            "Batch processing complete",
            batch_id=batch_id,
            total=len(records),
            succeeded=succeeded,
            failed=failed,
            dlq=dlq_count,
            latency_ms=round(elapsed, 2),
            throughput=round(self._metrics.throughput_per_second, 2),
        )

        return BatchResult(
            batch_id=batch_id,
            total=len(records),
            succeeded=succeeded,
            failed=failed,
            dlq_count=dlq_count,
            results=results,
            metrics=self._metrics,
            latency_ms=elapsed,
        )

    def get_stage_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for each pipeline stage."""
        return {k: v.to_dict() for k, v in self._metrics.stage_metrics.items()}

    def reset_metrics(self) -> None:
        """Reset all pipeline metrics."""
        self._metrics = PipelineMetrics()
        self._dlq.clear()

    # -------------------------------------------------------------------------
    # Stage Implementations
    # -------------------------------------------------------------------------

    def _run_validation(self, record: dict[str, Any], txn_id: str) -> dict[str, Any] | None:
        """Stage 1: Schema validation."""
        stage = PipelineStage.VALIDATION.value
        start = time.perf_counter()

        try:
            result = self._validator.validate(record)
            latency = (time.perf_counter() - start) * 1000

            if not result.is_valid:
                self._quarantine.quarantine(record, result)
                self._record_stage_failure(stage)
                self._route_to_dlq(record, stage, f"Validation errors: {len(result.errors)}")
                return None

            self._record_stage_success(stage, latency)
            record["_validation_warnings"] = len(result.warnings)
            return record

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            logger.error("Validation stage error", transaction_id=txn_id, error=str(e))
            return self._handle_stage_error(stage, record, txn_id, str(e))

    def _run_rules_engine(self, record: dict[str, Any], txn_id: str) -> dict[str, Any] | None:
        """Stage 1b: Business rules evaluation."""
        stage = PipelineStage.VALIDATION.value
        start = time.perf_counter()

        try:
            result = self._rules_engine.evaluate(record)
            latency = (time.perf_counter() - start) * 1000

            if result.overall_action == RuleAction.BLOCK:
                self._record_stage_failure(stage)
                self._route_to_dlq(
                    record,
                    "rules_engine_block",
                    f"Blocked by rules: {[r.rule_id for r in result.triggered_rules]}",
                )
                return None

            # Attach rule evaluation metadata
            record["_rules_triggered"] = result.total_rules_triggered
            record["_rules_action"] = result.overall_action.value
            self._record_stage_success(stage, latency)
            return record

        except Exception as e:
            logger.error("Rules engine error", transaction_id=txn_id, error=str(e))
            return self._handle_stage_error(stage, record, txn_id, str(e))

    def _run_cleaning(self, record: dict[str, Any], txn_id: str) -> dict[str, Any] | None:
        """Stage 2: Data cleaning."""
        stage = PipelineStage.CLEANING.value
        start = time.perf_counter()

        try:
            result = self._cleaner.clean(record)
            latency = (time.perf_counter() - start) * 1000

            if result.is_duplicate:
                self._record_stage_skip(stage)
                self._route_to_dlq(record, stage, "Duplicate record detected")
                return None

            self._record_stage_success(stage, latency)
            return result.record

        except Exception as e:
            logger.error("Cleaning stage error", transaction_id=txn_id, error=str(e))
            return self._handle_stage_error(stage, record, txn_id, str(e))

    def _run_normalization(self, record: dict[str, Any], txn_id: str) -> dict[str, Any] | None:
        """Stage 3: Data normalization."""
        stage = PipelineStage.NORMALIZATION.value
        start = time.perf_counter()

        try:
            result = self._normalizer.normalize(record)
            latency = (time.perf_counter() - start) * 1000
            self._record_stage_success(stage, latency)
            return result.record

        except Exception as e:
            logger.error("Normalization stage error", transaction_id=txn_id, error=str(e))
            return self._handle_stage_error(stage, record, txn_id, str(e))

    def _run_feature_engineering(
        self, record: dict[str, Any], txn_id: str
    ) -> dict[str, Any] | None:
        """Stage 4: Feature engineering."""
        stage = PipelineStage.FEATURE_ENGINEERING.value
        start = time.perf_counter()

        try:
            result = self._feature_engineer.compute_features(record)
            latency = (time.perf_counter() - start) * 1000

            if not result.is_success:
                self._record_stage_failure(stage)
                return self._handle_stage_error(
                    stage, record, txn_id, result.error or "Feature computation failed"
                )

            # Merge features into record
            record.update(result.features)
            record["_feature_count"] = len(result.features)
            self._record_stage_success(stage, latency)
            return record

        except Exception as e:
            logger.error("Feature engineering error", transaction_id=txn_id, error=str(e))
            return self._handle_stage_error(stage, record, txn_id, str(e))

    def _run_enrichment(self, record: dict[str, Any], txn_id: str) -> dict[str, Any] | None:
        """Stage 5: Multi-source enrichment (geo, device, merchant, velocity)."""
        stage = PipelineStage.ENRICHMENT.value
        start = time.perf_counter()

        try:
            # Geo enrichment
            geo_result = self._geo_enricher.enrich(record)
            if geo_result.is_success:
                record.update(geo_result.to_dict())

            # Device enrichment
            device_result = self._device_enricher.enrich(record)
            if device_result.is_success:
                record.update(device_result.to_dict())

            # Merchant enrichment
            merchant_result = self._merchant_enricher.enrich(record)
            if merchant_result.is_success:
                record.update(merchant_result.to_dict())

            # Velocity enrichment
            velocity_result = self._velocity_calculator.evaluate(record)
            if velocity_result.is_success:
                record.update(velocity_result.to_dict())

            latency = (time.perf_counter() - start) * 1000
            record["_enrichment_latency_ms"] = round(latency, 4)
            self._record_stage_success(stage, latency)
            return record

        except Exception as e:
            logger.error("Enrichment stage error", transaction_id=txn_id, error=str(e))
            return self._handle_stage_error(stage, record, txn_id, str(e))

    # -------------------------------------------------------------------------
    # Error Handling Helpers
    # -------------------------------------------------------------------------

    def _handle_stage_error(
        self, stage: str, record: dict[str, Any], txn_id: str, error: str
    ) -> dict[str, Any] | None:
        """Handle a stage error according to the configured policy."""
        policy = self._error_policy.get(stage, StageErrorPolicy.SKIP)

        if policy == StageErrorPolicy.HALT:
            self._route_to_dlq(record, stage, error)
            return None

        # SKIP: log warning and pass through the record as-is
        logger.warning(
            "Stage error skipped",
            stage=stage,
            transaction_id=txn_id,
            error=error,
            policy="skip",
        )
        record[f"_{stage}_error"] = error
        return record

    def _route_to_dlq(self, record: dict[str, Any], stage: str, reason: str) -> None:
        """Route a failed record to the dead-letter queue."""
        dlq_entry = {
            "dlq_id": uuid.uuid4().hex,
            "original_record": record,
            "failed_stage": stage,
            "failure_reason": reason,
            "timestamp": time.time(),
        }

        with self._lock:
            self._dlq.append(dlq_entry)

        if self._on_dlq:
            try:
                self._on_dlq(record, stage, reason)
            except Exception as e:
                logger.error("DLQ callback failed", error=str(e))

    def _record_stage_success(self, stage: str, latency_ms: float) -> None:
        """Record a successful stage processing."""
        if stage in self._metrics.stage_metrics:
            self._metrics.stage_metrics[stage].record_success(latency_ms)

    def _record_stage_failure(self, stage: str) -> None:
        """Record a stage processing failure."""
        if stage in self._metrics.stage_metrics:
            self._metrics.stage_metrics[stage].record_failure()

    def _record_stage_skip(self, stage: str) -> None:
        """Record a skipped record at a stage."""
        if stage in self._metrics.stage_metrics:
            self._metrics.stage_metrics[stage].record_skip()

    def _make_failure(
        self, txn_id: str, stage: PipelineStage, error: str, start: float
    ) -> PipelineResult:
        """Create a failure PipelineResult."""
        elapsed = (time.perf_counter() - start) * 1000
        return PipelineResult(
            transaction_id=txn_id,
            success=False,
            stage_failed=stage.value,
            error=error,
            dlq=True,
            latency_ms=elapsed,
        )
