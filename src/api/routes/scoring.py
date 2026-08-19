"""Unified Scoring API routes — single, batch, and retrieval endpoints.

Provides endpoints for:
- POST /api/v1/score — score a single transaction
- POST /api/v1/score/batch — score a batch of transactions
- GET /api/v1/score/{transaction_id} — retrieve a cached score
- GET /api/v1/score/metrics — pipeline metrics
- PUT /api/v1/score/weights — update ensemble weights
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.api.middleware.auth import require_permission, verify_api_key
from src.utils.constants import API_PREFIX

if TYPE_CHECKING:
    from src.fraud_detection.scoring_pipeline import ScoringPipeline

logger = structlog.get_logger(__name__)

router = APIRouter(prefix=f"{API_PREFIX}/score", tags=["Scoring"])


# ── Request/Response Schemas ─────────────────────────────────────────


class ScoreTransactionRequest(BaseModel):
    """Request schema for scoring a single transaction."""

    model_config = ConfigDict(str_strip_whitespace=True)

    transaction_id: str = Field(..., min_length=1, max_length=64)
    customer_id: str = Field(..., min_length=1, max_length=64)
    transaction_amount: float = Field(..., gt=0)
    transaction_currency: str = Field("USD", max_length=3)
    transaction_type: str = Field(...)
    channel: str = Field(...)
    merchant_id: str | None = Field(None, max_length=64)
    merchant_name: str | None = Field(None, max_length=256)
    merchant_category_code: str | None = Field(None, max_length=10)
    ip_address: str | None = Field(None, max_length=45)
    device_id: str | None = Field(None, max_length=128)
    geo_country: str | None = Field(None, max_length=3)
    geo_latitude: float | None = Field(None, ge=-90, le=90)
    geo_longitude: float | None = Field(None, ge=-180, le=180)
    is_international: bool = Field(False)
    transaction_timestamp: str | None = Field(None)
    features: dict[str, float] | None = Field(
        None,
        description="Pre-computed feature vector. Bypasses feature store lookup.",
    )
    context: dict[str, Any] | None = Field(
        None,
        description="Enrichment context for rule engine evaluation.",
    )
    use_cache: bool = Field(True, description="Whether to use cached scores.")


class BatchScoreRequest(BaseModel):
    """Request schema for batch scoring."""

    transactions: list[ScoreTransactionRequest] = Field(..., min_length=1, max_length=1000)


class MethodScoreResponse(BaseModel):
    """Individual scoring method result."""

    method: str
    raw_score: float
    normalized_score: float
    weight: float
    weighted_score: float
    latency_ms: float
    success: bool
    error: str | None = None


class ScoreResponse(BaseModel):
    """Response schema for a single unified score."""

    transaction_id: str
    final_score: float = Field(..., ge=0.0, le=1.0)
    risk_classification: str
    method_scores: list[MethodScoreResponse]
    ensemble_weights_used: dict[str, float]
    total_latency_ms: float
    methods_succeeded: int
    methods_failed: int
    alert_recommended: bool
    auto_block_recommended: bool
    cached: bool
    scoring_version: str


class BatchScoreResponse(BaseModel):
    """Response schema for batch scoring."""

    scores: list[ScoreResponse]
    total: int
    successful: int
    failed: int
    batch_latency_ms: float


class UpdateWeightsRequest(BaseModel):
    """Request to update ensemble weights."""

    rule_score: float = Field(..., ge=0.0, le=1.0)
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    ml_score: float = Field(..., ge=0.0, le=1.0)

    @field_validator("ml_score")
    @classmethod
    def weights_must_sum_to_one(cls, v: float, info: Any) -> float:
        rule = info.data.get("rule_score", 0.0)
        anomaly = info.data.get("anomaly_score", 0.0)
        total = rule + anomaly + v
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v


class MetricsResponse(BaseModel):
    """Scoring pipeline metrics."""

    total_scored: int
    avg_latency_ms: float
    classification_distribution: dict[str, int]
    cache_stats: dict[str, Any] | None


# ── Dependency: Pipeline Instance ────────────────────────────────────

_pipeline_instance: "ScoringPipeline | None" = None


def get_scoring_pipeline() -> "ScoringPipeline":
    """Get or create the singleton scoring pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        from src.fraud_detection.scoring_pipeline import ScoringPipeline

        _pipeline_instance = ScoringPipeline()
    return _pipeline_instance


def set_scoring_pipeline(pipeline: "ScoringPipeline") -> None:
    """Override the scoring pipeline (for testing or reconfiguration)."""
    global _pipeline_instance
    _pipeline_instance = pipeline


# ── Helper ───────────────────────────────────────────────────────────


def _request_to_transaction_dict(req: ScoreTransactionRequest) -> dict[str, Any]:
    """Convert API request to internal transaction dict format."""
    txn = {
        "external_transaction_id": req.transaction_id,
        "transaction_id": req.transaction_id,
        "customer_id": req.customer_id,
        "transaction_amount": req.transaction_amount,
        "transaction_currency": req.transaction_currency,
        "transaction_type": req.transaction_type,
        "channel": req.channel,
        "is_international": req.is_international,
    }
    if req.merchant_id:
        txn["merchant_id"] = req.merchant_id
    if req.merchant_name:
        txn["merchant_name"] = req.merchant_name
    if req.merchant_category_code:
        txn["merchant_category_code"] = req.merchant_category_code
    if req.ip_address:
        txn["ip_address"] = req.ip_address
    if req.device_id:
        txn["device_id"] = req.device_id
    if req.geo_country:
        txn["geo_country"] = req.geo_country
    if req.geo_latitude is not None:
        txn["geo_latitude"] = req.geo_latitude
    if req.geo_longitude is not None:
        txn["geo_longitude"] = req.geo_longitude
    if req.transaction_timestamp:
        txn["transaction_timestamp"] = req.transaction_timestamp
    if req.features:
        txn.update(req.features)
    return txn


def _unified_score_to_response(score: Any) -> ScoreResponse:
    """Convert internal UnifiedScore to API response."""
    return ScoreResponse(
        transaction_id=score.transaction_id,
        final_score=round(score.final_score, 6),
        risk_classification=score.risk_classification.value,
        method_scores=[
            MethodScoreResponse(
                method=ms.method,
                raw_score=round(ms.raw_score, 6),
                normalized_score=round(ms.normalized_score, 6),
                weight=ms.weight,
                weighted_score=round(ms.weighted_score, 6),
                latency_ms=round(ms.latency_ms, 3),
                success=ms.success,
                error=ms.error,
            )
            for ms in score.method_scores
        ],
        ensemble_weights_used=score.ensemble_weights_used,
        total_latency_ms=round(score.total_latency_ms, 3),
        methods_succeeded=score.methods_succeeded,
        methods_failed=score.methods_failed,
        alert_recommended=score.alert_recommended,
        auto_block_recommended=score.auto_block_recommended,
        cached=score.cached,
        scoring_version=score.scoring_version,
    )


# ── Endpoints ────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a single transaction",
    description="Run unified fraud scoring (rule + anomaly + ML) on a single transaction.",
)
async def score_transaction(
    request: ScoreTransactionRequest,
    pipeline: "ScoringPipeline" = Depends(get_scoring_pipeline),
    _api_key: str = Depends(verify_api_key),
) -> ScoreResponse:
    txn_dict = _request_to_transaction_dict(request)
    context = request.context

    try:
        score = await pipeline.score_transaction(
            txn_dict, context=context, use_cache=request.use_cache
        )
    except Exception as exc:
        logger.error(
            "scoring_endpoint_error",
            transaction_id=request.transaction_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {type(exc).__name__}",
        )

    return _unified_score_to_response(score)


@router.post(
    "/batch",
    response_model=BatchScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a batch of transactions",
    description="Run unified fraud scoring on multiple transactions in parallel.",
)
async def score_batch(
    request: BatchScoreRequest,
    pipeline: "ScoringPipeline" = Depends(get_scoring_pipeline),
    _api_key: str = Depends(verify_api_key),
) -> BatchScoreResponse:
    start = time.perf_counter()

    txn_dicts = [_request_to_transaction_dict(t) for t in request.transactions]
    contexts = [t.context for t in request.transactions]

    try:
        scores = await pipeline.score_batch(txn_dicts, contexts=contexts)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("batch_scoring_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch scoring failed: {type(exc).__name__}",
        )

    batch_latency = (time.perf_counter() - start) * 1000

    return BatchScoreResponse(
        scores=[_unified_score_to_response(s) for s in scores],
        total=len(request.transactions),
        successful=len(scores),
        failed=len(request.transactions) - len(scores),
        batch_latency_ms=round(batch_latency, 3),
    )


@router.get(
    "/{transaction_id}",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve a cached score",
    description="Retrieve a previously computed score from cache by transaction ID.",
)
async def get_score(
    transaction_id: str,
    pipeline: "ScoringPipeline" = Depends(get_scoring_pipeline),
    _api_key: str = Depends(verify_api_key),
) -> ScoreResponse:
    if pipeline._cache is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Caching is disabled; score not available.",
        )

    # Search cache by iterating (transaction_id-based lookup)
    # The cache is keyed by hash, so we need to check stored scores
    with pipeline._cache._lock:
        for _, (_, score) in pipeline._cache._cache.items():
            if score.transaction_id == transaction_id:
                score.cached = True
                return _unified_score_to_response(score)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Score not found for transaction_id: {transaction_id}",
    )


@router.get(
    "/metrics/summary",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get scoring pipeline metrics",
    description="Retrieve aggregated scoring pipeline performance metrics.",
)
async def get_metrics(
    pipeline: "ScoringPipeline" = Depends(get_scoring_pipeline),
    _api_key: str = Depends(verify_api_key),
) -> MetricsResponse:
    m = pipeline.metrics
    return MetricsResponse(
        total_scored=m["total_scored"],
        avg_latency_ms=m["avg_latency_ms"],
        classification_distribution=m["classification_distribution"],
        cache_stats=m["cache_stats"],
    )


@router.put(
    "/weights",
    response_model=dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Update ensemble weights",
    description="Dynamically update the ensemble scoring weights.",
)
async def update_weights(
    request: UpdateWeightsRequest,
    pipeline: "ScoringPipeline" = Depends(get_scoring_pipeline),
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    new_weights = {
        "rule_score": request.rule_score,
        "anomaly_score": request.anomaly_score,
        "ml_score": request.ml_score,
    }
    try:
        pipeline.update_weights(new_weights)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return {
        "status": "updated",
        "weights": pipeline.weights,
    }
