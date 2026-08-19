"""Risk scores API routes — model serving, health, and monitoring endpoints.

Provides endpoints for:
- Real-time risk scoring (single and batch)
- Model health and monitoring status
- Model version info and A/B test status
- Model hot-reload trigger
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from src.api.middleware.auth import require_permission, verify_api_key
from src.utils.constants import API_PREFIX

logger = structlog.get_logger(__name__)

router = APIRouter(prefix=f"{API_PREFIX}/risk-scores", tags=["Risk Scores"])


# --- Request/Response Schemas ---


class RiskScoreRequest(BaseModel):
    """Request schema for scoring a single transaction."""

    model_config = ConfigDict(str_strip_whitespace=True)

    transaction_id: str = Field(..., min_length=1, max_length=64)
    customer_id: str = Field(..., min_length=1, max_length=64)
    transaction_amount: float = Field(..., gt=0)
    transaction_currency: str = Field("USD", max_length=3)
    transaction_type: str = Field(...)
    channel: str = Field(...)
    merchant_category_code: str | None = Field(None, max_length=10)
    ip_address: str | None = Field(None, max_length=45)
    device_id: str | None = Field(None, max_length=128)
    geo_country: str | None = Field(None, max_length=3)
    is_international: bool = Field(False)
    features: dict[str, float] | None = Field(
        None, description="Pre-computed feature vector. If provided, bypasses feature store lookup."
    )


class BatchRiskScoreRequest(BaseModel):
    """Request schema for batch scoring."""

    transactions: list[RiskScoreRequest] = Field(..., min_length=1, max_length=1000)


class RiskScoreResponse(BaseModel):
    """Response schema for a single risk score."""

    transaction_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    prediction_latency_ms: float
    ab_test_variant: str | None = None


class BatchRiskScoreResponse(BaseModel):
    """Response schema for batch scoring."""

    scores: list[RiskScoreResponse]
    total: int
    batch_latency_ms: float
    model_version: str


class ModelHealthResponse(BaseModel):
    """Response schema for model health endpoint."""

    status: str
    model_name: str
    model_version: str
    is_ready: bool
    prediction_count: int
    avg_latency_ms: float
    error_rate: float
    prediction_psi: float | None = None
    active_alerts: int


class ModelInfoResponse(BaseModel):
    """Response schema for model info."""

    model_name: str
    active_version: str
    has_fallback: bool
    ab_tests: list[dict[str, Any]] = Field(default_factory=list)
    serving_stats: dict[str, Any]


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str
    request_id: str | None = None


# --- Dependency: Model Server ---


def _get_model_server(request: Request):
    """Retrieve model server from app state."""
    server = getattr(request.app.state, "model_server", None)
    if server is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model server not initialized. Service is starting up.",
        )
    return server


def _get_model_monitor(request: Request):
    """Retrieve model monitor from app state."""
    return getattr(request.app.state, "model_monitor", None)


# --- Endpoints ---


@router.post(
    "/predict",
    response_model=RiskScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a single transaction",
    description="Compute fraud risk score for a single transaction in real-time.",
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        503: {"model": ErrorResponse, "description": "Model not available"},
    },
)
async def predict_risk_score(
    request_body: RiskScoreRequest,
    request: Request,
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> RiskScoreResponse:
    """Score a single transaction for fraud risk."""
    server = _get_model_server(request)
    monitor = _get_model_monitor(request)

    start = time.perf_counter()

    # Build feature vector
    if request_body.features:
        feature_values = np.array(
            [request_body.features.get(f, 0.0) for f in _get_feature_names(server)]
        ).reshape(1, -1)
    else:
        feature_values = _build_features_from_request(request_body, server)

    # Check for A/B test routing
    ab_variant: str | None = None
    user_id = request_body.customer_id
    active_tests = _get_active_ab_tests(request)
    if active_tests:
        test_name = active_tests[0]
        registry = getattr(request.app.state, "model_registry", None)
        if registry:
            assigned_version = registry.resolve_ab_assignment(test_name, user_id)
            ab_variant = assigned_version

    # Run prediction
    try:
        predictions = server.predict(feature_values, user_id=user_id)
        score = float(predictions[0]) if predictions.ndim > 0 else float(predictions)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Classify risk level
    risk_level = _classify_risk(score)

    # Record in monitor
    if monitor:
        features_dict = request_body.features or {}
        monitor.record_prediction(
            score=score,
            features=features_dict if features_dict else None,
            latency_ms=elapsed_ms,
        )

    return RiskScoreResponse(
        transaction_id=request_body.transaction_id,
        risk_score=round(score, 6),
        risk_level=risk_level,
        confidence=_compute_confidence(score),
        model_version=server.active_version,
        prediction_latency_ms=round(elapsed_ms, 3),
        ab_test_variant=ab_variant,
    )


@router.post(
    "/predict/batch",
    response_model=BatchRiskScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a batch of transactions",
    description="Compute fraud risk scores for up to 1000 transactions in a single request.",
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        503: {"model": ErrorResponse, "description": "Model not available"},
    },
)
async def predict_batch(
    request_body: BatchRiskScoreRequest,
    request: Request,
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> BatchRiskScoreResponse:
    """Batch score multiple transactions."""
    server = _get_model_server(request)
    monitor = _get_model_monitor(request)

    start = time.perf_counter()
    feature_names = _get_feature_names(server)

    # Build feature matrix
    feature_arrays: list[np.ndarray] = []
    for txn in request_body.transactions:
        if txn.features:
            arr = np.array([txn.features.get(f, 0.0) for f in feature_names])
        else:
            arr = _build_single_feature_vector(txn, feature_names)
        feature_arrays.append(arr)

    feature_matrix = np.vstack([a.reshape(1, -1) for a in feature_arrays])

    # Run batch prediction
    try:
        predictions = server.predict(feature_matrix)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Build responses
    scores: list[RiskScoreResponse] = []
    for i, txn in enumerate(request_body.transactions):
        score = float(predictions[i])
        scores.append(
            RiskScoreResponse(
                transaction_id=txn.transaction_id,
                risk_score=round(score, 6),
                risk_level=_classify_risk(score),
                confidence=_compute_confidence(score),
                model_version=server.active_version,
                prediction_latency_ms=round(elapsed_ms / len(request_body.transactions), 3),
            )
        )

    # Record batch in monitor
    if monitor:
        monitor.record_batch(
            scores=predictions, feature_matrix=feature_matrix, latency_ms=elapsed_ms
        )

    return BatchRiskScoreResponse(
        scores=scores,
        total=len(scores),
        batch_latency_ms=round(elapsed_ms, 3),
        model_version=server.active_version,
    )


@router.get(
    "/health",
    response_model=ModelHealthResponse,
    summary="Model health status",
    description="Get the current model serving health and monitoring metrics.",
)
async def model_health(
    request: Request,
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> ModelHealthResponse:
    """Return model health and monitoring status."""
    server = _get_model_server(request)
    monitor = _get_model_monitor(request)

    stats = server.stats

    # Add monitoring data if available
    error_rate = 0.0
    psi = None
    active_alerts = 0
    if monitor:
        health = monitor.get_health_status()
        error_rate = health.get("error_rate", 0.0)
        psi = health.get("prediction_psi")
        active_alerts = health.get("active_alerts", 0)

    return ModelHealthResponse(
        status="healthy" if server.is_ready else "unavailable",
        model_name=stats["model_name"],
        model_version=stats["active_version"],
        is_ready=server.is_ready,
        prediction_count=stats["prediction_count"],
        avg_latency_ms=stats["avg_latency_ms"],
        error_rate=error_rate,
        prediction_psi=psi,
        active_alerts=active_alerts,
    )


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Model serving info",
    description="Get information about the currently served model, fallback, and A/B tests.",
)
async def model_info(
    request: Request,
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> ModelInfoResponse:
    """Return model serving information."""
    server = _get_model_server(request)
    registry = getattr(request.app.state, "model_registry", None)

    ab_tests: list[dict[str, Any]] = []
    if registry:
        for test_name, test_data in registry._registry.get("ab_tests", {}).items():
            if test_data.get("is_active", False):
                ab_tests.append({"name": test_name, **test_data})

    return ModelInfoResponse(
        model_name=server._model_name,
        active_version=server.active_version,
        has_fallback=server.stats["has_fallback"],
        ab_tests=ab_tests,
        serving_stats=server.stats,
    )


@router.post(
    "/reload",
    status_code=status.HTTP_200_OK,
    summary="Trigger model hot-reload",
    description="Check for model updates and hot-reload if a newer version is available.",
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def reload_model(
    request: Request,
    _auth: dict[str, Any] = Depends(require_permission("admin")),
) -> dict[str, Any]:
    """Trigger a model hot-reload from registry."""
    server = _get_model_server(request)

    previous_version = server.active_version
    reloaded = server.hot_reload()

    if reloaded:
        logger.info(
            "model_hot_reloaded",
            previous_version=previous_version,
            new_version=server.active_version,
        )
        return {
            "reloaded": True,
            "previous_version": previous_version,
            "current_version": server.active_version,
        }

    return {
        "reloaded": False,
        "current_version": server.active_version,
        "message": "Already on latest version",
    }


@router.get(
    "/monitoring/alerts",
    summary="Get monitoring alerts",
    description="Retrieve recent model monitoring alerts (drift, performance, quality).",
)
async def get_monitoring_alerts(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get recent monitoring alerts."""
    monitor = _get_model_monitor(request)
    if monitor is None:
        return {"alerts": [], "total": 0}

    alerts = monitor.get_alerts(limit=limit)
    return {"alerts": alerts, "total": len(alerts)}


@router.post(
    "/monitoring/check",
    summary="Run monitoring checks",
    description="Manually trigger all monitoring checks (drift, performance, quality).",
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def run_monitoring_checks(
    request: Request,
    _auth: dict[str, Any] = Depends(require_permission("admin")),
) -> dict[str, Any]:
    """Run all monitoring checks and return alerts."""
    monitor = _get_model_monitor(request)
    if monitor is None:
        return {"alerts": [], "health": {}, "message": "Monitor not configured"}

    alerts = monitor.run_all_checks()
    health = monitor.get_health_status()

    return {
        "alerts": [a.to_dict() for a in alerts],
        "health": health,
        "alerts_triggered": len(alerts),
    }


# --- Helper Functions ---


def _classify_risk(score: float) -> str:
    """Map score to risk level."""
    if score >= 0.95:
        return "critical"
    elif score >= 0.8:
        return "high"
    elif score >= 0.5:
        return "medium"
    elif score >= 0.3:
        return "low"
    return "minimal"


def _compute_confidence(score: float) -> float:
    """Compute confidence from score (higher at extremes)."""
    return round(2.0 * abs(score - 0.5), 4)


def _get_feature_names(server) -> list[str]:
    """Get feature names from model server metadata."""
    if hasattr(server, "_primary_metadata") and server._primary_metadata:
        return server._primary_metadata.feature_names or []
    return []


def _get_active_ab_tests(request: Request) -> list[str]:
    """Get names of active A/B tests."""
    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        return []
    tests = registry._registry.get("ab_tests", {})
    return [name for name, data in tests.items() if data.get("is_active", False)]


def _build_features_from_request(req: RiskScoreRequest, server) -> np.ndarray:
    """Build feature vector from request fields when pre-computed features not provided."""
    feature_names = _get_feature_names(server)
    return _build_single_feature_vector(req, feature_names).reshape(1, -1)


def _build_single_feature_vector(req: RiskScoreRequest, feature_names: list[str]) -> np.ndarray:
    """Build a single feature vector from transaction request data."""
    # Map request fields to potential feature values
    field_mapping: dict[str, float] = {
        "transaction_amount": req.transaction_amount,
        "is_international": float(req.is_international),
    }

    values = []
    for name in feature_names:
        values.append(field_mapping.get(name, 0.0))

    return np.array(values, dtype=np.float64)
