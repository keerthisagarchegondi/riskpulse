"""Rule management API routes for RiskPulse Rules Engine.

Provides CRUD endpoints for business rules, evaluation testing,
and audit trail access.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from src.api.middleware.auth import require_permission, verify_api_key
from src.utils.constants import API_PREFIX, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from src.validation.rules_engine import (
    RuleAction,
    RulesEngine,
    RuleSeverity,
    get_rules_engine,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix=f"{API_PREFIX}/rules", tags=["Rules Engine"])


# --- Request/Response Schemas ---


class RuleConditionSchema(BaseModel):
    """Schema for rule condition definition."""

    model_config = ConfigDict(extra="allow")

    field: str | None = None
    operator: str | None = None
    value: Any = None
    type: str | None = None
    conditions: list[dict[str, Any]] | None = None


class RuleCreateRequest(BaseModel):
    """Schema for creating a new rule."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(..., min_length=1, max_length=64, description="Unique rule identifier")
    name: str = Field(..., min_length=1, max_length=128, description="Human-readable rule name")
    description: str = Field(default="", max_length=512, description="Rule description")
    version: str = Field(default="1.0.0", max_length=20, description="Rule version (semver)")
    priority: int = Field(default=100, ge=1, le=1000, description="Execution priority (lower = higher)")
    enabled: bool = Field(default=True, description="Whether the rule is active")
    severity: str = Field(default="medium", description="Rule severity level")
    category: str = Field(default="custom", max_length=64, description="Rule category")
    condition: dict[str, Any] = Field(..., description="Rule condition definition")
    action: str = Field(default="flag", description="Action on trigger: block, flag, allow")
    schedule: dict[str, Any] | None = Field(default=None, description="Time-based activation schedule")
    tags: list[str] = Field(default_factory=list, description="Rule tags for filtering")


class RuleUpdateRequest(BaseModel):
    """Schema for updating an existing rule."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=512)
    version: str | None = Field(default=None, max_length=20)
    priority: int | None = Field(default=None, ge=1, le=1000)
    enabled: bool | None = None
    severity: str | None = None
    category: str | None = Field(default=None, max_length=64)
    condition: dict[str, Any] | None = None
    action: str | None = None
    schedule: dict[str, Any] | None = None
    tags: list[str] | None = None


class RuleResponse(BaseModel):
    """Schema for a single rule response."""

    id: str
    name: str
    description: str
    version: str
    priority: int
    enabled: bool
    severity: str
    category: str
    condition: dict[str, Any]
    action: str
    schedule: dict[str, Any] | None = None
    tags: list[str] = []


class RuleListResponse(BaseModel):
    """Response for listing rules."""

    rules: list[RuleResponse]
    total: int
    categories: list[str]


class RuleEvaluateRequest(BaseModel):
    """Request to evaluate a transaction against rules."""

    model_config = ConfigDict(extra="allow")

    transaction: dict[str, Any] = Field(..., description="Transaction data to evaluate")


class RuleEvaluateResponse(BaseModel):
    """Response from rule evaluation."""

    transaction_id: str
    overall_action: str
    triggered_rules: list[dict[str, Any]]
    total_rules_evaluated: int
    total_rules_triggered: int
    highest_severity: str | None = None
    latency_ms: float
    short_circuited: bool


class AuditTrailResponse(BaseModel):
    """Response containing audit trail records."""

    records: list[dict[str, Any]]
    total: int


class AuditStatsResponse(BaseModel):
    """Response containing audit statistics."""

    total_evaluations: int
    total_triggered: int
    trigger_rate: float
    unique_transactions: int
    rules_tracked: int
    breakdown: dict[str, int]


class RuleEngineStatusResponse(BaseModel):
    """Rules engine status information."""

    total_rules: int
    enabled_rules: int
    categories: list[str]
    velocity_tracked_entities: int
    audit_trail_records: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str


# --- Dependency ---


def get_engine() -> RulesEngine:
    """FastAPI dependency to get rules engine instance."""
    return get_rules_engine()


# --- Endpoints ---


@router.get(
    "",
    response_model=RuleListResponse,
    summary="List all rules",
    description="Get all configured business rules with optional filtering.",
)
async def list_rules(
    category: str | None = Query(default=None, description="Filter by category"),
    enabled_only: bool = Query(default=False, description="Only return enabled rules"),
    engine: RulesEngine = Depends(get_engine),
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> RuleListResponse:
    """List all configured business rules."""
    rules = engine.get_rules(category=category, enabled_only=enabled_only)
    return RuleListResponse(
        rules=rules,
        total=len(rules),
        categories=engine.get_categories(),
    )


@router.get(
    "/status",
    response_model=RuleEngineStatusResponse,
    summary="Get rules engine status",
    description="Get current status of the rules engine including rule counts and metrics.",
)
async def get_status(
    engine: RulesEngine = Depends(get_engine),
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> RuleEngineStatusResponse:
    """Get rules engine operational status."""
    return RuleEngineStatusResponse(
        total_rules=engine.total_rules,
        enabled_rules=engine.enabled_rules,
        categories=engine.get_categories(),
        velocity_tracked_entities=engine.velocity_tracker.tracked_entities,
        audit_trail_records=engine.audit_trail.total_records,
    )


@router.get(
    "/{rule_id}",
    response_model=RuleResponse,
    summary="Get a specific rule",
    description="Get the full definition of a specific rule by its ID.",
    responses={404: {"model": ErrorResponse}},
)
async def get_rule(
    rule_id: str,
    engine: RulesEngine = Depends(get_engine),
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> RuleResponse:
    """Get a single rule by ID."""
    rule = engine.get_rule(rule_id)
    if not rule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule '{rule_id}' not found.",
        )
    return RuleResponse(**rule)


@router.post(
    "",
    response_model=RuleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new rule",
    description="Add a new business rule to the engine (in-memory, not persisted to YAML).",
    responses={
        409: {"model": ErrorResponse, "description": "Rule already exists"},
    },
)
async def create_rule(
    rule_data: RuleCreateRequest,
    engine: RulesEngine = Depends(get_engine),
    _auth: dict[str, Any] = Depends(require_permission("admin")),
) -> RuleResponse:
    """Create a new business rule."""
    try:
        engine.add_rule(rule_data.model_dump())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    rule = engine.get_rule(rule_data.id)
    return RuleResponse(**rule)


@router.patch(
    "/{rule_id}",
    response_model=RuleResponse,
    summary="Update an existing rule",
    description="Partially update an existing rule's properties.",
    responses={404: {"model": ErrorResponse}},
)
async def update_rule(
    rule_id: str,
    updates: RuleUpdateRequest,
    engine: RulesEngine = Depends(get_engine),
    _auth: dict[str, Any] = Depends(require_permission("admin")),
) -> RuleResponse:
    """Update an existing rule."""
    update_data = updates.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update provided.",
        )
    result = engine.update_rule(rule_id, update_data)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule '{rule_id}' not found.",
        )
    return RuleResponse(**result)


@router.delete(
    "/{rule_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a rule",
    description="Remove a rule from the engine (in-memory).",
    responses={404: {"model": ErrorResponse}},
)
async def delete_rule(
    rule_id: str,
    engine: RulesEngine = Depends(get_engine),
    _auth: dict[str, Any] = Depends(require_permission("admin")),
) -> None:
    """Delete a rule by ID."""
    if not engine.remove_rule(rule_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule '{rule_id}' not found.",
        )


@router.post(
    "/{rule_id}/enable",
    response_model=RuleResponse,
    summary="Enable a rule",
    description="Enable a previously disabled rule.",
    responses={404: {"model": ErrorResponse}},
)
async def enable_rule(
    rule_id: str,
    engine: RulesEngine = Depends(get_engine),
    _auth: dict[str, Any] = Depends(require_permission("admin")),
) -> RuleResponse:
    """Enable a rule."""
    if not engine.enable_rule(rule_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule '{rule_id}' not found.",
        )
    return RuleResponse(**engine.get_rule(rule_id))


@router.post(
    "/{rule_id}/disable",
    response_model=RuleResponse,
    summary="Disable a rule",
    description="Disable a rule without deleting it.",
    responses={404: {"model": ErrorResponse}},
)
async def disable_rule(
    rule_id: str,
    engine: RulesEngine = Depends(get_engine),
    _auth: dict[str, Any] = Depends(require_permission("admin")),
) -> RuleResponse:
    """Disable a rule."""
    if not engine.disable_rule(rule_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rule '{rule_id}' not found.",
        )
    return RuleResponse(**engine.get_rule(rule_id))


@router.post(
    "/evaluate",
    response_model=RuleEvaluateResponse,
    summary="Evaluate a transaction",
    description="Test a transaction against all active rules. Used for testing and debugging.",
)
async def evaluate_transaction(
    request: RuleEvaluateRequest,
    engine: RulesEngine = Depends(get_engine),
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> RuleEvaluateResponse:
    """Evaluate a transaction against all rules."""
    result = engine.evaluate(request.transaction)
    return RuleEvaluateResponse(
        transaction_id=result.transaction_id,
        overall_action=result.overall_action.value,
        triggered_rules=[r.to_dict() for r in result.triggered_rules],
        total_rules_evaluated=result.total_rules_evaluated,
        total_rules_triggered=result.total_rules_triggered,
        highest_severity=result.highest_severity.value if result.highest_severity else None,
        latency_ms=result.latency_ms,
        short_circuited=result.short_circuited,
    )


@router.post(
    "/reload",
    status_code=status.HTTP_200_OK,
    summary="Force reload rules",
    description="Force the engine to reload rules from the YAML configuration file.",
)
async def reload_rules(
    engine: RulesEngine = Depends(get_engine),
    _auth: dict[str, Any] = Depends(require_permission("admin")),
) -> dict[str, Any]:
    """Force reload rules from configuration file."""
    engine.force_reload()
    return {
        "message": "Rules reloaded successfully",
        "total_rules": engine.total_rules,
        "enabled_rules": engine.enabled_rules,
    }


# --- Audit Trail Endpoints ---


@router.get(
    "/audit/stats",
    response_model=AuditStatsResponse,
    summary="Get audit statistics",
    description="Get aggregated statistics about rule evaluations.",
)
async def get_audit_stats(
    engine: RulesEngine = Depends(get_engine),
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> AuditStatsResponse:
    """Get audit trail statistics."""
    stats = engine.audit_trail.get_stats()
    return AuditStatsResponse(**stats)


@router.get(
    "/audit/recent",
    response_model=AuditTrailResponse,
    summary="Get recent audit records",
    description="Get the most recent rule evaluation audit records.",
)
async def get_recent_audit(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum records to return"),
    engine: RulesEngine = Depends(get_engine),
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> AuditTrailResponse:
    """Get recent audit trail records."""
    records = engine.audit_trail.get_recent(limit=limit)
    return AuditTrailResponse(records=records, total=len(records))


@router.get(
    "/audit/transaction/{transaction_id}",
    response_model=AuditTrailResponse,
    summary="Get audit for transaction",
    description="Get all rule evaluations for a specific transaction.",
)
async def get_transaction_audit(
    transaction_id: str,
    engine: RulesEngine = Depends(get_engine),
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> AuditTrailResponse:
    """Get audit trail for a specific transaction."""
    records = engine.audit_trail.get_by_transaction(transaction_id)
    return AuditTrailResponse(records=records, total=len(records))


@router.get(
    "/audit/rule/{rule_id}",
    response_model=AuditTrailResponse,
    summary="Get audit for rule",
    description="Get recent evaluation history for a specific rule.",
)
async def get_rule_audit(
    rule_id: str,
    limit: int = Query(default=100, ge=1, le=500, description="Maximum records to return"),
    engine: RulesEngine = Depends(get_engine),
    _auth: dict[str, Any] = Depends(verify_api_key),
) -> AuditTrailResponse:
    """Get audit trail for a specific rule."""
    records = engine.audit_trail.get_by_rule(rule_id, limit=limit)
    return AuditTrailResponse(records=records, total=len(records))
