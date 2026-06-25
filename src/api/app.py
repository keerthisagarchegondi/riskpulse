"""FastAPI application factory for RiskPulse API.

Creates and configures the FastAPI application with middleware, routes,
exception handlers, and lifecycle management.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.utils.config import get_settings
from src.utils.constants import APP_NAME, APP_VERSION, API_PREFIX
from src.utils.logger import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    settings = get_settings()
    configure_logging()
    logger.info(
        "api_starting",
        service=APP_NAME,
        version=APP_VERSION,
        environment=settings.environment,
    )

    # Initialize Redis connection for rate limiting (optional)
    redis_client = None
    try:
        import redis.asyncio as aioredis

        redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        app.state.redis_client = redis_client
        logger.info("redis_connected", url=settings.redis_url)
    except Exception as exc:
        logger.warning("redis_unavailable", error=str(exc), msg="Rate limiting will use in-memory fallback")
        app.state.redis_client = None

    logger.info("api_started", service=APP_NAME)

    yield

    # Shutdown
    logger.info("api_shutting_down", service=APP_NAME)
    if redis_client is not None:
        await redis_client.aclose()
    logger.info("api_shutdown_complete", service=APP_NAME)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=f"{APP_NAME} API",
        description="Fraud Analytics & Risk Intelligence Platform — Transaction Ingestion API",
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # --- Middleware (order matters: last added = first executed) ---

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get("api.cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Logging middleware
    from src.api.middleware.logging_middleware import LoggingMiddleware

    app.add_middleware(LoggingMiddleware)

    # Rate limiting middleware
    from src.api.middleware.rate_limiter import RateLimitMiddleware

    app.add_middleware(RateLimitMiddleware, redis_client=None)

    # --- Exception Handlers ---

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Return structured 422 errors with field-level detail."""
        errors = []
        for error in exc.errors():
            loc = " -> ".join(str(l) for l in error["loc"])
            errors.append({
                "field": loc,
                "message": error["msg"],
                "type": error["type"],
            })
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "detail": errors,
                "request_id": getattr(request.state, "correlation_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all handler to prevent leaking internal details."""
        logger.exception(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred. Please try again later.",
                "request_id": getattr(request.state, "correlation_id", None),
            },
        )

    # --- Routes ---

    from src.api.routes.health import router as health_router
    from src.api.routes.transactions import router as transactions_router

    app.include_router(health_router)
    app.include_router(transactions_router)

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": APP_NAME,
            "version": APP_VERSION,
            "docs": "/docs",
        }

    return app


# Application instance for uvicorn
app = create_app()
