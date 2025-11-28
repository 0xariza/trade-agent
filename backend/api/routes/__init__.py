"""
API Routes for Alpha Arena Trading Bot.

Provides REST API endpoints for:
- Health checks
- Trading status
- Performance metrics
- Position management
- Configuration
"""

from fastapi import APIRouter
from .health import router as health_router
from .trading import router as trading_router
from .metrics import router as metrics_router

# Main API router
api_router = APIRouter(prefix="/api/v1")

# Include sub-routers
api_router.include_router(health_router, tags=["Health"])
api_router.include_router(trading_router, tags=["Trading"])
api_router.include_router(metrics_router, tags=["Metrics"])

