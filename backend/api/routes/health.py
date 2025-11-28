"""
Health Check Endpoints for Alpha Arena.

Critical for:
- Monitoring system health
- Load balancer health checks
- Alerting integration
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict, Any
import logging
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter()

# Global references (set by app on startup)
_trading_scheduler = None
_paper_exchange = None
_market_data_provider = None


def set_dependencies(scheduler, exchange, market_data):
    """Set dependencies for health checks."""
    global _trading_scheduler, _paper_exchange, _market_data_provider
    _trading_scheduler = scheduler
    _paper_exchange = exchange
    _market_data_provider = market_data


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns 200 if service is running.
    Used by load balancers and monitoring.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "service": "alpha-arena-trading-bot"
    }


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with component status.
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check scheduler
    if _trading_scheduler:
        scheduler_running = getattr(_trading_scheduler.scheduler, 'running', False)
        health["components"]["scheduler"] = {
            "status": "healthy" if scheduler_running else "degraded",
            "running": scheduler_running,
            "cycle_count": getattr(_trading_scheduler, 'cycle_count', 0)
        }
    else:
        health["components"]["scheduler"] = {"status": "not_initialized"}
    
    # Check exchange
    if _paper_exchange:
        try:
            balance = _paper_exchange.get_balance("USDT")
            positions = len(_paper_exchange.get_all_positions())
            health["components"]["exchange"] = {
                "status": "healthy",
                "usdt_balance": round(balance, 2),
                "open_positions": positions
            }
        except Exception as e:
            health["components"]["exchange"] = {
                "status": "error",
                "error": str(e)
            }
    else:
        health["components"]["exchange"] = {"status": "not_initialized"}
    
    # Check market data provider
    if _market_data_provider:
        try:
            # Try to fetch a simple price
            ohlcv = await _market_data_provider.get_ohlcv("BTC/USDT", limit=1)
            if not ohlcv.empty:
                health["components"]["market_data"] = {
                    "status": "healthy",
                    "btc_price": round(ohlcv.iloc[-1]['close'], 2)
                }
            else:
                health["components"]["market_data"] = {"status": "no_data"}
        except Exception as e:
            health["components"]["market_data"] = {
                "status": "error",
                "error": str(e)
            }
    else:
        health["components"]["market_data"] = {"status": "not_initialized"}
    
    # Determine overall status
    statuses = [c.get("status") for c in health["components"].values()]
    if "error" in statuses:
        health["status"] = "unhealthy"
    elif "degraded" in statuses or "not_initialized" in statuses:
        health["status"] = "degraded"
    
    return health


@router.get("/health/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness probe for Kubernetes.
    
    Returns 200 only if all critical components are ready.
    """
    checks = {
        "scheduler_ready": _trading_scheduler is not None,
        "exchange_ready": _paper_exchange is not None,
        "market_data_ready": _market_data_provider is not None
    }
    
    all_ready = all(checks.values())
    
    if not all_ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "checks": checks}
        )
    
    return {
        "status": "ready",
        "checks": checks
    }


@router.get("/health/live")
async def liveness_check() -> Dict[str, str]:
    """
    Liveness probe for Kubernetes.
    
    Simple check that the service is alive.
    """
    return {"status": "alive"}

