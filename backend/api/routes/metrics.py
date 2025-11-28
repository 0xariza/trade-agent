"""
Metrics API Endpoints for Alpha Arena.

Provides performance analytics and Prometheus metrics export.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics")

# Global references
_trading_scheduler = None
_paper_exchange = None
_performance_analyzer = None


def set_dependencies(scheduler, exchange, analyzer=None):
    """Set dependencies."""
    global _trading_scheduler, _paper_exchange, _performance_analyzer
    _trading_scheduler = scheduler
    _paper_exchange = exchange
    _performance_analyzer = analyzer


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get comprehensive performance metrics.
    """
    if not _paper_exchange:
        raise HTTPException(status_code=503, detail="Exchange not initialized")
    
    # Use performance analyzer if available
    if _performance_analyzer:
        trades = _paper_exchange.closed_positions
        _performance_analyzer.add_trades(trades)
        _performance_analyzer.set_equity_curve(_paper_exchange.trade_history)
        
        metrics = _performance_analyzer.calculate_metrics()
        return metrics.to_dict()
    
    # Fallback to basic stats
    return _paper_exchange.get_performance_summary()


@router.get("/performance/by-symbol")
async def get_performance_by_symbol() -> Dict[str, Any]:
    """
    Get performance metrics grouped by symbol.
    """
    if not _paper_exchange:
        raise HTTPException(status_code=503, detail="Exchange not initialized")
    
    if not _performance_analyzer:
        raise HTTPException(status_code=503, detail="Performance analyzer not initialized")
    
    trades = _paper_exchange.closed_positions
    _performance_analyzer.add_trades(trades)
    
    by_symbol = _performance_analyzer.analyze_by_symbol()
    
    return {
        symbol: metrics.to_dict()
        for symbol, metrics in by_symbol.items()
    }


@router.get("/performance/losing-patterns")
async def get_losing_patterns() -> Dict[str, Any]:
    """
    Analyze patterns in losing trades.
    
    Critical for improving strategy.
    """
    if not _paper_exchange:
        raise HTTPException(status_code=503, detail="Exchange not initialized")
    
    if not _performance_analyzer:
        raise HTTPException(status_code=503, detail="Performance analyzer not initialized")
    
    trades = _paper_exchange.closed_positions
    _performance_analyzer.add_trades(trades)
    
    return _performance_analyzer.get_losing_patterns()


@router.get("/prometheus", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    """
    Export metrics in Prometheus format.
    
    Usage:
        Scrape this endpoint from Prometheus to collect metrics.
    """
    lines = []
    
    # Type declarations
    lines.append("# HELP alpha_arena_portfolio_value_usd Current portfolio value in USD")
    lines.append("# TYPE alpha_arena_portfolio_value_usd gauge")
    
    lines.append("# HELP alpha_arena_open_positions Number of open positions")
    lines.append("# TYPE alpha_arena_open_positions gauge")
    
    lines.append("# HELP alpha_arena_total_trades_count Total number of trades executed")
    lines.append("# TYPE alpha_arena_total_trades_count counter")
    
    lines.append("# HELP alpha_arena_win_rate_percent Win rate percentage")
    lines.append("# TYPE alpha_arena_win_rate_percent gauge")
    
    lines.append("# HELP alpha_arena_drawdown_percent Current drawdown percentage")
    lines.append("# TYPE alpha_arena_drawdown_percent gauge")
    
    lines.append("# HELP alpha_arena_total_pnl_usd Total profit/loss in USD")
    lines.append("# TYPE alpha_arena_total_pnl_usd gauge")
    
    # Values
    if _paper_exchange:
        balance = _paper_exchange.get_balance("USDT")
        positions = len(_paper_exchange.get_all_positions())
        perf = _paper_exchange.get_performance_summary()
        
        lines.append(f"alpha_arena_portfolio_value_usd {balance:.2f}")
        lines.append(f"alpha_arena_open_positions {positions}")
        lines.append(f"alpha_arena_total_trades_count {perf.get('total_trades', 0)}")
        lines.append(f"alpha_arena_win_rate_percent {perf.get('win_rate', 0):.1f}")
        lines.append(f"alpha_arena_total_pnl_usd {perf.get('total_pnl', 0):.2f}")
    
    if _trading_scheduler:
        dd_status = _trading_scheduler.drawdown_manager.get_status_summary()
        lines.append(f"alpha_arena_drawdown_percent {dd_status.get('current_drawdown_pct', 0):.2f}")
        
        # Trading mode as label
        mode = dd_status.get('trading_mode', 'unknown')
        lines.append(f'alpha_arena_trading_mode{{mode="{mode}"}} 1')
    
    return "\n".join(lines)


@router.get("/summary")
async def get_summary() -> Dict[str, Any]:
    """
    Get a quick summary for dashboard display.
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "portfolio": {},
        "trading": {},
        "performance": {}
    }
    
    if _paper_exchange:
        balance = _paper_exchange.get_balance("USDT")
        positions = _paper_exchange.get_all_positions()
        perf = _paper_exchange.get_performance_summary()
        
        summary["portfolio"] = {
            "usdt_balance": round(balance, 2),
            "open_positions": len(positions),
            "initial_balance": _paper_exchange.initial_balance
        }
        
        summary["performance"] = {
            "total_trades": perf.get('total_trades', 0),
            "win_rate": round(perf.get('win_rate', 0), 1),
            "total_pnl": round(perf.get('total_pnl', 0), 2)
        }
    
    if _trading_scheduler:
        dd = _trading_scheduler.drawdown_manager.get_status_summary()
        
        summary["trading"] = {
            "mode": dd.get('trading_mode', 'unknown'),
            "position_multiplier": dd.get('position_multiplier', 1.0),
            "current_drawdown_pct": round(dd.get('current_drawdown_pct', 0), 2),
            "consecutive_losses": dd.get('consecutive_losses', 0),
            "cycle_count": _trading_scheduler.cycle_count
        }
    
    return summary

