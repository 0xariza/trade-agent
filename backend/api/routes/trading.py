"""
Trading API Endpoints for Alpha Arena.

Provides endpoints for:
- Current positions
- Trade history
- Portfolio status
- Manual overrides (with caution)
"""

from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trading")

# Global references
_trading_scheduler = None
_paper_exchange = None


def set_dependencies(scheduler, exchange):
    """Set dependencies."""
    global _trading_scheduler, _paper_exchange
    _trading_scheduler = scheduler
    _paper_exchange = exchange


class ClosePositionRequest(BaseModel):
    """Request to manually close a position."""
    symbol: str
    reason: str = "manual"


@router.get("/status")
async def get_trading_status() -> Dict[str, Any]:
    """
    Get current trading status.
    """
    if not _trading_scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
    
    return {
        "is_running": _trading_scheduler.is_running,
        "cycle_count": _trading_scheduler.cycle_count,
        "stop_check_count": _trading_scheduler.stop_check_count,
        "daily_trades": _trading_scheduler.daily_trades_count,
        "trading_mode": _trading_scheduler.drawdown_manager.trading_mode.value,
        "position_multiplier": _trading_scheduler.drawdown_manager.position_size_multiplier,
        "symbols": _trading_scheduler.symbols
    }


@router.get("/positions")
async def get_positions() -> Dict[str, Any]:
    """
    Get all open positions.
    """
    if not _paper_exchange:
        raise HTTPException(status_code=503, detail="Exchange not initialized")
    
    positions = _paper_exchange.get_all_positions()
    
    return {
        "count": len(positions),
        "positions": {
            symbol: {
                "side": pos.side,
                "amount": pos.amount,
                "entry_price": pos.entry_price,
                "entry_time": pos.entry_time.isoformat(),
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "entry_reasoning": pos.entry_reasoning[:200] if pos.entry_reasoning else ""
            }
            for symbol, pos in positions.items()
        }
    }


@router.get("/positions/{symbol}")
async def get_position(symbol: str) -> Dict[str, Any]:
    """
    Get a specific position.
    """
    if not _paper_exchange:
        raise HTTPException(status_code=503, detail="Exchange not initialized")
    
    # URL decode the symbol
    symbol = symbol.replace("-", "/")
    
    position = _paper_exchange.get_position(symbol)
    
    if not position:
        raise HTTPException(status_code=404, detail=f"No position found for {symbol}")
    
    return {
        "symbol": position.symbol,
        "side": position.side,
        "amount": position.amount,
        "entry_price": position.entry_price,
        "entry_time": position.entry_time.isoformat(),
        "stop_loss": position.stop_loss,
        "take_profit": position.take_profit,
        "trailing_stop": position.trailing_stop,
        "entry_reasoning": position.entry_reasoning,
        "agent_name": position.agent_name,
        "market_regime": position.market_regime,
        "highest_price": position.highest_price,
        "lowest_price": position.lowest_price
    }


@router.get("/portfolio")
async def get_portfolio() -> Dict[str, Any]:
    """
    Get portfolio overview.
    """
    if not _paper_exchange:
        raise HTTPException(status_code=503, detail="Exchange not initialized")
    
    balance = _paper_exchange.get_balance("USDT")
    positions = _paper_exchange.get_all_positions()
    performance = _paper_exchange.get_performance_summary()
    
    return {
        "usdt_balance": round(balance, 2),
        "open_positions": len(positions),
        "initial_balance": _paper_exchange.initial_balance,
        "performance": performance
    }


@router.get("/history")
async def get_trade_history(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(50, ge=1, le=500, description="Max trades to return"),
    exit_reason: Optional[str] = Query(None, description="Filter by exit reason")
) -> Dict[str, Any]:
    """
    Get trade history.
    """
    if not _paper_exchange:
        raise HTTPException(status_code=503, detail="Exchange not initialized")
    
    trades = _paper_exchange.closed_positions.copy()
    
    # Filters
    if symbol:
        symbol = symbol.replace("-", "/")
        trades = [t for t in trades if t.get('symbol') == symbol]
    
    if exit_reason:
        trades = [t for t in trades if t.get('exit_reason') == exit_reason]
    
    # Limit
    trades = trades[-limit:]
    
    return {
        "count": len(trades),
        "trades": trades
    }


@router.get("/drawdown")
async def get_drawdown_status() -> Dict[str, Any]:
    """
    Get drawdown manager status.
    """
    if not _trading_scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
    
    return _trading_scheduler.drawdown_manager.get_status_summary()


@router.post("/positions/close")
async def close_position(request: ClosePositionRequest) -> Dict[str, Any]:
    """
    Manually close a position.
    
    ⚠️ Use with caution - this bypasses normal exit logic.
    """
    if not _paper_exchange:
        raise HTTPException(status_code=503, detail="Exchange not initialized")
    
    symbol = request.symbol.replace("-", "/")
    position = _paper_exchange.get_position(symbol)
    
    if not position:
        raise HTTPException(status_code=404, detail=f"No position found for {symbol}")
    
    # Get current price (simplified - would normally fetch from market)
    # In production, this should fetch the actual current price
    current_price = position.entry_price  # Placeholder
    
    logger.warning(f"Manual position close requested for {symbol}")
    
    result = _paper_exchange.close_position(
        symbol=symbol,
        exit_price=current_price,
        reason=f"manual_{request.reason}"
    )
    
    if result:
        return {
            "status": "closed",
            "symbol": symbol,
            "exit_price": result['exit_price'],
            "pnl": result['pnl'],
            "pnl_pct": result['pnl_pct']
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to close position")


@router.get("/memory")
async def get_agent_memory() -> Dict[str, Any]:
    """
    Get agent memory summary.
    """
    if not _trading_scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
    
    memory = _trading_scheduler.memory
    
    return {
        "total_trades_in_memory": len(memory.trades),
        "open_positions_tracked": len(memory.open_positions),
        "performance_by_symbol": memory.performance,
        "lessons_learned": memory.lessons[-10:],  # Last 10
        "win_rate_overall": memory.get_win_rate()
    }

