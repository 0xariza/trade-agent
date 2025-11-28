"""
State Manager - Handles persistence of bot state across restarts.

Saves and restores:
- Open positions
- Account balances
- Agent memory (trades, performance, lessons)
- Configuration
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert

from .db import get_db, AsyncSessionLocal
from .models import BotState, Position, TradeHistory, PerformanceStats

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages persistent state for the trading bot.
    
    Ensures no data is lost on restart by saving critical state to database.
    """
    
    def __init__(self):
        self.last_save_time: Optional[datetime] = None
    
    # ==================== GENERIC STATE ====================
    
    async def save_state(self, key: str, value: Any) -> bool:
        """Save a generic state value."""
        try:
            async with get_db() as db:
                # Upsert: insert or update
                stmt = insert(BotState).values(
                    key=key,
                    value=value,
                    updated_at=datetime.utcnow()
                ).on_conflict_do_update(
                    index_elements=['key'],
                    set_={'value': value, 'updated_at': datetime.utcnow()}
                )
                await db.execute(stmt)
                await db.commit()
                logger.debug(f"State saved: {key}")
                return True
        except Exception as e:
            logger.error(f"Failed to save state '{key}': {e}")
            return False
    
    async def load_state(self, key: str, default: Any = None) -> Any:
        """Load a generic state value."""
        try:
            async with get_db() as db:
                result = await db.execute(
                    select(BotState).where(BotState.key == key)
                )
                row = result.scalar_one_or_none()
                if row:
                    return row.value
                return default
        except Exception as e:
            logger.error(f"Failed to load state '{key}': {e}")
            return default
    
    # ==================== BALANCES ====================
    
    async def save_balances(self, balances: Dict[str, float]) -> bool:
        """Save account balances."""
        return await self.save_state('balances', balances)
    
    async def load_balances(self, default_balance: float = 10000.0) -> Dict[str, float]:
        """Load account balances."""
        balances = await self.load_state('balances')
        if balances:
            return balances
        return {"USDT": default_balance}
    
    # ==================== POSITIONS ====================
    
    async def save_positions(self, positions: Dict[str, Dict[str, Any]]) -> bool:
        """Save open positions to database."""
        try:
            async with get_db() as db:
                # First, mark all existing positions as closed (in case they were closed)
                # We'll update/insert the current open ones
                
                for symbol, pos_data in positions.items():
                    # Check if position exists
                    result = await db.execute(
                        select(Position).where(
                            Position.symbol == symbol,
                            Position.status == "open"
                        )
                    )
                    existing = result.scalar_one_or_none()
                    
                    if existing:
                        # Update existing
                        existing.amount = pos_data['amount']
                        existing.stop_loss = pos_data['stop_loss']
                        existing.take_profit = pos_data['take_profit']
                        existing.trailing_stop = pos_data.get('trailing_stop')
                    else:
                        # Insert new
                        new_pos = Position(
                            symbol=symbol,
                            side=pos_data['side'],
                            amount=pos_data['amount'],
                            entry_price=pos_data['entry_price'],
                            entry_time=datetime.fromisoformat(pos_data['entry_time']) if isinstance(pos_data['entry_time'], str) else pos_data['entry_time'],
                            stop_loss=pos_data['stop_loss'],
                            take_profit=pos_data['take_profit'],
                            trailing_stop=pos_data.get('trailing_stop'),
                            max_hold_hours=pos_data.get('max_hold_hours', 72),
                            entry_reasoning=pos_data.get('entry_reasoning', ''),
                            agent_name=pos_data.get('agent_name', ''),
                            market_regime=pos_data.get('market_regime', ''),
                            status="open"
                        )
                        db.add(new_pos)
                
                await db.commit()
                logger.info(f"Saved {len(positions)} open positions")
                return True
        except Exception as e:
            logger.error(f"Failed to save positions: {e}")
            return False
    
    async def load_positions(self) -> Dict[str, Dict[str, Any]]:
        """Load open positions from database."""
        try:
            async with get_db() as db:
                result = await db.execute(
                    select(Position).where(Position.status == "open")
                )
                rows = result.scalars().all()
                
                positions = {}
                for row in rows:
                    positions[row.symbol] = {
                        'symbol': row.symbol,
                        'side': row.side,
                        'amount': row.amount,
                        'entry_price': row.entry_price,
                        'entry_time': row.entry_time.isoformat(),
                        'stop_loss': row.stop_loss,
                        'take_profit': row.take_profit,
                        'trailing_stop': row.trailing_stop,
                        'max_hold_hours': row.max_hold_hours,
                        'entry_reasoning': row.entry_reasoning,
                        'agent_name': row.agent_name,
                        'market_regime': row.market_regime
                    }
                
                logger.info(f"Loaded {len(positions)} open positions from database")
                return positions
        except Exception as e:
            logger.error(f"Failed to load positions: {e}")
            return {}
    
    async def close_position_in_db(self, symbol: str, exit_price: float, exit_reason: str, pnl: float, pnl_pct: float) -> bool:
        """Mark a position as closed in database."""
        try:
            async with get_db() as db:
                result = await db.execute(
                    select(Position).where(
                        Position.symbol == symbol,
                        Position.status == "open"
                    )
                )
                pos = result.scalar_one_or_none()
                
                if pos:
                    pos.status = "closed"
                    pos.exit_price = exit_price
                    pos.exit_time = datetime.utcnow()
                    pos.exit_reason = exit_reason
                    pos.pnl = pnl
                    pos.pnl_pct = pnl_pct
                    await db.commit()
                    logger.info(f"Position {symbol} marked as closed in DB")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to close position in DB: {e}")
            return False
    
    # ==================== TRADE HISTORY ====================
    
    async def save_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Save a completed trade to history."""
        try:
            async with get_db() as db:
                trade = TradeHistory(
                    symbol=trade_data['symbol'],
                    side=trade_data['side'],
                    entry_price=trade_data['entry_price'],
                    exit_price=trade_data['exit_price'],
                    amount=trade_data.get('amount', 0),
                    pnl=trade_data['pnl'],
                    pnl_pct=trade_data['pnl_pct'],
                    exit_reason=trade_data['exit_reason'],
                    entry_reasoning=trade_data.get('entry_reasoning', ''),
                    market_regime=trade_data.get('market_regime', ''),
                    hold_duration_hours=trade_data.get('hold_duration_hours', 0),
                    entry_time=datetime.fromisoformat(trade_data['entry_time']) if isinstance(trade_data.get('entry_time'), str) else datetime.utcnow(),
                    exit_time=datetime.fromisoformat(trade_data['exit_time']) if isinstance(trade_data.get('exit_time'), str) else datetime.utcnow(),
                    agent_name=trade_data.get('agent_name', '')
                )
                db.add(trade)
                await db.commit()
                logger.debug(f"Trade saved to history: {trade_data['symbol']}")
                return True
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
            return False
    
    async def load_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Load recent trade history."""
        try:
            async with get_db() as db:
                result = await db.execute(
                    select(TradeHistory)
                    .order_by(TradeHistory.exit_time.desc())
                    .limit(limit)
                )
                rows = result.scalars().all()
                
                trades = []
                for row in rows:
                    trades.append({
                        'symbol': row.symbol,
                        'side': row.side,
                        'entry_price': row.entry_price,
                        'exit_price': row.exit_price,
                        'amount': row.amount,
                        'pnl': row.pnl,
                        'pnl_pct': row.pnl_pct,
                        'exit_reason': row.exit_reason,
                        'entry_reasoning': row.entry_reasoning,
                        'market_regime': row.market_regime,
                        'hold_duration_hours': row.hold_duration_hours,
                        'entry_time': row.entry_time.isoformat(),
                        'exit_time': row.exit_time.isoformat()
                    })
                
                logger.info(f"Loaded {len(trades)} trades from history")
                return trades
        except Exception as e:
            logger.error(f"Failed to load trade history: {e}")
            return []
    
    # ==================== PERFORMANCE STATS ====================
    
    async def save_performance_stats(self, stats: Dict[str, Dict[str, Any]]) -> bool:
        """Save performance statistics per symbol."""
        try:
            async with get_db() as db:
                for symbol, stat_data in stats.items():
                    # Upsert
                    stmt = insert(PerformanceStats).values(
                        symbol=symbol,
                        total_trades=stat_data.get('total_trades', 0),
                        winning_trades=stat_data.get('winning_trades', 0),
                        losing_trades=stat_data.get('losing_trades', 0),
                        total_pnl=stat_data.get('total_pnl', 0.0),
                        total_pnl_pct=stat_data.get('total_pnl_pct', 0.0),
                        avg_hold_hours=stat_data.get('avg_hold_hours', 0.0),
                        stop_loss_count=stat_data.get('stop_loss_count', 0),
                        take_profit_count=stat_data.get('take_profit_count', 0),
                        signal_close_count=stat_data.get('signal_close_count', 0),
                        best_trade_pnl=stat_data.get('best_trade', {}).get('pnl') if stat_data.get('best_trade') else None,
                        worst_trade_pnl=stat_data.get('worst_trade', {}).get('pnl') if stat_data.get('worst_trade') else None,
                        updated_at=datetime.utcnow()
                    ).on_conflict_do_update(
                        index_elements=['symbol'],
                        set_={
                            'total_trades': stat_data.get('total_trades', 0),
                            'winning_trades': stat_data.get('winning_trades', 0),
                            'losing_trades': stat_data.get('losing_trades', 0),
                            'total_pnl': stat_data.get('total_pnl', 0.0),
                            'total_pnl_pct': stat_data.get('total_pnl_pct', 0.0),
                            'avg_hold_hours': stat_data.get('avg_hold_hours', 0.0),
                            'stop_loss_count': stat_data.get('stop_loss_count', 0),
                            'take_profit_count': stat_data.get('take_profit_count', 0),
                            'signal_close_count': stat_data.get('signal_close_count', 0),
                            'best_trade_pnl': stat_data.get('best_trade', {}).get('pnl') if stat_data.get('best_trade') else None,
                            'worst_trade_pnl': stat_data.get('worst_trade', {}).get('pnl') if stat_data.get('worst_trade') else None,
                            'updated_at': datetime.utcnow()
                        }
                    )
                    await db.execute(stmt)
                
                await db.commit()
                logger.info(f"Saved performance stats for {len(stats)} symbols")
                return True
        except Exception as e:
            logger.error(f"Failed to save performance stats: {e}")
            return False
    
    async def load_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Load performance statistics."""
        try:
            async with get_db() as db:
                result = await db.execute(select(PerformanceStats))
                rows = result.scalars().all()
                
                stats = {}
                for row in rows:
                    stats[row.symbol] = {
                        'total_trades': row.total_trades,
                        'winning_trades': row.winning_trades,
                        'losing_trades': row.losing_trades,
                        'total_pnl': row.total_pnl,
                        'total_pnl_pct': row.total_pnl_pct,
                        'avg_hold_hours': row.avg_hold_hours,
                        'stop_loss_count': row.stop_loss_count,
                        'take_profit_count': row.take_profit_count,
                        'signal_close_count': row.signal_close_count,
                        'best_trade': {'pnl': row.best_trade_pnl} if row.best_trade_pnl else None,
                        'worst_trade': {'pnl': row.worst_trade_pnl} if row.worst_trade_pnl else None
                    }
                
                logger.info(f"Loaded performance stats for {len(stats)} symbols")
                return stats
        except Exception as e:
            logger.error(f"Failed to load performance stats: {e}")
            return {}
    
    # ==================== AGENT MEMORY ====================
    
    async def save_memory(self, memory_data: Dict[str, Any]) -> bool:
        """Save agent memory (lessons, etc.)."""
        return await self.save_state('agent_memory', memory_data)
    
    async def load_memory(self) -> Dict[str, Any]:
        """Load agent memory."""
        return await self.load_state('agent_memory', {})
    
    # ==================== DAILY STATS ====================
    
    async def save_daily_stats(self, stats: Dict[str, Any]) -> bool:
        """Save daily trading statistics."""
        return await self.save_state('daily_stats', stats)
    
    async def load_daily_stats(self) -> Dict[str, Any]:
        """Load daily trading statistics."""
        return await self.load_state('daily_stats', {
            'start_balance': 0,
            'trades_count': 0,
            'reset_date': datetime.now().date().isoformat()
        })
    
    # ==================== FULL STATE ====================
    
    async def save_full_state(
        self,
        balances: Dict[str, float],
        positions: Dict[str, Dict[str, Any]],
        memory_data: Dict[str, Any],
        daily_stats: Dict[str, Any]
    ) -> bool:
        """Save complete bot state."""
        success = True
        success &= await self.save_balances(balances)
        success &= await self.save_positions(positions)
        success &= await self.save_memory(memory_data)
        success &= await self.save_daily_stats(daily_stats)
        
        if memory_data.get('performance'):
            success &= await self.save_performance_stats(memory_data['performance'])
        
        self.last_save_time = datetime.utcnow()
        
        if success:
            logger.info("✅ Full state saved successfully")
        else:
            logger.warning("⚠️ Some state failed to save")
        
        return success
    
    async def load_full_state(self) -> Dict[str, Any]:
        """Load complete bot state."""
        balances = await self.load_balances()
        positions = await self.load_positions()
        memory = await self.load_memory()
        daily_stats = await self.load_daily_stats()
        performance = await self.load_performance_stats()
        trade_history = await self.load_trade_history(limit=100)
        
        logger.info("📂 Full state loaded from database")
        
        return {
            'balances': balances,
            'positions': positions,
            'memory': memory,
            'daily_stats': daily_stats,
            'performance': performance,
            'trade_history': trade_history
        }


# Singleton instance
state_manager = StateManager()


