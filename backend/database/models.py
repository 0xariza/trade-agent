from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Enum
from datetime import datetime
from .db import Base
import enum


class PositionStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED_STOP_LOSS = "closed_stop_loss"
    CLOSED_TAKE_PROFIT = "closed_take_profit"
    CLOSED_SIGNAL = "closed_signal"
    CLOSED_MANUAL = "closed_manual"


class Position(Base):
    """Tracks open and closed positions with stop-loss and take-profit."""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)  # 'long' or 'short'
    amount = Column(Float)
    entry_price = Column(Float)
    entry_time = Column(DateTime, default=datetime.utcnow)
    
    # Risk management
    stop_loss = Column(Float)
    take_profit = Column(Float)
    trailing_stop = Column(Float, nullable=True)  # Dynamic trailing stop
    max_hold_hours = Column(Integer, default=72)  # Time-based exit (default 3 days)
    
    # Exit info
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String, nullable=True)  # stop_loss, take_profit, signal, manual, time_stop
    
    # Performance
    pnl = Column(Float, nullable=True)  # Realized P&L in quote currency
    pnl_pct = Column(Float, nullable=True)  # Realized P&L percentage
    
    # Status
    status = Column(String, default="open")  # open, closed
    
    # Metadata
    entry_reasoning = Column(String, nullable=True)
    agent_name = Column(String, nullable=True)
    market_regime = Column(String, nullable=True)  # trending, ranging, volatile


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)  # 'buy' or 'sell'
    amount = Column(Float)
    price = Column(Float)
    cost = Column(Float, nullable=True)
    commission = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    strategy = Column(String, nullable=True)  # e.g., "RSI_Divergence"
    position_id = Column(Integer, nullable=True)  # Link to position


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    trend = Column(String)  # 'bullish', 'bearish'
    confidence = Column(String)
    reasoning = Column(String)
    raw_response = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)


class BotState(Base):
    """Stores bot state for persistence across restarts."""
    __tablename__ = "bot_state"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)  # e.g., 'balances', 'memory', 'config'
    value = Column(JSON)  # Serialized state data
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TradeHistory(Base):
    """Stores completed trades for memory system."""
    __tablename__ = "trade_history"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)  # 'long' or 'short'
    entry_price = Column(Float)
    exit_price = Column(Float)
    amount = Column(Float)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    exit_reason = Column(String)  # stop_loss, take_profit, signal, time_stop
    entry_reasoning = Column(String, nullable=True)
    market_regime = Column(String, nullable=True)
    hold_duration_hours = Column(Float)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime, default=datetime.utcnow)
    agent_name = Column(String, nullable=True)


class PerformanceStats(Base):
    """Stores performance statistics per symbol."""
    __tablename__ = "performance_stats"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    total_pnl_pct = Column(Float, default=0.0)
    avg_hold_hours = Column(Float, default=0.0)
    stop_loss_count = Column(Integer, default=0)
    take_profit_count = Column(Integer, default=0)
    signal_close_count = Column(Integer, default=0)
    best_trade_pnl = Column(Float, nullable=True)
    worst_trade_pnl = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
