"""
Partial Exit Manager for Alpha Arena.

Implements sophisticated exit strategies:
1. Scale-out at 1R (50% position)
2. Move stop to breakeven after 1R
3. Trail remaining position
4. Dynamic take-profit adjustment

Week 9-10 Implementation.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ExitStage(Enum):
    """Stages of partial exit strategy."""
    INITIAL = "initial"      # Full position, original stops
    FIRST_TARGET = "1r"      # Hit 1R, took partial, moved stop to BE
    TRAILING = "trailing"    # Past 1.5R, trailing stop active
    FINAL = "final"          # Position fully closed


@dataclass
class PartialExitState:
    """State of partial exits for a position."""
    symbol: str
    entry_price: float
    initial_amount: float
    current_amount: float
    stage: ExitStage
    original_stop_loss: float
    current_stop_loss: float
    original_take_profit: float
    breakeven_price: float
    
    # Partial exit history
    exits: List[Dict[str, Any]] = field(default_factory=list)
    
    # P&L tracking
    realized_pnl: float = 0.0
    highest_price_seen: float = 0.0
    lowest_price_seen: float = float('inf')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "stage": self.stage.value,
            "initial_amount": self.initial_amount,
            "current_amount": self.current_amount,
            "remaining_pct": (self.current_amount / self.initial_amount) * 100,
            "realized_pnl": self.realized_pnl,
            "exits_count": len(self.exits),
            "current_stop_loss": self.current_stop_loss
        }


class PartialExitManager:
    """
    Manages partial profit-taking strategies.
    
    Strategy:
    1. Entry: Full position with stop at 2 ATR
    2. At 1R: Exit 50%, move stop to breakeven
    3. At 1.5R: Start trailing stop
    4. Trail until stopped out or TP hit
    
    This locks in profits while allowing winners to run.
    """
    
    def __init__(
        self,
        exit_at_1r_pct: float = 0.50,        # Exit 50% at 1R
        move_to_breakeven_at_1r: bool = True,
        start_trailing_at_r: float = 1.5,    # Start trailing at 1.5R
        trailing_atr_multiplier: float = 1.0,
        min_trail_distance_pct: float = 0.005  # 0.5% minimum trail distance
    ):
        """
        Args:
            exit_at_1r_pct: Percentage of position to exit at 1R
            move_to_breakeven_at_1r: Move stop to breakeven after 1R exit
            start_trailing_at_r: R-multiple to start trailing
            trailing_atr_multiplier: ATR multiplier for trail distance
            min_trail_distance_pct: Minimum trailing distance as % of price
        """
        self.exit_at_1r_pct = exit_at_1r_pct
        self.move_to_breakeven_at_1r = move_to_breakeven_at_1r
        self.start_trailing_at_r = start_trailing_at_r
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.min_trail_distance_pct = min_trail_distance_pct
        
        # Track state for each position
        self.states: Dict[str, PartialExitState] = {}
        
        logger.info(
            f"PartialExitManager initialized. "
            f"Exit {exit_at_1r_pct*100}% at 1R, Trail at {start_trailing_at_r}R"
        )
    
    def register_position(
        self,
        symbol: str,
        entry_price: float,
        amount: float,
        stop_loss: float,
        take_profit: float,
        side: str = "long"
    ) -> PartialExitState:
        """
        Register a new position for partial exit management.
        """
        # Calculate breakeven (entry + fees approximation)
        fee_estimate = entry_price * 0.002  # 0.2% round trip
        breakeven = entry_price + fee_estimate if side == "long" else entry_price - fee_estimate
        
        state = PartialExitState(
            symbol=symbol,
            entry_price=entry_price,
            initial_amount=amount,
            current_amount=amount,
            stage=ExitStage.INITIAL,
            original_stop_loss=stop_loss,
            current_stop_loss=stop_loss,
            original_take_profit=take_profit,
            breakeven_price=breakeven,
            highest_price_seen=entry_price,
            lowest_price_seen=entry_price
        )
        
        self.states[symbol] = state
        
        logger.info(
            f"Registered {symbol} for partial exits. "
            f"Entry: ${entry_price:,.2f}, Amount: {amount:.4f}"
        )
        
        return state
    
    def check_exit_triggers(
        self,
        symbol: str,
        current_price: float,
        atr: float,
        side: str = "long"
    ) -> Tuple[Optional[str], float, float]:
        """
        Check if any partial exit should be triggered.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            atr: Current ATR for trailing
            side: 'long' or 'short'
            
        Returns:
            Tuple of (exit_reason, exit_amount, new_stop_loss)
            exit_reason is None if no exit triggered
        """
        if symbol not in self.states:
            return None, 0, 0
        
        state = self.states[symbol]
        
        # Update price tracking
        if side == "long":
            state.highest_price_seen = max(state.highest_price_seen, current_price)
        else:
            state.lowest_price_seen = min(state.lowest_price_seen, current_price)
        
        # Calculate R-multiple
        risk_per_unit = abs(state.entry_price - state.original_stop_loss)
        if risk_per_unit <= 0:
            return None, 0, state.current_stop_loss
        
        if side == "long":
            current_r = (current_price - state.entry_price) / risk_per_unit
        else:
            current_r = (state.entry_price - current_price) / risk_per_unit
        
        exit_reason = None
        exit_amount = 0
        new_stop = state.current_stop_loss
        
        # Stage: INITIAL -> Check for 1R exit
        if state.stage == ExitStage.INITIAL:
            if current_r >= 1.0:
                # Hit 1R! Take partial profit
                exit_amount = state.current_amount * self.exit_at_1r_pct
                exit_reason = "partial_1r"
                
                # Move stop to breakeven
                if self.move_to_breakeven_at_1r:
                    new_stop = state.breakeven_price
                
                # Update state
                state.current_amount -= exit_amount
                state.current_stop_loss = new_stop
                state.stage = ExitStage.FIRST_TARGET
                state.exits.append({
                    "reason": "1r_target",
                    "price": current_price,
                    "amount": exit_amount,
                    "timestamp": datetime.now().isoformat(),
                    "r_multiple": current_r
                })
                
                pnl = (current_price - state.entry_price) * exit_amount if side == "long" else \
                      (state.entry_price - current_price) * exit_amount
                state.realized_pnl += pnl
                
                logger.info(
                    f"🎯 {symbol} hit 1R! Exiting {exit_amount:.4f} ({self.exit_at_1r_pct*100}%). "
                    f"P&L: ${pnl:,.2f}. Stop moved to breakeven: ${new_stop:,.2f}"
                )
        
        # Stage: FIRST_TARGET -> Check for trailing start
        elif state.stage == ExitStage.FIRST_TARGET:
            if current_r >= self.start_trailing_at_r:
                state.stage = ExitStage.TRAILING
                logger.info(f"📈 {symbol} at {current_r:.1f}R. Starting trailing stop.")
        
        # Stage: TRAILING -> Update trailing stop
        if state.stage == ExitStage.TRAILING:
            trail_distance = max(
                atr * self.trailing_atr_multiplier,
                current_price * self.min_trail_distance_pct
            )
            
            if side == "long":
                potential_stop = state.highest_price_seen - trail_distance
                if potential_stop > state.current_stop_loss:
                    new_stop = potential_stop
                    state.current_stop_loss = new_stop
                    logger.debug(f"{symbol} trailing stop updated to ${new_stop:,.2f}")
            else:
                potential_stop = state.lowest_price_seen + trail_distance
                if potential_stop < state.current_stop_loss:
                    new_stop = potential_stop
                    state.current_stop_loss = new_stop
        
        return exit_reason, exit_amount, new_stop
    
    def record_full_exit(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        side: str = "long"
    ) -> Dict[str, Any]:
        """
        Record when a position is fully closed.
        
        Returns summary of all partial exits.
        """
        if symbol not in self.states:
            return {}
        
        state = self.states[symbol]
        
        # Calculate final P&L
        if side == "long":
            final_pnl = (exit_price - state.entry_price) * state.current_amount
        else:
            final_pnl = (state.entry_price - exit_price) * state.current_amount
        
        state.realized_pnl += final_pnl
        state.stage = ExitStage.FINAL
        
        state.exits.append({
            "reason": exit_reason,
            "price": exit_price,
            "amount": state.current_amount,
            "timestamp": datetime.now().isoformat(),
            "final": True
        })
        
        summary = {
            "symbol": symbol,
            "entry_price": state.entry_price,
            "initial_amount": state.initial_amount,
            "total_realized_pnl": state.realized_pnl,
            "exits": state.exits,
            "highest_price_seen": state.highest_price_seen,
            "lowest_price_seen": state.lowest_price_seen,
            "final_exit_reason": exit_reason
        }
        
        # Clean up
        del self.states[symbol]
        
        logger.info(
            f"Position {symbol} fully closed. "
            f"Total P&L: ${state.realized_pnl:,.2f} across {len(state.exits)} exits"
        )
        
        return summary
    
    def get_position_state(self, symbol: str) -> Optional[PartialExitState]:
        """Get current state for a position."""
        return self.states.get(symbol)
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all position states."""
        return {
            symbol: state.to_dict()
            for symbol, state in self.states.items()
        }
    
    def should_take_partial(self, symbol: str) -> bool:
        """Check if position is ready for partial exit."""
        if symbol not in self.states:
            return False
        
        state = self.states[symbol]
        return state.stage == ExitStage.INITIAL
    
    def get_adjusted_stop(self, symbol: str) -> Optional[float]:
        """Get the current adjusted stop-loss for a position."""
        if symbol not in self.states:
            return None
        
        return self.states[symbol].current_stop_loss
    
    def clear_position(self, symbol: str):
        """Remove position tracking (e.g., if closed elsewhere)."""
        if symbol in self.states:
            del self.states[symbol]
            logger.info(f"Cleared partial exit tracking for {symbol}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics on partial exits."""
        total_positions = len(self.states)
        
        stage_counts = {}
        for state in self.states.values():
            stage = state.stage.value
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        total_realized = sum(s.realized_pnl for s in self.states.values())
        
        return {
            "active_positions": total_positions,
            "by_stage": stage_counts,
            "total_realized_pnl_open": total_realized
        }

