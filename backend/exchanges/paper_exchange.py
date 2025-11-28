import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from backend.database.db import get_db
from backend.database.models import Trade, Position as DBPosition

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open trading position with risk management."""
    symbol: str
    side: str  # 'long' or 'short'
    amount: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    max_hold_hours: int = 72  # Default 3 days
    entry_reasoning: str = ""
    agent_name: str = ""
    market_regime: str = ""
    highest_price: float = field(default=0.0)  # For trailing stop
    lowest_price: float = field(default=float('inf'))  # For trailing stop (shorts)
    
    def __post_init__(self):
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L (needs current price)."""
        return 0.0  # Will be calculated with current price
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L given current price."""
        if self.side == 'long':
            return (current_price - self.entry_price) * self.amount
        else:  # short
            return (self.entry_price - current_price) * self.amount
    
    def get_unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.side == 'long':
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # short
            return ((self.entry_price - current_price) / self.entry_price) * 100
    
    def should_stop_loss(self, current_price: float) -> bool:
        """Check if stop-loss is triggered."""
        if self.side == 'long':
            return current_price <= self.stop_loss
        else:  # short
            return current_price >= self.stop_loss
    
    def should_take_profit(self, current_price: float) -> bool:
        """Check if take-profit is triggered."""
        if self.side == 'long':
            return current_price >= self.take_profit
        else:  # short
            return current_price <= self.take_profit
    
    def should_time_stop(self) -> bool:
        """Check if max hold time exceeded."""
        hold_duration = datetime.now() - self.entry_time
        return hold_duration > timedelta(hours=self.max_hold_hours)
    
    def update_trailing_stop(self, current_price: float, atr: float, multiplier: float = 1.5) -> bool:
        """Update trailing stop if price moves in our favor. Returns True if updated."""
        if not self.trailing_stop:
            return False
        
        if self.side == 'long':
            if current_price > self.highest_price:
                self.highest_price = current_price
                new_stop = current_price - (atr * multiplier)
                if new_stop > self.stop_loss:
                    old_stop = self.stop_loss
                    self.stop_loss = new_stop
                    logger.info(f"Trailing stop updated: {old_stop:.2f} -> {new_stop:.2f}")
                    return True
        else:  # short
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                new_stop = current_price + (atr * multiplier)
                if new_stop < self.stop_loss:
                    old_stop = self.stop_loss
                    self.stop_loss = new_stop
                    logger.info(f"Trailing stop updated: {old_stop:.2f} -> {new_stop:.2f}")
                    return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "amount": self.amount,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop": self.trailing_stop,
            "entry_reasoning": self.entry_reasoning
        }


class PaperExchange:
    """
    Simulates an exchange for paper trading with proper position management.
    """
    
    def __init__(self, initial_balance: float = 10000.0, slippage_pct: float = 0.001, fee_pct: float = 0.001):
        self.initial_balance = initial_balance
        self.balance = {"USDT": initial_balance}
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.closed_positions: List[Dict[str, Any]] = []  # History of closed positions
        self.trade_history: List[Dict[str, Any]] = []
        self.slippage_pct = slippage_pct  # 0.1% default
        self.fee_pct = fee_pct  # 0.1% default (Binance Spot)
        logger.info(f"Paper Exchange initialized. Balance: ${initial_balance:,.2f} | Slippage: {slippage_pct*100}% | Fee: {fee_pct*100}%")

    def get_balance(self, asset: str) -> float:
        return self.balance.get(asset, 0.0)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get open position for symbol."""
        return self.positions.get(symbol)
    
    def get_position_amount(self, symbol: str) -> float:
        """Get position amount (for backward compatibility)."""
        pos = self.positions.get(symbol)
        if pos:
            return pos.amount
        # Fallback to balance for the base asset
        base_asset = symbol.split('/')[0]
        return self.balance.get(base_asset, 0.0)
    
    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for symbol."""
        return symbol in self.positions
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all open positions."""
        return self.positions.copy()
    
    def open_position(
        self,
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        trailing_stop: bool = False,
        max_hold_hours: int = 72,
        entry_reasoning: str = "",
        agent_name: str = "",
        market_regime: str = ""
    ) -> Optional[Position]:
        """
        Open a new position with stop-loss and take-profit.
        """
        base_asset, quote_asset = symbol.split('/')
        
        # Apply slippage
        execution_price = entry_price * (1 + self.slippage_pct) if side == 'long' else entry_price * (1 - self.slippage_pct)
        
        # Calculate cost
        cost = amount * execution_price
        fee = cost * self.fee_pct
        total_cost = cost + fee
        
        # Check balance
        if self.balance.get(quote_asset, 0) < total_cost:
            logger.warning(f"Insufficient {quote_asset} balance. Need ${total_cost:,.2f}, have ${self.balance.get(quote_asset, 0):,.2f}")
            return None
        
        # Check if position already exists
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}. Close it first or add to position.")
            return None
        
        # Deduct balance
        self.balance[quote_asset] = self.balance.get(quote_asset, 0) - total_cost
        if base_asset not in self.balance:
            self.balance[base_asset] = 0.0
        self.balance[base_asset] += amount
        
        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            amount=amount,
            entry_price=execution_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=stop_loss if trailing_stop else None,
            max_hold_hours=max_hold_hours,
            entry_reasoning=entry_reasoning,
            agent_name=agent_name,
            market_regime=market_regime
        )
        
        self.positions[symbol] = position
        
        # Record trade
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": "buy" if side == "long" else "sell",
            "amount": amount,
            "price": execution_price,
            "fee": fee,
            "cost": total_cost,
            "type": "open_position",
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
        self.trade_history.append(trade_record)
        
        logger.info(
            f"📈 POSITION OPENED: {side.upper()} {amount:.4f} {base_asset} @ ${execution_price:,.2f} | "
            f"SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f} | Fee: ${fee:.2f}"
        )
        
        return position
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "signal"  # stop_loss, take_profit, signal, manual, time_stop
    ) -> Optional[Dict[str, Any]]:
        """
        Close an existing position.
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return None
        
        position = self.positions[symbol]
        base_asset, quote_asset = symbol.split('/')
        
        # Apply slippage (opposite direction)
        if position.side == 'long':
            execution_price = exit_price * (1 - self.slippage_pct)
        else:
            execution_price = exit_price * (1 + self.slippage_pct)
        
        # Calculate proceeds
        proceeds = position.amount * execution_price
        fee = proceeds * self.fee_pct
        net_proceeds = proceeds - fee
        
        # Calculate P&L
        if position.side == 'long':
            pnl = net_proceeds - (position.amount * position.entry_price)
        else:
            pnl = (position.amount * position.entry_price) - net_proceeds
        
        pnl_pct = (pnl / (position.amount * position.entry_price)) * 100
        
        # Update balance
        self.balance[base_asset] = self.balance.get(base_asset, 0) - position.amount
        self.balance[quote_asset] = self.balance.get(quote_asset, 0) + net_proceeds
        
        # Record closed position
        closed_record = {
            "symbol": symbol,
            "side": position.side,
            "amount": position.amount,
            "entry_price": position.entry_price,
            "entry_time": position.entry_time.isoformat(),
            "exit_price": execution_price,
            "exit_time": datetime.now().isoformat(),
            "exit_reason": reason,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "hold_duration_hours": (datetime.now() - position.entry_time).total_seconds() / 3600,
            "entry_reasoning": position.entry_reasoning
        }
        self.closed_positions.append(closed_record)
        
        # Record trade
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": "sell" if position.side == "long" else "buy",
            "amount": position.amount,
            "price": execution_price,
            "fee": fee,
            "proceeds": net_proceeds,
            "type": "close_position",
            "reason": reason,
            "pnl": pnl,
            "pnl_pct": pnl_pct
        }
        self.trade_history.append(trade_record)
        
        # Remove position
        del self.positions[symbol]
        
        # Log with emoji based on outcome
        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{emoji} POSITION CLOSED ({reason.upper()}): {position.side.upper()} {position.amount:.4f} {base_asset} | "
            f"Entry: ${position.entry_price:,.2f} -> Exit: ${execution_price:,.2f} | "
            f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)"
        )
        
        return closed_record
    
    def check_stops(self, current_prices: Dict[str, float], atrs: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Check all positions for stop-loss, take-profit, and time stops.
        Returns list of closed positions.
        """
        closed = []
        positions_to_check = list(self.positions.keys())
        
        for symbol in positions_to_check:
            if symbol not in current_prices:
                continue
            
            position = self.positions[symbol]
            current_price = current_prices[symbol]
            
            # Update trailing stop if enabled
            if atrs and symbol in atrs and position.trailing_stop:
                position.update_trailing_stop(current_price, atrs[symbol])
            
            # Check stop-loss
            if position.should_stop_loss(current_price):
                result = self.close_position(symbol, current_price, "stop_loss")
                if result:
                    closed.append(result)
                continue
            
            # Check take-profit
            if position.should_take_profit(current_price):
                result = self.close_position(symbol, current_price, "take_profit")
                if result:
                    closed.append(result)
                continue
            
            # Check time stop
            if position.should_time_stop():
                result = self.close_position(symbol, current_price, "time_stop")
                if result:
                    closed.append(result)
                continue
        
        return closed
    
    def execute_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[Dict[str, Any]]:
        """
        Execute a simulated order with slippage and fees.
        DEPRECATED: Use open_position() and close_position() instead.
        """
        base_asset, quote_asset = symbol.split('/')
        
        # Ensure assets exist in balance
        if base_asset not in self.balance:
            self.balance[base_asset] = 0.0
        if quote_asset not in self.balance:
            self.balance[quote_asset] = 0.0
        
        # Apply Slippage
        if side.lower() == 'buy':
            execution_price = price * (1 + self.slippage_pct)
        else:
            execution_price = price * (1 - self.slippage_pct)
            
        cost = amount * execution_price
        fee = cost * self.fee_pct
        
        if side.lower() == 'buy':
            total_cost = cost + fee
            if self.balance[quote_asset] >= total_cost:
                self.balance[quote_asset] -= total_cost
                self.balance[base_asset] += amount
                
                trade_record = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "side": "buy",
                    "amount": amount,
                    "price": execution_price,
                    "fee": fee,
                    "cost": total_cost
                }
                self.trade_history.append(trade_record)
                logger.info(f"Executed BUY: {amount} {base_asset} @ ${execution_price:,.2f} (Fee: ${fee:.2f})")
                return trade_record
            else:
                logger.warning(f"Insufficient {quote_asset} balance to buy {amount} {base_asset}")
                return None
                
        elif side.lower() == 'sell':
            if self.balance[base_asset] >= amount:
                self.balance[base_asset] -= amount
                proceeds = cost - fee
                self.balance[quote_asset] += proceeds
                
                trade_record = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "side": "sell",
                    "amount": amount,
                    "price": execution_price,
                    "fee": fee,
                    "proceeds": proceeds
                }
                self.trade_history.append(trade_record)
                logger.info(f"Executed SELL: {amount} {base_asset} @ ${execution_price:,.2f} (Fee: ${fee:.2f})")
                return trade_record
            else:
                logger.warning(f"Insufficient {base_asset} balance to sell {amount} {base_asset}")
                return None
        
        return None

    async def execute_order_async(self, symbol: str, side: str, amount: float, price: float) -> Optional[Dict[str, Any]]:
        """
        Async version of execute_order that also saves to DB.
        """
        trade_record = self.execute_order(symbol, side, amount, price)
        
        if trade_record:
            try:
                async with get_db() as db:
                    trade = Trade(
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        price=trade_record['price'],
                        cost=trade_record.get('cost'),
                        commission=trade_record.get('fee'),
                        timestamp=datetime.fromisoformat(trade_record['timestamp'])
                    )
                    db.add(trade)
                    await db.commit()
                    logger.info(f"Trade saved to DB: {trade.id}")
            except Exception as e:
                logger.error(f"Failed to save trade to DB: {e}")
        
        return trade_record
    
    async def save_position_to_db(self, position: Position) -> Optional[int]:
        """Save position to database."""
        try:
            async with get_db() as db:
                db_position = DBPosition(
                    symbol=position.symbol,
                    side=position.side,
                    amount=position.amount,
                    entry_price=position.entry_price,
                    entry_time=position.entry_time,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    trailing_stop=position.trailing_stop,
                    max_hold_hours=position.max_hold_hours,
                    entry_reasoning=position.entry_reasoning,
                    agent_name=position.agent_name,
                    market_regime=position.market_regime,
                    status="open"
                )
                db.add(db_position)
                await db.commit()
                logger.info(f"Position saved to DB: {db_position.id}")
                return db_position.id
        except Exception as e:
            logger.error(f"Failed to save position to DB: {e}")
            return None

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value in USDT.
        """
        total_value = self.get_balance("USDT")
        for asset, amount in self.balance.items():
            if asset == "USDT" or amount == 0:
                continue
            price = current_prices.get(f"{asset}/USDT", 0.0)
            total_value += amount * price
        return total_value
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of trading performance."""
        if not self.closed_positions:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl_pct": 0,
                "best_trade": None,
                "worst_trade": None
            }
        
        winning = [p for p in self.closed_positions if p['pnl'] > 0]
        losing = [p for p in self.closed_positions if p['pnl'] <= 0]
        
        total_pnl = sum(p['pnl'] for p in self.closed_positions)
        avg_pnl_pct = sum(p['pnl_pct'] for p in self.closed_positions) / len(self.closed_positions)
        
        best = max(self.closed_positions, key=lambda x: x['pnl'])
        worst = min(self.closed_positions, key=lambda x: x['pnl'])
        
        # Group by exit reason
        by_reason = {}
        for p in self.closed_positions:
            reason = p['exit_reason']
            if reason not in by_reason:
                by_reason[reason] = {"count": 0, "pnl": 0}
            by_reason[reason]["count"] += 1
            by_reason[reason]["pnl"] += p['pnl']
        
        return {
            "total_trades": len(self.closed_positions),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(self.closed_positions) * 100 if self.closed_positions else 0,
            "total_pnl": total_pnl,
            "avg_pnl_pct": avg_pnl_pct,
            "best_trade": best,
            "worst_trade": worst,
            "by_exit_reason": by_reason,
            "open_positions": len(self.positions)
        }
    
    # ==================== STATE PERSISTENCE ====================
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for persistence."""
        return {
            'balances': self.balance.copy(),
            'positions': {s: p.to_dict() for s, p in self.positions.items()},
            'closed_positions': self.closed_positions.copy(),
            'trade_history': self.trade_history.copy(),
            'initial_balance': self.initial_balance
        }
    
    def restore_state(self, state: Dict[str, Any]):
        """Restore state from saved data."""
        if 'balances' in state:
            self.balance = state['balances'].copy()
            logger.info(f"Restored balances: {self.balance}")
        
        if 'positions' in state:
            self.positions = {}
            for symbol, pos_data in state['positions'].items():
                self.positions[symbol] = Position(
                    symbol=pos_data['symbol'],
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
                    market_regime=pos_data.get('market_regime', '')
                )
            logger.info(f"Restored {len(self.positions)} positions")
        
        if 'closed_positions' in state:
            self.closed_positions = state['closed_positions'].copy()
            logger.info(f"Restored {len(self.closed_positions)} closed positions")
        
        if 'trade_history' in state:
            self.trade_history = state['trade_history'].copy()
        
        if 'initial_balance' in state:
            self.initial_balance = state['initial_balance']
    
    @classmethod
    def from_state(cls, state: Dict[str, Any], slippage_pct: float = 0.001, fee_pct: float = 0.001) -> 'PaperExchange':
        """Create a PaperExchange instance from saved state."""
        initial_balance = state.get('initial_balance', 10000.0)
        exchange = cls(
            initial_balance=initial_balance,
            slippage_pct=slippage_pct,
            fee_pct=fee_pct
        )
        exchange.restore_state(state)
        return exchange
