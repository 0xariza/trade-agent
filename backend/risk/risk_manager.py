import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode based on drawdown level."""
    NORMAL = "normal"           # Full position sizes
    CAUTIOUS = "cautious"       # 75% position sizes
    DEFENSIVE = "defensive"     # 50% position sizes
    RECOVERY = "recovery"       # 25% position sizes
    HALTED = "halted"          # No trading


@dataclass
class StopLevels:
    """Stop-loss and take-profit levels."""
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    risk_reward_ratio: float = 2.0


@dataclass
class DrawdownState:
    """Current drawdown state."""
    peak_equity: float
    current_equity: float
    current_drawdown_pct: float
    max_drawdown_pct: float
    trading_mode: TradingMode
    position_size_multiplier: float
    days_in_drawdown: int
    consecutive_losses: int
    last_updated: datetime = field(default_factory=datetime.now)


class DrawdownManager:
    """
    Manages drawdown tracking and position sizing adjustments.
    
    Features:
    - Tracks peak equity (high water mark)
    - Calculates rolling drawdown
    - Reduces position sizes during drawdowns
    - Implements weekly/monthly loss limits
    - Recovery mode after large losses
    - Consecutive loss tracking
    """
    
    def __init__(
        self,
        initial_equity: float = 10000.0,
        max_drawdown_pct: float = 0.15,          # 15% max drawdown before halt
        cautious_drawdown_pct: float = 0.05,      # 5% drawdown -> cautious mode
        defensive_drawdown_pct: float = 0.08,     # 8% drawdown -> defensive mode
        recovery_drawdown_pct: float = 0.12,      # 12% drawdown -> recovery mode
        weekly_loss_limit_pct: float = 0.08,      # 8% weekly loss limit
        monthly_loss_limit_pct: float = 0.15,     # 15% monthly loss limit
        max_consecutive_losses: int = 5,          # Max consecutive losses before pause
        recovery_win_streak: int = 3              # Wins needed to exit recovery
    ):
        # Thresholds
        self.max_drawdown_pct = max_drawdown_pct
        self.cautious_drawdown_pct = cautious_drawdown_pct
        self.defensive_drawdown_pct = defensive_drawdown_pct
        self.recovery_drawdown_pct = recovery_drawdown_pct
        self.weekly_loss_limit_pct = weekly_loss_limit_pct
        self.monthly_loss_limit_pct = monthly_loss_limit_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.recovery_win_streak = recovery_win_streak
        
        # State tracking
        self.peak_equity = initial_equity
        self.current_equity = initial_equity
        self.initial_equity = initial_equity
        self.max_drawdown_seen = 0.0
        
        # Time-based tracking
        self.week_start_equity = initial_equity
        self.month_start_equity = initial_equity
        self.week_start_date = datetime.now()
        self.month_start_date = datetime.now()
        self.drawdown_start_date: Optional[datetime] = None
        
        # Trade tracking
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.recent_trades: List[Dict[str, Any]] = []  # Last 20 trades
        
        # Current mode
        self.trading_mode = TradingMode.NORMAL
        self.position_size_multiplier = 1.0
        
        logger.info(
            f"DrawdownManager initialized. Max DD: {max_drawdown_pct*100}% | "
            f"Cautious: {cautious_drawdown_pct*100}% | Defensive: {defensive_drawdown_pct*100}%"
        )
    
    def update_equity(self, new_equity: float) -> DrawdownState:
        """
        Update equity and recalculate drawdown state.
        Call this after every trade or periodically.
        """
        self.current_equity = new_equity
        
        # Update peak (high water mark)
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
            self.drawdown_start_date = None  # Exited drawdown
        elif self.drawdown_start_date is None and new_equity < self.peak_equity:
            self.drawdown_start_date = datetime.now()
        
        # Calculate current drawdown
        current_dd = 0.0
        if self.peak_equity > 0:
            current_dd = (self.peak_equity - new_equity) / self.peak_equity
        
        # Update max drawdown seen
        if current_dd > self.max_drawdown_seen:
            self.max_drawdown_seen = current_dd
        
        # Check weekly/monthly limits
        self._check_period_limits()
        
        # Determine trading mode
        old_mode = self.trading_mode
        self._update_trading_mode(current_dd)
        
        if old_mode != self.trading_mode:
            logger.warning(f"Trading mode changed: {old_mode.value} -> {self.trading_mode.value}")
        
        # Calculate days in drawdown
        days_in_dd = 0
        if self.drawdown_start_date:
            days_in_dd = (datetime.now() - self.drawdown_start_date).days
        
        return DrawdownState(
            peak_equity=self.peak_equity,
            current_equity=self.current_equity,
            current_drawdown_pct=current_dd * 100,
            max_drawdown_pct=self.max_drawdown_seen * 100,
            trading_mode=self.trading_mode,
            position_size_multiplier=self.position_size_multiplier,
            days_in_drawdown=days_in_dd,
            consecutive_losses=self.consecutive_losses
        )
    
    def _update_trading_mode(self, current_dd: float):
        """Update trading mode based on drawdown level."""
        # Check for halt condition
        if current_dd >= self.max_drawdown_pct:
            self.trading_mode = TradingMode.HALTED
            self.position_size_multiplier = 0.0
            return
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.trading_mode = TradingMode.RECOVERY
            self.position_size_multiplier = 0.25
            return
        
        # Drawdown-based modes
        if current_dd >= self.recovery_drawdown_pct:
            self.trading_mode = TradingMode.RECOVERY
            self.position_size_multiplier = 0.25
        elif current_dd >= self.defensive_drawdown_pct:
            self.trading_mode = TradingMode.DEFENSIVE
            self.position_size_multiplier = 0.50
        elif current_dd >= self.cautious_drawdown_pct:
            self.trading_mode = TradingMode.CAUTIOUS
            self.position_size_multiplier = 0.75
        else:
            # Check if we're in recovery mode and need to stay there
            if self.trading_mode == TradingMode.RECOVERY:
                if self.consecutive_wins >= self.recovery_win_streak:
                    self.trading_mode = TradingMode.NORMAL
                    self.position_size_multiplier = 1.0
                    logger.info(f"Exiting RECOVERY mode after {self.consecutive_wins} consecutive wins")
            else:
                self.trading_mode = TradingMode.NORMAL
                self.position_size_multiplier = 1.0
    
    def _check_period_limits(self):
        """Check and reset weekly/monthly limits."""
        now = datetime.now()
        
        # Weekly reset (every Monday)
        if now.weekday() < self.week_start_date.weekday() or (now - self.week_start_date).days >= 7:
            weekly_return = (self.current_equity - self.week_start_equity) / self.week_start_equity
            
            if weekly_return < -self.weekly_loss_limit_pct:
                logger.critical(f"Weekly loss limit hit: {weekly_return*100:.1f}%")
                self.trading_mode = TradingMode.HALTED
            
            # Reset for new week
            self.week_start_equity = self.current_equity
            self.week_start_date = now
        
        # Monthly reset (1st of month)
        if now.month != self.month_start_date.month:
            monthly_return = (self.current_equity - self.month_start_equity) / self.month_start_equity
            
            if monthly_return < -self.monthly_loss_limit_pct:
                logger.critical(f"Monthly loss limit hit: {monthly_return*100:.1f}%")
            
            # Reset for new month
            self.month_start_equity = self.current_equity
            self.month_start_date = now
    
    def record_trade(self, pnl: float, pnl_pct: float):
        """Record a completed trade and update streak tracking."""
        trade_record = {
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'timestamp': datetime.now()
        }
        
        self.recent_trades.append(trade_record)
        if len(self.recent_trades) > 20:
            self.recent_trades = self.recent_trades[-20:]
        
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            if self.consecutive_losses >= self.max_consecutive_losses:
                logger.warning(f"⚠️ {self.consecutive_losses} consecutive losses! Entering recovery mode.")
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        if self.trading_mode == TradingMode.HALTED:
            return False, f"Trading HALTED: Max drawdown ({self.max_drawdown_pct*100}%) exceeded"
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            if self.consecutive_wins < self.recovery_win_streak:
                return True, f"RECOVERY mode: {self.consecutive_losses} losses, need {self.recovery_win_streak - self.consecutive_wins} wins to recover"
        
        return True, f"Trading allowed in {self.trading_mode.value} mode"
    
    def get_adjusted_position_size(self, base_size: float) -> float:
        """Adjust position size based on current drawdown state."""
        adjusted = base_size * self.position_size_multiplier
        
        if self.position_size_multiplier < 1.0:
            logger.info(
                f"Position size adjusted: {base_size:.6f} -> {adjusted:.6f} "
                f"({self.trading_mode.value} mode, {self.position_size_multiplier*100:.0f}%)"
            )
        
        return adjusted
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get current drawdown status summary."""
        current_dd = 0.0
        if self.peak_equity > 0:
            current_dd = (self.peak_equity - self.current_equity) / self.peak_equity
        
        weekly_pnl = 0.0
        if self.week_start_equity > 0:
            weekly_pnl = (self.current_equity - self.week_start_equity) / self.week_start_equity
        
        return {
            'trading_mode': self.trading_mode.value,
            'position_multiplier': self.position_size_multiplier,
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'current_drawdown_pct': current_dd * 100,
            'max_drawdown_seen_pct': self.max_drawdown_seen * 100,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'weekly_pnl_pct': weekly_pnl * 100,
            'days_in_drawdown': (datetime.now() - self.drawdown_start_date).days if self.drawdown_start_date else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            'peak_equity': self.peak_equity,
            'current_equity': self.current_equity,
            'initial_equity': self.initial_equity,
            'max_drawdown_seen': self.max_drawdown_seen,
            'week_start_equity': self.week_start_equity,
            'month_start_equity': self.month_start_equity,
            'week_start_date': self.week_start_date.isoformat(),
            'month_start_date': self.month_start_date.isoformat(),
            'drawdown_start_date': self.drawdown_start_date.isoformat() if self.drawdown_start_date else None,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'trading_mode': self.trading_mode.value,
            'position_size_multiplier': self.position_size_multiplier
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DrawdownManager':
        """Restore from persisted data."""
        manager = cls(initial_equity=data.get('initial_equity', 10000.0))
        
        manager.peak_equity = data.get('peak_equity', manager.initial_equity)
        manager.current_equity = data.get('current_equity', manager.initial_equity)
        manager.max_drawdown_seen = data.get('max_drawdown_seen', 0.0)
        manager.week_start_equity = data.get('week_start_equity', manager.initial_equity)
        manager.month_start_equity = data.get('month_start_equity', manager.initial_equity)
        manager.consecutive_losses = data.get('consecutive_losses', 0)
        manager.consecutive_wins = data.get('consecutive_wins', 0)
        manager.position_size_multiplier = data.get('position_size_multiplier', 1.0)
        
        if data.get('week_start_date'):
            manager.week_start_date = datetime.fromisoformat(data['week_start_date'])
        if data.get('month_start_date'):
            manager.month_start_date = datetime.fromisoformat(data['month_start_date'])
        if data.get('drawdown_start_date'):
            manager.drawdown_start_date = datetime.fromisoformat(data['drawdown_start_date'])
        
        mode_str = data.get('trading_mode', 'normal')
        manager.trading_mode = TradingMode(mode_str)
        
        return manager


class RiskManager:
    """
    Enforces risk management rules for trading with stop-loss and take-profit.
    """
    
    def __init__(
        self,
        max_position_size_pct: float = 0.10,
        max_daily_loss_pct: float = 0.05,
        default_risk_reward: float = 2.0,
        default_atr_stop_multiplier: float = 2.0,
        max_hold_hours: int = 72,
        enable_trailing_stop: bool = True
    ):
        self.max_position_size_pct = max_position_size_pct  # Max 10% of portfolio per trade
        self.max_daily_loss_pct = max_daily_loss_pct  # Max 5% daily loss
        self.default_risk_reward = default_risk_reward  # Default 2:1 reward to risk
        self.default_atr_stop_multiplier = default_atr_stop_multiplier  # Stop at 2x ATR
        self.max_hold_hours = max_hold_hours  # Max position hold time
        self.enable_trailing_stop = enable_trailing_stop

    def check_circuit_breaker(self, current_balance: float, daily_start_balance: float) -> bool:
        """
        Check if the daily loss limit has been reached.
        Returns True if trading should be HALTED.
        """
        if daily_start_balance <= 0:
            return False

        loss = daily_start_balance - current_balance
        loss_pct = loss / daily_start_balance
        
        if loss_pct >= self.max_daily_loss_pct:
            logger.critical(f"CIRCUIT BREAKER TRIGGERED! Daily Loss: {loss_pct*100:.2f}% > Limit: {self.max_daily_loss_pct*100:.2f}%")
            return True
            
        return False

    def validate_trade(self, signal: str, price: float, amount: float, portfolio_value: float, market_data: Dict[str, Any]) -> bool:
        """
        Validate if a trade should be executed based on risk rules.
        Returns True if safe to trade, False otherwise.
        """
        # 1. Check Position Sizing
        trade_value = price * amount
        max_trade_value = portfolio_value * self.max_position_size_pct
        
        if trade_value > max_trade_value:
            logger.warning(f"RISK REJECT: Trade value ${trade_value:,.2f} exceeds max allowed ${max_trade_value:,.2f} ({self.max_position_size_pct*100}%)")
            return False

        # 2. Check Technical Indicators (RSI)
        rsi = market_data.get('indicators', {}).get('rsi')
        
        if rsi is not None:
            if signal == 'buy' and rsi > 70:
                logger.warning(f"RISK REJECT: Buying when RSI is {rsi} (Overbought > 70)")
                return False
            
            if signal == 'sell' and rsi < 30:
                logger.warning(f"RISK REJECT: Selling when RSI is {rsi} (Oversold < 30)")
                return False

        return True

    def check_trade_risk(self, symbol: str, signal: str, amount: float, price: float, account_balance: float) -> bool:
        """
        Check if a trade meets risk requirements.
        This is a simplified version that checks position sizing only.
        For full validation including technical indicators, use validate_trade().
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            signal: 'buy' or 'sell'
            amount: Amount of asset to trade
            price: Current price
            account_balance: Current account balance
            
        Returns:
            True if trade is safe, False otherwise
        """
        # Check Position Sizing
        trade_value = price * amount
        max_trade_value = account_balance * self.max_position_size_pct
        
        if trade_value > max_trade_value:
            logger.warning(
                f"RISK REJECT: Trade value ${trade_value:,.2f} exceeds max allowed "
                f"${max_trade_value:,.2f} ({self.max_position_size_pct*100}%)"
            )
            return False
        
        # Ensure we have enough balance
        if signal == 'buy' and trade_value > account_balance:
            logger.warning(f"RISK REJECT: Insufficient balance. Need ${trade_value:,.2f}, have ${account_balance:,.2f}")
            return False
        
        return True

    def calculate_position_size(self, portfolio_value: float, current_price: float, atr: float, risk_per_trade_pct: float = 0.01, atr_multiplier: float = 2.0) -> float:
        """
        Calculate position size based on volatility (ATR).
        Formula: Position Size = (Portfolio Value * Risk %) / (ATR * Multiplier)
        
        Returns the amount of asset to buy (e.g., BTC amount).
        """
        if atr <= 0:
            logger.warning("ATR is 0 or negative. Defaulting to min size.")
            return 0.0001 # Safe default

        risk_amount = portfolio_value * risk_per_trade_pct
        stop_loss_distance = atr * atr_multiplier
        
        # Position Value = Risk Amount / (Stop Loss Distance / Price) 
        # Actually simpler: Position Amount = Risk Amount / Stop Loss Distance
        # Example: Risk $100. Stop Loss is $500 away. Position Amount = 100 / 500 = 0.2 BTC.
        # If price moves $500 against us, we lose 0.2 * 500 = $100.
        
        position_amount = risk_amount / stop_loss_distance
        
        # Cap at max position size
        max_position_value = portfolio_value * self.max_position_size_pct
        max_position_amount = max_position_value / current_price
        
        final_amount = min(position_amount, max_position_amount)
        
        logger.info(f"Dynamic Sizing: Risk=${risk_amount:.2f} | ATR=${atr:.2f} | Calc Amount={position_amount:.4f} | Final={final_amount:.4f}")
        
        return final_amount
    
    def calculate_stop_levels(
        self,
        entry_price: float,
        side: str,  # 'long' or 'short'
        atr: float,
        risk_reward: float = None,
        atr_multiplier: float = None,
        support_level: float = None,
        resistance_level: float = None
    ) -> StopLevels:
        """
        Calculate stop-loss and take-profit levels based on ATR and risk-reward ratio.
        
        Args:
            entry_price: Entry price for the position
            side: 'long' or 'short'
            atr: Average True Range (volatility measure)
            risk_reward: Risk-reward ratio (default from config)
            atr_multiplier: ATR multiplier for stop distance (default from config)
            support_level: Optional support level for smarter stop placement
            resistance_level: Optional resistance level for smarter TP placement
        
        Returns:
            StopLevels with stop_loss, take_profit, and optional trailing_stop
        """
        risk_reward = risk_reward or self.default_risk_reward
        atr_multiplier = atr_multiplier or self.default_atr_stop_multiplier
        
        # Default to 1% of price if ATR is 0
        if atr <= 0:
            atr = entry_price * 0.01
            logger.warning(f"ATR is 0, using 1% of price: {atr:.2f}")
        
        # Calculate stop distance
        stop_distance = atr * atr_multiplier
        
        if side == 'long':
            # Long position: stop below entry, take-profit above
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * risk_reward)
            
            # Use support level if provided and it's tighter
            if support_level and support_level < entry_price:
                # Place stop just below support with some buffer
                buffer = atr * 0.5
                potential_stop = support_level - buffer
                if potential_stop > stop_loss:
                    stop_loss = potential_stop
                    logger.info(f"Using support-based stop: ${stop_loss:.2f}")
            
            # Use resistance level for take-profit if provided
            if resistance_level and resistance_level > entry_price:
                # Target just below resistance
                potential_tp = resistance_level - (atr * 0.25)
                if potential_tp < take_profit:
                    take_profit = potential_tp
                    logger.info(f"Using resistance-based TP: ${take_profit:.2f}")
        
        else:  # short
            # Short position: stop above entry, take-profit below
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * risk_reward)
            
            # Use resistance level if provided
            if resistance_level and resistance_level > entry_price:
                buffer = atr * 0.5
                potential_stop = resistance_level + buffer
                if potential_stop < stop_loss:
                    stop_loss = potential_stop
                    logger.info(f"Using resistance-based stop: ${stop_loss:.2f}")
            
            # Use support level for take-profit if provided
            if support_level and support_level < entry_price:
                potential_tp = support_level + (atr * 0.25)
                if potential_tp > take_profit:
                    take_profit = potential_tp
                    logger.info(f"Using support-based TP: ${take_profit:.2f}")
        
        # Calculate trailing stop (starts at stop-loss level)
        trailing_stop = stop_loss if self.enable_trailing_stop else None
        
        logger.info(
            f"Stop Levels ({side.upper()}): Entry=${entry_price:,.2f} | "
            f"SL=${stop_loss:,.2f} ({((abs(entry_price - stop_loss) / entry_price) * 100):.1f}%) | "
            f"TP=${take_profit:,.2f} ({((abs(take_profit - entry_price) / entry_price) * 100):.1f}%) | "
            f"R:R={risk_reward}"
        )
        
        return StopLevels(
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            risk_reward_ratio=risk_reward
        )
    
    def calculate_position_with_stops(
        self,
        portfolio_value: float,
        current_price: float,
        atr: float,
        side: str = 'long',
        risk_per_trade_pct: float = 0.01,
        risk_reward: float = None
    ) -> Tuple[float, StopLevels]:
        """
        Calculate position size AND stop levels together.
        This ensures position size is based on actual stop distance.
        
        Returns:
            Tuple of (position_amount, StopLevels)
        """
        # First calculate stop levels
        stop_levels = self.calculate_stop_levels(
            entry_price=current_price,
            side=side,
            atr=atr,
            risk_reward=risk_reward
        )
        
        # Calculate actual stop distance
        stop_distance = abs(current_price - stop_levels.stop_loss)
        
        # Position size based on risk and stop distance
        risk_amount = portfolio_value * risk_per_trade_pct
        
        if stop_distance <= 0:
            logger.warning("Stop distance is 0, using minimum position size")
            position_amount = 0.0001
        else:
            # If stop is hit, we lose risk_amount
            # position_amount * stop_distance = risk_amount
            position_amount = risk_amount / stop_distance
        
        # Cap at max position size
        max_position_value = portfolio_value * self.max_position_size_pct
        max_position_amount = max_position_value / current_price
        
        final_amount = min(position_amount, max_position_amount)
        
        # Calculate actual risk with final amount
        actual_risk = final_amount * stop_distance
        actual_risk_pct = (actual_risk / portfolio_value) * 100
        
        logger.info(
            f"Position Sizing: Risk ${risk_amount:.2f} ({risk_per_trade_pct*100}%) | "
            f"Stop Distance ${stop_distance:.2f} | Amount {final_amount:.6f} | "
            f"Actual Risk ${actual_risk:.2f} ({actual_risk_pct:.2f}%)"
        )
        
        return final_amount, stop_levels
    
    def validate_stop_levels(
        self,
        stop_levels: StopLevels,
        current_price: float,
        side: str
    ) -> Tuple[bool, str]:
        """
        Validate that stop levels are sensible.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        if side == 'long':
            if stop_levels.stop_loss >= current_price:
                return False, f"Stop-loss {stop_levels.stop_loss} must be below entry {current_price} for long"
            if stop_levels.take_profit <= current_price:
                return False, f"Take-profit {stop_levels.take_profit} must be above entry {current_price} for long"
        else:
            if stop_levels.stop_loss <= current_price:
                return False, f"Stop-loss {stop_levels.stop_loss} must be above entry {current_price} for short"
            if stop_levels.take_profit >= current_price:
                return False, f"Take-profit {stop_levels.take_profit} must be below entry {current_price} for short"
        
        # Check minimum R:R ratio
        risk = abs(current_price - stop_levels.stop_loss)
        reward = abs(stop_levels.take_profit - current_price)
        actual_rr = reward / risk if risk > 0 else 0
        
        if actual_rr < 1.0:
            return False, f"Risk-reward ratio {actual_rr:.2f} is below minimum 1.0"
        
        # Check stop isn't too tight (less than 0.5% from entry)
        stop_pct = (risk / current_price) * 100
        if stop_pct < 0.5:
            logger.warning(f"Stop-loss is very tight: {stop_pct:.2f}%")
        
        return True, "Valid"
    
    def adjust_for_market_regime(
        self,
        stop_levels: StopLevels,
        market_regime: str,
        atr: float
    ) -> StopLevels:
        """
        Adjust stop levels based on market regime.
        
        - Trending: Wider stops to avoid noise
        - Ranging: Tighter stops, target mean reversion
        - Volatile: Much wider stops or reduce position
        """
        multiplier = 1.0
        
        if market_regime == "trending":
            multiplier = 1.2  # 20% wider stops in trends
        elif market_regime == "ranging":
            multiplier = 0.8  # 20% tighter stops in ranges
        elif market_regime == "volatile":
            multiplier = 1.5  # 50% wider stops in volatile markets
        
        if multiplier != 1.0:
            # Adjust stop distance
            entry = (stop_levels.stop_loss + stop_levels.take_profit) / 2  # Approximate entry
            
            if stop_levels.stop_loss < entry:  # Long position
                stop_distance = entry - stop_levels.stop_loss
                new_stop_distance = stop_distance * multiplier
                stop_levels.stop_loss = entry - new_stop_distance
                stop_levels.take_profit = entry + (new_stop_distance * stop_levels.risk_reward_ratio)
            else:  # Short position
                stop_distance = stop_levels.stop_loss - entry
                new_stop_distance = stop_distance * multiplier
                stop_levels.stop_loss = entry + new_stop_distance
                stop_levels.take_profit = entry - (new_stop_distance * stop_levels.risk_reward_ratio)
            
            logger.info(f"Adjusted stops for {market_regime} regime (x{multiplier})")
        
        return stop_levels
