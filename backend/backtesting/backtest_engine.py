"""
Professional Backtesting Engine for Strategy Validation.

Features:
- Historical data fetching from exchange
- Multi-timeframe indicator calculation
- Strategy simulation with realistic execution
- Performance metrics (Sharpe, Sortino, Max Drawdown, Win Rate)
- Trade-by-trade analysis
- Equity curve generation
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: TradeDirection
    entry_price: float
    exit_price: float
    amount: float
    stop_loss: float
    take_profit: float
    exit_reason: str  # 'stop_loss', 'take_profit', 'signal', 'end_of_data'
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    
    # Market conditions at entry
    entry_rsi: float = 0.0
    entry_adx: float = 0.0
    entry_regime: str = ""
    
    def __post_init__(self):
        # Calculate P&L
        if self.direction == TradeDirection.LONG:
            gross_pnl = (self.exit_price - self.entry_price) * self.amount
        else:
            gross_pnl = (self.entry_price - self.exit_price) * self.amount
        
        self.pnl = gross_pnl - self.fees - self.slippage
        if self.entry_price > 0 and self.amount > 0:
            self.pnl_pct = (self.pnl / (self.entry_price * self.amount)) * 100


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    # Performance metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration_hours: float = 0.0
    
    # Risk metrics
    calmar_ratio: float = 0.0
    avg_risk_per_trade_pct: float = 0.0
    
    # Detailed data
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'period': f"{self.start_date.date()} to {self.end_date.date()}",
            'initial_capital': self.initial_capital,
            'final_capital': round(self.final_capital, 2),
            'total_return_pct': round(self.total_return_pct, 2),
            'annualized_return_pct': round(self.annualized_return_pct, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 2),
            'total_trades': self.total_trades,
            'win_rate': round(self.win_rate, 1),
            'profit_factor': round(self.profit_factor, 2),
            'avg_trade_duration_hours': round(self.avg_trade_duration_hours, 1),
            'calmar_ratio': round(self.calmar_ratio, 2)
        }
    
    def print_summary(self):
        """Print a formatted summary of the backtest results."""
        print("\n" + "=" * 70)
        print("📊 BACKTEST RESULTS")
        print("=" * 70)
        print(f"Symbol: {self.symbol}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.final_capital:,.2f}")
        
        print("\n📈 PERFORMANCE")
        print("-" * 40)
        print(f"Total Return: {self.total_return_pct:+.2f}%")
        print(f"Annualized Return: {self.annualized_return_pct:+.2f}%")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {self.sortino_ratio:.2f}")
        print(f"Calmar Ratio: {self.calmar_ratio:.2f}")
        
        print("\n📉 RISK")
        print("-" * 40)
        print(f"Max Drawdown: {self.max_drawdown_pct:.2f}%")
        print(f"Max DD Duration: {self.max_drawdown_duration_days} days")
        
        print("\n🎯 TRADES")
        print("-" * 40)
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {self.win_rate:.1f}%")
        print(f"Avg Win: {self.avg_win_pct:+.2f}%")
        print(f"Avg Loss: {self.avg_loss_pct:.2f}%")
        print(f"Profit Factor: {self.profit_factor:.2f}")
        print(f"Avg Duration: {self.avg_trade_duration_hours:.1f} hours")
        
        # Grade the strategy
        print("\n🏆 STRATEGY GRADE")
        print("-" * 40)
        grade = self._calculate_grade()
        print(f"Overall: {grade}")
        print("=" * 70)
    
    def _calculate_grade(self) -> str:
        """Calculate a letter grade for the strategy."""
        score = 0
        
        # Sharpe ratio scoring (max 30 points)
        if self.sharpe_ratio >= 2.0:
            score += 30
        elif self.sharpe_ratio >= 1.5:
            score += 25
        elif self.sharpe_ratio >= 1.0:
            score += 20
        elif self.sharpe_ratio >= 0.5:
            score += 10
        
        # Win rate scoring (max 20 points)
        if self.win_rate >= 60:
            score += 20
        elif self.win_rate >= 50:
            score += 15
        elif self.win_rate >= 40:
            score += 10
        
        # Profit factor scoring (max 25 points)
        if self.profit_factor >= 2.0:
            score += 25
        elif self.profit_factor >= 1.5:
            score += 20
        elif self.profit_factor >= 1.2:
            score += 15
        elif self.profit_factor >= 1.0:
            score += 5
        
        # Max drawdown scoring (max 25 points)
        if self.max_drawdown_pct <= 10:
            score += 25
        elif self.max_drawdown_pct <= 15:
            score += 20
        elif self.max_drawdown_pct <= 20:
            score += 15
        elif self.max_drawdown_pct <= 30:
            score += 10
        
        # Convert score to grade
        if score >= 90:
            return "A+ (Excellent - Production Ready)"
        elif score >= 80:
            return "A (Very Good - Consider Production)"
        elif score >= 70:
            return "B+ (Good - Needs Minor Tuning)"
        elif score >= 60:
            return "B (Acceptable - Needs Improvement)"
        elif score >= 50:
            return "C (Mediocre - Significant Changes Needed)"
        elif score >= 40:
            return "D (Poor - Major Overhaul Required)"
        else:
            return "F (Failing - Do Not Use)"


class BacktestEngine:
    """
    Engine for running backtests on trading strategies.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_pct: float = 0.001,  # 0.1% per trade
        slippage_pct: float = 0.0005,  # 0.05% slippage
        risk_per_trade_pct: float = 0.01,  # 1% risk per trade
        max_position_pct: float = 0.10  # 10% max position
    ):
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_pct = max_position_pct
        
        # State
        self.capital = initial_capital
        self.position: Optional[Dict[str, Any]] = None
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict[str, Any]] = []
        
    def reset(self):
        """Reset the engine for a new backtest."""
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
    
    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 90
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        """
        import ccxt.async_support as ccxt
        
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        try:
            since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
            
            all_ohlcv = []
            while True:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                if len(ohlcv) < 1000:
                    break
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        finally:
            await exchange.close()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataframe."""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        
        # ADX
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(df['high'] - df['low'])
        tr2 = pd.DataFrame(abs(df['high'] - df['close'].shift(1)))
        tr3 = pd.DataFrame(abs(df['low'] - df['close'].shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df['atr'] = atr
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = abs(100 * (minus_dm.rolling(14).mean() / atr))
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        df['adx'] = dx.rolling(14).mean()
        
        # EMAs
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Stochastic RSI
        rsi_min = df['rsi'].rolling(14).min()
        rsi_max = df['rsi'].rolling(14).max()
        df['stoch_rsi'] = (df['rsi'] - rsi_min) / (rsi_max - rsi_min) * 100
        df['stoch_rsi_k'] = df['stoch_rsi'].rolling(3).mean()
        df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(3).mean()
        
        # OBV
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        
        return df
    
    def detect_market_regime(self, row: pd.Series) -> str:
        """Detect market regime from indicators."""
        adx = row.get('adx', 0)
        atr_pct = (row.get('atr', 0) / row['close']) * 100 if row['close'] > 0 else 0
        
        if adx > 25:
            return "trending"
        elif atr_pct > 3:  # High volatility
            return "volatile"
        else:
            return "ranging"
    
    def generate_signal(self, row: pd.Series, prev_row: pd.Series = None) -> Tuple[str, float, Dict[str, Any]]:
        """
        Generate trading signal using rule-based logic.
        
        Returns:
            Tuple of (signal, confidence, metadata)
            signal: 'BUY', 'SELL', or 'HOLD'
            confidence: 0.0 to 1.0
            metadata: Additional information about the signal
        """
        score = 0
        reasons = []
        
        rsi = row.get('rsi', 50)
        adx = row.get('adx', 0)
        macd = row.get('macd', 0)
        macd_signal = row.get('macd_signal', 0)
        ema_9 = row.get('ema_9', 0)
        ema_21 = row.get('ema_21', 0)
        stoch_rsi_k = row.get('stoch_rsi_k', 50)
        stoch_rsi_d = row.get('stoch_rsi_d', 50)
        close = row['close']
        bb_lower = row.get('bb_lower', close)
        bb_upper = row.get('bb_upper', close)
        
        regime = self.detect_market_regime(row)
        
        # 1. Trend Analysis (40 points max)
        if ema_9 > ema_21:
            score += 20
            reasons.append("EMA 9 > 21 (bullish)")
        elif ema_9 < ema_21:
            score -= 20
            reasons.append("EMA 9 < 21 (bearish)")
        
        if macd > macd_signal:
            score += 20
            reasons.append("MACD bullish")
        elif macd < macd_signal:
            score -= 20
            reasons.append("MACD bearish")
        
        # 2. Momentum Analysis (30 points max)
        if rsi < 30:
            score += 25
            reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi > 70:
            score -= 25
            reasons.append(f"RSI overbought ({rsi:.0f})")
        elif rsi < 45:
            score += 10
        elif rsi > 55:
            score -= 10
        
        # Stochastic RSI crossover
        if prev_row is not None:
            prev_stoch_k = prev_row.get('stoch_rsi_k', 50)
            if stoch_rsi_k > stoch_rsi_d and prev_stoch_k <= prev_row.get('stoch_rsi_d', 50):
                if stoch_rsi_k < 30:
                    score += 15
                    reasons.append("Stoch RSI bullish cross from oversold")
            elif stoch_rsi_k < stoch_rsi_d and prev_stoch_k >= prev_row.get('stoch_rsi_d', 50):
                if stoch_rsi_k > 70:
                    score -= 15
                    reasons.append("Stoch RSI bearish cross from overbought")
        
        # 3. Bollinger Band (15 points max)
        if close < bb_lower:
            score += 15
            reasons.append("Price below BB lower (oversold)")
        elif close > bb_upper:
            score -= 15
            reasons.append("Price above BB upper (overbought)")
        
        # 4. ADX Trend Strength (15 points max)
        if adx > 25:
            # Strong trend - amplify signal
            if score > 0:
                score += 15
                reasons.append(f"Strong trend (ADX {adx:.0f})")
            elif score < 0:
                score -= 15
                reasons.append(f"Strong downtrend (ADX {adx:.0f})")
        else:
            # Weak trend - reduce signal strength
            score = int(score * 0.7)
            reasons.append(f"Weak trend (ADX {adx:.0f})")
        
        # Determine signal
        confidence = min(abs(score) / 100, 1.0)
        
        if score >= 50:
            signal = "STRONG_BUY"
        elif score >= 35:
            signal = "MODERATE_BUY"
        elif score >= 20 and rsi < 35:
            signal = "WEAK_BUY"
        elif score <= -50:
            signal = "STRONG_SELL"
        elif score <= -35:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        metadata = {
            'score': score,
            'regime': regime,
            'reasons': reasons,
            'rsi': rsi,
            'adx': adx,
            'macd_hist': row.get('macd_hist', 0)
        }
        
        return signal, confidence, metadata
    
    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        confidence: float
    ) -> float:
        """Calculate position size based on risk and confidence."""
        # Risk amount based on capital and confidence
        base_risk = capital * self.risk_per_trade_pct
        adjusted_risk = base_risk * confidence
        
        # Calculate amount based on stop loss distance
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0:
            return 0
        
        amount = adjusted_risk / stop_distance
        
        # Apply max position limit
        max_amount = (capital * self.max_position_pct) / entry_price
        amount = min(amount, max_amount)
        
        return amount
    
    def calculate_stops(
        self,
        entry_price: float,
        atr: float,
        direction: TradeDirection,
        regime: str
    ) -> Tuple[float, float]:
        """Calculate stop-loss and take-profit levels."""
        # Base multipliers
        sl_mult = 2.0
        tp_mult = 4.0  # 2:1 risk-reward
        
        # Adjust for market regime
        if regime == "volatile":
            sl_mult *= 1.3
            tp_mult *= 0.8
        elif regime == "ranging":
            sl_mult *= 0.8
            tp_mult *= 1.2
        
        stop_distance = atr * sl_mult
        tp_distance = atr * tp_mult
        
        if direction == TradeDirection.LONG:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance
        
        return stop_loss, take_profit
    
    def apply_slippage(self, price: float, direction: TradeDirection, is_entry: bool) -> float:
        """Apply realistic slippage to price."""
        slippage = price * self.slippage_pct
        
        if direction == TradeDirection.LONG:
            if is_entry:
                return price + slippage  # Pay more on entry
            else:
                return price - slippage  # Get less on exit
        else:
            if is_entry:
                return price - slippage
            else:
                return price + slippage
    
    async def run_backtest(
        self,
        symbol: str,
        days: int = 90,
        timeframe: str = "1h"
    ) -> BacktestResult:
        """
        Run a full backtest on historical data.
        """
        self.reset()
        
        print(f"📥 Fetching {days} days of {symbol} data...")
        df = await self.fetch_historical_data(symbol, timeframe, days)
        
        print(f"📊 Calculating indicators on {len(df)} candles...")
        df = self.calculate_indicators(df)
        
        # Drop rows with NaN values (from indicator warmup)
        df = df.dropna()
        
        print(f"🚀 Running backtest...")
        
        prev_row = None
        for timestamp, row in df.iterrows():
            current_price = row['close']
            atr = row.get('atr', current_price * 0.02)
            
            # Track equity
            equity = self.capital
            if self.position:
                if self.position['direction'] == TradeDirection.LONG:
                    unrealized = (current_price - self.position['entry_price']) * self.position['amount']
                else:
                    unrealized = (self.position['entry_price'] - current_price) * self.position['amount']
                equity += unrealized
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'price': current_price
            })
            
            # Check existing position for stops
            if self.position:
                exit_reason = None
                exit_price = current_price
                
                if self.position['direction'] == TradeDirection.LONG:
                    if row['low'] <= self.position['stop_loss']:
                        exit_reason = 'stop_loss'
                        exit_price = self.position['stop_loss']
                    elif row['high'] >= self.position['take_profit']:
                        exit_reason = 'take_profit'
                        exit_price = self.position['take_profit']
                else:
                    if row['high'] >= self.position['stop_loss']:
                        exit_reason = 'stop_loss'
                        exit_price = self.position['stop_loss']
                    elif row['low'] <= self.position['take_profit']:
                        exit_reason = 'take_profit'
                        exit_price = self.position['take_profit']
                
                if exit_reason:
                    self._close_position(timestamp, exit_price, exit_reason)
            
            # Generate signal only if not in position
            if not self.position:
                signal, confidence, metadata = self.generate_signal(row, prev_row)
                
                if signal in ['STRONG_BUY', 'MODERATE_BUY', 'WEAK_BUY']:
                    direction = TradeDirection.LONG
                    stop_loss, take_profit = self.calculate_stops(
                        current_price, atr, direction, metadata['regime']
                    )
                    
                    # Apply slippage to entry
                    entry_price = self.apply_slippage(current_price, direction, True)
                    
                    amount = self.calculate_position_size(
                        self.capital, entry_price, stop_loss, confidence
                    )
                    
                    if amount > 0:
                        self._open_position(
                            timestamp, direction, entry_price, amount,
                            stop_loss, take_profit, metadata
                        )
                
                elif signal in ['STRONG_SELL', 'SELL']:
                    # For now, only close long positions on sell signal
                    # Could extend to short selling if needed
                    pass
            
            prev_row = row
        
        # Close any remaining position at end
        if self.position:
            final_price = df.iloc[-1]['close']
            self._close_position(df.index[-1], final_price, 'end_of_data')
        
        # Calculate results
        result = self._calculate_results(symbol, df.index[0], df.index[-1])
        
        return result
    
    def _open_position(
        self,
        timestamp: datetime,
        direction: TradeDirection,
        entry_price: float,
        amount: float,
        stop_loss: float,
        take_profit: float,
        metadata: Dict[str, Any]
    ):
        """Open a new position."""
        # Calculate fees
        fees = entry_price * amount * self.fee_pct
        
        # Deduct from capital
        cost = entry_price * amount + fees
        if cost > self.capital:
            amount = (self.capital * 0.95) / (entry_price * (1 + self.fee_pct))
            cost = entry_price * amount + (entry_price * amount * self.fee_pct)
        
        self.capital -= cost
        
        self.position = {
            'entry_time': timestamp,
            'direction': direction,
            'entry_price': entry_price,
            'amount': amount,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'fees': fees,
            'metadata': metadata
        }
    
    def _close_position(self, timestamp: datetime, exit_price: float, exit_reason: str):
        """Close the current position."""
        if not self.position:
            return
        
        direction = self.position['direction']
        
        # Apply slippage to exit
        exit_price = self.apply_slippage(exit_price, direction, False)
        
        # Calculate exit fees
        exit_fees = exit_price * self.position['amount'] * self.fee_pct
        total_fees = self.position['fees'] + exit_fees
        
        # Calculate P&L
        if direction == TradeDirection.LONG:
            gross_pnl = (exit_price - self.position['entry_price']) * self.position['amount']
        else:
            gross_pnl = (self.position['entry_price'] - exit_price) * self.position['amount']
        
        net_pnl = gross_pnl - total_fees
        
        # Add back to capital
        self.capital += (self.position['entry_price'] * self.position['amount']) + net_pnl
        
        # Record trade
        trade = BacktestTrade(
            entry_time=self.position['entry_time'],
            exit_time=timestamp,
            symbol="",  # Will be set in results
            direction=direction,
            entry_price=self.position['entry_price'],
            exit_price=exit_price,
            amount=self.position['amount'],
            stop_loss=self.position['stop_loss'],
            take_profit=self.position['take_profit'],
            exit_reason=exit_reason,
            fees=total_fees,
            entry_rsi=self.position['metadata'].get('rsi', 0),
            entry_adx=self.position['metadata'].get('adx', 0),
            entry_regime=self.position['metadata'].get('regime', '')
        )
        self.trades.append(trade)
        
        self.position = None
    
    def _calculate_results(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Calculate backtest performance metrics."""
        result = BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            trades=[t for t in self.trades],
            equity_curve=self.equity_curve
        )
        
        # Set symbol on trades
        for trade in result.trades:
            trade.symbol = symbol
        
        # Total return
        result.total_return_pct = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        # Annualized return
        days = (end_date - start_date).days
        if days > 0:
            result.annualized_return_pct = ((1 + result.total_return_pct / 100) ** (365 / days) - 1) * 100
        
        # Trade statistics
        result.total_trades = len(self.trades)
        
        if result.total_trades > 0:
            winning = [t for t in self.trades if t.pnl > 0]
            losing = [t for t in self.trades if t.pnl <= 0]
            
            result.winning_trades = len(winning)
            result.losing_trades = len(losing)
            result.win_rate = (result.winning_trades / result.total_trades) * 100
            
            if winning:
                result.avg_win_pct = sum(t.pnl_pct for t in winning) / len(winning)
            if losing:
                result.avg_loss_pct = sum(t.pnl_pct for t in losing) / len(losing)
            
            total_wins = sum(t.pnl for t in winning)
            total_losses = abs(sum(t.pnl for t in losing))
            result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Average trade duration
            durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in self.trades]
            result.avg_trade_duration_hours = sum(durations) / len(durations)
        
        # Calculate drawdown
        if self.equity_curve:
            equity = [e['equity'] for e in self.equity_curve]
            peak = equity[0]
            max_dd = 0
            dd_start = 0
            max_dd_duration = 0
            current_dd_start = None
            
            for i, e in enumerate(equity):
                if e > peak:
                    peak = e
                    if current_dd_start is not None:
                        duration = i - current_dd_start
                        max_dd_duration = max(max_dd_duration, duration)
                    current_dd_start = None
                else:
                    dd = (peak - e) / peak * 100
                    if dd > max_dd:
                        max_dd = dd
                    if current_dd_start is None:
                        current_dd_start = i
            
            result.max_drawdown_pct = max_dd
            # Estimate days from candle count (assuming 1h candles)
            result.max_drawdown_duration_days = max_dd_duration // 24
        
        # Sharpe & Sortino ratios
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i-1]['equity']
                curr_equity = self.equity_curve[i]['equity']
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)
            
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                
                # Annualize (assuming hourly data)
                annual_factor = np.sqrt(24 * 365)
                
                if std_return > 0:
                    result.sharpe_ratio = (avg_return / std_return) * annual_factor
                
                # Sortino (downside deviation only)
                negative_returns = [r for r in returns if r < 0]
                if negative_returns:
                    downside_std = np.std(negative_returns)
                    if downside_std > 0:
                        result.sortino_ratio = (avg_return / downside_std) * annual_factor
        
        # Calmar ratio
        if result.max_drawdown_pct > 0:
            result.calmar_ratio = result.annualized_return_pct / result.max_drawdown_pct
        
        return result


async def run_quick_backtest(symbol: str = "BTC/USDT", days: int = 90):
    """Quick backtest runner for testing."""
    engine = BacktestEngine(
        initial_capital=10000,
        fee_pct=0.001,
        slippage_pct=0.0005
    )
    
    result = await engine.run_backtest(symbol, days)
    result.print_summary()
    
    return result


if __name__ == "__main__":
    asyncio.run(run_quick_backtest())


