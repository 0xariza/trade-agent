"""
Performance Analytics Module for Alpha Arena.

Calculates and tracks key trading metrics:
- Sharpe Ratio
- Sortino Ratio
- Win Rate by symbol/timeframe/regime
- Profit Factor
- Maximum Drawdown
- Trade Analysis by exit reason

This is essential for Week 1-2 validation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    
    # Return metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    
    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown metrics
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    avg_drawdown_pct: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L metrics
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Duration metrics
    avg_hold_hours: float = 0.0
    avg_winning_hold_hours: float = 0.0
    avg_losing_hold_hours: float = 0.0
    
    # Exit analysis
    stop_loss_count: int = 0
    take_profit_count: int = 0
    signal_exit_count: int = 0
    time_stop_count: int = 0
    
    # Streak metrics
    current_streak: int = 0  # + for wins, - for losses
    longest_win_streak: int = 0
    longest_lose_streak: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "return_metrics": {
                "total_return_pct": round(self.total_return_pct, 2),
                "annualized_return_pct": round(self.annualized_return_pct, 2),
            },
            "risk_adjusted": {
                "sharpe_ratio": round(self.sharpe_ratio, 2),
                "sortino_ratio": round(self.sortino_ratio, 2),
                "calmar_ratio": round(self.calmar_ratio, 2),
            },
            "drawdown": {
                "max_drawdown_pct": round(self.max_drawdown_pct, 2),
                "current_drawdown_pct": round(self.current_drawdown_pct, 2),
                "max_duration_days": self.max_drawdown_duration_days,
            },
            "trade_stats": {
                "total_trades": self.total_trades,
                "win_rate": round(self.win_rate, 1),
                "profit_factor": round(self.profit_factor, 2),
                "expectancy": round(self.expectancy, 2),
            },
            "pnl": {
                "total_pnl": round(self.total_pnl, 2),
                "avg_win": round(self.avg_win, 2),
                "avg_loss": round(self.avg_loss, 2),
                "largest_win": round(self.largest_win, 2),
                "largest_loss": round(self.largest_loss, 2),
            },
            "exits": {
                "stop_loss": self.stop_loss_count,
                "take_profit": self.take_profit_count,
                "signal": self.signal_exit_count,
                "time_stop": self.time_stop_count,
            },
            "streaks": {
                "current": self.current_streak,
                "longest_win": self.longest_win_streak,
                "longest_lose": self.longest_lose_streak,
            }
        }
    
    def print_report(self):
        """Print a formatted performance report."""
        print("\n" + "=" * 70)
        print("📊 PERFORMANCE ANALYTICS REPORT")
        print("=" * 70)
        
        # Returns
        print("\n📈 RETURNS")
        print("-" * 40)
        print(f"  Total Return: {self.total_return_pct:+.2f}%")
        print(f"  Annualized Return: {self.annualized_return_pct:+.2f}%")
        
        # Risk-Adjusted
        print("\n⚖️ RISK-ADJUSTED METRICS")
        print("-" * 40)
        print(f"  Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {self.sortino_ratio:.2f}")
        print(f"  Calmar Ratio: {self.calmar_ratio:.2f}")
        
        # Sharpe interpretation
        if self.sharpe_ratio >= 2.0:
            print("  → Excellent risk-adjusted returns ✅")
        elif self.sharpe_ratio >= 1.0:
            print("  → Good risk-adjusted returns")
        elif self.sharpe_ratio >= 0.5:
            print("  → Acceptable, room for improvement")
        else:
            print("  → Poor risk-adjusted returns ⚠️")
        
        # Drawdown
        print("\n📉 DRAWDOWN")
        print("-" * 40)
        print(f"  Max Drawdown: {self.max_drawdown_pct:.2f}%")
        print(f"  Current Drawdown: {self.current_drawdown_pct:.2f}%")
        print(f"  Max DD Duration: {self.max_drawdown_duration_days} days")
        
        # Trade Stats
        print("\n🎯 TRADE STATISTICS")
        print("-" * 40)
        print(f"  Total Trades: {self.total_trades}")
        print(f"  Win Rate: {self.win_rate:.1f}%")
        print(f"  Profit Factor: {self.profit_factor:.2f}")
        print(f"  Expectancy: ${self.expectancy:.2f} per trade")
        
        # P&L
        print("\n💰 P&L ANALYSIS")
        print("-" * 40)
        print(f"  Total P&L: ${self.total_pnl:,.2f}")
        print(f"  Avg Win: ${self.avg_win:,.2f}")
        print(f"  Avg Loss: ${self.avg_loss:,.2f}")
        print(f"  Largest Win: ${self.largest_win:,.2f}")
        print(f"  Largest Loss: ${self.largest_loss:,.2f}")
        
        # Exit Analysis
        print("\n🚪 EXIT ANALYSIS")
        print("-" * 40)
        if self.total_trades > 0:
            sl_pct = (self.stop_loss_count / self.total_trades) * 100
            tp_pct = (self.take_profit_count / self.total_trades) * 100
            sig_pct = (self.signal_exit_count / self.total_trades) * 100
            ts_pct = (self.time_stop_count / self.total_trades) * 100
            print(f"  Stop-Loss: {self.stop_loss_count} ({sl_pct:.1f}%)")
            print(f"  Take-Profit: {self.take_profit_count} ({tp_pct:.1f}%)")
            print(f"  Signal Exit: {self.signal_exit_count} ({sig_pct:.1f}%)")
            print(f"  Time Stop: {self.time_stop_count} ({ts_pct:.1f}%)")
            
            # Exit analysis insights
            if sl_pct > 50:
                print("  ⚠️ High stop-loss rate - consider wider stops or better entries")
            if tp_pct > 40:
                print("  ✅ Good take-profit rate!")
        
        # Streaks
        print("\n🔥 STREAKS")
        print("-" * 40)
        print(f"  Current Streak: {self.current_streak:+d}")
        print(f"  Longest Win Streak: {self.longest_win_streak}")
        print(f"  Longest Lose Streak: {self.longest_lose_streak}")
        
        # Overall Grade
        print("\n" + "=" * 70)
        grade = self._calculate_grade()
        print(f"🏆 OVERALL GRADE: {grade}")
        print("=" * 70 + "\n")
    
    def _calculate_grade(self) -> str:
        """Calculate letter grade based on metrics."""
        score = 0
        
        # Sharpe (max 25 points)
        if self.sharpe_ratio >= 2.0:
            score += 25
        elif self.sharpe_ratio >= 1.5:
            score += 20
        elif self.sharpe_ratio >= 1.0:
            score += 15
        elif self.sharpe_ratio >= 0.5:
            score += 10
        
        # Win Rate (max 20 points)
        if self.win_rate >= 55:
            score += 20
        elif self.win_rate >= 50:
            score += 15
        elif self.win_rate >= 45:
            score += 10
        elif self.win_rate >= 40:
            score += 5
        
        # Profit Factor (max 25 points)
        if self.profit_factor >= 2.0:
            score += 25
        elif self.profit_factor >= 1.5:
            score += 20
        elif self.profit_factor >= 1.2:
            score += 15
        elif self.profit_factor >= 1.0:
            score += 5
        
        # Max Drawdown (max 20 points)
        if self.max_drawdown_pct <= 10:
            score += 20
        elif self.max_drawdown_pct <= 15:
            score += 15
        elif self.max_drawdown_pct <= 20:
            score += 10
        elif self.max_drawdown_pct <= 25:
            score += 5
        
        # Take-Profit vs Stop-Loss ratio (max 10 points)
        if self.total_trades > 0:
            tp_ratio = self.take_profit_count / max(1, self.stop_loss_count)
            if tp_ratio >= 1.5:
                score += 10
            elif tp_ratio >= 1.0:
                score += 7
            elif tp_ratio >= 0.5:
                score += 3
        
        # Convert to grade
        if score >= 90:
            return "A+ (Excellent)"
        elif score >= 80:
            return "A (Very Good)"
        elif score >= 70:
            return "B+ (Good)"
        elif score >= 60:
            return "B (Acceptable)"
        elif score >= 50:
            return "C (Needs Work)"
        elif score >= 40:
            return "D (Poor)"
        else:
            return "F (Failing)"


class PerformanceAnalyzer:
    """
    Analyzes trading performance from trade history.
    """
    
    def __init__(self, initial_capital: float = 10000.0, risk_free_rate: float = 0.04):
        """
        Args:
            initial_capital: Starting capital for return calculations
            risk_free_rate: Annual risk-free rate for Sharpe calculation (default 4%)
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
    
    def add_trade(self, trade: Dict[str, Any]):
        """Add a completed trade to the analyzer."""
        self.trades.append(trade)
    
    def add_trades(self, trades: List[Dict[str, Any]]):
        """Add multiple trades."""
        self.trades.extend(trades)
    
    def set_equity_curve(self, equity_curve: List[Dict[str, Any]]):
        """Set the equity curve for drawdown calculations."""
        self.equity_curve = equity_curve
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        metrics = PerformanceMetrics()
        
        if not self.trades:
            return metrics
        
        # Basic counts
        metrics.total_trades = len(self.trades)
        
        winning = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing = [t for t in self.trades if t.get('pnl', 0) <= 0]
        
        metrics.winning_trades = len(winning)
        metrics.losing_trades = len(losing)
        metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100
        
        # P&L metrics
        metrics.total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        
        if winning:
            wins = [t.get('pnl', 0) for t in winning]
            metrics.avg_win = sum(wins) / len(wins)
            metrics.largest_win = max(wins)
        
        if losing:
            losses = [abs(t.get('pnl', 0)) for t in losing]
            metrics.avg_loss = sum(losses) / len(losses)
            metrics.largest_loss = max(losses)
        
        # Profit Factor
        total_wins = sum(t.get('pnl', 0) for t in winning)
        total_losses = abs(sum(t.get('pnl', 0) for t in losing))
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Expectancy
        if metrics.total_trades > 0:
            metrics.expectancy = metrics.total_pnl / metrics.total_trades
        
        # Return metrics
        if self.initial_capital > 0:
            metrics.total_return_pct = (metrics.total_pnl / self.initial_capital) * 100
            
            # Annualized return
            if self.trades:
                first_trade = min(t.get('entry_time', datetime.now()) for t in self.trades 
                                  if isinstance(t.get('entry_time'), datetime))
                last_trade = max(t.get('exit_time', datetime.now()) for t in self.trades
                                 if isinstance(t.get('exit_time'), datetime))
                
                if isinstance(first_trade, str):
                    first_trade = datetime.fromisoformat(first_trade)
                if isinstance(last_trade, str):
                    last_trade = datetime.fromisoformat(last_trade)
                
                days = max(1, (last_trade - first_trade).days)
                if days > 0:
                    metrics.annualized_return_pct = (
                        (1 + metrics.total_return_pct / 100) ** (365 / days) - 1
                    ) * 100
        
        # Duration metrics
        durations = []
        win_durations = []
        lose_durations = []
        
        for t in self.trades:
            entry = t.get('entry_time')
            exit_time = t.get('exit_time')
            
            if entry and exit_time:
                if isinstance(entry, str):
                    entry = datetime.fromisoformat(entry)
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time)
                
                hours = (exit_time - entry).total_seconds() / 3600
                durations.append(hours)
                
                if t.get('pnl', 0) > 0:
                    win_durations.append(hours)
                else:
                    lose_durations.append(hours)
        
        if durations:
            metrics.avg_hold_hours = sum(durations) / len(durations)
        if win_durations:
            metrics.avg_winning_hold_hours = sum(win_durations) / len(win_durations)
        if lose_durations:
            metrics.avg_losing_hold_hours = sum(lose_durations) / len(lose_durations)
        
        # Exit reasons
        for t in self.trades:
            reason = t.get('exit_reason', '').lower()
            if 'stop' in reason:
                metrics.stop_loss_count += 1
            elif 'take' in reason or 'profit' in reason or 'tp' in reason:
                metrics.take_profit_count += 1
            elif 'signal' in reason:
                metrics.signal_exit_count += 1
            elif 'time' in reason:
                metrics.time_stop_count += 1
        
        # Streaks
        current_streak = 0
        max_win_streak = 0
        max_lose_streak = 0
        temp_streak = 0
        
        for t in self.trades:
            if t.get('pnl', 0) > 0:
                if temp_streak > 0:
                    temp_streak += 1
                else:
                    temp_streak = 1
                max_win_streak = max(max_win_streak, temp_streak)
            else:
                if temp_streak < 0:
                    temp_streak -= 1
                else:
                    temp_streak = -1
                max_lose_streak = max(max_lose_streak, abs(temp_streak))
            
            current_streak = temp_streak
        
        metrics.current_streak = current_streak
        metrics.longest_win_streak = max_win_streak
        metrics.longest_lose_streak = max_lose_streak
        
        # Drawdown calculations
        if self.equity_curve:
            self._calculate_drawdown_metrics(metrics)
        
        # Risk-adjusted metrics
        self._calculate_risk_adjusted_metrics(metrics)
        
        return metrics
    
    def _calculate_drawdown_metrics(self, metrics: PerformanceMetrics):
        """Calculate drawdown-related metrics."""
        if not self.equity_curve:
            return
        
        equity = [e.get('equity', e.get('value', 0)) for e in self.equity_curve]
        
        if not equity:
            return
        
        peak = equity[0]
        max_dd = 0
        current_dd = 0
        dd_start_idx = None
        max_dd_duration = 0
        current_dd_duration = 0
        
        for i, e in enumerate(equity):
            if e > peak:
                # New peak
                peak = e
                if dd_start_idx is not None:
                    duration = i - dd_start_idx
                    max_dd_duration = max(max_dd_duration, duration)
                dd_start_idx = None
                current_dd_duration = 0
            else:
                # In drawdown
                dd = (peak - e) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
                current_dd = dd
                
                if dd_start_idx is None:
                    dd_start_idx = i
                current_dd_duration += 1
        
        metrics.max_drawdown_pct = max_dd
        metrics.current_drawdown_pct = current_dd
        # Estimate days from data points (assuming hourly)
        metrics.max_drawdown_duration_days = max_dd_duration // 24
    
    def _calculate_risk_adjusted_metrics(self, metrics: PerformanceMetrics):
        """Calculate Sharpe, Sortino, Calmar ratios."""
        if not self.trades or metrics.total_trades < 5:
            return
        
        # Calculate daily returns from trades
        # Group trades by day
        daily_pnl = defaultdict(float)
        
        for t in self.trades:
            exit_time = t.get('exit_time')
            if exit_time:
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time)
                day = exit_time.date()
                daily_pnl[day] += t.get('pnl', 0)
        
        if not daily_pnl:
            return
        
        # Convert to returns
        capital = self.initial_capital
        daily_returns = []
        
        for day in sorted(daily_pnl.keys()):
            pnl = daily_pnl[day]
            if capital > 0:
                ret = pnl / capital
                daily_returns.append(ret)
                capital += pnl
        
        if len(daily_returns) < 5:
            return
        
        metrics.daily_returns = daily_returns
        
        # Sharpe Ratio
        avg_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        # Annualize (252 trading days)
        annual_return = avg_return * 252
        annual_std = std_return * np.sqrt(252)
        
        if annual_std > 0:
            metrics.sharpe_ratio = (annual_return - self.risk_free_rate) / annual_std
        
        # Sortino Ratio (using downside deviation)
        negative_returns = [r for r in daily_returns if r < 0]
        if negative_returns:
            downside_std = np.std(negative_returns) * np.sqrt(252)
            if downside_std > 0:
                metrics.sortino_ratio = (annual_return - self.risk_free_rate) / downside_std
        
        # Calmar Ratio
        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.annualized_return_pct / metrics.max_drawdown_pct
    
    def analyze_by_symbol(self) -> Dict[str, PerformanceMetrics]:
        """Analyze performance grouped by symbol."""
        by_symbol = defaultdict(list)
        
        for t in self.trades:
            symbol = t.get('symbol', 'UNKNOWN')
            by_symbol[symbol].append(t)
        
        results = {}
        for symbol, trades in by_symbol.items():
            analyzer = PerformanceAnalyzer(self.initial_capital)
            analyzer.add_trades(trades)
            results[symbol] = analyzer.calculate_metrics()
        
        return results
    
    def analyze_by_regime(self) -> Dict[str, PerformanceMetrics]:
        """Analyze performance grouped by market regime."""
        by_regime = defaultdict(list)
        
        for t in self.trades:
            regime = t.get('market_regime', 'unknown')
            by_regime[regime].append(t)
        
        results = {}
        for regime, trades in by_regime.items():
            analyzer = PerformanceAnalyzer(self.initial_capital)
            analyzer.add_trades(trades)
            results[regime] = analyzer.calculate_metrics()
        
        return results
    
    def analyze_by_exit_reason(self) -> Dict[str, Dict[str, Any]]:
        """Analyze trade outcomes by exit reason."""
        by_reason = defaultdict(list)
        
        for t in self.trades:
            reason = t.get('exit_reason', 'unknown')
            by_reason[reason].append(t)
        
        results = {}
        for reason, trades in by_reason.items():
            pnls = [t.get('pnl', 0) for t in trades]
            pnl_pcts = [t.get('pnl_pct', 0) for t in trades]
            
            results[reason] = {
                'count': len(trades),
                'total_pnl': sum(pnls),
                'avg_pnl': sum(pnls) / len(pnls) if pnls else 0,
                'avg_pnl_pct': sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0,
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls) * 100 if pnls else 0
            }
        
        return results
    
    def get_losing_patterns(self) -> Dict[str, Any]:
        """
        Identify patterns in losing trades.
        
        This is crucial for Phase 1 validation - understanding WHY trades fail.
        """
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        
        if not losing_trades:
            return {"message": "No losing trades found"}
        
        patterns = {
            "total_losing_trades": len(losing_trades),
            "total_loss": sum(t.get('pnl', 0) for t in losing_trades),
            
            # By symbol
            "by_symbol": {},
            
            # By entry conditions
            "high_rsi_entries": 0,  # RSI > 60 at entry
            "low_adx_entries": 0,   # ADX < 20 at entry
            "overbought_entries": 0,  # RSI > 65
            
            # By exit type
            "quick_stop_outs": 0,  # Stopped out in < 2 hours
            "slow_bleed": 0,  # Lost over 12+ hours
            
            # Common issues
            "issues_identified": []
        }
        
        # Analyze each losing trade
        symbol_losses = defaultdict(lambda: {"count": 0, "loss": 0})
        
        for t in losing_trades:
            symbol = t.get('symbol', 'UNKNOWN')
            symbol_losses[symbol]['count'] += 1
            symbol_losses[symbol]['loss'] += t.get('pnl', 0)
            
            # Entry condition analysis (if available in trade data)
            entry_rsi = t.get('entry_rsi', t.get('rsi', 50))
            entry_adx = t.get('entry_adx', t.get('adx', 25))
            
            if entry_rsi > 60:
                patterns['high_rsi_entries'] += 1
            if entry_rsi > 65:
                patterns['overbought_entries'] += 1
            if entry_adx < 20:
                patterns['low_adx_entries'] += 1
            
            # Duration analysis
            hold_hours = t.get('hold_duration_hours', 0)
            if hold_hours < 2:
                patterns['quick_stop_outs'] += 1
            elif hold_hours > 12:
                patterns['slow_bleed'] += 1
        
        patterns['by_symbol'] = dict(symbol_losses)
        
        # Identify top issues
        issues = []
        
        if patterns['high_rsi_entries'] > len(losing_trades) * 0.4:
            issues.append(f"⚠️ {patterns['high_rsi_entries']} trades entered with RSI > 60 - Consider stricter RSI filter")
        
        if patterns['low_adx_entries'] > len(losing_trades) * 0.3:
            issues.append(f"⚠️ {patterns['low_adx_entries']} trades entered with ADX < 20 - Avoid ranging markets")
        
        if patterns['quick_stop_outs'] > len(losing_trades) * 0.3:
            issues.append(f"⚠️ {patterns['quick_stop_outs']} quick stop-outs - Stops may be too tight")
        
        if patterns['overbought_entries'] > 0:
            issues.append(f"⚠️ {patterns['overbought_entries']} entries in overbought conditions - Never buy RSI > 65")
        
        # Symbol-specific issues
        worst_symbol = max(symbol_losses.items(), key=lambda x: abs(x[1]['loss'])) if symbol_losses else None
        if worst_symbol and worst_symbol[1]['count'] >= 3:
            issues.append(f"⚠️ {worst_symbol[0]} has {worst_symbol[1]['count']} losses (${abs(worst_symbol[1]['loss']):,.0f}) - Review this asset")
        
        patterns['issues_identified'] = issues
        
        return patterns
    
    def export_report(self, filepath: str):
        """Export full performance report to JSON."""
        metrics = self.calculate_metrics()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "initial_capital": self.initial_capital,
            "overall_metrics": metrics.to_dict(),
            "by_symbol": {
                symbol: m.to_dict() 
                for symbol, m in self.analyze_by_symbol().items()
            },
            "by_regime": {
                regime: m.to_dict()
                for regime, m in self.analyze_by_regime().items()
            },
            "by_exit_reason": self.analyze_by_exit_reason(),
            "losing_patterns": self.get_losing_patterns(),
            "total_trades_analyzed": len(self.trades)
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report exported to {filepath}")
        return filepath

