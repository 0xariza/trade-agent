"""
Agent Memory System - Provides context from past trades and decisions.

This module gives the LLM agent memory of:
1. Recent trades (wins/losses and why)
2. Current open positions
3. Performance statistics per symbol
4. Recent market conditions and how agent reacted
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class TradeMemory:
    """Memory of a single trade."""
    symbol: str
    side: str  # long/short
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # stop_loss, take_profit, signal, time_stop
    entry_reasoning: str
    market_regime: str
    hold_duration_hours: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_prompt_string(self) -> str:
        """Format for LLM prompt."""
        emoji = "✅" if self.pnl > 0 else "❌"
        return (
            f"{emoji} {self.symbol} {self.side.upper()}: "
            f"${self.entry_price:,.0f} → ${self.exit_price:,.0f} | "
            f"P&L: {self.pnl_pct:+.1f}% | "
            f"Exit: {self.exit_reason} | "
            f"Held: {self.hold_duration_hours:.0f}h"
        )


@dataclass
class DecisionMemory:
    """Memory of a trading decision (including holds)."""
    symbol: str
    decision: str  # buy, sell, hold
    confidence: str
    reasoning: str
    market_snapshot: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    outcome: Optional[str] = None  # What happened after? (used for reflection)


class AgentMemory:
    """
    Manages agent memory for contextual trading decisions.
    
    Provides the LLM with:
    - Recent trade history (what worked, what didn't)
    - Current open positions (what we're holding)
    - Performance stats (win rate, best/worst symbols)
    - Recent decisions (to avoid flip-flopping)
    """
    
    def __init__(
        self,
        max_trade_memory: int = 50,
        max_decision_memory: int = 20,
        performance_lookback_days: int = 30
    ):
        self.max_trade_memory = max_trade_memory
        self.max_decision_memory = max_decision_memory
        self.performance_lookback_days = performance_lookback_days
        
        # Trade history (deque for efficient append/pop)
        self.trades: deque[TradeMemory] = deque(maxlen=max_trade_memory)
        
        # Recent decisions per symbol
        self.decisions: Dict[str, deque[DecisionMemory]] = {}
        
        # Performance stats per symbol
        self.performance: Dict[str, Dict[str, Any]] = {}
        
        # Current open positions (synced from exchange)
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        
        # Lessons learned (from reflection)
        self.lessons: List[str] = []
        
        logger.info(f"Agent Memory initialized. Max trades: {max_trade_memory}")
    
    def add_trade(self, trade: TradeMemory):
        """Record a completed trade."""
        self.trades.append(trade)
        self._update_performance(trade)
        logger.info(f"Trade recorded: {trade.symbol} {trade.side} P&L: {trade.pnl_pct:+.1f}%")
    
    def add_trade_from_dict(self, trade_dict: Dict[str, Any]):
        """Record a trade from dictionary (e.g., from closed position)."""
        trade = TradeMemory(
            symbol=trade_dict['symbol'],
            side=trade_dict['side'],
            entry_price=trade_dict['entry_price'],
            exit_price=trade_dict['exit_price'],
            pnl=trade_dict['pnl'],
            pnl_pct=trade_dict['pnl_pct'],
            exit_reason=trade_dict['exit_reason'],
            entry_reasoning=trade_dict.get('entry_reasoning', ''),
            market_regime=trade_dict.get('market_regime', 'unknown'),
            hold_duration_hours=trade_dict.get('hold_duration_hours', 0),
            timestamp=datetime.fromisoformat(trade_dict['exit_time']) if 'exit_time' in trade_dict else datetime.now()
        )
        self.add_trade(trade)
    
    def add_decision(self, symbol: str, decision: DecisionMemory):
        """Record a trading decision."""
        if symbol not in self.decisions:
            self.decisions[symbol] = deque(maxlen=self.max_decision_memory)
        self.decisions[symbol].append(decision)
    
    def update_open_positions(self, positions: Dict[str, Any]):
        """Sync open positions from exchange."""
        self.open_positions = {}
        for symbol, pos in positions.items():
            if hasattr(pos, 'to_dict'):
                self.open_positions[symbol] = pos.to_dict()
            else:
                self.open_positions[symbol] = pos
    
    def _update_performance(self, trade: TradeMemory):
        """Update performance statistics for a symbol."""
        symbol = trade.symbol
        
        if symbol not in self.performance:
            self.performance[symbol] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'total_pnl_pct': 0.0,
                'best_trade': None,
                'worst_trade': None,
                'avg_hold_hours': 0.0,
                'stop_loss_count': 0,
                'take_profit_count': 0,
                'signal_close_count': 0
            }
        
        perf = self.performance[symbol]
        perf['total_trades'] += 1
        perf['total_pnl'] += trade.pnl
        perf['total_pnl_pct'] += trade.pnl_pct
        
        if trade.pnl > 0:
            perf['winning_trades'] += 1
        else:
            perf['losing_trades'] += 1
        
        # Track best/worst
        if perf['best_trade'] is None or trade.pnl > perf['best_trade']['pnl']:
            perf['best_trade'] = {'pnl': trade.pnl, 'pnl_pct': trade.pnl_pct}
        if perf['worst_trade'] is None or trade.pnl < perf['worst_trade']['pnl']:
            perf['worst_trade'] = {'pnl': trade.pnl, 'pnl_pct': trade.pnl_pct}
        
        # Track exit reasons
        if trade.exit_reason == 'stop_loss':
            perf['stop_loss_count'] += 1
        elif trade.exit_reason == 'take_profit':
            perf['take_profit_count'] += 1
        elif trade.exit_reason == 'signal':
            perf['signal_close_count'] += 1
        
        # Update average hold time
        total_hold = perf['avg_hold_hours'] * (perf['total_trades'] - 1) + trade.hold_duration_hours
        perf['avg_hold_hours'] = total_hold / perf['total_trades']
    
    def get_recent_trades(self, symbol: str = None, n: int = 10) -> List[TradeMemory]:
        """Get recent trades, optionally filtered by symbol."""
        trades = list(self.trades)
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        return trades[-n:]
    
    def get_win_rate(self, symbol: str = None) -> float:
        """Get win rate, optionally for specific symbol."""
        if symbol and symbol in self.performance:
            perf = self.performance[symbol]
            if perf['total_trades'] > 0:
                return (perf['winning_trades'] / perf['total_trades']) * 100
            return 0.0
        
        # Overall win rate
        total = sum(p['total_trades'] for p in self.performance.values())
        wins = sum(p['winning_trades'] for p in self.performance.values())
        return (wins / total * 100) if total > 0 else 0.0
    
    def get_last_decision(self, symbol: str) -> Optional[DecisionMemory]:
        """Get the last decision for a symbol."""
        if symbol in self.decisions and self.decisions[symbol]:
            return self.decisions[symbol][-1]
        return None
    
    def add_lesson(self, lesson: str):
        """Add a lesson learned (from reflection)."""
        self.lessons.append(lesson)
        if len(self.lessons) > 20:
            self.lessons = self.lessons[-20:]  # Keep last 20
    
    def build_context_for_prompt(
        self,
        symbol: str,
        include_all_symbols: bool = False
    ) -> str:
        """
        Build context string to include in LLM prompt.
        
        This gives the agent memory of past performance and current state.
        """
        sections = []
        
        # 1. Current Position
        if symbol in self.open_positions:
            pos = self.open_positions[symbol]
            sections.append(
                f"📍 CURRENT POSITION:\n"
                f"  {symbol}: {pos['side'].upper()} {pos['amount']:.4f} @ ${pos['entry_price']:,.2f}\n"
                f"  Stop-Loss: ${pos['stop_loss']:,.2f} | Take-Profit: ${pos['take_profit']:,.2f}\n"
                f"  Entry Reason: {pos.get('entry_reasoning', 'N/A')[:100]}"
            )
        else:
            sections.append(f"📍 CURRENT POSITION: None for {symbol}")
        
        # 2. Recent Trades for this symbol
        recent = self.get_recent_trades(symbol, n=5)
        if recent:
            trade_lines = [t.to_prompt_string() for t in recent]
            sections.append(
                f"📊 RECENT TRADES ({symbol}):\n" + "\n".join(f"  {line}" for line in trade_lines)
            )
        
        # 3. Performance Stats for this symbol
        if symbol in self.performance:
            perf = self.performance[symbol]
            win_rate = (perf['winning_trades'] / perf['total_trades'] * 100) if perf['total_trades'] > 0 else 0
            sections.append(
                f"📈 PERFORMANCE ({symbol}):\n"
                f"  Trades: {perf['total_trades']} | Win Rate: {win_rate:.0f}%\n"
                f"  Total P&L: ${perf['total_pnl']:,.2f} ({perf['total_pnl_pct']:+.1f}%)\n"
                f"  Avg Hold: {perf['avg_hold_hours']:.1f}h\n"
                f"  Exits: SL={perf['stop_loss_count']} | TP={perf['take_profit_count']} | Signal={perf['signal_close_count']}"
            )
        
        # 4. Last Decision (to avoid flip-flopping)
        last_decision = self.get_last_decision(symbol)
        if last_decision:
            time_ago = (datetime.now() - last_decision.timestamp).total_seconds() / 60
            sections.append(
                f"⏱️ LAST DECISION ({time_ago:.0f} min ago):\n"
                f"  {last_decision.decision.upper()} ({last_decision.confidence})\n"
                f"  Reason: {last_decision.reasoning[:100]}..."
            )
        
        # 5. Overall Portfolio Performance
        if include_all_symbols and self.performance:
            total_trades = sum(p['total_trades'] for p in self.performance.values())
            total_wins = sum(p['winning_trades'] for p in self.performance.values())
            total_pnl = sum(p['total_pnl'] for p in self.performance.values())
            
            if total_trades > 0:
                sections.append(
                    f"💼 PORTFOLIO PERFORMANCE:\n"
                    f"  Total Trades: {total_trades} | Overall Win Rate: {(total_wins/total_trades*100):.0f}%\n"
                    f"  Total P&L: ${total_pnl:,.2f}"
                )
        
        # 6. Lessons Learned
        if self.lessons:
            recent_lessons = self.lessons[-3:]  # Last 3 lessons
            sections.append(
                f"🎓 LESSONS LEARNED:\n" + "\n".join(f"  - {lesson}" for lesson in recent_lessons)
            )
        
        # 7. Open Positions Count
        open_count = len(self.open_positions)
        if open_count > 0:
            symbols = list(self.open_positions.keys())
            sections.append(f"📌 OPEN POSITIONS: {open_count} ({', '.join(symbols)})")
        
        return "\n\n".join(sections)
    
    def get_trading_rules_from_memory(self) -> str:
        """
        Generate trading rules based on past performance.
        
        E.g., "Avoid buying BTC when RSI > 65, last 3 trades lost"
        """
        rules = []
        
        for symbol, perf in self.performance.items():
            if perf['total_trades'] < 5:
                continue
            
            win_rate = (perf['winning_trades'] / perf['total_trades']) * 100
            
            # Low win rate warning
            if win_rate < 40:
                rules.append(f"⚠️ {symbol}: Low win rate ({win_rate:.0f}%). Be extra cautious.")
            
            # Too many stop losses
            if perf['stop_loss_count'] > perf['take_profit_count'] * 2:
                rules.append(f"⚠️ {symbol}: Many stop-losses. Consider wider stops or smaller positions.")
            
            # Short hold times with losses
            if perf['avg_hold_hours'] < 2 and win_rate < 50:
                rules.append(f"⚠️ {symbol}: Quick exits with losses. Wait for better entries.")
        
        # Check recent losing streak
        recent_trades = list(self.trades)[-5:]
        losses = [t for t in recent_trades if t.pnl < 0]
        if len(losses) >= 4:
            rules.append("🛑 LOSING STREAK: 4+ recent losses. Consider reducing position size.")
        
        return "\n".join(rules) if rules else "No specific warnings based on history."
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory to dictionary (for persistence)."""
        return {
            'trades': [
                {
                    'symbol': t.symbol,
                    'side': t.side,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'exit_reason': t.exit_reason,
                    'entry_reasoning': t.entry_reasoning,
                    'market_regime': t.market_regime,
                    'hold_duration_hours': t.hold_duration_hours,
                    'timestamp': t.timestamp.isoformat()
                }
                for t in self.trades
            ],
            'performance': self.performance,
            'lessons': self.lessons,
            'open_positions': self.open_positions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMemory':
        """Restore memory from dictionary."""
        memory = cls()
        
        for t in data.get('trades', []):
            trade = TradeMemory(
                symbol=t['symbol'],
                side=t['side'],
                entry_price=t['entry_price'],
                exit_price=t['exit_price'],
                pnl=t['pnl'],
                pnl_pct=t['pnl_pct'],
                exit_reason=t['exit_reason'],
                entry_reasoning=t.get('entry_reasoning', ''),
                market_regime=t.get('market_regime', 'unknown'),
                hold_duration_hours=t.get('hold_duration_hours', 0),
                timestamp=datetime.fromisoformat(t['timestamp'])
            )
            memory.trades.append(trade)
        
        memory.performance = data.get('performance', {})
        memory.lessons = data.get('lessons', [])
        memory.open_positions = data.get('open_positions', {})
        
        return memory


