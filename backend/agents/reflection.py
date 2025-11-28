"""
Self-Reflection System for Trading Agent.

Analyzes closed trades to:
1. Identify patterns in losing trades
2. Generate actionable lessons
3. Adjust trading behavior based on past mistakes
4. Build a knowledge base of what works/doesn't work

This implements a "post-trade analysis" loop that professional traders do manually.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class TradeOutcome(Enum):
    """Trade outcome classification."""
    BIG_WIN = "big_win"          # > 3% profit
    SMALL_WIN = "small_win"      # 0-3% profit
    SMALL_LOSS = "small_loss"    # 0-3% loss
    BIG_LOSS = "big_loss"        # > 3% loss
    STOPPED_OUT = "stopped_out"  # Hit stop-loss
    TIME_STOPPED = "time_stopped"  # Held too long


@dataclass
class ReflectionInsight:
    """A single insight from trade reflection."""
    category: str  # "entry_timing", "stop_placement", "position_sizing", etc.
    insight: str   # The actual insight text
    severity: str  # "info", "warning", "critical"
    actionable_rule: str  # What to do differently
    confidence: float  # 0-1, how confident we are in this insight
    supporting_trades: int  # How many trades support this insight
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_prompt_string(self) -> str:
        """Format for LLM prompt."""
        emoji = {"info": "💡", "warning": "⚠️", "critical": "🛑"}.get(self.severity, "📝")
        return f"{emoji} [{self.category.upper()}] {self.insight}\n   → {self.actionable_rule}"


@dataclass 
class TradeAnalysis:
    """Analysis of a single trade."""
    trade_data: Dict[str, Any]
    outcome: TradeOutcome
    issues_identified: List[str]
    what_went_well: List[str]
    market_context: Dict[str, Any]
    
    def was_preventable(self) -> bool:
        """Was this loss preventable based on available information?"""
        return len(self.issues_identified) > 0


class TradeReflector:
    """
    Analyzes trades and generates lessons learned.
    
    Features:
    - Pattern detection in losing trades
    - Rule-based issue identification
    - Batch analysis for pattern discovery
    - Lesson generation for agent prompts
    - Performance tracking by condition
    """
    
    def __init__(
        self,
        min_trades_for_pattern: int = 3,
        lookback_trades: int = 50,
        reflection_after_n_losses: int = 3
    ):
        self.min_trades_for_pattern = min_trades_for_pattern
        self.lookback_trades = lookback_trades
        self.reflection_after_n_losses = reflection_after_n_losses
        
        # Store all analyses
        self.trade_analyses: List[TradeAnalysis] = []
        
        # Generated insights
        self.insights: List[ReflectionInsight] = []
        
        # Pattern tracking
        self.patterns: Dict[str, Dict[str, Any]] = {
            'entry_patterns': defaultdict(list),  # What conditions led to entries
            'exit_patterns': defaultdict(list),   # What caused exits
            'regime_performance': defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0}),
            'time_performance': defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0}),
            'indicator_states': defaultdict(lambda: {'wins': 0, 'losses': 0})
        }
        
        # Consecutive loss tracker for triggering reflection
        self.consecutive_losses = 0
        self.last_reflection_time = datetime.now()
        
        logger.info("TradeReflector initialized")
    
    def analyze_trade(self, trade: Dict[str, Any], market_data: Dict[str, Any] = None) -> TradeAnalysis:
        """
        Analyze a single closed trade.
        
        Args:
            trade: Closed trade data including entry/exit prices, PnL, exit reason
            market_data: Market conditions at entry/exit
        
        Returns:
            TradeAnalysis with identified issues and what went well
        """
        # Classify outcome
        pnl_pct = trade.get('pnl_pct', 0)
        exit_reason = trade.get('exit_reason', 'unknown')
        
        if exit_reason == 'stop_loss':
            outcome = TradeOutcome.STOPPED_OUT
        elif exit_reason == 'time_stop':
            outcome = TradeOutcome.TIME_STOPPED
        elif pnl_pct > 3:
            outcome = TradeOutcome.BIG_WIN
        elif pnl_pct > 0:
            outcome = TradeOutcome.SMALL_WIN
        elif pnl_pct > -3:
            outcome = TradeOutcome.SMALL_LOSS
        else:
            outcome = TradeOutcome.BIG_LOSS
        
        # Identify issues
        issues = self._identify_issues(trade, market_data, outcome)
        
        # Identify what went well
        positives = self._identify_positives(trade, market_data, outcome)
        
        # Create analysis
        analysis = TradeAnalysis(
            trade_data=trade,
            outcome=outcome,
            issues_identified=issues,
            what_went_well=positives,
            market_context=market_data or {}
        )
        
        self.trade_analyses.append(analysis)
        
        # Update patterns
        self._update_patterns(trade, outcome, market_data)
        
        # Track consecutive losses
        if outcome in [TradeOutcome.BIG_LOSS, TradeOutcome.SMALL_LOSS, TradeOutcome.STOPPED_OUT]:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Trigger reflection if needed
        if self.consecutive_losses >= self.reflection_after_n_losses:
            self._trigger_reflection("consecutive_losses")
        
        return analysis
    
    def _identify_issues(
        self,
        trade: Dict[str, Any],
        market_data: Dict[str, Any],
        outcome: TradeOutcome
    ) -> List[str]:
        """Identify potential issues with a trade."""
        issues = []
        
        if outcome in [TradeOutcome.BIG_WIN, TradeOutcome.SMALL_WIN]:
            return issues  # No issues for winning trades
        
        # 1. Entry Timing Issues
        entry_reasoning = trade.get('entry_reasoning', '').lower()
        
        # Chasing momentum
        if market_data:
            change_24h = market_data.get('change_24h', 0)
            if trade.get('side') == 'long' and change_24h > 5:
                issues.append("Entered long after large up move (+5%), possible top chase")
            elif trade.get('side') == 'short' and change_24h < -5:
                issues.append("Entered short after large down move (-5%), possible bottom chase")
        
        # 2. Stop-Loss Issues
        if outcome == TradeOutcome.STOPPED_OUT:
            stop_distance_pct = abs((trade.get('stop_loss', 0) - trade.get('entry_price', 1)) / trade.get('entry_price', 1)) * 100
            
            if stop_distance_pct < 1:
                issues.append(f"Stop-loss too tight ({stop_distance_pct:.1f}%), caught by noise")
            elif stop_distance_pct > 5:
                issues.append(f"Stop-loss too wide ({stop_distance_pct:.1f}%), excessive risk")
        
        # 3. Position Sizing Issues
        # If big loss, check if position was too large
        if outcome == TradeOutcome.BIG_LOSS:
            pnl = trade.get('pnl', 0)
            if abs(pnl) > 500:  # Arbitrary threshold
                issues.append(f"Large absolute loss (${abs(pnl):.0f}), position may have been too large")
        
        # 4. Market Regime Mismatch
        if market_data:
            regime = market_data.get('market_regime', 'unknown').lower()
            adx = market_data.get('adx', 0)
            
            if regime == 'ranging' and 'breakout' in entry_reasoning:
                issues.append("Traded breakout in ranging market")
            
            if adx < 20 and trade.get('side') == 'long':
                issues.append(f"Entered trend trade with weak ADX ({adx:.0f})")
        
        # 5. Indicator Conflicts
        if market_data:
            rsi = market_data.get('rsi', 50)
            
            if trade.get('side') == 'long' and rsi > 70:
                issues.append(f"Bought overbought (RSI {rsi:.0f})")
            elif trade.get('side') == 'short' and rsi < 30:
                issues.append(f"Sold oversold (RSI {rsi:.0f})")
        
        # 6. Time-Based Issues
        if outcome == TradeOutcome.TIME_STOPPED:
            hold_hours = trade.get('hold_duration_hours', 0)
            issues.append(f"Position held too long ({hold_hours:.0f}h), no exit triggered")
        
        # 7. Exit Timing
        if outcome == TradeOutcome.SMALL_LOSS:
            # Could have been a winner if held differently
            issues.append("Small loss - review if exit was premature or entry was mistimed")
        
        return issues
    
    def _identify_positives(
        self,
        trade: Dict[str, Any],
        market_data: Dict[str, Any],
        outcome: TradeOutcome
    ) -> List[str]:
        """Identify what went well in a trade."""
        positives = []
        
        # 1. Good risk management
        if outcome == TradeOutcome.STOPPED_OUT:
            positives.append("Stop-loss executed as planned, limited loss")
        
        if outcome == TradeOutcome.BIG_WIN:
            if trade.get('exit_reason') == 'take_profit':
                positives.append("Take-profit hit, captured full target")
            
            pnl_pct = trade.get('pnl_pct', 0)
            if pnl_pct > 5:
                positives.append(f"Exceptional trade (+{pnl_pct:.1f}%), analyze for repetition")
        
        # 2. Good timing
        if market_data:
            if trade.get('side') == 'long' and market_data.get('rsi', 50) < 35:
                positives.append("Good entry - bought oversold")
        
        # 3. Regime alignment
        if market_data:
            regime = market_data.get('market_regime', 'unknown').lower()
            if regime == 'trending' and outcome in [TradeOutcome.BIG_WIN, TradeOutcome.SMALL_WIN]:
                positives.append("Traded with the trend in trending market")
        
        return positives
    
    def _update_patterns(
        self,
        trade: Dict[str, Any],
        outcome: TradeOutcome,
        market_data: Dict[str, Any]
    ):
        """Update pattern tracking."""
        is_win = outcome in [TradeOutcome.BIG_WIN, TradeOutcome.SMALL_WIN]
        pnl = trade.get('pnl', 0)
        
        # Track by market regime
        if market_data:
            regime = market_data.get('market_regime', 'unknown')
            perf = self.patterns['regime_performance'][regime]
            perf['wins' if is_win else 'losses'] += 1
            perf['pnl'] += pnl
        
        # Track by hour of day (if available)
        entry_time = trade.get('entry_time')
        if entry_time:
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            hour = entry_time.hour
            perf = self.patterns['time_performance'][hour]
            perf['wins' if is_win else 'losses'] += 1
            perf['pnl'] += pnl
        
        # Track by indicator state
        if market_data:
            rsi = market_data.get('rsi', 50)
            rsi_bucket = 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'
            
            self.patterns['indicator_states'][f'rsi_{rsi_bucket}']['wins' if is_win else 'losses'] += 1
    
    def _trigger_reflection(self, trigger_reason: str):
        """Run reflection analysis when triggered."""
        logger.info(f"Triggering reflection due to: {trigger_reason}")
        
        # Get recent losing trades
        recent_losses = [
            a for a in self.trade_analyses[-10:]
            if a.outcome in [TradeOutcome.BIG_LOSS, TradeOutcome.SMALL_LOSS, TradeOutcome.STOPPED_OUT]
        ]
        
        if len(recent_losses) < 2:
            return
        
        # Find common issues
        issue_counts = defaultdict(int)
        for analysis in recent_losses:
            for issue in analysis.issues_identified:
                # Normalize issue text for grouping
                key = issue.split(',')[0].strip()
                issue_counts[key] += 1
        
        # Generate insights for recurring issues
        for issue, count in issue_counts.items():
            if count >= 2:  # Issue appeared in 2+ trades
                self._add_insight(
                    category="recurring_issue",
                    insight=f"{issue} (occurred in {count} recent losses)",
                    severity="warning" if count < 3 else "critical",
                    actionable_rule=self._generate_rule_for_issue(issue),
                    confidence=min(count / 5, 1.0),
                    supporting_trades=count
                )
        
        # Analyze pattern data
        self._analyze_patterns()
        
        # Reset consecutive loss counter
        self.consecutive_losses = 0
        self.last_reflection_time = datetime.now()
    
    def _generate_rule_for_issue(self, issue: str) -> str:
        """Generate an actionable rule based on identified issue."""
        issue_lower = issue.lower()
        
        if 'stop-loss too tight' in issue_lower:
            return "Widen stop-loss to at least 2x ATR to avoid noise"
        elif 'stop-loss too wide' in issue_lower:
            return "Reduce position size or tighten stop to limit risk"
        elif 'chase' in issue_lower or 'top' in issue_lower:
            return "Wait for pullback before entering, avoid chasing moves"
        elif 'overbought' in issue_lower:
            return "Do not buy when RSI > 70, wait for cooling"
        elif 'oversold' in issue_lower:
            return "Do not short when RSI < 30, wait for bounce"
        elif 'ranging' in issue_lower and 'breakout' in issue_lower:
            return "In ranging markets, trade mean reversion not breakouts"
        elif 'adx' in issue_lower:
            return "Require ADX > 25 for trend trades"
        elif 'held too long' in issue_lower:
            return "Review take-profit levels, consider partial exits"
        elif 'position' in issue_lower and 'large' in issue_lower:
            return "Reduce maximum position size during drawdowns"
        else:
            return "Review trade setup and wait for higher-conviction entries"
    
    def _analyze_patterns(self):
        """Analyze collected patterns for insights."""
        # Regime performance analysis
        for regime, perf in self.patterns['regime_performance'].items():
            total = perf['wins'] + perf['losses']
            if total >= self.min_trades_for_pattern:
                win_rate = perf['wins'] / total if total > 0 else 0
                
                if win_rate < 0.35:
                    self._add_insight(
                        category="regime_performance",
                        insight=f"Poor performance in {regime} markets ({win_rate*100:.0f}% win rate)",
                        severity="warning",
                        actionable_rule=f"Consider reducing size or sitting out {regime} markets",
                        confidence=min(total / 10, 1.0),
                        supporting_trades=total
                    )
                elif win_rate > 0.65:
                    self._add_insight(
                        category="regime_performance",
                        insight=f"Strong performance in {regime} markets ({win_rate*100:.0f}% win rate)",
                        severity="info",
                        actionable_rule=f"Increase allocation during {regime} market conditions",
                        confidence=min(total / 10, 1.0),
                        supporting_trades=total
                    )
        
        # Time-of-day analysis
        for hour, perf in self.patterns['time_performance'].items():
            total = perf['wins'] + perf['losses']
            if total >= self.min_trades_for_pattern:
                win_rate = perf['wins'] / total if total > 0 else 0
                
                if win_rate < 0.30:
                    self._add_insight(
                        category="timing",
                        insight=f"Poor performance at hour {hour}:00 ({win_rate*100:.0f}% win rate)",
                        severity="warning",
                        actionable_rule=f"Avoid trading around {hour}:00 or reduce size",
                        confidence=min(total / 8, 1.0),
                        supporting_trades=total
                    )
        
        # RSI state analysis
        for state, perf in self.patterns['indicator_states'].items():
            total = perf['wins'] + perf['losses']
            if total >= self.min_trades_for_pattern:
                win_rate = perf['wins'] / total if total > 0 else 0
                
                if 'overbought' in state and win_rate < 0.40:
                    self._add_insight(
                        category="indicator",
                        insight=f"Buying overbought conditions losing ({win_rate*100:.0f}% win rate)",
                        severity="critical",
                        actionable_rule="AVOID buying when RSI > 70",
                        confidence=min(total / 5, 1.0),
                        supporting_trades=total
                    )
    
    def _add_insight(
        self,
        category: str,
        insight: str,
        severity: str,
        actionable_rule: str,
        confidence: float,
        supporting_trades: int
    ):
        """Add a new insight, avoiding duplicates."""
        # Check for duplicate
        for existing in self.insights:
            if existing.category == category and existing.insight == insight:
                # Update if higher confidence
                if confidence > existing.confidence:
                    existing.confidence = confidence
                    existing.supporting_trades = supporting_trades
                    existing.timestamp = datetime.now()
                return
        
        # Add new insight
        insight_obj = ReflectionInsight(
            category=category,
            insight=insight,
            severity=severity,
            actionable_rule=actionable_rule,
            confidence=confidence,
            supporting_trades=supporting_trades
        )
        self.insights.append(insight_obj)
        
        logger.info(f"New insight: [{category}] {insight}")
        
        # Keep only recent insights (last 20)
        if len(self.insights) > 20:
            self.insights = sorted(
                self.insights,
                key=lambda x: (x.confidence, x.timestamp),
                reverse=True
            )[:20]
    
    def get_insights_for_prompt(self, max_insights: int = 5) -> str:
        """Get formatted insights for LLM prompt."""
        if not self.insights:
            return ""
        
        # Sort by severity and confidence
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        sorted_insights = sorted(
            self.insights,
            key=lambda x: (severity_order.get(x.severity, 3), -x.confidence)
        )[:max_insights]
        
        lines = ["🔍 SELF-REFLECTION INSIGHTS:"]
        for insight in sorted_insights:
            lines.append(f"  {insight.to_prompt_string()}")
        
        return "\n".join(lines)
    
    def get_trading_adjustments(self) -> Dict[str, Any]:
        """
        Get current trading adjustments based on insights.
        
        Returns rules that should be applied to trading decisions.
        """
        adjustments = {
            'avoid_overbought': False,
            'avoid_oversold': False,
            'require_trend': False,
            'widen_stops': False,
            'reduce_size': False,
            'avoid_regimes': [],
            'avoid_hours': [],
            'custom_rules': []
        }
        
        for insight in self.insights:
            if insight.confidence < 0.5:
                continue  # Skip low-confidence insights
            
            rule = insight.actionable_rule.lower()
            
            if 'rsi > 70' in rule or 'overbought' in rule:
                adjustments['avoid_overbought'] = True
            
            if 'rsi < 30' in rule or 'oversold' in rule:
                adjustments['avoid_oversold'] = True
            
            if 'adx' in rule or 'trend' in rule:
                adjustments['require_trend'] = True
            
            if 'widen' in rule and 'stop' in rule:
                adjustments['widen_stops'] = True
            
            if 'reduce' in rule and 'size' in rule:
                adjustments['reduce_size'] = True
            
            # Extract regime avoidance
            for regime in ['trending', 'ranging', 'volatile']:
                if regime in rule and ('avoid' in rule or 'sitting out' in rule):
                    adjustments['avoid_regimes'].append(regime)
            
            adjustments['custom_rules'].append(insight.actionable_rule)
        
        return adjustments
    
    def should_block_trade(
        self,
        side: str,
        market_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Check if a trade should be blocked based on learned rules.
        
        Returns:
            Tuple of (should_block, reason)
        """
        adjustments = self.get_trading_adjustments()
        
        rsi = market_data.get('rsi', 50)
        adx = market_data.get('adx', 25)
        regime = market_data.get('market_regime', '').lower()
        
        # Check overbought
        if side == 'long' and adjustments['avoid_overbought'] and rsi > 70:
            return True, f"Blocked: Learned not to buy overbought (RSI={rsi:.0f})"
        
        # Check oversold
        if side == 'short' and adjustments['avoid_oversold'] and rsi < 30:
            return True, f"Blocked: Learned not to sell oversold (RSI={rsi:.0f})"
        
        # Check trend requirement
        if adjustments['require_trend'] and adx < 25:
            return True, f"Blocked: Learned to require trend (ADX={adx:.0f} < 25)"
        
        # Check regime avoidance
        if regime in adjustments['avoid_regimes']:
            return True, f"Blocked: Learned to avoid {regime} markets"
        
        return False, ""
    
    def get_position_size_adjustment(self) -> float:
        """
        Get position size multiplier based on recent performance.
        
        Returns:
            Multiplier (0.5 to 1.0)
        """
        adjustments = self.get_trading_adjustments()
        
        if adjustments['reduce_size']:
            return 0.7
        
        # Reduce size during consecutive losses
        if self.consecutive_losses >= 2:
            return max(0.5, 1.0 - (self.consecutive_losses * 0.15))
        
        return 1.0
    
    def get_stop_adjustment(self, base_multiplier: float) -> float:
        """
        Adjust ATR stop multiplier based on insights.
        
        Args:
            base_multiplier: Original ATR multiplier for stops
        
        Returns:
            Adjusted multiplier
        """
        adjustments = self.get_trading_adjustments()
        
        if adjustments['widen_stops']:
            return base_multiplier * 1.25  # 25% wider
        
        return base_multiplier
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            'insights': [
                {
                    'category': i.category,
                    'insight': i.insight,
                    'severity': i.severity,
                    'actionable_rule': i.actionable_rule,
                    'confidence': i.confidence,
                    'supporting_trades': i.supporting_trades,
                    'timestamp': i.timestamp.isoformat()
                }
                for i in self.insights
            ],
            'patterns': {
                'regime_performance': dict(self.patterns['regime_performance']),
                'time_performance': {str(k): v for k, v in self.patterns['time_performance'].items()},
                'indicator_states': dict(self.patterns['indicator_states'])
            },
            'consecutive_losses': self.consecutive_losses,
            'last_reflection_time': self.last_reflection_time.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeReflector':
        """Restore from persisted data."""
        reflector = cls()
        
        for i in data.get('insights', []):
            insight = ReflectionInsight(
                category=i['category'],
                insight=i['insight'],
                severity=i['severity'],
                actionable_rule=i['actionable_rule'],
                confidence=i['confidence'],
                supporting_trades=i['supporting_trades'],
                timestamp=datetime.fromisoformat(i['timestamp'])
            )
            reflector.insights.append(insight)
        
        patterns = data.get('patterns', {})
        for regime, perf in patterns.get('regime_performance', {}).items():
            reflector.patterns['regime_performance'][regime] = perf
        for hour, perf in patterns.get('time_performance', {}).items():
            reflector.patterns['time_performance'][int(hour)] = perf
        for state, perf in patterns.get('indicator_states', {}).items():
            reflector.patterns['indicator_states'][state] = perf
        
        reflector.consecutive_losses = data.get('consecutive_losses', 0)
        if data.get('last_reflection_time'):
            reflector.last_reflection_time = datetime.fromisoformat(data['last_reflection_time'])
        
        return reflector


