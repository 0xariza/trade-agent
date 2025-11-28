"""
Spot Trading Signal Handler for Alpha Arena.

This module handles COMPLETE spot trading:
- BUY: Purchase asset with USDT
- SELL: Sell asset back to USDT

NOT futures/margin - pure spot trading only.

Signal Flow:
    Agent Decision → Pre-Trade Filters → Position Check → Execute
    
    BUY Path:
        1. Agent says BULLISH + BUY signal
        2. Check filters (ADX, RSI, BTC safety, etc.)
        3. If no existing position → Open position (BUY)
        
    SELL Path:
        1. Agent says BEARISH + SELL signal
        2. Check filters (don't sell into extreme oversold)
        3. If existing position → Close position (SELL)
"""

import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SpotAction(Enum):
    """Possible spot trading actions."""
    BUY = "buy"           # Open new position
    SELL = "sell"         # Close existing position
    HOLD = "hold"         # Do nothing
    BLOCKED = "blocked"   # Signal valid but blocked by filter


@dataclass
class SignalDecision:
    """Result of signal analysis."""
    action: SpotAction
    signal_type: str      # STRONG_BUY, MODERATE_BUY, SELL, HOLD
    confidence: str       # high, moderate, low
    position_multiplier: float  # Size multiplier (0.0 to 1.0)
    reasoning: str
    blocked_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "signal_type": self.signal_type,
            "confidence": self.confidence,
            "position_multiplier": self.position_multiplier,
            "reasoning": self.reasoning,
            "blocked_reason": self.blocked_reason
        }


class SpotSignalHandler:
    """
    Handles spot trading signals with complete BUY/SELL logic.
    
    Spot Trading Rules:
    - You can only SELL what you own
    - You can only BUY with available USDT
    - No leverage, no shorting
    """
    
    def __init__(
        self,
        # BUY filters
        min_adx_for_buy: float = 12.0,
        max_rsi_for_buy: float = 72.0,
        require_btc_safety: bool = True,
        require_macd_alignment: bool = True,
        min_timeframe_alignment: int = 2,  # 2 of 3 timeframes
        
        # SELL filters
        min_rsi_for_sell: float = 25.0,  # Don't sell into extreme oversold
        require_confirmation_for_sell: bool = True,
        
        # General
        min_confidence_for_trade: str = "moderate",  # low, moderate, high
        allow_weak_signals: bool = False
    ):
        self.min_adx_for_buy = min_adx_for_buy
        self.max_rsi_for_buy = max_rsi_for_buy
        self.require_btc_safety = require_btc_safety
        self.require_macd_alignment = require_macd_alignment
        self.min_timeframe_alignment = min_timeframe_alignment
        
        self.min_rsi_for_sell = min_rsi_for_sell
        self.require_confirmation_for_sell = require_confirmation_for_sell
        
        self.min_confidence_for_trade = min_confidence_for_trade
        self.allow_weak_signals = allow_weak_signals
        
        logger.info(
            f"SpotSignalHandler initialized. "
            f"ADX>{min_adx_for_buy}, RSI<{max_rsi_for_buy} for BUY"
        )
    
    def process_signal(
        self,
        agent_decision: Dict[str, Any],
        market_data: Dict[str, Any],
        has_position: bool,
        symbol: str
    ) -> SignalDecision:
        """
        Process an agent's decision and determine spot trading action.
        
        Args:
            agent_decision: Agent's analysis result
            market_data: Current market snapshot
            has_position: Whether we already hold this asset
            symbol: Trading symbol (e.g., "BTC/USDT")
            
        Returns:
            SignalDecision with action to take
        """
        signal_type = agent_decision.get("signal_type", "HOLD").upper()
        trend = agent_decision.get("trend", "neutral").lower()
        confidence = agent_decision.get("confidence", "low").lower()
        reasoning = agent_decision.get("reasoning", "")
        
        # Determine if this is a BUY or SELL signal
        is_buy_signal = "BUY" in signal_type and trend == "bullish"
        is_sell_signal = "SELL" in signal_type or trend == "bearish"
        
        # ============================================
        # CASE 1: BUY SIGNAL (No existing position)
        # ============================================
        if is_buy_signal and not has_position:
            passed, block_reason = self._check_buy_filters(
                symbol, market_data, signal_type, confidence
            )
            
            if not passed:
                return SignalDecision(
                    action=SpotAction.BLOCKED,
                    signal_type=signal_type,
                    confidence=confidence,
                    position_multiplier=0.0,
                    reasoning=reasoning,
                    blocked_reason=block_reason
                )
            
            # Calculate position size multiplier
            multiplier = self._calculate_buy_multiplier(signal_type, confidence)
            
            return SignalDecision(
                action=SpotAction.BUY,
                signal_type=signal_type,
                confidence=confidence,
                position_multiplier=multiplier,
                reasoning=reasoning
            )
        
        # ============================================
        # CASE 2: SELL SIGNAL (Has existing position)
        # ============================================
        elif is_sell_signal and has_position:
            passed, block_reason = self._check_sell_filters(
                symbol, market_data, signal_type, confidence
            )
            
            if not passed:
                return SignalDecision(
                    action=SpotAction.BLOCKED,
                    signal_type=signal_type,
                    confidence=confidence,
                    position_multiplier=0.0,
                    reasoning=reasoning,
                    blocked_reason=block_reason
                )
            
            return SignalDecision(
                action=SpotAction.SELL,
                signal_type=signal_type,
                confidence=confidence,
                position_multiplier=1.0,  # Full exit on SELL
                reasoning=reasoning
            )
        
        # ============================================
        # CASE 3: BUY signal but already have position
        # ============================================
        elif is_buy_signal and has_position:
            return SignalDecision(
                action=SpotAction.HOLD,
                signal_type=signal_type,
                confidence=confidence,
                position_multiplier=0.0,
                reasoning="Already holding position - cannot buy more",
                blocked_reason="Position exists"
            )
        
        # ============================================
        # CASE 4: SELL signal but no position
        # ============================================
        elif is_sell_signal and not has_position:
            return SignalDecision(
                action=SpotAction.HOLD,
                signal_type=signal_type,
                confidence=confidence,
                position_multiplier=0.0,
                reasoning="No position to sell",
                blocked_reason="No position"
            )
        
        # ============================================
        # CASE 5: HOLD signal
        # ============================================
        else:
            return SignalDecision(
                action=SpotAction.HOLD,
                signal_type=signal_type,
                confidence=confidence,
                position_multiplier=0.0,
                reasoning=reasoning or "No clear signal"
            )
    
    def _check_buy_filters(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        signal_type: str,
        confidence: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check all pre-trade filters for BUY signals.
        
        Returns:
            Tuple of (passed, block_reason)
        """
        timeframes = market_data.get('timeframes', {})
        tf_1h = timeframes.get('1h', {}).get('indicators', {})
        tf_4h = timeframes.get('4h', {}).get('indicators', {})
        tf_15m = timeframes.get('15m', {}).get('indicators', {})
        
        adx_1h = tf_1h.get('adx', 0)
        adx_4h = tf_4h.get('adx', 0)
        rsi_1h = tf_1h.get('rsi', 50)
        rsi_4h = tf_4h.get('rsi', 50)
        macd_1h = tf_1h.get('macd', 'neutral')
        macd_4h = tf_4h.get('macd', 'neutral')
        
        # FILTER 1: Confidence check
        confidence_levels = {"low": 1, "moderate": 2, "high": 3}
        min_conf = confidence_levels.get(self.min_confidence_for_trade, 2)
        current_conf = confidence_levels.get(confidence, 1)
        
        if current_conf < min_conf and "STRONG" not in signal_type:
            return False, f"Low confidence ({confidence})"
        
        # FILTER 2: Weak signal check
        if "WEAK" in signal_type and not self.allow_weak_signals:
            if adx_4h < 20:
                return False, f"Weak signal with weak trend (ADX={adx_4h:.0f})"
        
        # FILTER 3: BTC safety for altcoins
        if self.require_btc_safety and symbol != "BTC/USDT":
            btc_trend = market_data.get('btc_trend', {})
            if btc_trend and not btc_trend.get('is_safe_for_alts', True):
                return False, f"BTC bearish - not safe for alts"
        
        # FILTER 4: ADX trend strength
        if adx_4h < self.min_adx_for_buy and adx_1h < self.min_adx_for_buy:
            return False, f"No trend (ADX 1H={adx_1h:.0f}, 4H={adx_4h:.0f})"
        
        # FILTER 5: RSI overbought check
        if rsi_4h > self.max_rsi_for_buy and rsi_1h > self.max_rsi_for_buy - 5:
            return False, f"Overbought (RSI 1H={rsi_1h:.0f}, 4H={rsi_4h:.0f})"
        
        # FILTER 6: MACD alignment
        if self.require_macd_alignment:
            if macd_1h == 'bearish' and macd_4h == 'bearish':
                return False, "MACD bearish on both timeframes"
        
        # FILTER 7: Timeframe alignment
        bullish_count = 0
        for tf_ind in [tf_15m, tf_1h, tf_4h]:
            if tf_ind.get('macd') == 'bullish':
                bullish_count += 1
            elif tf_ind.get('rsi', 50) < 45:
                bullish_count += 0.5  # Partial credit for oversold
        
        if bullish_count < self.min_timeframe_alignment:
            return False, f"Poor timeframe alignment ({bullish_count:.1f}/3 bullish)"
        
        return True, None
    
    def _check_sell_filters(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        signal_type: str,
        confidence: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check filters for SELL signals.
        
        We want to SELL when:
        - Trend has reversed
        - Take profit target hit
        - Stop loss triggered
        
        We DON'T want to sell into:
        - Extreme oversold (potential bounce)
        """
        timeframes = market_data.get('timeframes', {})
        tf_1h = timeframes.get('1h', {}).get('indicators', {})
        tf_4h = timeframes.get('4h', {}).get('indicators', {})
        
        rsi_1h = tf_1h.get('rsi', 50)
        rsi_4h = tf_4h.get('rsi', 50)
        macd_1h = tf_1h.get('macd', 'neutral')
        macd_4h = tf_4h.get('macd', 'neutral')
        
        # FILTER 1: Don't sell into extreme oversold
        if rsi_4h < self.min_rsi_for_sell and rsi_1h < self.min_rsi_for_sell + 5:
            return False, f"Too oversold to sell (RSI 1H={rsi_1h:.0f}, 4H={rsi_4h:.0f})"
        
        # FILTER 2: Require some confirmation for sell
        if self.require_confirmation_for_sell:
            # At least one timeframe should show bearish MACD
            if macd_1h != 'bearish' and macd_4h != 'bearish':
                # Unless RSI shows we're overbought (should exit)
                if rsi_4h < 60:
                    return False, "No bearish confirmation for sell"
        
        return True, None
    
    def _calculate_buy_multiplier(
        self,
        signal_type: str,
        confidence: str
    ) -> float:
        """
        Calculate position size multiplier based on signal strength.
        
        Returns:
            Float between 0.3 and 1.0
        """
        # Base on signal type
        if "STRONG" in signal_type:
            base = 1.0
        elif "MODERATE" in signal_type:
            base = 0.7
        elif "WEAK" in signal_type:
            base = 0.4
        elif "OVERSOLD" in signal_type:
            base = 0.5  # Reversals are risky
        else:
            base = 0.5
        
        # Adjust by confidence
        if confidence == "high":
            base *= 1.0
        elif confidence == "moderate":
            base *= 0.85
        else:  # low
            base *= 0.6
        
        return max(0.3, min(1.0, base))
    
    def get_action_emoji(self, action: SpotAction) -> str:
        """Get emoji for action type."""
        return {
            SpotAction.BUY: "🟢",
            SpotAction.SELL: "🔴",
            SpotAction.HOLD: "⚪",
            SpotAction.BLOCKED: "🚫"
        }.get(action, "❓")


# Standalone function for easy integration
def process_spot_signal(
    agent_decision: Dict[str, Any],
    market_data: Dict[str, Any],
    has_position: bool,
    symbol: str,
    handler: SpotSignalHandler = None
) -> SignalDecision:
    """
    Process a spot trading signal.
    
    Simple wrapper for SpotSignalHandler.
    
    Example:
        decision = await agent.analyze_market(market_data)
        
        result = process_spot_signal(
            agent_decision=decision,
            market_data=market_data,
            has_position=exchange.has_position(symbol),
            symbol=symbol
        )
        
        if result.action == SpotAction.BUY:
            execute_buy(...)
        elif result.action == SpotAction.SELL:
            execute_sell(...)
    """
    if handler is None:
        handler = SpotSignalHandler()
    
    return handler.process_signal(
        agent_decision=agent_decision,
        market_data=market_data,
        has_position=has_position,
        symbol=symbol
    )

