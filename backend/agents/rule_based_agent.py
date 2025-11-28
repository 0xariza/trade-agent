"""
Rule-Based Trading Agent - Deterministic Signal Generation.

This agent uses pure technical analysis rules without LLM interpretation.
Benefits:
- 100% deterministic (same data = same signal)
- Fast execution (no API calls)
- Backtestable
- No API costs

The strategy uses multi-timeframe confluence with weighted scoring.
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    MODERATE_BUY = "MODERATE_BUY"
    WEAK_BUY = "WEAK_BUY"
    OVERSOLD_BUY = "OVERSOLD_BUY"
    HOLD = "HOLD"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class TimeframeScore:
    """Score for a single timeframe."""
    timeframe: str
    trend_score: float  # -40 to +40
    momentum_score: float  # -30 to +30
    volume_score: float  # -30 to +30
    total_score: float  # -100 to +100
    
    components: Dict[str, Any] = None
    
    def __post_init__(self):
        self.total_score = self.trend_score + self.momentum_score + self.volume_score


@dataclass
class TradingSignal:
    """Complete trading signal with all details."""
    signal_type: SignalType
    confidence: float  # 0 to 1
    trend: str  # bullish, bearish, neutral
    
    # Scores
    weighted_score: float
    timeframe_scores: Dict[str, TimeframeScore]
    confluence: str  # "2/3 Bullish", etc.
    
    # Market context
    market_regime: MarketRegime
    
    # Decision details
    reasoning: List[str]
    veto_active: bool = False
    veto_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "trend": self.trend,
            "weighted_score": round(self.weighted_score, 2),
            "confluence": self.confluence,
            "market_regime": self.market_regime.value,
            "reasoning": "; ".join(self.reasoning[:3]),
            "veto_active": self.veto_active,
            "veto_reason": self.veto_reason
        }


class RuleBasedAgent:
    """
    Deterministic rule-based trading agent.
    
    Strategy: Hierarchical Multi-Timeframe Confluence
    
    Timeframe Weights:
    - 15m: 1.0 (short-term entry timing)
    - 1h: 2.5 (medium-term trend)
    - 4h: 4.0 (big picture, has VETO power)
    
    Signal Thresholds:
    - STRONG_BUY: score > 50, confluence >= 66%
    - MODERATE_BUY: score > 35, confluence >= 66%
    - WEAK_BUY: score > 20, RSI < 35
    - SELL: score < -35, confluence >= 66%
    """
    
    # Timeframe weights
    WEIGHTS = {
        '15m': 1.0,
        '1h': 2.5,
        '4h': 4.0
    }
    
    # Thresholds
    STRONG_BUY_THRESHOLD = 50
    MODERATE_BUY_THRESHOLD = 35
    WEAK_BUY_THRESHOLD = 20
    SELL_THRESHOLD = -35
    STRONG_SELL_THRESHOLD = -50
    
    # Veto thresholds
    VETO_BULLISH = 40  # 4h score > 40 vetoes sell
    VETO_BEARISH = -40  # 4h score < -40 vetoes buy
    
    def __init__(self, config: Dict[str, Any] = None):
        self.name = "rule_based_trader"
        self.config = config or {}
        
        # Adjustable parameters
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.adx_trend_threshold = self.config.get('adx_trend', 25)
        self.require_confluence = self.config.get('require_confluence', True)
        
        logger.info(f"RuleBasedAgent initialized. RSI thresholds: {self.rsi_oversold}/{self.rsi_overbought}")
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signal.
        
        Args:
            market_data: Dict containing:
                - symbol: Trading pair
                - price: Current price
                - timeframes: Dict with '15m', '1h', '4h' indicator data
                - btc_trend: BTC trend data (critical for alts)
                - fear_greed: Fear & Greed index
                - news_sentiment: Optional news data
        
        Returns:
            Dict with signal_type, trend, confidence, reasoning
        """
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        # CRITICAL: Check BTC safety for alts
        if symbol != 'BTC/USDT':
            btc_trend = market_data.get('btc_trend', {})
            if btc_trend.get('trend') in ['bearish', 'strong_bearish']:
                return {
                    "signal_type": "HOLD",
                    "trend": "neutral",
                    "confidence": 0.9,
                    "reasoning": f"BTC is {btc_trend.get('trend')} ({btc_trend.get('change_24h', 0):+.1f}%) - avoid buying alts",
                    "btc_blocked": True
                }
        
        signal = self._generate_signal(market_data)
        result = signal.to_dict()
        
        # Add Fear & Greed context
        fng = market_data.get('fear_greed', {})
        if fng.get('signal') in ['strong_buy', 'strong_sell']:
            result['fear_greed'] = fng
        
        return result
    
    def _generate_signal(self, market_data: Dict[str, Any]) -> TradingSignal:
        """Generate complete trading signal."""
        timeframes = market_data.get('timeframes', {})
        
        # 1. Detect Market Regime (from 4h)
        regime = self._detect_regime(timeframes.get('4h', {}))
        
        # 2. Score each timeframe
        tf_scores = {}
        for tf in ['15m', '1h', '4h']:
            tf_data = timeframes.get(tf, {})
            indicators = tf_data.get('indicators', tf_data)
            tf_scores[tf] = self._score_timeframe(tf, indicators)
        
        # 3. Calculate weighted score
        total_weight = sum(self.WEIGHTS.values())
        weighted_score = sum(
            tf_scores[tf].total_score * self.WEIGHTS[tf]
            for tf in tf_scores
        ) / total_weight
        
        # 4. Calculate confluence
        bullish_count = sum(1 for s in tf_scores.values() if s.total_score > 10)
        bearish_count = sum(1 for s in tf_scores.values() if s.total_score < -10)
        neutral_count = 3 - bullish_count - bearish_count
        
        if bullish_count >= 2:
            confluence = f"{bullish_count}/3 Bullish"
            confluence_pct = bullish_count / 3
        elif bearish_count >= 2:
            confluence = f"{bearish_count}/3 Bearish"
            confluence_pct = bearish_count / 3
        else:
            confluence = "Mixed"
            confluence_pct = 0.33
        
        # 5. Check for veto conditions
        veto_active = False
        veto_reason = ""
        score_4h = tf_scores.get('4h', TimeframeScore('4h', 0, 0, 0, 0)).total_score
        
        if score_4h < self.VETO_BEARISH:
            veto_active = True
            veto_reason = f"4H strongly bearish ({score_4h:.0f})"
        elif score_4h > self.VETO_BULLISH:
            veto_reason = f"4H strongly bullish ({score_4h:.0f})"  # Will veto sells
        
        # 6. Check for oversold/overbought exceptions
        rsi_4h = timeframes.get('4h', {}).get('indicators', {}).get('rsi', 50)
        rsi_15m = timeframes.get('15m', {}).get('indicators', {}).get('rsi', 50)
        score_15m = tf_scores.get('15m', TimeframeScore('15m', 0, 0, 0, 0)).total_score
        
        # Extreme oversold sniper entry
        oversold_exception = False
        if rsi_4h < 15 and score_15m > 0:
            oversold_exception = True
            veto_active = False
            veto_reason = "OVERSOLD EXCEPTION: 4H RSI < 15"
        elif rsi_4h < 25 and score_15m > 40:
            oversold_exception = True
            veto_active = False
            veto_reason = "OVERSOLD REVERSAL: 4H RSI < 25, 15m bullish"
        
        # 7. Determine signal type
        reasoning = []
        
        # Add score breakdown to reasoning
        for tf, score in tf_scores.items():
            reasoning.append(f"{tf}: {score.total_score:+.0f}")
        reasoning.append(f"Weighted: {weighted_score:+.1f}")
        reasoning.append(f"Confluence: {confluence}")
        
        signal_type = SignalType.HOLD
        trend = "neutral"
        confidence = 0.5
        
        # Apply signal rules
        if oversold_exception:
            if rsi_4h < 15:
                signal_type = SignalType.OVERSOLD_BUY
                trend = "bullish"
                confidence = 0.7
                reasoning.append("Extreme oversold sniper entry")
            else:
                signal_type = SignalType.WEAK_BUY
                trend = "bullish"
                confidence = 0.6
                reasoning.append("Oversold reversal setup")
        
        elif weighted_score > self.STRONG_BUY_THRESHOLD and confluence_pct >= 0.66:
            if veto_active and score_4h < self.VETO_BEARISH:
                signal_type = SignalType.HOLD
                reasoning.append(f"BUY VETOED: {veto_reason}")
            else:
                signal_type = SignalType.STRONG_BUY
                trend = "bullish"
                confidence = min(0.6 + (weighted_score / 100) * 0.3, 0.9)
                reasoning.append("Strong buy signal with confluence")
        
        elif weighted_score > self.MODERATE_BUY_THRESHOLD and confluence_pct >= 0.66:
            if veto_active and score_4h < self.VETO_BEARISH:
                signal_type = SignalType.HOLD
                reasoning.append(f"BUY VETOED: {veto_reason}")
            else:
                signal_type = SignalType.MODERATE_BUY
                trend = "bullish"
                confidence = min(0.5 + (weighted_score / 100) * 0.3, 0.8)
                reasoning.append("Moderate buy with confluence")
        
        elif weighted_score > self.WEAK_BUY_THRESHOLD and rsi_15m < 35:
            if veto_active and score_4h < self.VETO_BEARISH:
                signal_type = SignalType.HOLD
                reasoning.append(f"BUY VETOED: {veto_reason}")
            else:
                signal_type = SignalType.WEAK_BUY
                trend = "bullish"
                confidence = 0.55
                reasoning.append(f"Weak buy - oversold RSI ({rsi_15m:.0f})")
        
        elif weighted_score < self.STRONG_SELL_THRESHOLD and confluence_pct >= 0.66:
            if score_4h > self.VETO_BULLISH:
                signal_type = SignalType.HOLD
                reasoning.append(f"SELL VETOED: {veto_reason}")
            else:
                signal_type = SignalType.STRONG_SELL
                trend = "bearish"
                confidence = min(0.6 + abs(weighted_score) / 100 * 0.3, 0.9)
                reasoning.append("Strong sell signal with confluence")
        
        elif weighted_score < self.SELL_THRESHOLD and confluence_pct >= 0.66:
            if score_4h > self.VETO_BULLISH:
                signal_type = SignalType.HOLD
                reasoning.append(f"SELL VETOED: {veto_reason}")
            else:
                signal_type = SignalType.SELL
                trend = "bearish"
                confidence = min(0.5 + abs(weighted_score) / 100 * 0.3, 0.8)
                reasoning.append("Sell signal with confluence")
        
        else:
            reasoning.append("No clear signal - waiting for better setup")
        
        return TradingSignal(
            signal_type=signal_type,
            confidence=confidence,
            trend=trend,
            weighted_score=weighted_score,
            timeframe_scores=tf_scores,
            confluence=confluence,
            market_regime=regime,
            reasoning=reasoning,
            veto_active=veto_active,
            veto_reason=veto_reason
        )
    
    def _detect_regime(self, tf_data: Dict[str, Any]) -> MarketRegime:
        """Detect market regime from 4h timeframe."""
        indicators = tf_data.get('indicators', tf_data)
        
        adx = indicators.get('adx', 0)
        atr = indicators.get('atr', 0)
        close = indicators.get('close', 1)
        ema_9 = indicators.get('ema_9', close)
        ema_21 = indicators.get('ema_21', close)
        
        # Calculate ATR as percentage
        atr_pct = (atr / close * 100) if close > 0 else 0
        
        if atr_pct > 4:  # Very high volatility
            return MarketRegime.VOLATILE
        elif adx > self.adx_trend_threshold:
            if ema_9 > ema_21:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING
    
    def _score_timeframe(self, timeframe: str, tf_data: Dict[str, Any]) -> TimeframeScore:
        """
        Score a timeframe from -100 to +100.
        
        Components:
        - Trend (40%): EMA alignment, MACD, ADX direction
        - Momentum (30%): RSI, Stochastic
        - Volume (30%): OBV trend, volume confirmation
        """
        components = {}
        
        # Handle both old format (indicators dict) and new format (nested)
        if 'indicators' in tf_data:
            indicators = tf_data['indicators']
            momentum_data = tf_data.get('momentum', {})
            volume_data = tf_data.get('volume', {})
            trend_signal = tf_data.get('trend', 'neutral')
        else:
            indicators = tf_data
            momentum_data = {}
            volume_data = {}
            trend_signal = 'neutral'
        
        # Extract indicators with defaults
        rsi = momentum_data.get('rsi', indicators.get('rsi', 50))
        stoch_k = momentum_data.get('stoch_k', indicators.get('stoch_k', 50))
        adx = indicators.get('adx', 0)
        macd_trend = indicators.get('macd', 'neutral')
        bb_position = indicators.get('bb_position', 'middle')
        obv_trend = volume_data.get('obv_trend', indicators.get('obv_trend', 'flat'))
        volume_confirms = volume_data.get('confirms_trend', False)
        ema_alignment = tf_data.get('ema_alignment', indicators.get('ema_cross', 'neutral'))
        
        # ============ TREND SCORE (40 points max) ============
        trend_score = 0
        
        # Use pre-calculated trend signal (20 points)
        if trend_signal in ['strong_bullish']:
            trend_score += 25
            components['trend'] = 'strong bullish'
        elif trend_signal in ['bullish']:
            trend_score += 15
            components['trend'] = 'bullish'
        elif trend_signal in ['strong_bearish']:
            trend_score -= 25
            components['trend'] = 'strong bearish'
        elif trend_signal in ['bearish']:
            trend_score -= 15
            components['trend'] = 'bearish'
        else:
            components['trend'] = 'neutral'
        
        # MACD confirmation (15 points)
        if isinstance(macd_trend, str):
            if macd_trend.lower() == 'bullish':
                trend_score += 15
                components['macd'] = 'bullish'
            elif macd_trend.lower() == 'bearish':
                trend_score -= 15
                components['macd'] = 'bearish'
        
        # ============ MOMENTUM SCORE (30 points max) ============
        momentum_score = 0
        
        # RSI (20 points) - more weight for extreme values
        if rsi < 25:
            momentum_score += 20
            components['rsi'] = f'extreme oversold ({rsi:.0f})'
        elif rsi < self.rsi_oversold:
            momentum_score += 12
            components['rsi'] = f'oversold ({rsi:.0f})'
        elif rsi > 75:
            momentum_score -= 20
            components['rsi'] = f'extreme overbought ({rsi:.0f})'
        elif rsi > self.rsi_overbought:
            momentum_score -= 12
            components['rsi'] = f'overbought ({rsi:.0f})'
        elif rsi < 45:
            momentum_score += 5
        elif rsi > 55:
            momentum_score -= 5
        components['rsi_value'] = rsi
        
        # Stochastic (10 points)
        if stoch_k < 20:
            momentum_score += 10
            components['stoch'] = 'oversold'
        elif stoch_k > 80:
            momentum_score -= 10
            components['stoch'] = 'overbought'
        
        # Bollinger Band position (bonus)
        if bb_position == 'below_lower':
            momentum_score += 5
            components['bb'] = 'below lower band'
        elif bb_position == 'above_upper':
            momentum_score -= 5
            components['bb'] = 'above upper band'
        
        # ============ VOLUME SCORE (30 points max) ============
        volume_score = 0
        
        # OBV trend (15 points)
        if obv_trend == 'rising':
            volume_score += 15
            components['obv'] = 'rising (bullish)'
        elif obv_trend == 'falling':
            volume_score -= 15
            components['obv'] = 'falling (bearish)'
        else:
            components['obv'] = 'flat'
        
        # Volume confirms trend (10 points)
        if volume_confirms:
            if trend_score > 0:
                volume_score += 10
            elif trend_score < 0:
                volume_score -= 10
            components['volume_confirm'] = True
        
        # ADX strength modifier (5 points bonus)
        if adx > 30:
            # Very strong trend - boost signal
            if trend_score > 0:
                volume_score += 5
            elif trend_score < 0:
                volume_score -= 5
            components['adx'] = f'very strong ({adx:.0f})'
        elif adx > 25:
            components['adx'] = f'strong ({adx:.0f})'
        elif adx < 15:
            # Very weak trend - dampen all signals
            trend_score = int(trend_score * 0.6)
            momentum_score = int(momentum_score * 0.6)
            components['adx'] = f'no trend ({adx:.0f})'
        else:
            components['adx'] = f'weak ({adx:.0f})'
        
        return TimeframeScore(
            timeframe=timeframe,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volume_score=volume_score,
            total_score=trend_score + momentum_score + volume_score,
            components=components
        )
    
    def get_signal_for_regime(
        self,
        regime: MarketRegime,
        weighted_score: float
    ) -> Tuple[str, str]:
        """Adjust signal interpretation based on market regime."""
        if regime == MarketRegime.TRENDING_UP:
            # In uptrend, be more aggressive on buys
            if weighted_score > 25:
                return "Buy the dip", "Trend following"
        
        elif regime == MarketRegime.TRENDING_DOWN:
            # In downtrend, be cautious
            if weighted_score > 50:
                return "Counter-trend buy", "Higher risk"
        
        elif regime == MarketRegime.RANGING:
            # In range, trade reversals
            if weighted_score > 40 or weighted_score < -40:
                return "Range trade", "Mean reversion"
        
        elif regime == MarketRegime.VOLATILE:
            # High volatility - wider stops needed
            return "Caution", "High volatility"
        
        return "Standard", ""

