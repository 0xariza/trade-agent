"""
Resilient Agent Wrapper for Alpha Arena.

Provides:
1. Automatic fallback to rule-based when LLM fails
2. Ensemble voting from multiple LLMs
3. LLM output validation against indicators
4. Retry logic with exponential backoff
5. Response caching for backtesting

Week 5-6 Implementation.
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from backend.utils.production_logger import get_logger, get_trade_logger

logger = get_logger(__name__)

# Import Telegram notifier for failure alerts
try:
    from backend.notifications.telegram_notifier import TelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


class AgentType(Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    RULE_BASED = "rule_based"


@dataclass
class AgentResponse:
    """Standardized response from any agent."""
    signal_type: str  # STRONG_BUY, MODERATE_BUY, HOLD, SELL
    trend: str  # bullish, bearish, neutral
    confidence: str  # high, moderate, low
    reasoning: str
    market_regime: str  # trending, ranging, volatile
    source_agent: str
    cached: bool = False
    validation_passed: bool = True
    validation_warnings: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "trend": self.trend,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "market_regime": self.market_regime,
            "source_agent": self.source_agent,
            "cached": self.cached,
            "validation_passed": self.validation_passed,
            "validation_warnings": self.validation_warnings or []
        }


class RuleBasedFallback:
    """
    Pure rule-based trading logic as fallback when LLM fails.
    
    Uses the same rules the LLM is prompted with:
    1. ADX check (trend strength)
    2. Timeframe alignment
    3. RSI zones
    4. MACD direction
    """
    
    def __init__(self):
        self.name = "rule_based_fallback"
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> AgentResponse:
        """Generate trading signal using pure rule-based logic."""
        
        timeframes = market_data.get('timeframes', {})
        tf_1h = timeframes.get('1h', {})
        tf_4h = timeframes.get('4h', {})
        tf_15m = timeframes.get('15m', {})
        
        # Extract indicators
        ind_1h = tf_1h.get('indicators', {})
        ind_4h = tf_4h.get('indicators', {})
        ind_15m = tf_15m.get('indicators', {})
        
        adx_1h = ind_1h.get('adx', 0)
        adx_4h = ind_4h.get('adx', 0)
        rsi_1h = ind_1h.get('rsi', 50)
        rsi_4h = ind_4h.get('rsi', 50)
        macd_1h = ind_1h.get('macd', 'neutral')
        macd_4h = ind_4h.get('macd', 'neutral')
        
        score = 0
        reasons = []
        
        # Rule 1: ADX Check (No trend = no trade)
        if adx_4h < 15 and adx_1h < 15:
            return AgentResponse(
                signal_type="HOLD",
                trend="neutral",
                confidence="high",
                reasoning="No trend detected. ADX < 15 on both timeframes.",
                market_regime="ranging",
                source_agent=self.name
            )
        
        # Rule 2: ADX strength scoring
        if adx_4h > 25:
            score += 20
            reasons.append(f"Strong trend (ADX 4H: {adx_4h:.0f})")
        elif adx_4h > 20:
            score += 10
            reasons.append(f"Moderate trend (ADX 4H: {adx_4h:.0f})")
        
        # Rule 3: MACD alignment
        if macd_1h == 'bullish' and macd_4h == 'bullish':
            score += 25
            reasons.append("MACD bullish on 1H and 4H")
        elif macd_1h == 'bullish' or macd_4h == 'bullish':
            score += 10
            reasons.append("MACD partially bullish")
        elif macd_1h == 'bearish' and macd_4h == 'bearish':
            score -= 25
            reasons.append("MACD bearish on both timeframes")
        
        # Rule 4: RSI zones
        if rsi_4h > 70:
            score -= 30
            reasons.append(f"Overbought (RSI 4H: {rsi_4h:.0f})")
        elif rsi_4h > 60:
            score -= 10
            reasons.append(f"RSI elevated ({rsi_4h:.0f})")
        elif rsi_4h < 30:
            score += 25
            reasons.append(f"Oversold (RSI 4H: {rsi_4h:.0f})")
        elif rsi_4h < 40:
            score += 10
            reasons.append(f"RSI favorable ({rsi_4h:.0f})")
        
        # Rule 5: EMA alignment
        ema_1h = tf_1h.get('ema_alignment', 'neutral')
        ema_4h = tf_4h.get('ema_alignment', 'neutral')
        
        if ema_1h == 'bullish' and ema_4h == 'bullish':
            score += 15
            reasons.append("EMA aligned bullish")
        elif ema_1h == 'bearish' and ema_4h == 'bearish':
            score -= 15
            reasons.append("EMA aligned bearish")
        
        # Determine signal
        if score >= 50:
            signal_type = "STRONG_BUY"
            trend = "bullish"
            confidence = "high"
        elif score >= 30:
            signal_type = "MODERATE_BUY"
            trend = "bullish"
            confidence = "moderate"
        elif score >= 15:
            signal_type = "WEAK_BUY"
            trend = "bullish"
            confidence = "low"
        elif score <= -30:
            signal_type = "SELL"
            trend = "bearish"
            confidence = "high"
        elif score <= -15:
            signal_type = "HOLD"
            trend = "bearish"
            confidence = "moderate"
        else:
            signal_type = "HOLD"
            trend = "neutral"
            confidence = "moderate"
        
        # Determine regime
        if adx_4h > 25:
            market_regime = "trending"
        elif ind_4h.get('bb_width', 0) > 5:
            market_regime = "volatile"
        else:
            market_regime = "ranging"
        
        return AgentResponse(
            signal_type=signal_type,
            trend=trend,
            confidence=confidence,
            reasoning=f"Rule-based analysis (score: {score}): " + "; ".join(reasons),
            market_regime=market_regime,
            source_agent=self.name
        )


class ResponseValidator:
    """
    Validates LLM responses against market data.
    
    Catches LLM hallucinations and inconsistencies.
    """
    
    def validate(
        self,
        response: AgentResponse,
        market_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate an agent response against market data.
        
        Returns:
            Tuple of (passed, warnings)
        """
        warnings = []
        
        timeframes = market_data.get('timeframes', {})
        tf_4h = timeframes.get('4h', {})
        ind_4h = tf_4h.get('indicators', {})
        
        rsi_4h = ind_4h.get('rsi', 50)
        adx_4h = ind_4h.get('adx', 25)
        macd_4h = ind_4h.get('macd', 'neutral')
        
        # Validation Rule 1: Don't buy in overbought
        if response.signal_type in ['STRONG_BUY', 'MODERATE_BUY'] and rsi_4h > 70:
            warnings.append(f"BUY signal with RSI={rsi_4h:.0f} (overbought)")
        
        # Validation Rule 2: High confidence needs strong trend
        if response.confidence == 'high' and adx_4h < 20:
            warnings.append(f"High confidence but ADX={adx_4h:.0f} (weak trend)")
        
        # Validation Rule 3: Bullish trend with bearish MACD
        if response.trend == 'bullish' and macd_4h == 'bearish':
            warnings.append("Bullish trend declared but MACD is bearish")
        
        # Validation Rule 4: Strong buy needs alignment
        if response.signal_type == 'STRONG_BUY':
            tf_1h = timeframes.get('1h', {})
            macd_1h = tf_1h.get('indicators', {}).get('macd', 'neutral')
            
            if macd_1h == 'bearish' or macd_4h == 'bearish':
                warnings.append("STRONG_BUY but MACD not aligned on all timeframes")
        
        # Validation Rule 5: Sell signal when oversold
        if response.signal_type == 'SELL' and rsi_4h < 30:
            warnings.append(f"SELL signal with RSI={rsi_4h:.0f} (oversold)")
        
        # Pass if no critical warnings
        passed = len(warnings) < 2  # Allow 1 minor warning
        
        return passed, warnings


class ResponseCache:
    """
    Caches LLM responses for backtesting efficiency.
    
    Uses a hash of market data to key responses.
    """
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
    
    def _make_key(self, market_data: Dict[str, Any]) -> str:
        """Create a cache key from market data."""
        # Use relevant fields only
        key_data = {
            'symbol': market_data.get('symbol'),
            'price': round(market_data.get('price', 0), -1),  # Round to nearest 10
            'timeframes': {
                tf: {
                    'rsi': round(data.get('indicators', {}).get('rsi', 50), 0),
                    'adx': round(data.get('indicators', {}).get('adx', 25), 0),
                    'macd': data.get('indicators', {}).get('macd', 'neutral')
                }
                for tf, data in market_data.get('timeframes', {}).items()
            }
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, market_data: Dict[str, Any]) -> Optional[AgentResponse]:
        """Get cached response if available and not expired."""
        key = self._make_key(market_data)
        
        if key in self.cache:
            cached = self.cache[key]
            cached_time = cached.get('timestamp')
            
            if cached_time and (datetime.now() - cached_time).seconds < self.ttl_seconds:
                response = AgentResponse(**cached['response'])
                response.cached = True
                return response
            else:
                # Expired
                del self.cache[key]
        
        return None
    
    def set(self, market_data: Dict[str, Any], response: AgentResponse):
        """Cache a response."""
        key = self._make_key(market_data)
        self.cache[key] = {
            'response': {
                'signal_type': response.signal_type,
                'trend': response.trend,
                'confidence': response.confidence,
                'reasoning': response.reasoning,
                'market_regime': response.market_regime,
                'source_agent': response.source_agent,
                'validation_passed': response.validation_passed,
                'validation_warnings': response.validation_warnings
            },
            'timestamp': datetime.now()
        }
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()


class ResilientAgent:
    """
    Wrapper that provides resilience for trading agents.
    
    Features:
    1. Retry with exponential backoff
    2. Fallback to rule-based on LLM failure
    3. Optional ensemble voting
    4. Response validation
    5. Caching for backtests
    6. Telegram notifications on failure
    """
    
    def __init__(
        self,
        primary_agent,
        fallback_agents: List = None,
        enable_ensemble: bool = False,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        enable_caching: bool = False,
        cache_ttl: int = 300,
        notify_on_failure: bool = True
    ):
        """
        Args:
            primary_agent: Main agent to use (e.g., GeminiAgent)
            fallback_agents: List of agents to try if primary fails
            enable_ensemble: Use voting from multiple agents
            max_retries: Max retry attempts for primary agent
            retry_delay: Base delay between retries (exponential)
            enable_caching: Cache responses (for backtesting)
            cache_ttl: Cache time-to-live in seconds
            notify_on_failure: Send Telegram notification when LLM fails
        """
        self.primary_agent = primary_agent
        self.fallback_agents = fallback_agents or []
        self.rule_based_fallback = RuleBasedFallback()
        self.enable_ensemble = enable_ensemble
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.notify_on_failure = notify_on_failure
        
        self.validator = ResponseValidator()
        self.cache = ResponseCache(cache_ttl) if enable_caching else None
        
        # Initialize Telegram notifier
        self.telegram = None
        if notify_on_failure and TELEGRAM_AVAILABLE:
            try:
                self.telegram = TelegramNotifier()
                logger.info("Telegram notifications enabled for LLM failures")
            except Exception as e:
                logger.warning(f"Could not initialize Telegram: {e}")
        
        # Stats
        self.stats = {
            'primary_success': 0,
            'primary_fail': 0,
            'fallback_used': 0,
            'rule_based_used': 0,
            'cache_hits': 0,
            'validation_failures': 0,
            'telegram_alerts_sent': 0
        }
        
        # Track last failure notification time to avoid spam
        self._last_failure_notify = None
        self._notify_cooldown_seconds = 300  # 5 minutes between notifications
        
        self.name = f"resilient_{getattr(primary_agent, 'name', 'agent')}"
        
        logger.info(
            f"ResilientAgent initialized. Primary: {getattr(primary_agent, 'name', 'unknown')}, "
            f"Ensemble: {enable_ensemble}, Retries: {max_retries}, "
            f"Telegram Alerts: {notify_on_failure and self.telegram is not None}"
        )
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market with full resilience:
        1. Check cache
        2. Try primary with retries
        3. Try fallback agents
        4. Fall back to rule-based
        5. Validate response
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(market_data)
            if cached:
                self.stats['cache_hits'] += 1
                return cached.to_dict()
        
        response = None
        
        # Ensemble mode: Get votes from multiple agents
        if self.enable_ensemble and self.fallback_agents:
            response = await self._ensemble_analyze(market_data)
        else:
            # Standard mode: Primary with fallback
            response = await self._analyze_with_fallback(market_data)
        
        # Validate response
        passed, warnings = self.validator.validate(response, market_data)
        response.validation_passed = passed
        response.validation_warnings = warnings
        
        if not passed:
            self.stats['validation_failures'] += 1
            logger.warning(f"Validation warnings: {warnings}")
            
            # If validation fails badly, use rule-based
            if len(warnings) >= 3:
                logger.warning("Too many validation warnings, using rule-based")
                response = await self.rule_based_fallback.analyze_market(market_data)
                response.reasoning += " [Overridden due to LLM validation failure]"
        
        # Cache the response
        if self.cache:
            self.cache.set(market_data, response)
        
        return response.to_dict()
    
    async def _notify_failure(self, error: str, symbol: str, fallback_used: str):
        """Send Telegram notification about LLM failure."""
        if not self.telegram or not self.notify_on_failure:
            return
        
        # Check cooldown to avoid spam
        now = datetime.now()
        if self._last_failure_notify:
            elapsed = (now - self._last_failure_notify).total_seconds()
            if elapsed < self._notify_cooldown_seconds:
                logger.debug(f"Skipping notification (cooldown: {elapsed:.0f}s)")
                return
        
        try:
            message = (
                f"⚠️ *LLM FAILURE ALERT*\n\n"
                f"🤖 Agent: `{getattr(self.primary_agent, 'name', 'Gemini')}`\n"
                f"📊 Symbol: `{symbol}`\n"
                f"❌ Error: `{error[:100]}`\n"
                f"🔄 Fallback: `{fallback_used}`\n"
                f"⏰ Time: `{now.strftime('%Y-%m-%d %H:%M:%S')}`\n\n"
                f"_Trading continues with {fallback_used}_"
            )
            
            await self.telegram.send_message(message)
            self._last_failure_notify = now
            self.stats['telegram_alerts_sent'] += 1
            
            logger.info(f"Telegram notification sent for LLM failure")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
    
    async def _analyze_with_fallback(self, market_data: Dict[str, Any]) -> AgentResponse:
        """Try primary agent with retries, then fallbacks, then rule-based."""
        
        symbol = market_data.get('symbol', 'UNKNOWN')
        last_error = None
        
        # Try primary agent with retries
        for attempt in range(self.max_retries):
            try:
                result = await self.primary_agent.analyze_market(market_data)
                self.stats['primary_success'] += 1
                return self._parse_response(result, getattr(self.primary_agent, 'name', 'primary'))
                
            except Exception as e:
                last_error = str(e)
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Primary agent attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
        
        self.stats['primary_fail'] += 1
        
        # Try fallback agents
        for agent in self.fallback_agents:
            try:
                result = await agent.analyze_market(market_data)
                self.stats['fallback_used'] += 1
                
                # Notify about fallback to secondary LLM
                await self._notify_failure(
                    error=last_error or "Unknown error",
                    symbol=symbol,
                    fallback_used=getattr(agent, 'name', 'Secondary LLM')
                )
                
                logger.info(f"Using fallback agent: {getattr(agent, 'name', 'fallback')}")
                return self._parse_response(result, getattr(agent, 'name', 'fallback'))
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Fallback agent failed: {e}")
        
        # Final fallback: Rule-based
        self.stats['rule_based_used'] += 1
        logger.info("All agents failed, using rule-based fallback")
        
        # Notify about fallback to rule-based
        await self._notify_failure(
            error=last_error or "All LLM agents failed",
            symbol=symbol,
            fallback_used="Rule-Based Agent"
        )
        
        return await self.rule_based_fallback.analyze_market(market_data)
    
    async def _ensemble_analyze(self, market_data: Dict[str, Any]) -> AgentResponse:
        """Get responses from multiple agents and vote."""
        
        all_agents = [self.primary_agent] + self.fallback_agents
        responses = []
        
        # Gather responses (with timeout)
        tasks = [
            asyncio.wait_for(
                agent.analyze_market(market_data),
                timeout=30.0
            )
            for agent in all_agents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for agent, result in zip(all_agents, results):
            if isinstance(result, Exception):
                logger.warning(f"Agent {getattr(agent, 'name', 'unknown')} failed: {result}")
            else:
                try:
                    parsed = self._parse_response(result, getattr(agent, 'name', 'unknown'))
                    responses.append(parsed)
                except Exception as e:
                    logger.warning(f"Failed to parse response: {e}")
        
        if not responses:
            # All failed, use rule-based
            self.stats['rule_based_used'] += 1
            return await self.rule_based_fallback.analyze_market(market_data)
        
        # Vote on signal type
        signal_votes = {}
        trend_votes = {}
        
        for r in responses:
            signal_votes[r.signal_type] = signal_votes.get(r.signal_type, 0) + 1
            trend_votes[r.trend] = trend_votes.get(r.trend, 0) + 1
        
        # Majority vote
        winning_signal = max(signal_votes.items(), key=lambda x: x[1])[0]
        winning_trend = max(trend_votes.items(), key=lambda x: x[1])[0]
        
        # Use the response that matches the winning vote
        for r in responses:
            if r.signal_type == winning_signal:
                r.reasoning = f"[Ensemble vote: {len(responses)} agents] " + r.reasoning
                return r
        
        return responses[0]  # Fallback to first
    
    def _parse_response(self, result: Any, source: str) -> AgentResponse:
        """Parse agent result into standardized response."""
        if isinstance(result, AgentResponse):
            return result
        
        if isinstance(result, dict):
            return AgentResponse(
                signal_type=result.get('signal_type', 'HOLD'),
                trend=result.get('trend', 'neutral'),
                confidence=result.get('confidence', 'low'),
                reasoning=result.get('reasoning', ''),
                market_regime=result.get('market_regime', 'unknown'),
                source_agent=source
            )
        
        if isinstance(result, str):
            # Try to parse JSON string
            try:
                cleaned = result.replace('```json', '').replace('```', '').strip()
                data = json.loads(cleaned)
                return self._parse_response(data, source)
            except:
                pass
        
        # Default
        return AgentResponse(
            signal_type='HOLD',
            trend='neutral',
            confidence='low',
            reasoning=f'Failed to parse response: {str(result)[:100]}',
            market_regime='unknown',
            source_agent=source
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        total = (
            self.stats['primary_success'] + 
            self.stats['primary_fail']
        )
        
        return {
            **self.stats,
            'primary_success_rate': (
                self.stats['primary_success'] / total * 100 
                if total > 0 else 0
            ),
            'total_requests': total
        }
    
    def reset_stats(self):
        """Reset statistics."""
        for key in self.stats:
            self.stats[key] = 0

