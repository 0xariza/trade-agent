import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

logger = logging.getLogger(__name__)

# Simple rate limiting for Gemini API
class GeminiRateLimiter:
    """Simple rate limiter for Gemini API calls."""
    
    def __init__(self, calls_per_minute: int = 30):
        self.calls_per_minute = calls_per_minute
        self.call_times = []
    
    async def wait_if_needed(self):
        """Wait if we're hitting rate limits."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Remove old calls
        self.call_times = [t for t in self.call_times if t > minute_ago]
        
        if len(self.call_times) >= self.calls_per_minute:
            # Wait until oldest call is > 1 minute ago
            wait_time = (self.call_times[0] - minute_ago).total_seconds() + 1
            if wait_time > 0:
                logger.info(f"Rate limit: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        self.call_times.append(now)

# Global rate limiter for Gemini
_gemini_rate_limiter = GeminiRateLimiter(calls_per_minute=30)

class GeminiAgent:
    """
    Trading agent using Google Gemini API with fallback key rotation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.name = "gemini_trader"
        # Use the stable 2.0 Flash model
        self.model_name = config.get("model", "models/gemini-2.0-flash") if config else "models/gemini-2.0-flash"
        
        # Load API keys from settings
        from backend.config import settings
        
        # Primary API key
        primary_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        if not primary_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or settings")
        
        # Fallback keys (comma-separated string)
        fallback_keys_str = settings.gemini_fallback_keys or os.getenv("GEMINI_FALLBACK_KEYS", "")
        fallback_keys = [k.strip() for k in fallback_keys_str.split(",") if k.strip()] if fallback_keys_str else []
        
        # Combine all keys: primary first, then fallbacks
        self.api_keys: List[str] = [primary_key] + fallback_keys
        self.current_key_index = 0
        self.failed_keys: set = set()  # Track keys that have failed
        
        # Initialize with primary key
        self._configure_api_key(self.api_keys[0])
        self.model = genai.GenerativeModel(self.model_name)
        
        logger.info(f"Gemini Agent initialized with model: {self.model_name}")
        logger.info(f"Total API keys available: {len(self.api_keys)} (primary + {len(fallback_keys)} fallback)")
    
    def _configure_api_key(self, api_key: str):
        """Configure Gemini with a specific API key."""
        genai.configure(api_key=api_key)
        logger.debug(f"Configured Gemini with API key: {api_key[:10]}...{api_key[-4:]}")
    
    def _rotate_to_next_key(self) -> bool:
        """
        Rotate to the next available API key.
        Returns True if a new key was found, False if all keys are exhausted.
        """
        # Try next keys in order
        for i in range(len(self.api_keys)):
            next_index = (self.current_key_index + 1) % len(self.api_keys)
            next_key = self.api_keys[next_index]
            
            # Skip keys that have already failed
            if next_key in self.failed_keys:
                self.current_key_index = next_index
                continue
            
            # Found a valid key
            self.current_key_index = next_index
            self._configure_api_key(next_key)
            self.model = genai.GenerativeModel(self.model_name)
            
            logger.warning(f"Rotated to fallback API key #{next_index + 1}/{len(self.api_keys)}")
            return True
        
        # All keys exhausted
        logger.error("All Gemini API keys have been exhausted!")
        return False
    
    def _is_quota_error(self, error: Exception) -> bool:
        """Check if error is a quota/rate limit error."""
        error_str = str(error).lower()
        quota_indicators = [
            "quota",
            "rate limit",
            "resource exhausted",
            "429",
            "503",
            "usage limit",
            "billing",
            "permission denied",
            "api key not valid"
        ]
        return any(indicator in error_str for indicator in quota_indicators)
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and return trading decision.
        """
        system_prompt = """
        You are a crypto SPOT trading agent. You can BUY or SELL assets.
        
        ### SPOT TRADING RULES
        
        **BUY** = Purchase the asset with USDT (go long)
        **SELL** = Sell the asset back to USDT (exit position)
        
        This is NOT futures trading. No shorting. No leverage.
        
        ---
        
        ### WHEN TO BUY (All conditions should align)
        
        1. **Trend is BULLISH**: EMA 9 > EMA 21, MACD bullish
        2. **Trend is STRONG**: ADX > 20 on 4H timeframe
        3. **Not overbought**: RSI < 70 on 4H
        4. **Timeframe alignment**: At least 2/3 timeframes bullish
        5. **BTC safe** (for altcoins): BTC not in downtrend
        
        BUY Signal Types:
        - STRONG_BUY: All conditions met, high confidence
        - MODERATE_BUY: Most conditions met, moderate confidence
        - WEAK_BUY: Some bullish signs but not ideal (smaller position)
        
        ---
        
        ### WHEN TO SELL (Exit existing position)
        
        1. **Trend REVERSAL**: MACD turns bearish on multiple timeframes
        2. **Overbought exhaustion**: RSI > 75 and starting to turn down
        3. **Breakdown**: Price breaks below key EMA (21 or 50)
        4. **Bearish divergence**: Price makes high but RSI makes lower high
        5. **BTC dump**: BTC dropping hard (for altcoins)
        
        SELL Signal Types:
        - STRONG_SELL: Clear reversal, exit immediately
        - SELL: Trend weakening, should exit
        
        ---
        
        ### WHEN TO HOLD
        
        - No clear trend (ADX < 15)
        - Mixed signals between timeframes
        - Already in position and trend still valid
        - No position and no buy signal
        
        ---
        
        ### RESPONSE FORMAT (JSON only)
        
        {
            "market_regime": "trending" | "ranging" | "volatile",
            "signal_type": "STRONG_BUY" | "MODERATE_BUY" | "WEAK_BUY" | "HOLD" | "SELL" | "STRONG_SELL",
            "trend": "bullish" | "bearish" | "neutral",
            "confidence": "high" | "moderate" | "low",
            "reasoning": "Explain your decision based on indicators"
        }
        
        Be decisive. If bullish, say BUY. If bearish, say SELL. Only HOLD when truly uncertain.
        """
        
        # Check if we have an open position
        open_positions = market_data.get('open_positions', {})
        symbol = market_data.get('symbol', 'BTC/USDT')
        has_position = symbol in open_positions
        position_info = ""
        
        if has_position:
            pos = open_positions[symbol]
            entry_price = pos.get('entry_price', 0)
            current_price = market_data.get('price', 0)
            if entry_price > 0:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                position_info = f"""
        
        ⚠️ YOU HAVE AN OPEN POSITION:
        - Entry Price: ${entry_price:,.2f}
        - Current Price: ${current_price:,.2f}
        - Unrealized P&L: {pnl_pct:+.2f}%
        - Consider SELL if trend is reversing!
        """
        else:
            position_info = "\n        📍 NO OPEN POSITION - Looking for BUY opportunity"
        
        user_prompt = f"""
        Analyze this market data for SPOT TRADING:
        
        Symbol: {symbol}
        Current Price: ${market_data.get('price', 0):,.2f}
        24h Change: {market_data.get('change_24h', 0):.2f}%
        {position_info}
        
        Multi-Timeframe Analysis:
        {json.dumps(market_data.get('timeframes', {}), indent=2)}
        
        BTC Trend: {market_data.get('btc_trend', {}).get('trend', 'unknown')}
        News Sentiment: {market_data.get('news_sentiment', {}).get('sentiment_label', 'neutral')} ({market_data.get('news_sentiment', {}).get('sentiment_score', 0):+.2f})
        
        Should we BUY, SELL, or HOLD? Respond in JSON format.
        """
        
        # Retry with key rotation on quota errors
        max_retries = len(self.api_keys)
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                await _gemini_rate_limiter.wait_if_needed()
                
                # Configure safety settings to avoid blocking legitimate trading analysis
                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    },
                ]

                # Get token limit from settings
                from backend.config import settings
                max_tokens = settings.llm_max_output_tokens
                
                # Gemini API call
                response = self.model.generate_content(
                    f"{system_prompt}\n\n{user_prompt}",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=max_tokens,
                    ),
                    safety_settings=safety_settings
                )
                
                # Parse response
                try:
                    response_text = response.text.strip()
                except Exception:
                    # Fallback if response.text fails (e.g. finish_reason is not STOP)
                    if response.candidates and response.candidates[0].content.parts:
                        response_text = response.candidates[0].content.parts[0].text.strip()
                    else:
                        raise ValueError(f"Empty response from Gemini. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
                
                # Remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                elif response_text.startswith("```"):
                    response_text = response_text.replace("```", "").strip()
                
                # Parse JSON
                decision = json.loads(response_text)
                return decision
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                logger.warning(f"Gemini API call failed (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}")
                
                # Check if it's a quota/rate limit error
                if self._is_quota_error(e):
                    # Mark current key as failed
                    current_key = self.api_keys[self.current_key_index]
                    self.failed_keys.add(current_key)
                    logger.warning(f"API key #{self.current_key_index + 1} hit quota limit. Rotating...")
                    
                    # Try next key
                    if self._rotate_to_next_key():
                        # Wait a bit before retrying with new key
                        await asyncio.sleep(1.0)
                        continue
                    else:
                        # All keys exhausted
                        logger.error("All Gemini API keys exhausted. Cannot proceed.")
                        break
                else:
                    # Not a quota error, don't retry with different key
                    logger.error(f"Non-quota error in Gemini analysis: {e}")
                    break
        
        # All retries failed
        logger.error(f"Error in Gemini analysis after {max_retries} attempts: {last_error}")
        # Return safe default
        return {
            "trend": "neutral",
            "confidence": "low",
            "signal_type": "HOLD",
            "reasoning": f"Error in analysis: {str(last_error)[:100] if last_error else 'Unknown error'}"
        }
