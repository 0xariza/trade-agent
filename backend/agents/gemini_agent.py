import os
import json
import logging
import asyncio
from typing import Dict, Any
from datetime import datetime, timedelta
import google.generativeai as genai

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
    Trading agent using Google Gemini API.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.name = "gemini_trader"
        # Use the stable 2.0 Flash model
        self.model_name = config.get("model", "models/gemini-2.0-flash") if config else "models/gemini-2.0-flash"
        
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        logger.info(f"Gemini Agent initialized with model: {self.model_name}")
    
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

            # Gemini API call
            response = self.model.generate_content(
                f"{system_prompt}\n\n{user_prompt}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2000, # Increased to avoid truncation
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
            logger.error(f"Error in Gemini analysis: {e}")
            # Return safe default
            return {
                "trend": "neutral",
                "confidence": "low",
                "reasoning": f"Error in analysis: {str(e)}"
            }
