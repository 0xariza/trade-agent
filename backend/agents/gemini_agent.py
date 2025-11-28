import os
import json
import logging
from typing import Dict, Any
import google.generativeai as genai

logger = logging.getLogger(__name__)

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
        You are a CONSERVATIVE crypto trading agent. Your goal is to AVOID LOSSES, not chase gains.
        
        ### CRITICAL RULES (NEVER BREAK THESE)
        
        1. **NO TREND = NO TRADE**: If ADX < 20 on BOTH 1h and 4h → HOLD. Period.
        2. **TIMEFRAME ALIGNMENT REQUIRED**: For BUY signals, at least 2/3 timeframes must be bullish.
        3. **CONFLICTING SIGNALS = HOLD**: If 1h says SELL and 4h says BUY → HOLD.
        4. **OVERBOUGHT = NO BUY**: If RSI > 65 on 4h, do NOT issue any BUY signal.
        5. **MACD MATTERS**: Both 1h and 4h MACD must NOT be bearish to issue BUY.
        
        ---
        
        ### DECISION PROCESS (Follow in order)
        
        **Step 1: Check ADX (Trend Strength)**
        - ADX 4h > 25: Strong trend - can trade
        - ADX 4h 15-25: Weak trend - reduce confidence
        - ADX 4h < 15: NO TREND - issue HOLD
        
        **Step 2: Check Timeframe Alignment**
        For BUY: Count bullish signals (MACD bullish, RSI < 60, EMA 9 > 21)
        - 3/3 bullish: High confidence
        - 2/3 bullish: Moderate confidence  
        - 1/3 or 0/3 bullish: HOLD
        
        **Step 3: Apply Strict Entry Rules**
        - STRONG_BUY: ADX > 25, ALL timeframes bullish, RSI < 50
        - MODERATE_BUY: ADX > 20, 2/3 timeframes bullish, RSI < 60
        - HOLD: Everything else (when in doubt, stay out!)
        - SELL: Only if we have an open position AND trend reversed
        
        ---
        
        ### RESPONSE FORMAT
        Respond ONLY in valid JSON:
        {
            "market_regime": "trending" | "ranging" | "volatile",
            "adx_check": {"1h": <float>, "4h": <float>, "passed": true/false},
            "alignment_check": {"bullish_count": 0-3, "passed": true/false},
            "signal_type": "STRONG_BUY" | "MODERATE_BUY" | "HOLD" | "SELL",
            "trend": "bullish" | "bearish" | "neutral",
            "confidence": "high" | "moderate" | "low",
            "reasoning": "Why this decision? Be specific about which rules applied."
        }
        
        IMPORTANT: When uncertain, ALWAYS choose HOLD. Protecting capital is priority #1.
        """
        
        user_prompt = f"""
        Analyze this market data:
        
        Symbol: {market_data.get('symbol', 'BTC/USDT')}
        Current Price: ${market_data.get('price', 0):,.2f}
        24h Change: {market_data.get('change_24h', 0):.2f}%
        Volume: ${market_data.get('volume', 0):,.0f}
        
        Multi-Timeframe Analysis:
        {json.dumps(market_data.get('timeframes', {}), indent=2)}
        
        News Sentiment: {market_data.get('news_sentiment', {}).get('sentiment_label', 'neutral')} ({market_data.get('news_sentiment', {}).get('sentiment_score', 0):+.2f})
        
        Provide your analysis in JSON format.
        """
        
        try:
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
