import os
import json
import logging
from typing import Dict, Any
from openai import AsyncOpenAI
from .base_agent import BaseTradingAgent

logger = logging.getLogger(__name__)

class QwenAgent(BaseTradingAgent):
    """
    Trading agent using local Ollama Qwen model via OpenAI SDK compatibility.
    """
    
    def __init__(self, name: str = "qwen_trader", config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Default to qwen3:8b if not specified, but user can override in config
        self.model = config.get("model", "qwen3:8b") if config else "qwen3:8b"
        
        # Ollama local endpoint
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.api_key = "ollama" # Ollama doesn't require a real key
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        logger.info(f"Qwen Agent initialized with model: {self.model} at {self.base_url}")

    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data using Qwen via Ollama with Hybrid Approach.
        """
        # 1. Hybrid Gatekeeper: Check if market is "interesting" enough for LLM
        is_interesting, reason = self._is_market_interesting(market_data)
        if not is_interesting:
            logger.info(f"Skipping LLM for {market_data.get('symbol')}: {reason}")
            return {
                "trend": "neutral",
                "confidence": "high",
                "reasoning": f"Hybrid Gatekeeper: Market is neutral/boring. {reason}",
                "signal_type": "HOLD",
                "weighted_score": 0,
                "confluence": "N/A"
            }

        # 2. LLM Analysis for Interesting Markets
        system_prompt = """
        You are an expert crypto trading bot acting as a "Chief Strategist".
        Your goal is to analyze market data using a "Hierarchical Confluence Approach" to make high-probability trading decisions.
        
        IMPORTANT: Respond ONLY with valid JSON. Do NOT include any "thinking" text, markdown blocks, or explanations outside the JSON.
        
        ### EXAMPLE RESPONSE:
        {
            "market_regime": "Trending",
            "scoring_breakdown": {
                "15m": "Bullish trend, EMA cross",
                "1h": "RSI neutral, low volume",
                "4h": "Strong uptrend, ADX > 25"
            },
            "scores": { "15m": 60, "1h": 10, "4h": 80 },
            "weighted_score": 55.5,
            "confluence": "2/3 Bullish",
            "signal_type": "STRONG_BUY",
            "trend": "bullish",
            "confidence": "high",
            "reasoning": "Strong 4H trend aligned with 15m momentum."
        }
        
        You have access to data for 3 timeframes:
        1. **15m (Junior Analyst)**: Short-term moves. Weight = 1.0
        2. **1h (Senior Analyst)**: Medium-term trends. Weight = 2.5
        3. **4h (Chief Strategist)**: Big picture. Weight = 4.0 (VETO POWER)
        
        ---
        
        ### THE STRATEGY (Execute this EXACTLY)
        
        **Step 1: Detect Market Regime (on 4h chart)**
        - **Trending**: ADX > 25. Follow the trend.
        - **Ranging**: ADX < 25. Buy dips, sell rips.
        - **Volatile**: High ATR. Be cautious.
        
        **Step 2: Score Each Timeframe (-100 to +100)**
        For EACH timeframe (15m, 1h, 4h), calculate a score:
        - **Trend (40%)**: EMA 9 > 21 (+20), Ichimoku Price > Cloud (+20). (Negative if bearish).
        - **Momentum (30%)**: RSI < 30 (+15), RSI > 70 (-15), Stoch RSI Bullish Cross (+15).
        - **Volume (30%)**: OBV rising with price (x1.3 boost), Weak volume (x0.7 penalty).
        
        **Step 3: Calculate Weighted Score**
        Formula: `(15m_Score * 1.0 + 1h_Score * 2.5 + 4h_Score * 4.0) / 7.5`
        
        **Step 4: Check Confluence & Rules**
        - **Rule 1 (Veto)**: If 4h Score is strongly Bearish (<-40), DO NOT BUY. If 4h is strongly Bullish (>+40), DO NOT SELL.
        - **Rule 2 (Extreme Oversold - SNIPER ENTRY)**: IF 4h RSI < 15 AND 15m Score > 0 -> **OVERSOLD_BUY** (Ignore Veto).
        - **Rule 3 (Oversold Reversal - EXCEPTION)**: IF 4h RSI < 20 AND 15m Score > +60 -> **WEAK_BUY** (Ignore Veto).
        - **Rule 4 (Thresholds)**:
            - **STRONG_BUY**: Weighted Score > +50 AND Confluence >= 66%.
            - **MODERATE_BUY**: Weighted Score > +35 AND Confluence >= 66%.
            - **WEAK_BUY**: Oversold Reversal condition met.
            - **OVERSOLD_BUY**: Extreme Oversold condition met.
            - **SELL**: Weighted Score < -35 AND Confluence >= 66%.
            - **HOLD**: Anything else.
            
        ---
        
        ### RESPONSE FORMAT
        Respond ONLY in JSON. Do not include markdown formatting.
        {
            "market_regime": "Trending/Ranging/Volatile",
            "scoring_breakdown": {
                "15m": "Explain calculation",
                "1h": "Explain calculation",
                "4h": "Explain calculation"
            },
            "scores": {
                "15m": <int>,
                "1h": <int>,
                "4h": <int>
            },
            "weighted_score": <float>,
            "confluence": "X/3 Bullish/Bearish",
            "signal_type": "STRONG_BUY" | "MODERATE_BUY" | "WEAK_BUY" | "OVERSOLD_BUY" | "SELL" | "HOLD",
            "trend": "bullish" | "bearish" | "neutral",
            "confidence": "high" | "moderate" | "low",
            "reasoning": "Brief summary of why you made this decision based on the scores and rules."
        }
        """
        
        # Build memory context
        memory_context = market_data.get('memory_context', '')
        trading_rules = market_data.get('trading_rules', '')
        
        user_prompt = f"""
        Analyze this market data:
        
        Symbol: {market_data.get('symbol', 'BTC/USDT')}
        Current Price: ${market_data.get('price', 0):,.2f}
        24h Change: {market_data.get('change_24h', 0):.2f}%
        Volume: ${market_data.get('volume', 0):,.0f}
        
        Multi-Timeframe Analysis:
        {json.dumps(market_data.get('timeframes', {}), indent=2)}
        
        News Sentiment: {market_data.get('news_sentiment', {}).get('sentiment_label', 'neutral')} ({market_data.get('news_sentiment', {}).get('sentiment_score', 0):+.2f})
        
        {'='*50}
        TRADING HISTORY & CONTEXT (Use this to inform your decision):
        {'='*50}
        {memory_context if memory_context else 'No trading history available.'}
        
        {f"WARNINGS FROM PAST PERFORMANCE:{chr(10)}{trading_rules}" if trading_rules else ""}
        {'='*50}
        
        IMPORTANT: Consider your past trades on this symbol. If you've been losing, be more conservative.
        If you already have an open position, consider whether to HOLD or SELL rather than opening new positions.
        
        Provide your analysis in JSON format.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content
            
            # Clean up response text
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            # Parse JSON
            try:
                decision = json.loads(response_text)
                return decision
            except json.JSONDecodeError:
                # Fallback: try to find JSON block with more robust regex
                import re
                # Find the first { and the last }
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse extracted JSON: {json_str[:100]}...")
                        print(f"\n[ERROR] Raw Qwen Response:\n{response_text}\n")
                        logger.debug(f"Raw response: {response_text}")
                        raise ValueError("Extracted JSON block is invalid")
                else:
                    logger.error(f"Could not find JSON block. Raw response: {response_text[:200]}...")
                    print(f"\n[ERROR] Raw Qwen Response (No JSON found):\n{response_text}\n")
                    raise ValueError("Could not find JSON block in response")
                    
        except Exception as e:
            logger.error(f"Error in Qwen analysis: {e}")
            return {
                "trend": "neutral",
                "confidence": "low",
                "reasoning": f"Error in analysis: {str(e)}"
            }

    def _is_market_interesting(self, market_data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Check if the market is interesting enough to warrant an LLM call.
        Returns (is_interesting, reason).
        """
        try:
            tf_1h = market_data.get('timeframes', {}).get('1h', {}).get('indicators', {})
            tf_4h = market_data.get('timeframes', {}).get('4h', {}).get('indicators', {})
            
            rsi_1h = tf_1h.get('rsi', 50)
            rsi_4h = tf_4h.get('rsi', 50)
            adx_1h = tf_1h.get('adx', 0)
            
            # 1. Check for Oversold/Overbought (Interesting!)
            if rsi_1h < 35 or rsi_1h > 65:
                return True, f"1H RSI is extreme ({rsi_1h})"
            if rsi_4h < 35 or rsi_4h > 65:
                return True, f"4H RSI is extreme ({rsi_4h})"
                
            # 2. Check for Strong Trend (Interesting!)
            if adx_1h > 25:
                return True, f"1H ADX indicates strong trend ({adx_1h})"
                
            # 3. Check for Ichimoku Signals (Interesting!)
            # If price is breaking out of cloud, etc.
            # For simplicity, we'll rely on RSI/ADX for now as the main gatekeepers.
            
            # If we get here, it's boring
            return False, f"RSI Neutral (1H:{rsi_1h}, 4H:{rsi_4h}), Weak Trend (ADX:{adx_1h})"
            
        except Exception as e:
            logger.error(f"Error in gatekeeper: {e}")
            return True, "Error in gatekeeper, defaulting to interesting"

    async def generate_signals(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on analysis.
        """
        # This method might be redundant if the analysis itself contains the signal,
        # but keeping it for compatibility with BaseTradingAgent interface if needed.
        return {"signal": "HOLD", "analysis": analysis}

    async def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade based on a signal.
        """
        return {"status": "executed", "signal": signal}
