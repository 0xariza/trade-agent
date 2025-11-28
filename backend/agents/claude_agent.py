import os
import json
import logging
from typing import Dict, Any
from anthropic import AsyncAnthropic
from .base_agent import BaseTradingAgent

logger = logging.getLogger(__name__)

class ClaudeAgent(BaseTradingAgent):
    """
    Trading agent using Anthropic's Claude Opus 4.5 model.
    """
    
    def __init__(self, name: str = "claude_trader", config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Default to Claude Opus 4.5
        # Model names (check Anthropic API docs for latest):
        # - claude-opus-4.5 (if available)
        # - claude-3-5-opus-20241022 (Opus 3.5, fallback)
        # - claude-3-opus-20240229 (older Opus)
        # You can override via config: {"model": "claude-opus-4.5"}
        if config and config.get("model"):
            self.model = config["model"]
        else:
            # Default to Opus 4.5 - adjust model name based on Anthropic's actual API
            # If "claude-opus-4.5" doesn't work, try "claude-3-5-opus-20241022"
            self.model = "claude-opus-4.5"
        
        # Get API key
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        # Initialize Anthropic client
        self.client = AsyncAnthropic(api_key=self.api_key)
        
        logger.info(f"Claude Agent initialized with model: {self.model}")

    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data using Claude Opus 4.5 with Hierarchical Confluence Approach.
        """
        system_prompt = """You are an expert crypto trading bot acting as a "Chief Strategist".
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

**Step 5: Consider News Sentiment**
- If sentiment is very negative (< -0.3), reduce confidence and avoid buying even if technicals are bullish.
- If sentiment is very positive (> 0.3), be cautious of FOMO - only buy if technicals strongly confirm.
- Neutral sentiment: rely primarily on technicals.

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
    "reasoning": "Brief summary of why you made this decision based on the scores, rules, and news sentiment."
}"""

        user_prompt = f"""Analyze this market data:

Symbol: {market_data.get('symbol', 'BTC/USDT')}
Current Price: ${market_data.get('price', 0):,.2f}
24h Change: {market_data.get('change_24h', 0):.2f}%
Volume: ${market_data.get('volume', 0):,.0f}

Multi-Timeframe Analysis:
{json.dumps(market_data.get('timeframes', {}), indent=2)}

News Sentiment: {market_data.get('news_sentiment', {}).get('sentiment_label', 'neutral')} ({market_data.get('news_sentiment', {}).get('sentiment_score', 0):+.2f})

Provide your analysis in JSON format following the exact structure specified."""

        try:
            # Make API call to Claude
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            
            # Extract response text
            response_text = message.content[0].text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            # Parse JSON
            try:
                decision = json.loads(response_text)
                logger.info(f"Claude analysis completed for {market_data.get('symbol')}: {decision.get('signal_type', 'HOLD')}")
                return decision
            except json.JSONDecodeError as e:
                # Fallback: try to find JSON block with more robust extraction
                import re
                # Find the first { and the last }
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx+1]
                    try:
                        decision = json.loads(json_str)
                        logger.warning(f"Extracted JSON from response (had extra text)")
                        return decision
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse extracted JSON: {json_str[:100]}...")
                        logger.debug(f"Raw response: {response_text}")
                        raise ValueError("Extracted JSON block is invalid")
                else:
                    logger.error(f"Could not find JSON block. Raw response: {response_text[:200]}...")
                    raise ValueError("Could not find JSON block in response")
                    
        except Exception as e:
            logger.error(f"Error in Claude analysis: {e}", exc_info=True)
            # Return safe default
            return {
                "trend": "neutral",
                "confidence": "low",
                "signal_type": "HOLD",
                "reasoning": f"Error in analysis: {str(e)}",
                "weighted_score": 0,
                "confluence": "N/A"
            }

    async def generate_signals(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on analysis.
        """
        return {"signal": "HOLD", "analysis": analysis}

    async def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade based on a signal.
        """
        return {"status": "executed", "signal": signal}

