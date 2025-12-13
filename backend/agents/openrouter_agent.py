import os
from typing import Dict, Any
from openai import AsyncOpenAI
from .base_agent import BaseTradingAgent

class OpenRouterAgent(BaseTradingAgent):
    """
    Trading agent using OpenRouter API to access various models.
    """
    
    def __init__(self, name: str = "openrouter_trader", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        # Default to a cost-effective but capable model, can be overridden in config
        self.model = config.get("model", "deepseek/deepseek-chat") if config else "deepseek/deepseek-chat"
    
    def _get_max_tokens(self) -> int:
        """Get max tokens from settings."""
        try:
            from backend.config import settings
            return settings.llm_max_output_tokens
        except:
            return 4000  # Default fallback

    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data using OpenRouter.
        """
        system_prompt = """
        You are an expert crypto trading agent. Analyze the provided market data and generate a trading signal.
        
        You have access to:
        1. Price & Volume: Current price, 24h change, volume.
        2. Technical Indicators:
           - RSI (Relative Strength Index): >70 Overbought, <30 Oversold.
           - MACD (Moving Average Convergence Divergence): Trend direction.
           - Bollinger Bands: Volatility and potential breakout levels.
           - ADX (Average Directional Index): Trend Strength. >25 means strong trend. <25 means ranging/choppy.
           - ATR (Average True Range): Volatility. High ATR means high risk/movement.
           - VWAP (Volume Weighted Average Price): Institutional benchmark. Price < VWAP in uptrend is good value.
        3. Multi-Timeframe Analysis: 15m, 1h, and 4h charts.
        4. News Sentiment: Market sentiment from recent crypto news (-1 = very negative, +1 = very positive).
        
        Strategy Guidelines:
        - If ADX > 25, follow the trend (MACD).
        - If ADX < 25, look for mean reversion (RSI overbought/oversold).
        - Use VWAP as a dynamic support/resistance level.
        - Confirm 1h signals with 4h trend.
        - **News Sentiment Rules**:
            - If sentiment is very negative (< -0.3), avoid buying even if technicals are bullish.
            - If sentiment is very positive (> 0.3), be cautious of FOMO - only buy if technicals confirm.
            - Neutral sentiment: rely on technicals.
        
        Respond ONLY in JSON format:
        {
            "trend": "bullish" | "bearish" | "neutral",
            "confidence": "high" | "moderate" | "low",
            "reasoning": "Brief explanation citing specific indicators AND news sentiment (e.g. 'ADX is 30 indicating strong trend, but news sentiment is negative at -0.4 suggesting caution...')"
        }
        """
        
        user_prompt = f"""
        Analyze the following market data and provide trading insights:
        {market_data}
        """
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # OpenRouter specific headers
            extra_headers={
                "HTTP-Referer": "https://github.com/alpha-arena", # Optional
                "X-Title": "Alpha Arena Trading Agent" # Optional
            },
            response_format={"type": "json_object"},
            max_tokens=self._get_max_tokens(),
            temperature=0.7
        )
        
        return response.choices[0].message.content

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
