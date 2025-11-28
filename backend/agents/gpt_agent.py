import os
from typing import Dict, Any
from openai import AsyncOpenAI
from .base_agent import BaseTradingAgent

class GPTAgent(BaseTradingAgent):
    """
    Trading agent using OpenAI's GPT models.
    """
    
    def __init__(self, name: str = "gpt_trader", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = "gpt-4o" # Using the latest flagship model

    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data using GPT-4o.
        """
        prompt = f"""
        Analyze the following market data and provide trading insights:
        {market_data}
        
        Return a JSON object with 'trend', 'confidence', and 'reasoning'.
        """
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert crypto trading analyst."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
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
