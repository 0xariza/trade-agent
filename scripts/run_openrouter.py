import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents import OpenRouterAgent

async def main():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        print("Please add it to your .env file.")
        return

    print(f"Initializing OpenRouter Agent with API Key: {api_key[:4]}...{api_key[-4:]}")
    
    # You can specify the model here. OpenRouter supports many models.
    # Examples: "openai/gpt-4o", "anthropic/claude-3.5-sonnet", "deepseek/deepseek-chat"
    agent = OpenRouterAgent(config={"model": "openai/gpt-4o"})
    print(f"Using model: {agent.model}")
    
    # Sample market data
    market_data = {
        "symbol": "SOL/USDT",
        "price": 145.0,
        "volume_24h": 500000000,
        "change_24h": 5.5,
        "indicators": {
            "rsi": 72,
            "macd": "bullish"
        }
    }
    
    print("\n--- Analyzing Market Data ---")
    print(f"Input: {market_data}")
    
    try:
        analysis = await agent.analyze_market(market_data)
        print("\n--- Agent Analysis ---")
        print(analysis)
    except Exception as e:
        print(f"\nError running agent: {e}")

if __name__ == "__main__":
    asyncio.run(main())
