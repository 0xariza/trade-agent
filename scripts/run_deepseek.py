import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents import DeepSeekAgent

async def main():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found in environment variables.")
        print("Please add it to your .env file.")
        return

    print(f"Initializing DeepSeek Agent with API Key: {api_key[:4]}...{api_key[-4:]}")
    agent = DeepSeekAgent()
    
    # Sample market data
    market_data = {
        "symbol": "BTC/USDT",
        "price": 95000.0,
        "volume_24h": 1500000000,
        "change_24h": 2.5,
        "indicators": {
            "rsi": 65,
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
