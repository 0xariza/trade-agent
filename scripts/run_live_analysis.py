import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents import OpenRouterAgent
from backend.data.market_data import MarketDataProvider

async def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Agent
    try:
        agent = OpenRouterAgent(config={"model": "openai/gpt-4o"})
    except ValueError as e:
        print(f"Agent Error: {e}")
        return

    # Initialize Data Provider
    market_data_provider = MarketDataProvider(exchange_id='binance')
    
    symbol = "BTC/USDT"
    print(f"Fetching live data for {symbol} from Binance...")
    
    try:
        # Fetch real data
        market_snapshot = await market_data_provider.get_market_snapshot(symbol)
        
        if not market_snapshot:
            print("Failed to fetch market data.")
            return

        print("\n--- Live Market Data ---")
        print(f"Price: ${market_snapshot['price']:,.2f}")
        print(f"RSI: {market_snapshot['indicators']['rsi']}")
        print(f"MACD: {market_snapshot['indicators']['macd']}")
        print(f"Bollinger: {market_snapshot['indicators']['bollinger_position']}")
        
        print("\n--- Agent Analysis ---")
        print("Asking OpenRouter Agent...")
        
        analysis = await agent.analyze_market(market_snapshot)
        print(analysis)
        
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        await market_data_provider.close()

if __name__ == "__main__":
    asyncio.run(main())
