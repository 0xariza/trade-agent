import asyncio
import os
import sys
import signal
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.gemini_agent import GeminiAgent
from backend.data.market_data import MarketDataProvider
from backend.data.news_sentiment import NewsSentimentProvider
from backend.scheduler.trade_scheduler import TradingScheduler
from backend.exchanges.paper_exchange import PaperExchange
from backend.risk.risk_manager import RiskManager
from backend.database.db import init_db


# Global reference for signal handler
scheduler_ref = None


async def shutdown(scheduler, market_data):
    """Graceful shutdown with state saving."""
    print("\n⏹️ Shutting down gracefully...")
    
    # Save state before exiting
    if scheduler:
        print("💾 Saving state to database...")
        await scheduler.save_state()
        print("✅ State saved successfully!")
    
    # Close market data connection
    if market_data:
        await market_data.close()
    
    print("👋 Bot stopped. State preserved for next restart.")


async def main():
    global scheduler_ref
    
    # Load environment variables
    load_dotenv()
    
    print("=" * 60)
    print("🚀 Alpha Arena Trading Bot Starting")
    print("=" * 60)

    # Initialize Database
    print("\n📦 Initializing Database...")
    await init_db()
    print("✅ Database ready")
    
    # Initialize Components
    market_data = None
    scheduler = None
    
    try:
        # Agent - Using Gemini
        print("\n🤖 Initializing Agent...")
        agent = GeminiAgent()
        print(f"   Model: {agent.model_name}")
        
        # Data Provider
        print("\n📊 Connecting to Market Data...")
        market_data = MarketDataProvider(exchange_id='binance')
        print(f"   Exchange: {market_data.exchange_id}")
        
        # News Sentiment Provider
        print("\n📰 Initializing News Provider...")
        news_provider = NewsSentimentProvider()
        
        # Paper Exchange (will be restored from state)
        print("\n💱 Initializing Paper Exchange...")
        paper_exchange = PaperExchange(initial_balance=10000.0)
        
        # Risk Manager with enhanced settings
        print("\n⚠️ Initializing Risk Manager...")
        risk_manager = RiskManager(
            max_position_size_pct=0.10,
            max_daily_loss_pct=0.05,
            default_risk_reward=2.0,
            enable_trailing_stop=True
        )
        print(f"   Max Position: {risk_manager.max_position_size_pct*100}%")
        print(f"   Risk/Reward: {risk_manager.default_risk_reward}:1")
        print(f"   Trailing Stop: {'Enabled' if risk_manager.enable_trailing_stop else 'Disabled'}")

        # Initialize Scheduler with multiple symbols
        # BEST PRACTICE: Split intervals for different tasks
        print("\n⏰ Initializing Scheduler...")
        scheduler = TradingScheduler(
            market_data_provider=market_data,
            news_provider=news_provider,
            agent=agent,
            paper_exchange=paper_exchange,
            risk_manager=risk_manager,
            # Split intervals (best practice for trading bots)
            analysis_interval_minutes=15,  # Full LLM analysis every 15 min (matches 15m timeframe)
            stop_check_seconds=30,         # Fast SL/TP check every 30 seconds
            state_save_minutes=5,          # Auto-save state every 5 minutes
            symbols=[
                "BTC/USDT", "ETH/USDT", "SOL/USDT", 
                "BNB/USDT", "AVAX/USDT", "LINK/USDT", 
                "DOGE/USDT", "XRP/USDT", "ZEC/USDT", 
                "OP/USDT", "ARB/USDT", "POL/USDT"
            ]
        )
        scheduler_ref = scheduler
        
        # Restore previous state from database
        print("\n" + "=" * 60)
        await scheduler.restore_state()
        print("=" * 60)
        
        # Start the scheduler
        print("\n🏁 Starting Trading Scheduler...")
        scheduler.start()
        
        print("\n" + "=" * 60)
        print("✅ Bot is running! Press Ctrl+C to stop.")
        print("=" * 60 + "\n")
        
        # Keep the script running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        await shutdown(scheduler, market_data)
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save state even on error
        if scheduler:
            try:
                await scheduler.save_state()
                print("💾 State saved after error")
            except:
                pass
        
        if market_data:
            await market_data.close()


if __name__ == "__main__":
    asyncio.run(main())
