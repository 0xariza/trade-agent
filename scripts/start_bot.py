"""
Alpha Arena Trading Bot - Main Entry Point

FULLY INTEGRATED with all new components:
- ResilientAgent (LLM fallback)
- CorrelationManager (portfolio protection)
- Centralized Settings
- Performance Analytics
- Spot Trading (BUY + SELL)
- Production Logging with Telegram Notifications
"""

import asyncio
import os
import sys
import signal
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize production logging FIRST (before other imports)
from backend.utils.production_logger import setup_logging, get_logger

# Setup logging based on environment
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
setup_logging(
    level=LOG_LEVEL,
    log_dir="logs",
    json_logs=False,  # Human-readable for console
    enable_telegram=True  # Send errors to Telegram
)
logger = get_logger("start_bot")

# Core components - Agent imports
from backend.agents.resilient_agent import ResilientAgent
from backend.config.settings import AgentType
from backend.data.market_data import MarketDataProvider
from backend.data.news_sentiment import NewsSentimentProvider
from backend.scheduler.trade_scheduler import TradingScheduler
from backend.exchanges.paper_exchange import PaperExchange
from backend.risk.risk_manager import RiskManager
from backend.risk.correlation_manager import CorrelationManager
from backend.database.db import init_db

# New components
from backend.config import settings
from backend.monitoring.performance_analytics import PerformanceAnalyzer

# Global references for graceful shutdown
scheduler_ref = None
market_data_ref = None
paper_exchange_ref = None
shutdown_event = asyncio.Event()
shutdown_requested = False


# Global reference for signal handler
scheduler_ref = None
performance_analyzer = None


def _create_agent(agent_type: AgentType):
    """
    Factory function to create the appropriate agent based on settings.
    """
    if agent_type == AgentType.GEMINI:
        from backend.agents.gemini_agent import GeminiAgent
        return GeminiAgent()
    
    elif agent_type == AgentType.OPENROUTER:
        from backend.agents.openrouter_agent import OpenRouterAgent
        return OpenRouterAgent(config={"model": settings.openrouter_model})
    
    elif agent_type == AgentType.OPENAI:
        from backend.agents.gpt_agent import GPTAgent
        return GPTAgent()
    
    elif agent_type == AgentType.CLAUDE:
        from backend.agents.claude_agent import ClaudeAgent
        return ClaudeAgent()
    
    elif agent_type == AgentType.DEEPSEEK:
        from backend.agents.deepseek_agent import DeepSeekAgent
        return DeepSeekAgent()
    
    elif agent_type == AgentType.QWEN:
        from backend.agents.qwen_agent import QwenAgent
        return QwenAgent()
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


async def print_final_report(paper_exchange):
    """Print performance report on shutdown."""
    global performance_analyzer
    
    print("\n" + "=" * 60)
    print("📊 FINAL PERFORMANCE REPORT")
    print("=" * 60)
    
    if paper_exchange.closed_positions:
        performance_analyzer = PerformanceAnalyzer(
            initial_capital=settings.initial_balance
        )
        performance_analyzer.add_trades(paper_exchange.closed_positions)
        
        metrics = performance_analyzer.calculate_metrics()
        metrics.print_report()
        
        # Identify losing patterns
        print("\n🔍 LOSING TRADE PATTERNS:")
        patterns = performance_analyzer.get_losing_patterns()
        for issue in patterns.get('issues_identified', []):
            print(f"  {issue}")
        
        # Export report
        try:
            report_file = f"reports/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("reports", exist_ok=True)
            performance_analyzer.export_report(report_file)
            print(f"\n💾 Report saved to: {report_file}")
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")
    else:
        print("  No closed trades to analyze.")
    
    print("=" * 60 + "\n")


def signal_handler(signum, frame):
    """Handle shutdown signals (SIGINT, SIGTERM)."""
    global shutdown_requested
    signal_name = signal.Signals(signum).name
    logger.info(f"Received {signal_name} signal, initiating graceful shutdown...")
    print(f"\n⚠️  Received {signal_name} signal. Shutting down gracefully...")
    shutdown_requested = True
    shutdown_event.set()


async def shutdown(scheduler, market_data, paper_exchange):
    """Graceful shutdown with state saving and reporting."""
    print("\n⏹️ Shutting down gracefully...")
    logger.info("Starting graceful shutdown sequence")
    
    # Stop scheduler first (stops new trades)
    if scheduler:
        try:
            logger.info("Stopping trading scheduler...")
            scheduler.stop()
            print("⏸️  Trading scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    
    # Print final performance report
    if paper_exchange:
        try:
            await print_final_report(paper_exchange)
        except Exception as e:
            logger.error(f"Error printing final report: {e}")
    
    # Save state before exiting (CRITICAL - don't lose positions!)
    if scheduler:
        try:
            print("💾 Saving state to database...")
            logger.info("Saving bot state to database...")
            await scheduler.save_state()
            print("✅ State saved successfully!")
            logger.info("State saved successfully")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            print(f"⚠️  Warning: Failed to save state: {e}")
    
    # Close market data connection
    if market_data:
        try:
            logger.info("Closing market data connection...")
            await market_data.close()
        except Exception as e:
            logger.error(f"Error closing market data: {e}")
    
    print("👋 Bot stopped. State preserved for next restart.")
    logger.info("Graceful shutdown completed")


async def run_preflight_check() -> bool:
    """Quick validation before starting."""
    print("\n🔍 Running quick preflight check...")
    
    # Check Gemini API key
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key or len(gemini_key) < 10:
        print("❌ GEMINI_API_KEY not set or invalid!")
        print("   Set it in .env file: GEMINI_API_KEY=your_key_here")
        return False
    print("  ✓ Gemini API key found")
    
    # Check if we can import all modules
    try:
        from backend.config import settings
        print(f"  ✓ Config loaded ({len(settings.trading_symbols)} symbols)")
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False
    
    # Check exchange connectivity (with fallback)
    try:
        from backend.data.market_data import MarketDataProvider
        
        # Build API credentials dict
        api_credentials = {
            'kraken': {
                'api_key': settings.kraken_api_key if settings.kraken_api_key else None,
                'api_secret': settings.kraken_api_secret if settings.kraken_api_secret else None,
                'passphrase': None
            },
            'kucoin': {
                'api_key': settings.kucoin_api_key if settings.kucoin_api_key else None,
                'api_secret': settings.kucoin_api_secret if settings.kucoin_api_secret else None,
                'passphrase': settings.kucoin_passphrase if settings.kucoin_passphrase else None
            }
        }
        
        print(f"  → Trying primary: {settings.exchange_id}")
        provider = await MarketDataProvider.create_with_fallback(
            primary_exchange=settings.exchange_id,
            fallback_exchanges=settings.exchange_fallbacks,
            api_credentials=api_credentials
        )
        
        ohlcv = await provider.get_ohlcv("BTC/USDT", limit=1)
        if not ohlcv.empty:
            btc_price = ohlcv.iloc[-1]['close']
            print(f"  ✓ Exchange connected: {provider.exchange_id} (BTC: ${btc_price:,.0f})")
        await provider.close()
    except Exception as e:
        print(f"❌ All exchanges failed: {e}")
        return False
    
    print("  ✓ All checks passed!\n")
    return True


async def main():
    global scheduler_ref, market_data_ref, paper_exchange_ref
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info("Signal handlers registered for graceful shutdown")
    
    # Load environment variables
    load_dotenv()
    
    logger.info("=" * 60)
    logger.info("🚀 ALPHA ARENA TRADING BOT v2.0")
    logger.info("   Spot Trading | Resilient LLM | Production Logging")
    logger.info("=" * 60)
    
    print("=" * 60)
    print("🚀 ALPHA ARENA TRADING BOT v2.0")
    print("   Spot Trading | Resilient LLM | Correlation Protection")
    print("=" * 60)
    
    # Run preflight check
    if not await run_preflight_check():
        logger.error("Preflight check failed")
        print("\n❌ Preflight check failed. Fix errors and try again.")
        print("   For detailed check: python scripts/preflight_check.py")
        return
    
    # Show configuration
    logger.info(f"Config: mode={settings.trading_mode.value}, risk={settings.risk_profile.value}")
    print(f"\n⚙️ CONFIGURATION (from settings.py)")
    print(f"   Trading Mode: {settings.trading_mode.value}")
    print(f"   Risk Profile: {settings.risk_profile.value}")
    print(f"   Initial Balance: ${settings.initial_balance:,.2f}")
    print(f"   Max Position: {settings.max_position_size_pct*100}%")
    print(f"   Max Drawdown: {settings.max_drawdown_pct*100}%")
    print(f"   Max Concurrent Positions: {settings.max_concurrent_positions}")

    # Initialize Database (optional - bot can run without it for basic paper trading)
    print("\n📦 Initializing Database...")
    try:
        await init_db()
        print("✅ Database ready")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
        print(f"⚠️  Database not available: {str(e)[:100]}")
        print("   Bot will continue, but some features may be limited.")
        print("   To fix: Install/start PostgreSQL or set DATABASE_URL in .env")
    
    # Initialize Components
    market_data = None
    scheduler = None
    paper_exchange = None
    
    # Store references for signal handler
    market_data_ref = None
    paper_exchange_ref = None
    
    try:
        # ====================================
        # 1. AGENT WITH RESILIENCE
        # ====================================
        print("\n🤖 Initializing Agent with Resilience...")
        print(f"   Selected Agent Type: {settings.agent_type.value}")
        
        # Create agent based on settings
        primary_agent = _create_agent(settings.agent_type)
        
        # Wrap with ResilientAgent for fallback protection
        if settings.fallback_to_rule_based:
            agent = ResilientAgent(
                primary_agent=primary_agent,
                max_retries=settings.llm_max_retries,
                retry_delay=settings.llm_retry_delay_seconds,
                enable_caching=False  # Enable for backtesting
            )
            print(f"   Primary: {getattr(primary_agent, 'model_name', settings.agent_type.value)}")
            print(f"   Fallback: Rule-based (enabled)")
            print(f"   Max Retries: {settings.llm_max_retries}")
        else:
            agent = primary_agent
            print(f"   Model: {getattr(primary_agent, 'model_name', settings.agent_type.value)}")
            print(f"   Fallback: Disabled")
        
        # ====================================
        # 2. DATA PROVIDERS
        # ====================================
        print("\n📊 Connecting to Market Data (with fallback)...")
        
        # Build API credentials dict for all exchanges
        api_credentials = {
            'kraken': {
                'api_key': settings.kraken_api_key if settings.kraken_api_key else None,
                'api_secret': settings.kraken_api_secret if settings.kraken_api_secret else None,
                'passphrase': None
            },
            'kucoin': {
                'api_key': settings.kucoin_api_key if settings.kucoin_api_key else None,
                'api_secret': settings.kucoin_api_secret if settings.kucoin_api_secret else None,
                'passphrase': settings.kucoin_passphrase if settings.kucoin_passphrase else None
            }
        }
        
        # Try primary exchange first, then fallbacks
        print(f"   Primary: {settings.exchange_id}")
        print(f"   Fallbacks: {', '.join(settings.exchange_fallbacks)}")
        
        market_data = await MarketDataProvider.create_with_fallback(
            primary_exchange=settings.exchange_id,
            fallback_exchanges=settings.exchange_fallbacks,
            api_credentials=api_credentials
        )
        market_data_ref = market_data  # Store for signal handler
        
        print(f"   ✓ Connected to: {market_data.exchange_id}")
        if api_credentials.get(market_data.exchange_id, {}).get('api_key'):
            print(f"   API: Authenticated")
        else:
            print(f"   API: Public data only")
        
        print("\n📰 Initializing News Provider...")
        news_provider = NewsSentimentProvider()
        
        # ====================================
        # 3. PAPER EXCHANGE
        # ====================================
        print("\n💱 Initializing Paper Exchange...")
        paper_exchange = PaperExchange(
            initial_balance=settings.initial_balance,
            slippage_pct=settings.slippage_pct,
            fee_pct=settings.fee_pct
        )
        paper_exchange_ref = paper_exchange  # Store for signal handler
        print(f"   Initial Balance: ${settings.initial_balance:,.2f}")
        print(f"   Slippage: {settings.slippage_pct*100}%")
        print(f"   Fee: {settings.fee_pct*100}%")
        
        # ====================================
        # 4. RISK MANAGER
        # ====================================
        print("\n⚠️ Initializing Risk Manager...")
        risk_manager = RiskManager(
            max_position_size_pct=settings.max_position_size_pct,
            max_daily_loss_pct=settings.max_daily_loss_pct,
            default_risk_reward=settings.default_risk_reward,
            default_atr_stop_multiplier=settings.default_atr_stop_multiplier,
            max_hold_hours=settings.max_hold_hours,
            enable_trailing_stop=settings.enable_trailing_stop
        )
        print(f"   Max Position: {settings.max_position_size_pct*100}%")
        print(f"   Risk/Reward: {settings.default_risk_reward}:1")
        print(f"   ATR Stop Multiplier: {settings.default_atr_stop_multiplier}x")
        print(f"   Trailing Stop: {'Enabled' if settings.enable_trailing_stop else 'Disabled'}")
        
        # ====================================
        # 5. CORRELATION MANAGER
        # ====================================
        print("\n📈 Initializing Correlation Manager...")
        corr_manager = CorrelationManager(
            max_correlation_threshold=settings.max_correlation_exposure,
            max_sector_exposure_pct=settings.max_sector_exposure_pct,
            max_concurrent_positions=settings.max_concurrent_positions
        )
        print(f"   Max Correlation: {settings.max_correlation_exposure}")
        print(f"   Max Sector Exposure: {settings.max_sector_exposure_pct*100}%")
        print(f"   Max Positions: {settings.max_concurrent_positions}")
        
        # ====================================
        # 6. TRADING SCHEDULER
        # ====================================
        print("\n⏰ Initializing Scheduler...")
        scheduler = TradingScheduler(
            market_data_provider=market_data,
            news_provider=news_provider,
            agent=agent,
            paper_exchange=paper_exchange,
            risk_manager=risk_manager,
            analysis_interval_minutes=settings.analysis_interval_minutes,
            stop_check_seconds=settings.stop_check_seconds,
            state_save_minutes=settings.state_save_minutes,
            symbols=settings.trading_symbols
        )
        
        # Attach correlation manager to scheduler
        scheduler.corr_manager = corr_manager
        
        scheduler_ref = scheduler
        
        print(f"   Analysis Interval: {settings.analysis_interval_minutes} min")
        print(f"   Stop Check: {settings.stop_check_seconds} sec")
        print(f"   Symbols: {len(settings.trading_symbols)}")
        for sym in settings.trading_symbols:
            print(f"      - {sym}")
        
        # ====================================
        # 7. RESTORE STATE
        # ====================================
        print("\n" + "=" * 60)
        await scheduler.restore_state()
        
        # Update correlation matrix with initial data
        print("\n📊 Updating correlation matrix...")
        await corr_manager.update_correlation_matrix(market_data, settings.trading_symbols)
        print("=" * 60)
        
        # ====================================
        # 8. START TRADING
        # ====================================
        print("\n🏁 Starting Trading Scheduler...")
        scheduler.start()
        
        print("\n" + "=" * 60)
        print("✅ BOT IS RUNNING!")
        print("")
        print("   💱 Mode: SPOT TRADING (BUY + SELL)")
        print("   🛡️ Protection: Resilient LLM + Correlation Limits")
        print("   📊 Analysis: Every 15 minutes")
        print("   🛑 Stop Check: Every 30 seconds")
        print("")
        print("   Press Ctrl+C to stop and see performance report")
        print("=" * 60 + "\n")
        
        # Keep the script running until shutdown requested
        logger.info("Bot is running. Waiting for shutdown signal...")
        while not shutdown_requested:
            try:
                # Wait for shutdown event or timeout after 1 second
                await asyncio.wait_for(shutdown_event.wait(), timeout=1.0)
                break  # Shutdown requested
            except asyncio.TimeoutError:
                continue  # Continue running
        
        # Shutdown was requested
        await shutdown(scheduler, market_data, paper_exchange)
            
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
        await shutdown(scheduler, market_data, paper_exchange)
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
        
        # Print report even on error
        if paper_exchange:
            await print_final_report(paper_exchange)
        
        if market_data:
            await market_data.close()


if __name__ == "__main__":
    asyncio.run(main())
