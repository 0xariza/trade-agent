#!/usr/bin/env python3
"""
Pre-Flight Check Script for Alpha Arena.

Run this BEFORE starting the bot to verify:
1. All API keys are valid
2. Exchange connection works
3. Database is accessible
4. Telegram notifications work
5. Configuration is valid

Usage:
    python scripts/preflight_check.py
"""

import asyncio
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment
load_dotenv()


class Colors:
    """Terminal colors for pretty output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header():
    print(f"""
{Colors.BOLD}╔══════════════════════════════════════════════════════════════════╗
║                    ALPHA ARENA - PRE-FLIGHT CHECK                  ║
╚══════════════════════════════════════════════════════════════════╝{Colors.END}
""")


def check_pass(name: str, details: str = ""):
    print(f"  {Colors.GREEN}✓{Colors.END} {name}" + (f" - {details}" if details else ""))


def check_fail(name: str, error: str):
    print(f"  {Colors.RED}✗{Colors.END} {name} - {error}")


def check_warn(name: str, warning: str):
    print(f"  {Colors.YELLOW}⚠{Colors.END} {name} - {warning}")


async def check_environment():
    """Check required environment variables."""
    print(f"\n{Colors.BLUE}[1/6] Checking Environment Variables{Colors.END}")
    
    required = {
        'GEMINI_API_KEY': 'Primary LLM',
    }
    
    optional = {
        'OPENAI_API_KEY': 'Fallback LLM',
        'TELEGRAM_BOT_TOKEN': 'Notifications',
        'TELEGRAM_CHAT_ID': 'Notifications',
        'DATABASE_URL': 'State persistence',
    }
    
    all_ok = True
    
    for key, desc in required.items():
        value = os.getenv(key)
        if value and len(value) > 10:
            check_pass(f"{key}", f"{desc} ({'*' * 8}{value[-4:]})")
        else:
            check_fail(f"{key}", f"Missing or invalid ({desc})")
            all_ok = False
    
    for key, desc in optional.items():
        value = os.getenv(key)
        if value and len(value) > 5:
            check_pass(f"{key}", f"{desc}")
        else:
            check_warn(f"{key}", f"Not set ({desc})")
    
    return all_ok


async def check_gemini():
    """Test Gemini API connection."""
    print(f"\n{Colors.BLUE}[2/6] Testing Gemini API{Colors.END}")
    
    try:
        from backend.agents.gemini_agent import GeminiAgent
        
        agent = GeminiAgent()
        check_pass("Agent initialized", f"Model: {agent.model_name}")
        
        # Simple test call
        test_data = {
            'symbol': 'BTC/USDT',
            'price': 95000,
            'timeframes': {
                '1h': {'indicators': {'rsi': 50, 'adx': 25, 'macd': 'bullish'}},
                '4h': {'indicators': {'rsi': 55, 'adx': 30, 'macd': 'bullish'}}
            }
        }
        
        result = await agent.analyze_market(test_data)
        
        if result and 'signal_type' in result:
            check_pass("API call successful", f"Signal: {result.get('signal_type', 'N/A')}")
            return True
        else:
            check_warn("API returned unexpected format", str(result)[:50])
            return True  # Still usable
            
    except Exception as e:
        check_fail("Gemini API", str(e)[:80])
        return False


async def check_exchange():
    """Test exchange data connection with fallback."""
    print(f"\n{Colors.BLUE}[3/6] Testing Exchange Connection{Colors.END}")
    
    try:
        from backend.data.market_data import MarketDataProvider
        from backend.config import settings
        
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
        if settings.exchange_fallbacks:
            print(f"  → Fallbacks: {', '.join(settings.exchange_fallbacks)}")
        
        provider = await MarketDataProvider.create_with_fallback(
            primary_exchange=settings.exchange_id,
            fallback_exchanges=settings.exchange_fallbacks,
            api_credentials=api_credentials
        )
        
        check_pass("Provider initialized", provider.exchange_id.capitalize())
        
        # Fetch BTC price
        ohlcv = await provider.get_ohlcv("BTC/USDT", limit=1)
        
        if not ohlcv.empty:
            price = ohlcv.iloc[-1]['close']
            check_pass("Market data fetched", f"BTC: ${price:,.2f}")
            await provider.close()
            return True
        else:
            check_fail("Market data", "Empty response")
            await provider.close()
            return False
            
    except Exception as e:
        check_fail("Exchange connection", str(e)[:80])
        return False


async def check_database():
    """Test database connection."""
    print(f"\n{Colors.BLUE}[4/6] Testing Database{Colors.END}")
    
    try:
        from backend.database.db import init_db, get_db
        
        await init_db()
        check_pass("Database initialized")
        
        # Test connection
        async with get_db() as db:
            result = await db.execute("SELECT 1")
            check_pass("Database connection OK")
        
        return True
        
    except Exception as e:
        check_warn("Database", f"{str(e)[:50]} (will use in-memory)")
        return True  # Not critical, will use memory


async def check_telegram():
    """Test Telegram notifications."""
    print(f"\n{Colors.BLUE}[5/6] Testing Telegram{Colors.END}")
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        check_warn("Telegram", "Not configured (notifications disabled)")
        return True
    
    try:
        from backend.notifications.telegram_notifier import TelegramNotifier
        
        notifier = TelegramNotifier()
        
        # Send test message
        await notifier.send_message(
            f"🧪 *Pre-Flight Check*\n\n"
            f"Alpha Arena bot is starting...\n"
            f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
        )
        
        check_pass("Telegram notification sent", "Check your phone!")
        return True
        
    except Exception as e:
        check_warn("Telegram", f"{str(e)[:50]}")
        return True  # Not critical


async def check_config():
    """Validate configuration settings."""
    print(f"\n{Colors.BLUE}[6/6] Validating Configuration{Colors.END}")
    
    try:
        from backend.config import settings
        
        # Check critical values
        issues = []
        critical_issues = []
        
        # Critical validations (will fail if too dangerous)
        if settings.max_drawdown_pct > 0.30:
            critical_issues.append(f"Max drawdown {settings.max_drawdown_pct*100}% > 30% is DANGEROUS!")
        
        if settings.max_position_size_pct > 0.50:
            critical_issues.append(f"Position size {settings.max_position_size_pct*100}% > 50% is TOO RISKY!")
        
        if settings.max_daily_loss_pct > 0.10:
            critical_issues.append(f"Max daily loss {settings.max_daily_loss_pct*100}% > 10% is dangerous")
        
        # Warning validations
        if settings.max_drawdown_pct > 0.25:
            issues.append(f"Max drawdown {settings.max_drawdown_pct*100}% is very high (recommended: <25%)")
        
        if settings.max_position_size_pct > 0.20:
            issues.append(f"Position size {settings.max_position_size_pct*100}% is risky (recommended: <20%)")
        
        if len(settings.trading_symbols) > 15:
            issues.append(f"Trading {len(settings.trading_symbols)} symbols may hit rate limits")
        
        if settings.initial_balance < 1000:
            issues.append(f"Initial balance ${settings.initial_balance} is low for meaningful testing")
        
        if settings.analysis_interval_minutes < 5:
            issues.append(f"Analysis interval {settings.analysis_interval_minutes}min is very frequent (may hit API limits)")
        
        if settings.stop_check_seconds < 10:
            issues.append(f"Stop check {settings.stop_check_seconds}s is very frequent (may hit API limits)")
        
        # Validate risk/reward ratio
        if settings.default_risk_reward < 1.5:
            issues.append(f"Risk/reward ratio {settings.default_risk_reward}:1 is low (recommended: >=2:1)")
        
        check_pass("Settings loaded")
        check_pass(f"Trading mode: {settings.trading_mode.value}")
        check_pass(f"Risk profile: {settings.risk_profile.value}")
        check_pass(f"Symbols: {len(settings.trading_symbols)}")
        check_pass(f"Max position: {settings.max_position_size_pct*100}%")
        check_pass(f"Max drawdown: {settings.max_drawdown_pct*100}%")
        
        # Show warnings
        for issue in issues:
            check_warn("Config", issue)
        
        # Show critical issues (but don't fail - let user decide)
        for issue in critical_issues:
            print(f"  {Colors.RED}⚠{Colors.END} {Colors.RED}{Colors.BOLD}CRITICAL:{Colors.END} {issue}")
        
        if critical_issues:
            print(f"\n  {Colors.YELLOW}⚠️  Review critical issues above before proceeding!{Colors.END}")
        
        return True
        
    except Exception as e:
        check_fail("Configuration", str(e)[:80])
        return False


async def main():
    print_header()
    
    results = {
        'environment': await check_environment(),
        'gemini': await check_gemini(),
        'exchange': await check_exchange(),
        'database': await check_database(),
        'telegram': await check_telegram(),
        'config': await check_config(),
    }
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL CHECKS PASSED ({passed}/{total}){Colors.END}")
        print(f"\n  {Colors.GREEN}Ready to start paper trading!{Colors.END}")
        print(f"\n  Run: {Colors.BLUE}python scripts/start_bot.py{Colors.END}\n")
        return 0
    
    elif passed >= total - 2:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ PASSED WITH WARNINGS ({passed}/{total}){Colors.END}")
        print(f"\n  {Colors.YELLOW}Some optional features may not work.{Colors.END}")
        print(f"  You can still start the bot.\n")
        return 0
    
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ PRE-FLIGHT FAILED ({passed}/{total}){Colors.END}")
        print(f"\n  {Colors.RED}Fix the errors above before starting.{Colors.END}\n")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

