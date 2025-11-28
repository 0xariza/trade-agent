import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.notifications.telegram_notifier import TelegramNotifier

async def main():
    load_dotenv()
    
    print("--- Testing Telegram Notifier ---\n")
    
    notifier = TelegramNotifier()
    
    if not notifier.enabled:
        print("❌ Telegram not configured.")
        print("\nTo enable:")
        print("1. Create bot via @BotFather")
        print("2. Get bot token")
        print("3. Get chat ID from: https://api.telegram.org/bot<TOKEN>/getUpdates")
        print("4. Add to .env:")
        print("   TELEGRAM_BOT_TOKEN=your_token")
        print("   TELEGRAM_CHAT_ID=your_chat_id")
        return
    
    print("✅ Telegram configured. Sending test messages...\n")
    
    # Test 1: Startup message
    print("1. Sending startup message...")
    await notifier.send_startup_message()
    await asyncio.sleep(2)
    
    # Test 2: Trade alert
    print("2. Sending trade alert...")
    await notifier.send_trade_alert(
        symbol="BTC/USDT",
        side="buy",
        amount=0.01,
        price=84500.00,
        reasoning="RSI oversold (28), MACD bullish crossover, ADX showing strong trend (32)"
    )
    await asyncio.sleep(2)
    
    # Test 3: Daily summary
    print("3. Sending daily summary...")
    await notifier.send_daily_summary(
        pnl=250.50,
        pnl_pct=2.5,
        trades_count=5,
        start_balance=10000.00,
        end_balance=10250.50
    )
    await asyncio.sleep(2)
    
    # Test 4: Circuit breaker
    print("4. Sending circuit breaker alert...")
    await notifier.send_circuit_breaker_alert(
        loss_pct=-5.2,
        current_balance=9480.00,
        start_balance=10000.00
    )
    
    print("\n✅ All test messages sent! Check your Telegram.")

if __name__ == "__main__":
    asyncio.run(main())
