import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents import OpenRouterAgent
from backend.data.market_data import MarketDataProvider
from backend.exchanges.paper_exchange import PaperExchange
from backend.backtesting.backtester import Backtester

# Define crash periods
CRASH_PERIODS = {
    "May 2021 Crypto Crash": {
        "start": "2021-05-10",
        "end": "2021-05-23",
        "description": "BTC dropped from ~58k to ~30k (-48%)"
    },
    "March 2020 COVID Crash": {
        "start": "2020-03-08",
        "end": "2020-03-16",
        "description": "BTC dropped from ~9k to ~3.8k (-58%)"
    }
}

async def run_stress_test(period_name, start_date, end_date, description):
    """
    Run backtest on a specific crash period.
    """
    print(f"\n{'='*60}")
    print(f"STRESS TEST: {period_name}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Context: {description}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize components
        agent = OpenRouterAgent(config={"model": "openai/gpt-4o"})
        market_data = MarketDataProvider(exchange_id='binance')
        paper_exchange = PaperExchange(initial_balance=10000.0, slippage_pct=0.001, fee_pct=0.001)
        
        # Fetch historical data for the period
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        # Fetch OHLCV data
        print(f"Fetching historical data from {start_date} to {end_date}...")
        
        # Use the async exchange from market_data
        # Note: ccxt async exchanges need to be loaded first
        await market_data.exchange.load_markets()
        
        # Fetch data in chunks (ccxt has limits)
        all_candles = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            candles = await market_data.exchange.fetch_ohlcv('BTC/USDT', '1h', since=current_ts, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            current_ts = candles[-1][0] + 1  # Move to next timestamp
            
            # Stop if we've reached the end
            if candles[-1][0] >= end_ts:
                break
        
        # Filter to exact period
        filtered_candles = [c for c in all_candles if start_ts <= c[0] <= end_ts]
        
        if not filtered_candles:
            print(f"No data found for period {start_date} to {end_date}")
            return
        
        print(f"Loaded {len(filtered_candles)} candles")
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(filtered_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate indicators (same as backtester)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma_20'] + (df['std_20'] * 2)
        df['lower_band'] = df['sma_20'] - (df['std_20'] * 2)
        
        # Run simplified backtest (without LLM to save costs)
        # We'll use a simple strategy: Buy when RSI < 30, Sell when RSI > 70
        print("Running stress test with simple RSI strategy (to save API costs)...")
        
        initial_balance = 10000.0
        equity_curve = [initial_balance]
        
        for index, row in df.iterrows():
            price = row['close']
            rsi = row['rsi']
            
            # Simple strategy
            if pd.notna(rsi):
                if rsi < 30 and paper_exchange.get_balance("USDT") > 100:
                    # Buy signal
                    amount = 0.01
                    paper_exchange.execute_order("BTC/USDT", "buy", amount, price)
                elif rsi > 70 and paper_exchange.get_position("BTC") > 0.001:
                    # Sell signal
                    amount = 0.01
                    paper_exchange.execute_order("BTC/USDT", "sell", amount, price)
            
            # Track equity
            current_equity = paper_exchange.get_portfolio_value({"BTC/USDT": price})
            equity_curve.append(current_equity)
        
        # Calculate metrics
        import numpy as np
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        final_equity = equity_curve[-1]
        total_return_pct = ((final_equity - initial_balance) / initial_balance) * 100
        
        if returns.std() != 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(24 * 365)
        else:
            sharpe_ratio = 0
        
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown_pct = drawdown.min() * 100
        
        # Report
        print(f"\n--- STRESS TEST RESULTS: {period_name} ---")
        print(f"Initial Equity:   ${initial_balance:,.2f}")
        print(f"Final Equity:     ${final_equity:,.2f}")
        print(f"Total Return:     {total_return_pct:.2f}%")
        print(f"Sharpe Ratio:     {sharpe_ratio:.2f}")
        print(f"Max Drawdown:     {max_drawdown_pct:.2f}%")
        print(f"Trades Executed:  {len(paper_exchange.trade_history)}")
        print(f"Survival:         {'✓ YES' if final_equity > 0 else '✗ BLOWN UP'}")
        print(f"{'='*60}\n")
        
        await market_data.close()
        
    except Exception as e:
        print(f"Error during stress test: {e}")
        import traceback
        traceback.print_exc()

async def main():
    load_dotenv()
    
    print("\n" + "="*60)
    print("ALPHA ARENA - STRESS TESTING")
    print("Testing strategy survival on historical crash data")
    print("="*60)
    
    for period_name, config in CRASH_PERIODS.items():
        await run_stress_test(
            period_name,
            config["start"],
            config["end"],
            config["description"]
        )
    
    print("\n" + "="*60)
    print("STRESS TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
