import asyncio
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
import logging

# Configure logging
logger = logging.getLogger(__name__)

import numpy as np

class Backtester:
    """
    Simulates trading strategy on historical data.
    """
    
    def __init__(self, agent, market_data_provider, paper_exchange):
        self.agent = agent
        self.market_data_provider = market_data_provider
        self.paper_exchange = paper_exchange
        self.results = []
        self.equity_curve = [] # Track equity over time

    async def run(self, symbol: str, timeframe: str = '1h', limit: int = 50):
        """
        Run the backtest.
        """
        print(f"Starting Backtest for {symbol} ({timeframe}) - Last {limit} candles")
        
        # 1. Fetch Historical Data
        # We need enough data to calculate indicators (e.g., +100 candles for warmup)
        fetch_limit = limit + 100 
        df = await self.market_data_provider.get_ohlcv(symbol, timeframe, limit=fetch_limit)
        
        if df.empty:
            print("No data found for backtest.")
            return

        # Calculate Indicators on the full dataset first
        # (Simulating that we have history available)
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['sma_20'] + (df['std_20'] * 2)
        df['lower_band'] = df['sma_20'] - (df['std_20'] * 2)

        # 2. Iterate through the "Test" period (the last `limit` candles)
        test_data = df.iloc[-limit:].copy()
        
        # Initial Equity
        initial_balance = self.paper_exchange.get_balance("USDT")
        self.equity_curve.append(initial_balance)
        
        for index, row in test_data.iterrows():
            timestamp = row['timestamp']
            price = row['close']
            
            print(f"Processing Candle: {timestamp} | Price: ${price:,.2f}")
            
            # Construct Snapshot for Agent
            # Note: In a real backtest, we should be careful not to look ahead.
            # Here we use the indicators calculated on the full history up to this point.
            
            macd_signal = "bullish" if row['macd'] > row['signal'] else "bearish"
            
            # 24h change approximation
            change_24h = 0.0 
            
            snapshot = {
                "symbol": symbol,
                "price": price,
                "volume_24h": row['volume'],
                "change_24h": change_24h, # Placeholder
                "indicators": {
                    "rsi": round(row['rsi'], 2) if pd.notna(row['rsi']) else 50,
                    "macd": macd_signal,
                    "macd_value": round(row['macd'], 2) if pd.notna(row['macd']) else 0,
                    "bollinger_position": "upper" if price > row['upper_band'] else "lower" if price < row['lower_band'] else "middle"
                },
                "last_updated": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
            }
            
            # 3. Agent Analysis
            # WARNING: This calls the LLM API.
            try:
                analysis_json = await self.agent.analyze_market(snapshot)
                
                # Parse Decision
                import json
                if isinstance(analysis_json, str):
                    cleaned_json = analysis_json.replace('```json', '').replace('```', '').strip()
                    decision = json.loads(cleaned_json)
                else:
                    decision = analysis_json
                
                trend = decision.get("trend", "").lower()
                confidence = decision.get("confidence", "low").lower()
                
                # 4. Execute Trade (Paper)
                trade_amount = 0.001
                signal = None
                
                if "bullish" in trend and ("high" in confidence or "moderate" in confidence):
                    signal = 'buy'
                elif "bearish" in trend:
                    signal = 'sell'
                
                if signal:
                    self.paper_exchange.execute_order(symbol, signal, trade_amount, price)
                    print(f"  -> Action: {signal.upper()}")
                else:
                    print(f"  -> Action: HOLD")
                    
            except Exception as e:
                print(f"  -> Error: {e}")
            
            # Track Equity
            current_equity = self.paper_exchange.get_portfolio_value({symbol: price})
            self.equity_curve.append(current_equity)
                
        # 5. Report Results
        self.calculate_metrics(test_data)

    def calculate_metrics(self, test_data):
        """
        Calculate and print advanced performance metrics.
        """
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        final_equity = self.equity_curve[-1]
        initial_equity = self.equity_curve[0]
        total_return_pct = ((final_equity - initial_equity) / initial_equity) * 100
        
        # Sharpe Ratio (Assuming hourly data, annualized)
        # Risk-free rate assumed 0 for simplicity
        if returns.std() != 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(24 * 365) 
        else:
            sharpe_ratio = 0
            
        # Max Drawdown
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown_pct = drawdown.min() * 100
        
        # Win Rate
        trades = self.paper_exchange.trade_history
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0] # Note: PaperExchange needs to track PnL per trade to calculate this accurately. 
        # For now, we'll just count trades, as PnL calculation per trade is complex without a closed trade list.
        # We will skip Win Rate for now or implement a simple version if PnL is available.
        
        print("\n--- Advanced Backtest Report ---")
        print(f"Initial Equity:   ${initial_equity:,.2f}")
        print(f"Final Equity:     ${final_equity:,.2f}")
        print(f"Total Return:     {total_return_pct:.2f}%")
        print(f"Sharpe Ratio:     {sharpe_ratio:.2f} (Annualized)")
        print(f"Max Drawdown:     {max_drawdown_pct:.2f}%")
        print(f"Total Trades:     {len(trades)}")
        print("--------------------------------")
