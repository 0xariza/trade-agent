#!/usr/bin/env python3
"""
Backtest Runner Script.

Usage:
    python scripts/run_backtest.py                    # Default BTC 90 days
    python scripts/run_backtest.py --symbol ETH/USDT  # Specific symbol
    python scripts/run_backtest.py --days 180         # Custom period
    python scripts/run_backtest.py --all              # Run all symbols
"""

import asyncio
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.backtesting.backtest_engine import BacktestEngine, BacktestResult


# Default symbols to test
DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT"
]


async def run_single_backtest(symbol: str, days: int) -> BacktestResult:
    """Run backtest on a single symbol."""
    engine = BacktestEngine(
        initial_capital=10000,
        fee_pct=0.001,
        slippage_pct=0.0005,
        risk_per_trade_pct=0.01,
        max_position_pct=0.10
    )
    
    result = await engine.run_backtest(symbol, days)
    return result


async def run_all_backtests(symbols: list, days: int):
    """Run backtests on multiple symbols."""
    results = []
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Running backtest: {symbol}")
        print('='*60)
        
        try:
            result = await run_single_backtest(symbol, days)
            result.print_summary()
            results.append(result)
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Print comparison
    if len(results) > 1:
        print("\n" + "="*70)
        print("📊 BACKTEST COMPARISON")
        print("="*70)
        print(f"{'Symbol':<12} {'Return':>10} {'Sharpe':>8} {'Win Rate':>10} {'Max DD':>10} {'Trades':>8}")
        print("-"*70)
        
        for r in sorted(results, key=lambda x: x.total_return_pct, reverse=True):
            print(
                f"{r.symbol:<12} "
                f"{r.total_return_pct:>9.1f}% "
                f"{r.sharpe_ratio:>8.2f} "
                f"{r.win_rate:>9.1f}% "
                f"{r.max_drawdown_pct:>9.1f}% "
                f"{r.total_trades:>8}"
            )
        
        # Calculate portfolio metrics
        avg_return = sum(r.total_return_pct for r in results) / len(results)
        avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results)
        avg_win_rate = sum(r.win_rate for r in results) / len(results)
        
        print("-"*70)
        print(f"{'AVERAGE':<12} {avg_return:>9.1f}% {avg_sharpe:>8.2f} {avg_win_rate:>9.1f}%")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Run strategy backtest')
    parser.add_argument('--symbol', '-s', type=str, default='BTC/USDT',
                        help='Trading symbol (e.g., BTC/USDT)')
    parser.add_argument('--days', '-d', type=int, default=90,
                        help='Number of days to backtest')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Run backtest on all default symbols')
    parser.add_argument('--capital', '-c', type=float, default=10000,
                        help='Initial capital')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔬 ALPHA ARENA STRATEGY BACKTESTER")
    print("="*60)
    print(f"Period: {args.days} days")
    print(f"Initial Capital: ${args.capital:,.2f}")
    
    if args.all:
        print(f"Symbols: {', '.join(DEFAULT_SYMBOLS)}")
        asyncio.run(run_all_backtests(DEFAULT_SYMBOLS, args.days))
    else:
        print(f"Symbol: {args.symbol}")
        result = asyncio.run(run_single_backtest(args.symbol, args.days))
        result.print_summary()
        
        # Save detailed results
        if result.total_trades > 0:
            print("\n📝 Trade Details:")
            print("-"*70)
            for i, trade in enumerate(result.trades[-10:], 1):
                emoji = "✅" if trade.pnl > 0 else "❌"
                print(
                    f"  {i}. {emoji} {trade.direction.value.upper()} "
                    f"${trade.entry_price:,.0f} → ${trade.exit_price:,.0f} "
                    f"| P&L: {trade.pnl_pct:+.2f}% "
                    f"| Exit: {trade.exit_reason}"
                )


if __name__ == "__main__":
    main()
