import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.claude_agent import ClaudeAgent
from backend.data.market_data import MarketDataProvider
from backend.data.news_sentiment import NewsSentimentProvider

async def main():
    # Load environment variables
    load_dotenv()
    
    print("--- Claude Opus 4.5 Trading Agent Test ---")
    
    # Initialize Agent with Opus 4.5
    # Model options:
    # - "claude-opus-4.5" (Opus 4.5 - default, try this first)
    # - "claude-3-5-opus-20241022" (Opus 3.5 fallback if 4.5 not available)
    # - "claude-3-opus-20240229" (older Opus)
    agent = ClaudeAgent(
        name="claude_opus_trader",
        config={"model": "claude-opus-4.5"}  # Opus 4.5
    )
    print(f"Agent initialized: {agent.name} ({agent.model})")
    
    # Initialize Data Providers
    market_data = MarketDataProvider(exchange_id='binance')
    news_provider = NewsSentimentProvider()
    
    try:
        # Get market snapshot
        symbol = "BTC/USDT"
        print(f"\nFetching market data for {symbol}...")
        market_snapshot = await market_data.get_market_snapshot(symbol)
        
        print(f"Price: ${market_snapshot['price']:,.2f}")
        
        # Get news sentiment
        print("Fetching news sentiment...")
        news_sentiment = await news_provider.get_market_sentiment()
        market_snapshot['news_sentiment'] = news_sentiment
        print(f"News Sentiment: {news_sentiment['sentiment_label']} ({news_sentiment['sentiment_score']:+.2f})")
        
        # Analyze with Claude
        print("\n" + "="*50)
        print("Claude Opus 4.5 Analysis:")
        print("="*50)
        analysis = await agent.analyze_market(market_snapshot)
        
        print(f"\nDecision: {analysis.get('signal_type', 'HOLD')}")
        print(f"Trend: {analysis.get('trend', 'neutral').upper()}")
        print(f"Confidence: {analysis.get('confidence', 'low').upper()}")
        
        if 'weighted_score' in analysis:
            print(f"Weighted Score: {analysis.get('weighted_score'):.2f}")
        if 'confluence' in analysis:
            print(f"Confluence: {analysis.get('confluence')}")
        if 'market_regime' in analysis:
            print(f"Market Regime: {analysis.get('market_regime')}")
        
        print(f"\nReasoning:\n{analysis.get('reasoning', 'N/A')}")
        
        if 'scoring_breakdown' in analysis:
            print("\nScoring Breakdown:")
            for tf, explanation in analysis.get('scoring_breakdown', {}).items():
                score = analysis.get('scores', {}).get(tf, 'N/A')
                print(f"  {tf}: {score} - {explanation}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await market_data.close()
        print("\nTest completed.")

if __name__ == "__main__":
    asyncio.run(main())

