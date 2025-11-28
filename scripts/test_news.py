import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data.news_sentiment import NewsSentimentProvider

async def main():
    load_dotenv()
    
    print("--- Testing News Sentiment Provider ---\n")
    
    provider = NewsSentimentProvider()
    
    # Test fetching news
    print("Fetching latest crypto news...")
    sentiment_data = await provider.get_market_sentiment()
    
    print(f"\nSentiment Score: {sentiment_data['sentiment_score']}")
    print(f"Sentiment Label: {sentiment_data['sentiment_label'].upper()}")
    print(f"News Count: {sentiment_data['news_count']}")
    
    print("\nTop Headlines:")
    for i, headline in enumerate(sentiment_data['top_headlines'], 1):
        print(f"{i}. {headline['title']}")
        print(f"   Sentiment: {headline['sentiment']:+.2f}\n")

if __name__ == "__main__":
    asyncio.run(main())
