import os
import logging
import asyncio
import aiohttp
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class NewsSentimentProvider:
    """
    Fetches crypto news and analyzes sentiment.
    """
    
    def __init__(self):
        self.api_key = os.getenv("CRYPTOCOMPARE_API_KEY")
        if not self.api_key:
            logger.warning("CRYPTOCOMPARE_API_KEY not found. News sentiment will be unavailable.")
        self.base_url = "https://min-api.cryptocompare.com/data/v2/news/"
        
        # Sentiment keywords
        self.positive_keywords = [
            'bullish', 'surge', 'rally', 'gain', 'profit', 'adoption', 
            'breakthrough', 'partnership', 'upgrade', 'positive', 'growth',
            'soar', 'jump', 'rise', 'institutional', 'approval'
        ]
        
        self.negative_keywords = [
            'bearish', 'crash', 'drop', 'fall', 'loss', 'hack', 'scam',
            'regulation', 'ban', 'lawsuit', 'fraud', 'decline', 'plunge',
            'fear', 'uncertainty', 'doubt', 'fud', 'sell-off', 'collapse'
        ]
    
    async def get_latest_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch latest crypto news from CryptoCompare.
        """
        if not self.api_key:
            logger.warning("No CryptoCompare API key. News sentiment disabled.")
            return []
        
        try:
            params = {
                'api_key': self.api_key,
                'lang': 'EN'
            }
            
            # Create SSL context that doesn't verify certificates (safe for public APIs)
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Try different response structures
                        news_items = []
                        
                        if isinstance(data, list):
                            # Direct list response
                            news_items = data
                        elif isinstance(data, dict):
                            # Try nested structures
                            if 'Data' in data:
                                data_obj = data['Data']
                                if isinstance(data_obj, list):
                                    news_items = data_obj
                                elif isinstance(data_obj, dict) and 'List' in data_obj:
                                    news_items = data_obj['List']
                        
                        if news_items:
                            logger.info(f"Fetched {len(news_items)} news items")
                            return news_items[:limit]
                        else:
                            logger.warning("No news items found in API response")
                            return []
                    else:
                        logger.error(f"API returned status {response.status}")
                        return []
        except asyncio.TimeoutError:
            logger.error("News API request timed out")
            return []
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Simple keyword-based sentiment analysis.
        Returns a score from -1 (very negative) to +1 (very positive).
        """
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        total_count = positive_count + negative_count
        
        if total_count == 0:
            return 0.0  # Neutral
        
        # Calculate sentiment score
        sentiment = (positive_count - negative_count) / total_count
        return sentiment
    
    async def get_market_sentiment(self) -> Dict[str, Any]:
        """
        Get overall market sentiment from recent news.
        """
        news_items = await self.get_latest_news(limit=20)
        
        if not news_items:
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "news_count": 0,
                "top_headlines": []
            }
        
        # Analyze sentiment for each news item
        sentiments = []
        headlines = []
        
        for item in news_items:
            title = item.get('title', '')
            body = item.get('body', '')
            
            # Combine title and body for analysis (title weighted more)
            combined_text = f"{title} {title} {body}"
            sentiment = self.analyze_sentiment(combined_text)
            sentiments.append(sentiment)
            
            # Store top headlines
            if len(headlines) < 5:
                headlines.append({
                    "title": title,
                    "sentiment": sentiment,
                    "published": item.get('published_on', 0)
                })
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        
        # Classify sentiment
        if avg_sentiment > 0.2:
            label = "positive"
        elif avg_sentiment < -0.2:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "sentiment_score": round(avg_sentiment, 2),
            "sentiment_label": label,
            "news_count": len(news_items),
            "top_headlines": headlines[:3]  # Top 3 headlines
        }
