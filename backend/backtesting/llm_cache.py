"""
LLM Response Cache for Backtesting.

Solves the problem of:
- Expensive API calls during backtest
- Slow backtest execution
- Inconsistent results between runs

Usage:
    cache = BacktestLLMCache("./cache/backtest_responses.json")
    
    # Wrap your agent
    cached_agent = cache.wrap_agent(GeminiAgent())
    
    # Run backtest with cached responses
    for candle in historical_data:
        response = await cached_agent.analyze_market(market_data)
        # First run: Calls LLM, caches response
        # Subsequent runs: Returns cached response instantly
"""

import json
import hashlib
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BacktestLLMCache:
    """
    Persistent cache for LLM responses during backtesting.
    
    Benefits:
    - Run backtest once to populate cache
    - Subsequent runs are instant (no API calls)
    - Consistent results across runs
    - Save money on API costs
    """
    
    def __init__(
        self,
        cache_file: str = "./cache/backtest_llm_cache.json",
        enabled: bool = True
    ):
        self.cache_file = Path(cache_file)
        self.enabled = enabled
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        
        # Create cache directory
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache
        self._load_cache()
        
        logger.info(f"BacktestLLMCache initialized. Entries: {len(self.cache)}")
    
    def _load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached responses")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _make_key(self, market_data: Dict[str, Any]) -> str:
        """
        Create a cache key from market data.
        
        Key is based on:
        - Symbol
        - Price (rounded)
        - Key indicators (RSI, ADX, MACD)
        
        We DON'T include timestamp so similar market conditions hit cache.
        """
        timeframes = market_data.get('timeframes', {})
        
        key_data = {
            'symbol': market_data.get('symbol', 'UNKNOWN'),
            'price_bucket': int(market_data.get('price', 0) / 100) * 100,  # Round to 100s
        }
        
        # Add indicator snapshots for each timeframe
        for tf in ['15m', '1h', '4h']:
            if tf in timeframes:
                ind = timeframes[tf].get('indicators', {})
                key_data[f'{tf}_rsi'] = round(ind.get('rsi', 50) / 5) * 5  # Round to 5s
                key_data[f'{tf}_adx'] = round(ind.get('adx', 25) / 5) * 5
                key_data[f'{tf}_macd'] = ind.get('macd', 'neutral')
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response if available."""
        if not self.enabled:
            return None
        
        key = self._make_key(market_data)
        
        if key in self.cache:
            self.hits += 1
            response = self.cache[key]['response']
            response['_cached'] = True
            return response
        
        self.misses += 1
        return None
    
    def set(self, market_data: Dict[str, Any], response: Dict[str, Any]):
        """Cache a response."""
        if not self.enabled:
            return
        
        key = self._make_key(market_data)
        
        self.cache[key] = {
            'response': response,
            'cached_at': datetime.now().isoformat(),
            'symbol': market_data.get('symbol'),
            'price': market_data.get('price')
        }
        
        # Save every 100 entries
        if len(self.cache) % 100 == 0:
            self._save_cache()
    
    def wrap_agent(self, agent):
        """
        Wrap an agent to use caching.
        
        Returns a proxy that checks cache before calling the real agent.
        """
        cache = self
        
        class CachedAgentProxy:
            def __init__(self, real_agent):
                self.real_agent = real_agent
                self.name = f"cached_{getattr(real_agent, 'name', 'agent')}"
            
            async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
                # Check cache first
                cached = cache.get(market_data)
                if cached:
                    return cached
                
                # Call real agent
                response = await self.real_agent.analyze_market(market_data)
                
                # Cache the response
                if isinstance(response, dict):
                    cache.set(market_data, response)
                
                return response
            
            def __getattr__(self, name):
                return getattr(self.real_agent, name)
        
        return CachedAgentProxy(agent)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'total_entries': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate_pct': round(hit_rate, 1),
            'cache_file': str(self.cache_file),
            'enabled': self.enabled
        }
    
    def clear(self):
        """Clear the cache."""
        self.cache = {}
        self.hits = 0
        self.misses = 0
        
        if self.cache_file.exists():
            self.cache_file.unlink()
        
        logger.info("Cache cleared")
    
    def save(self):
        """Manually save cache to disk."""
        self._save_cache()
        logger.info(f"Cache saved: {len(self.cache)} entries")


# Example usage in backtest:
"""
from backend.backtesting.llm_cache import BacktestLLMCache
from backend.agents import GeminiAgent

# Initialize cache
cache = BacktestLLMCache()

# Wrap your agent
agent = GeminiAgent()
cached_agent = cache.wrap_agent(agent)

# Run backtest - first run populates cache
for candle in historical_data:
    market_data = prepare_market_data(candle)
    
    # First run: Calls Gemini API (~500ms)
    # Second run: Returns from cache (~1ms)
    response = await cached_agent.analyze_market(market_data)
    
    process_signal(response)

# Save cache at end
cache.save()

# Print stats
print(cache.get_stats())
# {'hits': 450, 'misses': 50, 'hit_rate_pct': 90.0}
"""

