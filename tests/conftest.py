"""
Pytest configuration and fixtures for Alpha Arena tests.
"""
import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_market_data():
    """Mock market data for testing."""
    return {
        "symbol": "BTC/USDT",
        "price": 50000.0,
        "timeframes": {
            "15m": {
                "rsi": 55.0,
                "macd": {"macd": 100, "signal": 95, "histogram": 5},
                "bb": {"upper": 51000, "middle": 50000, "lower": 49000},
                "adx": 25.0,
                "atr": 500.0,
                "ema_9": 50100,
                "ema_21": 49900,
                "ema_50": 49800,
            },
            "1h": {
                "rsi": 60.0,
                "macd": {"macd": 150, "signal": 140, "histogram": 10},
                "bb": {"upper": 52000, "middle": 50000, "lower": 48000},
                "adx": 30.0,
                "atr": 800.0,
            },
            "4h": {
                "rsi": 65.0,
                "macd": {"macd": 200, "signal": 180, "histogram": 20},
                "bb": {"upper": 53000, "middle": 50000, "lower": 47000},
                "adx": 35.0,
                "atr": 1200.0,
            },
        },
        "btc_trend": {"trend": "bullish", "change_1h": 0.5, "change_24h": 2.0},
        "news_sentiment": {"sentiment_label": "neutral", "sentiment_score": 0.0},
        "funding_rate": 0.01,
        "fear_greed": 60,
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM agent response."""
    return {
        "signal_type": "BUY",
        "confidence": 0.75,
        "reasoning": "Strong bullish momentum with RSI at 60 and MACD crossing above signal",
        "entry_price": 50000.0,
        "stop_loss": 49000.0,
        "take_profit": 52000.0,
        "market_regime": "trending",
    }


@pytest.fixture
def mock_paper_exchange():
    """Mock paper exchange for testing."""
    exchange = Mock()
    exchange.get_balance = Mock(return_value=10000.0)
    exchange.get_all_positions = Mock(return_value={})
    exchange.open_position = AsyncMock(return_value=True)
    exchange.close_position = AsyncMock(return_value={"pnl": 100.0, "pnl_pct": 1.0})
    exchange.get_position = Mock(return_value=None)
    return exchange


@pytest.fixture
def mock_market_data_provider():
    """Mock market data provider."""
    provider = AsyncMock()
    provider.get_ohlcv = AsyncMock(return_value=None)
    provider.get_btc_trend = AsyncMock(return_value={"trend": "bullish", "change_1h": 0.5})
    provider.close = AsyncMock()
    return provider


@pytest.fixture
def mock_risk_manager():
    """Mock risk manager."""
    manager = Mock()
    manager.calculate_position_size = Mock(return_value=0.1)
    manager.calculate_stop_loss = Mock(return_value=49000.0)
    manager.calculate_take_profit = Mock(return_value=52000.0)
    manager.check_risk_limits = Mock(return_value=(True, "OK"))
    return manager


@pytest.fixture
def sample_settings():
    """Sample settings for testing."""
    from backend.config import Settings
    
    return Settings(
        trading_mode="paper",
        risk_profile="conservative",
        initial_balance=10000.0,
        max_position_size_pct=0.10,
        max_drawdown_pct=0.15,
        trading_symbols=["BTC/USDT", "ETH/USDT"],
    )


@pytest.fixture
def mock_agent():
    """Mock trading agent."""
    agent = AsyncMock()
    agent.analyze_market = AsyncMock(return_value={
        "signal_type": "BUY",
        "confidence": 0.75,
        "reasoning": "Test reasoning",
    })
    agent.name = "MockAgent"
    return agent

