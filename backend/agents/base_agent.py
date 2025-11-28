from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from .memory import AgentMemory


class BaseTradingAgent(ABC):
    """
    Abstract base class for all trading agents.
    
    Features:
    - Memory system for learning from past trades
    - Standardized interface for market analysis
    - Context-aware prompts
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.memory: Optional[AgentMemory] = None
    
    def set_memory(self, memory: AgentMemory):
        """Attach memory system to agent."""
        self.memory = memory
    
    def get_memory_context(self, symbol: str) -> str:
        """Get memory context for prompt."""
        if self.memory:
            return self.memory.build_context_for_prompt(symbol, include_all_symbols=True)
        return "No trading history available."
    
    def get_trading_rules(self) -> str:
        """Get learned trading rules from memory."""
        if self.memory:
            return self.memory.get_trading_rules_from_memory()
        return ""
    
    def record_decision(self, symbol: str, decision: str, confidence: str, reasoning: str, market_snapshot: Dict[str, Any]):
        """Record a trading decision to memory."""
        if self.memory:
            from .memory import DecisionMemory
            self.memory.add_decision(
                symbol=symbol,
                decision=DecisionMemory(
                    symbol=symbol,
                    decision=decision,
                    confidence=confidence,
                    reasoning=reasoning,
                    market_snapshot=market_snapshot
                )
            )

    @abstractmethod
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and return insights.
        
        Args:
            market_data: Dictionary containing:
                - symbol: Trading pair (e.g., "BTC/USDT")
                - price: Current price
                - timeframes: Multi-timeframe indicator data
                - news_sentiment: Sentiment analysis
                - memory_context: (Optional) Past trade context
        
        Returns:
            Dictionary with:
                - trend: "bullish" | "bearish" | "neutral"
                - confidence: "high" | "moderate" | "low"
                - signal_type: "STRONG_BUY" | "MODERATE_BUY" | "WEAK_BUY" | "SELL" | "HOLD"
                - reasoning: Explanation of decision
        """
        pass

    async def analyze_with_memory(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data with memory context included.
        
        This is the preferred method to call as it includes past trade history.
        """
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        # Add memory context to market data
        market_data['memory_context'] = self.get_memory_context(symbol)
        market_data['trading_rules'] = self.get_trading_rules()
        
        # Sync positions if memory available
        if self.memory and 'open_positions' in market_data:
            self.memory.update_open_positions(market_data['open_positions'])
        
        # Call the actual analysis
        result = await self.analyze_market(market_data)
        
        # Record the decision
        if isinstance(result, dict):
            self.record_decision(
                symbol=symbol,
                decision=result.get('signal_type', result.get('trend', 'hold')),
                confidence=result.get('confidence', 'low'),
                reasoning=result.get('reasoning', ''),
                market_snapshot=market_data
            )
        
        return result

    @abstractmethod
    async def generate_signals(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on analysis.
        """
        pass

    @abstractmethod
    async def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade based on a signal.
        """
        pass
