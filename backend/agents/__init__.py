from .base_agent import BaseTradingAgent
from .memory import AgentMemory, TradeMemory, DecisionMemory
from .reflection import TradeReflector, ReflectionInsight, TradeAnalysis
from .rule_based_agent import RuleBasedAgent, SignalType, MarketRegime
from .hybrid_agent import HybridAgent, RulesOnlyAgent
from .gemini_agent import GeminiAgent
from .gpt_agent import GPTAgent
from .deepseek_agent import DeepSeekAgent
from .qwen_agent import QwenAgent
from .openrouter_agent import OpenRouterAgent

# Optional imports - only if dependencies are installed
try:
    from .claude_agent import ClaudeAgent
except ImportError:
    ClaudeAgent = None  # anthropic not installed

__all__ = [
    "BaseTradingAgent",
    "AgentMemory",
    "TradeMemory",
    "DecisionMemory",
    "TradeReflector",
    "ReflectionInsight",
    "TradeAnalysis",
    "RuleBasedAgent",
    "SignalType",
    "MarketRegime",
    "HybridAgent",
    "RulesOnlyAgent",
    "GeminiAgent",
    "GPTAgent",
    "DeepSeekAgent",
    "QwenAgent",
    "OpenRouterAgent",
    "ClaudeAgent"
]
