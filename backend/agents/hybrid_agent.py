"""
Hybrid Trading Agent - Rules + LLM Confirmation.

This agent combines the best of both approaches:
1. Rule-based system generates signals (fast, deterministic)
2. LLM confirms/rejects signals (adds context, catches edge cases)

Benefits:
- Consistent baseline from rules
- LLM provides additional filtering
- Reduces false positives
- Can explain decisions in natural language
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import google.generativeai as genai

from .rule_based_agent import RuleBasedAgent, SignalType, TradingSignal

logger = logging.getLogger(__name__)


class HybridAgent:
    """
    Hybrid agent combining rule-based signals with LLM confirmation.
    
    Flow:
    1. Rule-based agent generates signal
    2. If signal is actionable (BUY/SELL):
       - LLM reviews the signal with context
       - LLM can CONFIRM, REDUCE, or REJECT
    3. If signal is HOLD:
       - Skip LLM call (save costs)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.name = "hybrid_trader"
        self.config = config or {}
        
        # Initialize rule-based agent
        self.rule_agent = RuleBasedAgent(config)
        
        # Initialize LLM (Gemini)
        self.use_llm = self.config.get('use_llm', True)
        self.llm_model = None
        
        if self.use_llm:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                model_name = self.config.get("model", "models/gemini-2.0-flash")
                self.llm_model = genai.GenerativeModel(model_name)
                logger.info(f"Hybrid Agent initialized with LLM: {model_name}")
            else:
                logger.warning("No GEMINI_API_KEY found. Running in rules-only mode.")
                self.use_llm = False
        
        # Track LLM usage
        self.llm_calls = 0
        self.llm_confirmations = 0
        self.llm_rejections = 0
        
        logger.info(f"Hybrid Agent initialized. LLM enabled: {self.use_llm}")
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data using hybrid approach.
        
        Args:
            market_data: Market data with indicators
            
        Returns:
            Trading decision with signal, confidence, reasoning
        """
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        # Step 1: Get rule-based signal
        rule_signal = await self.rule_agent.analyze_market(market_data)
        signal_type = rule_signal.get('signal_type', 'HOLD')
        
        # Step 2: Check if LLM confirmation needed
        if not self.use_llm or self.llm_model is None:
            # Rules-only mode
            return rule_signal
        
        # Only call LLM for actionable signals
        actionable_signals = [
            'STRONG_BUY', 'MODERATE_BUY', 'WEAK_BUY', 'OVERSOLD_BUY',
            'SELL', 'STRONG_SELL', 'WEAK_SELL'
        ]
        
        if signal_type not in actionable_signals:
            # HOLD signal - no LLM needed
            rule_signal['llm_confirmed'] = None
            rule_signal['llm_note'] = "Skipped (HOLD signal)"
            return rule_signal
        
        # Step 3: Get LLM confirmation
        try:
            llm_decision = await self._get_llm_confirmation(
                market_data, rule_signal, signal_type
            )
            self.llm_calls += 1
            
            # Step 4: Combine decisions
            final_signal = self._combine_decisions(rule_signal, llm_decision)
            return final_signal
            
        except Exception as e:
            logger.error(f"LLM confirmation failed: {e}")
            # Fall back to rule-based signal
            rule_signal['llm_confirmed'] = None
            rule_signal['llm_note'] = f"LLM error: {str(e)[:50]}"
            return rule_signal
    
    async def _get_llm_confirmation(
        self,
        market_data: Dict[str, Any],
        rule_signal: Dict[str, Any],
        signal_type: str
    ) -> Dict[str, Any]:
        """
        Get LLM confirmation for a trading signal.
        """
        symbol = market_data.get('symbol', 'UNKNOWN')
        price = market_data.get('price', 0)
        
        prompt = f"""
You are a trading risk manager reviewing a signal from an automated trading system.

SIGNAL TO REVIEW:
- Symbol: {symbol}
- Current Price: ${price:,.2f}
- Signal: {signal_type}
- Confidence: {rule_signal.get('confidence', 0):.0%}
- Reasoning: {rule_signal.get('reasoning', 'N/A')}
- Market Regime: {rule_signal.get('market_regime', 'unknown')}
- Weighted Score: {rule_signal.get('weighted_score', 0)}

MARKET DATA:
- 24h Change: {market_data.get('change_24h', 0):.2f}%
- News Sentiment: {market_data.get('news_sentiment', {}).get('sentiment_label', 'N/A')}

CONTEXT (if available):
{market_data.get('memory_context', 'No history available')}

YOUR TASK:
1. Review if this signal makes sense given the market context
2. Look for any red flags the rules might have missed
3. Consider news/sentiment that rules don't account for

RESPOND IN JSON ONLY:
{{
    "decision": "CONFIRM" | "REDUCE" | "REJECT",
    "confidence_adjustment": 0.0 to 1.0 (multiplier, 1.0 = no change),
    "reason": "Brief explanation",
    "red_flags": ["list", "of", "concerns"] or [],
    "proceed": true/false
}}

IMPORTANT:
- CONFIRM = Execute as planned
- REDUCE = Execute with 50% position size
- REJECT = Do not execute
"""
        
        try:
            response = self.llm_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temp for more consistent decisions
                    max_output_tokens=500,
                )
            )
            
            response_text = response.text.strip()
            
            # Clean JSON
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            return json.loads(response_text)
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response: {response_text[:100]}")
            return {"decision": "CONFIRM", "confidence_adjustment": 1.0, "reason": "Parse error - defaulting to confirm"}
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _combine_decisions(
        self,
        rule_signal: Dict[str, Any],
        llm_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine rule-based signal with LLM decision.
        """
        decision = llm_decision.get('decision', 'CONFIRM')
        
        # Track stats
        if decision == 'REJECT':
            self.llm_rejections += 1
        else:
            self.llm_confirmations += 1
        
        # Start with rule signal
        final_signal = rule_signal.copy()
        
        if decision == 'REJECT':
            # Override to HOLD
            final_signal['signal_type'] = 'HOLD'
            final_signal['original_signal'] = rule_signal.get('signal_type')
            final_signal['confidence'] = 0.3
            final_signal['llm_confirmed'] = False
            final_signal['llm_note'] = f"REJECTED: {llm_decision.get('reason', 'No reason')}"
            
            # Add red flags to reasoning
            red_flags = llm_decision.get('red_flags', [])
            if red_flags:
                final_signal['llm_note'] += f" | Flags: {', '.join(red_flags)}"
        
        elif decision == 'REDUCE':
            # Keep signal but reduce confidence
            adjustment = llm_decision.get('confidence_adjustment', 0.5)
            final_signal['confidence'] = rule_signal.get('confidence', 0.5) * adjustment
            final_signal['llm_confirmed'] = True
            final_signal['llm_note'] = f"REDUCED: {llm_decision.get('reason', 'Proceed with caution')}"
            final_signal['position_multiplier'] = 0.5  # Signal to use 50% size
        
        else:  # CONFIRM
            adjustment = llm_decision.get('confidence_adjustment', 1.0)
            final_signal['confidence'] = min(
                rule_signal.get('confidence', 0.5) * adjustment,
                0.95
            )
            final_signal['llm_confirmed'] = True
            final_signal['llm_note'] = f"CONFIRMED: {llm_decision.get('reason', 'Proceed')}"
        
        return final_signal
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        total_decisions = self.llm_confirmations + self.llm_rejections
        rejection_rate = (self.llm_rejections / total_decisions * 100) if total_decisions > 0 else 0
        
        return {
            'llm_calls': self.llm_calls,
            'confirmations': self.llm_confirmations,
            'rejections': self.llm_rejections,
            'rejection_rate': f"{rejection_rate:.1f}%"
        }


class RulesOnlyAgent(RuleBasedAgent):
    """
    Convenience class for pure rule-based trading without LLM.
    Same as RuleBasedAgent but with a clearer name.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "rules_only_trader"


