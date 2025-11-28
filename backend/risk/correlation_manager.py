"""
Correlation Manager - Avoid Concentrated Risk.

When BTC drops, most altcoins drop too. This manager:
1. Groups assets by correlation
2. Limits exposure to correlated assets
3. Calculates portfolio-level risk
4. Blocks trades that would over-concentrate risk
"""

import logging
from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorrelationGroup:
    """A group of correlated assets."""
    name: str
    leader: str  # Primary asset (e.g., BTC)
    members: Set[str]
    correlation_threshold: float = 0.7
    max_group_exposure_pct: float = 0.20  # Max 20% of portfolio in one group


# Pre-defined correlation groups for crypto
CRYPTO_CORRELATION_GROUPS = {
    'btc_ecosystem': CorrelationGroup(
        name='BTC Ecosystem',
        leader='BTC/USDT',
        members={'BTC/USDT', 'WBTC/USDT'},
        max_group_exposure_pct=0.15
    ),
    'eth_ecosystem': CorrelationGroup(
        name='ETH Ecosystem',
        leader='ETH/USDT',
        members={'ETH/USDT', 'STETH/USDT', 'LDO/USDT'},
        max_group_exposure_pct=0.15
    ),
    'layer1_alts': CorrelationGroup(
        name='Layer 1 Alts',
        leader='ETH/USDT',
        members={'SOL/USDT', 'AVAX/USDT', 'ADA/USDT', 'DOT/USDT', 'NEAR/USDT'},
        max_group_exposure_pct=0.20
    ),
    'layer2': CorrelationGroup(
        name='Layer 2',
        leader='ETH/USDT',
        members={'OP/USDT', 'ARB/USDT', 'MATIC/USDT', 'POL/USDT'},
        max_group_exposure_pct=0.15
    ),
    'defi_blue_chips': CorrelationGroup(
        name='DeFi Blue Chips',
        leader='ETH/USDT',
        members={'LINK/USDT', 'UNI/USDT', 'AAVE/USDT', 'MKR/USDT'},
        max_group_exposure_pct=0.15
    ),
    'meme_coins': CorrelationGroup(
        name='Meme Coins',
        leader='DOGE/USDT',
        members={'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT'},
        max_group_exposure_pct=0.10  # Lower exposure for memes
    ),
    'exchange_tokens': CorrelationGroup(
        name='Exchange Tokens',
        leader='BNB/USDT',
        members={'BNB/USDT', 'FTT/USDT', 'CRO/USDT'},
        max_group_exposure_pct=0.10
    ),
    'privacy_coins': CorrelationGroup(
        name='Privacy Coins',
        leader='XMR/USDT',
        members={'XMR/USDT', 'ZEC/USDT', 'DASH/USDT'},
        max_group_exposure_pct=0.10
    )
}


@dataclass
class ExposureReport:
    """Report on current portfolio exposure."""
    total_exposure_pct: float
    group_exposures: Dict[str, float]  # group_name -> exposure_pct
    warnings: List[str]
    blocked_symbols: Set[str]
    concentration_score: float  # 0 = well diversified, 100 = highly concentrated


class CorrelationManager:
    """
    Manages correlation-based risk limits.
    
    Features:
    - Tracks exposure by correlation group
    - Blocks trades that would over-concentrate
    - Calculates real-time portfolio concentration
    - Suggests hedging opportunities
    """
    
    def __init__(
        self,
        max_single_asset_pct: float = 0.10,       # Max 10% in any single asset
        max_correlated_group_pct: float = 0.25,   # Max 25% in correlated group
        max_total_long_pct: float = 0.80,         # Max 80% total long exposure
        max_positions: int = 8                     # Max concurrent positions
    ):
        self.max_single_asset_pct = max_single_asset_pct
        self.max_correlated_group_pct = max_correlated_group_pct
        self.max_total_long_pct = max_total_long_pct
        self.max_positions = max_positions
        
        # Use pre-defined groups
        self.groups = CRYPTO_CORRELATION_GROUPS.copy()
        
        # Track current positions
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # Track historical correlations (optional, for dynamic calculation)
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(
            f"CorrelationManager initialized. Max single: {max_single_asset_pct*100}%, "
            f"Max group: {max_correlated_group_pct*100}%, Max positions: {max_positions}"
        )
    
    def update_position(self, symbol: str, value: float, side: str = 'long'):
        """
        Update position tracking.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            value: Position value in USDT
            side: 'long' or 'short'
        """
        if value <= 0:
            if symbol in self.positions:
                del self.positions[symbol]
        else:
            self.positions[symbol] = {
                'value': value,
                'side': side,
                'updated_at': datetime.now()
            }
    
    def get_symbol_group(self, symbol: str) -> Optional[str]:
        """Find which correlation group a symbol belongs to."""
        for group_id, group in self.groups.items():
            if symbol in group.members:
                return group_id
        return None
    
    def can_open_position(
        self,
        symbol: str,
        proposed_value: float,
        portfolio_value: float
    ) -> Tuple[bool, str, float]:
        """
        Check if a new position can be opened.
        
        Args:
            symbol: Symbol to trade
            proposed_value: Value of proposed position
            portfolio_value: Total portfolio value
            
        Returns:
            Tuple of (allowed, reason, max_allowed_value)
        """
        reasons = []
        max_allowed = proposed_value
        
        # 1. Check max positions
        if len(self.positions) >= self.max_positions and symbol not in self.positions:
            return False, f"Max positions ({self.max_positions}) reached", 0
        
        # 2. Check single asset limit
        proposed_pct = proposed_value / portfolio_value
        if proposed_pct > self.max_single_asset_pct:
            max_allowed = portfolio_value * self.max_single_asset_pct
            reasons.append(f"Single asset limit: max {self.max_single_asset_pct*100}%")
        
        # 3. Check correlated group exposure
        group_id = self.get_symbol_group(symbol)
        if group_id:
            group = self.groups[group_id]
            current_group_value = self._get_group_exposure_value(group_id)
            new_group_value = current_group_value + proposed_value
            new_group_pct = new_group_value / portfolio_value
            
            group_limit = min(group.max_group_exposure_pct, self.max_correlated_group_pct)
            
            if new_group_pct > group_limit:
                available = (group_limit * portfolio_value) - current_group_value
                max_allowed = min(max_allowed, max(0, available))
                reasons.append(
                    f"{group.name} limit: {new_group_pct*100:.1f}% > {group_limit*100}%"
                )
        
        # 4. Check total long exposure
        total_long = sum(
            p['value'] for p in self.positions.values() if p['side'] == 'long'
        )
        new_total = total_long + proposed_value
        new_total_pct = new_total / portfolio_value
        
        if new_total_pct > self.max_total_long_pct:
            available = (self.max_total_long_pct * portfolio_value) - total_long
            max_allowed = min(max_allowed, max(0, available))
            reasons.append(f"Total long limit: {new_total_pct*100:.1f}% > {self.max_total_long_pct*100}%")
        
        # Decision
        if max_allowed <= 0:
            return False, " | ".join(reasons), 0
        elif max_allowed < proposed_value:
            return True, f"Reduced: {' | '.join(reasons)}", max_allowed
        else:
            return True, "OK", proposed_value
    
    def _get_group_exposure_value(self, group_id: str) -> float:
        """Get total exposure value for a correlation group."""
        if group_id not in self.groups:
            return 0
        
        group = self.groups[group_id]
        total = 0
        for symbol in group.members:
            if symbol in self.positions:
                total += self.positions[symbol]['value']
        return total
    
    def get_exposure_report(self, portfolio_value: float) -> ExposureReport:
        """
        Generate a full exposure report.
        """
        group_exposures = {}
        warnings = []
        blocked_symbols = set()
        
        # Calculate group exposures
        for group_id, group in self.groups.items():
            group_value = self._get_group_exposure_value(group_id)
            group_pct = (group_value / portfolio_value * 100) if portfolio_value > 0 else 0
            group_exposures[group.name] = group_pct
            
            if group_pct > group.max_group_exposure_pct * 100:
                warnings.append(f"{group.name}: {group_pct:.1f}% (limit: {group.max_group_exposure_pct*100}%)")
                # Block further positions in this group
                blocked_symbols.update(group.members)
        
        # Total exposure
        total_value = sum(p['value'] for p in self.positions.values())
        total_exposure_pct = (total_value / portfolio_value * 100) if portfolio_value > 0 else 0
        
        if total_exposure_pct > self.max_total_long_pct * 100:
            warnings.append(f"Total exposure: {total_exposure_pct:.1f}% (limit: {self.max_total_long_pct*100}%)")
        
        # Position count warning
        if len(self.positions) >= self.max_positions:
            warnings.append(f"Max positions: {len(self.positions)}/{self.max_positions}")
        
        # Calculate concentration score (Herfindahl-Hirschman Index)
        if self.positions and portfolio_value > 0:
            weights = [p['value'] / portfolio_value for p in self.positions.values()]
            hhi = sum(w**2 for w in weights) * 10000  # Scale to 0-10000
            # Normalize to 0-100 (single position = 100, perfectly diversified = ~0)
            concentration_score = min(100, (hhi / 100))
        else:
            concentration_score = 0
        
        return ExposureReport(
            total_exposure_pct=total_exposure_pct,
            group_exposures=group_exposures,
            warnings=warnings,
            blocked_symbols=blocked_symbols,
            concentration_score=concentration_score
        )
    
    def get_diversification_suggestions(self, portfolio_value: float) -> List[str]:
        """
        Suggest ways to improve diversification.
        """
        suggestions = []
        report = self.get_exposure_report(portfolio_value)
        
        # High concentration in one group
        for group_name, exposure in report.group_exposures.items():
            if exposure > 15:
                suggestions.append(
                    f"Consider reducing {group_name} exposure ({exposure:.1f}%)"
                )
        
        # Too few positions
        if len(self.positions) < 3 and report.total_exposure_pct > 20:
            suggestions.append(
                f"Only {len(self.positions)} positions. Consider diversifying."
            )
        
        # Single large position
        if self.positions:
            largest = max(self.positions.values(), key=lambda x: x['value'])
            largest_pct = largest['value'] / portfolio_value * 100
            if largest_pct > 10:
                suggestions.append(
                    f"Largest position is {largest_pct:.1f}% of portfolio"
                )
        
        # Under-exposed
        if report.total_exposure_pct < 30:
            suggestions.append(
                f"Low exposure ({report.total_exposure_pct:.1f}%). Opportunities may be missed."
            )
        
        return suggestions
    
    def get_hedge_opportunities(self) -> List[Dict[str, Any]]:
        """
        Suggest hedging opportunities based on current exposure.
        """
        opportunities = []
        
        # Check for high BTC correlation exposure
        btc_group_exposure = self._get_group_exposure_value('btc_ecosystem')
        alt_exposure = sum(
            self._get_group_exposure_value(g) 
            for g in ['layer1_alts', 'layer2', 'defi_blue_chips']
        )
        
        total_long = btc_group_exposure + alt_exposure
        
        if total_long > 5000:  # Significant exposure
            opportunities.append({
                'type': 'hedge',
                'suggestion': 'Consider BTC short hedge',
                'reason': f'High long exposure: ${total_long:,.0f}',
                'hedge_size': total_long * 0.3  # 30% hedge
            })
        
        return opportunities
    
    def sync_with_exchange(self, positions: Dict[str, Any], current_prices: Dict[str, float]):
        """
        Sync position tracking with exchange state.
        
        Args:
            positions: Dict of symbol -> position object
            current_prices: Dict of symbol -> current price
        """
        self.positions.clear()
        
        for symbol, pos in positions.items():
            if hasattr(pos, 'amount') and hasattr(pos, 'side'):
                price = current_prices.get(symbol, pos.entry_price if hasattr(pos, 'entry_price') else 0)
                value = pos.amount * price
                self.update_position(symbol, value, pos.side)
            elif isinstance(pos, dict):
                amount = pos.get('amount', 0)
                price = current_prices.get(symbol, pos.get('entry_price', 0))
                value = amount * price
                self.update_position(symbol, value, pos.get('side', 'long'))
    
    def print_status(self, portfolio_value: float):
        """Print current correlation/exposure status."""
        report = self.get_exposure_report(portfolio_value)
        
        print("\n" + "=" * 50)
        print("📊 CORRELATION & EXPOSURE STATUS")
        print("=" * 50)
        print(f"Total Exposure: {report.total_exposure_pct:.1f}%")
        print(f"Positions: {len(self.positions)}/{self.max_positions}")
        print(f"Concentration Score: {report.concentration_score:.0f}/100")
        
        if report.group_exposures:
            print("\nGroup Exposures:")
            for group, pct in sorted(report.group_exposures.items(), key=lambda x: -x[1]):
                if pct > 0:
                    bar = "█" * int(pct / 2)
                    print(f"  {group}: {pct:.1f}% {bar}")
        
        if report.warnings:
            print("\n⚠️ Warnings:")
            for w in report.warnings:
                print(f"  - {w}")
        
        if report.blocked_symbols:
            print(f"\n🚫 Blocked: {', '.join(list(report.blocked_symbols)[:5])}")
        
        print("=" * 50)


