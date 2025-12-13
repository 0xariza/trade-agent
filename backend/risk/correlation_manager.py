"""
Portfolio Correlation Manager for Alpha Arena.

Prevents overexposure to correlated assets.
Key features:
- Real-time correlation matrix calculation
- Sector/category exposure limits
- Position blocking when correlation too high
- Diversification scoring

Week 3-4 Implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import asyncio

logger = logging.getLogger(__name__)


# Asset categories for sector exposure limits
ASSET_CATEGORIES = {
    # Layer 1 Blockchains
    "L1": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "NEAR/USDT", "APT/USDT", "SUI/USDT"],
    
    # Layer 2 / Scaling
    "L2": ["MATIC/USDT", "ARB/USDT", "OP/USDT"],
    
    # DeFi
    "DEFI": ["UNI/USDT", "AAVE/USDT", "LINK/USDT", "MKR/USDT", "SNX/USDT", "CRV/USDT"],
    
    # Exchange Tokens
    "CEX": ["BNB/USDT", "OKB/USDT", "FTT/USDT", "CRO/USDT"],
    
    # Meme Coins
    "MEME": ["DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "BONK/USDT"],
    
    # AI/Compute
    "AI": ["FET/USDT", "RNDR/USDT", "TAO/USDT", "OCEAN/USDT"],
    
    # Gaming/Metaverse
    "GAMING": ["AXS/USDT", "SAND/USDT", "MANA/USDT", "IMX/USDT"],
    
    # Stablecoins (shouldn't trade these but for completeness)
    "STABLE": ["USDT/USD", "USDC/USDT", "DAI/USDT"],
}


class CorrelationManager:
    """
    Manages portfolio correlation and diversification.
    
    Critical for preventing compounded losses when correlated assets dump together.
    """
    
    def __init__(
        self,
        max_correlation_threshold: float = 0.7,
        max_sector_exposure_pct: float = 0.40,
        max_concurrent_positions: int = 6,
        correlation_lookback_days: int = 30,
        btc_correlation_weight: float = 0.3  # BTC correlation matters more
    ):
        """
        Args:
            max_correlation_threshold: Block new position if correlation > this
            max_sector_exposure_pct: Max % of portfolio in one sector
            max_concurrent_positions: Max open positions at once
            correlation_lookback_days: Days of data for correlation calc
            btc_correlation_weight: Extra weight for BTC correlation
        """
        self.max_correlation_threshold = max_correlation_threshold
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.correlation_lookback_days = correlation_lookback_days
        self.btc_correlation_weight = btc_correlation_weight
        
        # Cached correlation matrix
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._last_correlation_update: Optional[datetime] = None
        self._correlation_cache_ttl = 3600  # 1 hour
        
        # Price history cache for correlation calculation
        self._price_history: Dict[str, pd.Series] = {}
        
        logger.info(
            f"CorrelationManager initialized. Max corr: {max_correlation_threshold}, "
            f"Max sector: {max_sector_exposure_pct*100}%, Max positions: {max_concurrent_positions}"
        )
    
    def get_asset_category(self, symbol: str) -> str:
        """Get the category/sector for an asset."""
        for category, symbols in ASSET_CATEGORIES.items():
            if symbol in symbols:
                return category
        return "OTHER"
    
    def can_open_position(
        self,
        symbol: str,
        position_size_usd: float,
        current_positions: Dict[str, Dict[str, Any]],
        portfolio_value: float
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be opened based on correlation and diversification rules.
        
        Args:
            symbol: Symbol to potentially trade
            position_size_usd: Size of new position in USD
            current_positions: Dict of current open positions {symbol: position_data}
            portfolio_value: Total portfolio value
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Rule 1: Max concurrent positions
        if len(current_positions) >= self.max_concurrent_positions:
            return False, f"Max positions reached ({self.max_concurrent_positions})"
        
        # Rule 2: Already have position in this symbol
        if symbol in current_positions:
            return False, f"Already have open position in {symbol}"
        
        # Rule 3: Sector exposure limit
        new_category = self.get_asset_category(symbol)
        sector_exposure = self._calculate_sector_exposure(
            new_category, position_size_usd, current_positions, portfolio_value
        )
        
        if sector_exposure > self.max_sector_exposure_pct:
            return False, (
                f"Sector {new_category} exposure would be {sector_exposure*100:.1f}% "
                f"(limit: {self.max_sector_exposure_pct*100:.1f}%)"
                )
        
        # Rule 4: Correlation check with existing positions
        if current_positions and self._correlation_matrix is not None:
            max_corr, corr_symbol = self._get_max_correlation(symbol, current_positions)
            
            if max_corr > self.max_correlation_threshold:
                return False, (
                    f"High correlation ({max_corr:.2f}) with {corr_symbol} "
                    f"(limit: {self.max_correlation_threshold})"
                )
        
        # Rule 5: BTC exposure check for alts
        if symbol != "BTC/USDT" and "BTC/USDT" in current_positions:
            # When BTC position is open, limit alt exposure
            btc_pos = current_positions["BTC/USDT"]
            btc_size = btc_pos.get('amount', 0) * btc_pos.get('entry_price', 0)
            
            if btc_size > portfolio_value * 0.15:  # BTC is > 15% of portfolio
                # Check BTC correlation
                btc_corr = self._get_correlation(symbol, "BTC/USDT")
                if btc_corr > 0.8:
                    return False, (
                        f"High BTC correlation ({btc_corr:.2f}) while BTC position is large"
                    )
        
        return True, "All correlation checks passed"
    
    def _calculate_sector_exposure(
        self,
        category: str,
        new_position_size: float,
        current_positions: Dict[str, Dict[str, Any]],
        portfolio_value: float
    ) -> float:
        """Calculate exposure to a sector including the new position."""
        sector_exposure = new_position_size
        
        for symbol, pos in current_positions.items():
            pos_category = self.get_asset_category(symbol)
            if pos_category == category:
                pos_size = pos.get('amount', 0) * pos.get('entry_price', 0)
                sector_exposure += pos_size
        
        return sector_exposure / portfolio_value if portfolio_value > 0 else 0
    
    def _get_max_correlation(
        self,
        symbol: str,
        current_positions: Dict[str, Dict[str, Any]]
    ) -> Tuple[float, str]:
        """Get maximum correlation between symbol and any current position."""
        if self._correlation_matrix is None:
            return 0.0, ""
        
        max_corr = 0.0
        corr_symbol = ""
        
        for pos_symbol in current_positions.keys():
            corr = self._get_correlation(symbol, pos_symbol)
        
            # Weight BTC correlation higher
            if pos_symbol == "BTC/USDT":
                corr = corr * (1 + self.btc_correlation_weight)
            
            if corr > max_corr:
                max_corr = corr
                corr_symbol = pos_symbol
        
        return min(max_corr, 1.0), corr_symbol  # Cap at 1.0
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols from the matrix."""
        if self._correlation_matrix is None:
            return 0.5  # Assume moderate correlation if no data
        
        try:
            return abs(self._correlation_matrix.loc[symbol1, symbol2])
        except KeyError:
            return 0.5  # Default if not in matrix
    
    async def update_correlation_matrix(
        self,
        market_data_provider,
        symbols: List[str]
    ):
        """
        Update the correlation matrix with recent price data.
        
        Should be called periodically (e.g., every hour).
        """
        # Check cache
        if (
            self._last_correlation_update and
            (datetime.now() - self._last_correlation_update).seconds < self._correlation_cache_ttl
        ):
            return
        
        logger.info(f"Updating correlation matrix for {len(symbols)} symbols...")
        
        try:
            # PERFORMANCE: Fetch price history for all symbols in parallel
            price_data = {}
            
            # Create tasks for all symbols
            fetch_tasks = {
                symbol: market_data_provider.get_ohlcv(
                    symbol, 
                    timeframe='1h',
                    limit=self.correlation_lookback_days * 24
                )
                for symbol in symbols
            }
            
            # Fetch all in parallel
            import asyncio
            results = await asyncio.gather(*fetch_tasks.values(), return_exceptions=True)
            
            # Process results
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to get data for {symbol}: {result}")
                    continue
                if not result.empty:
                    price_data[symbol] = result['close']
            
            if len(price_data) < 2:
                logger.warning("Not enough data for correlation calculation")
                return
            
            # Create DataFrame and calculate returns
            df = pd.DataFrame(price_data)
            returns = df.pct_change().dropna()
            
            # Calculate correlation matrix
            self._correlation_matrix = returns.corr()
            self._last_correlation_update = datetime.now()
            
            logger.info(f"Correlation matrix updated. Shape: {self._correlation_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Error updating correlation matrix: {e}")
    
    def get_diversification_score(
        self,
        positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate a diversification score for the current portfolio.
        
        Score from 0-100:
        - 0-30: Poorly diversified (high risk)
        - 30-60: Moderately diversified
        - 60-100: Well diversified
        """
        if not positions:
            return {"score": 100, "rating": "N/A - No positions", "issues": []}
        
        score = 100
        issues = []
        
        # 1. Number of positions (max 20 points)
        n_positions = len(positions)
        if n_positions == 1:
            score -= 20
            issues.append("Only 1 position - no diversification")
        elif n_positions == 2:
            score -= 10
            issues.append("Only 2 positions - limited diversification")
        
        # 2. Sector concentration (max 30 points)
        sector_counts = defaultdict(int)
        for symbol in positions:
            category = self.get_asset_category(symbol)
            sector_counts[category] += 1
        
        if sector_counts:
            max_sector_pct = max(sector_counts.values()) / n_positions
            if max_sector_pct > 0.6:
                score -= 30
                issues.append(f"High sector concentration ({max_sector_pct*100:.0f}% in one sector)")
            elif max_sector_pct > 0.4:
                score -= 15
                issues.append(f"Moderate sector concentration ({max_sector_pct*100:.0f}%)")
        
        # 3. BTC correlation exposure (max 25 points)
        if self._correlation_matrix is not None:
            btc_corr_sum = 0
            for symbol in positions:
                if symbol != "BTC/USDT":
                    btc_corr = self._get_correlation(symbol, "BTC/USDT")
                    btc_corr_sum += btc_corr
            
            avg_btc_corr = btc_corr_sum / max(1, n_positions - 1)
            if avg_btc_corr > 0.8:
                score -= 25
                issues.append(f"All positions highly correlated with BTC ({avg_btc_corr:.2f})")
            elif avg_btc_corr > 0.6:
                score -= 15
                issues.append(f"High avg BTC correlation ({avg_btc_corr:.2f})")
        
        # 4. Average pairwise correlation (max 25 points)
        if self._correlation_matrix is not None and n_positions > 1:
            total_corr = 0
            pair_count = 0
            
            symbols = list(positions.keys())
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    corr = self._get_correlation(symbols[i], symbols[j])
                    total_corr += corr
                    pair_count += 1
            
            avg_corr = total_corr / max(1, pair_count)
            if avg_corr > 0.7:
                score -= 25
                issues.append(f"High avg portfolio correlation ({avg_corr:.2f})")
            elif avg_corr > 0.5:
                score -= 12
                issues.append(f"Moderate portfolio correlation ({avg_corr:.2f})")
        
        # Determine rating
        if score >= 70:
            rating = "Well Diversified ✅"
        elif score >= 50:
            rating = "Moderately Diversified"
        elif score >= 30:
            rating = "Poorly Diversified ⚠️"
        else:
            rating = "High Concentration Risk 🔴"
        
        return {
            "score": max(0, score),
            "rating": rating,
            "issues": issues,
            "sector_distribution": dict(sector_counts),
            "position_count": n_positions
        }
    
    def get_position_size_multiplier(
        self,
        symbol: str,
        current_positions: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Get a multiplier for position size based on correlation.
        
        If new position is highly correlated with existing, reduce size.
        """
        if not current_positions:
            return 1.0
        
        max_corr, _ = self._get_max_correlation(symbol, current_positions)
        
        # Scale multiplier: 1.0 at corr=0, 0.5 at corr=0.7, 0.25 at corr=0.9
        if max_corr < 0.3:
            return 1.0
        elif max_corr < 0.5:
            return 0.85
        elif max_corr < 0.7:
            return 0.7
        elif max_corr < 0.8:
            return 0.5
        else:
            return 0.25
    
    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get a summary of the correlation matrix."""
        if self._correlation_matrix is None:
            return {"status": "Not calculated", "last_update": None}
        
        matrix = self._correlation_matrix
        
        # Find highest correlations (excluding diagonal)
        np.fill_diagonal(matrix.values, 0)
        
        high_corr_pairs = []
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                corr = matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    high_corr_pairs.append({
                        "pair": f"{matrix.index[i]} / {matrix.columns[j]}",
                        "correlation": round(corr, 3)
                    })
        
        # Sort by correlation
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            "status": "Calculated",
            "last_update": self._last_correlation_update.isoformat() if self._last_correlation_update else None,
            "symbols_count": len(matrix),
            "avg_correlation": round(matrix.values[np.triu_indices_from(matrix.values, k=1)].mean(), 3),
            "high_correlation_pairs": high_corr_pairs[:10]  # Top 10
        }
    
    def print_matrix(self):
        """Print the correlation matrix (for debugging)."""
        if self._correlation_matrix is None:
            print("Correlation matrix not calculated")
            return
        
        print("\n" + "=" * 60)
        print("📊 CORRELATION MATRIX")
        print("=" * 60)
        print(self._correlation_matrix.round(2).to_string())
        print("=" * 60 + "\n")
