"""
Realistic Trade Execution Model.

Simulates real-world trading conditions:
1. Dynamic slippage based on order size and volatility
2. Partial fills for large orders
3. Market impact for significant positions
4. Order book simulation
5. Latency simulation
"""

import random
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    TAKE_PROFIT = "take_profit"


class FillStatus(Enum):
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"


@dataclass
class OrderFill:
    """Represents a single fill of an order."""
    fill_price: float
    fill_amount: float
    fee: float
    timestamp: datetime
    slippage_pct: float
    

@dataclass
class ExecutionResult:
    """Result of order execution."""
    status: FillStatus
    fills: List[OrderFill]
    total_amount: float
    average_price: float
    total_cost: float
    total_fees: float
    total_slippage_pct: float
    execution_time_ms: float
    unfilled_amount: float = 0.0
    rejection_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.value,
            'fills': len(self.fills),
            'total_amount': self.total_amount,
            'average_price': self.average_price,
            'total_cost': self.total_cost,
            'total_fees': self.total_fees,
            'slippage_pct': self.total_slippage_pct,
            'execution_time_ms': self.execution_time_ms
        }


class RealisticExecutionEngine:
    """
    Simulates realistic order execution.
    
    Features:
    - Dynamic slippage based on order size relative to volume
    - Partial fills for large orders
    - Market impact modeling
    - Random execution delays
    - Stop-loss slippage (often worse than limit orders)
    """
    
    # Typical liquidity thresholds (as % of 24h volume)
    INSTANT_FILL_THRESHOLD = 0.001   # 0.1% of volume fills instantly
    PARTIAL_FILL_THRESHOLD = 0.01    # 1% of volume may have partial fills
    LARGE_ORDER_THRESHOLD = 0.05     # 5% of volume = significant market impact
    
    # Slippage parameters
    BASE_SLIPPAGE = 0.0005           # 0.05% base slippage
    VOLATILITY_MULTIPLIER = 2.0       # Double slippage in volatile markets
    STOP_LOSS_EXTRA_SLIPPAGE = 0.002  # Extra 0.2% on stop-loss fills
    
    def __init__(
        self,
        base_fee_pct: float = 0.001,      # 0.1% Binance spot
        maker_rebate_pct: float = 0.0002,  # 0.02% maker rebate
        enable_partial_fills: bool = True,
        enable_market_impact: bool = True,
        latency_range_ms: Tuple[int, int] = (50, 200)
    ):
        self.base_fee_pct = base_fee_pct
        self.maker_rebate_pct = maker_rebate_pct
        self.enable_partial_fills = enable_partial_fills
        self.enable_market_impact = enable_market_impact
        self.latency_range = latency_range_ms
        
        # Track market impact (temporary price impact from our orders)
        self.pending_impact: Dict[str, float] = {}
        
        logger.info(
            f"RealisticExecutionEngine initialized. "
            f"Fee: {base_fee_pct*100}% | Partial fills: {enable_partial_fills}"
        )
    
    def execute_market_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        amount: float,
        current_price: float,
        volume_24h: float,
        volatility_pct: float = 1.0,  # ATR as % of price
        order_type: OrderType = OrderType.MARKET
    ) -> ExecutionResult:
        """
        Execute a market order with realistic conditions.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order amount in base currency
            current_price: Current market price
            volume_24h: 24-hour trading volume
            volatility_pct: Current volatility (ATR/price * 100)
            order_type: Type of order (affects slippage)
        
        Returns:
            ExecutionResult with fill details
        """
        start_time = datetime.now()
        fills = []
        remaining_amount = amount
        order_value = amount * current_price
        
        # Calculate order size relative to volume
        if volume_24h > 0:
            order_size_pct = order_value / volume_24h
        else:
            order_size_pct = 0.01  # Assume medium size if no volume data
        
        # Simulate execution latency
        latency_ms = random.randint(self.latency_range[0], self.latency_range[1])
        
        # Check for rejection (very rare)
        if self._should_reject_order(order_size_pct):
            return ExecutionResult(
                status=FillStatus.REJECTED,
                fills=[],
                total_amount=0,
                average_price=0,
                total_cost=0,
                total_fees=0,
                total_slippage_pct=0,
                execution_time_ms=latency_ms,
                unfilled_amount=amount,
                rejection_reason="Order too large for current liquidity"
            )
        
        # Determine if partial fill needed
        if self.enable_partial_fills and order_size_pct > self.PARTIAL_FILL_THRESHOLD:
            # Large order - split into multiple fills
            num_fills = min(5, int(order_size_pct / self.INSTANT_FILL_THRESHOLD) + 1)
            fill_amounts = self._split_order(amount, num_fills)
        else:
            # Small order - single fill
            fill_amounts = [amount]
        
        # Execute each fill
        cumulative_impact = 0.0
        
        for i, fill_amount in enumerate(fill_amounts):
            # Calculate slippage for this fill
            slippage = self._calculate_slippage(
                base_price=current_price,
                fill_number=i,
                total_fills=len(fill_amounts),
                order_size_pct=order_size_pct,
                volatility_pct=volatility_pct,
                side=side,
                order_type=order_type,
                cumulative_impact=cumulative_impact
            )
            
            # Calculate fill price
            if side == 'buy':
                fill_price = current_price * (1 + slippage)
            else:
                fill_price = current_price * (1 - slippage)
            
            # Calculate fee
            fill_cost = fill_amount * fill_price
            fee = fill_cost * self.base_fee_pct
            
            # Create fill record
            fill = OrderFill(
                fill_price=fill_price,
                fill_amount=fill_amount,
                fee=fee,
                timestamp=datetime.now(),
                slippage_pct=slippage * 100
            )
            fills.append(fill)
            
            # Update cumulative impact for next fill
            if self.enable_market_impact:
                cumulative_impact += slippage * 0.3  # 30% of slippage carries over
            
            remaining_amount -= fill_amount
        
        # Calculate totals
        total_amount = sum(f.fill_amount for f in fills)
        total_cost = sum(f.fill_amount * f.fill_price for f in fills)
        total_fees = sum(f.fee for f in fills)
        average_price = total_cost / total_amount if total_amount > 0 else 0
        
        # Calculate total slippage
        if side == 'buy':
            total_slippage_pct = ((average_price - current_price) / current_price) * 100
        else:
            total_slippage_pct = ((current_price - average_price) / current_price) * 100
        
        status = FillStatus.FILLED if remaining_amount < 0.0001 else FillStatus.PARTIAL
        
        return ExecutionResult(
            status=status,
            fills=fills,
            total_amount=total_amount,
            average_price=average_price,
            total_cost=total_cost,
            total_fees=total_fees,
            total_slippage_pct=total_slippage_pct,
            execution_time_ms=latency_ms,
            unfilled_amount=remaining_amount
        )
    
    def _calculate_slippage(
        self,
        base_price: float,
        fill_number: int,
        total_fills: int,
        order_size_pct: float,
        volatility_pct: float,
        side: str,
        order_type: OrderType,
        cumulative_impact: float
    ) -> float:
        """Calculate slippage for a single fill."""
        # Base slippage
        slippage = self.BASE_SLIPPAGE
        
        # Add volatility component
        volatility_factor = min(volatility_pct / 1.0, 3.0)  # Cap at 3x
        slippage += self.BASE_SLIPPAGE * volatility_factor * self.VOLATILITY_MULTIPLIER
        
        # Add order size impact
        if order_size_pct > self.INSTANT_FILL_THRESHOLD:
            size_impact = (order_size_pct / self.INSTANT_FILL_THRESHOLD - 1) * 0.001
            slippage += min(size_impact, 0.01)  # Cap at 1%
        
        # Add cumulative market impact from previous fills
        slippage += cumulative_impact
        
        # Add progressive slippage for later fills
        if total_fills > 1:
            fill_factor = fill_number / total_fills
            slippage += slippage * fill_factor * 0.5  # 50% more slippage on last fill
        
        # Extra slippage for stop-loss orders (worse execution)
        if order_type == OrderType.STOP_MARKET:
            slippage += self.STOP_LOSS_EXTRA_SLIPPAGE
        
        # Add random component (market microstructure noise)
        noise = random.uniform(-0.0002, 0.0002)  # ±0.02%
        slippage += noise
        
        # Ensure non-negative
        return max(0, slippage)
    
    def _split_order(self, amount: float, num_fills: int) -> List[float]:
        """Split order into multiple fills with randomized sizes."""
        fills = []
        remaining = amount
        
        for i in range(num_fills - 1):
            # Random size between 10% and 40% of remaining
            fill_pct = random.uniform(0.1, 0.4)
            fill_amount = remaining * fill_pct
            fills.append(fill_amount)
            remaining -= fill_amount
        
        fills.append(remaining)  # Last fill gets the rest
        return fills
    
    def _should_reject_order(self, order_size_pct: float) -> bool:
        """Check if order should be rejected due to size."""
        if order_size_pct > 0.1:  # > 10% of daily volume
            # 10% chance of rejection for very large orders
            return random.random() < 0.1
        return False
    
    def simulate_stop_loss_fill(
        self,
        symbol: str,
        amount: float,
        stop_price: float,
        current_price: float,
        volume_24h: float,
        is_gap_down: bool = False
    ) -> ExecutionResult:
        """
        Simulate stop-loss fill (often worse than regular orders).
        
        Args:
            symbol: Trading pair
            amount: Position amount
            stop_price: Stop-loss trigger price
            current_price: Current market price (may be below stop)
            volume_24h: 24h volume
            is_gap_down: True if price gapped through stop (worse execution)
        """
        # Stop-loss typically fills at or below stop price
        if is_gap_down:
            # Gap down - fill at current price (worse than stop)
            execution_price = current_price
        else:
            # Normal stop trigger - small slippage from stop
            slippage = self.STOP_LOSS_EXTRA_SLIPPAGE
            slippage += random.uniform(0, 0.003)  # Up to 0.3% extra
            execution_price = stop_price * (1 - slippage)
        
        return self.execute_market_order(
            symbol=symbol,
            side='sell',  # Stop-loss is always a sell for longs
            amount=amount,
            current_price=execution_price,
            volume_24h=volume_24h,
            volatility_pct=2.0,  # Assume higher volatility on stop triggers
            order_type=OrderType.STOP_MARKET
        )
    
    def estimate_execution_cost(
        self,
        amount: float,
        price: float,
        volume_24h: float,
        side: str
    ) -> Dict[str, float]:
        """
        Estimate total execution cost before placing order.
        
        Returns:
            Dict with estimated costs
        """
        order_value = amount * price
        
        if volume_24h > 0:
            order_size_pct = order_value / volume_24h
        else:
            order_size_pct = 0.01
        
        # Estimate slippage
        estimated_slippage = self.BASE_SLIPPAGE
        if order_size_pct > self.INSTANT_FILL_THRESHOLD:
            estimated_slippage += (order_size_pct - self.INSTANT_FILL_THRESHOLD) * 0.1
        
        # Fees
        estimated_fees = order_value * self.base_fee_pct
        
        # Total cost impact
        slippage_cost = order_value * estimated_slippage
        
        return {
            'estimated_slippage_pct': estimated_slippage * 100,
            'estimated_slippage_cost': slippage_cost,
            'estimated_fees': estimated_fees,
            'total_execution_cost': slippage_cost + estimated_fees,
            'execution_cost_pct': (slippage_cost + estimated_fees) / order_value * 100
        }


def integrate_with_paper_exchange(paper_exchange, execution_engine: RealisticExecutionEngine):
    """
    Integrate the realistic execution engine with PaperExchange.
    
    This patches the exchange's execution methods to use realistic simulation.
    """
    original_open = paper_exchange.open_position
    
    def realistic_open_position(
        symbol, side, amount, entry_price,
        stop_loss, take_profit, **kwargs
    ):
        # Get realistic execution
        volume_24h = kwargs.pop('volume_24h', entry_price * amount * 1000)  # Estimate
        volatility_pct = kwargs.pop('volatility_pct', 1.0)
        
        result = execution_engine.execute_market_order(
            symbol=symbol,
            side='buy' if side == 'long' else 'sell',
            amount=amount,
            current_price=entry_price,
            volume_24h=volume_24h,
            volatility_pct=volatility_pct
        )
        
        if result.status == FillStatus.REJECTED:
            logger.warning(f"Order rejected: {result.rejection_reason}")
            return None
        
        # Use actual execution price
        actual_price = result.average_price
        actual_amount = result.total_amount
        
        # Call original with realistic price
        return original_open(
            symbol=symbol,
            side=side,
            amount=actual_amount,
            entry_price=actual_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            **kwargs
        )
    
    # Patch the method
    paper_exchange.open_position = realistic_open_position
    paper_exchange.execution_engine = execution_engine
    
    logger.info("Paper exchange upgraded with realistic execution engine")


