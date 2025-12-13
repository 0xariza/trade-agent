"""
Unit tests for RiskManager.
"""
import pytest
from backend.risk.risk_manager import RiskManager


class TestRiskManager:
    """Test RiskManager functionality."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create a RiskManager instance for testing."""
        return RiskManager(
            max_position_size_pct=0.10,
            max_daily_loss_pct=0.05,
            default_risk_reward=2.0,
            default_atr_stop_multiplier=2.0,
            max_hold_hours=72,
            enable_trailing_stop=True
        )
    
    def test_calculate_position_size(self, risk_manager):
        """Test position size calculation."""
        portfolio_value = 10000.0
        entry_price = 50000.0
        stop_loss = 49000.0
        
        size = risk_manager.calculate_position_size(
            portfolio_value=portfolio_value,
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        
        # Should respect max_position_size_pct
        position_value = size * entry_price
        position_pct = position_value / portfolio_value
        
        assert position_pct <= 0.10  # max_position_size_pct
        assert size > 0
    
    def test_calculate_stop_loss(self, risk_manager):
        """Test stop loss calculation."""
        entry_price = 50000.0
        atr = 500.0
        
        stop_loss = risk_manager.calculate_stop_loss(
            entry_price=entry_price,
            atr=atr,
            side="long"
        )
        
        # Stop loss should be below entry for long
        assert stop_loss < entry_price
        # Should be approximately 2 ATR away
        expected_stop = entry_price - (atr * 2.0)
        assert abs(stop_loss - expected_stop) < 100
    
    def test_calculate_take_profit(self, risk_manager):
        """Test take profit calculation."""
        entry_price = 50000.0
        stop_loss = 49000.0
        
        take_profit = risk_manager.calculate_take_profit(
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_reward=2.0
        )
        
        # Take profit should be above entry for long
        assert take_profit > entry_price
        # Should be 2x the risk distance
        risk = entry_price - stop_loss
        expected_tp = entry_price + (risk * 2.0)
        assert abs(take_profit - expected_tp) < 100
    
    def test_check_risk_limits(self, risk_manager):
        """Test risk limit checks."""
        # Test within limits
        can_trade, reason = risk_manager.check_risk_limits(
            current_balance=10000.0,
            daily_pnl=-100.0,  # -1% daily loss
            open_positions_count=2
        )
        assert can_trade is True
        
        # Test daily loss limit exceeded
        can_trade, reason = risk_manager.check_risk_limits(
            current_balance=10000.0,
            daily_pnl=-600.0,  # -6% daily loss (exceeds 5%)
            open_positions_count=2
        )
        assert can_trade is False
        assert "daily loss" in reason.lower()

