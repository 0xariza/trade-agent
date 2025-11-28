"""
Trading Dashboard with Kill Switch.

Features:
- Real-time P&L tracking
- Position monitoring
- Risk metrics display
- Emergency kill switch
- Trade log
- Performance charts (text-based for terminal)
"""

import os
import json
import asyncio
import signal
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class TradingStatus(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class DashboardMetrics:
    """Current trading metrics."""
    # Portfolio
    current_equity: float = 0.0
    starting_equity: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    
    # Positions
    open_positions: int = 0
    position_exposure_pct: float = 0.0
    largest_position_pct: float = 0.0
    
    # Risk
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    trading_mode: str = "normal"
    
    # Performance
    trades_today: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Status
    status: TradingStatus = TradingStatus.RUNNING
    last_signal: str = ""
    last_signal_time: datetime = field(default_factory=datetime.now)
    uptime_hours: float = 0.0


class KillSwitch:
    """
    Emergency kill switch for trading.
    
    Triggered by:
    - Manual activation
    - Max drawdown exceeded
    - Rapid losses
    - API errors
    - System errors
    """
    
    def __init__(
        self,
        max_drawdown_pct: float = 15.0,
        rapid_loss_threshold: float = 5.0,  # 5% in 1 hour
        max_consecutive_losses: int = 5,
        on_kill: Optional[Callable] = None
    ):
        self.max_drawdown_pct = max_drawdown_pct
        self.rapid_loss_threshold = rapid_loss_threshold
        self.max_consecutive_losses = max_consecutive_losses
        self.on_kill = on_kill
        
        self.is_active = False
        self.kill_reason = ""
        self.kill_time: Optional[datetime] = None
        
        # Tracking
        self.equity_history: List[Dict[str, Any]] = []
        self.consecutive_losses = 0
        
        logger.info(f"KillSwitch initialized. Max DD: {max_drawdown_pct}%")
    
    def record_equity(self, equity: float):
        """Record equity for monitoring."""
        self.equity_history.append({
            'equity': equity,
            'timestamp': datetime.now()
        })
        
        # Keep last hour only
        cutoff = datetime.now() - timedelta(hours=1)
        self.equity_history = [
            e for e in self.equity_history
            if e['timestamp'] > cutoff
        ]
    
    def record_trade(self, pnl: float):
        """Record trade result."""
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Check consecutive loss trigger
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.trigger(f"Consecutive losses: {self.consecutive_losses}")
    
    def check_drawdown(self, current_equity: float, peak_equity: float) -> bool:
        """Check if drawdown trigger should fire."""
        if peak_equity <= 0:
            return False
        
        drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100
        
        if drawdown_pct >= self.max_drawdown_pct:
            self.trigger(f"Max drawdown exceeded: {drawdown_pct:.1f}%")
            return True
        
        return False
    
    def check_rapid_loss(self) -> bool:
        """Check for rapid loss (5% in 1 hour)."""
        if len(self.equity_history) < 2:
            return False
        
        oldest = self.equity_history[0]
        newest = self.equity_history[-1]
        
        if oldest['equity'] <= 0:
            return False
        
        loss_pct = ((oldest['equity'] - newest['equity']) / oldest['equity']) * 100
        
        if loss_pct >= self.rapid_loss_threshold:
            self.trigger(f"Rapid loss: {loss_pct:.1f}% in last hour")
            return True
        
        return False
    
    def trigger(self, reason: str):
        """Activate kill switch."""
        if self.is_active:
            return
        
        self.is_active = True
        self.kill_reason = reason
        self.kill_time = datetime.now()
        
        logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
        print("\n" + "=" * 60)
        print("🚨 EMERGENCY STOP - KILL SWITCH ACTIVATED")
        print("=" * 60)
        print(f"Reason: {reason}")
        print(f"Time: {self.kill_time}")
        print("All trading has been halted.")
        print("=" * 60 + "\n")
        
        if self.on_kill:
            self.on_kill(reason)
    
    def reset(self, confirm: bool = False):
        """Reset kill switch (requires confirmation)."""
        if not confirm:
            logger.warning("Kill switch reset requires confirmation=True")
            return False
        
        self.is_active = False
        self.kill_reason = ""
        self.kill_time = None
        self.consecutive_losses = 0
        
        logger.info("Kill switch has been reset")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get kill switch status."""
        return {
            'is_active': self.is_active,
            'reason': self.kill_reason,
            'kill_time': self.kill_time.isoformat() if self.kill_time else None,
            'consecutive_losses': self.consecutive_losses
        }


class TradingDashboard:
    """
    Real-time trading dashboard for terminal display.
    """
    
    def __init__(
        self,
        scheduler=None,
        exchange=None,
        risk_manager=None,
        refresh_interval: int = 30
    ):
        self.scheduler = scheduler
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.refresh_interval = refresh_interval
        
        self.metrics = DashboardMetrics()
        self.start_time = datetime.now()
        self.trade_log: List[Dict[str, Any]] = []
        
        # Kill switch
        self.kill_switch = KillSwitch(
            max_drawdown_pct=15.0,
            rapid_loss_threshold=5.0,
            max_consecutive_losses=5,
            on_kill=self._on_kill_switch
        )
        
        # Status
        self.status = TradingStatus.RUNNING
        self._running = False
        
        logger.info("Trading Dashboard initialized")
    
    def _on_kill_switch(self, reason: str):
        """Handle kill switch activation."""
        self.status = TradingStatus.EMERGENCY_STOP
        
        # Stop scheduler if available
        if self.scheduler and hasattr(self.scheduler, 'stop'):
            self.scheduler.stop()
        
        # Close all positions if possible
        if self.exchange:
            self._emergency_close_all()
    
    def _emergency_close_all(self):
        """Emergency close all positions."""
        if not self.exchange:
            return
        
        positions = self.exchange.get_all_positions()
        for symbol, pos in positions.items():
            try:
                # Get current price (estimate)
                price = pos.entry_price  # Fallback
                self.exchange.close_position(symbol, price, "emergency_stop")
                logger.info(f"Emergency closed: {symbol}")
            except Exception as e:
                logger.error(f"Failed to emergency close {symbol}: {e}")
    
    def update_metrics(self):
        """Update dashboard metrics from connected components."""
        if self.exchange:
            usdt = self.exchange.get_balance("USDT")
            positions = self.exchange.get_all_positions()
            
            self.metrics.current_equity = usdt
            self.metrics.open_positions = len(positions)
            
            # Calculate exposure
            if usdt > 0:
                total_position_value = sum(
                    p.amount * p.entry_price for p in positions.values()
                )
                self.metrics.position_exposure_pct = (total_position_value / usdt) * 100
                
                if positions:
                    largest = max(positions.values(), key=lambda p: p.amount * p.entry_price)
                    self.metrics.largest_position_pct = (
                        (largest.amount * largest.entry_price) / usdt * 100
                    )
        
        if self.risk_manager and hasattr(self.risk_manager, 'drawdown_manager'):
            dm = self.risk_manager.drawdown_manager
            self.metrics.current_drawdown_pct = dm.current_drawdown_pct if hasattr(dm, 'current_drawdown_pct') else 0
            self.metrics.max_drawdown_pct = dm.max_drawdown_seen * 100 if hasattr(dm, 'max_drawdown_seen') else 0
            self.metrics.trading_mode = dm.trading_mode.value if hasattr(dm, 'trading_mode') else "unknown"
        
        # Uptime
        self.metrics.uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        self.metrics.status = self.status
    
    def record_trade(self, trade: Dict[str, Any]):
        """Record a trade for display."""
        self.trade_log.append(trade)
        if len(self.trade_log) > 100:
            self.trade_log = self.trade_log[-100:]
        
        # Update kill switch
        pnl = trade.get('pnl', 0)
        self.kill_switch.record_trade(pnl)
    
    def record_signal(self, signal: str, symbol: str):
        """Record latest signal."""
        self.metrics.last_signal = f"{symbol}: {signal}"
        self.metrics.last_signal_time = datetime.now()
    
    def print_dashboard(self):
        """Print current dashboard to terminal."""
        self.update_metrics()
        m = self.metrics
        
        # Clear screen (optional)
        # os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" + "=" * 70)
        print("📊 ALPHA ARENA TRADING DASHBOARD")
        print("=" * 70)
        
        # Status bar
        status_emoji = {
            TradingStatus.RUNNING: "🟢",
            TradingStatus.PAUSED: "🟡",
            TradingStatus.STOPPED: "🔴",
            TradingStatus.EMERGENCY_STOP: "🚨"
        }
        print(f"Status: {status_emoji.get(m.status, '❓')} {m.status.value.upper()}")
        print(f"Uptime: {m.uptime_hours:.1f} hours")
        
        # Kill switch status
        ks = self.kill_switch.get_status()
        if ks['is_active']:
            print(f"⚠️  KILL SWITCH ACTIVE: {ks['reason']}")
        
        print("\n" + "-" * 70)
        print("💰 PORTFOLIO")
        print("-" * 70)
        print(f"  Current Equity:    ${m.current_equity:,.2f}")
        print(f"  Daily P&L:         ${m.daily_pnl:,.2f} ({m.daily_pnl_pct:+.2f}%)")
        print(f"  Total P&L:         ${m.total_pnl:,.2f} ({m.total_pnl_pct:+.2f}%)")
        
        print("\n" + "-" * 70)
        print("📈 POSITIONS")
        print("-" * 70)
        print(f"  Open Positions:    {m.open_positions}")
        print(f"  Total Exposure:    {m.position_exposure_pct:.1f}%")
        print(f"  Largest Position:  {m.largest_position_pct:.1f}%")
        
        print("\n" + "-" * 70)
        print("⚠️  RISK")
        print("-" * 70)
        print(f"  Current Drawdown:  {m.current_drawdown_pct:.1f}%")
        print(f"  Max Drawdown:      {m.max_drawdown_pct:.1f}%")
        print(f"  Trading Mode:      {m.trading_mode.upper()}")
        
        print("\n" + "-" * 70)
        print("📝 RECENT ACTIVITY")
        print("-" * 70)
        print(f"  Last Signal:       {m.last_signal}")
        
        # Recent trades
        if self.trade_log:
            print("\n  Last 3 Trades:")
            for trade in self.trade_log[-3:]:
                pnl = trade.get('pnl', 0)
                emoji = "✅" if pnl > 0 else "❌"
                print(f"    {emoji} {trade.get('symbol', '?')}: ${pnl:+,.2f}")
        
        print("\n" + "=" * 70)
        print("Commands: [P]ause | [R]esume | [K]ill Switch | [Q]uit")
        print("=" * 70)
    
    def pause_trading(self):
        """Pause trading."""
        self.status = TradingStatus.PAUSED
        if self.scheduler and hasattr(self.scheduler, 'pause'):
            self.scheduler.pause()
        logger.info("Trading paused")
    
    def resume_trading(self):
        """Resume trading."""
        if self.kill_switch.is_active:
            logger.warning("Cannot resume - kill switch is active")
            return False
        
        self.status = TradingStatus.RUNNING
        if self.scheduler and hasattr(self.scheduler, 'resume'):
            self.scheduler.resume()
        logger.info("Trading resumed")
        return True
    
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """Manually activate kill switch."""
        self.kill_switch.trigger(reason)
    
    async def run_display_loop(self):
        """Run continuous dashboard display."""
        self._running = True
        
        while self._running:
            self.print_dashboard()
            await asyncio.sleep(self.refresh_interval)
    
    def stop(self):
        """Stop dashboard."""
        self._running = False
        self.status = TradingStatus.STOPPED


class AlertManager:
    """
    Manages alerts for significant events.
    """
    
    def __init__(self, telegram_notifier=None):
        self.telegram = telegram_notifier
        self.alert_history: List[Dict[str, Any]] = []
        
        # Alert thresholds
        self.large_loss_threshold_pct = 2.0
        self.large_win_threshold_pct = 3.0
        self.drawdown_warning_pct = 8.0
    
    async def check_and_alert(self, metrics: DashboardMetrics):
        """Check metrics and send alerts if needed."""
        alerts = []
        
        # Drawdown warning
        if metrics.current_drawdown_pct > self.drawdown_warning_pct:
            alerts.append({
                'type': 'warning',
                'message': f"⚠️ Drawdown at {metrics.current_drawdown_pct:.1f}%",
                'severity': 'high'
            })
        
        # Mode changes
        if metrics.trading_mode != 'normal':
            alerts.append({
                'type': 'info',
                'message': f"Trading mode: {metrics.trading_mode.upper()}",
                'severity': 'medium'
            })
        
        for alert in alerts:
            await self._send_alert(alert)
    
    async def alert_trade(self, trade: Dict[str, Any]):
        """Send alert for significant trade."""
        pnl_pct = trade.get('pnl_pct', 0)
        
        if pnl_pct < -self.large_loss_threshold_pct:
            await self._send_alert({
                'type': 'loss',
                'message': f"❌ Large loss: {trade.get('symbol')} {pnl_pct:.1f}%",
                'severity': 'high'
            })
        elif pnl_pct > self.large_win_threshold_pct:
            await self._send_alert({
                'type': 'win',
                'message': f"✅ Large win: {trade.get('symbol')} +{pnl_pct:.1f}%",
                'severity': 'low'
            })
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """Send an alert."""
        self.alert_history.append({
            **alert,
            'timestamp': datetime.now()
        })
        
        logger.info(f"Alert: {alert['message']}")
        
        if self.telegram and alert.get('severity') in ['high', 'critical']:
            try:
                await self.telegram.send_message(alert['message'])
            except Exception as e:
                logger.error(f"Failed to send Telegram alert: {e}")


