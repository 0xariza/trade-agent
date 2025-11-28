"""
Production Logging System for Alpha Arena.

Features:
- Structured JSON logging for production
- Console logging for development
- File rotation (daily)
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Correlation IDs for request tracing
- Telegram alerts for critical errors
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import traceback

# Try to import Telegram notifier
try:
    from backend.notifications.telegram_notifier import TelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for production."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info[0] else None
            }
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data["data"] = record.extra_data
        
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Colored console output for development."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Format timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Format message
        formatted = f"{color}[{timestamp}] {record.levelname:8}{self.RESET} | {record.name}: {record.getMessage()}"
        
        # Add exception if present
        if record.exc_info:
            formatted += f"\n{color}{traceback.format_exception(*record.exc_info)[-1].strip()}{self.RESET}"
        
        return formatted


class TelegramHandler(logging.Handler):
    """Send critical logs to Telegram."""
    
    def __init__(self, min_level: int = logging.ERROR):
        super().__init__(level=min_level)
        self.notifier = None
        self._last_notify = {}
        self._cooldown_seconds = 60  # Don't spam same error
        
        if TELEGRAM_AVAILABLE:
            try:
                self.notifier = TelegramNotifier()
            except Exception as e:
                print(f"Warning: Could not initialize Telegram handler: {e}")
    
    def emit(self, record: logging.LogRecord):
        if not self.notifier:
            return
        
        # Check cooldown for same message
        msg_key = f"{record.name}:{record.lineno}:{record.getMessage()[:50]}"
        now = datetime.now()
        
        if msg_key in self._last_notify:
            elapsed = (now - self._last_notify[msg_key]).total_seconds()
            if elapsed < self._cooldown_seconds:
                return
        
        self._last_notify[msg_key] = now
        
        try:
            emoji = "🔴" if record.levelname == "ERROR" else "🚨"
            
            message = (
                f"{emoji} *{record.levelname}*\n\n"
                f"📍 `{record.name}`\n"
                f"💬 {record.getMessage()[:500]}\n"
                f"⏰ `{now.strftime('%Y-%m-%d %H:%M:%S')}`"
            )
            
            if record.exc_info and record.exc_info[1]:
                message += f"\n\n❌ `{str(record.exc_info[1])[:200]}`"
            
            # Use asyncio to send (non-blocking)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.notifier.send_message(message))
                else:
                    loop.run_until_complete(self.notifier.send_message(message))
            except RuntimeError:
                # No event loop, skip telegram
                pass
                
        except Exception as e:
            # Don't let telegram errors break logging
            print(f"Telegram logging error: {e}")


class ProductionLogger:
    """
    Production-ready logging configuration.
    
    Usage:
        from backend.utils.production_logger import setup_logging, get_logger
        
        setup_logging(level="INFO", log_dir="logs")
        logger = get_logger("my_module")
        
        logger.info("Trade executed", extra={"symbol": "BTC/USDT", "amount": 0.1})
    """
    
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    
    def __init__(
        self,
        level: str = "INFO",
        log_dir: str = "logs",
        json_logs: bool = False,
        enable_telegram: bool = True,
        app_name: str = "alpha-arena"
    ):
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.log_dir = Path(log_dir)
        self.json_logs = json_logs
        self.enable_telegram = enable_telegram
        self.app_name = app_name
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup root logger
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Configure the root logger."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        
        if self.json_logs:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(ConsoleFormatter())
        
        root_logger.addHandler(console_handler)
        
        # File Handler (rotating daily)
        log_file = self.log_dir / f"{self.app_name}.log"
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=30,  # Keep 30 days
            encoding='utf-8'
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)
        
        # Error file (separate file for errors)
        error_file = self.log_dir / f"{self.app_name}_errors.log"
        error_handler = logging.handlers.TimedRotatingFileHandler(
            error_file,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(error_handler)
        
        # Telegram Handler for critical errors
        if self.enable_telegram:
            telegram_handler = TelegramHandler(min_level=logging.ERROR)
            root_logger.addHandler(telegram_handler)
        
        # Reduce noise from third-party libraries
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('ccxt').setLevel(logging.WARNING)
        logging.getLogger('apscheduler').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the given name."""
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]


# Global instance
_logger_instance: Optional[ProductionLogger] = None


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    json_logs: bool = False,
    enable_telegram: bool = True
) -> ProductionLogger:
    """
    Setup production logging.
    
    Call this once at application startup.
    
    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        json_logs: Use JSON format for console (always JSON for files)
        enable_telegram: Send ERROR+ logs to Telegram
    """
    global _logger_instance
    _logger_instance = ProductionLogger(
        level=level,
        log_dir=log_dir,
        json_logs=json_logs,
        enable_telegram=enable_telegram
    )
    return _logger_instance


def get_logger(name: str) -> logging.Logger:
    """Get a logger. Call setup_logging() first."""
    if _logger_instance is None:
        setup_logging()
    return _logger_instance.get_logger(name)


# Trade-specific logger for important events
class TradeLogger:
    """
    Dedicated logger for trade events with Telegram notifications.
    
    Usage:
        trade_logger = TradeLogger()
        await trade_logger.log_buy("BTC/USDT", 0.1, 96000, "Bullish signal")
        await trade_logger.log_sell("BTC/USDT", 0.1, 97000, 100, 1.04, "Take profit")
    """
    
    def __init__(self):
        self.logger = get_logger("trades")
        self.notifier = None
        
        if TELEGRAM_AVAILABLE:
            try:
                self.notifier = TelegramNotifier()
            except Exception:
                pass
    
    async def log_buy(
        self,
        symbol: str,
        amount: float,
        price: float,
        reason: str,
        stop_loss: float = None,
        take_profit: float = None
    ):
        """Log a BUY order."""
        self.logger.info(
            f"BUY {symbol}: {amount:.6f} @ ${price:,.2f}",
            extra={
                "event": "BUY",
                "symbol": symbol,
                "amount": amount,
                "price": price,
                "reason": reason
            }
        )
        
        if self.notifier:
            try:
                sl_text = f"${stop_loss:,.2f}" if stop_loss else "N/A"
                tp_text = f"${take_profit:,.2f}" if take_profit else "N/A"
                
                message = (
                    f"🟢 *SPOT BUY*\n\n"
                    f"📊 Symbol: `{symbol}`\n"
                    f"💰 Amount: `{amount:.6f}`\n"
                    f"💵 Price: `${price:,.2f}`\n"
                    f"🛑 Stop-Loss: `{sl_text}`\n"
                    f"🎯 Take-Profit: `{tp_text}`\n\n"
                    f"📝 _{reason[:100]}_"
                )
                await self.notifier.send_message(message)
            except Exception as e:
                self.logger.warning(f"Failed to send Telegram: {e}")
    
    async def log_sell(
        self,
        symbol: str,
        amount: float,
        price: float,
        pnl: float,
        pnl_pct: float,
        reason: str
    ):
        """Log a SELL order."""
        self.logger.info(
            f"SELL {symbol}: {amount:.6f} @ ${price:,.2f} | P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)",
            extra={
                "event": "SELL",
                "symbol": symbol,
                "amount": amount,
                "price": price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "reason": reason
            }
        )
        
        if self.notifier:
            try:
                emoji = "✅" if pnl > 0 else "❌"
                
                message = (
                    f"🔴 *SPOT SELL* {emoji}\n\n"
                    f"📊 Symbol: `{symbol}`\n"
                    f"💰 Amount: `{amount:.6f}`\n"
                    f"💵 Exit Price: `${price:,.2f}`\n"
                    f"📈 P&L: `${pnl:,.2f}` (`{pnl_pct:+.2f}%`)\n\n"
                    f"📝 _{reason}_"
                )
                await self.notifier.send_message(message)
            except Exception as e:
                self.logger.warning(f"Failed to send Telegram: {e}")
    
    async def log_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float
    ):
        """Log a stop-loss trigger."""
        self.logger.warning(
            f"STOP-LOSS {symbol}: Entry ${entry_price:,.2f} -> Exit ${exit_price:,.2f} | Loss: ${pnl:,.2f}",
            extra={
                "event": "STOP_LOSS",
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl
            }
        )
        
        if self.notifier:
            try:
                message = (
                    f"🛑 *STOP-LOSS TRIGGERED*\n\n"
                    f"📊 Symbol: `{symbol}`\n"
                    f"📍 Entry: `${entry_price:,.2f}`\n"
                    f"📍 Exit: `${exit_price:,.2f}`\n"
                    f"💸 Loss: `${pnl:,.2f}` (`{pnl_pct:+.2f}%`)\n\n"
                    f"_Position closed automatically_"
                )
                await self.notifier.send_message(message)
            except Exception as e:
                self.logger.warning(f"Failed to send Telegram: {e}")
    
    async def log_take_profit(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float
    ):
        """Log a take-profit trigger."""
        self.logger.info(
            f"TAKE-PROFIT {symbol}: Entry ${entry_price:,.2f} -> Exit ${exit_price:,.2f} | Profit: ${pnl:,.2f}",
            extra={
                "event": "TAKE_PROFIT",
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl
            }
        )
        
        if self.notifier:
            try:
                message = (
                    f"🎯 *TAKE-PROFIT HIT* ✅\n\n"
                    f"📊 Symbol: `{symbol}`\n"
                    f"📍 Entry: `${entry_price:,.2f}`\n"
                    f"📍 Exit: `${exit_price:,.2f}`\n"
                    f"💰 Profit: `${pnl:,.2f}` (`{pnl_pct:+.2f}%`)\n\n"
                    f"_Target reached!_"
                )
                await self.notifier.send_message(message)
            except Exception as e:
                self.logger.warning(f"Failed to send Telegram: {e}")
    
    async def log_error(self, error: str, context: str = ""):
        """Log an error with Telegram notification."""
        self.logger.error(f"{context}: {error}" if context else error)
        
        if self.notifier:
            try:
                message = (
                    f"🚨 *ERROR*\n\n"
                    f"📍 Context: `{context or 'Unknown'}`\n"
                    f"❌ Error: `{error[:300]}`\n"
                    f"⏰ Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
                )
                await self.notifier.send_message(message)
            except Exception:
                pass


# Global trade logger instance
_trade_logger: Optional[TradeLogger] = None


def get_trade_logger() -> TradeLogger:
    """Get the trade logger instance."""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return _trade_logger

