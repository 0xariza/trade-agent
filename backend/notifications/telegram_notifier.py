import os
import logging
import re
from typing import Optional
import asyncio

logger = logging.getLogger(__name__)


def escape_markdown(text: str) -> str:
    """
    Escape special characters for Telegram Markdown.
    """
    if not text:
        return ""
    
    # Characters that need escaping in Markdown mode
    escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    for char in escape_chars:
        text = text.replace(char, f'\\{char}')
    
    return text


def escape_markdown_v2(text: str) -> str:
    """
    Escape special characters for Telegram MarkdownV2.
    More strict escaping for MarkdownV2 mode.
    """
    if not text:
        return ""
    
    # All special chars for MarkdownV2
    escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!', '\\']
    
    result = ""
    for char in text:
        if char in escape_chars:
            result += f'\\{char}'
        else:
            result += char
    
    return result


class TelegramNotifier:
    """
    Sends notifications to Telegram for trading events.
    """
    
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not found. Notifications disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Telegram notifier initialized")
    
    async def send_message(self, message: str, parse_mode: str = "HTML"):
        """
        Send a message to Telegram.
        Uses HTML by default as it's more forgiving with special characters.
        Falls back to plain text if formatting fails.
        """
        if not self.enabled:
            return
        
        try:
            from telegram import Bot
            bot = Bot(token=self.bot_token)
            
            try:
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=parse_mode
                )
            except Exception as parse_error:
                # If parsing fails, try without formatting
                logger.warning(f"Parse mode failed, sending plain text: {parse_error}")
                # Strip HTML/Markdown tags for plain text
                plain_text = re.sub(r'<[^>]+>', '', message)  # Remove HTML tags
                plain_text = re.sub(r'\*([^*]+)\*', r'\1', plain_text)  # Remove *bold*
                plain_text = re.sub(r'_([^_]+)_', r'\1', plain_text)  # Remove _italic_
                
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=plain_text,
                    parse_mode=None
                )
                
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        return (
            text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
        )
    
    async def send_trade_alert(self, symbol: str, side: str, amount: float, price: float, reasoning: str):
        """
        Send trade execution alert.
        """
        emoji = "🟢" if side.lower() == "buy" else "🔴"
        
        # Escape the reasoning text to prevent parsing issues
        safe_reasoning = self._escape_html(reasoning)
        
        message = f"""{emoji} <b>TRADE EXECUTED</b>

<b>Symbol:</b> {symbol}
<b>Action:</b> {side.upper()}
<b>Amount:</b> {amount:.4f}
<b>Price:</b> ${price:,.2f}

<b>Reasoning:</b>
{safe_reasoning}"""
        
        await self.send_message(message, parse_mode="HTML")
    
    async def send_position_closed(self, symbol: str, side: str, entry_price: float, 
                                    exit_price: float, pnl: float, pnl_pct: float, 
                                    exit_reason: str):
        """
        Send position closed alert.
        """
        emoji = "💰" if pnl >= 0 else "💸"
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        
        message = f"""{emoji} <b>POSITION CLOSED</b>

<b>Symbol:</b> {symbol}
<b>Side:</b> {side.upper()}
<b>Entry:</b> ${entry_price:,.2f}
<b>Exit:</b> ${exit_price:,.2f}

{pnl_emoji} <b>P&amp;L:</b> ${pnl:,.2f} ({pnl_pct:+.2f}%)
<b>Reason:</b> {self._escape_html(exit_reason)}"""
        
        await self.send_message(message, parse_mode="HTML")
    
    async def send_circuit_breaker_alert(self, loss_pct: float, current_balance: float, start_balance: float):
        """
        Send circuit breaker trigger alert.
        """
        message = f"""⚠️ <b>CIRCUIT BREAKER TRIGGERED</b>

<b>Daily Loss:</b> {loss_pct:.2f}%
<b>Current Balance:</b> ${current_balance:,.2f}
<b>Start Balance:</b> ${start_balance:,.2f}

🛑 Trading halted for today."""
        
        await self.send_message(message, parse_mode="HTML")
    
    async def send_drawdown_alert(self, drawdown_pct: float, mode: str, action: str):
        """
        Send drawdown warning alert.
        """
        emoji = "⚠️" if drawdown_pct < 10 else "🚨"
        
        message = f"""{emoji} <b>DRAWDOWN ALERT</b>

<b>Current Drawdown:</b> {drawdown_pct:.2f}%
<b>Trading Mode:</b> {mode.upper()}
<b>Action:</b> {self._escape_html(action)}"""
        
        await self.send_message(message, parse_mode="HTML")
    
    async def send_daily_summary(self, pnl: float, pnl_pct: float, trades_count: int, 
                                  start_balance: float, end_balance: float,
                                  wins: int = 0, losses: int = 0):
        """
        Send daily performance summary.
        """
        emoji = "📈" if pnl >= 0 else "📉"
        win_rate = (wins / trades_count * 100) if trades_count > 0 else 0
        
        message = f"""{emoji} <b>DAILY SUMMARY</b>

<b>P&amp;L:</b> ${pnl:,.2f} ({pnl_pct:+.2f}%)
<b>Trades:</b> {trades_count} (W: {wins} / L: {losses})
<b>Win Rate:</b> {win_rate:.1f}%
<b>Start:</b> ${start_balance:,.2f}
<b>End:</b> ${end_balance:,.2f}"""
        
        await self.send_message(message, parse_mode="HTML")
    
    async def send_btc_warning(self, btc_trend: str, btc_change: float):
        """
        Send BTC trend warning.
        """
        message = f"""⚠️ <b>BTC TREND WARNING</b>

<b>BTC Trend:</b> {btc_trend.upper()}
<b>24h Change:</b> {btc_change:+.2f}%

Alt coin buys are being blocked until BTC stabilizes."""
        
        await self.send_message(message, parse_mode="HTML")
    
    async def send_error_alert(self, error_msg: str):
        """
        Send error alert.
        """
        safe_error = self._escape_html(str(error_msg)[:500])  # Limit length
        
        message = f"""❌ <b>ERROR</b>

{safe_error}"""
        
        await self.send_message(message, parse_mode="HTML")
    
    async def send_startup_message(self, symbols: list = None, balance: float = None):
        """
        Send bot startup notification.
        """
        symbols_str = ", ".join(symbols) if symbols else "Multiple pairs"
        balance_str = f"${balance:,.2f}" if balance else "N/A"
        
        message = f"""🤖 <b>ALPHA ARENA BOT STARTED</b>

<b>Monitoring:</b> {symbols_str}
<b>Balance:</b> {balance_str}

<b>Features Active:</b>
• BTC Trend Filter
• Stop-Loss/Take-Profit
• Drawdown Management
• Self-Reflection

You will receive alerts for:
• Trade executions
• Position closures
• Circuit breaker triggers
• Daily summaries

Good luck! 🚀"""
        
        await self.send_message(message, parse_mode="HTML")
    
    async def send_reflection_insight(self, insight: str, action: str):
        """
        Send a self-reflection insight.
        """
        message = f"""🧠 <b>AGENT LEARNED</b>

<b>Insight:</b>
{self._escape_html(insight)}

<b>Action Taken:</b>
{self._escape_html(action)}"""
        
        await self.send_message(message, parse_mode="HTML")
