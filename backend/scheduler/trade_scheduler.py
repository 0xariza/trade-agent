import asyncio
import json
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime
from typing import Dict, Any, List
from backend.data.market_data import MarketDataProvider
from backend.data.news_sentiment import NewsSentimentProvider
from backend.agents import OpenRouterAgent
from backend.agents.memory import AgentMemory
from backend.exchanges.paper_exchange import PaperExchange, Position
from backend.risk.risk_manager import RiskManager, DrawdownManager
from backend.agents.reflection import TradeReflector
from backend.notifications.telegram_notifier import TelegramNotifier
from backend.database.state_manager import state_manager
from backend.trading.spot_signal_handler import SpotSignalHandler, SpotAction
from backend.utils.production_logger import get_logger, get_trade_logger

logger = get_logger(__name__)


class TradingScheduler:
    """
    Orchestrates the trading loop with proper position management,
    stop-loss checking, take-profit execution, agent memory, and state persistence.
    """
    
    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        news_provider: NewsSentimentProvider,
        agent,  # Accept any agent type
        paper_exchange: PaperExchange,
        risk_manager: RiskManager,
        # Split intervals for different tasks (best practice)
        analysis_interval_minutes: int = 15,   # LLM analysis every 15 min
        stop_check_seconds: int = 30,          # SL/TP check every 30 sec
        state_save_minutes: int = 5,           # Save state every 5 min
        symbols: list = None,
        # Legacy support
        interval_minutes: int = None  # Deprecated, use analysis_interval_minutes
    ):
        self.market_data_provider = market_data_provider
        self.news_provider = news_provider
        self.agent = agent
        self.paper_exchange = paper_exchange
        self.risk_manager = risk_manager
        self.telegram_notifier = TelegramNotifier()
        self.scheduler = AsyncIOScheduler()
        
        # Use new interval settings (with backwards compatibility)
        if interval_minutes is not None:
            print("⚠️ 'interval_minutes' is deprecated. Use 'analysis_interval_minutes' instead.")
            self.analysis_interval_minutes = interval_minutes
        else:
            self.analysis_interval_minutes = analysis_interval_minutes
        
        self.stop_check_seconds = stop_check_seconds
        self.state_save_minutes = state_save_minutes
        
        self.is_running = False
        self.is_stop_checking = False  # Separate lock for stop checks
        self.symbols = symbols if symbols else ["BTC/USDT"]
        self.cycle_count = 0
        self.stop_check_count = 0

        # Track daily balance for circuit breaker
        self.daily_start_balance = self.paper_exchange.get_balance("USDT") 
        self.last_reset_date = datetime.now().date()
        self.daily_trades_count = 0
        
        # Cache for ATR values (for trailing stops)
        self.atr_cache: Dict[str, float] = {}
        
        # Initialize agent memory
        self.memory = AgentMemory(
            max_trade_memory=100,
            max_decision_memory=50,
            performance_lookback_days=30
        )
        
        # Attach memory to agent if it supports it
        if hasattr(self.agent, 'set_memory'):
            self.agent.set_memory(self.memory)
            print(f"🧠 Agent Memory attached to {getattr(self.agent, 'name', 'agent')}")
        
        # State persistence
        self.state_loaded = False
        
        # Initialize drawdown manager
        self.drawdown_manager = DrawdownManager(
            initial_equity=self.paper_exchange.get_balance("USDT"),
            max_drawdown_pct=0.15,           # 15% max drawdown
            cautious_drawdown_pct=0.05,       # 5% -> cautious
            defensive_drawdown_pct=0.08,      # 8% -> defensive
            recovery_drawdown_pct=0.12,       # 12% -> recovery
            weekly_loss_limit_pct=0.08,       # 8% weekly limit
            monthly_loss_limit_pct=0.15,      # 15% monthly limit
            max_consecutive_losses=5,         # 5 losses -> pause
            recovery_win_streak=3             # 3 wins to recover
        )
        print(f"📉 Drawdown Manager initialized. Max DD: 15%")
        
        # Initialize trade reflector for self-learning
        self.reflector = TradeReflector(
            min_trades_for_pattern=3,
            lookback_trades=50,
            reflection_after_n_losses=3
        )
        print(f"🔍 Trade Reflector initialized. Self-learning enabled.")
        
        # Initialize spot signal handler
        self.spot_handler = SpotSignalHandler(
            min_adx_for_buy=12.0,
            max_rsi_for_buy=72.0,
            require_btc_safety=True,
            min_rsi_for_sell=25.0
        )
        print(f"💱 Spot Trading Handler initialized. BUY + SELL enabled.")
        
        # Initialize trade logger for production logging + Telegram
        self.trade_logger = get_trade_logger()
        logger.info("Trade logger initialized with Telegram notifications")

    def start(self):
        """
        Start the trading scheduler with split intervals.
        
        Schedules:
        1. Stop-loss/TP check: Every 30 seconds (fast, low cost)
        2. Full analysis: Every 15 minutes (matches 15m timeframe)
        3. State save: Every 5 minutes (crash safety)
        """
        # Job 1: Fast stop-loss/TP check (every 30 seconds)
        self.scheduler.add_job(
            self._fast_stop_check,
            'interval',
            seconds=self.stop_check_seconds,
            id='stop_check',
            name='Stop-Loss/TP Check'
        )
        
        # Job 2: Full analysis cycle (every 15 minutes)
        self.scheduler.add_job(
            self.run_trading_cycle,
            'interval',
            minutes=self.analysis_interval_minutes,
            id='analysis',
            name='Full Market Analysis'
        )
        
        # Job 3: State persistence (every 5 minutes)
        self.scheduler.add_job(
            self.save_state,
            'interval',
            minutes=self.state_save_minutes,
            id='state_save',
            name='State Persistence'
        )
        
        self.scheduler.start()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  TRADING SCHEDULER STARTED                                       ║
╠══════════════════════════════════════════════════════════════════╣
║  📊 Full Analysis:    Every {self.analysis_interval_minutes:2d} minutes                           ║
║  🛑 Stop-Loss Check:  Every {self.stop_check_seconds:2d} seconds                           ║
║  💾 State Save:       Every {self.state_save_minutes:2d} minutes                            ║
║  📈 Symbols:          {len(self.symbols):2d} pairs                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")
        
        # Send startup notification
        asyncio.create_task(
            self.telegram_notifier.send_startup_message(
                symbols=self.symbols,
                balance=self.paper_exchange.get_balance("USDT")
            )
        )
    
    async def restore_state(self) -> bool:
        """Restore state from database on startup."""
        try:
            print("📂 Loading saved state from database...")
            
            # Load full state
            state = await state_manager.load_full_state()
            
            if not state.get('balances') and not state.get('positions'):
                print("📝 No saved state found. Starting fresh.")
                return False
            
            # Restore balances and positions to exchange
            self.paper_exchange.restore_state({
                'balances': state.get('balances', {}),
                'positions': state.get('positions', {})
            })
            
            # Restore daily stats
            daily_stats = state.get('daily_stats', {})
            if daily_stats:
                saved_date = daily_stats.get('reset_date')
                if saved_date == datetime.now().date().isoformat():
                    self.daily_start_balance = daily_stats.get('start_balance', self.paper_exchange.get_balance("USDT"))
                    self.daily_trades_count = daily_stats.get('trades_count', 0)
                    print(f"  📊 Restored daily stats: Start=${self.daily_start_balance:,.2f}, Trades={self.daily_trades_count}")
            
            # Restore memory from trade history
            trade_history = state.get('trade_history', [])
            for trade in reversed(trade_history):  # Oldest first
                self.memory.add_trade_from_dict(trade)
            
            # Restore performance stats
            if state.get('performance'):
                self.memory.performance = state['performance']
            
            # Restore lessons
            memory_data = state.get('memory', {})
            if memory_data.get('lessons'):
                self.memory.lessons = memory_data['lessons']
            
            # Restore drawdown manager state
            if memory_data.get('drawdown'):
                self.drawdown_manager = DrawdownManager.from_dict(memory_data['drawdown'])
                print(f"  📉 Drawdown state restored: Mode={self.drawdown_manager.trading_mode.value}")
            
            # Restore reflector state
            if memory_data.get('reflector'):
                self.reflector = TradeReflector.from_dict(memory_data['reflector'])
                insights_count = len(self.reflector.insights)
                print(f"  🔍 Reflector restored: {insights_count} insights")
            
            # Summary
            positions = self.paper_exchange.get_all_positions()
            usdt = self.paper_exchange.get_balance("USDT")
            
            print(f"✅ State restored successfully!")
            print(f"  💰 USDT Balance: ${usdt:,.2f}")
            print(f"  📈 Open Positions: {len(positions)}")
            print(f"  📊 Trade History: {len(trade_history)} trades")
            print(f"  🧠 Performance Data: {len(self.memory.performance)} symbols")
            
            if positions:
                for symbol, pos in positions.items():
                    print(f"     - {symbol}: {pos.side.upper()} {pos.amount:.4f} @ ${pos.entry_price:,.2f}")
            
            self.state_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore state: {e}")
            print(f"⚠️ Failed to restore state: {e}")
            return False
    
    async def save_state(self) -> bool:
        """Save current state to database."""
        try:
            # Get exchange state
            exchange_state = self.paper_exchange.get_state()
            
            # Prepare memory data (including drawdown state and reflector)
            memory_data = {
                'lessons': self.memory.lessons,
                'performance': self.memory.performance,
                'drawdown': self.drawdown_manager.to_dict(),
                'reflector': self.reflector.to_dict()
            }
            
            # Daily stats
            daily_stats = {
                'start_balance': self.daily_start_balance,
                'trades_count': self.daily_trades_count,
                'reset_date': self.last_reset_date.isoformat()
            }
            
            # Save everything
            success = await state_manager.save_full_state(
                balances=exchange_state['balances'],
                positions=exchange_state['positions'],
                memory_data=memory_data,
                daily_stats=daily_stats
            )
            
            if success:
                logger.debug("State saved successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    async def _fast_stop_check(self):
        """
        Fast stop-loss/take-profit check (runs every 30 seconds).
        
        This is a lightweight check that only:
        1. Fetches current prices
        2. Checks if any stops are hit
        3. Closes positions if needed
        
        No LLM calls, minimal API usage.
        """
        if self.is_stop_checking:
            return
        
        self.is_stop_checking = True
        try:
            positions = self.paper_exchange.get_all_positions()
            
            if not positions:
                return
            
            self.stop_check_count += 1
            
            # Only log every 4th check (every 2 minutes) to reduce noise
            verbose = self.stop_check_count % 4 == 0
            
            # Get current prices for all symbols with positions
            current_prices = {}
            for symbol in positions.keys():
                try:
                    ohlcv = await self.market_data_provider.get_ohlcv(symbol, limit=1)
                    if not ohlcv.empty:
                        current_prices[symbol] = ohlcv.iloc[-1]['close']
                except Exception as e:
                    logger.warning(f"Failed to get price for {symbol}: {e}")
            
            if not current_prices:
                return
            
            # Update trailing stops with current ATR (if cached)
            for symbol, pos in positions.items():
                if symbol in current_prices and symbol in self.atr_cache:
                    self.paper_exchange.update_trailing_stop(
                        symbol, 
                        current_prices[symbol], 
                        self.atr_cache[symbol]
                    )
            
            # Check if any stops are hit
            closed = self.paper_exchange.check_stops(current_prices, self.atr_cache)
            
            # Process closed positions
            for closed_pos in closed:
                emoji = "🛑" if closed_pos['exit_reason'] == 'stop_loss' else "🎯" if closed_pos['exit_reason'] == 'take_profit' else "⏰"
                reason = closed_pos.get('exit_reason', 'unknown').replace('_', ' ').title()
                
                print(f"\n{emoji} POSITION CLOSED: {closed_pos['symbol']}")
                print(f"   Exit: ${closed_pos['exit_price']:,.2f} | P&L: ${closed_pos['pnl']:,.2f} ({closed_pos['pnl_pct']:+.2f}%)")
                print(f"   Reason: {reason}")
                
                # Record to memory (use add_trade_from_dict for dict input)
                self.memory.add_trade_from_dict(closed_pos)
                
                # Record outcome for drawdown manager
                self.drawdown_manager.record_trade(
                    pnl=closed_pos['pnl'],
                    pnl_pct=closed_pos['pnl_pct']
                )
                
                # Analyze trade with reflector
                self.reflector.analyze_trade({
                    'symbol': closed_pos['symbol'],
                    'side': closed_pos['side'],
                    'entry_price': closed_pos['entry_price'],
                    'exit_price': closed_pos['exit_price'],
                    'pnl': closed_pos['pnl'],
                    'pnl_pct': closed_pos['pnl_pct'],
                    'exit_reason': closed_pos['exit_reason'],
                    'entry_reasoning': closed_pos.get('entry_reasoning', ''),
                    'market_regime': closed_pos.get('market_regime', 'unknown'),
                    'hold_hours': closed_pos.get('hold_hours', 0)
                })
                
                # Send Telegram notification
                await self.telegram_notifier.send_position_closed(
                    symbol=closed_pos['symbol'],
                    side=closed_pos['side'],
                    entry_price=closed_pos['entry_price'],
                    exit_price=closed_pos['exit_price'],
                    pnl=closed_pos['pnl'],
                    pnl_pct=closed_pos['pnl_pct'],
                    exit_reason=reason
                )
                
                self.daily_trades_count += 1
            
            # Log current position status (every 2 minutes)
            if verbose and positions:
                remaining = self.paper_exchange.get_all_positions()
                if remaining:
                    print(f"\n📊 Position Check ({self.stop_check_count}):")
                    for symbol, pos in remaining.items():
                        if symbol in current_prices:
                            price = current_prices[symbol]
                            pnl_pct = ((price - pos.entry_price) / pos.entry_price) * 100
                            if pos.side == 'short':
                                pnl_pct = -pnl_pct
                            sl_dist = ((pos.stop_loss - price) / price) * 100 if pos.stop_loss else 0
                            tp_dist = ((pos.take_profit - price) / price) * 100 if pos.take_profit else 0
                            print(f"   {symbol}: {pnl_pct:+.2f}% | SL: {sl_dist:+.1f}% | TP: {tp_dist:+.1f}%")
                            
        except Exception as e:
            logger.error(f"Error in fast stop check: {e}")
        finally:
            self.is_stop_checking = False

    async def run_trading_cycle(self):
        """Execute one trading cycle for all symbols."""
        if self.is_running:
            return
        
        self.is_running = True
        try:
            # 0. Daily Reset Logic
            await self._handle_daily_reset()

            # 1. Check Circuit Breaker (Global)
            current_portfolio_value = await self._calculate_portfolio_value()
            
            # Update drawdown manager
            dd_state = self.drawdown_manager.update_equity(current_portfolio_value)
            
            # Check if trading is allowed
            can_trade, reason = self.drawdown_manager.can_trade()
            if not can_trade:
                print(f"⚠️ TRADING HALTED: {reason}")
                await self.telegram_notifier.send_message(
                    f"🛑 *TRADING HALTED*\n\n{reason}\n\n"
                    f"Current DD: {dd_state.current_drawdown_pct:.1f}%\n"
                    f"Peak: ${dd_state.peak_equity:,.2f}\n"
                    f"Current: ${dd_state.current_equity:,.2f}"
                )
                return
            
            # Log trading mode if not normal
            if dd_state.trading_mode.value != "normal":
                print(f"⚠️ Trading in {dd_state.trading_mode.value.upper()} mode "
                      f"(DD: {dd_state.current_drawdown_pct:.1f}%, Size: {dd_state.position_size_multiplier*100:.0f}%)")
            
            if self.risk_manager.check_circuit_breaker(current_portfolio_value, self.daily_start_balance):
                loss_pct = ((current_portfolio_value - self.daily_start_balance) / self.daily_start_balance) * 100
                print("⚠️ TRADING HALTED: Circuit Breaker Triggered (Daily Loss Limit Reached)")
                await self.telegram_notifier.send_circuit_breaker_alert(
                    loss_pct=loss_pct,
                    current_balance=current_portfolio_value,
                    start_balance=self.daily_start_balance
                )
                return

            # 2. CHECK ALL STOP-LOSSES (also done by fast check, but ensure consistency)
            await self._check_all_stops()

            # 3. Iterate through each symbol for new signals
            self.cycle_count += 1
            print(f"\n{'='*60}")
            print(f"  ANALYSIS CYCLE #{self.cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            # Process symbols with controlled concurrency
            # Process 3 at a time to avoid rate limits while being faster than sequential
            batch_size = 3
            for i in range(0, len(self.symbols), batch_size):
                batch = self.symbols[i:i + batch_size]
                tasks = [self._process_symbol_safe(symbol) for symbol in batch]
                await asyncio.gather(*tasks)
                
                # Small delay between batches to avoid rate limits
                if i + batch_size < len(self.symbols):
                    await asyncio.sleep(1)
            
            # 4. Print Portfolio Summary
            await self._print_portfolio_summary()
            
            # Note: State saving is now handled by a separate scheduled job

        except Exception as e:
            print(f"Error in trading cycle: {e}")
            import traceback
            traceback.print_exc()
            # Save state on error to prevent data loss
            await self.save_state()
        finally:
            self.is_running = False

    async def _handle_daily_reset(self):
        """Handle daily P&L reset and reporting."""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            total_value = await self._calculate_portfolio_value()
            pnl = total_value - self.daily_start_balance
            pnl_pct = (pnl / self.daily_start_balance) * 100 if self.daily_start_balance > 0 else 0
            
            await self.telegram_notifier.send_daily_summary(
                pnl=pnl,
                pnl_pct=pnl_pct,
                trades_count=self.daily_trades_count,
                start_balance=self.daily_start_balance,
                end_balance=total_value
            )
            
            self.daily_start_balance = total_value
            self.last_reset_date = current_date
            self.daily_trades_count = 0
            print(f"--- DAILY RESET: New Start Balance ${self.daily_start_balance:,.2f} ---")

    async def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        total_value = self.paper_exchange.get_balance("USDT")
        
        for sym in self.symbols:
            ohlcv = await self.market_data_provider.get_ohlcv(sym, limit=1)
            price = ohlcv.iloc[-1]['close'] if not ohlcv.empty else 0
            asset = sym.split('/')[0]
            qty = self.paper_exchange.get_balance(asset)
            total_value += qty * price
        
        return total_value

    async def _check_all_stops(self):
        """Check stop-loss and take-profit for all open positions."""
        positions = self.paper_exchange.get_all_positions()
        
        # Sync positions with memory
        self.memory.update_open_positions(
            {s: p.to_dict() for s, p in positions.items()}
        )
        
        if not positions:
            return
        
        print(f"\n📊 Checking {len(positions)} open position(s)...")
        
        # Get current prices for all symbols with positions
        current_prices = {}
        for symbol in positions.keys():
            ohlcv = await self.market_data_provider.get_ohlcv(symbol, limit=1)
            if not ohlcv.empty:
                current_prices[symbol] = ohlcv.iloc[-1]['close']
        
        # Check stops
        closed = self.paper_exchange.check_stops(current_prices, self.atr_cache)
        
        # Send notifications and record to memory
        for closed_pos in closed:
            # Record to agent memory
            self.memory.add_trade_from_dict(closed_pos)
            
            # Log with production logger + Telegram based on exit reason
            if closed_pos['exit_reason'] == 'stop_loss':
                await self.trade_logger.log_stop_loss(
                    symbol=closed_pos['symbol'],
                    entry_price=closed_pos['entry_price'],
                    exit_price=closed_pos['exit_price'],
                    pnl=closed_pos['pnl'],
                    pnl_pct=closed_pos['pnl_pct']
                )
                lesson = f"Stop-loss hit on {closed_pos['symbol']}. Entry: ${closed_pos['entry_price']:,.0f}, Loss: {closed_pos['pnl_pct']:.1f}%"
                self.memory.add_lesson(lesson)
            elif closed_pos['exit_reason'] == 'take_profit':
                await self.trade_logger.log_take_profit(
                    symbol=closed_pos['symbol'],
                    entry_price=closed_pos['entry_price'],
                    exit_price=closed_pos['exit_price'],
                    pnl=closed_pos['pnl'],
                    pnl_pct=closed_pos['pnl_pct']
                )
            else:
                # Other exits (trailing stop, max hold, etc.)
                await self.trade_logger.log_sell(
                    symbol=closed_pos['symbol'],
                    amount=closed_pos.get('amount', 0),
                    price=closed_pos['exit_price'],
                    pnl=closed_pos['pnl'],
                    pnl_pct=closed_pos['pnl_pct'],
                    reason=closed_pos['exit_reason'].replace('_', ' ').title()
                )
            
            self.daily_trades_count += 1
            
            # Record trade to drawdown manager
            self.drawdown_manager.record_trade(
                pnl=closed_pos['pnl'],
                pnl_pct=closed_pos['pnl_pct']
            )
            
            # Self-reflection: Analyze the closed trade
            market_context = {
                'market_regime': closed_pos.get('market_regime', 'unknown'),
                'rsi': closed_pos.get('rsi', 50),
                'adx': closed_pos.get('adx', 25),
                'change_24h': closed_pos.get('change_24h', 0)
            }
            analysis = self.reflector.analyze_trade(closed_pos, market_context)
            
            # Log insights from reflection
            if analysis.issues_identified:
                print(f"  🔍 Issues identified: {', '.join(analysis.issues_identified[:2])}")
            if analysis.what_went_well:
                print(f"  ✨ Positives: {', '.join(analysis.what_went_well[:2])}")
            
            # Save trade to database for persistence
            await state_manager.save_trade(closed_pos)
            await state_manager.close_position_in_db(
                symbol=closed_pos['symbol'],
                exit_price=closed_pos['exit_price'],
                exit_reason=closed_pos['exit_reason'],
                pnl=closed_pos['pnl'],
                pnl_pct=closed_pos['pnl_pct']
            )

    async def _process_symbol_safe(self, symbol: str):
        """Wrapper that catches errors to prevent one symbol from crashing the batch."""
        try:
            await self._process_symbol(symbol)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            print(f"  ❌ Error processing {symbol}: {str(e)[:50]}")
    
    async def _process_symbol(self, symbol: str):
        """Process a single symbol for trading signals."""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing {symbol}")
        
        # Fetch Market Data
        market_snapshot = await self.market_data_provider.get_market_snapshot(symbol)
        if not market_snapshot:
            print(f"Skipping {symbol}: Failed to fetch market data.")
            return

        current_price = market_snapshot['price']
                
        # Cache ATR for trailing stops
        tf_1h = market_snapshot['timeframes'].get('1h', {}).get('indicators', {})
        tf_4h = market_snapshot['timeframes'].get('4h', {}).get('indicators', {})
        
        # Get ATR with proper fallback
        atr = tf_1h.get('atr', 0)
        if atr == 0 or atr is None:
            # Try 4H ATR
            atr = tf_4h.get('atr', 0)
        if atr == 0 or atr is None:
            # Fallback: Use 2% of price as proxy for ATR
            atr = current_price * 0.02
            print(f"  ⚠️ ATR unavailable, using 2% fallback: {atr:.4f}")
        
        self.atr_cache[symbol] = atr
        
        # Print current position status
        position = self.paper_exchange.get_position(symbol)
        if position:
            unrealized_pnl = position.get_unrealized_pnl(current_price)
            unrealized_pnl_pct = position.get_unrealized_pnl_pct(current_price)
            print(f"  📍 OPEN POSITION: {position.side.upper()} {position.amount:.4f} @ ${position.entry_price:,.2f}")
            print(f"     Current: ${current_price:,.2f} | P&L: ${unrealized_pnl:,.2f} ({unrealized_pnl_pct:+.2f}%)")
            print(f"     SL: ${position.stop_loss:,.2f} | TP: ${position.take_profit:,.2f}")
                
        # Print Indicator Summary
        print(f"  Price: ${current_price:,.2f}")
        print(f"  1H: RSI={tf_1h.get('rsi', 'N/A')} | MACD={tf_1h.get('macd', 'N/A')} | ADX={tf_1h.get('adx', 'N/A')}")
        print(f"  4H: RSI={tf_4h.get('rsi', 'N/A')} | MACD={tf_4h.get('macd', 'N/A')} | ADX={tf_4h.get('adx', 'N/A')}")

        # Fetch News Sentiment
        news_sentiment = await self.news_provider.get_market_sentiment()
        market_snapshot['news_sentiment'] = news_sentiment
        print(f"  News: {news_sentiment['sentiment_label']} ({news_sentiment['sentiment_score']:+.2f})")

        # Add memory context to market snapshot
        market_snapshot['memory_context'] = self.memory.build_context_for_prompt(symbol)
        
        # Add reflection insights if available
        reflection_insights = self.reflector.get_insights_for_prompt(max_insights=3)
        if reflection_insights:
            market_snapshot['memory_context'] += f"\n\n{reflection_insights}"
        market_snapshot['trading_rules'] = self.memory.get_trading_rules_from_memory()
        market_snapshot['open_positions'] = {
            s: p.to_dict() for s, p in self.paper_exchange.get_all_positions().items()
        }
        
        # Print memory summary
        win_rate = self.memory.get_win_rate(symbol)
        recent_trades = self.memory.get_recent_trades(symbol, n=3)
        if recent_trades:
            print(f"  🧠 Memory: {len(recent_trades)} recent trades on {symbol}, Win rate: {win_rate:.0f}%")

        # Agent Analysis
        print("  Analyzing...")
        try:
            # Use analyze_with_memory if available, otherwise regular analyze
            if hasattr(self.agent, 'analyze_with_memory'):
                analysis_json = await self.agent.analyze_with_memory(market_snapshot)
            else:
                analysis_json = await self.agent.analyze_market(market_snapshot)
                
            # Parse response
            if isinstance(analysis_json, str):
                cleaned_json = analysis_json.replace('```json', '').replace('```', '').strip()
                decision = json.loads(cleaned_json)
            else:
                decision = analysis_json
                    
            print(f"  Agent: {decision.get('trend', 'neutral').upper()} ({decision.get('confidence', 'low')})")
            
            # Handle signal
            await self._handle_signal(symbol, decision, current_price, atr, market_snapshot)
            
        except json.JSONDecodeError as e:
            print(f"  ❌ Failed to parse agent response: {e}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    async def _handle_signal(
        self,
        symbol: str,
        decision: Dict[str, Any],
        current_price: float,
        atr: float,
        market_snapshot: Dict[str, Any]
    ):
        """
        Handle trading signal using SpotSignalHandler.
        
        SPOT TRADING:
        - BUY: Purchase asset with USDT (open position)
        - SELL: Sell asset back to USDT (close position)
        """
        reasoning = decision.get("reasoning", "No reasoning provided")
        market_regime = decision.get("market_regime", "unknown")
        
        # Check if we already have a position
        has_position = self.paper_exchange.has_position(symbol)
        
        # Use SpotSignalHandler for clean BUY/SELL logic
        signal_decision = self.spot_handler.process_signal(
            agent_decision=decision,
            market_data=market_snapshot,
            has_position=has_position,
            symbol=symbol
        )
        
        # Get emoji and log
        emoji = self.spot_handler.get_action_emoji(signal_decision.action)
        print(f"  {emoji} Action: {signal_decision.action.value.upper()} (Signal: {signal_decision.signal_type})")
        
        if signal_decision.blocked_reason:
            print(f"  🚫 Blocked: {signal_decision.blocked_reason}")
        
        # Determine action
        multiplier = signal_decision.position_multiplier
        
        # Execute action based on SpotAction
        if signal_decision.action == SpotAction.BUY:
            print(f"  🟢 Executing SPOT BUY for {symbol}")
            await self._open_position(
                symbol=symbol,
                side="long",
                current_price=current_price,
                atr=atr,
                multiplier=multiplier,
                reasoning=reasoning,
                market_regime=market_regime,
                market_snapshot=market_snapshot
            )
        
        elif signal_decision.action == SpotAction.SELL:
            print(f"  🔴 Executing SPOT SELL for {symbol}")
            result = self.paper_exchange.close_position(symbol, current_price, "signal_sell")
            if result:
                # Record to memory
                self.memory.add_trade_from_dict(result)
                
                # Record to drawdown manager
                self.drawdown_manager.record_trade(
                    pnl=result['pnl'],
                    pnl_pct=result['pnl_pct']
                )
                
                # Save to database for persistence
                await state_manager.save_trade(result)
                await state_manager.close_position_in_db(
                    symbol=symbol,
                    exit_price=result['exit_price'],
                    exit_reason="signal",
                    pnl=result['pnl'],
                    pnl_pct=result['pnl_pct']
                )
                
                self.daily_trades_count += 1
                
                # Log SELL with production logger + Telegram
                await self.trade_logger.log_sell(
                    symbol=symbol,
                    amount=result['amount'],
                    price=result['exit_price'],
                    pnl=result['pnl'],
                    pnl_pct=result['pnl_pct'],
                    reason="Agent SELL signal"
                )

    async def _open_position(
        self,
        symbol: str,
        side: str,
        current_price: float,
        atr: float,
        multiplier: float,
        reasoning: str,
        market_regime: str,
        market_snapshot: Dict[str, Any] = None
    ):
        """Open a new position with proper risk management, drawdown adjustment, and self-learning."""
        market_snapshot = market_snapshot or {}
        portfolio_value = self.paper_exchange.get_balance("USDT")
        
        # Calculate position size AND stop levels together
        base_amount, stop_levels = self.risk_manager.calculate_position_with_stops(
            portfolio_value=portfolio_value,
                            current_price=current_price,
            atr=atr,
            side=side,
            risk_per_trade_pct=0.01  # 1% risk per trade
        )
        
        # Adjust for market regime
        stop_levels = self.risk_manager.adjust_for_market_regime(
            stop_levels=stop_levels,
            market_regime=market_regime.lower() if market_regime else "unknown",
            atr=atr
        )
        
        # Apply signal strength multiplier
        trade_amount = base_amount * multiplier
        
        # Apply drawdown-based position size adjustment
        trade_amount = self.drawdown_manager.get_adjusted_position_size(trade_amount)
        
        # Apply reflection-based position size adjustment
        reflection_adjustment = self.reflector.get_position_size_adjustment()
        if reflection_adjustment < 1.0:
            trade_amount *= reflection_adjustment
            print(f"  ⚠️ Position reduced to {reflection_adjustment*100:.0f}% based on past performance")
        
        if trade_amount <= 0:
            print(f"  ❌ Position size is 0 (drawdown protection)")
            return
        
        # Check if trade should be blocked based on learned rules
        market_context = {
            'rsi': market_snapshot.get('rsi', 50),
            'adx': market_snapshot.get('adx', 25),
            'market_regime': market_regime
        }
        should_block, block_reason = self.reflector.should_block_trade(side, market_context)
        if should_block:
            print(f"  🛑 {block_reason}")
            return
        
        # Validate stop levels
        is_valid, reason = self.risk_manager.validate_stop_levels(
            stop_levels=stop_levels,
            current_price=current_price,
            side=side
        )
        
        if not is_valid:
            print(f"  ❌ Invalid stop levels: {reason}")
            return
        
        # Check risk limits
        is_safe = self.risk_manager.check_trade_risk(
            symbol=symbol,
            signal="buy" if side == "long" else "sell",
            amount=trade_amount,
            price=current_price,
            account_balance=portfolio_value
        )
        
        if not is_safe:
            print(f"  ❌ Risk check failed")
            return
        
        # Check correlation limits (if manager attached)
        if hasattr(self, 'corr_manager') and self.corr_manager:
            trade_value = trade_amount * current_price
            current_positions = {
                s: p.to_dict() for s, p in self.paper_exchange.get_all_positions().items()
            }
            
            can_open, corr_reason = self.corr_manager.can_open_position(
                symbol=symbol,
                position_size_usd=trade_value,
                current_positions=current_positions,
                portfolio_value=portfolio_value
            )
            
            if not can_open:
                print(f"  🚫 Correlation block: {corr_reason}")
                return
            
            # Apply correlation-based size reduction
            size_mult = self.corr_manager.get_position_size_multiplier(symbol, current_positions)
            if size_mult < 1.0:
                old_amount = trade_amount
                trade_amount *= size_mult
                print(f"  ⚠️ Reduced size due to correlation: {old_amount:.6f} → {trade_amount:.6f}")
        
        # Open position
        position = self.paper_exchange.open_position(
            symbol=symbol,
            side=side,
            amount=trade_amount,
            entry_price=current_price,
            stop_loss=stop_levels.stop_loss,
            take_profit=stop_levels.take_profit,
            trailing_stop=self.risk_manager.enable_trailing_stop,
            max_hold_hours=self.risk_manager.max_hold_hours,
            entry_reasoning=reasoning,
            agent_name=getattr(self.agent, 'name', 'unknown'),
            market_regime=market_regime
        )
        
        if position:
            self.daily_trades_count += 1
            
            # Log BUY with production logger + Telegram
            await self.trade_logger.log_buy(
                symbol=symbol,
                amount=trade_amount,
                price=current_price,
                reason=reasoning[:100],
                stop_loss=stop_levels.stop_loss,
                take_profit=stop_levels.take_profit
            )
            
            # Save to DB
            await self.paper_exchange.save_position_to_db(position)

    async def _print_portfolio_summary(self):
        """Print portfolio summary including drawdown status."""
        usdt_bal = self.paper_exchange.get_balance("USDT")
        positions = self.paper_exchange.get_all_positions()
        dd_status = self.drawdown_manager.get_status_summary()
        
        print(f"\n{'='*60}")
        print(f"💰 PORTFOLIO SUMMARY")
        print(f"{'='*60}")
        print(f"  USDT Balance: ${usdt_bal:,.2f}")
        print(f"  Open Positions: {len(positions)}")
        
        if positions:
            for symbol, pos in positions.items():
                # Get current price
                ohlcv = await self.market_data_provider.get_ohlcv(symbol, limit=1)
                if not ohlcv.empty:
                    current_price = ohlcv.iloc[-1]['close']
                    pnl = pos.get_unrealized_pnl(current_price)
                    pnl_pct = pos.get_unrealized_pnl_pct(current_price)
                    print(f"  {symbol}: {pos.amount:.4f} @ ${pos.entry_price:,.2f} | P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        
        # Drawdown status
        print(f"\n📉 DRAWDOWN STATUS")
        print(f"  Mode: {dd_status['trading_mode'].upper()} (Size: {dd_status['position_multiplier']*100:.0f}%)")
        print(f"  Peak: ${dd_status['peak_equity']:,.2f} | Current: ${dd_status['current_equity']:,.2f}")
        print(f"  Drawdown: {dd_status['current_drawdown_pct']:.1f}% | Max Seen: {dd_status['max_drawdown_seen_pct']:.1f}%")
        print(f"  Streak: {dd_status['consecutive_wins']}W / {dd_status['consecutive_losses']}L")
        if dd_status['days_in_drawdown'] > 0:
            print(f"  Days in DD: {dd_status['days_in_drawdown']}")
        
        # Performance summary
        perf = self.paper_exchange.get_performance_summary()
        if perf['total_trades'] > 0:
            print(f"\n📈 PERFORMANCE (Today)")
            print(f"  Trades: {perf['total_trades']} | Win Rate: {perf['win_rate']:.1f}%")
            print(f"  Total P&L: ${perf['total_pnl']:,.2f}")
        
        # Self-reflection insights
        if self.reflector.insights:
            print(f"\n🔍 SELF-LEARNING INSIGHTS ({len(self.reflector.insights)})")
            for insight in self.reflector.insights[:3]:
                emoji = {"info": "💡", "warning": "⚠️", "critical": "🛑"}.get(insight.severity, "📝")
                print(f"  {emoji} {insight.insight[:60]}...")
        
        # Trading adjustments active
        adjustments = self.reflector.get_trading_adjustments()
        active_rules = []
        if adjustments['avoid_overbought']:
            active_rules.append("No buy RSI>70")
        if adjustments['avoid_oversold']:
            active_rules.append("No sell RSI<30")
        if adjustments['require_trend']:
            active_rules.append("Require ADX>25")
        if adjustments['reduce_size']:
            active_rules.append("Size reduced")
        
        if active_rules:
            print(f"\n🛡️ ACTIVE PROTECTIONS: {' | '.join(active_rules)}")
        
        print(f"{'='*60}\n")
