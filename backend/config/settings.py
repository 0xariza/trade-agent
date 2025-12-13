"""
Centralized Configuration for Alpha Arena.

All hardcoded values should be moved here.
Uses Pydantic Settings for validation and .env file support.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from enum import Enum


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class AgentType(str, Enum):
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Usage:
        from backend.config.settings import settings
        print(settings.max_drawdown_pct)
    """
    
    # =========================
    # API KEYS
    # =========================
    # Note: gemini_api_key is defined below in AGENT SELECTION section with fallback support
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openrouter_api_key: str = Field(default="", env="OPENROUTER_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    
    # Telegram
    telegram_bot_token: str = Field(default="", env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", env="TELEGRAM_CHAT_ID")
    
    # =========================
    # TRADING MODE
    # =========================
    trading_mode: TradingMode = Field(default=TradingMode.PAPER)
    risk_profile: RiskProfile = Field(default=RiskProfile.CONSERVATIVE)
    
    # =========================
    # AGENT SELECTION
    # =========================
    # Primary agent: Gemini (falls back to rule-based on failure)
    agent_type: AgentType = Field(default=AgentType.GEMINI, env="AGENT_TYPE")
    openrouter_model: str = Field(default="google/gemini-2.0-flash-001", env="OPENROUTER_MODEL")
    
    # Gemini API Keys (with fallback support)
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    gemini_fallback_keys: str = Field(
        default="",
        env="GEMINI_FALLBACK_KEYS"  # Comma-separated list of fallback API keys
    )
    # Example: GEMINI_FALLBACK_KEYS=key1,key2,key3
    
    # LLM Token Configuration
    llm_max_output_tokens: int = Field(default=4000, ge=500, le=32000, env="LLM_MAX_OUTPUT_TOKENS")
    # Increased from 2000 to 4000 for more detailed analysis
    # Can be set up to 32000 for models that support it
    
    # =========================
    # EXCHANGE SELECTION
    # =========================
    # Options: binance, binanceus, kraken, kucoin, bybit, okx, coinbase
    exchange_id: str = Field(default="binance", env="EXCHANGE_ID")
    
    # Fallback exchanges (tried in order if primary fails)
    exchange_fallbacks: List[str] = Field(
        default=["kraken", "kucoin", "bybit", "okx"],
        env="EXCHANGE_FALLBACKS"
    )
    
    # Exchange API credentials (optional - not needed for public data)
    kraken_api_key: str = Field(default="", env="KRAKEN_API_KEY")
    kraken_api_secret: str = Field(default="", env="KRAKEN_API_SECRET")
    kucoin_api_key: str = Field(default="", env="KUCOIN_API_KEY")
    kucoin_api_secret: str = Field(default="", env="KUCOIN_API_SECRET")
    kucoin_passphrase: str = Field(default="", env="KUCOIN_API_PASSPHRASE")  # KuCoin requires passphrase
    
    # =========================
    # DATABASE CONFIGURATION
    # =========================
    # PostgreSQL is the default (recommended for production)
    # Format: postgresql+asyncpg://user:password@host:port/database
    # Example: postgresql+asyncpg://alpha_user:alpha_password@localhost:5432/alpha_arena
    # Database URL
    # Default: SQLite (works out of the box, no setup required)
    # For PostgreSQL: postgresql+asyncpg://user:password@localhost:5432/dbname
    database_url: str = Field(
        default="sqlite+aiosqlite:///./alpha_arena.db",
        env="DATABASE_URL"
    )
    
    # =========================
    # SYMBOLS & SCHEDULING
    # =========================
    # Trading symbols - Top cryptocurrencies with good liquidity
    # Can be overridden via TRADING_SYMBOLS env var (comma-separated)
    trading_symbols: List[str] = Field(
        default=[
            # Top 4 (Major)
            "BTC/USDT",   # Bitcoin
            "ETH/USDT",   # Ethereum
            "SOL/USDT",   # Solana
            "BNB/USDT",   # Binance Coin
            
            # Top Altcoins (High liquidity)
            "XRP/USDT",   # Ripple
            "ADA/USDT",   # Cardano
            "DOGE/USDT",  # Dogecoin
            "AVAX/USDT",  # Avalanche
            "MATIC/USDT", # Polygon
            "DOT/USDT",   # Polkadot
            "LINK/USDT",  # Chainlink
            "UNI/USDT",   # Uniswap
            "LTC/USDT",   # Litecoin
            "ATOM/USDT",  # Cosmos
        ],
        env="TRADING_SYMBOLS"  # Can override via env: TRADING_SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT
    )
    
    analysis_interval_minutes: int = Field(default=15, ge=1, le=60)
    stop_check_seconds: int = Field(default=30, ge=5, le=300)
    state_save_minutes: int = Field(default=5, ge=1, le=30)
    
    # Performance settings
    parallel_analysis_batch_size: int = Field(default=3, ge=1, le=10, env="PARALLEL_BATCH_SIZE")
    # Number of symbols to process in parallel (higher = faster but more API calls)
    
    parallel_analysis_batch_delay_seconds: float = Field(default=1.0, ge=0.1, le=5.0, env="BATCH_DELAY_SECONDS")
    # Delay between batches to avoid rate limits
    
    market_data_cache_ttl_seconds: int = Field(default=60, ge=10, le=300, env="MARKET_DATA_CACHE_TTL")
    # Cache market snapshots for this many seconds (reduces API calls)
    
    # =========================
    # RISK MANAGEMENT
    # =========================
    # Position sizing
    max_position_size_pct: float = Field(default=0.10, ge=0.01, le=0.50)
    risk_per_trade_pct: float = Field(default=0.01, ge=0.005, le=0.05)
    max_concurrent_positions: int = Field(default=6, ge=1, le=20)
    
    # Drawdown thresholds
    max_drawdown_pct: float = Field(default=0.15, ge=0.05, le=0.30)
    cautious_drawdown_pct: float = Field(default=0.05, ge=0.02, le=0.10)
    defensive_drawdown_pct: float = Field(default=0.08, ge=0.03, le=0.15)
    recovery_drawdown_pct: float = Field(default=0.12, ge=0.05, le=0.20)
    
    # Loss limits
    max_daily_loss_pct: float = Field(default=0.05, ge=0.02, le=0.10)
    weekly_loss_limit_pct: float = Field(default=0.08, ge=0.03, le=0.15)
    monthly_loss_limit_pct: float = Field(default=0.15, ge=0.05, le=0.25)
    max_consecutive_losses: int = Field(default=5, ge=2, le=10)
    
    # =========================
    # STOP-LOSS / TAKE-PROFIT
    # =========================
    default_risk_reward: float = Field(default=2.0, ge=1.0, le=5.0)
    default_atr_stop_multiplier: float = Field(default=2.0, ge=1.0, le=4.0)
    enable_trailing_stop: bool = Field(default=True)
    max_hold_hours: int = Field(default=72, ge=1, le=168)
    
    # Dynamic ATR multipliers by regime
    atr_mult_trending: float = Field(default=2.0)
    atr_mult_ranging: float = Field(default=1.5)
    atr_mult_volatile: float = Field(default=2.5)
    
    # =========================
    # PRE-TRADE FILTERS
    # =========================
    # NOTE: Original values were TOO STRICT (blocking 80%+ of trades)
    # Relaxed defaults below - tune based on your backtest results
    
    min_adx_threshold: float = Field(default=12.0, ge=5.0, le=30.0)  # Was 15
    strong_adx_threshold: float = Field(default=20.0, ge=15.0, le=40.0)  # Was 25
    max_rsi_for_buy: float = Field(default=72.0, ge=60.0, le=85.0)  # Was 70
    min_rsi_for_sell: float = Field(default=28.0, ge=15.0, le=40.0)  # Was 30
    require_btc_safety_for_alts: bool = Field(default=True)
    min_timeframe_alignment: int = Field(default=2)  # 2 out of 3
    
    # NEW: Filter strictness control (conservative, moderate, aggressive)
    # "conservative" = original strict filters
    # "moderate" = balanced (recommended)
    # "aggressive" = looser filters, more trades
    filter_strictness: str = Field(default="moderate")
    
    # NEW: Allow override filters when confidence is high
    allow_high_confidence_override: bool = Field(default=True)
    
    # =========================
    # LLM SETTINGS
    # =========================
    primary_llm: str = Field(default="gemini")  # gemini, openai, claude
    fallback_to_rule_based: bool = Field(default=True)
    enable_ensemble_voting: bool = Field(default=False)
    llm_temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    llm_max_retries: int = Field(default=3, ge=1, le=5)
    llm_retry_delay_seconds: float = Field(default=2.0)
    
    # =========================
    # PAPER EXCHANGE
    # =========================
    initial_balance: float = Field(default=10000.0, ge=100.0)
    slippage_pct: float = Field(default=0.001, ge=0.0, le=0.01)
    fee_pct: float = Field(default=0.001, ge=0.0, le=0.01)
    
    # =========================
    # CORRELATION LIMITS
    # =========================
    max_correlation_exposure: float = Field(default=0.7)  # Max correlation between positions
    max_sector_exposure_pct: float = Field(default=0.30)  # Max 30% in one "sector" (e.g., L1s)
    
    # =========================
    # PARTIAL PROFIT TAKING
    # =========================
    enable_partial_exits: bool = Field(default=False)
    partial_exit_at_1r_pct: float = Field(default=0.50)  # Exit 50% at 1R
    move_stop_to_breakeven_after_1r: bool = Field(default=True)
    
    # =========================
    # MEMORY & LEARNING
    # =========================
    # INCREASED from original 100/50 which was too limited for 12 symbols
    max_trade_memory: int = Field(default=500, ge=100, le=2000)  # Was 100
    max_decision_memory: int = Field(default=200, ge=50, le=1000)  # Was 50
    performance_lookback_days: int = Field(default=60, ge=14, le=180)  # Was 30
    reflection_after_n_losses: int = Field(default=3, ge=2, le=5)
    
    # NEW: Per-symbol memory allocation
    min_trades_per_symbol: int = Field(default=20)  # Ensure each symbol has history
    
    # NEW: Enable ML-based pattern learning (future feature)
    enable_pattern_learning: bool = Field(default=False)
    
    # =========================
    # MONITORING
    # =========================
    enable_prometheus_metrics: bool = Field(default=True)
    prometheus_port: int = Field(default=8000)
    enable_health_checks: bool = Field(default=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
    
    def get_atr_multiplier(self, regime: str) -> float:
        """Get ATR multiplier based on market regime."""
        regime_map = {
            "trending": self.atr_mult_trending,
            "ranging": self.atr_mult_ranging,
            "volatile": self.atr_mult_volatile,
        }
        return regime_map.get(regime.lower(), self.default_atr_stop_multiplier)
    
    def get_max_positions_for_profile(self) -> int:
        """Adjust max positions based on risk profile."""
        profile_map = {
            RiskProfile.CONSERVATIVE: max(3, self.max_concurrent_positions // 2),
            RiskProfile.MODERATE: self.max_concurrent_positions,
            RiskProfile.AGGRESSIVE: min(20, int(self.max_concurrent_positions * 1.5)),
        }
        return profile_map.get(self.risk_profile, self.max_concurrent_positions)


# Singleton instance
settings = Settings()

