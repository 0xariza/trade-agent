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
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openrouter_api_key: str = Field(default="", env="OPENROUTER_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    
    # Telegram
    telegram_bot_token: str = Field(default="", env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", env="TELEGRAM_CHAT_ID")
    
    # Database
    database_url: str = Field(
        default="sqlite:///./alpha_arena.db",
        env="DATABASE_URL"
    )
    
    # =========================
    # TRADING MODE
    # =========================
    trading_mode: TradingMode = Field(default=TradingMode.PAPER)
    risk_profile: RiskProfile = Field(default=RiskProfile.CONSERVATIVE)
    
    # =========================
    # SYMBOLS & SCHEDULING
    # =========================
    # Start with fewer symbols to avoid rate limits
    # Expand after testing is stable
    trading_symbols: List[str] = Field(
        default=[
            "BTC/USDT", "ETH/USDT", "SOL/USDT",
            "BNB/USDT"  # Start with 4, add more after validation
        ]
    )
    
    analysis_interval_minutes: int = Field(default=15, ge=1, le=60)
    stop_check_seconds: int = Field(default=30, ge=5, le=300)
    state_save_minutes: int = Field(default=5, ge=1, le=30)
    
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

