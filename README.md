# 🏆 Alpha Arena

**AI-Powered Cryptocurrency Trading Bot with Multi-Model Support**

A professional-grade automated trading system that combines LLM intelligence with traditional technical analysis for cryptocurrency markets.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Paper%20Trading-yellow.svg)

---

## 🎯 Features

### 🤖 AI-Powered Analysis
- **Multi-Model Support**: Gemini, GPT-4, DeepSeek, Qwen, Claude via OpenRouter
- **Hierarchical Confluence Strategy**: Analyzes 15m, 1h, 4h timeframes with weighted scoring
- **Agent Memory System**: Learns from past trades to improve decisions
- **Self-Reflection Loop**: Automated post-trade analysis and rule generation

### 📊 Advanced Technical Analysis
- **Core Indicators**: RSI, MACD, Bollinger Bands, ADX, ATR, EMA (9/21/50)
- **Volume Analysis**: OBV with trend detection
- **Crypto-Specific**: BTC Correlation, Funding Rates, Fear & Greed Index
- **Multi-Timeframe**: 15-minute, 1-hour, and 4-hour analysis

### ⚠️ Risk Management
- **Dynamic Position Sizing**: Based on ATR volatility
- **Stop-Loss/Take-Profit**: Automatic calculation with trailing stops
- **Drawdown Management**: Reduces position sizes during losing streaks
- **Circuit Breaker**: Halts trading at daily/weekly loss limits
- **Pre-Trade Filters**: ADX check, timeframe alignment, BTC safety for alts

### 🔔 Notifications & Monitoring
- **Telegram Integration**: Real-time trade alerts, daily summaries
- **Position Tracking**: Live P&L, stop distances, portfolio value
- **State Persistence**: Saves/restores positions across restarts

---

## 🏗️ Architecture

```
alpha-arena/
├── backend/
│   ├── agents/           # AI trading agents (Gemini, GPT, etc.)
│   │   ├── base_agent.py
│   │   ├── gemini_agent.py
│   │   ├── memory.py         # Agent memory system
│   │   ├── reflection.py     # Self-learning module
│   │   ├── rule_based_agent.py
│   │   └── hybrid_agent.py
│   ├── data/             # Market data providers
│   │   ├── market_data.py    # OHLCV, indicators, BTC trend
│   │   └── news_sentiment.py # News analysis
│   ├── exchanges/        # Exchange interfaces
│   │   ├── paper_exchange.py # Paper trading simulator
│   │   └── realistic_execution.py
│   ├── risk/             # Risk management
│   │   ├── risk_manager.py   # Position sizing, stops
│   │   └── correlation_manager.py
│   ├── scheduler/        # Trading loop orchestration
│   │   └── trade_scheduler.py
│   ├── database/         # Persistence layer
│   │   ├── models.py
│   │   ├── db.py
│   │   └── state_manager.py
│   └── notifications/    # Alerting
│       └── telegram_notifier.py
├── frontend/             # Next.js dashboard (optional)
├── scripts/              # Utility scripts
│   ├── start_bot.py      # Main entry point
│   ├── run_backtest.py
│   └── init_db.py
├── config/               # Configuration files
└── docker/               # Docker setup
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL (or SQLite for local testing)
- API Keys: Binance (data), Gemini/OpenAI (AI)

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/alpha-arena.git
cd alpha-arena

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key

# Optional: Fallback Gemini API keys (comma-separated)
# If primary key hits quota/rate limit, bot will automatically rotate to fallback keys
GEMINI_FALLBACK_KEYS=AIzaSyDXhEYXxCd_Ma6iRSp-SjFeXClpgouZ7qw,AIzaSyCUAsBJT2YfrR7X12UJuCLhK7Kes9Jv6ys,AIzaSyBn8_77NlAvw556ZpSzBpyVqm04M1rpInc

# Optional (for other agents)
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key

# Telegram Notifications (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Database
DATABASE_URL=sqlite:///./alpha_arena.db
# Or PostgreSQL: postgresql://user:pass@localhost:5432/alpha_arena
```

### 3. Initialize Database

```bash
python scripts/init_db.py
```

### 4. Start Trading Bot

```bash
python scripts/start_bot.py
```

---

## ⚙️ Configuration

### Trading Parameters

Edit `scripts/start_bot.py`:

```python
# Symbols to trade
symbols = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT",
    "BNB/USDT", "AVAX/USDT", "LINK/USDT"
]

# Risk settings
risk_manager = RiskManager(
    max_position_size_pct=0.10,  # 10% max per trade
    max_daily_loss_pct=0.05,     # 5% daily loss limit
    default_risk_reward=2.0,     # 2:1 reward/risk ratio
    enable_trailing_stop=True
)

# Scheduler intervals
scheduler = TradingScheduler(
    analysis_interval_minutes=15,  # Full analysis every 15 min
    stop_check_seconds=30,         # SL/TP check every 30 sec
    state_save_minutes=5           # Save state every 5 min
)
```

### Agent Selection

```python
# Choose your AI agent
from backend.agents.gemini_agent import GeminiAgent
from backend.agents.gpt_agent import GPTAgent
from backend.agents.hybrid_agent import HybridAgent

agent = GeminiAgent()  # or GPTAgent(), HybridAgent()
```

---

## 📈 Trading Strategy

### Entry Criteria (All must pass)

| Filter | Rule |
|--------|------|
| **BTC Safety** | BTC must not be in bearish trend for alt trades |
| **ADX Check** | ADX > 15 on both 1H and 4H (trend present) |
| **Alignment** | At least 2/3 timeframes must agree |
| **RSI** | Not overbought (RSI < 70) |
| **Confidence** | Agent confidence must be moderate or high |

### Exit Criteria

| Type | Trigger |
|------|---------|
| **Stop-Loss** | 2x ATR below entry |
| **Take-Profit** | 2:1 reward/risk ratio |
| **Trailing Stop** | Follows price, locks in gains |
| **Time Stop** | Max 72 hours hold time |
| **Signal Exit** | Agent issues SELL signal |

### Position Sizing

```
Position Size = (Portfolio × Risk%) / Stop Distance
                    ↓
          Capped at 10% of portfolio
                    ↓
          Adjusted by drawdown mode (25-100%)
```

---

## 📊 Monitoring

### Console Output

```
╔══════════════════════════════════════════════════════════════════╗
║  TRADING SCHEDULER STARTED                                       ║
╠══════════════════════════════════════════════════════════════════╣
║  📊 Full Analysis:    Every 15 minutes                           ║
║  🛑 Stop-Loss Check:  Every 30 seconds                           ║
║  💾 State Save:       Every  5 minutes                           ║
║  📈 Symbols:          12 pairs                                   ║
╚══════════════════════════════════════════════════════════════════╝

[2025-11-27 18:38:40] Processing BTC/USDT
  Price: $96,500.00
  1H: RSI=45.2 | MACD=bullish | ADX=28.5
  4H: RSI=52.1 | MACD=bullish | ADX=32.1
  News: positive (+0.30)
  Agent: BULLISH (high)
  Action: OPEN_LONG (Signal: MODERATE_BUY)
```

### Telegram Alerts

- 🟢 Trade opened
- 🔴 Trade closed
- 💰 Position P&L
- ⚠️ Drawdown warnings
- 📊 Daily summary

---

## 🧪 Paper Trading

The bot runs in **paper trading mode** by default:

```python
paper_exchange = PaperExchange(initial_balance=10000.0)
```

No real money is at risk. All trades are simulated.

---

## 🔧 Development

### Run Tests

```bash
pytest backend/tests/
```

### Run Backtest

```bash
python scripts/run_backtest.py --symbol BTC/USDT --days 30
```

### Lint & Format

```bash
# Install dev dependencies
pip install black flake8 mypy

# Format
black backend/

# Lint
flake8 backend/
```

---

## ⚠️ Disclaimer

**This software is for educational purposes only.**

- Trading cryptocurrencies involves substantial risk of loss
- Past performance does not guarantee future results
- Never trade with money you cannot afford to lose
- Always test thoroughly in paper trading before considering live trading
- The authors are not responsible for any financial losses

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/alpha-arena/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/alpha-arena/discussions)

---

<p align="center">
  <b>Built with ❤️ for the crypto community</b>
</p>
