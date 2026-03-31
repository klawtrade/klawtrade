<div align="center">

# KlawTrade

### The Next Generation of Wealth Creation

<<<<<<< Updated upstream
Works out of the box in **simulation mode** (no API keys needed). Optionally connects to **Alpaca** and **5** other brokers for monetary or paper trading.
=======
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()
[![Website](https://img.shields.io/badge/Website-klawtrade.com-orange.svg)](https://klawtrade.com)

**Open-source algorithmic trading platform with rule-based strategies, 14-check risk management, and 6 broker integrations.**

[Website](https://klawtrade.com) &bull; [Documentation](https://klawtrade.com/docs) &bull; [GitHub Issues](https://github.com/klawtrade/klawtrade/issues)

</div>

---

## Quick Demo

Get a Python trading bot running in three commands -- no API keys required:

```bash
pip install klawtrade
klawtrade init my-trading-bot && cd my-trading-bot
klawtrade start
```

Open **http://localhost:8080** to see the real-time trading dashboard.

---

## Features

### :bar_chart: Rule-Based Trading Strategies

Seven deterministic strategies built for automated trading: **SMA crossover**, **RSI**, **MACD**, **Bollinger Bands**, **Stochastic RSI**, **Momentum**, and **Mean Reversion**. Every signal includes a confidence score and full audit trail.

### :shield: 14-Check Risk Gate

Every trade passes through all 14 risk checks before execution -- no short-circuit evaluation. Covers position sizing, daily loss limits, drawdown protection, sector allocation, correlated exposure, cash reserves, volume thresholds, and spread width.

### :zap: 7-Trigger Circuit Breaker

Automatic trading halt on consecutive losses, daily loss limit, weekly loss limit, max drawdown, VIX spike, error rate threshold, or manual kill switch. Protects capital during volatile markets.

### :chart_with_upwards_trend: Real-Time Dashboard

WebSocket-powered dashboard at `localhost:8080` with live equity curve, open positions, trade log, portfolio metrics, and a one-click kill switch for emergency stops.

### :link: 6 Broker Integrations

Connect to **Alpaca**, **Interactive Brokers**, **Coinbase**, **Binance**, **Kraken**, or **Tradier**. Runs in simulation mode with zero configuration -- add API keys when you are ready for live paper trading or real execution.

### :test_tube: Historical Backtesting

Backtest any strategy against historical data. Outputs Sharpe ratio, win rate, max drawdown, profit factor, and full trade logs for quantitative analysis.

---
>>>>>>> Stashed changes

## Quick Start Guide

### Option 1: pip install (recommended)

```bash
pip install klawtrade
klawtrade init my-trading-bot
cd my-trading-bot
klawtrade start
```

### Option 2: Clone + uv

```bash
git clone https://github.com/klawtrade/klawtrade.git
cd klawtrade
uv sync
uv run klawtrade start
```

### Option 3: Docker

```bash
git clone https://github.com/klawtrade/klawtrade.git
cd klawtrade
cp .env.example .env
docker compose up -d
```

Works out of the box in **simulation mode** -- no API keys needed. Open **http://localhost:8080** to see the dashboard.

---

## Architecture

```
Signal Generation ──> Risk Gate (14 checks) ──> Circuit Breaker (7 triggers) ──> Broker Execution
```

```
src/
  main.py                 # Orchestrator loop
  cli.py                  # CLI entry point
  config.py               # Pydantic config loader
  strategy/
    engine.py             # Strategy runner + risk gate
    rules/
      momentum.py         # SMA crossover + RSI + MACD
      mean_reversion.py   # Bollinger + Stoch RSI
  risk/
    manager.py            # 14-check risk gate
    circuit_breaker.py    # Auto-halt system
    position_sizer.py     # Kelly criterion sizing
    limits.py             # Hard limits
  execution/
    broker.py             # Abstract broker interface
    sim_broker.py         # Simulated broker
    alpaca_broker.py      # Alpaca paper/live broker
    order_manager.py      # Order lifecycle
  market_data/
    provider.py           # Abstract data provider
    simulated_data.py     # Simulated price feeds
    technical.py          # pandas-ta indicators
    aggregator.py         # Unified snapshot builder
  portfolio/
    state.py              # Portfolio tracking
  storage/
    database.py           # SQLite async storage
    models.py             # Data models
  dashboard/
    app.py                # FastAPI + WebSocket dashboard
```

---

## Broker Setup

| Broker | Environment Variables | Optional Dependency |
|---|---|---|
| **Simulation** | None required | -- |
| **Alpaca** | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Included by default |
| **Interactive Brokers** | `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID` | `pip install klawtrade[ibkr]` |
| **Coinbase** | `COINBASE_API_KEY`, `COINBASE_API_SECRET` | `pip install klawtrade[coinbase]` |
| **Binance** | `BINANCE_API_KEY`, `BINANCE_API_SECRET` | `pip install klawtrade[binance]` |
| **Kraken** | `KRAKEN_API_KEY`, `KRAKEN_API_SECRET` | `pip install klawtrade[kraken]` |
| **Tradier** | `TRADIER_ACCESS_TOKEN`, `TRADIER_ACCOUNT_ID` | `pip install klawtrade[tradier]` |

Install all broker integrations at once:

```bash
pip install klawtrade[all-brokers]
```

Add your keys to a `.env` file in the project root. KlawTrade auto-detects available credentials and connects to the appropriate broker.

---

## Dashboard

The built-in trading dashboard runs at **http://localhost:8080** and provides:

- **Live equity curve** -- portfolio value over time with WebSocket updates
- **Open positions** -- current holdings with unrealized P&L
- **Trade log** -- full history of executed trades
- **Portfolio metrics** -- daily return, total return, Sharpe ratio, max drawdown
- **Kill switch** -- one-click emergency stop for all trading activity

No additional setup required -- the dashboard starts automatically with `klawtrade start`.

---

## Configuration

Edit `config/settings.yaml` to customize the trading system:

```yaml
system:
  heartbeat_interval_seconds: 30
  always_on: true               # Trade outside market hours (simulation)

risk:
  max_daily_loss_pct: 0.03      # 3% daily loss limit
  max_drawdown_pct: 0.15        # 15% max drawdown
  max_single_position_pct: 0.10 # 10% max position size
  max_sector_allocation: 0.30   # 30% max sector exposure
  min_cash_reserve_pct: 0.10    # 10% cash reserve

portfolio:
  starting_capital: 100000

strategies:
  - momentum                    # SMA crossover + RSI + MACD
  - mean_reversion              # Bollinger Bands + Stoch RSI

dashboard:
  port: 8080
```

See [full configuration reference](https://klawtrade.com/docs/configuration) for all options.

---

## Backtesting

Run a historical backtest against any strategy:

```bash
klawtrade backtest --strategy momentum --start 2024-01-01 --end 2024-12-31
```

Sample output:

```
Strategy: Momentum (SMA + RSI + MACD)
Period: 2024-01-01 to 2024-12-31
Total Trades: 142
Win Rate: 58.4%
Sharpe Ratio: 1.87
Max Drawdown: -8.2%
Profit Factor: 1.64
Total Return: 24.3%
```

---

## CLI Commands

```bash
klawtrade start              # Start the trading system
klawtrade start --sim        # Force simulation mode
klawtrade start -p 9090      # Custom dashboard port
klawtrade start -c path.yaml # Custom config file
klawtrade init [directory]   # Initialize a new project
klawtrade status             # Check if KlawTrade is running
klawtrade backtest           # Run historical backtesting
```

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch and open a pull request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**[Website](https://klawtrade.com)** &bull; **[Documentation](https://klawtrade.com/docs)** &bull; **[GitHub Issues](https://github.com/klawtrade/klawtrade/issues)** &bull; **[PyPI](https://pypi.org/project/klawtrade/)**

Built for algorithmic traders, quantitative analysts, and anyone automating their trading strategies with Python.

</div>
