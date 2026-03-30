# KlawTrade

Autonomous algorithmic trading system with deterministic strategies, 14-check risk management, circuit breakers, and a real-time dashboard.

Works out of the box in **simulation mode** (no API keys needed). Optionally connects to **Alpaca** for real paper trading.

## Quick Start Guide

### Option 1: pip install

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

Open **http://localhost:8080** to see the dashboard.

## Features

- **Rule-based strategies** -- Momentum (SMA crossover + RSI + MACD) and Mean Reversion (Bollinger Bands + Stochastic RSI)
- **14-check risk manager** -- Position sizing, daily loss limits, drawdown protection, sector allocation, correlated exposure, cash reserves, and more
- **Circuit breakers** -- Auto-halt on consecutive losses, daily/weekly loss, max drawdown, or system errors
- **Kelly criterion position sizing** -- Optimal bet sizing with half-Kelly conservative default
- **Real-time dashboard** -- Equity curve, open positions, trade log, kill switch, WebSocket updates
- **Simulation mode** -- Runs without any API keys using simulated market data and a mock broker
- **Alpaca integration** -- Drop in your API keys for real paper trading
- **SQLite storage** -- All signals, orders, trades, and portfolio snapshots persisted

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
system:
  heartbeat_interval_seconds: 30
  always_on: true        # Trade outside market hours (simulation)

risk:
  max_daily_loss_pct: 0.03
  max_drawdown_pct: 0.15
  max_single_position_pct: 0.10

portfolio:
  starting_capital: 100000

dashboard:
  port: 8080
```

## Alpaca Paper Trading

1. Create a free account at [alpaca.markets](https://alpaca.markets)
2. Generate paper trading API keys
3. Add them to your `.env` file:

```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

4. Restart KlawTrade -- it auto-detects the keys and switches to Alpaca

## CLI Commands

```bash
klawtrade start              # Start trading system
klawtrade start --sim        # Force simulation mode
klawtrade start -p 9090      # Custom dashboard port
klawtrade start -c path.yaml # Custom config file
klawtrade init [directory]   # Initialize project directory
klawtrade status             # Check if KlawTrade is running
```

## Architecture

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
  utils/
    logging.py            # Structured JSON logging
    time_utils.py         # Market hours utilities
```

## Risk Management

Every signal passes through **all 14 checks** before execution (no short-circuit):

1. Symbol blacklist
2. Circuit breaker status
3. Signal expiry
4. Confidence threshold (min 70%)
5. Daily trade count limit
6. Max open positions
7. Volume threshold
8. Spread width
9. Position size limit (max 10% of portfolio)
10. Sector allocation (max 30%)
11. Correlated exposure (max 40%)
12. Cash reserve (min 10%)
13. Daily loss limit (max 3%)
14. Drawdown limit (max 15%)

## License

MIT
