"""Configuration loader and validation using Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class TradingHoursConfig(BaseModel):
    start: str = "09:30"
    end: str = "16:00"


class SystemConfig(BaseModel):
    mode: str = "paper"
    timezone: str = "US/Eastern"
    trading_hours: TradingHoursConfig = TradingHoursConfig()
    heartbeat_interval_seconds: int = 30
    max_concurrent_orders: int = 5
    always_on: bool = False

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("paper", "live"):
            raise ValueError(f"mode must be 'paper' or 'live', got '{v}'")
        return v


class BrokerConfig(BaseModel):
    provider: str = "alpaca"
    paper: bool = True


class CircuitBreakerConfig(BaseModel):
    consecutive_losses_halt: int = 5
    halt_duration_minutes: int = 60
    vix_threshold: float = 35.0
    daily_pnl_check_interval: int = 300


class OrderGuardsConfig(BaseModel):
    require_limit_orders: bool = True
    max_slippage_pct: float = 0.005
    min_volume_threshold: int = 100000
    max_spread_pct: float = 0.02
    blacklisted_symbols: list[str] = Field(default_factory=list)


class RiskConfig(BaseModel):
    max_portfolio_allocation: float = 0.90
    max_single_position_pct: float = 0.10
    max_sector_allocation_pct: float = 0.30
    max_correlated_exposure_pct: float = 0.40

    max_daily_loss_pct: float = 0.03
    max_weekly_loss_pct: float = 0.07
    max_drawdown_pct: float = 0.15
    max_single_trade_loss_pct: float = 0.02

    max_open_positions: int = 15
    max_daily_trades: int = 50
    min_cash_reserve_pct: float = 0.10

    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()
    order_guards: OrderGuardsConfig = OrderGuardsConfig()


class LLMConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2048
    temperature: float = 0.3
    max_retries: int = 3
    timeout_seconds: int = 30
    max_daily_api_cost_usd: float = 10.0


class AnalysisConfig(BaseModel):
    full_analysis_interval_minutes: int = 15
    quick_scan_interval_minutes: int = 5
    news_check_interval_minutes: int = 30


class SignalsConfig(BaseModel):
    min_confidence: float = 0.70
    required_confirmations: int = 2
    max_signal_age_seconds: int = 300


class UniverseConfig(BaseModel):
    watchlist: list[str] = Field(default_factory=lambda: [
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA",
        "GOOGL", "AMZN", "META", "TSLA", "AMD",
    ])
    scan_for_new: bool = False
    max_universe_size: int = 25


class StrategyConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    signals: SignalsConfig = SignalsConfig()
    universe: UniverseConfig = UniverseConfig()


class PortfolioConfig(BaseModel):
    starting_capital: float = 100000.0
    benchmark: str = "SPY"
    rebalance_check_interval_hours: int = 4


class DashboardConfig(BaseModel):
    enabled: bool = True
    port: int = 8080
    auth_enabled: bool = False
    update_interval_ms: int = 1000


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/klawtrade.log"
    max_size_mb: int = 100
    rotation: str = "daily"
    log_all_signals: bool = True
    log_all_risk_checks: bool = True


class KlawTradeConfig(BaseModel):
    system: SystemConfig = SystemConfig()
    broker: BrokerConfig = BrokerConfig()
    risk: RiskConfig = RiskConfig()
    strategy: StrategyConfig = StrategyConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    dashboard: DashboardConfig = DashboardConfig()
    logging: LoggingConfig = LoggingConfig()


def load_config(config_path: Optional[Path] = None) -> KlawTradeConfig:
    """Load and validate configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

    if not config_path.exists():
        return KlawTradeConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    return KlawTradeConfig(**raw)
