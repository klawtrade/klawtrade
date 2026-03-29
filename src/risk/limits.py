"""Hard limit definitions loaded from config."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskLimits:
    """Immutable risk limits — loaded once from config, never modified at runtime."""
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

    min_volume_threshold: int = 100000
    max_spread_pct: float = 0.02
    blacklisted_symbols: tuple[str, ...] = ()

    min_confidence: float = 0.70
    max_signal_age_seconds: int = 300

    consecutive_losses_halt: int = 5
    halt_duration_minutes: int = 60
    vix_threshold: float = 35.0


def limits_from_config(risk_cfg: "RiskConfig", strategy_cfg: "StrategyConfig") -> RiskLimits:  # noqa: F821
    """Build RiskLimits from config objects."""
    return RiskLimits(
        max_portfolio_allocation=risk_cfg.max_portfolio_allocation,
        max_single_position_pct=risk_cfg.max_single_position_pct,
        max_sector_allocation_pct=risk_cfg.max_sector_allocation_pct,
        max_correlated_exposure_pct=risk_cfg.max_correlated_exposure_pct,
        max_daily_loss_pct=risk_cfg.max_daily_loss_pct,
        max_weekly_loss_pct=risk_cfg.max_weekly_loss_pct,
        max_drawdown_pct=risk_cfg.max_drawdown_pct,
        max_single_trade_loss_pct=risk_cfg.max_single_trade_loss_pct,
        max_open_positions=risk_cfg.max_open_positions,
        max_daily_trades=risk_cfg.max_daily_trades,
        min_cash_reserve_pct=risk_cfg.min_cash_reserve_pct,
        min_volume_threshold=risk_cfg.order_guards.min_volume_threshold,
        max_spread_pct=risk_cfg.order_guards.max_spread_pct,
        blacklisted_symbols=tuple(risk_cfg.order_guards.blacklisted_symbols),
        min_confidence=strategy_cfg.signals.min_confidence,
        max_signal_age_seconds=strategy_cfg.signals.max_signal_age_seconds,
        consecutive_losses_halt=risk_cfg.circuit_breaker.consecutive_losses_halt,
        halt_duration_minutes=risk_cfg.circuit_breaker.halt_duration_minutes,
        vix_threshold=risk_cfg.circuit_breaker.vix_threshold,
    )
