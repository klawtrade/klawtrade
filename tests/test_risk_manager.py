"""Tests for the risk manager — targeting 100% branch coverage."""

from datetime import datetime, timedelta, timezone

import pytest

from src.risk.circuit_breaker import CircuitBreaker
from src.risk.limits import RiskLimits
from src.risk.manager import RiskManager
from src.storage.models import (
    MarketSnapshot,
    PortfolioState,
    Position,
    RiskCheckResult,
    RiskRejectionReason,
    SignalAction,
    TradeSignal,
)


def _limits(**overrides) -> RiskLimits:
    defaults = dict(
        max_portfolio_allocation=0.90,
        max_single_position_pct=0.10,
        max_sector_allocation_pct=0.30,
        max_correlated_exposure_pct=0.40,
        max_daily_loss_pct=0.03,
        max_weekly_loss_pct=0.07,
        max_drawdown_pct=0.15,
        max_single_trade_loss_pct=0.02,
        max_open_positions=15,
        max_daily_trades=50,
        min_cash_reserve_pct=0.10,
        min_volume_threshold=100000,
        max_spread_pct=0.02,
        blacklisted_symbols=("GME", "AMC"),
        min_confidence=0.70,
        max_signal_age_seconds=300,
        consecutive_losses_halt=5,
        halt_duration_minutes=60,
        vix_threshold=35.0,
    )
    defaults.update(overrides)
    return RiskLimits(**defaults)


def _portfolio(**overrides) -> PortfolioState:
    defaults = dict(
        timestamp=datetime.now(timezone.utc),
        cash=50000.0,
        total_equity=100000.0,
        positions=[],
        daily_pnl=0.0,
        daily_pnl_pct=0.0,
        weekly_pnl=0.0,
        weekly_pnl_pct=0.0,
        total_pnl=0.0,
        total_pnl_pct=0.0,
        peak_equity=100000.0,
        current_drawdown_pct=0.0,
        trades_today=0,
        consecutive_losses=0,
        win_rate=0.5,
    )
    defaults.update(overrides)
    return PortfolioState(**defaults)


def _signal(
    symbol: str = "AAPL",
    action: SignalAction = SignalAction.BUY,
    confidence: float = 0.85,
    quantity: int = 10,
    limit_price: float = 200.0,
    stop_loss: float | None = 195.0,
    take_profit: float | None = 210.0,
    expires_in_seconds: int = 300,
) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        suggested_quantity=quantity,
        strategy_name="test",
        confirming_indicators=["RSI", "MACD"],
        suggested_limit_price=limit_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        reasoning="test signal",
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds),
    )


def _snapshot(
    symbol: str = "AAPL",
    price: float = 200.0,
    bid: float = 199.90,
    ask: float = 200.10,
    daily_volume: int = 500000,
    sector: str | None = None,
    correlation_group: str | None = None,
) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        price=price,
        bid=bid,
        ask=ask,
        volume=1000,
        daily_volume=daily_volume,
        vwap=price,
        daily_change_pct=0.5,
        sector=sector,
        correlation_group=correlation_group,
    )


def _make_rm(limits: RiskLimits | None = None) -> RiskManager:
    lim = limits or _limits()
    cb = CircuitBreaker(lim)
    return RiskManager(lim, cb)


class TestBasicApproval:
    def test_valid_signal_approved(self):
        rm = _make_rm()
        result = rm.check(_signal(), _portfolio(), _snapshot())
        assert result.approved is True
        assert result.rejection_reasons == []
        assert result.adjusted_quantity is not None
        assert result.adjusted_quantity > 0

    def test_sell_signal_approved(self):
        rm = _make_rm()
        port = _portfolio(positions=[
            Position(
                symbol="AAPL", quantity=50, avg_entry_price=190.0,
                current_price=200.0, market_value=10000.0,
                unrealized_pnl=500.0, unrealized_pnl_pct=5.0,
                opened_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
            ),
        ])
        sig = _signal(action=SignalAction.SELL, quantity=10)
        result = rm.check(sig, port, _snapshot())
        assert result.approved is True

    def test_hold_signal_approved(self):
        rm = _make_rm()
        sig = _signal(action=SignalAction.HOLD)
        result = rm.check(sig, _portfolio())
        assert result.approved is True

    def test_close_signal_approved(self):
        rm = _make_rm()
        sig = _signal(action=SignalAction.CLOSE)
        result = rm.check(sig, _portfolio())
        assert result.approved is True


class TestBlacklist:
    def test_blacklisted_symbol_rejected(self):
        rm = _make_rm()
        sig = _signal(symbol="GME")
        result = rm.check(sig, _portfolio(), _snapshot(symbol="GME"))
        assert not result.approved
        assert RiskRejectionReason.BLACKLISTED in result.rejection_reasons

    def test_non_blacklisted_passes(self):
        rm = _make_rm()
        sig = _signal(symbol="AAPL")
        result = rm.check(sig, _portfolio(), _snapshot())
        assert RiskRejectionReason.BLACKLISTED not in result.rejection_reasons


class TestCircuitBreaker:
    def test_circuit_breaker_active_rejects(self):
        lim = _limits()
        cb = CircuitBreaker(lim)
        cb.manual_halt()
        rm = RiskManager(lim, cb)
        result = rm.check(_signal(), _portfolio(), _snapshot())
        assert not result.approved
        assert RiskRejectionReason.CIRCUIT_BREAKER in result.rejection_reasons

    def test_circuit_breaker_inactive_passes(self):
        rm = _make_rm()
        result = rm.check(_signal(), _portfolio(), _snapshot())
        assert RiskRejectionReason.CIRCUIT_BREAKER not in result.rejection_reasons


class TestSignalExpiry:
    def test_expired_signal_rejected(self):
        rm = _make_rm()
        sig = _signal(expires_in_seconds=-10)  # already expired
        result = rm.check(sig, _portfolio(), _snapshot())
        assert not result.approved
        assert RiskRejectionReason.SIGNAL_EXPIRED in result.rejection_reasons

    def test_valid_signal_passes(self):
        rm = _make_rm()
        sig = _signal(expires_in_seconds=300)
        result = rm.check(sig, _portfolio(), _snapshot())
        assert RiskRejectionReason.SIGNAL_EXPIRED not in result.rejection_reasons


class TestConfidence:
    def test_low_confidence_rejected(self):
        rm = _make_rm()
        sig = _signal(confidence=0.50)
        result = rm.check(sig, _portfolio(), _snapshot())
        assert not result.approved
        assert RiskRejectionReason.LOW_CONFIDENCE in result.rejection_reasons

    def test_exactly_at_threshold_passes(self):
        rm = _make_rm(_limits(min_confidence=0.70))
        sig = _signal(confidence=0.70)
        result = rm.check(sig, _portfolio(), _snapshot())
        assert RiskRejectionReason.LOW_CONFIDENCE not in result.rejection_reasons

    def test_just_below_threshold_rejected(self):
        rm = _make_rm(_limits(min_confidence=0.70))
        sig = _signal(confidence=0.699)
        result = rm.check(sig, _portfolio(), _snapshot())
        assert RiskRejectionReason.LOW_CONFIDENCE in result.rejection_reasons


class TestDailyTrades:
    def test_max_daily_trades_rejected(self):
        rm = _make_rm(_limits(max_daily_trades=10))
        port = _portfolio(trades_today=10)
        result = rm.check(_signal(), port, _snapshot())
        assert not result.approved
        assert RiskRejectionReason.MAX_DAILY_TRADES in result.rejection_reasons

    def test_below_max_passes(self):
        rm = _make_rm(_limits(max_daily_trades=50))
        port = _portfolio(trades_today=49)
        result = rm.check(_signal(), port, _snapshot())
        assert RiskRejectionReason.MAX_DAILY_TRADES not in result.rejection_reasons


class TestMaxPositions:
    def test_at_max_positions_rejected(self):
        rm = _make_rm(_limits(max_open_positions=2))
        positions = [
            Position(
                symbol=s, quantity=10, avg_entry_price=100.0,
                current_price=100.0, market_value=1000.0,
                unrealized_pnl=0.0, unrealized_pnl_pct=0.0,
                opened_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
            )
            for s in ["MSFT", "GOOGL"]
        ]
        port = _portfolio(positions=positions)
        result = rm.check(_signal(), port, _snapshot())
        assert RiskRejectionReason.MAX_POSITIONS in result.rejection_reasons

    def test_sell_at_max_positions_allowed(self):
        """Sells should still work even at max positions."""
        rm = _make_rm(_limits(max_open_positions=1))
        positions = [
            Position(
                symbol="AAPL", quantity=50, avg_entry_price=190.0,
                current_price=200.0, market_value=10000.0,
                unrealized_pnl=500.0, unrealized_pnl_pct=5.0,
                opened_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
            ),
        ]
        port = _portfolio(positions=positions)
        sig = _signal(action=SignalAction.SELL, quantity=10)
        result = rm.check(sig, port, _snapshot())
        # Sell doesn't check max positions
        assert RiskRejectionReason.MAX_POSITIONS not in result.rejection_reasons


class TestVolume:
    def test_low_volume_rejected(self):
        rm = _make_rm(_limits(min_volume_threshold=500000))
        snap = _snapshot(daily_volume=100000)
        result = rm.check(_signal(), _portfolio(), snap)
        assert RiskRejectionReason.LOW_VOLUME in result.rejection_reasons

    def test_sufficient_volume_passes(self):
        rm = _make_rm()
        snap = _snapshot(daily_volume=1000000)
        result = rm.check(_signal(), _portfolio(), snap)
        assert RiskRejectionReason.LOW_VOLUME not in result.rejection_reasons

    def test_no_snapshot_skips_volume_check(self):
        rm = _make_rm()
        result = rm.check(_signal(), _portfolio(), None)
        assert RiskRejectionReason.LOW_VOLUME not in result.rejection_reasons


class TestSpread:
    def test_wide_spread_rejected(self):
        rm = _make_rm(_limits(max_spread_pct=0.01))
        snap = _snapshot(bid=195.0, ask=200.0)  # 2.5% spread
        result = rm.check(_signal(), _portfolio(), snap)
        assert RiskRejectionReason.WIDE_SPREAD in result.rejection_reasons

    def test_tight_spread_passes(self):
        rm = _make_rm()
        snap = _snapshot(bid=199.95, ask=200.05)  # 0.05% spread
        result = rm.check(_signal(), _portfolio(), snap)
        assert RiskRejectionReason.WIDE_SPREAD not in result.rejection_reasons

    def test_zero_bid_skips_spread(self):
        rm = _make_rm()
        snap = _snapshot(bid=0, ask=200.0)
        result = rm.check(_signal(), _portfolio(), snap)
        assert RiskRejectionReason.WIDE_SPREAD not in result.rejection_reasons

    def test_zero_ask_skips_spread(self):
        rm = _make_rm()
        snap = _snapshot(bid=200.0, ask=0)
        result = rm.check(_signal(), _portfolio(), snap)
        assert RiskRejectionReason.WIDE_SPREAD not in result.rejection_reasons


class TestPositionSize:
    def test_exceeds_max_position_pct(self):
        rm = _make_rm(_limits(max_single_position_pct=0.05))
        # 100 shares * $200 = $20,000 = 20% of $100k portfolio
        sig = _signal(quantity=100, limit_price=200.0)
        result = rm.check(sig, _portfolio(), _snapshot())
        assert RiskRejectionReason.MAX_POSITION_SIZE in result.rejection_reasons

    def test_existing_position_counts(self):
        rm = _make_rm(_limits(max_single_position_pct=0.10))
        positions = [
            Position(
                symbol="AAPL", quantity=40, avg_entry_price=200.0,
                current_price=200.0, market_value=8000.0,
                unrealized_pnl=0.0, unrealized_pnl_pct=0.0,
                opened_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
            ),
        ]
        port = _portfolio(positions=positions)
        sig = _signal(quantity=20, limit_price=200.0)  # +$4000 = $12k total = 12%
        result = rm.check(sig, port, _snapshot())
        assert RiskRejectionReason.MAX_POSITION_SIZE in result.rejection_reasons

    def test_within_position_limit_passes(self):
        rm = _make_rm(_limits(max_single_position_pct=0.10))
        sig = _signal(quantity=10, limit_price=200.0)  # $2,000 = 2%
        result = rm.check(sig, _portfolio(), _snapshot())
        assert RiskRejectionReason.MAX_POSITION_SIZE not in result.rejection_reasons


class TestSectorAllocation:
    def test_exceeds_sector_limit(self):
        rm = _make_rm(_limits(max_sector_allocation_pct=0.20))
        positions = [
            Position(
                symbol="MSFT", quantity=100, avg_entry_price=150.0,
                current_price=150.0, market_value=15000.0,
                unrealized_pnl=0.0, unrealized_pnl_pct=0.0,
                opened_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
                sector="tech",
            ),
        ]
        port = _portfolio(positions=positions)
        snap = _snapshot(sector="tech")
        sig = _signal(quantity=30, limit_price=200.0)  # +$6,000 tech = $21k = 21%
        result = rm.check(sig, port, snap)
        assert RiskRejectionReason.SECTOR_LIMIT in result.rejection_reasons

    def test_no_sector_info_skips(self):
        rm = _make_rm()
        snap = _snapshot(sector=None)
        result = rm.check(_signal(), _portfolio(), snap)
        assert RiskRejectionReason.SECTOR_LIMIT not in result.rejection_reasons


class TestCorrelatedExposure:
    def test_exceeds_correlation_limit(self):
        rm = _make_rm(_limits(max_correlated_exposure_pct=0.20))
        positions = [
            Position(
                symbol="MSFT", quantity=100, avg_entry_price=150.0,
                current_price=150.0, market_value=15000.0,
                unrealized_pnl=0.0, unrealized_pnl_pct=0.0,
                opened_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
                correlation_group="big_tech",
            ),
        ]
        port = _portfolio(positions=positions)
        snap = _snapshot(correlation_group="big_tech")
        sig = _signal(quantity=30, limit_price=200.0)
        result = rm.check(sig, port, snap)
        assert RiskRejectionReason.CORRELATED_EXPOSURE in result.rejection_reasons

    def test_no_correlation_group_skips(self):
        rm = _make_rm()
        snap = _snapshot(correlation_group=None)
        result = rm.check(_signal(), _portfolio(), snap)
        assert RiskRejectionReason.CORRELATED_EXPOSURE not in result.rejection_reasons


class TestCashReserve:
    def test_insufficient_cash_rejected(self):
        rm = _make_rm(_limits(min_cash_reserve_pct=0.10))
        port = _portfolio(cash=12000.0, total_equity=100000.0)
        sig = _signal(quantity=20, limit_price=200.0)  # costs $4,000, leaves $8k < $10k reserve
        result = rm.check(sig, port, _snapshot())
        assert RiskRejectionReason.INSUFFICIENT_CASH in result.rejection_reasons

    def test_sufficient_cash_passes(self):
        rm = _make_rm()
        port = _portfolio(cash=50000.0)
        sig = _signal(quantity=5, limit_price=200.0)  # costs $1,000
        result = rm.check(sig, port, _snapshot())
        assert RiskRejectionReason.INSUFFICIENT_CASH not in result.rejection_reasons


class TestDailyLoss:
    def test_daily_loss_exceeded_rejected(self):
        rm = _make_rm(_limits(max_daily_loss_pct=0.03))
        port = _portfolio(daily_pnl_pct=-0.03)
        result = rm.check(_signal(), port, _snapshot())
        assert RiskRejectionReason.MAX_DAILY_LOSS in result.rejection_reasons

    def test_daily_loss_within_limit_passes(self):
        rm = _make_rm()
        port = _portfolio(daily_pnl_pct=-0.01)
        result = rm.check(_signal(), port, _snapshot())
        assert RiskRejectionReason.MAX_DAILY_LOSS not in result.rejection_reasons


class TestDrawdown:
    def test_max_drawdown_exceeded_rejected(self):
        rm = _make_rm(_limits(max_drawdown_pct=0.15))
        port = _portfolio(current_drawdown_pct=0.15)
        result = rm.check(_signal(), port, _snapshot())
        assert RiskRejectionReason.MAX_DRAWDOWN in result.rejection_reasons

    def test_drawdown_within_limit_passes(self):
        rm = _make_rm()
        port = _portfolio(current_drawdown_pct=0.05)
        result = rm.check(_signal(), port, _snapshot())
        assert RiskRejectionReason.MAX_DRAWDOWN not in result.rejection_reasons


class TestMultipleRejections:
    def test_collects_all_rejection_reasons(self):
        """Verify we don't short-circuit — all violations are reported."""
        rm = _make_rm(_limits(
            min_confidence=0.90,
            max_daily_trades=5,
            max_open_positions=1,
        ))
        positions = [
            Position(
                symbol="MSFT", quantity=10, avg_entry_price=100.0,
                current_price=100.0, market_value=1000.0,
                unrealized_pnl=0.0, unrealized_pnl_pct=0.0,
                opened_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
            ),
        ]
        port = _portfolio(positions=positions, trades_today=5)
        sig = _signal(symbol="GME", confidence=0.50)
        snap = _snapshot(symbol="GME")
        result = rm.check(sig, port, snap)
        assert not result.approved
        assert RiskRejectionReason.BLACKLISTED in result.rejection_reasons
        assert RiskRejectionReason.LOW_CONFIDENCE in result.rejection_reasons
        assert RiskRejectionReason.MAX_DAILY_TRADES in result.rejection_reasons
        assert RiskRejectionReason.MAX_POSITIONS in result.rejection_reasons


class TestPositionSizing:
    def test_quantity_adjusted_down(self):
        rm = _make_rm(_limits(max_single_position_pct=0.05))
        port = _portfolio(cash=50000.0)
        sig = _signal(quantity=100, limit_price=200.0)  # wants $20k, max is $5k
        # This would exceed position size, so it should be rejected
        result = rm.check(sig, port, _snapshot())
        assert RiskRejectionReason.MAX_POSITION_SIZE in result.rejection_reasons

    def test_adjusted_quantity_zero_means_rejection(self):
        rm = _make_rm(_limits(min_cash_reserve_pct=0.99))
        port = _portfolio(cash=1000.0, total_equity=100000.0)
        sig = _signal(quantity=10, limit_price=200.0)
        result = rm.check(sig, port, _snapshot())
        assert not result.approved


class TestRiskScore:
    def test_clean_signal_low_risk(self):
        rm = _make_rm()
        result = rm.check(_signal(confidence=0.95), _portfolio(), _snapshot())
        assert result.risk_score < 0.3

    def test_rejected_signal_max_risk(self):
        rm = _make_rm()
        sig = _signal(symbol="GME")
        result = rm.check(sig, _portfolio(), _snapshot(symbol="GME"))
        assert result.risk_score == 1.0

    def test_high_drawdown_increases_risk(self):
        rm = _make_rm()
        port = _portfolio(current_drawdown_pct=0.10)
        result = rm.check(_signal(), port, _snapshot())
        assert result.risk_score > 0.15


class TestEdgeCases:
    def test_empty_portfolio(self):
        rm = _make_rm()
        port = _portfolio(cash=100000.0, positions=[])
        result = rm.check(_signal(), port, _snapshot())
        assert result.approved is True

    def test_zero_price_signal(self):
        rm = _make_rm()
        sig = _signal(limit_price=0.0)
        snap = _snapshot(price=0.0)
        result = rm.check(sig, _portfolio(), snap)
        # Should handle gracefully without division by zero
        assert isinstance(result, RiskCheckResult)

    def test_no_snapshot(self):
        rm = _make_rm()
        result = rm.check(_signal(), _portfolio(), None)
        # Should work without market data (skips volume/spread checks)
        assert isinstance(result, RiskCheckResult)

    def test_signal_without_stop_loss(self):
        rm = _make_rm()
        sig = _signal(stop_loss=None)
        result = rm.check(sig, _portfolio(), _snapshot())
        assert isinstance(result, RiskCheckResult)

    def test_zero_equity_portfolio(self):
        rm = _make_rm()
        port = _portfolio(total_equity=0.0, cash=0.0)
        sig = _signal(quantity=1, limit_price=10.0)
        result = rm.check(sig, port, _snapshot(price=10.0))
        assert isinstance(result, RiskCheckResult)
