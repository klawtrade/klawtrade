"""Tests for position sizer."""

from datetime import datetime, timezone

import pytest

from src.risk.limits import RiskLimits
from src.risk.position_sizer import PositionSizer
from src.storage.models import (
    PortfolioState,
    Position,
    SignalAction,
    TradeSignal,
)


def _limits(**overrides) -> RiskLimits:
    defaults = dict(
        max_single_position_pct=0.10,
        min_cash_reserve_pct=0.10,
        max_single_trade_loss_pct=0.02,
    )
    defaults.update(overrides)
    return RiskLimits(**defaults)


def _portfolio(cash: float = 50000.0, equity: float = 100000.0, positions=None) -> PortfolioState:
    return PortfolioState(
        timestamp=datetime.now(timezone.utc),
        cash=cash,
        total_equity=equity,
        positions=positions or [],
    )


def _signal(action=SignalAction.BUY, quantity=100, stop_loss=None) -> TradeSignal:
    return TradeSignal(
        symbol="AAPL",
        action=action,
        confidence=0.85,
        suggested_quantity=quantity,
        strategy_name="test",
        stop_loss_price=stop_loss,
    )


class TestBuyQuantity:
    def test_respects_suggested_quantity(self):
        ps = PositionSizer(_limits(max_single_position_pct=1.0))
        qty = ps.calculate_quantity(_signal(quantity=5), _portfolio(), 200.0)
        assert qty == 5

    def test_limited_by_position_pct(self):
        ps = PositionSizer(_limits(max_single_position_pct=0.05))
        # Max = 5% of $100k = $5k / $200 = 25 shares
        qty = ps.calculate_quantity(_signal(quantity=100), _portfolio(), 200.0)
        assert qty == 25

    def test_limited_by_cash_reserve(self):
        ps = PositionSizer(_limits(max_single_position_pct=1.0, min_cash_reserve_pct=0.10))
        port = _portfolio(cash=15000.0, equity=100000.0)
        # Available = $15k - $10k reserve = $5k / $200 = 25 shares
        qty = ps.calculate_quantity(_signal(quantity=100), port, 200.0)
        assert qty == 25

    def test_limited_by_stop_loss_risk(self):
        ps = PositionSizer(_limits(max_single_trade_loss_pct=0.02, max_single_position_pct=1.0))
        # Max risk = 2% of $100k = $2,000. Stop at $195, price $200 = $5/share risk
        # Max by risk = $2,000 / $5 = 400 shares
        # But also limited by cash: $50k - $10k reserve = $40k / $200 = 200
        sig = _signal(quantity=1000, stop_loss=195.0)
        qty = ps.calculate_quantity(sig, _portfolio(), 200.0)
        assert qty == 200

    def test_zero_price(self):
        ps = PositionSizer(_limits())
        qty = ps.calculate_quantity(_signal(), _portfolio(), 0.0)
        assert qty == 0


class TestSellQuantity:
    def test_sell_limited_to_held_quantity(self):
        ps = PositionSizer(_limits())
        positions = [
            Position(
                symbol="AAPL", quantity=30, avg_entry_price=190.0,
                current_price=200.0, market_value=6000.0,
                unrealized_pnl=300.0, unrealized_pnl_pct=5.0,
                opened_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
            ),
        ]
        port = _portfolio(positions=positions)
        sig = _signal(action=SignalAction.SELL, quantity=100)
        qty = ps.calculate_quantity(sig, port, 200.0)
        assert qty == 30

    def test_sell_no_position(self):
        ps = PositionSizer(_limits())
        sig = _signal(action=SignalAction.SELL, quantity=10)
        qty = ps.calculate_quantity(sig, _portfolio(), 200.0)
        assert qty == 0

    def test_close_limited_to_held(self):
        ps = PositionSizer(_limits())
        positions = [
            Position(
                symbol="AAPL", quantity=20, avg_entry_price=190.0,
                current_price=200.0, market_value=4000.0,
                unrealized_pnl=200.0, unrealized_pnl_pct=5.0,
                opened_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
            ),
        ]
        port = _portfolio(positions=positions)
        sig = _signal(action=SignalAction.CLOSE, quantity=50)
        qty = ps.calculate_quantity(sig, port, 200.0)
        assert qty == 20


class TestKellyCriterion:
    def test_positive_edge(self):
        ps = PositionSizer(_limits(max_single_position_pct=0.10))
        # 60% win rate, avg win $200, avg loss $100
        kelly = ps.kelly_criterion(0.60, 200.0, 100.0)
        assert 0 < kelly <= 0.10  # Capped at max position pct

    def test_no_edge(self):
        ps = PositionSizer(_limits())
        # 50% win rate, equal win/loss
        kelly = ps.kelly_criterion(0.50, 100.0, 100.0)
        assert kelly == 0.0

    def test_zero_avg_loss(self):
        ps = PositionSizer(_limits())
        kelly = ps.kelly_criterion(0.60, 200.0, 0.0)
        assert kelly == 0.0

    def test_zero_win_rate(self):
        ps = PositionSizer(_limits())
        kelly = ps.kelly_criterion(0.0, 200.0, 100.0)
        assert kelly == 0.0

    def test_100_pct_win_rate(self):
        ps = PositionSizer(_limits())
        kelly = ps.kelly_criterion(1.0, 200.0, 100.0)
        assert kelly == 0.0

    def test_negative_edge_returns_zero(self):
        ps = PositionSizer(_limits())
        # 30% win rate, equal win/loss -> negative Kelly
        kelly = ps.kelly_criterion(0.30, 100.0, 100.0)
        assert kelly == 0.0
