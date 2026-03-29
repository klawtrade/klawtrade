"""Portfolio state manager — tracks positions, P&L, and risk metrics."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.storage.models import Order, OrderStatus, Position, PortfolioState

logger = logging.getLogger(__name__)


class PortfolioStateManager:
    """Maintains an in-memory view of the current portfolio state.

    Updated each heartbeat from broker data and filled orders.
    """

    def __init__(self, starting_capital: float) -> None:
        self._starting_capital = starting_capital
        self._cash = starting_capital
        self._positions: list[Position] = []
        self._peak_equity = starting_capital

        # Daily / weekly tracking (reset by the orchestrator at day/week boundaries)
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._day_start_equity = starting_capital
        self._week_start_equity = starting_capital

        # Trade stats
        self._trades_today = 0
        self._consecutive_losses = 0
        self._total_wins = 0
        self._total_trades = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_from_positions(self, positions: list[Position], cash: float) -> None:
        """Refresh portfolio state from broker-reported positions and cash."""
        self._positions = positions
        self._cash = cash

        equity = self._total_equity()

        # Track peak for drawdown calculation
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Recompute daily / weekly P&L
        self._daily_pnl = equity - self._day_start_equity
        self._weekly_pnl = equity - self._week_start_equity

    def record_trade(self, order: Order, was_win: bool) -> None:
        """Record the outcome of a completed trade."""
        if order.status not in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            return

        self._trades_today += 1
        self._total_trades += 1

        if was_win:
            self._total_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

    def get_state(self) -> PortfolioState:
        """Build a snapshot of the current portfolio state."""
        equity = self._total_equity()
        total_pnl = equity - self._starting_capital
        total_pnl_pct = (total_pnl / self._starting_capital) if self._starting_capital > 0 else 0.0
        daily_pnl_pct = (self._daily_pnl / self._day_start_equity) if self._day_start_equity > 0 else 0.0
        weekly_pnl_pct = (self._weekly_pnl / self._week_start_equity) if self._week_start_equity > 0 else 0.0

        drawdown_pct = 0.0
        if self._peak_equity > 0:
            drawdown_pct = (self._peak_equity - equity) / self._peak_equity

        win_rate = (self._total_wins / self._total_trades) if self._total_trades > 0 else 0.0

        return PortfolioState(
            timestamp=datetime.now(timezone.utc),
            cash=self._cash,
            total_equity=equity,
            positions=list(self._positions),
            daily_pnl=self._daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl=self._weekly_pnl,
            weekly_pnl_pct=weekly_pnl_pct,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            peak_equity=self._peak_equity,
            current_drawdown_pct=drawdown_pct,
            trades_today=self._trades_today,
            consecutive_losses=self._consecutive_losses,
            win_rate=win_rate,
        )

    def reset_daily(self) -> None:
        """Call at the start of a new trading day."""
        self._day_start_equity = self._total_equity()
        self._daily_pnl = 0.0
        self._trades_today = 0

    def reset_weekly(self) -> None:
        """Call at the start of a new trading week."""
        self._week_start_equity = self._total_equity()
        self._weekly_pnl = 0.0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _total_equity(self) -> float:
        """Cash + market value of all positions."""
        position_value = sum(p.market_value for p in self._positions)
        return self._cash + position_value
