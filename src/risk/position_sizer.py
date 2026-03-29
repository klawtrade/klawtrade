"""Position sizing — determines how many shares to buy/sell.

Uses fixed fractional sizing by default. Kelly criterion available
for more aggressive sizing when win rate data is sufficient.
"""

from __future__ import annotations

from src.risk.limits import RiskLimits
from src.storage.models import PortfolioState, TradeSignal, SignalAction


class PositionSizer:
    """Calculate position sizes respecting risk limits."""

    def __init__(self, limits: RiskLimits) -> None:
        self._limits = limits

    def calculate_quantity(
        self,
        signal: TradeSignal,
        portfolio: PortfolioState,
        current_price: float,
    ) -> int:
        """Calculate the maximum allowed quantity for a trade.

        Returns the minimum of:
        1. Signal's suggested quantity
        2. Max single position size (% of equity)
        3. Available cash (respecting reserve)
        4. Risk-per-trade limit (based on stop loss distance)
        """
        if signal.action in (SignalAction.SELL, SignalAction.CLOSE):
            # For sells/closes, limit to what we hold
            for pos in portfolio.positions:
                if pos.symbol == signal.symbol:
                    return min(signal.suggested_quantity, pos.quantity)
            return 0

        # Max position value by portfolio percentage
        max_position_value = portfolio.total_equity * self._limits.max_single_position_pct
        max_by_position = int(max_position_value / current_price) if current_price > 0 else 0

        # Available cash after reserving minimum
        min_reserve = portfolio.total_equity * self._limits.min_cash_reserve_pct
        available_cash = max(0.0, portfolio.cash - min_reserve)
        max_by_cash = int(available_cash / current_price) if current_price > 0 else 0

        # Risk-per-trade sizing (if stop loss is set)
        max_by_risk = max_by_position  # default to position limit
        if signal.stop_loss_price and signal.stop_loss_price < current_price:
            risk_per_share = current_price - signal.stop_loss_price
            max_risk_dollars = portfolio.total_equity * self._limits.max_single_trade_loss_pct
            if risk_per_share > 0:
                max_by_risk = int(max_risk_dollars / risk_per_share)

        # Take the minimum of all constraints
        quantity = min(
            signal.suggested_quantity,
            max_by_position,
            max_by_cash,
            max_by_risk,
        )

        return max(0, quantity)

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Calculate Kelly fraction for position sizing.

        Returns a fraction of capital to risk (0.0 to 1.0).
        We use half-Kelly for safety.
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        win_loss_ratio = avg_win / abs(avg_loss)
        kelly = win_rate - (1 - win_rate) / win_loss_ratio

        # Half-Kelly for safety, capped at max position size
        half_kelly = max(0.0, kelly / 2)
        return min(half_kelly, self._limits.max_single_position_pct)
