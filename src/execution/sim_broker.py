"""Simulated broker for paper trading / running without API keys.

Fills limit orders immediately if the limit price is at or better than the
current price, applying small random slippage (0-0.1%).  Tracks positions,
fills, and P&L in memory.  Thread-safe via a reentrant lock.
"""

from __future__ import annotations

import logging
import random
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.execution.broker import Broker
from src.storage.models import Order, OrderStatus, Position

logger = logging.getLogger(__name__)


class SimBroker(Broker):
    """In-memory simulated broker."""

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        slippage_max_pct: float = 0.001,
    ) -> None:
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._slippage_max_pct = slippage_max_pct

        # symbol -> _SimPosition
        self._positions: dict[str, _SimPosition] = {}
        # broker_order_id -> Order
        self._orders: dict[str, Order] = {}
        # Completed fills for audit trail
        self._fills: list[dict] = []
        self._realized_pnl: float = 0.0

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Broker interface
    # ------------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """Submit a limit order.  Attempts immediate fill."""
        with self._lock:
            broker_id = str(uuid.uuid4())
            order.broker_order_id = broker_id
            order.status = OrderStatus.SUBMITTED

            self._orders[broker_id] = order

            # Attempt immediate fill for limit orders
            filled = self._try_fill(order)
            if filled:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PENDING

            logger.info(
                "SimBroker order %s: %s %d %s @ %.4f -> %s",
                broker_id,
                order.side,
                order.quantity,
                order.symbol,
                order.limit_price,
                order.status.value,
            )
            return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return False
            if order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                order.status = OrderStatus.CANCELLED
                logger.info("SimBroker cancelled order %s", order_id)
                return True
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current order status."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return OrderStatus.REJECTED
            return order.status

    async def get_positions(self) -> list[Position]:
        """Return all open positions."""
        with self._lock:
            now = datetime.now(timezone.utc)
            positions: list[Position] = []
            for sym, sim_pos in self._positions.items():
                if sim_pos.quantity == 0:
                    continue
                market_value = sim_pos.quantity * sim_pos.current_price
                cost_basis = sim_pos.quantity * sim_pos.avg_entry_price
                unrealized = market_value - cost_basis
                unrealized_pct = (unrealized / cost_basis) if cost_basis != 0 else 0.0
                positions.append(
                    Position(
                        symbol=sym,
                        quantity=sim_pos.quantity,
                        avg_entry_price=sim_pos.avg_entry_price,
                        current_price=sim_pos.current_price,
                        market_value=market_value,
                        unrealized_pnl=unrealized,
                        unrealized_pnl_pct=unrealized_pct,
                        opened_at=sim_pos.opened_at,
                        last_updated=now,
                    )
                )
            return positions

    async def get_account(self) -> dict:
        """Return account summary."""
        with self._lock:
            positions_value = sum(
                p.quantity * p.current_price
                for p in self._positions.values()
                if p.quantity > 0
            )
            total_equity = self._cash + positions_value
            return {
                "cash": self._cash,
                "positions_value": positions_value,
                "total_equity": total_equity,
                "initial_cash": self._initial_cash,
                "realized_pnl": self._realized_pnl,
                "total_fills": len(self._fills),
            }

    # ------------------------------------------------------------------
    # Price feed (call this to update simulated market prices)
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float) -> None:
        """Update the current price for a symbol (for P&L tracking)."""
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].current_price = price

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_fill(self, order: Order) -> bool:
        """Attempt to fill a limit order immediately.

        A buy limit fills if the limit_price >= current best ask (we simulate
        with the limit price itself since we don't have a real order book).
        A sell limit fills if the limit_price <= current best bid.

        Applies random slippage of 0 to slippage_max_pct.
        """
        slippage_pct = random.uniform(0, self._slippage_max_pct)

        if order.side == "buy":
            # Slippage works against us on buys (price slightly higher)
            fill_price = order.limit_price * (1.0 + slippage_pct)
            cost = fill_price * order.quantity

            if cost > self._cash:
                order.status = OrderStatus.REJECTED
                return False

            self._cash -= cost
            self._add_to_position(order.symbol, order.quantity, fill_price)

        elif order.side == "sell":
            # Slippage works against us on sells (price slightly lower)
            fill_price = order.limit_price * (1.0 - slippage_pct)

            pos = self._positions.get(order.symbol)
            if pos is None or pos.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                return False

            proceeds = fill_price * order.quantity
            self._cash += proceeds
            self._remove_from_position(order.symbol, order.quantity, fill_price)
        else:
            return False

        now = datetime.now(timezone.utc)
        order.filled_at = now
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.slippage = abs(fill_price - order.limit_price)

        self._fills.append({
            "broker_order_id": order.broker_order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "limit_price": order.limit_price,
            "fill_price": fill_price,
            "slippage": order.slippage,
            "timestamp": now.isoformat(),
        })

        return True

    def _add_to_position(self, symbol: str, quantity: int, price: float) -> None:
        """Add shares to a position, updating average entry price."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            total_cost = pos.avg_entry_price * pos.quantity + price * quantity
            pos.quantity += quantity
            pos.avg_entry_price = total_cost / pos.quantity if pos.quantity > 0 else 0.0
            pos.current_price = price
        else:
            self._positions[symbol] = _SimPosition(
                quantity=quantity,
                avg_entry_price=price,
                current_price=price,
                opened_at=datetime.now(timezone.utc),
            )

    def _remove_from_position(self, symbol: str, quantity: int, price: float) -> None:
        """Remove shares from a position and realise P&L."""
        pos = self._positions.get(symbol)
        if pos is None:
            return

        pnl = (price - pos.avg_entry_price) * quantity
        self._realized_pnl += pnl

        pos.quantity -= quantity
        pos.current_price = price

        if pos.quantity <= 0:
            del self._positions[symbol]


class _SimPosition:
    """Internal position tracker for SimBroker."""

    __slots__ = ("quantity", "avg_entry_price", "current_price", "opened_at")

    def __init__(
        self,
        quantity: int,
        avg_entry_price: float,
        current_price: float,
        opened_at: datetime,
    ) -> None:
        self.quantity = quantity
        self.avg_entry_price = avg_entry_price
        self.current_price = current_price
        self.opened_at = opened_at
