"""Alpaca broker for real paper (or live) trading.

Requires ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.
Uses the alpaca-py SDK to submit limit orders to Alpaca's paper trading API.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus as AlpacaOrderStatus, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, GetOrdersRequest

from src.execution.broker import Broker
from src.storage.models import Order, OrderStatus, Position

logger = logging.getLogger(__name__)

# Map Alpaca order status -> our OrderStatus
_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PARTIAL,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "replaced": OrderStatus.CANCELLED,
    "pending_cancel": OrderStatus.PENDING,
    "pending_replace": OrderStatus.PENDING,
    "rejected": OrderStatus.REJECTED,
    "stopped": OrderStatus.CANCELLED,
    "suspended": OrderStatus.PENDING,
}


def alpaca_keys_present() -> bool:
    """Check if Alpaca API keys are configured in the environment."""
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    return bool(key and secret and key != "your_paper_key_here")


class AlpacaBroker(Broker):
    """Real Alpaca paper/live broker using the alpaca-py SDK."""

    def __init__(self, paper: bool = True) -> None:
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set. "
                "Get your keys at https://app.alpaca.markets"
            )

        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        self._paper = paper
        logger.info("AlpacaBroker initialised (paper=%s)", paper)

    async def submit_order(self, order: Order) -> Order:
        """Submit a limit order to Alpaca."""
        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL

        request = LimitOrderRequest(
            symbol=order.symbol,
            qty=order.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=round(order.limit_price, 2),
        )

        try:
            alpaca_order = self._client.submit_order(request)
            order.broker_order_id = str(alpaca_order.id)
            order.status = _STATUS_MAP.get(
                alpaca_order.status.value if hasattr(alpaca_order.status, 'value') else str(alpaca_order.status),
                OrderStatus.SUBMITTED,
            )

            # Check if already filled
            status_str = alpaca_order.status.value if hasattr(alpaca_order.status, 'value') else str(alpaca_order.status)
            if status_str == "filled":
                order.status = OrderStatus.FILLED
                order.filled_at = alpaca_order.filled_at or datetime.now(timezone.utc)
                order.filled_price = float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else order.limit_price
                order.filled_quantity = int(alpaca_order.filled_qty) if alpaca_order.filled_qty else order.quantity

            logger.info(
                "Alpaca order submitted: %s %d %s @ %.2f -> %s (id=%s)",
                order.side, order.quantity, order.symbol,
                order.limit_price, order.status.value, order.broker_order_id,
            )
            return order

        except Exception as e:
            logger.error("Alpaca order submission failed: %s", e)
            order.status = OrderStatus.REJECTED
            return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by its Alpaca order ID."""
        try:
            self._client.cancel_order_by_id(order_id)
            logger.info("Alpaca order %s cancelled", order_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel Alpaca order %s: %s", order_id, e)
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current status of an order from Alpaca."""
        try:
            alpaca_order = self._client.get_order_by_id(order_id)
            status_str = alpaca_order.status.value if hasattr(alpaca_order.status, 'value') else str(alpaca_order.status)
            return _STATUS_MAP.get(status_str, OrderStatus.PENDING)
        except Exception as e:
            logger.error("Failed to get order status for %s: %s", order_id, e)
            return OrderStatus.REJECTED

    async def get_positions(self) -> list[Position]:
        """Return all currently open positions from Alpaca."""
        try:
            alpaca_positions = self._client.get_all_positions()
            now = datetime.now(timezone.utc)
            positions: list[Position] = []

            for ap in alpaca_positions:
                qty = int(ap.qty)
                avg_entry = float(ap.avg_entry_price)
                current = float(ap.current_price)
                market_val = float(ap.market_value)
                unrealized = float(ap.unrealized_pl)
                unrealized_pct = float(ap.unrealized_plpc)

                positions.append(
                    Position(
                        symbol=ap.symbol,
                        quantity=abs(qty),
                        avg_entry_price=avg_entry,
                        current_price=current,
                        market_value=abs(market_val),
                        unrealized_pnl=unrealized,
                        unrealized_pnl_pct=unrealized_pct,
                        opened_at=now,  # Alpaca doesn't expose per-position open time easily
                        last_updated=now,
                    )
                )

            return positions
        except Exception as e:
            logger.error("Failed to get Alpaca positions: %s", e)
            return []

    async def get_account(self) -> dict:
        """Return Alpaca account information."""
        try:
            account = self._client.get_account()
            return {
                "cash": float(account.cash),
                "positions_value": float(account.long_market_value),
                "total_equity": float(account.equity),
                "initial_cash": float(account.last_equity),
                "buying_power": float(account.buying_power),
                "daytrading_buying_power": float(account.daytrading_buying_power) if account.daytrading_buying_power else 0.0,
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked,
            }
        except Exception as e:
            logger.error("Failed to get Alpaca account: %s", e)
            return {
                "cash": 0.0,
                "positions_value": 0.0,
                "total_equity": 0.0,
                "initial_cash": 0.0,
            }
