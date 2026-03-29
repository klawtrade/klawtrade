"""Order lifecycle management.

Takes approved RiskCheckResult objects, creates limit Orders, submits them
through the broker, tracks status, handles timeouts, and links orders back
to the originating signals.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.execution.broker import Broker
from src.storage.models import (
    Order,
    OrderStatus,
    RiskCheckResult,
    SignalAction,
)

logger = logging.getLogger(__name__)

# Orders that haven't filled after this duration are cancelled.
DEFAULT_ORDER_TIMEOUT = timedelta(minutes=15)


class OrderManager:
    """Creates, submits, and tracks orders through their full lifecycle."""

    def __init__(
        self,
        broker: Broker,
        order_timeout: timedelta = DEFAULT_ORDER_TIMEOUT,
    ) -> None:
        self._broker = broker
        self._order_timeout = order_timeout

        # order_id -> Order
        self._active_orders: dict[str, Order] = {}
        # signal_id -> order_id (for linking)
        self._signal_order_map: dict[str, str] = {}
        # All completed/cancelled orders
        self._completed_orders: list[Order] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_approved(self, result: RiskCheckResult) -> Optional[Order]:
        """Convert an approved RiskCheckResult into a submitted Order.

        Returns the Order object (with broker status updated), or None if
        the result was not approved or order creation failed.
        """
        if not result.approved:
            logger.debug(
                "Skipping non-approved result for %s", result.signal.symbol
            )
            return None

        signal = result.signal
        order = self._create_order(result)

        try:
            submitted_order = await self._broker.submit_order(order)
            self._active_orders[submitted_order.id] = submitted_order
            self._signal_order_map[signal.id] = submitted_order.id

            logger.info(
                "Order %s submitted for signal %s: %s %d %s @ %.4f",
                submitted_order.id,
                signal.id,
                submitted_order.side,
                submitted_order.quantity,
                submitted_order.symbol,
                submitted_order.limit_price,
            )

            # If the broker already filled it (e.g. sim broker), move to completed
            if submitted_order.status == OrderStatus.FILLED:
                self._move_to_completed(submitted_order.id)

            return submitted_order

        except Exception:
            logger.exception(
                "Failed to submit order for signal %s on %s",
                signal.id,
                signal.symbol,
            )
            order.status = OrderStatus.REJECTED
            self._completed_orders.append(order)
            return order

    async def process_batch(
        self, results: list[RiskCheckResult]
    ) -> list[Order]:
        """Process a batch of risk-check results, submitting approved ones.

        Returns list of all Order objects created (regardless of fill status).
        """
        orders: list[Order] = []
        for result in results:
            order = await self.process_approved(result)
            if order is not None:
                orders.append(order)
        return orders

    async def check_timeouts(self) -> list[Order]:
        """Cancel any active orders that have exceeded the timeout.

        Returns list of orders that were cancelled.
        """
        now = datetime.now(timezone.utc)
        cancelled: list[Order] = []

        # Iterate over a copy since we mutate during the loop
        for order_id, order in list(self._active_orders.items()):
            age = now - order.submitted_at
            if age > self._order_timeout:
                try:
                    success = await self._broker.cancel_order(
                        order.broker_order_id or order_id
                    )
                    if success:
                        order.status = OrderStatus.CANCELLED
                        logger.info(
                            "Order %s timed out after %s, cancelled",
                            order_id,
                            age,
                        )
                    else:
                        # Refresh status — might have filled in the meantime
                        order.status = await self._broker.get_order_status(
                            order.broker_order_id or order_id
                        )
                except Exception:
                    logger.exception(
                        "Error cancelling timed-out order %s", order_id
                    )
                    order.status = OrderStatus.CANCELLED

                self._move_to_completed(order_id)
                cancelled.append(order)

        return cancelled

    async def sync_order_statuses(self) -> None:
        """Poll the broker for updated statuses on all active orders."""
        for order_id, order in list(self._active_orders.items()):
            try:
                new_status = await self._broker.get_order_status(
                    order.broker_order_id or order_id
                )
                if new_status != order.status:
                    logger.info(
                        "Order %s status changed: %s -> %s",
                        order_id,
                        order.status.value,
                        new_status.value,
                    )
                    order.status = new_status

                if new_status in (
                    OrderStatus.FILLED,
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                ):
                    self._move_to_completed(order_id)

            except Exception:
                logger.exception(
                    "Failed to sync status for order %s", order_id
                )

    def get_order_for_signal(self, signal_id: str) -> Optional[Order]:
        """Look up the order linked to a given signal ID."""
        order_id = self._signal_order_map.get(signal_id)
        if order_id is None:
            return None
        # Check active first, then completed
        order = self._active_orders.get(order_id)
        if order is not None:
            return order
        for completed in self._completed_orders:
            if completed.id == order_id:
                return completed
        return None

    @property
    def active_order_count(self) -> int:
        return len(self._active_orders)

    @property
    def completed_orders(self) -> list[Order]:
        return list(self._completed_orders)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_order(self, result: RiskCheckResult) -> Order:
        """Build an Order from an approved RiskCheckResult."""
        signal = result.signal

        # Determine side
        if signal.action == SignalAction.BUY:
            side = "buy"
        elif signal.action in (SignalAction.SELL, SignalAction.CLOSE):
            side = "sell"
        else:
            side = "sell"  # Default to sell for unknown actions

        # Use risk-adjusted quantity if available, otherwise signal's suggestion
        quantity = result.adjusted_quantity or signal.suggested_quantity

        # Use risk-adjusted price if available, otherwise signal's suggestion
        limit_price = result.adjusted_limit_price or signal.suggested_limit_price
        if limit_price is None or limit_price <= 0:
            # Fallback — should not happen in practice
            limit_price = 0.0
            logger.warning(
                "No limit price for signal %s on %s", signal.id, signal.symbol
            )

        return Order(
            id=str(uuid.uuid4()),
            signal_id=signal.id,
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            order_type="limit",
            limit_price=limit_price,
            status=OrderStatus.PENDING,
            submitted_at=datetime.now(timezone.utc),
            stop_loss_price=signal.stop_loss_price,
            take_profit_price=signal.take_profit_price,
        )

    def _move_to_completed(self, order_id: str) -> None:
        """Move an order from active to completed tracking."""
        order = self._active_orders.pop(order_id, None)
        if order is not None:
            self._completed_orders.append(order)
