"""Abstract broker interface for order execution."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.storage.models import Order, OrderStatus, Position


class Broker(ABC):
    """Base class that all broker implementations must extend."""

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """Submit an order to the broker.

        Returns the order with updated status and broker_order_id.
        """
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if cancellation succeeded."""
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Query the current status of an order by its broker order ID."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Return all currently open positions."""
        ...

    @abstractmethod
    async def get_account(self) -> dict:
        """Return account information (cash, equity, buying power, etc.)."""
        ...
