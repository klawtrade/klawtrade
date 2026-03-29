"""Abstract base class for market data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.storage.models import MarketSnapshot


class DataProvider(ABC):
    """Interface that all market data providers must implement."""

    @abstractmethod
    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        """Return a point-in-time market snapshot for *symbol*."""
        ...

    @abstractmethod
    async def get_bars(
        self, symbol: str, timeframe: str, limit: int
    ) -> pd.DataFrame:
        """Return OHLCV bars as a DataFrame.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"AAPL"``).
        timeframe:
            Bar size such as ``"1m"``, ``"5m"``, ``"1h"``, ``"1d"``.
        limit:
            Maximum number of bars to return.

        Returns
        -------
        pd.DataFrame
            Columns: ``open, high, low, close, volume`` indexed by datetime.
        """
        ...

    @abstractmethod
    async def get_account_info(self) -> dict:
        """Return account information (cash, equity, buying power, etc.)."""
        ...
