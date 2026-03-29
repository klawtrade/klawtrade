"""Aggregator that combines a data provider with technical analysis."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

from src.market_data.provider import DataProvider
from src.market_data.technical import calculate_indicators
from src.storage.models import MarketSnapshot

logger = logging.getLogger(__name__)


class MarketDataAggregator:
    """Build complete :class:`MarketSnapshot` objects with indicators.

    The aggregator fetches a raw snapshot from the data provider, pulls
    historical bars, computes technical indicators, and merges
    everything into a single snapshot.

    Parameters
    ----------
    provider:
        Any concrete :class:`DataProvider` implementation.
    bar_timeframe:
        Timeframe string passed to ``provider.get_bars``.
    bar_limit:
        Number of historical bars to request (more bars = more accurate
        long-period indicators like SMA-200).
    """

    def __init__(
        self,
        provider: DataProvider,
        bar_timeframe: str = "1d",
        bar_limit: int = 250,
    ) -> None:
        self._provider = provider
        self._bar_timeframe = bar_timeframe
        self._bar_limit = bar_limit

    @property
    def provider(self) -> DataProvider:
        return self._provider

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        """Return a fully-enriched snapshot for *symbol*.

        Steps:
        1. Fetch a raw price snapshot from the provider.
        2. Fetch OHLCV bars for the symbol.
        3. Calculate technical indicators from the bars.
        4. Merge indicator values into the snapshot.
        """
        snapshot = await self._provider.get_snapshot(symbol)

        try:
            bars = await self._provider.get_bars(
                symbol, self._bar_timeframe, self._bar_limit
            )
            indicators = calculate_indicators(bars)
            snapshot = replace(snapshot, **{
                k: v for k, v in indicators.items() if v is not None
            })
        except Exception:
            logger.exception(
                "Failed to compute indicators for %s; returning raw snapshot",
                symbol,
            )

        return snapshot

    async def get_watchlist_snapshots(
        self, watchlist: list[str]
    ) -> dict[str, MarketSnapshot]:
        """Return enriched snapshots for every symbol in *watchlist*.

        Symbols that fail are logged and skipped — they will not appear
        in the returned dict.
        """
        snapshots: dict[str, MarketSnapshot] = {}
        for symbol in watchlist:
            try:
                snapshots[symbol] = await self.get_snapshot(symbol)
            except Exception:
                logger.exception(
                    "Failed to get snapshot for %s; skipping", symbol
                )
        return snapshots

    async def get_account_info(self) -> dict:
        """Proxy through to the underlying provider."""
        return await self._provider.get_account_info()
