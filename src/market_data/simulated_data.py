"""Simulated market data provider for testing without API keys."""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.market_data.provider import DataProvider
from src.storage.models import MarketSnapshot

logger = logging.getLogger(__name__)

# Plausible starting prices (approximate real-world levels).
DEFAULT_STARTING_PRICES: dict[str, float] = {
    "SPY": 530.0,
    "QQQ": 460.0,
    "AAPL": 200.0,
    "MSFT": 400.0,
    "NVDA": 130.0,
    "GOOGL": 175.0,
    "AMZN": 190.0,
    "META": 500.0,
    "TSLA": 250.0,
    "AMD": 160.0,
    "NFLX": 650.0,
    "JPM": 200.0,
    "V": 280.0,
    "DIS": 110.0,
    "BA": 190.0,
}

# Typical daily volume ranges (shares).
DEFAULT_VOLUME_RANGES: dict[str, tuple[int, int]] = {
    "SPY": (50_000_000, 120_000_000),
    "QQQ": (30_000_000, 80_000_000),
    "AAPL": (40_000_000, 100_000_000),
    "MSFT": (15_000_000, 40_000_000),
    "NVDA": (20_000_000, 60_000_000),
    "GOOGL": (15_000_000, 35_000_000),
    "AMZN": (10_000_000, 30_000_000),
    "META": (10_000_000, 25_000_000),
    "TSLA": (50_000_000, 150_000_000),
    "AMD": (30_000_000, 80_000_000),
}

_DEFAULT_VOLUME_RANGE: tuple[int, int] = (100_000, 10_000_000)

# Spread as a fraction of price for each liquidity tier.
_SPREAD_BPS: dict[str, float] = {
    "SPY": 0.0001,
    "QQQ": 0.0001,
    "AAPL": 0.0002,
    "MSFT": 0.0002,
    "NVDA": 0.0003,
}

_DEFAULT_SPREAD_BPS: float = 0.0005


class SimulatedDataProvider(DataProvider):
    """Generate realistic price data via seeded random walks.

    Prices are deterministic for a given *seed*, making backtests
    reproducible.

    Parameters
    ----------
    starting_prices:
        Mapping of symbol -> initial price.  Falls back to
        ``DEFAULT_STARTING_PRICES`` for known symbols, or 100.0 for
        unknowns.
    seed:
        RNG seed for reproducibility.
    starting_capital:
        Starting capital reported by :meth:`get_account_info`.
    tick_std:
        Standard deviation of per-tick Gaussian returns (fraction).
    """

    def __init__(
        self,
        starting_prices: Optional[dict[str, float]] = None,
        seed: int = 42,
        starting_capital: float = 100_000.0,
        tick_std: float = 0.001,
    ) -> None:
        self._base_prices: dict[str, float] = {
            **DEFAULT_STARTING_PRICES,
            **(starting_prices or {}),
        }
        self._seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._starting_capital = starting_capital
        self._tick_std = tick_std

        # Track the *current* simulated price per symbol so successive
        # calls form a coherent random walk.
        self._current_prices: dict[str, float] = {}
        # Track how many ticks have elapsed per symbol (used as a
        # secondary seed for bar generation).
        self._tick_count: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_symbol(self, symbol: str) -> float:
        """Initialise tracking state for *symbol* if needed and return
        the current price."""
        if symbol not in self._current_prices:
            self._current_prices[symbol] = self._base_prices.get(symbol, 100.0)
            self._tick_count[symbol] = 0
        return self._current_prices[symbol]

    def _advance_price(self, symbol: str) -> float:
        """Move the price by one tick and return the new price."""
        price = self._ensure_symbol(symbol)
        ret = self._rng.gauss(0, self._tick_std)
        new_price = round(price * (1.0 + ret), 4)
        # Prevent prices from going to zero or negative.
        new_price = max(new_price, 0.01)
        self._current_prices[symbol] = new_price
        self._tick_count[symbol] = self._tick_count.get(symbol, 0) + 1
        return new_price

    def _spread(self, symbol: str) -> float:
        return _SPREAD_BPS.get(symbol, _DEFAULT_SPREAD_BPS)

    def _volume(self, symbol: str) -> int:
        lo, hi = DEFAULT_VOLUME_RANGES.get(symbol, _DEFAULT_VOLUME_RANGE)
        return self._rng.randint(lo, hi)

    # ------------------------------------------------------------------
    # DataProvider interface
    # ------------------------------------------------------------------

    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        price = self._advance_price(symbol)
        half_spread = price * self._spread(symbol)
        bid = round(price - half_spread, 4)
        ask = round(price + half_spread, 4)

        volume = self._volume(symbol)
        daily_volume = volume * self._rng.randint(3, 8)

        # Simulate a plausible daily change from the base price.
        base = self._base_prices.get(symbol, 100.0)
        daily_change_pct = round((price - base) / base * 100, 4)

        # Simple VWAP estimate: midpoint biased slightly toward the
        # close.
        vwap = round((bid + ask + price) / 3, 4)

        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            price=price,
            bid=bid,
            ask=ask,
            volume=volume,
            daily_volume=daily_volume,
            vwap=vwap,
            daily_change_pct=daily_change_pct,
        )

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        limit: int = 200,
    ) -> pd.DataFrame:
        """Generate OHLCV bars using a deterministic random walk.

        The walk is seeded per-symbol so that repeated calls with the
        same parameters always return the same data.
        """
        self._ensure_symbol(symbol)

        # Derive a per-call seed so that bar history is stable.
        call_seed = hash((self._seed, symbol, timeframe, limit)) & 0xFFFFFFFF
        rng = np.random.default_rng(call_seed)

        base_price = self._base_prices.get(symbol, 100.0)

        # Build the series of close prices.
        returns = rng.normal(0, self._tick_std, size=limit)
        cum_returns = np.cumsum(returns)
        closes = base_price * np.exp(cum_returns)
        closes = np.maximum(closes, 0.01)

        # Derive OHLV from close prices.
        intraday_noise = rng.uniform(0.001, 0.008, size=limit)
        highs = closes * (1 + intraday_noise)
        lows = closes * (1 - intraday_noise)
        # Opens: previous close with small gap.
        opens = np.empty_like(closes)
        opens[0] = base_price
        opens[1:] = closes[:-1] * (1 + rng.normal(0, 0.001, size=limit - 1))

        vol_lo, vol_hi = DEFAULT_VOLUME_RANGES.get(symbol, _DEFAULT_VOLUME_RANGE)
        volumes = rng.integers(vol_lo, vol_hi, size=limit)

        # Build a datetime index.
        tf_deltas: dict[str, timedelta] = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
        }
        delta = tf_deltas.get(timeframe, timedelta(days=1))
        end = datetime.now(timezone.utc)
        index = pd.date_range(end=end, periods=limit, freq=delta)

        df = pd.DataFrame(
            {
                "open": np.round(opens, 4),
                "high": np.round(highs, 4),
                "low": np.round(lows, 4),
                "close": np.round(closes, 4),
                "volume": volumes,
            },
            index=index,
        )
        df.index.name = "timestamp"
        return df

    async def get_account_info(self) -> dict:
        return {
            "cash": self._starting_capital,
            "equity": self._starting_capital,
            "buying_power": self._starting_capital * 2,
            "currency": "USD",
            "status": "ACTIVE",
            "provider": "simulated",
        }
