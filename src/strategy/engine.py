"""Strategy engine — runs all active strategies and funnels signals through risk management."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from src.risk.manager import RiskManager
from src.storage.models import (
    MarketSnapshot,
    PortfolioState,
    RiskCheckResult,
    TradeSignal,
)

logger = logging.getLogger(__name__)


class Strategy(Protocol):
    """Protocol that all strategy modules must satisfy.

    Each strategy module exposes a ``generate_signals`` function.
    """

    def generate_signals(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> list[TradeSignal]: ...


class _ModuleStrategy:
    """Adapter that wraps a strategy *module* (with a module-level
    ``generate_signals`` function) so it satisfies the ``Strategy`` protocol.
    """

    def __init__(self, module: Any) -> None:
        self._fn = module.generate_signals

    def generate_signals(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> list[TradeSignal]:
        return self._fn(snapshot, portfolio)


class StrategyEngine:
    """Runs all active strategies against market data and passes each
    resulting signal through the risk manager.
    """

    def __init__(
        self,
        strategies: list[Any],
        risk_manager: RiskManager,
        config: Any = None,
    ) -> None:
        self._strategies: list[Strategy] = []
        for s in strategies:
            if hasattr(s, "generate_signals") and callable(s.generate_signals):
                # Already satisfies the protocol (class instance)
                self._strategies.append(s)
            elif hasattr(s, "generate_signals") and callable(getattr(s, "generate_signals")):
                # It's a module with a generate_signals function
                self._strategies.append(_ModuleStrategy(s))
            else:
                raise TypeError(
                    f"Strategy {s!r} does not expose a generate_signals callable"
                )

        self._risk_manager = risk_manager
        self._config = config

    @property
    def strategy_count(self) -> int:
        return len(self._strategies)

    def analyze(
        self,
        snapshots: dict[str, MarketSnapshot],
        portfolio: PortfolioState,
    ) -> list[RiskCheckResult]:
        """Run all strategies, collect signals, pass each through risk manager.

        Args:
            snapshots: Mapping of symbol -> current MarketSnapshot.
            portfolio: Current portfolio state.

        Returns:
            List of RiskCheckResult (one per signal generated).
        """
        all_signals: list[TradeSignal] = []

        for strategy in self._strategies:
            for symbol, snapshot in snapshots.items():
                try:
                    signals = strategy.generate_signals(snapshot, portfolio)
                    all_signals.extend(signals)
                except Exception:
                    logger.exception(
                        "Strategy %r failed on %s",
                        strategy,
                        symbol,
                    )

        logger.info(
            "Collected %d signals from %d strategies across %d symbols",
            len(all_signals),
            len(self._strategies),
            len(snapshots),
        )

        results: list[RiskCheckResult] = []
        for signal in all_signals:
            snapshot = snapshots.get(signal.symbol)
            try:
                result = self._risk_manager.check(signal, portfolio, snapshot)
                results.append(result)
            except Exception:
                logger.exception(
                    "Risk check failed for signal %s on %s",
                    signal.id,
                    signal.symbol,
                )

        approved_count = sum(1 for r in results if r.approved)
        logger.info(
            "Risk check complete: %d/%d signals approved",
            approved_count,
            len(results),
        )

        return results
