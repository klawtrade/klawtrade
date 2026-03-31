"""Main orchestrator loop for the KlawTrade trading system."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

import uvicorn
from dotenv import load_dotenv

from src.config import KlawTradeConfig, load_config
from src.dashboard.app import Dashboard
from src.execution import create_broker, SUPPORTED_BROKERS
from src.execution.order_manager import OrderManager
from src.execution.sim_broker import SimBroker
from src.market_data.aggregator import MarketDataAggregator
from src.market_data.simulated_data import SimulatedDataProvider
from src.portfolio.state import PortfolioStateManager
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.limits import RiskLimits, limits_from_config
from src.risk.manager import RiskManager
from src.storage.database import Database
from src.storage.models import (
    MarketSnapshot,
    Order,
    OrderStatus,
    PortfolioState,
    Position,
    TradeSignal,
)
from src.strategy.engine import StrategyEngine
from src.strategy.rules import momentum, mean_reversion
from src.utils.logging import setup_logging
from src.utils.time_utils import is_market_open, time_until_market_open

logger = logging.getLogger(__name__)


class KlawTrade:
    """Top-level orchestrator — the heartbeat of the system."""

    def __init__(self, config: Optional[KlawTradeConfig] = None) -> None:
        config_path = os.environ.get("KLAWTRADE_CONFIG")
        if config is not None:
            self.config = config
        elif config_path:
            from pathlib import Path
            self.config = load_config(Path(config_path))
        else:
            self.config = load_config()

        # CLI overrides
        port_override = os.environ.get("KLAWTRADE_PORT")
        if port_override:
            self.config.dashboard.port = int(port_override)
        self.running = False

        # Core components (initialised in startup())
        self.limits: Optional[RiskLimits] = None
        self.circuit_breaker: Optional[CircuitBreaker] = None
        self.risk_manager: Optional[RiskManager] = None
        self.portfolio_mgr: Optional[PortfolioStateManager] = None
        self.db: Optional[Database] = None
        self.dashboard: Optional[Dashboard] = None
        self.aggregator: Optional[MarketDataAggregator] = None
        self.strategy_engine: Optional[StrategyEngine] = None
        self.broker: Optional[Any] = None  # SimBroker or any Broker subclass
        self.order_manager: Optional[OrderManager] = None
        self._broker_mode: str = "simulated"

        self._dashboard_task: Optional[asyncio.Task[None]] = None
        self._last_day: Optional[int] = None
        self._last_week: Optional[int] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialise all components and verify readiness."""
        # Load .env file for API keys
        load_dotenv()

        # Logging
        setup_logging(
            level=self.config.logging.level,
            log_file=self.config.logging.file,
            max_size_mb=self.config.logging.max_size_mb,
            rotation=self.config.logging.rotation,
        )
        logger.info("Starting KlawTrade trading system")

        # Risk layer
        self.limits = limits_from_config(self.config.risk, self.config.strategy)
        self.circuit_breaker = CircuitBreaker(self.limits)
        self.risk_manager = RiskManager(self.limits, self.circuit_breaker)
        logger.info("Risk manager initialised")

        # Portfolio
        self.portfolio_mgr = PortfolioStateManager(
            starting_capital=self.config.portfolio.starting_capital,
        )
        logger.info(
            "Portfolio manager initialised",
            extra={"starting_capital": self.config.portfolio.starting_capital},
        )

        # Market data
        data_provider = SimulatedDataProvider(seed=42)
        self.aggregator = MarketDataAggregator(provider=data_provider)
        logger.info("Market data aggregator initialised (simulated)")

        # Strategy engine
        self.strategy_engine = StrategyEngine(
            strategies=[momentum, mean_reversion],
            risk_manager=self.risk_manager,
        )
        logger.info(
            "Strategy engine initialised",
            extra={"strategies": self.strategy_engine.strategy_count},
        )

        # Broker + order manager (auto-detect from config provider)
        force_sim = os.environ.get("KLAWTRADE_FORCE_SIM") == "1"
        if force_sim:
            self.broker = SimBroker(initial_cash=self.config.portfolio.starting_capital)
            self._broker_mode = "simulated"
            logger.info("SimBroker initialised (forced via KLAWTRADE_FORCE_SIM)")
        else:
            provider = self.config.broker.provider
            self.broker = create_broker(
                provider=provider,
                paper=self.config.broker.paper,
            )
            if isinstance(self.broker, SimBroker):
                self._broker_mode = "simulated"
                self.broker = SimBroker(initial_cash=self.config.portfolio.starting_capital)
                logger.info("SimBroker initialised (no %s keys found)", provider)
            else:
                self._broker_mode = provider
                logger.info(
                    "%s broker initialised (paper=%s)",
                    provider.upper(),
                    self.config.broker.paper,
                )
        self.order_manager = OrderManager(broker=self.broker)

        # Database
        self.db = Database()
        await self.db.initialize()
        logger.info("Database initialised")

        # Dashboard
        if self.config.dashboard.enabled:
            self.dashboard = Dashboard()
            self.dashboard.set_callbacks(
                get_state=self._get_dashboard_state,
                get_trades=self._get_dashboard_trades,
                get_signals=self._get_dashboard_signals,
                kill_switch=self._kill_switch,
                resume_switch=self._resume_switch,
            )
            self._dashboard_task = asyncio.create_task(self._run_dashboard())
            logger.info(
                "Dashboard started",
                extra={"port": self.config.dashboard.port},
            )

        # Day / week tracking for resets
        now = datetime.now(timezone.utc)
        self._last_day = now.day
        self._last_week = now.isocalendar()[1]

        if self._broker_mode == "simulated":
            mode_label = "SIMULATION"
        else:
            trade_type = "PAPER TRADING" if self.config.broker.paper else "LIVE TRADING"
            mode_label = f"{trade_type} ({self._broker_mode.upper()})"
        logger.info(
            "KlawTrade initialised",
            extra={"mode": self.config.system.mode, "broker": self._broker_mode},
        )
        print(f"KlawTrade initialised in {mode_label} mode  |  Dashboard: http://localhost:{self.config.dashboard.port}")

    async def heartbeat(self) -> None:
        """Single iteration of the main trading loop."""
        assert self.circuit_breaker is not None
        assert self.portfolio_mgr is not None
        assert self.db is not None

        now = datetime.now(timezone.utc)

        # -- Day / week boundary resets --
        if self._last_day is not None and now.day != self._last_day:
            self.portfolio_mgr.reset_daily()
            self._last_day = now.day
            logger.info("Daily counters reset")

        if self._last_week is not None and now.isocalendar()[1] != self._last_week:
            self.portfolio_mgr.reset_weekly()
            self._last_week = now.isocalendar()[1]
            logger.info("Weekly counters reset")

        # 1. Check circuit breakers
        if self.circuit_breaker.is_active:
            logger.info(
                "Circuit breaker active, skipping heartbeat",
                extra={"reason": self.circuit_breaker.halt_reason.value if self.circuit_breaker.halt_reason else "unknown"},
            )
            await self._push_dashboard_update()
            return

        # 2. Check market hours (skip if always_on mode)
        if not self.config.system.always_on:
            tz = self.config.system.timezone
            hours = self.config.system.trading_hours
            if not is_market_open(tz, hours):
                ttopen = time_until_market_open(tz, hours)
                logger.debug(
                    "Market closed",
                    extra={"time_until_open_seconds": ttopen.total_seconds()},
                )
                await self._push_dashboard_update()
                return

        # 3. Fetch market data
        snapshots = await self.aggregator.get_watchlist_snapshots(
            self.config.strategy.universe.watchlist
        )
        logger.debug("Fetched %d snapshots", len(snapshots))

        # Update sim broker prices
        for sym, snap in snapshots.items():
            self.broker.update_price(sym, snap.price)

        # 4. Run strategy analysis + risk checks
        state = self.portfolio_mgr.get_state()
        results = self.strategy_engine.analyze(snapshots, state)

        # 5. Process approved signals -> orders
        for result in results:
            rejection_strs = [r.value for r in result.rejection_reasons]
            await self.db.save_signal(result.signal, result.approved, rejection_strs)

            if result.approved:
                order = await self.order_manager.process_approved(result)
                if order and order.status == OrderStatus.FILLED:
                    was_win = (order.filled_price or 0) >= order.limit_price if order.side == "sell" else True
                    self.portfolio_mgr.record_trade(order, was_win)
                    await self.db.save_order(order)
                    logger.info(
                        "Trade executed",
                        extra={"symbol": order.symbol, "side": order.side,
                               "qty": order.quantity, "price": order.filled_price},
                    )

        # 6. Check open order timeouts
        await self.order_manager.check_timeouts()

        # 7. Update portfolio state from broker
        positions = await self.broker.get_positions()
        account = await self.broker.get_account()
        self.portfolio_mgr.update_from_positions(positions, account["cash"])
        state = self.portfolio_mgr.get_state()

        # 8. Portfolio-level risk checks
        self.circuit_breaker.check_consecutive_losses(state.consecutive_losses)
        self.circuit_breaker.check_daily_loss(state.daily_pnl_pct)
        self.circuit_breaker.check_weekly_loss(state.weekly_pnl_pct)
        self.circuit_breaker.check_drawdown(state.current_drawdown_pct)

        # 9. Periodic snapshot
        await self.db.save_portfolio_snapshot(state)

        # 10. Push dashboard update
        await self._push_dashboard_update()

        logger.debug(
            "Heartbeat complete",
            extra={
                "equity": state.total_equity,
                "daily_pnl": state.daily_pnl,
                "positions": len(state.positions),
            },
        )

    async def run(self) -> None:
        """Main loop — runs until stopped."""
        await self.startup()
        self.running = True
        logger.info("Main loop started")

        while self.running:
            try:
                await self.heartbeat()
            except Exception:
                logger.exception("Heartbeat error")
                if self.circuit_breaker:
                    self.circuit_breaker.increment_error_count()
            await asyncio.sleep(self.config.system.heartbeat_interval_seconds)

    async def shutdown(self) -> None:
        """Graceful shutdown — save state and close connections."""
        logger.info("Shutting down KlawTrade")
        self.running = False

        # Save final snapshot
        if self.portfolio_mgr and self.db:
            try:
                await self.db.save_portfolio_snapshot(self.portfolio_mgr.get_state())
            except Exception:
                logger.exception("Failed to save final snapshot")

        # Stop dashboard
        if self._dashboard_task and not self._dashboard_task.done():
            self._dashboard_task.cancel()
            try:
                await self._dashboard_task
            except asyncio.CancelledError:
                pass

        # Close database
        if self.db:
            await self.db.close()

        logger.info("KlawTrade shutdown complete")

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------

    def _get_dashboard_state(self) -> dict:
        """Build a JSON-safe state dict for the dashboard."""
        if not self.portfolio_mgr:
            return {}
        state = self.portfolio_mgr.get_state()
        positions = [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
                "unrealized_pnl_pct": p.unrealized_pnl_pct,
            }
            for p in state.positions
        ]
        return {
            "timestamp": state.timestamp.isoformat(),
            "cash": state.cash,
            "total_equity": state.total_equity,
            "positions": positions,
            "daily_pnl": state.daily_pnl,
            "daily_pnl_pct": state.daily_pnl_pct,
            "weekly_pnl": state.weekly_pnl,
            "weekly_pnl_pct": state.weekly_pnl_pct,
            "total_pnl": state.total_pnl,
            "total_pnl_pct": state.total_pnl_pct,
            "peak_equity": state.peak_equity,
            "current_drawdown_pct": state.current_drawdown_pct,
            "trades_today": state.trades_today,
            "consecutive_losses": state.consecutive_losses,
            "win_rate": state.win_rate,
            "circuit_breaker_active": self.circuit_breaker.is_active if self.circuit_breaker else False,
            "circuit_breaker_reason": (
                self.circuit_breaker.halt_reason.value
                if self.circuit_breaker and self.circuit_breaker.halt_reason
                else None
            ),
        }

    async def _get_dashboard_trades(self) -> list[dict]:
        if self.db:
            return await self.db.get_recent_trades(limit=50)
        return []

    async def _get_dashboard_signals(self) -> list[dict]:
        if self.db:
            return await self.db.get_recent_signals(limit=50)
        return []

    def _kill_switch(self) -> None:
        if self.circuit_breaker:
            self.circuit_breaker.manual_halt()
            logger.critical("KILL SWITCH activated from dashboard")

    def _resume_switch(self) -> None:
        if self.circuit_breaker:
            self.circuit_breaker.manual_resume()
            logger.info("Trading resumed from dashboard")

    async def _push_dashboard_update(self) -> None:
        if self.dashboard:
            state = self._get_dashboard_state()
            await self.dashboard.broadcast_state(state)

    async def _run_dashboard(self) -> None:
        """Run the FastAPI dashboard server in the background."""
        assert self.dashboard is not None
        config = uvicorn.Config(
            app=self.dashboard.app,
            host="0.0.0.0",
            port=self.config.dashboard.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    """CLI entry point for ``python -m src.main``."""
    sentinel = KlawTrade()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handle_signal(sig: int, frame: Any) -> None:
        logger.info("Received signal %s, shutting down", sig)
        sentinel.running = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        loop.run_until_complete(sentinel.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(sentinel.shutdown())
        loop.close()


if __name__ == "__main__":
    main()
