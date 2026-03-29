"""SQLite storage for trade history, signals, and portfolio snapshots."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from src.storage.models import Order, OrderStatus, PortfolioState, TradeSignal

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    suggested_quantity INTEGER NOT NULL,
    strategy_name TEXT NOT NULL,
    confirming_indicators TEXT,
    suggested_limit_price REAL,
    stop_loss_price REAL,
    take_profit_price REAL,
    reasoning TEXT,
    timestamp TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    approved INTEGER NOT NULL,
    rejection_reasons TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    order_type TEXT NOT NULL,
    limit_price REAL NOT NULL,
    status TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    broker_order_id TEXT,
    stop_loss_price REAL,
    take_profit_price REAL,
    filled_at TEXT,
    filled_price REAL,
    filled_quantity INTEGER,
    fees REAL DEFAULT 0.0,
    slippage REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    pnl REAL DEFAULT 0.0,
    pnl_pct REAL DEFAULT 0.0,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    strategy_name TEXT,
    FOREIGN KEY (order_id) REFERENCES orders(id)
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cash REAL NOT NULL,
    total_equity REAL NOT NULL,
    daily_pnl REAL DEFAULT 0.0,
    daily_pnl_pct REAL DEFAULT 0.0,
    weekly_pnl REAL DEFAULT 0.0,
    weekly_pnl_pct REAL DEFAULT 0.0,
    total_pnl REAL DEFAULT 0.0,
    total_pnl_pct REAL DEFAULT 0.0,
    peak_equity REAL DEFAULT 0.0,
    current_drawdown_pct REAL DEFAULT 0.0,
    trades_today INTEGER DEFAULT 0,
    consecutive_losses INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0,
    positions_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_opened_at ON trades(opened_at);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_timestamp ON portfolio_snapshots(timestamp);
"""


class Database:
    """Async SQLite storage for the KlawTrade trading system."""

    def __init__(self, db_path: str | Path = "data/klawtrade.db") -> None:
        self._db_path = Path(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Create the database and tables."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        logger.info("Database initialized", extra={"path": str(self._db_path)})

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def save_signal(
        self,
        signal: TradeSignal,
        approved: bool,
        rejection_reasons: list[str],
    ) -> None:
        """Persist a trade signal and its risk check outcome."""
        assert self._db is not None, "Database not initialized"
        await self._db.execute(
            """INSERT OR REPLACE INTO signals
               (id, symbol, action, confidence, suggested_quantity, strategy_name,
                confirming_indicators, suggested_limit_price, stop_loss_price,
                take_profit_price, reasoning, timestamp, expires_at,
                approved, rejection_reasons)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.id,
                signal.symbol,
                signal.action.value,
                signal.confidence,
                signal.suggested_quantity,
                signal.strategy_name,
                json.dumps(signal.confirming_indicators),
                signal.suggested_limit_price,
                signal.stop_loss_price,
                signal.take_profit_price,
                signal.reasoning,
                signal.timestamp.isoformat(),
                signal.expires_at.isoformat(),
                int(approved),
                json.dumps(rejection_reasons),
            ),
        )
        await self._db.commit()

    async def save_order(self, order: Order) -> None:
        """Persist an order."""
        assert self._db is not None, "Database not initialized"
        await self._db.execute(
            """INSERT OR REPLACE INTO orders
               (id, signal_id, symbol, side, quantity, order_type, limit_price,
                status, submitted_at, broker_order_id, stop_loss_price,
                take_profit_price, filled_at, filled_price, filled_quantity,
                fees, slippage)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                order.id,
                order.signal_id,
                order.symbol,
                order.side,
                order.quantity,
                order.order_type,
                order.limit_price,
                order.status.value,
                order.submitted_at.isoformat(),
                order.broker_order_id,
                order.stop_loss_price,
                order.take_profit_price,
                order.filled_at.isoformat() if order.filled_at else None,
                order.filled_price,
                order.filled_quantity,
                order.fees,
                order.slippage,
            ),
        )
        await self._db.commit()

    async def save_portfolio_snapshot(self, state: PortfolioState) -> None:
        """Persist a portfolio state snapshot."""
        assert self._db is not None, "Database not initialized"
        positions_json = json.dumps([
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
        ])
        await self._db.execute(
            """INSERT INTO portfolio_snapshots
               (timestamp, cash, total_equity, daily_pnl, daily_pnl_pct,
                weekly_pnl, weekly_pnl_pct, total_pnl, total_pnl_pct,
                peak_equity, current_drawdown_pct, trades_today,
                consecutive_losses, win_rate, positions_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                state.timestamp.isoformat(),
                state.cash,
                state.total_equity,
                state.daily_pnl,
                state.daily_pnl_pct,
                state.weekly_pnl,
                state.weekly_pnl_pct,
                state.total_pnl,
                state.total_pnl_pct,
                state.peak_equity,
                state.current_drawdown_pct,
                state.trades_today,
                state.consecutive_losses,
                state.win_rate,
                positions_json,
            ),
        )
        await self._db.commit()

    async def get_recent_trades(self, limit: int = 50) -> list[dict]:
        """Return the most recent completed trades."""
        assert self._db is not None, "Database not initialized"
        cursor = await self._db.execute(
            """SELECT o.id, o.symbol, o.side, o.quantity, o.limit_price,
                      o.filled_price, o.filled_at, o.status, o.fees, o.slippage,
                      s.strategy_name, s.confidence
               FROM orders o
               LEFT JOIN signals s ON o.signal_id = s.id
               WHERE o.status IN (?, ?)
               ORDER BY o.filled_at DESC
               LIMIT ?""",
            (OrderStatus.FILLED.value, OrderStatus.PARTIAL.value, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_recent_signals(self, limit: int = 50) -> list[dict]:
        """Return the most recent signals."""
        assert self._db is not None, "Database not initialized"
        cursor = await self._db.execute(
            """SELECT id, symbol, action, confidence, suggested_quantity,
                      strategy_name, reasoning, timestamp, approved,
                      rejection_reasons
               FROM signals
               ORDER BY timestamp DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["rejection_reasons"] = json.loads(d["rejection_reasons"] or "[]")
            results.append(d)
        return results

    async def get_daily_stats(self) -> dict:
        """Return aggregate stats for today."""
        assert self._db is not None, "Database not initialized"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        cursor = await self._db.execute(
            """SELECT COUNT(*) as total_signals,
                      SUM(CASE WHEN approved = 1 THEN 1 ELSE 0 END) as approved_signals
               FROM signals
               WHERE timestamp LIKE ?""",
            (f"{today}%",),
        )
        signal_stats = dict(await cursor.fetchone())

        cursor = await self._db.execute(
            """SELECT COUNT(*) as total_orders,
                      SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) as filled_orders,
                      SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) as cancelled_orders
               FROM orders
               WHERE submitted_at LIKE ?""",
            (OrderStatus.FILLED.value, OrderStatus.CANCELLED.value, f"{today}%"),
        )
        order_stats = dict(await cursor.fetchone())

        return {
            "date": today,
            "total_signals": signal_stats.get("total_signals", 0) or 0,
            "approved_signals": signal_stats.get("approved_signals", 0) or 0,
            "total_orders": order_stats.get("total_orders", 0) or 0,
            "filled_orders": order_stats.get("filled_orders", 0) or 0,
            "cancelled_orders": order_stats.get("cancelled_orders", 0) or 0,
        }
