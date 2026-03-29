"""Backtesting engine for KlawTrade.

Downloads historical OHLCV data via yfinance, replays it bar-by-bar through
the strategy and risk pipelines, tracks portfolio state, and computes
performance metrics.
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.config import KlawTradeConfig, load_config
from src.execution.sim_broker import SimBroker
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.limits import RiskLimits
from src.risk.manager import RiskManager
from src.storage.models import (
    MarketSnapshot,
    Order,
    OrderStatus,
    PortfolioState,
    Position,
    RiskCheckResult,
    SignalAction,
    TradeSignal,
)
from src.strategy.rules import mean_reversion, momentum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = _ema(series, 12)
    ema26 = _ema(series, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    return macd_line, signal_line


def _bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = _sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def _stoch_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    rsi_vals = _rsi(series, period)
    rsi_min = rsi_vals.rolling(window=period, min_periods=period).min()
    rsi_max = rsi_vals.rolling(window=period, min_periods=period).max()
    denom = rsi_max - rsi_min
    return ((rsi_vals - rsi_min) / denom.replace(0, np.nan)).clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_historical_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data from yfinance for the given symbols.

    Returns a dict mapping symbol -> DataFrame with columns:
        Open, High, Low, Close, Volume
    indexed by datetime.
    """
    import yfinance as yf

    data: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                logger.warning("No data returned for %s", symbol)
                continue

            # Keep only the columns we need
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logger.warning("Missing columns %s for %s, skipping", missing, symbol)
                continue

            df = df[required_cols].copy()
            df.dropna(subset=["Close"], inplace=True)

            if len(df) < 2:
                logger.warning("Insufficient data for %s (%d rows), skipping", symbol, len(df))
                continue

            data[symbol] = df
            logger.info("Downloaded %d bars for %s", len(df), symbol)

        except Exception:
            logger.exception("Failed to download data for %s", symbol)

    return data


def enrich_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator columns to a price DataFrame in-place."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["sma_20"] = _sma(close, 20)
    df["sma_50"] = _sma(close, 50)
    df["sma_200"] = _sma(close, 200)
    df["ema_12"] = _ema(close, 12)
    df["ema_26"] = _ema(close, 26)
    df["rsi_14"] = _rsi(close, 14)

    macd_line, macd_signal = _macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal

    bb_upper, bb_mid, bb_lower = _bollinger_bands(close)
    df["bollinger_upper"] = bb_upper
    df["bollinger_mid"] = bb_mid
    df["bollinger_lower"] = bb_lower

    df["atr_14"] = _atr(high, low, close, 14)
    df["volume_sma_20"] = _sma(volume.astype(float), 20)
    df["stoch_rsi"] = _stoch_rsi(close, 14)
    df["daily_change_pct"] = close.pct_change() * 100.0
    df["vwap"] = (close * volume).cumsum() / volume.cumsum()

    return df


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    """Record of a completed round-trip trade."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    strategy: str


@dataclass
class BacktestMetrics:
    """Summary performance metrics from a backtest run."""
    start_date: str
    end_date: str
    symbols: list[str]
    strategies: list[str]
    starting_capital: float
    ending_equity: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_trade_pnl: float
    best_trade_pnl: float
    worst_trade_pnl: float
    avg_holding_period_days: float
    exposure_pct: float
    signals_generated: int
    signals_approved: int
    signals_rejected: int
    daily_returns: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def summary_report(self) -> str:
        """Generate a formatted text summary report."""
        sep = "=" * 64
        thin_sep = "-" * 64
        lines = [
            "",
            sep,
            "  KLAWTRADE BACKTEST REPORT",
            sep,
            "",
            f"  Period:           {self.start_date}  -->  {self.end_date}",
            f"  Symbols:          {', '.join(self.symbols)}",
            f"  Strategies:       {', '.join(self.strategies)}",
            "",
            thin_sep,
            "  RETURNS",
            thin_sep,
            f"  Starting Capital: ${self.starting_capital:>14,.2f}",
            f"  Ending Equity:    ${self.ending_equity:>14,.2f}",
            f"  Total Return:     {self.total_return_pct:>14.2f}%",
            f"  Annualized:       {self.annualized_return_pct:>14.2f}%",
            "",
            thin_sep,
            "  RISK-ADJUSTED",
            thin_sep,
            f"  Sharpe Ratio:     {self.sharpe_ratio:>14.3f}",
            f"  Sortino Ratio:    {self.sortino_ratio:>14.3f}",
            f"  Max Drawdown:     {self.max_drawdown_pct:>14.2f}%",
            f"  Max DD Duration:  {self.max_drawdown_duration_days:>14d} days",
            "",
            thin_sep,
            "  TRADES",
            thin_sep,
            f"  Total Trades:     {self.total_trades:>14d}",
            f"  Winning:          {self.winning_trades:>14d}",
            f"  Losing:           {self.losing_trades:>14d}",
            f"  Win Rate:         {self.win_rate:>14.1f}%",
            f"  Profit Factor:    {self.profit_factor:>14.3f}",
            "",
            f"  Avg Win:          {self.avg_win_pct:>14.2f}%",
            f"  Avg Loss:         {self.avg_loss_pct:>14.2f}%",
            f"  Avg Trade P&L:    ${self.avg_trade_pnl:>14,.2f}",
            f"  Best Trade:       ${self.best_trade_pnl:>14,.2f}",
            f"  Worst Trade:      ${self.worst_trade_pnl:>14,.2f}",
            f"  Avg Hold Period:  {self.avg_holding_period_days:>14.1f} days",
            "",
            thin_sep,
            "  SIGNALS",
            thin_sep,
            f"  Generated:        {self.signals_generated:>14d}",
            f"  Approved:         {self.signals_approved:>14d}",
            f"  Rejected:         {self.signals_rejected:>14d}",
            f"  Approval Rate:    {(self.signals_approved / max(1, self.signals_generated)) * 100:>14.1f}%",
            f"  Exposure:         {self.exposure_pct:>14.1f}%",
            "",
            sep,
        ]
        return "\n".join(lines)


def _compute_metrics(
    trades: list[BacktestTrade],
    equity_curve: list[float],
    daily_returns: list[float],
    starting_capital: float,
    start_date: str,
    end_date: str,
    symbols: list[str],
    strategies: list[str],
    signals_generated: int,
    signals_approved: int,
    signals_rejected: int,
    invested_days: int,
    total_days: int,
) -> BacktestMetrics:
    """Compute all backtest performance metrics."""
    ending_equity = equity_curve[-1] if equity_curve else starting_capital
    total_return_pct = ((ending_equity - starting_capital) / starting_capital) * 100.0

    # Annualized return
    n_years = total_days / 252.0 if total_days > 0 else 1.0
    if ending_equity > 0 and starting_capital > 0 and n_years > 0:
        annualized_return_pct = ((ending_equity / starting_capital) ** (1.0 / n_years) - 1.0) * 100.0
    else:
        annualized_return_pct = 0.0

    # Sharpe ratio (annualized, assuming 252 trading days, risk-free rate ~ 0)
    dr = np.array(daily_returns) if daily_returns else np.array([0.0])
    mean_daily = float(np.mean(dr))
    std_daily = float(np.std(dr, ddof=1)) if len(dr) > 1 else 0.0
    sharpe_ratio = (mean_daily / std_daily * math.sqrt(252)) if std_daily > 0 else 0.0

    # Sortino ratio (uses downside deviation)
    downside = dr[dr < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino_ratio = (mean_daily / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0

    # Max drawdown
    eq = np.array(equity_curve) if equity_curve else np.array([starting_capital])
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / np.where(running_max > 0, running_max, 1.0)
    max_drawdown_pct = float(np.min(drawdowns)) * 100.0  # negative value

    # Max drawdown duration
    max_dd_duration = 0
    current_dd_duration = 0
    for i in range(len(eq)):
        if eq[i] < running_max[i]:
            current_dd_duration += 1
            max_dd_duration = max(max_dd_duration, current_dd_duration)
        else:
            current_dd_duration = 0

    # Trade statistics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    n_wins = len(winning_trades)
    n_losses = len(losing_trades)
    win_rate = (n_wins / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0.0
    gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    avg_win_pct = float(np.mean([t.pnl_pct for t in winning_trades])) if winning_trades else 0.0
    avg_loss_pct = float(np.mean([t.pnl_pct for t in losing_trades])) if losing_trades else 0.0
    avg_trade_pnl = float(np.mean([t.pnl for t in trades])) if trades else 0.0
    best_trade_pnl = max((t.pnl for t in trades), default=0.0)
    worst_trade_pnl = min((t.pnl for t in trades), default=0.0)

    holding_periods = [
        (t.exit_time - t.entry_time).total_seconds() / 86400.0 for t in trades
    ]
    avg_holding_period_days = float(np.mean(holding_periods)) if holding_periods else 0.0

    exposure_pct = (invested_days / max(1, total_days)) * 100.0

    return BacktestMetrics(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        strategies=strategies,
        starting_capital=starting_capital,
        ending_equity=ending_equity,
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized_return_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_duration_days=max_dd_duration,
        total_trades=total_trades,
        winning_trades=n_wins,
        losing_trades=n_losses,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        avg_trade_pnl=avg_trade_pnl,
        best_trade_pnl=best_trade_pnl,
        worst_trade_pnl=worst_trade_pnl,
        avg_holding_period_days=avg_holding_period_days,
        exposure_pct=exposure_pct,
        signals_generated=signals_generated,
        signals_approved=signals_approved,
        signals_rejected=signals_rejected,
        daily_returns=daily_returns,
        equity_curve=equity_curve,
    )


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Replays historical market data through KlawTrade strategies and risk
    management, tracking portfolio state and computing performance metrics.

    Usage::

        engine = BacktestEngine(
            symbols=["AAPL", "MSFT"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            starting_capital=100_000,
            strategy_names=["momentum"],
        )
        metrics = engine.run()
        print(metrics.summary_report())
    """

    def __init__(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        starting_capital: float = 100_000.0,
        strategy_names: list[str] | None = None,
        config: KlawTradeConfig | None = None,
        risk_limits: RiskLimits | None = None,
    ) -> None:
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.starting_capital = starting_capital
        self.strategy_names = strategy_names or ["momentum", "mean_reversion"]

        self._config = config or KlawTradeConfig()

        # Build risk infrastructure with relaxed limits for backtesting
        if risk_limits is not None:
            self._limits = risk_limits
        else:
            self._limits = RiskLimits(
                max_portfolio_allocation=0.90,
                max_single_position_pct=0.10,
                max_sector_allocation_pct=0.30,
                max_correlated_exposure_pct=0.40,
                max_daily_loss_pct=0.03,
                max_weekly_loss_pct=0.07,
                max_drawdown_pct=0.15,
                max_single_trade_loss_pct=0.02,
                max_open_positions=15,
                max_daily_trades=50,
                min_cash_reserve_pct=0.10,
                min_volume_threshold=0,  # relaxed for historical data
                max_spread_pct=1.0,  # relaxed — no live spread in daily bars
                min_confidence=0.30,  # lower for backtesting to see more trades
                max_signal_age_seconds=86400,  # 1 day
                consecutive_losses_halt=5,
                halt_duration_minutes=60,
                vix_threshold=35.0,
            )

        self._circuit_breaker = CircuitBreaker(self._limits)
        self._risk_manager = RiskManager(self._limits, self._circuit_breaker)
        self._broker = SimBroker(
            initial_cash=starting_capital,
            slippage_max_pct=0.001,
        )

        # Strategy dispatch
        self._strategy_funcs: dict[str, object] = {
            "momentum": momentum,
            "mean_reversion": mean_reversion,
        }

        # State tracking
        self._trades: list[BacktestTrade] = []
        self._equity_curve: list[float] = []
        self._daily_returns: list[float] = []
        self._signals_generated = 0
        self._signals_approved = 0
        self._signals_rejected = 0
        self._invested_days = 0

        # Open trade tracking for round-trip P&L
        # symbol -> list of {entry_price, quantity, entry_time, strategy}
        self._open_entries: dict[str, list[dict]] = {}

    def run(self) -> BacktestMetrics:
        """Execute the full backtest and return performance metrics."""
        logger.info(
            "Starting backtest: %s to %s, symbols=%s, strategies=%s, capital=$%,.2f",
            self.start_date, self.end_date, self.symbols,
            self.strategy_names, self.starting_capital,
        )

        # 1. Download data
        print(f"\nDownloading historical data for {', '.join(self.symbols)}...")
        raw_data = download_historical_data(
            self.symbols, self.start_date, self.end_date
        )

        if not raw_data:
            raise ValueError(
                f"No historical data available for any of: {self.symbols}"
            )

        # 2. Enrich with indicators
        print("Computing technical indicators...")
        enriched: dict[str, pd.DataFrame] = {}
        for symbol, df in raw_data.items():
            enriched[symbol] = enrich_with_indicators(df)

        # 3. Build a unified date index
        all_dates: set[datetime] = set()
        for df in enriched.values():
            all_dates.update(df.index.to_pydatetime())
        sorted_dates = sorted(all_dates)

        if not sorted_dates:
            raise ValueError("No trading dates found in downloaded data")

        print(
            f"Replaying {len(sorted_dates)} bars across "
            f"{len(enriched)} symbols...\n"
        )

        # 4. Replay bar-by-bar
        prev_equity = self.starting_capital
        self._equity_curve.append(self.starting_capital)

        for bar_idx, bar_dt in enumerate(sorted_dates):
            # Reset daily counters at start of each day
            daily_signals: list[TradeSignal] = []

            for symbol, df in enriched.items():
                if bar_dt not in df.index:
                    continue

                row = df.loc[bar_dt]
                snapshot = self._row_to_snapshot(symbol, bar_dt, row)

                # Update broker price for P&L tracking
                self._broker.update_price(symbol, snapshot.price)

                # Build current portfolio state
                portfolio = self._build_portfolio_state(bar_dt)

                # Run each active strategy
                for strat_name in self.strategy_names:
                    strat_module = self._strategy_funcs.get(strat_name)
                    if strat_module is None:
                        continue

                    try:
                        signals = strat_module.generate_signals(snapshot, portfolio)
                    except Exception:
                        logger.exception(
                            "Strategy %s error on %s at %s",
                            strat_name, symbol, bar_dt,
                        )
                        continue

                    for signal in signals:
                        self._signals_generated += 1
                        daily_signals.append(signal)

                        # Run through risk manager
                        result = self._risk_manager.check(
                            signal, portfolio, snapshot
                        )

                        if result.approved:
                            self._signals_approved += 1
                            self._execute_signal(
                                signal, result, snapshot, bar_dt
                            )
                        else:
                            self._signals_rejected += 1

            # End-of-day equity tracking
            current_equity = self._get_total_equity()
            self._equity_curve.append(current_equity)

            daily_ret = (
                (current_equity - prev_equity) / prev_equity
                if prev_equity > 0
                else 0.0
            )
            self._daily_returns.append(daily_ret)

            # Track invested days (any open positions)
            if self._broker._positions:
                self._invested_days += 1

            prev_equity = current_equity

        # 5. Close any remaining positions at final prices
        self._close_all_remaining(sorted_dates[-1] if sorted_dates else datetime.now(timezone.utc))

        # 6. Final equity
        final_equity = self._get_total_equity()
        self._equity_curve.append(final_equity)

        # 7. Compute metrics
        metrics = _compute_metrics(
            trades=self._trades,
            equity_curve=self._equity_curve,
            daily_returns=self._daily_returns,
            starting_capital=self.starting_capital,
            start_date=self.start_date,
            end_date=self.end_date,
            symbols=list(enriched.keys()),
            strategies=self.strategy_names,
            signals_generated=self._signals_generated,
            signals_approved=self._signals_approved,
            signals_rejected=self._signals_rejected,
            invested_days=self._invested_days,
            total_days=len(sorted_dates),
        )

        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_opt_float(val: object) -> Optional[float]:
        """Safely convert a value to Optional[float], returning None for NaN."""
        if val is None:
            return None
        try:
            f = float(val)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except (TypeError, ValueError):
            return None

    def _row_to_snapshot(
        self,
        symbol: str,
        bar_dt: datetime,
        row: pd.Series,
    ) -> MarketSnapshot:
        """Convert a DataFrame row to a MarketSnapshot."""
        price = float(row["Close"])
        volume = int(row["Volume"]) if not pd.isna(row["Volume"]) else 0
        opt = self._to_opt_float

        return MarketSnapshot(
            symbol=symbol,
            timestamp=bar_dt if bar_dt.tzinfo else bar_dt.replace(tzinfo=timezone.utc),
            price=price,
            bid=price,  # daily bars have no bid/ask; approximate
            ask=price,
            volume=volume,
            daily_volume=volume,
            vwap=float(row.get("vwap", price)) if opt(row.get("vwap")) is not None else price,
            daily_change_pct=opt(row.get("daily_change_pct")) or 0.0,
            rsi_14=opt(row.get("rsi_14")),
            macd=opt(row.get("macd")),
            macd_signal=opt(row.get("macd_signal")),
            sma_20=opt(row.get("sma_20")),
            sma_50=opt(row.get("sma_50")),
            sma_200=opt(row.get("sma_200")),
            ema_12=opt(row.get("ema_12")),
            ema_26=opt(row.get("ema_26")),
            bollinger_upper=opt(row.get("bollinger_upper")),
            bollinger_lower=opt(row.get("bollinger_lower")),
            bollinger_mid=opt(row.get("bollinger_mid")),
            atr_14=opt(row.get("atr_14")),
            volume_sma_20=opt(row.get("volume_sma_20")),
            stoch_rsi=opt(row.get("stoch_rsi")),
        )

    def _build_portfolio_state(self, ts: datetime) -> PortfolioState:
        """Build a PortfolioState from the SimBroker's current state."""
        positions = asyncio.get_event_loop().run_until_complete(
            self._broker.get_positions()
        ) if asyncio.get_event_loop().is_running() else self._sync_get_positions()

        cash = self._broker._cash
        positions_value = sum(p.market_value for p in positions)
        total_equity = cash + positions_value
        peak = max(self._equity_curve) if self._equity_curve else self.starting_capital
        peak = max(peak, total_equity)

        drawdown = ((peak - total_equity) / peak) if peak > 0 else 0.0

        # Daily P&L approximation
        prev_equity = self._equity_curve[-1] if self._equity_curve else self.starting_capital
        daily_pnl = total_equity - prev_equity
        daily_pnl_pct = (daily_pnl / prev_equity) if prev_equity > 0 else 0.0

        total_pnl = total_equity - self.starting_capital
        total_pnl_pct = (total_pnl / self.starting_capital) if self.starting_capital > 0 else 0.0

        ts_aware = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)

        return PortfolioState(
            timestamp=ts_aware,
            cash=cash,
            total_equity=total_equity,
            positions=positions,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            peak_equity=peak,
            current_drawdown_pct=drawdown,
            trades_today=0,  # simplified for backtest
        )

    def _sync_get_positions(self) -> list[Position]:
        """Synchronously retrieve positions from the async SimBroker."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._broker.get_positions())
        finally:
            loop.close()

    def _execute_signal(
        self,
        signal: TradeSignal,
        result: RiskCheckResult,
        snapshot: MarketSnapshot,
        bar_dt: datetime,
    ) -> None:
        """Execute an approved signal through the SimBroker."""
        if signal.action == SignalAction.BUY:
            quantity = result.adjusted_quantity or signal.suggested_quantity
            if quantity <= 0:
                return

            order = Order(
                id=str(uuid.uuid4()),
                signal_id=signal.id,
                symbol=signal.symbol,
                side="buy",
                quantity=quantity,
                order_type="limit",
                limit_price=snapshot.price,
                status=OrderStatus.PENDING,
                submitted_at=bar_dt.replace(tzinfo=timezone.utc) if not bar_dt.tzinfo else bar_dt,
                stop_loss_price=signal.stop_loss_price,
                take_profit_price=signal.take_profit_price,
            )

            filled_order = self._sync_submit_order(order)
            if filled_order.status == OrderStatus.FILLED:
                entry = {
                    "entry_price": filled_order.filled_price or snapshot.price,
                    "quantity": filled_order.filled_quantity or quantity,
                    "entry_time": bar_dt.replace(tzinfo=timezone.utc) if not bar_dt.tzinfo else bar_dt,
                    "strategy": signal.strategy_name,
                }
                self._open_entries.setdefault(signal.symbol, []).append(entry)

        elif signal.action in (SignalAction.SELL, SignalAction.CLOSE):
            quantity = signal.suggested_quantity

            order = Order(
                id=str(uuid.uuid4()),
                signal_id=signal.id,
                symbol=signal.symbol,
                side="sell",
                quantity=quantity,
                order_type="limit",
                limit_price=snapshot.price,
                status=OrderStatus.PENDING,
                submitted_at=bar_dt.replace(tzinfo=timezone.utc) if not bar_dt.tzinfo else bar_dt,
            )

            filled_order = self._sync_submit_order(order)
            if filled_order.status == OrderStatus.FILLED:
                self._record_exit(
                    symbol=signal.symbol,
                    exit_price=filled_order.filled_price or snapshot.price,
                    exit_quantity=filled_order.filled_quantity or quantity,
                    exit_time=bar_dt.replace(tzinfo=timezone.utc) if not bar_dt.tzinfo else bar_dt,
                )

    def _sync_submit_order(self, order: Order) -> Order:
        """Synchronously submit an order to the async SimBroker."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._broker.submit_order(order))
        finally:
            loop.close()

    def _record_exit(
        self,
        symbol: str,
        exit_price: float,
        exit_quantity: int,
        exit_time: datetime,
    ) -> None:
        """Match an exit against open entries (FIFO) and record BacktestTrades."""
        entries = self._open_entries.get(symbol, [])
        remaining = exit_quantity

        while remaining > 0 and entries:
            entry = entries[0]
            fill_qty = min(remaining, entry["quantity"])

            pnl = (exit_price - entry["entry_price"]) * fill_qty
            pnl_pct = (
                ((exit_price - entry["entry_price"]) / entry["entry_price"]) * 100.0
                if entry["entry_price"] > 0
                else 0.0
            )

            self._trades.append(
                BacktestTrade(
                    symbol=symbol,
                    side="long",
                    entry_price=entry["entry_price"],
                    exit_price=exit_price,
                    quantity=fill_qty,
                    entry_time=entry["entry_time"],
                    exit_time=exit_time,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    strategy=entry["strategy"],
                )
            )

            entry["quantity"] -= fill_qty
            remaining -= fill_qty

            if entry["quantity"] <= 0:
                entries.pop(0)

        # Clean up empty entry lists
        if not entries:
            self._open_entries.pop(symbol, None)

    def _close_all_remaining(self, final_dt: datetime) -> None:
        """Close all open positions at their last known price."""
        final_dt_aware = final_dt.replace(tzinfo=timezone.utc) if not final_dt.tzinfo else final_dt

        for symbol, sim_pos in list(self._broker._positions.items()):
            if sim_pos.quantity <= 0:
                continue

            order = Order(
                id=str(uuid.uuid4()),
                signal_id="backtest-close",
                symbol=symbol,
                side="sell",
                quantity=sim_pos.quantity,
                order_type="limit",
                limit_price=sim_pos.current_price,
                status=OrderStatus.PENDING,
                submitted_at=final_dt_aware,
            )

            filled_order = self._sync_submit_order(order)
            if filled_order.status == OrderStatus.FILLED:
                self._record_exit(
                    symbol=symbol,
                    exit_price=filled_order.filled_price or sim_pos.current_price,
                    exit_quantity=filled_order.filled_quantity or sim_pos.quantity,
                    exit_time=final_dt_aware,
                )

    def _get_total_equity(self) -> float:
        """Get total equity from SimBroker."""
        loop = asyncio.new_event_loop()
        try:
            account = loop.run_until_complete(self._broker.get_account())
            return account["total_equity"]
        finally:
            loop.close()
