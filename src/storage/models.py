"""Core data models for the Sentinel trading system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional


class SignalAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class RiskRejectionReason(Enum):
    MAX_POSITION_SIZE = "max_position_size_exceeded"
    MAX_DAILY_LOSS = "max_daily_loss_exceeded"
    MAX_WEEKLY_LOSS = "max_weekly_loss_exceeded"
    MAX_DRAWDOWN = "max_drawdown_exceeded"
    MAX_POSITIONS = "max_open_positions_exceeded"
    MAX_DAILY_TRADES = "max_daily_trades_exceeded"
    INSUFFICIENT_CASH = "insufficient_cash_reserve"
    LOW_VOLUME = "below_volume_threshold"
    WIDE_SPREAD = "spread_too_wide"
    BLACKLISTED = "symbol_blacklisted"
    CIRCUIT_BREAKER = "circuit_breaker_active"
    CORRELATED_EXPOSURE = "correlated_exposure_exceeded"
    SECTOR_LIMIT = "sector_allocation_exceeded"
    LOW_CONFIDENCE = "signal_confidence_too_low"
    SIGNAL_EXPIRED = "signal_expired"


@dataclass
class MarketSnapshot:
    """Point-in-time market data for a single symbol."""
    symbol: str
    timestamp: datetime
    price: float
    bid: float
    ask: float
    volume: int
    daily_volume: int
    vwap: float
    daily_change_pct: float
    # Technical indicators
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_mid: Optional[float] = None
    atr_14: Optional[float] = None
    volume_sma_20: Optional[float] = None
    stoch_rsi: Optional[float] = None
    # Metadata
    sector: Optional[str] = None
    correlation_group: Optional[str] = None


def _make_signal_id() -> str:
    return str(uuid.uuid4())


def _default_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=300)


@dataclass
class TradeSignal:
    """Output from the strategy engine."""
    symbol: str
    action: SignalAction
    confidence: float
    suggested_quantity: int
    strategy_name: str
    confirming_indicators: list[str] = field(default_factory=list)
    suggested_limit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    reasoning: str = ""
    id: str = field(default_factory=_make_signal_id)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=_default_expiry)
    raw_llm_response: str = ""


@dataclass
class RiskCheckResult:
    """Output from the risk manager."""
    approved: bool
    signal: TradeSignal
    rejection_reasons: list[RiskRejectionReason] = field(default_factory=list)
    adjusted_quantity: Optional[int] = None
    adjusted_limit_price: Optional[float] = None
    risk_score: float = 0.0
    notes: str = ""


@dataclass
class Order:
    """An order submitted to the broker."""
    id: str
    signal_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    order_type: str  # "limit"
    limit_price: float
    status: OrderStatus
    submitted_at: datetime
    broker_order_id: Optional[str] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: Optional[int] = None
    fees: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """A current portfolio position."""
    symbol: str
    quantity: int
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    opened_at: datetime
    last_updated: datetime
    sector: Optional[str] = None
    correlation_group: Optional[str] = None


@dataclass
class PortfolioState:
    """Complete portfolio snapshot."""
    timestamp: datetime
    cash: float
    total_equity: float
    positions: list[Position] = field(default_factory=list)
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    weekly_pnl: float = 0.0
    weekly_pnl_pct: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    win_rate: float = 0.0
    sharpe_ratio: Optional[float] = None
