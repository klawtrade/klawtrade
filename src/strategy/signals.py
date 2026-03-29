"""Convenience factories for creating trade signals."""

from __future__ import annotations

from typing import Optional

from src.storage.models import SignalAction, TradeSignal


def create_buy_signal(
    symbol: str,
    confidence: float,
    quantity: int,
    limit_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    strategy_name: str = "",
    indicators: Optional[list[str]] = None,
    reasoning: str = "",
) -> TradeSignal:
    """Create a BUY trade signal with sensible defaults."""
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.BUY,
        confidence=max(0.0, min(1.0, confidence)),
        suggested_quantity=max(1, quantity),
        strategy_name=strategy_name,
        confirming_indicators=indicators or [],
        suggested_limit_price=limit_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        reasoning=reasoning,
    )


def create_sell_signal(
    symbol: str,
    confidence: float,
    quantity: int,
    limit_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    strategy_name: str = "",
    indicators: Optional[list[str]] = None,
    reasoning: str = "",
) -> TradeSignal:
    """Create a SELL trade signal with sensible defaults."""
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.SELL,
        confidence=max(0.0, min(1.0, confidence)),
        suggested_quantity=max(1, quantity),
        strategy_name=strategy_name,
        confirming_indicators=indicators or [],
        suggested_limit_price=limit_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        reasoning=reasoning,
    )


def create_close_signal(
    symbol: str,
    confidence: float,
    quantity: int,
    limit_price: Optional[float] = None,
    strategy_name: str = "",
    indicators: Optional[list[str]] = None,
    reasoning: str = "",
) -> TradeSignal:
    """Create a CLOSE trade signal with sensible defaults.

    Close signals typically don't need stop-loss or take-profit since
    we're exiting an existing position.
    """
    return TradeSignal(
        symbol=symbol,
        action=SignalAction.CLOSE,
        confidence=max(0.0, min(1.0, confidence)),
        suggested_quantity=max(1, quantity),
        strategy_name=strategy_name,
        confirming_indicators=indicators or [],
        suggested_limit_price=limit_price,
        reasoning=reasoning,
    )
