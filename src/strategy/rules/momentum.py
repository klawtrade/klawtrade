"""Rule-based momentum strategy.

Pure Python, no LLM. Generates BUY/CLOSE signals based on moving-average
crossovers, RSI, and MACD confirmation.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.storage.models import MarketSnapshot, PortfolioState, TradeSignal
from src.strategy.signals import create_buy_signal, create_close_signal

logger = logging.getLogger(__name__)

STRATEGY_NAME = "momentum"

# Indicator weights for confidence scoring
_WEIGHT_SMA_CROSS = 0.30   # SMA20 > SMA50
_WEIGHT_RSI = 0.20         # RSI < 70
_WEIGHT_MACD = 0.25        # MACD > signal
_WEIGHT_TREND = 0.25       # price > SMA200


def _has_position(symbol: str, portfolio: PortfolioState) -> Optional[int]:
    """Return current position quantity for symbol, or None."""
    for pos in portfolio.positions:
        if pos.symbol == symbol:
            return pos.quantity
    return None


def generate_signals(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
) -> list[TradeSignal]:
    """Evaluate momentum rules against a single market snapshot.

    BUY when: SMA20 > SMA50 AND RSI < 70 AND MACD > MACD signal AND price > SMA200
    CLOSE when: SMA20 < SMA50 OR RSI > 80 OR price < SMA50

    Indicators that are None are skipped (and confidence is reduced).
    """
    signals: list[TradeSignal] = []
    symbol = snapshot.symbol
    price = snapshot.price

    # --- Check CLOSE / exit conditions first ---
    held_qty = _has_position(symbol, portfolio)
    if held_qty is not None and held_qty > 0:
        close_signal = _evaluate_close(snapshot, held_qty)
        if close_signal is not None:
            signals.append(close_signal)
            return signals  # If we're closing, don't also generate a buy

    # --- Check BUY conditions ---
    buy_signal = _evaluate_buy(snapshot, portfolio)
    if buy_signal is not None:
        signals.append(buy_signal)

    return signals


def _evaluate_buy(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
) -> Optional[TradeSignal]:
    """Check momentum BUY conditions and build a signal if met."""
    price = snapshot.price
    confirming: list[str] = []
    reasons: list[str] = []
    total_weight = 0.0
    weighted_score = 0.0

    # 1. SMA crossover: SMA20 > SMA50
    if snapshot.sma_20 is not None and snapshot.sma_50 is not None:
        total_weight += _WEIGHT_SMA_CROSS
        if snapshot.sma_20 > snapshot.sma_50:
            weighted_score += _WEIGHT_SMA_CROSS
            confirming.append("sma_crossover")
            reasons.append(f"SMA20 ({snapshot.sma_20:.2f}) > SMA50 ({snapshot.sma_50:.2f})")
        else:
            return None  # Hard requirement
    else:
        reasons.append("SMA20/SMA50 unavailable, skipped")

    # 2. RSI < 70 (not overbought)
    if snapshot.rsi_14 is not None:
        total_weight += _WEIGHT_RSI
        if snapshot.rsi_14 < 70:
            weighted_score += _WEIGHT_RSI
            confirming.append("rsi_not_overbought")
            reasons.append(f"RSI ({snapshot.rsi_14:.1f}) < 70")
        else:
            return None  # Hard requirement
    else:
        reasons.append("RSI unavailable, skipped")

    # 3. MACD > MACD signal
    if snapshot.macd is not None and snapshot.macd_signal is not None:
        total_weight += _WEIGHT_MACD
        if snapshot.macd > snapshot.macd_signal:
            weighted_score += _WEIGHT_MACD
            confirming.append("macd_bullish")
            reasons.append(f"MACD ({snapshot.macd:.4f}) > signal ({snapshot.macd_signal:.4f})")
        else:
            return None  # Hard requirement
    else:
        reasons.append("MACD unavailable, skipped")

    # 4. Price > SMA200 (long-term uptrend)
    if snapshot.sma_200 is not None:
        total_weight += _WEIGHT_TREND
        if price > snapshot.sma_200:
            weighted_score += _WEIGHT_TREND
            confirming.append("above_sma200")
            reasons.append(f"Price ({price:.2f}) > SMA200 ({snapshot.sma_200:.2f})")
        else:
            return None  # Hard requirement
    else:
        reasons.append("SMA200 unavailable, skipped")

    # Need at least one confirming indicator
    if not confirming:
        return None

    # Confidence = weighted score / total possible weight, penalised for missing data
    if total_weight > 0:
        raw_confidence = weighted_score / total_weight
    else:
        raw_confidence = 0.0

    # Penalise if some indicators were missing
    data_completeness = total_weight / (_WEIGHT_SMA_CROSS + _WEIGHT_RSI + _WEIGHT_MACD + _WEIGHT_TREND)
    confidence = raw_confidence * data_completeness

    if confidence < 0.3:
        return None

    # Stop loss and take profit using ATR
    stop_loss, take_profit = _compute_stop_take(price, snapshot.atr_14, direction="buy")

    # Suggest a small default quantity (engine/risk manager will adjust)
    quantity = 1

    return create_buy_signal(
        symbol=snapshot.symbol,
        confidence=confidence,
        quantity=quantity,
        limit_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        strategy_name=STRATEGY_NAME,
        indicators=confirming,
        reasoning="; ".join(reasons),
    )


def _evaluate_close(
    snapshot: MarketSnapshot,
    held_qty: int,
) -> Optional[TradeSignal]:
    """Check momentum CLOSE/exit conditions."""
    price = snapshot.price
    confirming: list[str] = []
    reasons: list[str] = []

    triggered = False

    # CLOSE when SMA20 < SMA50
    if snapshot.sma_20 is not None and snapshot.sma_50 is not None:
        if snapshot.sma_20 < snapshot.sma_50:
            triggered = True
            confirming.append("sma_bearish_cross")
            reasons.append(f"SMA20 ({snapshot.sma_20:.2f}) < SMA50 ({snapshot.sma_50:.2f})")

    # CLOSE when RSI > 80
    if snapshot.rsi_14 is not None:
        if snapshot.rsi_14 > 80:
            triggered = True
            confirming.append("rsi_overbought")
            reasons.append(f"RSI ({snapshot.rsi_14:.1f}) > 80")

    # CLOSE when price < SMA50
    if snapshot.sma_50 is not None:
        if price < snapshot.sma_50:
            triggered = True
            confirming.append("price_below_sma50")
            reasons.append(f"Price ({price:.2f}) < SMA50 ({snapshot.sma_50:.2f})")

    if not triggered:
        return None

    # Confidence based on how many exit conditions fired
    confidence = min(1.0, 0.5 + 0.2 * len(confirming))

    return create_close_signal(
        symbol=snapshot.symbol,
        confidence=confidence,
        quantity=held_qty,
        limit_price=price,
        strategy_name=STRATEGY_NAME,
        indicators=confirming,
        reasoning="; ".join(reasons),
    )


def _compute_stop_take(
    price: float,
    atr: Optional[float],
    direction: str = "buy",
) -> tuple[Optional[float], Optional[float]]:
    """Compute stop-loss and take-profit using ATR.

    Stop loss = price - 2*ATR, Take profit = price + 3*ATR (>= 2:1 R/R).
    Returns (None, None) if ATR is unavailable.
    """
    if atr is None or atr <= 0:
        return None, None

    if direction == "buy":
        stop_loss = round(price - 2.0 * atr, 4)
        take_profit = round(price + 3.0 * atr, 4)
    else:
        stop_loss = round(price + 2.0 * atr, 4)
        take_profit = round(price - 3.0 * atr, 4)

    return stop_loss, take_profit
