"""Rule-based mean reversion strategy.

Pure Python, no LLM. Generates BUY/CLOSE signals when price deviates
significantly from its mean (Bollinger Bands) with RSI and Stochastic RSI
confirmation.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.storage.models import MarketSnapshot, PortfolioState, TradeSignal
from src.strategy.signals import create_buy_signal, create_close_signal

logger = logging.getLogger(__name__)

STRATEGY_NAME = "mean_reversion"

# Indicator weights for confidence scoring
_WEIGHT_BOLLINGER = 0.35
_WEIGHT_RSI = 0.35
_WEIGHT_STOCH_RSI = 0.30


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
    """Evaluate mean-reversion rules against a single market snapshot.

    BUY when: price < bollinger_lower AND RSI < 30 AND stoch_rsi < 0.2
    CLOSE when: price > bollinger_upper AND RSI > 70 AND stoch_rsi > 0.8

    Indicators that are None are skipped (and confidence is reduced).
    """
    signals: list[TradeSignal] = []
    symbol = snapshot.symbol

    # --- Check CLOSE / exit conditions first ---
    held_qty = _has_position(symbol, portfolio)
    if held_qty is not None and held_qty > 0:
        close_signal = _evaluate_close(snapshot, held_qty)
        if close_signal is not None:
            signals.append(close_signal)
            return signals

    # --- Check BUY conditions ---
    buy_signal = _evaluate_buy(snapshot, portfolio)
    if buy_signal is not None:
        signals.append(buy_signal)

    return signals


def _evaluate_buy(
    snapshot: MarketSnapshot,
    portfolio: PortfolioState,
) -> Optional[TradeSignal]:
    """Check mean-reversion BUY conditions."""
    price = snapshot.price
    confirming: list[str] = []
    reasons: list[str] = []
    total_weight = 0.0
    weighted_score = 0.0

    # 1. Price below lower Bollinger Band
    if snapshot.bollinger_lower is not None:
        total_weight += _WEIGHT_BOLLINGER
        if price < snapshot.bollinger_lower:
            weighted_score += _WEIGHT_BOLLINGER
            confirming.append("below_bollinger_lower")
            reasons.append(
                f"Price ({price:.2f}) < Bollinger lower ({snapshot.bollinger_lower:.2f})"
            )
        else:
            return None  # Hard requirement
    else:
        reasons.append("Bollinger lower unavailable, skipped")

    # 2. RSI < 30 (oversold)
    if snapshot.rsi_14 is not None:
        total_weight += _WEIGHT_RSI
        if snapshot.rsi_14 < 30:
            weighted_score += _WEIGHT_RSI
            confirming.append("rsi_oversold")
            reasons.append(f"RSI ({snapshot.rsi_14:.1f}) < 30")
        else:
            return None  # Hard requirement
    else:
        reasons.append("RSI unavailable, skipped")

    # 3. Stochastic RSI < 0.2
    if snapshot.stoch_rsi is not None:
        total_weight += _WEIGHT_STOCH_RSI
        if snapshot.stoch_rsi < 0.2:
            weighted_score += _WEIGHT_STOCH_RSI
            confirming.append("stoch_rsi_oversold")
            reasons.append(f"Stoch RSI ({snapshot.stoch_rsi:.2f}) < 0.2")
        else:
            return None  # Hard requirement
    else:
        reasons.append("Stochastic RSI unavailable, skipped")

    if not confirming:
        return None

    # Confidence = weighted score adjusted for data completeness
    if total_weight > 0:
        raw_confidence = weighted_score / total_weight
    else:
        raw_confidence = 0.0

    data_completeness = total_weight / (_WEIGHT_BOLLINGER + _WEIGHT_RSI + _WEIGHT_STOCH_RSI)
    confidence = raw_confidence * data_completeness

    if confidence < 0.3:
        return None

    # Stop/take using ATR with mean-reversion targets
    stop_loss, take_profit = _compute_stop_take(price, snapshot.atr_14, snapshot.bollinger_mid)

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
    """Check mean-reversion CLOSE conditions.

    CLOSE when: price > bollinger_upper AND RSI > 70 AND stoch_rsi > 0.8
    Any single condition is an OR trigger for exit.
    """
    price = snapshot.price
    confirming: list[str] = []
    reasons: list[str] = []

    triggered = False

    # Price above upper Bollinger Band
    if snapshot.bollinger_upper is not None:
        if price > snapshot.bollinger_upper:
            triggered = True
            confirming.append("above_bollinger_upper")
            reasons.append(
                f"Price ({price:.2f}) > Bollinger upper ({snapshot.bollinger_upper:.2f})"
            )

    # RSI > 70 (overbought)
    if snapshot.rsi_14 is not None:
        if snapshot.rsi_14 > 70:
            triggered = True
            confirming.append("rsi_overbought")
            reasons.append(f"RSI ({snapshot.rsi_14:.1f}) > 70")

    # Stochastic RSI > 0.8
    if snapshot.stoch_rsi is not None:
        if snapshot.stoch_rsi > 0.8:
            triggered = True
            confirming.append("stoch_rsi_overbought")
            reasons.append(f"Stoch RSI ({snapshot.stoch_rsi:.2f}) > 0.8")

    if not triggered:
        return None

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
    bollinger_mid: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    """Compute stop-loss and take-profit for a mean-reversion buy.

    Stop loss = price - 2*ATR.
    Take profit = bollinger_mid (reversion target) if available, else price + 3*ATR.
    Ensures at least a 2:1 reward-to-risk ratio.
    """
    if atr is None or atr <= 0:
        # Fall back to bollinger mid if available
        if bollinger_mid is not None:
            return None, round(bollinger_mid, 4)
        return None, None

    stop_loss = round(price - 2.0 * atr, 4)
    risk = price - stop_loss  # = 2 * ATR

    if bollinger_mid is not None and bollinger_mid > price:
        reward = bollinger_mid - price
        # Enforce minimum 2:1 R/R
        if reward >= 2.0 * risk:
            take_profit = round(bollinger_mid, 4)
        else:
            take_profit = round(price + 3.0 * atr, 4)
    else:
        take_profit = round(price + 3.0 * atr, 4)

    return stop_loss, take_profit
