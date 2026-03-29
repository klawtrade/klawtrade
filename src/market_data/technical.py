"""Technical indicator calculations using pandas-ta."""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


def calculate_indicators(
    df: pd.DataFrame,
    rsi_length: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_periods: tuple[int, ...] = (20, 50, 200),
    ema_periods: tuple[int, ...] = (12, 26),
    bb_length: int = 20,
    bb_std: float = 2.0,
    atr_length: int = 14,
    vol_sma_length: int = 20,
    stoch_rsi_length: int = 14,
) -> dict[str, Optional[float]]:
    """Compute technical indicators on an OHLCV DataFrame.

    Parameters
    ----------
    df:
        Must contain columns ``open, high, low, close, volume`` and have
        at least a few rows.  More rows yield more accurate indicators.

    Returns
    -------
    dict
        Keys correspond to :class:`~src.storage.models.MarketSnapshot`
        indicator fields.  Values are ``None`` when there is insufficient
        data to compute a given indicator.
    """
    result: dict[str, Optional[float]] = {
        "rsi_14": None,
        "macd": None,
        "macd_signal": None,
        "sma_20": None,
        "sma_50": None,
        "sma_200": None,
        "ema_12": None,
        "ema_26": None,
        "bollinger_upper": None,
        "bollinger_lower": None,
        "bollinger_mid": None,
        "atr_14": None,
        "volume_sma_20": None,
        "stoch_rsi": None,
    }

    if df.empty or len(df) < 2:
        logger.warning("Insufficient data for indicator calculation (%d rows)", len(df))
        return result

    close: pd.Series = df["close"]
    high: pd.Series = df["high"]
    low: pd.Series = df["low"]
    volume: pd.Series = df["volume"]

    # --- RSI ---
    result["rsi_14"] = _last_valid(ta.rsi(close, length=rsi_length))

    # --- MACD ---
    macd_df = ta.macd(close, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if macd_df is not None and not macd_df.empty:
        macd_col = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
        signal_col = f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"
        result["macd"] = _last_valid_from_df(macd_df, macd_col)
        result["macd_signal"] = _last_valid_from_df(macd_df, signal_col)

    # --- SMAs ---
    for period in sma_periods:
        key = f"sma_{period}"
        if key in result:
            result[key] = _last_valid(ta.sma(close, length=period))

    # --- EMAs ---
    for period in ema_periods:
        key = f"ema_{period}"
        if key in result:
            result[key] = _last_valid(ta.ema(close, length=period))

    # --- Bollinger Bands ---
    bb_df = ta.bbands(close, length=bb_length, std=bb_std)
    if bb_df is not None and not bb_df.empty:
        result["bollinger_upper"] = _last_valid_from_df(
            bb_df, f"BBU_{bb_length}_{bb_std}"
        )
        result["bollinger_mid"] = _last_valid_from_df(
            bb_df, f"BBM_{bb_length}_{bb_std}"
        )
        result["bollinger_lower"] = _last_valid_from_df(
            bb_df, f"BBL_{bb_length}_{bb_std}"
        )

    # --- ATR ---
    result["atr_14"] = _last_valid(ta.atr(high, low, close, length=atr_length))

    # --- Volume SMA ---
    result["volume_sma_20"] = _last_valid(ta.sma(volume, length=vol_sma_length))

    # --- Stochastic RSI ---
    stoch_rsi_df = ta.stochrsi(close, length=stoch_rsi_length)
    if stoch_rsi_df is not None and not stoch_rsi_df.empty:
        # Use the %K line (first column).
        k_col = stoch_rsi_df.columns[0]
        result["stoch_rsi"] = _last_valid_from_df(stoch_rsi_df, k_col)

    # Round everything for cleanliness.
    for key, value in result.items():
        if value is not None:
            result[key] = round(value, 4)

    return result


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _last_valid(series: Optional[pd.Series]) -> Optional[float]:
    """Return the last non-NaN value from *series*, or ``None``."""
    if series is None or series.empty:
        return None
    last = series.dropna()
    if last.empty:
        return None
    return float(last.iloc[-1])


def _last_valid_from_df(df: pd.DataFrame, col: str) -> Optional[float]:
    """Return the last non-NaN value for *col* in *df*, or ``None``."""
    if col not in df.columns:
        return None
    return _last_valid(df[col])
