"""Market hours and timezone utilities."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from src.config import TradingHoursConfig


def _parse_time(t: str) -> time:
    """Parse an HH:MM string into a time object."""
    parts = t.strip().split(":")
    return time(int(parts[0]), int(parts[1]))


def _now_in_tz(timezone_str: str) -> datetime:
    """Return the current datetime in the given timezone."""
    return datetime.now(ZoneInfo(timezone_str))


def is_market_open(timezone_str: str, hours: TradingHoursConfig) -> bool:
    """Check whether the market is currently open.

    Returns False on weekends (Saturday=5, Sunday=6) and outside
    the configured trading hours window.
    """
    now = _now_in_tz(timezone_str)

    # Weekends are always closed
    if now.weekday() >= 5:
        return False

    market_open = _parse_time(hours.start)
    market_close = _parse_time(hours.end)
    current_time = now.time()

    return market_open <= current_time < market_close


def time_until_market_open(
    timezone_str: str,
    hours: TradingHoursConfig,
) -> timedelta:
    """Return the time remaining until the next market open.

    If the market is currently open, returns ``timedelta(0)``.
    Accounts for weekends by skipping forward to Monday.
    """
    now = _now_in_tz(timezone_str)
    market_open = _parse_time(hours.start)
    market_close = _parse_time(hours.end)
    current_time = now.time()

    # If market is open right now, return zero
    if now.weekday() < 5 and market_open <= current_time < market_close:
        return timedelta(0)

    # Find the next trading day
    target = now.replace(
        hour=market_open.hour,
        minute=market_open.minute,
        second=0,
        microsecond=0,
    )

    if now.weekday() < 5 and current_time < market_open:
        # Same day, before open
        pass
    else:
        # Move to next day
        target += timedelta(days=1)

    # Skip weekends
    while target.weekday() >= 5:
        target += timedelta(days=1)

    return target - now


def time_until_market_close(
    timezone_str: str,
    hours: TradingHoursConfig,
) -> timedelta:
    """Return the time remaining until market close.

    If the market is already closed, returns ``timedelta(0)``.
    """
    now = _now_in_tz(timezone_str)
    market_open = _parse_time(hours.start)
    market_close = _parse_time(hours.end)
    current_time = now.time()

    if now.weekday() >= 5:
        return timedelta(0)

    if not (market_open <= current_time < market_close):
        return timedelta(0)

    close_dt = now.replace(
        hour=market_close.hour,
        minute=market_close.minute,
        second=0,
        microsecond=0,
    )
    return close_dt - now
