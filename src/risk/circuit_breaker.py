"""Circuit breaker — automatic trading halt system.

Once triggered, ALL trading stops for the configured duration.
Cannot be overridden by the LLM or any other component.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum

from src.risk.limits import RiskLimits

logger = logging.getLogger(__name__)


class HaltReason(Enum):
    CONSECUTIVE_LOSSES = "consecutive_losses"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    WEEKLY_LOSS_LIMIT = "weekly_loss_limit"
    MAX_DRAWDOWN = "max_drawdown"
    VIX_THRESHOLD = "vix_threshold"
    SYSTEM_ERRORS = "system_errors"
    MANUAL = "manual"


class CircuitBreaker:
    """Automatic trading halt system."""

    def __init__(self, limits: RiskLimits) -> None:
        self._limits = limits
        self._halt_until: datetime | None = None
        self._halt_reason: HaltReason | None = None
        self._error_timestamps: list[datetime] = []
        self._max_errors_in_window = 3
        self._error_window_seconds = 300  # 5 minutes

    @property
    def is_active(self) -> bool:
        """Check if circuit breaker is currently active."""
        if self._halt_until is None:
            return False
        now = datetime.now(timezone.utc)
        if now >= self._halt_until:
            self._halt_until = None
            self._halt_reason = None
            logger.info("Circuit breaker expired, trading resumed")
            return False
        return True

    @property
    def halt_reason(self) -> HaltReason | None:
        if self.is_active:
            return self._halt_reason
        return None

    @property
    def halt_until(self) -> datetime | None:
        if self.is_active:
            return self._halt_until
        return None

    def trigger(self, reason: HaltReason, duration_minutes: int | None = None) -> None:
        """Activate the circuit breaker."""
        duration = duration_minutes or self._limits.halt_duration_minutes
        self._halt_until = datetime.now(timezone.utc) + timedelta(minutes=duration)
        self._halt_reason = reason
        logger.critical(
            "CIRCUIT BREAKER TRIGGERED",
            extra={"reason": reason.value, "halt_until": self._halt_until.isoformat()},
        )

    def check_consecutive_losses(self, consecutive_losses: int) -> bool:
        """Check if consecutive losses exceed threshold. Returns True if triggered."""
        if consecutive_losses >= self._limits.consecutive_losses_halt:
            self.trigger(HaltReason.CONSECUTIVE_LOSSES)
            return True
        return False

    def check_daily_loss(self, daily_pnl_pct: float) -> bool:
        """Check if daily loss exceeds limit. Returns True if triggered."""
        if daily_pnl_pct <= -self._limits.max_daily_loss_pct:
            self.trigger(HaltReason.DAILY_LOSS_LIMIT)
            return True
        return False

    def check_weekly_loss(self, weekly_pnl_pct: float) -> bool:
        """Check if weekly loss exceeds limit. Returns True if triggered."""
        if weekly_pnl_pct <= -self._limits.max_weekly_loss_pct:
            self.trigger(HaltReason.WEEKLY_LOSS_LIMIT)
            return True
        return False

    def check_drawdown(self, drawdown_pct: float) -> bool:
        """Check if drawdown exceeds limit. Returns True if triggered."""
        if drawdown_pct >= self._limits.max_drawdown_pct:
            self.trigger(HaltReason.MAX_DRAWDOWN)
            return True
        return False

    def check_vix(self, vix: float) -> bool:
        """Check if VIX exceeds threshold. Returns True if triggered."""
        if vix >= self._limits.vix_threshold:
            self.trigger(HaltReason.VIX_THRESHOLD)
            return True
        return False

    def increment_error_count(self) -> bool:
        """Record a system error. Returns True if circuit breaker triggered."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self._error_window_seconds)
        self._error_timestamps = [t for t in self._error_timestamps if t > cutoff]
        self._error_timestamps.append(now)

        if len(self._error_timestamps) >= self._max_errors_in_window:
            self.trigger(HaltReason.SYSTEM_ERRORS)
            self._error_timestamps.clear()
            return True
        return False

    def manual_halt(self, duration_minutes: int | None = None) -> None:
        """Manually trigger the circuit breaker (e.g., from dashboard kill switch)."""
        self.trigger(HaltReason.MANUAL, duration_minutes=duration_minutes or 999999)

    def manual_resume(self) -> None:
        """Manually resume trading (clear circuit breaker)."""
        self._halt_until = None
        self._halt_reason = None
        logger.info("Circuit breaker manually cleared, trading resumed")

    def reset(self) -> None:
        """Full reset — clears halt and error history."""
        self._halt_until = None
        self._halt_reason = None
        self._error_timestamps.clear()
