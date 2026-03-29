"""Tests for circuit breaker — targeting 100% branch coverage."""

from datetime import datetime, timedelta, timezone

import pytest

from src.risk.circuit_breaker import CircuitBreaker, HaltReason
from src.risk.limits import RiskLimits


def _limits(**overrides) -> RiskLimits:
    defaults = dict(
        consecutive_losses_halt=5,
        halt_duration_minutes=60,
        vix_threshold=35.0,
    )
    defaults.update(overrides)
    return RiskLimits(**defaults)


class TestIsActive:
    def test_initially_inactive(self):
        cb = CircuitBreaker(_limits())
        assert cb.is_active is False
        assert cb.halt_reason is None
        assert cb.halt_until is None

    def test_active_after_trigger(self):
        cb = CircuitBreaker(_limits())
        cb.trigger(HaltReason.MANUAL)
        assert cb.is_active is True
        assert cb.halt_reason == HaltReason.MANUAL
        assert cb.halt_until is not None

    def test_expires_after_duration(self):
        cb = CircuitBreaker(_limits(halt_duration_minutes=0))
        cb.trigger(HaltReason.MANUAL, duration_minutes=0)
        # Force expiry by setting halt_until to past
        cb._halt_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert cb.is_active is False
        assert cb.halt_reason is None
        assert cb.halt_until is None


class TestConsecutiveLosses:
    def test_below_threshold(self):
        cb = CircuitBreaker(_limits(consecutive_losses_halt=5))
        assert cb.check_consecutive_losses(4) is False
        assert cb.is_active is False

    def test_at_threshold(self):
        cb = CircuitBreaker(_limits(consecutive_losses_halt=5))
        assert cb.check_consecutive_losses(5) is True
        assert cb.is_active is True
        assert cb.halt_reason == HaltReason.CONSECUTIVE_LOSSES

    def test_above_threshold(self):
        cb = CircuitBreaker(_limits(consecutive_losses_halt=5))
        assert cb.check_consecutive_losses(10) is True
        assert cb.is_active is True


class TestDailyLoss:
    def test_within_limit(self):
        cb = CircuitBreaker(_limits(max_daily_loss_pct=0.03))
        assert cb.check_daily_loss(-0.02) is False
        assert cb.is_active is False

    def test_at_limit(self):
        cb = CircuitBreaker(_limits(max_daily_loss_pct=0.03))
        assert cb.check_daily_loss(-0.03) is True
        assert cb.is_active is True
        assert cb.halt_reason == HaltReason.DAILY_LOSS_LIMIT

    def test_positive_pnl(self):
        cb = CircuitBreaker(_limits())
        assert cb.check_daily_loss(0.05) is False


class TestWeeklyLoss:
    def test_within_limit(self):
        cb = CircuitBreaker(_limits(max_weekly_loss_pct=0.07))
        assert cb.check_weekly_loss(-0.05) is False

    def test_at_limit(self):
        cb = CircuitBreaker(_limits(max_weekly_loss_pct=0.07))
        assert cb.check_weekly_loss(-0.07) is True
        assert cb.halt_reason == HaltReason.WEEKLY_LOSS_LIMIT


class TestDrawdown:
    def test_within_limit(self):
        cb = CircuitBreaker(_limits(max_drawdown_pct=0.15))
        assert cb.check_drawdown(0.10) is False

    def test_at_limit(self):
        cb = CircuitBreaker(_limits(max_drawdown_pct=0.15))
        assert cb.check_drawdown(0.15) is True
        assert cb.halt_reason == HaltReason.MAX_DRAWDOWN


class TestVIX:
    def test_below_threshold(self):
        cb = CircuitBreaker(_limits(vix_threshold=35.0))
        assert cb.check_vix(25.0) is False

    def test_at_threshold(self):
        cb = CircuitBreaker(_limits(vix_threshold=35.0))
        assert cb.check_vix(35.0) is True
        assert cb.halt_reason == HaltReason.VIX_THRESHOLD


class TestSystemErrors:
    def test_single_error_no_trigger(self):
        cb = CircuitBreaker(_limits())
        assert cb.increment_error_count() is False
        assert cb.is_active is False

    def test_three_errors_triggers(self):
        cb = CircuitBreaker(_limits())
        cb.increment_error_count()
        cb.increment_error_count()
        assert cb.increment_error_count() is True
        assert cb.halt_reason == HaltReason.SYSTEM_ERRORS

    def test_old_errors_expire(self):
        cb = CircuitBreaker(_limits())
        # Add errors in the past
        old_time = datetime.now(timezone.utc) - timedelta(seconds=600)
        cb._error_timestamps = [old_time, old_time]
        # New error should not trigger (old ones expired)
        assert cb.increment_error_count() is False


class TestManualControls:
    def test_manual_halt(self):
        cb = CircuitBreaker(_limits())
        cb.manual_halt()
        assert cb.is_active is True
        assert cb.halt_reason == HaltReason.MANUAL

    def test_manual_halt_custom_duration(self):
        cb = CircuitBreaker(_limits())
        cb.manual_halt(duration_minutes=5)
        assert cb.is_active is True

    def test_manual_resume(self):
        cb = CircuitBreaker(_limits())
        cb.manual_halt()
        assert cb.is_active is True
        cb.manual_resume()
        assert cb.is_active is False
        assert cb.halt_reason is None

    def test_resume_when_not_active(self):
        cb = CircuitBreaker(_limits())
        cb.manual_resume()  # should not raise
        assert cb.is_active is False


class TestReset:
    def test_full_reset(self):
        cb = CircuitBreaker(_limits())
        cb.manual_halt()
        cb.increment_error_count()
        cb.increment_error_count()
        cb.reset()
        assert cb.is_active is False
        assert cb.halt_reason is None
        assert len(cb._error_timestamps) == 0


class TestTriggerDuration:
    def test_default_duration(self):
        cb = CircuitBreaker(_limits(halt_duration_minutes=60))
        cb.trigger(HaltReason.MANUAL)
        expected_min = datetime.now(timezone.utc) + timedelta(minutes=59)
        expected_max = datetime.now(timezone.utc) + timedelta(minutes=61)
        assert expected_min < cb._halt_until < expected_max

    def test_custom_duration(self):
        cb = CircuitBreaker(_limits())
        cb.trigger(HaltReason.MANUAL, duration_minutes=5)
        expected_min = datetime.now(timezone.utc) + timedelta(minutes=4)
        expected_max = datetime.now(timezone.utc) + timedelta(minutes=6)
        assert expected_min < cb._halt_until < expected_max
