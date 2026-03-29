"""THE risk manager — deterministic, no LLM, no overrides.

Every trade signal passes through here. If it says no, the answer is no.
No appeal, no override, no "just this once."
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.risk.circuit_breaker import CircuitBreaker
from src.risk.limits import RiskLimits
from src.risk.position_sizer import PositionSizer
from src.storage.models import (
    MarketSnapshot,
    PortfolioState,
    RiskCheckResult,
    RiskRejectionReason,
    SignalAction,
    TradeSignal,
)

logger = logging.getLogger(__name__)


class RiskManager:
    """Deterministic risk management gate.

    Runs ALL checks on every signal. Collects ALL rejection reasons
    (does not short-circuit) so we can log and learn from multiple
    simultaneous violations.
    """

    def __init__(self, limits: RiskLimits, circuit_breaker: CircuitBreaker) -> None:
        self._limits = limits
        self._circuit_breaker = circuit_breaker
        self._position_sizer = PositionSizer(limits)

    def check(
        self,
        signal: TradeSignal,
        portfolio: PortfolioState,
        snapshot: MarketSnapshot | None = None,
    ) -> RiskCheckResult:
        """Run all risk checks on a signal.

        Args:
            signal: The trade signal to evaluate.
            portfolio: Current portfolio state.
            snapshot: Current market data for the symbol (optional).

        Returns:
            RiskCheckResult with approval status and any rejection reasons.
        """
        reasons: list[RiskRejectionReason] = []

        # 1. Is the symbol blacklisted?
        if signal.symbol in self._limits.blacklisted_symbols:
            reasons.append(RiskRejectionReason.BLACKLISTED)

        # 2. Is a circuit breaker active?
        if self._circuit_breaker.is_active:
            reasons.append(RiskRejectionReason.CIRCUIT_BREAKER)

        # 3. Has the signal expired?
        now = datetime.now(timezone.utc)
        if now > signal.expires_at:
            reasons.append(RiskRejectionReason.SIGNAL_EXPIRED)

        # 4. Is signal confidence above minimum?
        if signal.confidence < self._limits.min_confidence:
            reasons.append(RiskRejectionReason.LOW_CONFIDENCE)

        # 5. Have we hit max daily trades?
        if portfolio.trades_today >= self._limits.max_daily_trades:
            reasons.append(RiskRejectionReason.MAX_DAILY_TRADES)

        # Checks 6-14 only apply to BUY signals
        if signal.action == SignalAction.BUY:
            reasons.extend(self._check_buy_constraints(signal, portfolio, snapshot))

        # Build result
        approved = len(reasons) == 0

        adjusted_quantity = None
        if approved and signal.action == SignalAction.BUY:
            price = signal.suggested_limit_price or (snapshot.price if snapshot else 0)
            if price > 0:
                adjusted_quantity = self._position_sizer.calculate_quantity(
                    signal, portfolio, price
                )
                if adjusted_quantity == 0:
                    approved = False
                    reasons.append(RiskRejectionReason.INSUFFICIENT_CASH)

        result = RiskCheckResult(
            approved=approved,
            signal=signal,
            rejection_reasons=reasons,
            adjusted_quantity=adjusted_quantity,
            risk_score=self._calculate_risk_score(signal, portfolio, reasons),
        )

        logger.info(
            "Risk check completed",
            extra={
                "signal_id": signal.id,
                "symbol": signal.symbol,
                "action": signal.action.value,
                "approved": approved,
                "reasons": [r.value for r in reasons],
            },
        )

        return result

    def _check_buy_constraints(
        self,
        signal: TradeSignal,
        portfolio: PortfolioState,
        snapshot: MarketSnapshot | None,
    ) -> list[RiskRejectionReason]:
        """Run buy-specific risk checks."""
        reasons: list[RiskRejectionReason] = []

        # 6. Max open positions
        if len(portfolio.positions) >= self._limits.max_open_positions:
            reasons.append(RiskRejectionReason.MAX_POSITIONS)

        # 7. Volume threshold (if we have market data)
        if snapshot and snapshot.daily_volume < self._limits.min_volume_threshold:
            reasons.append(RiskRejectionReason.LOW_VOLUME)

        # 8. Spread check
        if snapshot and snapshot.bid > 0 and snapshot.ask > 0:
            spread_pct = (snapshot.ask - snapshot.bid) / snapshot.ask
            if spread_pct > self._limits.max_spread_pct:
                reasons.append(RiskRejectionReason.WIDE_SPREAD)

        # 9. Max single position size
        price = signal.suggested_limit_price or (snapshot.price if snapshot else 0)
        if price > 0 and portfolio.total_equity > 0:
            proposed_value = price * signal.suggested_quantity
            # Include existing position value if we already hold some
            existing_value = 0.0
            for pos in portfolio.positions:
                if pos.symbol == signal.symbol:
                    existing_value = pos.market_value
                    break
            total_position_value = existing_value + proposed_value
            if total_position_value > portfolio.total_equity * self._limits.max_single_position_pct:
                reasons.append(RiskRejectionReason.MAX_POSITION_SIZE)

        # 10. Sector allocation
        if snapshot and snapshot.sector:
            sector_value = sum(
                p.market_value for p in portfolio.positions
                if p.sector == snapshot.sector
            )
            if price > 0:
                new_sector_value = sector_value + (price * signal.suggested_quantity)
                if portfolio.total_equity > 0 and new_sector_value > portfolio.total_equity * self._limits.max_sector_allocation_pct:
                    reasons.append(RiskRejectionReason.SECTOR_LIMIT)

        # 11. Correlated exposure
        if snapshot and snapshot.correlation_group:
            corr_value = sum(
                p.market_value for p in portfolio.positions
                if p.correlation_group == snapshot.correlation_group
            )
            if price > 0:
                new_corr_value = corr_value + (price * signal.suggested_quantity)
                if portfolio.total_equity > 0 and new_corr_value > portfolio.total_equity * self._limits.max_correlated_exposure_pct:
                    reasons.append(RiskRejectionReason.CORRELATED_EXPOSURE)

        # 12. Cash reserve
        if price > 0:
            cost = price * signal.suggested_quantity
            remaining_cash = portfolio.cash - cost
            min_reserve = portfolio.total_equity * self._limits.min_cash_reserve_pct
            if remaining_cash < min_reserve:
                reasons.append(RiskRejectionReason.INSUFFICIENT_CASH)

        # 13. Daily loss limit check
        if portfolio.daily_pnl_pct <= -self._limits.max_daily_loss_pct:
            reasons.append(RiskRejectionReason.MAX_DAILY_LOSS)

        # 14. Drawdown limit check
        if portfolio.current_drawdown_pct >= self._limits.max_drawdown_pct:
            reasons.append(RiskRejectionReason.MAX_DRAWDOWN)

        return reasons

    def _calculate_risk_score(
        self,
        signal: TradeSignal,
        portfolio: PortfolioState,
        reasons: list[RiskRejectionReason],
    ) -> float:
        """Calculate a 0-1 risk score. Higher = riskier."""
        if reasons:
            return 1.0

        score = 0.0

        # Factor in drawdown proximity
        if self._limits.max_drawdown_pct > 0:
            score += 0.3 * (portfolio.current_drawdown_pct / self._limits.max_drawdown_pct)

        # Factor in daily loss proximity
        if self._limits.max_daily_loss_pct > 0:
            daily_loss_ratio = abs(min(0, portfolio.daily_pnl_pct)) / self._limits.max_daily_loss_pct
            score += 0.3 * daily_loss_ratio

        # Factor in position count proximity
        if self._limits.max_open_positions > 0:
            score += 0.2 * (len(portfolio.positions) / self._limits.max_open_positions)

        # Factor in confidence (inverse — low confidence = higher risk)
        score += 0.2 * (1.0 - signal.confidence)

        return min(1.0, max(0.0, score))
