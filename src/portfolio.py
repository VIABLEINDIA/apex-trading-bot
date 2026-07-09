"""Phase 3: The Portfolio Risk Manager (the Gatekeeper).

Enforces institutional capital-preservation rules so the bot cannot
over-leverage during a market surge:
  - Hard limit of `max_slots` concurrent open positions.
  - Exactly `capital_per_slot_pct` of total capital allocated per slot
    (unless risk-based sizing off an ATR stop distance produces a smaller
    quantity -- see evaluate_signal).
  - A ticker can never have two simultaneous open positions (dedup via
    `active_trades` hash-map).
  - Per-trade exits via an ATR-based stop + trailing profit-lock (see
    check_exits), not a fixed-% stop -- backtesting showed a fixed-%
    stop-loss makes results *worse* (see README "Risk management"); ATR
    scaling adapts to each stock's own volatility instead of imposing one
    blanket threshold.
"""
import logging
import math
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import Path

import pandas as pd

from src.config import settings

IST = "Asia/Kolkata"

logger = logging.getLogger(__name__)

# Graduated position-size reduction as today's drawdown grows: (drawdown
# fraction of total capital, size multiplier). Ramps size down in steps
# *before* the circuit breaker's hard stop at -DAILY_LOSS_LIMIT_PCT (2% by
# default) -- adapted from APEXBOT's 5-tier capital shield, but rescaled: its
# original tiers span up to >10% drawdown, which would never fire here since
# our circuit breaker already halts all new entries at 2%. These tiers instead
# occupy the 0-2% range so the graduated response has room to actually matter.
CAPITAL_SHIELD_TIERS = [
    (0.005, 1.00),   # DD < 0.5%: full size
    (0.010, 0.75),   # DD < 1.0%: 75% size
    (0.015, 0.50),   # DD < 1.5%: 50% size
    (0.020, 0.25),   # DD < 2.0%: 25% size (circuit breaker trips at 2.0%)
]


@dataclass
class Trade:
    ticker: str
    entry_price: float
    quantity: int
    opened_at: datetime
    order_id: str | None = None
    atr_at_entry: float | None = None
    stop_price: float | None = None
    trail_activated: bool = False


@dataclass
class Decision:
    approved: bool
    reason: str
    quantity: int = 0
    capital_allocated: float = 0.0


def _parse_hhmm(s: str) -> dt_time:
    hour, minute = s.split(":")
    return dt_time(int(hour), int(minute))


class PortfolioManager:
    def __init__(
        self,
        total_capital: float | None = None,
        max_slots: int | None = None,
        capital_per_slot_pct: float | None = None,
        daily_loss_limit_pct: float | None = None,
        max_positions_per_sector: int | None = None,
        sector_map: dict[str, str] | None = None,
        max_trades_per_hour: int | None = None,
        atr_stop_mult: float | None = None,
        atr_trail_activation_mult: float | None = None,
        atr_trail_distance_mult: float | None = None,
        risk_per_trade_pct: float | None = None,
        primary_session_end: str | None = None,
        continuation_session_end: str | None = None,
        continuation_confidence_bonus: float | None = None,
        continuation_size_mult: float | None = None,
        max_consecutive_losses: int | None = None,
        kill_switch_path: str | None = None,
        regime_gate: object | None = None,
    ):
        self.total_capital = total_capital if total_capital is not None else settings.risk.total_capital
        self.max_slots = max_slots if max_slots is not None else settings.risk.max_slots
        self.capital_per_slot_pct = (
            capital_per_slot_pct if capital_per_slot_pct is not None else settings.risk.capital_per_slot_pct
        )
        self.daily_loss_limit_pct = (
            daily_loss_limit_pct if daily_loss_limit_pct is not None else settings.risk.daily_loss_limit_pct
        )
        self.max_positions_per_sector = (
            max_positions_per_sector if max_positions_per_sector is not None
            else settings.risk.max_positions_per_sector
        )
        # {} (rather than None) means "no sector known for anyone" -- the
        # concentration check below becomes a no-op rather than raising, so
        # this stays backward-compatible for callers that don't supply one
        # (e.g. existing tests, or a universe without an Industry column).
        self.sector_map = sector_map or {}
        self.max_trades_per_hour = (
            max_trades_per_hour if max_trades_per_hour is not None else settings.risk.max_trades_per_hour
        )
        self.atr_stop_mult = atr_stop_mult if atr_stop_mult is not None else settings.risk.atr_stop_mult
        self.atr_trail_activation_mult = (
            atr_trail_activation_mult if atr_trail_activation_mult is not None
            else settings.risk.atr_trail_activation_mult
        )
        self.atr_trail_distance_mult = (
            atr_trail_distance_mult if atr_trail_distance_mult is not None
            else settings.risk.atr_trail_distance_mult
        )
        self.risk_per_trade_pct = (
            risk_per_trade_pct if risk_per_trade_pct is not None else settings.risk.risk_per_trade_pct
        )
        self.primary_session_end = _parse_hhmm(
            primary_session_end if primary_session_end is not None else settings.risk.primary_session_end
        )
        self.continuation_session_end = _parse_hhmm(
            continuation_session_end if continuation_session_end is not None
            else settings.risk.continuation_session_end
        )
        self.continuation_confidence_bonus = (
            continuation_confidence_bonus if continuation_confidence_bonus is not None
            else settings.risk.continuation_confidence_bonus
        )
        self.continuation_size_mult = (
            continuation_size_mult if continuation_size_mult is not None else settings.risk.continuation_size_mult
        )
        self.max_consecutive_losses = (
            max_consecutive_losses if max_consecutive_losses is not None else settings.risk.max_consecutive_losses
        )
        self.kill_switch_path = (
            kill_switch_path if kill_switch_path is not None else settings.risk.kill_switch_path
        )
        # Optional market-regime gate: pauses new entries (existing positions
        # still exit normally) independent of the model's own signal. None
        # (the default) is a full no-op, matching the pattern of every other
        # optional gate here (sector_map, timestamp, atr): existing callers
        # that don't supply one are unaffected. A single object covers both
        # halves of the gate -- a zero-arg `check() -> bool` (or, for a
        # simple stateless gate, just being directly callable) and an
        # optional `on_timestamp(timestamp)` for gates that need to know
        # "as of when" (see src/regime.py:RealizedVolatilityGate) -- one
        # parameter rather than two that a caller would otherwise have to
        # keep in sync by hand. Not wired into any default pipeline
        # (paper_trade.py/live_session.py) since it hasn't been backtested
        # yet -- same "implemented but unvalidated" caveat as the ATR-stop
        # mechanism, see docs/decisions/0003.
        self.regime_gate = regime_gate

        self.active_trades: dict[str, Trade] = {}
        # (date, hour) -> count. Bucketed on the timestamp the *caller* passes
        # to evaluate_signal/open_trade, not wall-clock time -- during a
        # backtest "now" is meaningless; what matters is the hour the
        # simulated bar belongs to.
        self._trade_counts_by_hour: dict[tuple, int] = {}

        self.realized_pnl_today = 0.0
        self.circuit_breaker_tripped = False
        self.circuit_breaker_reason: str | None = None
        self.consecutive_losses = 0

    @property
    def slot_capital(self) -> float:
        return self.total_capital * self.capital_per_slot_pct

    @property
    def open_slots(self) -> int:
        return len(self.active_trades)

    @property
    def has_open_slot(self) -> bool:
        return self.open_slots < self.max_slots

    @property
    def daily_loss_limit(self) -> float:
        return self.total_capital * self.daily_loss_limit_pct

    def is_already_active(self, ticker: str) -> bool:
        return ticker in self.active_trades

    def sector_position_count(self, sector: str) -> int:
        return sum(1 for t in self.active_trades if self.sector_map.get(t) == sector)

    @staticmethod
    def _hour_bucket(timestamp) -> tuple:
        return (timestamp.normalize(), timestamp.hour)

    def trades_opened_in_hour(self, timestamp) -> int:
        return self._trade_counts_by_hour.get(self._hour_bucket(timestamp), 0)

    def session_state(self, timestamp) -> tuple[bool, float, float]:
        """Returns (allowed, extra_confidence_required, size_multiplier) for
        the session `timestamp`'s wall-clock time falls into:
          - before primary_session_end: normal threshold, full size
          - before continuation_session_end: threshold + confidence bonus,
            reduced size
          - at/after continuation_session_end: no new entries at all

        `timestamp=None` skips session gating entirely (allowed, no bonus,
        full size) -- backward compatible for callers that don't care about
        pacing, matching the pattern used by the hourly-budget check.
        """
        if timestamp is None:
            return True, 0.0, 1.0

        t = timestamp.time()
        if t < self.primary_session_end:
            return True, 0.0, 1.0
        if t < self.continuation_session_end:
            return True, self.continuation_confidence_bonus, self.continuation_size_mult
        return False, 0.0, 0.0

    def mark_to_market(self, current_prices: dict[str, float]) -> float:
        """Unrealized PnL of every open position, using whatever prices the
        caller currently has (missing tickers are skipped, not assumed flat)."""
        total = 0.0
        for ticker, trade in self.active_trades.items():
            price = current_prices.get(ticker)
            if price is not None:
                total += (price - trade.entry_price) * trade.quantity
        return total

    def capital_shield_multiplier(self, current_prices: dict[str, float] | None = None) -> float:
        """Fraction of normal slot_capital to size the next trade at, based on
        today's drawdown so far (same realized+unrealized PnL basis as
        check_circuit_breaker). 1.0 means full size; ramps down as the day
        gets worse, before the circuit breaker's hard stop kicks in."""
        unrealized = self.mark_to_market(current_prices) if current_prices else 0.0
        total_pnl_today = self.realized_pnl_today + unrealized
        if total_pnl_today >= 0:
            return 1.0

        drawdown_pct = -total_pnl_today / self.total_capital
        for threshold, multiplier in CAPITAL_SHIELD_TIERS:
            if drawdown_pct < threshold:
                return multiplier
        return CAPITAL_SHIELD_TIERS[-1][1]  # deepest defined tier short of the circuit breaker's full stop

    def check_circuit_breaker(self, current_prices: dict[str, float] | None = None) -> bool:
        """Combine realized + unrealized PnL for the day; trip the breaker the
        first time it breaches -daily_loss_limit_pct of total capital.

        Returns True only on the call that *newly* trips it, so callers can
        force-flatten exactly once instead of re-triggering every tick.
        """
        if self.circuit_breaker_tripped:
            return False

        unrealized = self.mark_to_market(current_prices) if current_prices else 0.0
        total_pnl_today = self.realized_pnl_today + unrealized

        if total_pnl_today <= -self.daily_loss_limit:
            self.circuit_breaker_tripped = True
            self.circuit_breaker_reason = (
                f"daily PnL {total_pnl_today:.2f} breached -{self.daily_loss_limit:.2f} "
                f"({self.daily_loss_limit_pct:.1%} of capital)"
            )
            logger.critical("CIRCUIT BREAKER TRIPPED: %s", self.circuit_breaker_reason)
            return True
        return False

    def check_exits(self, current_prices: dict[str, float]) -> list[tuple[str, float, str]]:
        """Check every open position's ATR stop / trailing stop against the
        latest known price, ratcheting the trailing stop up as price moves in
        our favor (never down). Returns (ticker, exit_price, reason) for
        positions that should be closed now -- the caller is responsible for
        actually calling close_trade() and logging; this method only mutates
        trailing-stop state.

        Positions opened without an `atr` (stop_price is None) are skipped
        entirely -- this is a no-op unless the caller opts a trade into it by
        supplying `atr` to open_trade().
        """
        triggered = []
        for ticker, trade in self.active_trades.items():
            if trade.stop_price is None:
                continue
            price = current_prices.get(ticker)
            if price is None:
                continue

            if not trade.trail_activated:
                activation_distance = self.atr_trail_activation_mult * trade.atr_at_entry
                if price - trade.entry_price >= activation_distance:
                    trade.trail_activated = True
                    trade.stop_price = price - self.atr_trail_distance_mult * trade.atr_at_entry
            else:
                candidate_stop = price - self.atr_trail_distance_mult * trade.atr_at_entry
                if candidate_stop > trade.stop_price:
                    trade.stop_price = candidate_stop

            if price <= trade.stop_price:
                reason = "trailing stop" if trade.trail_activated else "ATR stop"
                triggered.append((ticker, price, reason))

        return triggered

    def reset_day(self) -> None:
        """Call at the start of each new trading day (paper-trade day rollover,
        or a fresh live_session.py process at market open)."""
        self.realized_pnl_today = 0.0
        self.circuit_breaker_tripped = False
        self.circuit_breaker_reason = None
        self._trade_counts_by_hour.clear()
        self.consecutive_losses = 0

    def is_kill_switch_engaged(self) -> bool:
        return bool(self.kill_switch_path) and Path(self.kill_switch_path).exists()

    def on_timestamp(self, timestamp) -> None:
        """Optional per-simulated-timestamp hook: forwards to
        `regime_gate.on_timestamp(timestamp)` if a regime_gate was supplied
        AND it exposes that method (a plain stateless callable/gate has
        nothing to update), otherwise a no-op. A backtest has no wall clock,
        so a regime check that needs "as of when" (like RealizedVolatilityGate)
        can't just call pd.Timestamp.now() the way live code could --
        scripts/paper_trade.py's simulate() calls this once per simulated
        timestamp, before evaluating any signals at that instant, so the gate
        can answer using only history up to that point (no look-ahead)."""
        if self.regime_gate is not None and hasattr(self.regime_gate, "on_timestamp"):
            self.regime_gate.on_timestamp(timestamp)

    def evaluate_signal(self, ticker: str, confidence: float, price: float, timestamp=None,
                        current_prices: dict[str, float] | None = None, atr: float | None = None) -> Decision:
        """Gatekeeper check run before any order is routed to the exchange.

        `timestamp`, if given, checks the hourly trade budget AND the
        trading-window session (see session_state) for that time. Omit it
        (e.g. in tests that don't care about pacing) and both checks are
        skipped entirely.

        `current_prices`, if given, feeds the capital shield's mark-to-market
        drawdown calc (see capital_shield_multiplier) so position sizing
        already reflects today's unrealized PnL, not just realized. Omit it
        to size off realized PnL alone.

        `atr`, if given, both sets the ATR-based stop distance (for
        check_exits later) and drives risk-based position sizing: the trade
        is sized so a stop-out loses `risk_per_trade_pct` of total capital,
        capped by the flat slot_capital allocation as a safety ceiling. Omit
        it to fall back to flat slot_capital sizing with no stop at all
        (matching the original no-stop-loss design).
        """
        if self.circuit_breaker_tripped:
            return Decision(False, f"circuit breaker tripped: {self.circuit_breaker_reason}")

        if self.is_kill_switch_engaged():
            return Decision(False, "kill switch engaged")

        if self.regime_gate is not None:
            regime_check = self.regime_gate.check if hasattr(self.regime_gate, "check") else self.regime_gate
            if not regime_check():
                return Decision(False, "unfavorable market regime")

        if self.consecutive_losses >= self.max_consecutive_losses:
            return Decision(False, f"consecutive loss limit reached ({self.max_consecutive_losses})")

        session_allowed, confidence_bonus, session_size_mult = self.session_state(timestamp)
        if not session_allowed:
            return Decision(False, "outside trading window (no new entries after continuation session)")

        effective_threshold = settings.risk.confidence_threshold + confidence_bonus
        if confidence <= effective_threshold:
            return Decision(False, f"confidence {confidence:.2%} below threshold ({effective_threshold:.2%} incl. session)")
        if self.is_already_active(ticker):
            return Decision(False, f"{ticker} already has an open position (dedup)")
        if not self.has_open_slot:
            return Decision(False, f"no free slot ({self.open_slots}/{self.max_slots} used)")

        sector = self.sector_map.get(ticker)
        if sector is not None and self.sector_position_count(sector) >= self.max_positions_per_sector:
            return Decision(False, f"sector limit reached: {sector} already has {self.max_positions_per_sector} open")

        if timestamp is not None and self.trades_opened_in_hour(timestamp) >= self.max_trades_per_hour:
            return Decision(False, f"hourly trade budget reached ({self.max_trades_per_hour}/hour)")

        if price <= 0:
            return Decision(False, "invalid price")

        shield_multiplier = self.capital_shield_multiplier(current_prices)
        size_mult = shield_multiplier * session_size_mult
        flat_quantity = math.floor(self.slot_capital * size_mult / price)

        quantity = flat_quantity
        if atr is not None and atr > 0:
            stop_distance = self.atr_stop_mult * atr
            risk_quantity = math.floor(self.total_capital * self.risk_per_trade_pct * size_mult / stop_distance)
            quantity = min(flat_quantity, risk_quantity)

        if quantity <= 0:
            return Decision(False, "position sizing rounded down to zero shares at current price")

        return Decision(True, "approved", quantity=quantity, capital_allocated=quantity * price)

    def open_trade(self, ticker: str, price: float, quantity: int, order_id: str | None = None,
                   timestamp=None, atr: float | None = None) -> Trade:
        # IST, not UTC: opened_at feeds src.costs' time-of-day slippage lookup,
        # which is keyed to NSE market hours (09:15-15:30 IST wall-clock).
        # `timestamp`, if given, is used verbatim instead of wall-clock "now"
        # -- essential during backtests, where "now" has nothing to do with
        # the simulated bar being processed. Live callers can omit it.
        opened_at = timestamp if timestamp is not None else pd.Timestamp.now(tz=IST)
        stop_price = price - self.atr_stop_mult * atr if atr is not None and atr > 0 else None
        trade = Trade(ticker=ticker, entry_price=price, quantity=quantity,
                       opened_at=opened_at, order_id=order_id,
                       atr_at_entry=atr, stop_price=stop_price)
        self.active_trades[ticker] = trade

        bucket = self._hour_bucket(opened_at)
        self._trade_counts_by_hour[bucket] = self._trade_counts_by_hour.get(bucket, 0) + 1

        logger.info("Opened trade: %s qty=%d @ %.2f stop=%s (slot %d/%d)",
                     ticker, quantity, price, f"{stop_price:.2f}" if stop_price else "n/a",
                     self.open_slots, self.max_slots)
        return trade

    def close_trade(self, ticker: str, exit_price: float) -> float | None:
        trade = self.active_trades.pop(ticker, None)
        if trade is None:
            logger.warning("close_trade called for %s but no active trade found", ticker)
            return None
        pnl = (exit_price - trade.entry_price) * trade.quantity
        self.realized_pnl_today += pnl
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0
        logger.info("Closed trade: %s pnl=%.2f", ticker, pnl)
        return pnl

    def flush_all(self) -> list[str]:
        """Return tickers still open at market close so the caller can force-exit them."""
        return list(self.active_trades.keys())
