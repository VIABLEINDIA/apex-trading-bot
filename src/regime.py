"""Reference market-regime gate: pauses new entries during high-volatility
"chop" in the broader market, independent of the model's own per-stock
signal. Wires into src/portfolio.py:PortfolioManager's `regime_gate` --
pass an object exposing `check()` (and, for a stateful gate like the one
below, `on_timestamp()`) at construction time; the default (`regime_gate=None`)
is a no-op, so nothing here changes any existing behavior unless a caller opts in.

Why this exists: every backtest so far (see README "Model performance") is
drawn from the same ~2-month calendar window and market regime -- a genuine
regime shift (sharp downtrend, high-vol chop) has never been tested, but is
exactly the situation a regime gate is meant to sit out rather than trade
through blindly.

**Not wired into any default pipeline (paper_trade.py/live_session.py), and
the threshold below is a starting guess, not backtested against this
strategy's actual behavior across regimes** -- same "implemented but
unvalidated" status as the ATR-stop mechanism (docs/decisions/0003) until
someone runs it through a sweep the way that was.
"""
import pandas as pd

BARS_PER_TRADING_DAY = 25  # ~09:15-15:30 IST in 15-min bars
TRADING_DAYS_PER_YEAR = 252


def realized_volatility_regime_ok(benchmark_close: pd.Series, lookback: int = 20,
                                   max_annualized_vol: float = 0.30) -> bool:
    """False when the benchmark's trailing realized volatility (annualized,
    from `lookback` bars of *intraday* returns) exceeds `max_annualized_vol`
    -- a simple "pause during high-vol chop" gate.

    Returns are computed within each calendar day only (`groupby` on the
    normalized date), so the overnight gap between one day's last bar and the
    next day's first bar is never included -- NSE runs a single session per
    day (09:15-15:30 IST, no overnight trading), so a calendar-day boundary
    always lines up with a real session boundary. Without this, a routine
    (non-extreme) overnight gap dominates a 20-bar lookback window and
    spuriously reports a high-vol regime for roughly the first `lookback`
    bars of every single trading day, regardless of that day's actual
    intraday volatility -- this isn't measuring the "chop" the gate exists
    to detect.

    Fails open (returns True, i.e. allows trading) when there isn't enough
    history yet to compute volatility, so this doesn't silently block all
    trading during warm-up -- consistent with how the rest of this codebase
    treats missing inputs as a no-op rather than a hard failure (see
    PortfolioManager's sector_map/atr/timestamp handling).
    """
    returns = benchmark_close.groupby(benchmark_close.index.normalize()).pct_change().dropna()
    if len(returns) < lookback:
        return True

    recent = returns.iloc[-lookback:]
    annualized_vol = recent.std() * (BARS_PER_TRADING_DAY * TRADING_DAYS_PER_YEAR) ** 0.5
    # A degenerate window (e.g. lookback=1) gives an undefined sample std
    # (ddof=1 is NaN for n=1) -- that's still "not enough history to compute
    # volatility", so it must fail open like the count check above, not fail
    # closed (NaN <= threshold is False, which would otherwise block trading).
    if pd.isna(annualized_vol):
        return True
    return bool(annualized_vol <= max_annualized_vol)


class RealizedVolatilityGate:
    """Stateful adapter so a point-in-time regime check can sit behind
    PortfolioManager's `regime_gate`, in a backtest as well as live.

    `check()` has to stay a plain zero-arg call (see tests/test_portfolio.py),
    but `realized_volatility_regime_ok` needs to know "as of when" to look at
    trailing history -- there's no wall clock in a backtest. This class splits
    the two: `on_timestamp(ts)` records the current simulated instant (wired
    to PortfolioManager.on_timestamp, called once per timestamp by
    scripts/paper_trade.py's simulate() before any signals at that instant are
    evaluated), and `check()` answers using only benchmark data up to that
    recorded instant, so there's no look-ahead into future volatility.

    Usage:
        gate = RealizedVolatilityGate(benchmark_close)
        PortfolioManager(regime_gate=gate)

    Live sessions can use this identically by calling `gate.on_timestamp(now)`
    on each tick instead, though that wiring hasn't been done in
    live_session.py yet -- same "implemented but unvalidated in this context"
    status as the rest of this module.
    """

    def __init__(self, benchmark_close: pd.Series, lookback: int = 20, max_annualized_vol: float = 0.30):
        self.benchmark_close = benchmark_close
        self.lookback = lookback
        self.max_annualized_vol = max_annualized_vol
        self._as_of: pd.Timestamp | None = None
        # Fail-open default (see check()'s docstring), matching
        # realized_volatility_regime_ok's own behavior before on_timestamp
        # has ever been called.
        self._cached_ok = True

    def on_timestamp(self, timestamp: pd.Timestamp) -> None:
        """Recomputes and caches the regime verdict for `timestamp`. Doing
        the (pandas slice + rolling-std) work here -- once per simulated
        timestamp -- instead of inside check() matters because
        PortfolioManager.evaluate_signal calls check() once per *candidate
        ticker*, not once per timestamp: on a 500-ticker sweep that's the
        difference between one recomputation per timestamp and one per
        (timestamp x candidates-at-that-timestamp)."""
        self._as_of = timestamp
        history = self.benchmark_close.loc[:timestamp]
        self._cached_ok = realized_volatility_regime_ok(
            history, lookback=self.lookback, max_annualized_vol=self.max_annualized_vol,
        )

    def check(self) -> bool:
        """Fails open (allows trading) if on_timestamp hasn't been called
        yet -- same fail-open default as realized_volatility_regime_ok during
        its own warm-up period."""
        return self._cached_ok
