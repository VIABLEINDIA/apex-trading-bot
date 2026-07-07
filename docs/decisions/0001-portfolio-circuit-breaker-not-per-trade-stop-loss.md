# ADR-0001: Portfolio-level circuit breaker instead of a per-trade stop-loss

## Status
Accepted

## Date
2026-07-06

## Context
The original strategy design had no per-trade exit at all beyond the end-of-day
flatten. Before adding one, `scripts/analyze_stoploss.py` re-fetched the real
intraday bar path for every trade logged across the walk-forward runs (500
trades) and checked what a fixed-percentage stop-loss would have done at
several thresholds (0.5% / 1.0% / 1.5% / 2.0% / 3.0%):

| Stop threshold | Trades stopped early | Total PnL | Win rate |
|---|---|---|---|
| None (baseline) | -- | +₹9,206.98 | 46.0% |
| 0.5% | 301/500 | +₹7,016.87 | 33.4% |
| 1.0% | 196/500 | +₹6,653.46 | 42.0% |
| 1.5% | 107/500 | +₹7,666.63 | 44.8% |
| 2.0% | 54/500 | +₹8,716.41 | 46.0% |
| 3.0% | 7/500 | +₹9,060.36 | 46.0% |

Every fixed-% stop-loss made results worse, not better. Median intraday
drawdown before close was only 0.73%, and most dips were noise around a
momentum trade that recovers by end of day -- a mechanical stop just cuts
winners off before that recovery. It also doesn't address the actual tail
risk: several traded names are circuit-band-prone on NSE, and a stop order
simply doesn't fill if a stock locks limit-down.

We still needed *some* protection against a genuinely bad day compounding
across many simultaneous positions (up to `MAX_SLOTS`, default 10).

## Decision
Implement a **portfolio-level circuit breaker** in `PortfolioManager`
(`src/portfolio.py`) instead of a per-trade stop-loss:

- `mark_to_market(current_prices)` sums unrealized PnL across all open
  positions; combined with `realized_pnl_today` (updated in `close_trade`),
  this gives the day's total PnL.
- `check_circuit_breaker(current_prices)` trips once
  (`circuit_breaker_tripped = True`) the first time that total breaches
  `-DAILY_LOSS_LIMIT_PCT * TOTAL_CAPITAL` (default 2%), and returns `True`
  only on the call that newly trips it -- callers force-flatten exactly once
  instead of re-triggering every tick.
- Once tripped, `evaluate_signal` rejects all new entries for the rest of the
  day, but existing open positions are otherwise untouched by this mechanism
  (they still rely on end-of-day flatten, or later the ATR stop from
  [[0003-atr-stop-trailing-session-gating-risk-sizing]]).
- `reset_day()` clears the flag and counters at the start of each new trading
  day.

## Alternatives Considered

### Fixed-percentage per-trade stop-loss
- Pros: simple, industry-standard, easy to reason about per trade
- Cons: empirically measured to reduce total PnL at every threshold tested;
  cuts winners off during normal intraday noise; doesn't protect against
  limit-down gaps anyway
- Rejected: the data above directly contradicts the "stops always help"
  assumption for this specific momentum strategy

### Multi-day peak-equity drawdown guard (halt if equity drops >20% from a
running high-water-mark across days, adapted from a later "JEANS" design)
- Pros: catches slow multi-day bleed, not just single bad days
- Cons: needs equity state to persist across `live_session.py`'s daily
  process restarts (a state file or DB table) -- a bigger lift than a
  same-day breaker
- Rejected for now: not adopted; worth revisiting if the system runs long
  enough that multi-day drawdown becomes a real concern distinct from
  single-day losses (see [[0004-consecutive-loss-halt-and-kill-switch]] for
  what *was* adopted from the same design)

## Consequences
- Protects against the untested tail (a bad day compounding across many
  positions) without cutting off the per-trade noise that the stop-loss
  analysis showed actively hurts this strategy.
- `DAILY_LOSS_LIMIT_PCT` (default 0.02) is the single tunable knob; the 2%
  ceiling was chosen so [[0002-sector-hourly-capital-shield-limits]]'s
  graduated capital-shield tiers (0-2% drawdown) have room to matter before
  the breaker fires.
- Does not protect an individual position from a large single-name move
  (gap down, circuit-band lock) between ticks -- only caps aggregate daily
  loss across the whole portfolio.
- This design was later revisited per-trade via ATR-based stops (see
  [[0003-atr-stop-trailing-session-gating-risk-sizing]]), which is a
  fundamentally different mechanism (volatility-adjusted, only ratchets in
  the trader's favor) from the fixed-% stop rejected here.
