# ADR-0002: Sector concentration, hourly trade budget, and graduated capital shield

## Status
Accepted

## Date
2026-07-06

## Context
`PortfolioManager`'s original design ([[0001-portfolio-circuit-breaker-not-per-trade-stop-loss]])
covered the "all-at-once" tail risk (a bad day breaching a daily loss limit)
but had three gaps that a separate, more mature Kotak-Neo-targeting reference
project (not part of this repo) already addressed with a 5-layer risk
architecture:

1. The 10-slot limit says nothing about *correlation* between slots -- 6 of
   10 slots could silently be correlated bank/finance names that move
   together, understating real portfolio risk versus what the slot count
   implies.
2. Nothing paces *entries* within a session -- a string of losses in a bad
   hour could be immediately followed by piling into more trades before the
   model's signals reflect changed conditions.
3. Position sizing was flat (100% of slot capital) right up until the
   circuit breaker's hard stop -- no graduated response as the day gets
   worse but hasn't yet breached the 2% ceiling.

The reference project's own thresholds were calibrated for a >10% drawdown
ceiling, which would never fire alongside our 2% circuit breaker -- so the
concepts were adapted and rescaled to our numbers, not copied verbatim.

## Decision
Add three independent, backward-compatible risk layers to `PortfolioManager`:

- **Sector concentration limit** (`MAX_POSITIONS_PER_SECTOR`, default 3):
  `evaluate_signal` rejects a new entry if `sector_position_count` for that
  ticker's NSE Industry (from `data/nifty500.csv`'s `Industry` column via
  `src/universe.py:load_sector_map`) already has `max_positions_per_sector`
  open. No-ops if no sector map is supplied (`sector_map={}`).
- **Hourly trade budget** (`MAX_TRADES_PER_HOUR`, default 4):
  `trades_opened_in_hour` buckets on an explicit `timestamp` argument passed
  to `evaluate_signal`/`open_trade` -- **never** wall-clock "now" internally.
  During a backtest, "now" is meaningless; what matters is the hour the
  *simulated* bar belongs to. `timestamp=None` skips the check entirely.
- **Capital shield** (`capital_shield_multiplier`): ramps position size down
  in steps -- 100% / 75% / 50% / 25% -- as today's realized+unrealized
  drawdown grows from 0% to 2% (`CAPITAL_SHIELD_TIERS` in `src/portfolio.py`),
  *before* the circuit breaker's hard stop at 2%. The tiers were deliberately
  rescaled to occupy the 0-2% range (rather than the reference project's
  >10% range) so the graduated response has room to actually matter given our
  tighter circuit breaker.

All three are no-ops if their inputs aren't supplied (no `sector_map`, no
`timestamp`, drawdown at/above zero) -- existing tests and call sites that
don't care about them are unaffected.

## Alternatives Considered

### Copy the reference project's thresholds verbatim
- Pros: already tuned and battle-tested in that project
- Rejected: its capital-shield tiers span up to a >10% drawdown ceiling,
  which would never fire given our circuit breaker trips at 2% -- the tiers
  would be dead code in practice. Rescaling to 0-2% was necessary for the
  mechanism to have any effect at all here.

### Hard sector allowlist/blocklist instead of a per-sector count limit
- Pros: simpler to reason about
- Rejected: a count limit degrades gracefully (caps concentration without
  banning any sector outright) and reuses the existing `Industry` column
  already loaded for other purposes

## Consequences
- Sector and hourly limits depend on data quality: `load_sector_map` returns
  `{}` if `data/nifty500.csv` is missing (matching `load_universe`'s own
  fallback), silently turning the sector check into a no-op rather than
  raising -- production runs need the real Nifty 500 CSV for this to do
  anything.
- The hourly budget's correctness depends on every caller passing a
  consistent `timestamp` (simulated-bar time in backtests, current tick time
  live) -- a caller that omits it gets no pacing at all, not a broken one.
- Capital shield and the circuit breaker share the same PnL basis
  (`realized_pnl_today + mark_to_market`), so they're consistent with each
  other by construction, but any future change to one's PnL calculation must
  be mirrored in the other or their thresholds stop meaning what their names
  imply.
