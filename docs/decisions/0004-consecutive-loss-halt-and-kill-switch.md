# ADR-0004: Consecutive-loss halt and manual kill switch (JEANS)

## Status
Accepted

## Date
2026-07-06

## Context
A third-generation rewrite in the same lineage as the reference projects
behind [[0002-sector-hourly-capital-shield-limits]] and
[[0003-atr-stop-trailing-session-gating-risk-sizing]] (APEXBOT -> LAKSHMI ->
"JEANS") adds several more safety checks. Two were cheap, non-conflicting,
and clearly applicable regardless of whether this project's model signal
turns out to have real edge:

1. The existing daily circuit breaker ([[0001-portfolio-circuit-breaker-not-per-trade-stop-loss]])
   only fires once *cumulative* PnL breaches a threshold -- a cold streak of
   many small losses in a row can be a meaningful signal on its own (model
   drift, a regime shift) well before it adds up to a 2% daily loss.
2. There was no way to pause live trading mid-session without editing `.env`
   and restarting the process -- awkward and slow if something looks wrong
   while `live_session.py` is running unattended.

JEANS also has a multi-day peak-equity drawdown guard (halt if equity drops
>20% from a running high-water-mark, tracked *across* days). That one was
**not** adopted -- it needs equity state to persist across
`live_session.py`'s daily process restarts (a state file or DB table), a
bigger lift than the two below, and not clearly justified before the
model's own edge is validated (see [[0003-atr-stop-trailing-session-gating-risk-sizing]]
for how that validation question is currently unresolved).

## Decision
Add two independent checks in `PortfolioManager.evaluate_signal`, both
before any sizing/session logic runs:

- **Consecutive-loss halt** (`MAX_CONSECUTIVE_LOSSES`, default 5):
  `close_trade` increments `consecutive_losses` on a loss, resets it to 0 on
  a win. Once the counter reaches the limit, no new entries are approved for
  the rest of the day. Independent of `DAILY_LOSS_LIMIT_PCT` -- this catches
  a cold streak of many small losses even before cumulative PnL breaches the
  daily circuit breaker. Resets in `reset_day()` alongside the other daily
  counters.
- **Manual kill switch** (`KILL_SWITCH_PATH`, default `./data/KILL_SWITCH`):
  `is_kill_switch_engaged()` checks whether that file exists, fresh on every
  call -- no caching, no polling loop. Creating the file blocks all new
  entries immediately (existing positions still exit normally via their
  stop/trail or day-end flatten); deleting it resumes trading. Useful as a
  manual override during a live session without touching `.env` or
  restarting the process.

## Alternatives Considered

### Multi-day peak-equity drawdown guard (full JEANS design)
- Pros: catches slow bleed across many days, which none of the existing
  daily-reset mechanisms can see
- Cons: needs equity state to persist across `live_session.py`'s daily
  process restarts -- a state file or DB table, meaningfully more
  infrastructure than a same-day counter or a file-existence check
- Rejected for now: not proportionate given the model's edge is itself
  still unvalidated (see [[0003-atr-stop-trailing-session-gating-risk-sizing]]);
  worth revisiting if the system runs long enough that multi-day drawdown
  becomes a real, distinct concern from single-day losses

### A remote/API-driven kill switch (e.g. a flag in the SQLite journal)
- Pros: could be toggled without filesystem access to the EC2 instance
- Rejected: the file-based approach needs zero new infrastructure and is
  trivially operable over SSH, which is already the deployment model (see
  README "Deployment (AWS EC2)")

## Consequences
- `MAX_CONSECUTIVE_LOSSES` and `DAILY_LOSS_LIMIT_PCT` are independent halts
  with different trigger conditions -- a config that sets one very loose and
  the other very tight can produce surprising behavior (e.g. many small
  losses halting trading well before the "2% of capital" framing would
  suggest). Tune them together, not independently.
- The kill switch has no audit trail of who created/removed the file or
  when -- acceptable for a single-operator system, but would need
  revisiting (e.g. logging kill-switch state transitions to the journal) if
  more than one person can touch the deploy.
- Both mechanisms are daily-scoped (reset in `reset_day()`); neither
  protects against a multi-day drawdown pattern -- see the rejected
  alternative above.
