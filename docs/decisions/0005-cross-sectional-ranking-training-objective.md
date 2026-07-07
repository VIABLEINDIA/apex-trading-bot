# ADR-0005: Cross-sectional ranking as an alternative training objective

## Status
Accepted (implementation) -- **result is a single promising data point, not a
validated improvement.** Same caveat pattern as
[[0003-atr-stop-trailing-session-gating-risk-sizing]]: do not read this ADR
as evidence the model is now profitable.

## Date
2026-07-07

## Context
`scripts/tune_model.py`'s sweep (see README "Model-signal sweep") showed every
absolute-direction model variant tested -- different label horizons (1/3/5
bars), a more regularized GBM -- landed within 0.003 of the same ~52%
holdout accuracy. That pointed at the training objective itself, not
hyperparameters, as the ceiling: the model has only ever been asked "will
this stock's price go up in isolation," independently per ticker.

Meanwhile `scripts/paper_trade.py`'s `simulate()` already **executes**
cross-sectionally: every candidate at a timestamp is scored, then slots are
filled strongest-confidence-first across the whole watchlist, not
ticker-by-ticker. The training objective was never changed to match that.

## Decision
Add `src/cross_sectional.py:build_cross_sectional_training_set`, which labels
each bar by its forward-return **percentile rank within the cross-section of
tickers with data at that same timestamp** (top/bottom quantile -> 1/0, dead
zone in between dropped -- same dead-zone concept as `add_labels`, but
relative instead of absolute), instead of an absolute epsilon threshold.
Feeds directly into `MomentumClassifier.train_prelabeled` (fixed in the same
change to genuinely time-split rather than evaluate in-sample -- see the
commit history for that latent bug).

`scripts/tune_model_cross_sectional.py` runs this alongside the existing
absolute-baseline variant against the same fetch and cutoff, mirroring
`tune_model.py`/`tune_strategy.py`'s "fetch once, vary cheaply" methodology
so the comparison is apples-to-apples rather than against a prior run's
numbers.

## Result (one window: 59-day train, 14-day holdout ending 2026-07-07, full
500-ticker universe, ATR-stop off, full-day session)

| Variant | Holdout acc | Net PnL | Net WR | Gross PnL | Gross WR |
|---|---|---|---|---|---|
| `cross_sectional_q20` | 0.521 | -₹3,007.59 | 35.0% | -₹819.69 | 43.0% |
| `cross_sectional_q30` | 0.524 | -₹3,113.57 | 39.0% | -₹837.36 | 49.0% |
| `absolute_baseline` | 0.520 | -₹4,806.70 | 32.0% | -₹2,557.86 | 38.0% |

Both cross-sectional variants clearly beat the absolute baseline on this
window -- 35-40% smaller net loss, meaningfully higher win rate on both gross
and net -- while holdout **accuracy is essentially unchanged** (~52% either
way). The reformulation appears to change *which* signals the model
selects, not how accurate it is at the underlying prediction task.

**All three configurations are still net-negative.**

## Why this is not being treated as a validated win
This project already has one directly analogous cautionary tale
([[0003-atr-stop-trailing-session-gating-risk-sizing]]): BALU's ATR-stop
mechanism looked clearly better in an early comparison, and a proper
multi-window sweep (`tune_strategy.py`) later showed the *same* baseline
configuration swinging from strongly positive to uniformly negative just
from the trailing window shifting by hours of wall-clock time -- a bigger
effect than the mechanism being compared. A single 14-day window beating
baseline here is the same shape of evidence that turned out to be
unreliable there. It would be inconsistent to trust it now just because the
result happens to point in a more encouraging direction.

## Alternatives Considered

### Declare this validated and flip the default training path
- Rejected outright for the same reason BALU's early result was rejected as
  sufficient evidence: one window is one window, regardless of which
  direction it points.

### Don't test this at all until a multi-window sweep can be run
- Rejected: the single-window result is still useful signal to decide
  *where to invest next* (worth a multi-window sweep) even though it isn't
  sufficient to trust on its own.

## Consequences
- The mechanism is merged and available (`build_cross_sectional_training_set`,
  `scripts/tune_model_cross_sectional.py`), fully additive -- the existing
  absolute-label training path (`build_training_set`, `train_multi`) is
  unchanged and remains the default.
- **Next step before trusting this**: run the same 7/14/21/28-day
  multi-window sweep this project already applies to every other strategy
  change, the same way `tune_strategy.py` did for BALU. That run was not
  done as part of this change (each window costs a full fetch+train+backtest
  cycle, ~1-1.5 hours at full-universe scale) -- treat the numbers above as
  the reason to run it, not a substitute for it.
- If a multi-window sweep confirms cross-sectional ranking is consistently
  least-bad (not just least-bad on one window), the natural next move is
  order-flow/microstructure features on top of it, since accuracy itself
  hasn't moved -- only which trades get selected.
