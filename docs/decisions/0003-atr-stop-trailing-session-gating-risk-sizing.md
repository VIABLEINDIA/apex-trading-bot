# ADR-0003: ATR stop / trailing profit-lock, session gating, and risk-based sizing (BALU)

## Status
Accepted (implementation) -- **flagged as an unproven regression, not a
validated improvement**, pending re-tuning or further evidence. See
"Post-implementation validation" below. Do not read "Accepted" here as
"proven to help."

## Date
2026-07-06

## Context
A specific tuned strategy config ("BALU", exported from the same reference
project referenced in [[0002-sector-hourly-capital-shield-limits]]) showed a
much stronger win/loss asymmetry than anything measured in this project
(avg win ~₹185 vs. avg loss ~₹-71, profit factor 3.23, across a genuinely
non-overlapping 4-walk walk-forward in that project). Its config implied
three structural differences from what this project had at the time, not
just more risk gates:

1. No per-trade exit mechanism at all beyond the portfolio circuit breaker
   (see [[0001-portfolio-circuit-breaker-not-per-trade-stop-loss]]) and
   end-of-day flatten.
2. No time-of-day awareness beyond the cost model's slippage table (which
   already flagged the close as the most expensive window to trade, but
   only priced that in rather than avoiding it).
3. Flat position sizing (`slot_capital`-based), with no link between
   position size and a stock's own volatility.

## Decision
Adapt (not copy verbatim) three mechanisms from BALU into `PortfolioManager`:

- **ATR-based stop + trailing profit-lock** (`ATR_STOP_MULT=0.75`,
  `ATR_TRAIL_ACTIVATION_MULT=0.25`, `ATR_TRAIL_DISTANCE_MULT=0.12`):
  `open_trade(..., atr=...)` sets an initial stop at `entry - 0.75*ATR`.
  `check_exits(current_prices)` activates a trailing stop once price has
  moved `0.25*ATR` in our favor, then ratchets it up (never down) to
  `price - 0.12*ATR`. This is fundamentally different from the fixed-%
  stop-loss rejected in [[0001-portfolio-circuit-breaker-not-per-trade-stop-loss]]:
  it's volatility-adjusted per stock and only tightens in the trader's favor.
  Positions opened without an `atr` argument get no stop at all -- fully
  backward compatible.
- **Trading-window session gating** (`session_state`): before
  `PRIMARY_SESSION_END` (10:15), normal threshold and full size. Before
  `CONTINUATION_SESSION_END` (13:15), threshold raised by
  `CONTINUATION_CONFIDENCE_BONUS` (+0.05) and size cut to
  `CONTINUATION_SIZE_MULT` (0.4x). After that, no new entries -- existing
  positions still exit normally.
- **Risk-based position sizing**: when `atr` is supplied, `evaluate_signal`
  sizes the trade so a stop-out loses `RISK_PER_TRADE_PCT` (1.3%) of total
  capital, capped by the existing flat `slot_capital`-based quantity as a
  safety ceiling.

All three are wired into `paper_trade.py`'s `simulate()` and
`live_session.py`, and are no-ops for any caller that doesn't pass
`atr`/`timestamp`.

## Alternatives Considered

### Port BALU's exact backtest results as evidence this will work here
- Rejected outright: BALU's ratios were tuned alongside BALU's own
  multi-factor rule-based scoring engine (technical + order-flow), not this
  project's GBM classifier (~52% holdout accuracy). A mechanism tuned for
  one signal's confidence distribution doesn't necessarily transfer to
  another untested signal.

### Wait to adopt any of this until the model's own edge is proven
- Pros: avoids compounding two unproven changes (weak model + new risk
  mechanism) at once
- Rejected (at the time): the structural gap was large enough (no per-trade
  exit at all) that it seemed worth testing early. In hindsight, per the
  validation below, this reasoning underweighted how much harder it is to
  attribute an outcome once two unproven variables move together.

## Post-implementation validation
`scripts/tune_strategy.py` swept ~15 `PortfolioManager` configurations
(BALU's exact ratios, wider ATR stops, session-only, sizing-only, and the
pre-BALU baseline with the mechanism disabled) against the same precomputed
features. **Every configuration -- including the exact pre-BALU baseline --
came back net-negative** on the 14-day holdout window that sweep ran
against. A same-window before/after comparison also showed net PnL negative
across all 4 walk-forward windows tested (7/14/21/28-day), vs. positive in 3
of 4 for the pre-BALU baseline in an earlier (different-window) batch.

The working theory (see `README.md` "Model-signal sweep") is that this
mechanism was tuned for BALU's own signal, not this project's ~52%-accuracy
GBM classifier, and porting the specific ratios without re-tuning against
this model's actual behavior doesn't transfer -- compounded by the extra
trade frequency (roughly doubled) adding transaction-cost drag against an
already-thin edge.

## Consequences
- **Do not treat this ADR's "Accepted" status as evidence the mechanism
  helps net PnL.** Treat it as an implemented, testable hypothesis that the
  available evidence currently contradicts.
- The pre-BALU configuration (flat sizing, no per-trade stop, full-day
  trading) remains the safer fallback baseline until either these parameters
  are re-tuned specifically against this model, or more evidence changes
  that assessment.
- `balu_default` remains the default in `.env`/`.env.example` only because it
  was the least-bad performer in one sweep -- not because it's validated.
- Any future re-tuning attempt should re-run `scripts/tune_strategy.py`
  across several different windows (not one) before drawing conclusions --
  a single sweep is one data point, and the swing between "strongly
  positive" and "uniformly negative" observed here happened just from the
  trailing-window shifting by hours of wall-clock time between runs.
