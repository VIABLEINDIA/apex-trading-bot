# Apex AI Trading Bot

Institutional-style, fully automated intraday momentum bot for NSE equities (Nifty 500), built from the "Apex AI Trading Bot" architecture blueprint. Screens the universe pre-market, scores momentum with a GradientBoostingClassifier, enforces a 10-slot portfolio risk cap, and (optionally) routes live orders through Kotak Neo.

## Safety default: paper trading

**The bot places zero real orders unless `LIVE_TRADING=true` is set in `.env`.** Every order is logged as a `PAPER-...` simulated fill until you explicitly flip that switch. Do not set it to `true` until you've run genuine out-of-sample validation (see below) and reviewed several sessions in `data/trading_journal.db` — see "Model performance" for why this matters here specifically.

## Two venvs -- this is required, not optional

`neo_api_client` (the Kotak Neo SDK) hard-pins `websockets==8.1` and `certifi==2022.12.7`. `yfinance` requires `websockets>=13`. These cannot be satisfied in the same Python environment, so the project is split into two processes that never share a venv:

| | venv (main) | venv-live |
|---|---|---|
| Installs from | `requirements.txt` | `requirements-live.txt` |
| Has | yfinance, scikit-learn, pandas | neo_api_client, scikit-learn, pandas |
| Runs | screener, training, backtests | the live execution loop only |
| Entry point | `src/premarket.py`, `scripts/*.py` | `src/live_session.py` |

`src/premarket.py` runs the screener and writes `data/watchlist.json`; `src/live_session.py` reads that file, connects to Kotak Neo, and trades. They hand off through that one file plus the shared SQLite journal (`data/trading_journal.db`) -- both of which are plain stdlib/pandas, no cross-venv dependency conflicts.

## Project layout

```
src/
  config.py        - env-driven settings (capital, risk, Kotak creds)
  screener.py       - Phase 1: Nifty 500 momentum screener (yfinance, 30-day ROC)
  market_data.py    - shared batched-download-with-retry helper (yfinance)
  features.py       - Phase 2: alpha features (EMA/RSI/ROC/MACD/Bollinger/ATR)
  model.py          - Phase 2: GradientBoostingClassifier train/predict/save/load
  portfolio.py       - Phase 3: risk manager -- slots, dedup, sector limit, hourly budget, circuit breaker, capital shield
  execution.py       - Phase 4: Kotak Neo auth/streaming/order routing (paper-safe)
  costs.py            - time-of-day slippage + STT/exchange-charge modeling for realistic net PnL
  universe.py          - ticker universe + sector map from data/nifty500.csv (no yfinance dep, safe for venv-live)
  database.py        - Phase 5: SQLite journal (trades, ai_signals)
  premarket.py        - MAIN venv entry point: run screener, write watchlist.json
  live_session.py     - LIVE venv entry point: connect, stream, trade, flush on shutdown
scripts/
  train_model.py           - fetch 15m bars, train, save the classifier
  paper_trade.py            - historical paper-trading simulation (no broker needed)
  validate_walkforward.py   - genuine out-of-sample validation (train/test date split)
  analyze_stoploss.py        - stop-loss sensitivity analysis against logged trades
  test_broker_connection.py  - Kotak Neo login/2FA/websocket smoke test, places NO orders
  deploy/ecosystem.config.js - PM2 config for live_session.py (venv-live)
  deploy/crontab.txt          - AWS EC2 cron: premarket -> pm2 start, then pm2 stop
tests/          - unit tests for portfolio and feature logic (no network/broker needed)
```

## Setup

1. Main venv (screener/training/backtesting):
   ```
   python -m venv venv && source venv/bin/activate   # venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Live venv (only needed once you're ready to connect to Kotak Neo):
   ```
   python -m venv venv-live
   venv-live/bin/pip install -r requirements-live.txt   # pulls neo_api_client from GitHub
   ```
3. `cp .env.example .env` and fill in your Kotak Neo credentials (consumer key/secret, mobile number, UCC, TOTP secret, MPIN from the developer console/app) directly in the file -- not somewhere these could be logged. Leave `LIVE_TRADING=false`.
4. Download the official Nifty 500 constituent list as `data/nifty500.csv` with a `Symbol` column (e.g. `curl -A "Mozilla/5.0" -o data/nifty500.csv https://archives.nseindia.com/content/indices/ind_nifty500list.csv`). Without this file, the screener falls back to a small 20-stock sample so the pipeline is still runnable for testing.
5. Train an initial model (main venv): `python scripts/train_model.py`. Pulls ~59 days of 15-minute bars per ticker and saves to `models/momentum_classifier.joblib`.
6. **Validate out-of-sample before trusting the model at all** -- see "Model performance" below.
7. Each trading day: `venv/bin/python -m src.premarket` (writes `data/watchlist.json`), then `venv-live/bin/python -m src.live_session` (connects and trades, paper mode by default).

## Running tests

```
pytest tests/ -v
```

Covers the portfolio risk manager (slot limits, dedup, capital math) and the feature pipeline (warm-up handling, label correctness) with synthetic data — no live API calls.

## Model performance -- read this before going live

An earlier naive workflow (train on "last 59 days", backtest on "last 30 days" via two separate yfinance pulls) looked profitable, but the two windows overlap almost completely -- yfinance always serves the most recent N days, so that backtest was measuring in-sample fit, not real edge.

`scripts/validate_walkforward.py` fixes this: it fetches history once, trains only on bars before a cutoff date, and only lets the simulator trade on bars strictly after it. `src/costs.py` additionally models realistic NSE intraday charges (STT, exchange fees, stamp duty, GST -- Kotak Neo's own MIS brokerage is zero) plus a 5bps/leg slippage assumption, since those aren't negligible against a ~0.15-0.2% average per-trade edge. Real out-of-sample runs so far (screener universe, all windows ending 2026-07-03), gross vs. net:

| Holdout | Trades | Win rate (gross / net) | PnL (gross / net) |
|---|---|---|---|
| 7-day | 50 | 46.0% / 44.0% | +₹896.76 / +₹265.44 |
| 14-day | 100 | 42.0% / 38.0% | +₹1,735.84 / +₹474.47 |
| 21-day | 150 | 48.7% / 44.0% | +₹3,600.71 / +₹1,702.96 |
| 28-day\* | 200 | 48.0% / 44.5% | +₹564.03 / **-₹2,014.85** |
| 10 liquid tickers, ~15-day holdout | 88 | 40.9% / n/a | -₹857.80 / n/a |

\* The 28-day run only got usable data for 200/500 tickers -- Yahoo Finance rate-limited the request and mislabeled ~50 tickers (including obviously-still-listed names like MARUTI, ONGC, NESTLEIND) as "possibly delisted". Treat this row as based on a smaller, likely non-representative slice of the universe rather than a clean fourth data point.

Net PnL stays positive in three of four windows but consistently shrinks 50-70% from gross, and **win rate is sub-50% in every single window** (gross or net). The three full-size windows are also nested/overlapping (all end "today"), not independent trials -- this rules out one specific overfitting failure mode (fitting to one arbitrary cutoff) without being three separate confirmations of edge. All of it is drawn from the same ~2-month market regime -- yfinance's 15-min history only goes back ~60 days, so a genuinely different regime (sharp downtrend, high-vol chop) hasn't been tested yet. **Do not set `LIVE_TRADING=true` on the strength of these numbers.** Before considering it:

- Keep running `validate_walkforward.py` as more calendar time (and therefore more independent holdout windows) becomes available.
- Re-run the 28-day window once Yahoo Finance rate-limiting clears, for a clean comparison.
- Treat a positive result on overlapping windows as suggestive, not confirmed.

```
python scripts/validate_walkforward.py --days 59 --holdout-days 14   # diagnostic only, doesn't touch the saved model
python scripts/validate_walkforward.py --days 59 --holdout-days 14 --save-model   # promote if you're satisfied
```

## Risk management: why there's a circuit breaker instead of a per-trade stop-loss

See `docs/decisions/` for ADRs recording the reasoning, alternatives considered,
and consequences behind each risk-management mechanism below in a more
structured form than this prose.

`scripts/analyze_stoploss.py` re-fetches the real intraday bar path for every trade logged across the walk-forward runs above (500 trades) and checks what a fixed-% stop-loss would have done:

| Stop threshold | Trades stopped early | Total PnL | Win rate |
|---|---|---|---|
| None (baseline) | -- | +₹9,206.98 | 46.0% |
| 0.5% | 301/500 | +₹7,016.87 | 33.4% |
| 1.0% | 196/500 | +₹6,653.46 | 42.0% |
| 1.5% | 107/500 | +₹7,666.63 | 44.8% |
| 2.0% | 54/500 | +₹8,716.41 | 46.0% |
| 3.0% | 7/500 | +₹9,060.36 | 46.0% |

**Every fixed-% stop-loss made results worse**, not better -- most intraday dips are noise around a momentum trade (median intraday drawdown before close was 0.73%) that recovers by end of day; a mechanical stop just cuts winners off before that recovery. It's also not a fix for the actual tail risk: several traded names are circuit-band-prone on NSE, and a stop order doesn't fill if a stock locks limit-down anyway.

Given that, `PortfolioManager` implements a **portfolio-level circuit breaker** instead of a per-trade stop-loss:

- Tracks `realized_pnl_today` (updated in `close_trade`) and can mark open positions to market via `mark_to_market(current_prices)`.
- `check_circuit_breaker(current_prices)` trips once (`circuit_breaker_tripped = True`) the first time realized + unrealized PnL breaches `-DAILY_LOSS_LIMIT_PCT * TOTAL_CAPITAL` (default 2%), and returns `True` only on the call that newly trips it, so callers force-flatten exactly once instead of re-triggering every tick.
- Once tripped, `evaluate_signal` rejects all new entries for the rest of the day ("circuit breaker tripped"), but doesn't touch the no-stop-loss exit logic for positions that were already fine.
- `reset_day()` clears the flag and counter at the start of each new trading day (called on day-rollover in `paper_trade.py`'s simulator; a fresh `live_session.py` process each morning starts clean automatically).

This protects against the untested tail (a bad day compounding across many positions) without cutting off the per-trade noise that the stop-loss analysis showed actively hurts this strategy. Tune the threshold with `DAILY_LOSS_LIMIT_PCT` in `.env`.

### Additional risk layers (adapted from a more mature reference implementation)

A separate, more mature Kotak-Neo-targeting project (not part of this repo) has a 5-layer risk architecture; three of its concepts were adapted here -- rescaled to fit our numbers, not copied verbatim, since its original thresholds were calibrated for a >10% drawdown ceiling that would never fire alongside our 2% circuit breaker:

- **Sector concentration limit** (`MAX_POSITIONS_PER_SECTOR`, default 3): caps simultaneous open positions in the same NSE Industry (from `data/nifty500.csv`'s `Industry` column, loaded via `src/universe.py:load_sector_map`), so the 10 slots can't silently become 6 correlated bank/finance names. No-ops if no sector map is supplied.
- **Hourly trade budget** (`MAX_TRADES_PER_HOUR`, default 4): caps new entries per clock hour, to avoid piling into more trades right after a string of losses in a bad hour. Bucketed on an explicit `timestamp` argument to `evaluate_signal`/`open_trade` (never wall-clock "now" internally) -- during a backtest, "now" is meaningless; what matters is the hour the *simulated* bar belongs to.
- **Capital shield** (graduated position sizing): `capital_shield_multiplier()` ramps position size down in steps (100% / 75% / 50% / 25%) as today's drawdown grows from 0% to 2%, *before* the circuit breaker's hard stop at 2% -- so a bad day gets progressively more conservative rather than trading at full size right up until the moment everything halts.

All three are backward-compatible no-ops if their inputs aren't supplied (no `sector_map`, no `timestamp`, drawdown at/above zero) -- existing tests and call sites that don't care about them are unaffected.

### ATR stop / trailing profit-lock, session restriction, and risk-based sizing (adapted from BALU)

A specific tuned strategy config ("BALU", exported from the same reference project referenced above) showed a much stronger win/loss asymmetry than anything measured here (avg win ~₹185 vs. avg loss ~₹-71, profit factor 3.23, across a genuinely non-overlapping 4-walk walk-forward). Its config implies three structural differences from what this project had, not just more risk gates -- adapted (not copied) into `PortfolioManager`:

- **ATR-based stop + trailing profit-lock** (`ATR_STOP_MULT=0.75`, `ATR_TRAIL_ACTIVATION_MULT=0.25`, `ATR_TRAIL_DISTANCE_MULT=0.12`): `open_trade(..., atr=...)` sets an initial stop at `entry - 0.75*ATR`. `check_exits(current_prices)` -- called every bar/tick for every open position -- activates a trailing stop once price has moved `0.25*ATR` in our favor, then ratchets it up (never down) to `price - 0.12*ATR`, closing the position the moment price falls through whichever stop is active. **This is fundamentally different from the fixed-% stop-loss already tested and rejected above**: it's volatility-adjusted per stock (a calm stock gets a tight stop, a choppy one gets room to breathe) and only tightens in the trader's favor, rather than imposing one blanket threshold that cuts every position's noise the same way. Positions opened without an `atr` argument get no stop at all (`stop_price` stays `None`) -- fully backward compatible.
- **Trading-window session gating** (`session_state`): before `PRIMARY_SESSION_END` (10:15), normal confidence threshold and full size. Before `CONTINUATION_SESSION_END` (13:15), the threshold is raised by `CONTINUATION_CONFIDENCE_BONUS` (+0.05) and size cut to `CONTINUATION_SIZE_MULT` (0.4x). After that, no new entries at all -- existing positions still exit normally via their stop/trail or the end-of-day flatten. This avoids the close entirely rather than just pricing in its cost, which our own time-of-day slippage table already flagged as the most expensive window to trade.
- **Risk-based position sizing**: when `atr` is supplied, `evaluate_signal` sizes the trade so a stop-out loses `RISK_PER_TRADE_PCT` (1.3%) of total capital -- `quantity = floor(total_capital * risk_per_trade_pct / (atr_stop_mult * atr))` -- capped by the existing flat `slot_capital`-based quantity as a safety ceiling (a very tight stop can never blow past the normal per-slot allocation). Without `atr`, sizing falls back to the original flat calculation unchanged.

All three are wired into `paper_trade.py`'s `simulate()` and `live_session.py` (ATR is derived from the existing `atr_pct` feature: `atr = atr_pct * price`), and are no-ops for any caller that doesn't pass `atr`/`timestamp`.

**Same-window before/after result** (2026-06-22 to 2026-07-06, full universe, 14-day holdout -- both runs use the identical window, so this isolates the effect of the code change rather than a market-regime shift):

| | Before (no ATR stop/session/sizing) | After (BALU changes active) |
|---|---|---|
| Trades | 100 | 158 |
| Gross PnL | -₹2,402.04 | -₹995.14 |
| Gross win rate | 42.0% | 40.5% |
| Net PnL | -₹4,661.50 | -₹2,684.70 |
| Net win rate | 34.0% | 30.4% |

Losses shrank substantially in that one same-window comparison (~58% smaller gross, ~42% smaller net) -- but a fuller 7/14/21/28-day batch under the new strategy tells a different, more sobering story:

| Holdout | Trades | Gross PnL | Gross WR | Net PnL | Net WR |
|---|---|---|---|---|---|
| 7-day | 87 | -₹22.30 | 48.3% | -₹1,006.92 | 35.6% |
| 14-day | 158 | -₹995.14 | 40.5% | -₹2,684.70 | 30.4% |
| 21-day | 217 | -₹1,015.83 | 47.9% | -₹3,451.92 | 34.1% |
| 28-day | 321 | +₹365.82 | 46.1% | -₹3,044.28 | 34.6% |

Compared to the *original* 4-window batch, from before any of today's BALU/JEANS changes:

| Holdout | Trades | Gross PnL | Gross WR | Net PnL | Net WR |
|---|---|---|---|---|---|
| 7-day | 50 | +₹896.76 | 46.0% | +₹265.44 | 44.0% |
| 14-day | 100 | +₹1,735.84 | 42.0% | +₹474.47 | 38.0% |
| 21-day | 150 | +₹3,600.71 | 48.7% | +₹1,702.96 | 44.0% |
| 28-day | 200 | +₹564.03 | 48.0% | -₹2,014.85 | 44.5% (degraded data, see note above) |

**Net PnL is negative in all 4 new-strategy windows, vs. positive in 3 of 4 original windows.** Net win rate is consistently lower (30-36% vs 38-44%). Trade count roughly doubled (87-321 vs 50-200) -- more trades at the same per-trade friction means more cumulative cost drag, which shows up as a wider gross-to-net gap across the board.

**Caveat:** the two batches aren't on identical windows (hours passed between them, so "last N days" pulled different underlying calendar data each time) -- not a rigorously controlled comparison. But the *consistency* of the decline across all 4 windows, not just one, makes pure noise a weaker explanation than for earlier single-window checks.

**Working theory for why this underperforms despite BALU's own strong backtest:** BALU's ATR-stop/session/sizing mechanism was tuned alongside BALU's own multi-factor rule-based scoring engine (technical + order-flow), not bolted onto a different, weaker signal (our GBM classifier, ~52% holdout accuracy). A stop/sizing mechanism tuned for one signal's confidence distribution doesn't necessarily transfer to another -- porting the specific ratios (0.75/0.25/0.12 ATR multiples, 1.3% risk-per-trade) without re-tuning them against *our* model's actual behavior may just not work, and the extra trade frequency compounds transaction costs against a weak edge.

**Bottom line: treat today's ATR-stop/session/sizing changes as an unproven regression, not a validated improvement**, until either the parameters get re-tuned specifically against this model, or more evidence changes that read. The pre-BALU-strategy configuration (flat sizing, no per-trade stop, full-day trading) was net-positive in 3 of 4 original windows and is the safer baseline to fall back to.

### Parameter sweep result -- the real finding

`scripts/tune_strategy.py` fetches data and trains the model **once**, then replays the same precomputed features through ~15 different `PortfolioManager` configurations (BALU's exact ratios, wider ATR stops, session-restriction-only, sizing-only, and the pre-BALU baseline with the mechanism fully disabled) -- turning what had been a 25-30 minute-per-configuration cost into a single run. This is what a proper tuning pass looks like, instead of guessing at one configuration at a time.

**Every single one of the ~15 configurations tested -- including the exact pre-BALU baseline -- came back net-negative** on the 14-day holdout window this ran against (full universe, same day the sweep ran):

| Configuration | Trades | Net PnL | Net WR |
|---|---|---|---|
| `balu_default` (best of the batch) | 155 | -₹2,301.88 | 34.8% |
| `atr_wide_stop1.0_balu_session` | 155 | -₹2,982.54 | 33.5% |
| `atr_wide_stop2.0_balu_session` | 158 | -₹3,047.93 | 36.7% |
| `atr_wide_stop1.5_balu_session` | 155 | -₹3,242.92 | 35.5% |
| `session_only_no_atr` | 100 | -₹3,579.32 | 34.0% |
| `baseline_off` (pre-BALU design) | 100 | -₹3,586.56 | 33.0% |
| ...(9 more, all worse)... | | down to -₹6,337.37 | |

This is the most important result of the whole exercise, and it cuts against every prior conclusion in this section: **the pre-BALU baseline itself is net-negative on this window**, even though the *same* baseline configuration was net-positive in 3 of 4 windows from the original batch earlier in the session. The swing between "strongly positive" and "uniformly negative" happened just from the underlying trailing-14-days window shifting by several hours of wall-clock time between runs -- a bigger effect than any risk-parameter choice tested here.

**What this means:** no amount of tuning within this parameter space (stop width, trailing distance, risk-per-trade, session boundaries) is going to produce a reliably profitable configuration, because the swings between windows are dominated by something else -- most likely the underlying model's signal genuinely doesn't have a stable, time-consistent edge (consistent with the ~52% holdout accuracy and thin/inconsistent edge documented throughout this file). `balu_default` topping this particular sweep is not an endorsement of it; it's the least-bad result in a batch that lost money across the board on this specific window. Picking it and calling it "tuned" would be fitting noise, not signal.

**Honest recommendation:** stop tuning risk/sizing parameters and treat the model's predictive signal itself as the actual bottleneck. Re-run `tune_strategy.py` across several different windows (not just one) to see whether any configuration is *consistently* least-bad, which would be a weak but real signal worth acting on -- a single sweep like this one is still just one data point. In the meantime, `balu_default` remains the default in `.env`/`.env.example` since it was the best performer here, but that should not be read as "validated" or "tuned to be profitable."

### Model-signal sweep -- the actual bottleneck confirmed

Acting on the recommendation above, `scripts/tune_model.py` applies the same "fetch/compute once, vary cheaply" principle to the model itself: features are computed once per ticker (they don't depend on the label horizon), and only `add_labels()` -- a cheap relabeling of an already-featurized frame -- differs per variant. Each variant still needs its own GBM training run (10-25 min on the full universe, unlike the risk-parameter sweep), so only 4 well-chosen candidates were tested: the current 1-bar-ahead label with default hyperparameters, longer 3-bar and 5-bar horizons (less noisy targets), and a more regularized GBM (fewer/shallower trees, lower learning rate) at the original horizon. All four were evaluated with the ATR-stop mechanism off (isolating model quality from the risk-parameter question already answered above).

| Variant | Holdout accuracy | Net PnL | Net WR | Gross PnL |
|---|---|---|---|---|
| `h1_regularized_gbm` (best) | 0.521 | -₹2,693.38 | 36.7% | -₹566.73 |
| `horizon3_default_gbm` | 0.523 | -₹3,189.18 | 31.6% | -₹1,033.57 |
| `horizon5_default_gbm` | 0.522 | -₹4,078.01 | 27.6% | -₹1,945.85 |
| `baseline_h1_default_gbm` | 0.520 | -₹4,650.24 | 33.7% | -₹2,534.87 |

**Every variant lands within 0.003 of ~52% holdout accuracy, and none is net-positive.** Neither a less noisy training target (3 or 5 bars ahead instead of 1) nor a more regularized model moves accuracy off that ceiling. This confirms the working theory from the risk-parameter sweep: the ~52% accuracy isn't an artifact of this specific horizon or hyperparameter choice -- it's a ceiling set by the current feature set (technical indicators computed from OHLCV alone: EMA/RSI/ROC/MACD/Bollinger/ATR/relative-strength-vs-Nifty).

**What this rules out, and what it doesn't:** it rules out "just retrain with a cleaner target" or "just regularize the model better" as fixes -- both were tried directly and neither helped. It does *not* rule out that a genuinely different feature set could still find signal (order-flow/microstructure data, which needs broker-level tick data we don't have; a cross-sectional ranking objective instead of per-stock absolute classification, which would be a bigger redesign since the model currently predicts each stock's direction independently rather than "which stocks look best relative to each other right now" -- arguably closer to what the strategy's cross-sectional ranking *execution* already assumes, but the *training objective* was never changed to match).

**A real bug found and fixed along the way:** `MomentumClassifier.__init__` used `model or GradientBoostingClassifier(...)` to pick a default when no model is supplied. This looks like a safe None-check but isn't: Python falls back to `__len__` for truthiness when `__bool__` isn't defined, and an *unfitted* `GradientBoostingClassifier` has no `estimators_` attribute yet, so `len(model)` raises `AttributeError` instead of evaluating as falsy. This was latent since day one but never triggered, because every previous caller passed either `None` or an already-*fitted* model from `load()` (which does have `estimators_`, so `len()` works and returns a truthy nonzero count). `scripts/tune_model.py` was the first caller to construct an unfitted custom model, exposing it. Fixed with an explicit `is not None` check.

### Consecutive-loss halt and manual kill switch (adapted from JEANS)

A third-generation rewrite in the same lineage (APEXBOT -> LAKSHMI -> "JEANS") adds two more cheap, non-conflicting safety checks, both in `evaluate_signal` before any sizing/session logic runs:

- **Consecutive-loss halt** (`MAX_CONSECUTIVE_LOSSES`, default 5): `close_trade` increments `consecutive_losses` on a loss, resets it to 0 on a win; once the counter reaches the limit, no new entries are approved for the rest of the day ("consecutive loss limit reached"). Independent of `DAILY_LOSS_LIMIT_PCT` -- catches a cold streak of many small losses even before cumulative PnL breaches the daily circuit breaker. Resets in `reset_day()` alongside the other daily counters.
- **Manual kill switch** (`KILL_SWITCH_PATH`, default `./data/KILL_SWITCH`): `is_kill_switch_engaged()` checks whether that file exists, fresh on every call -- no caching, no polling loop. Create the file to block all new entries immediately (existing positions still exit normally via their stop/trail or day-end flatten); delete it to resume. Useful as a manual override during a live session without touching `.env` or restarting the process.

JEANS also has a multi-day peak-equity drawdown guard (halt if equity drops >20% from its running high-water-mark, tracked *across* days, not reset daily) -- not adopted here, since it would need equity state to persist across `live_session.py`'s daily process restarts (a state file or DB table), a bigger lift than the two above. Worth revisiting if this system ever runs for long enough that multi-day drawdown becomes a real concern distinct from single-day losses.

## Historical paper-trading (no broker needed)

`scripts/paper_trade.py` replays the full screener -> features -> model -> portfolio pipeline against real historical bars, logging every simulated fill to the same journal the live system uses. It never imports `src/execution.py`, so it needs only the main venv:

```
python scripts/paper_trade.py                      # today's screener watchlist, 30 days
python scripts/paper_trade.py RELIANCE.NS TCS.NS --days 15
```

## Deployment (AWS EC2)

1. Provision an Ubuntu EC2 instance, attach a static Elastic IP, and whitelist that IP in the Kotak Neo developer console.
2. `git clone` the project, set up **both** venvs (see Setup above), install `pm2` (`npm i -g pm2`), and populate `.env`.
3. Edit `APP_DIR` in `scripts/deploy/crontab.txt` and `cwd` in `scripts/deploy/ecosystem.config.js` to the actual deploy path.
4. `crontab scripts/deploy/crontab.txt` — runs the screener (main venv) at 08:45 IST, then pm2-starts the live session (venv-live) only once that succeeds; stops it at 15:30 IST. Not paying for idle compute overnight.
5. PM2 keeps `live_session.py` alive and auto-restarts it if it crashes intraday (`max_restarts: 10`). `kill_timeout: 10000` gives the SIGTERM handler time to flush open positions before pm2 force-kills it.

## Walk-forward retraining (bi-weekly)

Per the blueprint's optimization step: pull `data/trading_journal.db` every two weeks and retrain on the most recent 14 days of live data so the model adapts to regime shifts (e.g., bullish surge -> high-volatility chop):

```
python scripts/train_model.py --days 14
```

The `ai_signals` table logs every signal the model produced — including rejected ones — so you can debug model drift even on days with no executed trades.

## Known limitations / things to validate before going live

- **Model edge is unproven** -- see "Model performance" above. This is the main blocker, not a code issue.
- **Order placement fields are verified, tick/websocket fields are not.** `execution.py`'s auth flow (`totp_login`/`totp_validate`), order placement (`transaction_type` "B"/"S", `trading_symbol` "RELIANCE-EQ" format, `nOrdNo` response) were cross-checked against a working reference implementation. The WebSocket tick field names (`ltp`, `tk`, `v`) and the Nifty 50 index instrument token (`"NIFTY 50"`) are still unverified guesses -- confirm both against a live session (`scripts/test_broker_connection.py`) before trusting them.
- We never poll the broker for confirmed fills or reconcile against actual order/position status (`ordSt`, `fldQty`, `avgPrc`, `rlzPL` fields exist on Kotak Neo's API but aren't used here) -- the system trusts its own in-memory state and the price it last saw, not a broker-confirmed fill price.
- The bundled 20-stock universe in `screener.py`/`universe.py` is a fallback for local testing only; production runs need the real `data/nifty500.csv`.
- Slippage is a time-of-day reference table (`src/costs.py`), not measured from our own order book -- treat net PnL as directional, not exact.
- Per-trade exits are now an ATR stop + trailing profit-lock (see "ATR stop / trailing profit-lock..." above), not the fixed-% stop that backtesting showed makes things worse -- but this new mechanism hasn't been validated yet either; treat it as an untested hypothesis until a fresh walk-forward run confirms it actually helps net PnL/win rate rather than just being a different, equally-unproven guess.
- The trading-window session restriction and risk-based ATR sizing are similarly new and unvalidated -- same caveat.
