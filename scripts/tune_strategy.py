"""Fast parameter sweep for the ATR-stop/session/risk-sizing mechanism (and
the confidence threshold), reusing the SAME fetched data and trained model
across every candidate configuration.

Why this exists: every previous tuning attempt ran a full
validate_walkforward.py end to end per configuration (~25-30 min each --
download + train + simulate), because that script re-fetches and re-trains
every time. But only the *simulation* step depends on the risk/sizing
parameters being tuned here -- the data and the trained model don't change
across candidates. Fetching and training ONCE, then replaying the same
precomputed features through dozens of PortfolioManager configurations, turns
a multi-hour sweep into a couple of minutes.

This is what actually answers the question the last few validation rounds
couldn't: whether BALU's ATR-stop/session/sizing ratios (or some other
combination, including "off") produce a genuine improvement for *this*
model, rather than assuming numbers tuned for a different signal engine
transfer unchanged.

Usage:
    python scripts/tune_strategy.py                       # full universe, defaults below
    python scripts/tune_strategy.py --days 59 --holdout-days 14
    python scripts/tune_strategy.py RELIANCE.NS TCS.NS ... # explicit tickers, faster iteration
"""
import argparse
import itertools
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.market_data import fetch_bars_by_ticker, fetch_benchmark_close  # noqa: E402
from src.model import MomentumClassifier  # noqa: E402
from src.portfolio import PortfolioManager  # noqa: E402
from src.regime import RealizedVolatilityGate  # noqa: E402
from src.screener import load_universe  # noqa: E402
from src.universe import load_sector_map  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_trade import simulate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Each combination is (label, kwargs-for-PortfolioManager, use_atr_stop).
# "off" candidates use use_atr_stop=False, so `atr` is never passed through
# regardless of what's computed -- exercises the pre-BALU-strategy baseline
# (flat sizing, no per-trade stop, full-day trading) as a real candidate,
# not just an assumption.
def build_candidates(benchmark_close: pd.Series | None = None) -> list[tuple[str, dict, bool]]:
    candidates = []

    # Baseline: mechanism off entirely, full-day trading (no session cutoff).
    candidates.append((
        "baseline_off",
        dict(primary_session_end="15:30", continuation_session_end="15:30"),
        False,
    ))

    # BALU's own ratios, for reference.
    candidates.append((
        "balu_default",
        dict(atr_stop_mult=0.75, atr_trail_activation_mult=0.25, atr_trail_distance_mult=0.12,
             risk_per_trade_pct=0.013, primary_session_end="10:15", continuation_session_end="13:15",
             continuation_confidence_bonus=0.05, continuation_size_mult=0.4),
        True,
    ))

    # Wider stops (less likely to be noise-triggered), full-day trading.
    for stop_mult in (1.0, 1.5, 2.0):
        for trail_activation, trail_distance in ((0.5, 0.25), (1.0, 0.5)):
            candidates.append((
                f"atr_wide_stop{stop_mult}_act{trail_activation}_trail{trail_distance}_fullday",
                dict(atr_stop_mult=stop_mult, atr_trail_activation_mult=trail_activation,
                     atr_trail_distance_mult=trail_distance, risk_per_trade_pct=0.013,
                     primary_session_end="15:30", continuation_session_end="15:30"),
                True,
            ))

    # Same wider stops, but WITH BALU's session restriction, to isolate
    # whether the stop or the session cutoff is driving the difference.
    for stop_mult in (1.0, 1.5, 2.0):
        candidates.append((
            f"atr_wide_stop{stop_mult}_balu_session",
            dict(atr_stop_mult=stop_mult, atr_trail_activation_mult=0.5, atr_trail_distance_mult=0.25,
                 risk_per_trade_pct=0.013, primary_session_end="10:15", continuation_session_end="13:15",
                 continuation_confidence_bonus=0.05, continuation_size_mult=0.4),
            True,
        ))

    # Session restriction alone, mechanism off -- isolates the session effect.
    candidates.append((
        "session_only_no_atr",
        dict(primary_session_end="10:15", continuation_session_end="13:15",
             continuation_confidence_bonus=0.05, continuation_size_mult=0.4),
        False,
    ))

    # Risk-based sizing alone (wide stop so it barely ever triggers, i.e. it's
    # mostly acting as a sizing rule, not an exit rule), full-day trading.
    for risk_pct in (0.01, 0.02, 0.03):
        candidates.append((
            f"sizing_only_risk{risk_pct}",
            dict(atr_stop_mult=3.0, atr_trail_activation_mult=1.0, atr_trail_distance_mult=0.5,
                 risk_per_trade_pct=risk_pct, primary_session_end="15:30", continuation_session_end="15:30"),
            True,
        ))

    # Market-regime gate (src/regime.py:RealizedVolatilityGate), applied on
    # top of the exact baseline_off config (mechanism off, full-day trading)
    # to isolate the gate's own effect -- same "change one thing at a time"
    # approach as session_only_no_atr/sizing_only above. Untested until now
    # (see docs/decisions/ and README: implemented but never backtested).
    if benchmark_close is not None:
        gate = RealizedVolatilityGate(benchmark_close)
        candidates.append((
            "baseline_off_regime_gated",
            dict(primary_session_end="15:30", continuation_session_end="15:30", regime_gate=gate),
            False,
        ))

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", help="Tickers to tune on. Defaults to the full universe.")
    parser.add_argument("--days", type=int, default=59, help="Total lookback window in days.")
    parser.add_argument("--holdout-days", type=int, default=14, help="Trailing days held out as the OOS test window.")
    args = parser.parse_args()

    tickers = args.tickers or load_universe()
    logger.info("Fetching %d days of 15m bars for %d tickers (ONCE, reused across every candidate)...",
                args.days, len(tickers))
    ticker_bars = fetch_bars_by_ticker(tickers, args.days)
    benchmark_close = fetch_benchmark_close(args.days)

    last_date = max(bars.index.max() for bars in ticker_bars.values()).normalize()
    cutoff = last_date - pd.Timedelta(days=args.holdout_days)
    logger.info("Train window: < %s | Held-out test window: >= %s", cutoff.date(), cutoff.date())

    train_bars = {ticker: bars[bars.index.normalize() < cutoff]
                  for ticker, bars in ticker_bars.items() if not bars[bars.index.normalize() < cutoff].empty}
    train_benchmark_close = benchmark_close[benchmark_close.index.normalize() < cutoff]

    model = MomentumClassifier()
    report = model.train_multi(list(train_bars.values()), benchmark_close=train_benchmark_close)
    logger.info("Trained ONCE on pre-cutoff data (%d tickers). Holdout accuracy=%.3f", len(train_bars), report["accuracy"])

    candidates = build_candidates(benchmark_close)
    logger.info("Sweeping %d configurations against the same held-out window...", len(candidates))

    sector_map = load_sector_map()
    results = []
    for label, pm_kwargs, use_atr_stop in candidates:
        portfolio = PortfolioManager(sector_map=sector_map, **pm_kwargs)
        result = simulate(
            ticker_bars, model, start_date=cutoff, benchmark_close=benchmark_close,
            portfolio=portfolio, use_atr_stop=use_atr_stop, log_to_db=False,
        )
        results.append((label, result))
        logger.info(
            "%-40s trades=%-4d gross=%9.2f (%.1f%%)  net=%9.2f (%.1f%%)",
            label, result["total_trades"], result["total_pnl"], result["win_rate"] * 100,
            result["total_net_pnl"], result["net_win_rate"] * 100,
        )

    results.sort(key=lambda r: r[1]["total_net_pnl"], reverse=True)
    logger.info("\n--- Ranked by net PnL (best first) ---")
    for label, result in results:
        logger.info(
            "%-40s net=%9.2f (%.1f%% WR, %d trades)  gross=%9.2f (%.1f%% WR)",
            label, result["total_net_pnl"], result["net_win_rate"] * 100, result["total_trades"],
            result["total_pnl"], result["win_rate"] * 100,
        )


if __name__ == "__main__":
    main()
