"""Compare the existing absolute-direction model against a cross-sectional
ranking reformulation, on the same held-out window, fetching data only once.

Why this exists: scripts/tune_model.py already showed that neither a cleaner
label horizon nor a more regularized GBM moves holdout accuracy off ~52% --
every variant of "will this stock's price go up, asked in isolation" landed
in the same place. The one thing not yet tried: reframing the *training
objective* to match what scripts/paper_trade.py's simulate() already does at
*execution* time -- score every candidate at a timestamp, then fill slots
strongest-confidence-first across the whole watchlist. That's a
cross-sectional ranking, but the model has only ever been trained to answer
"will this go up in isolation", never "will this outperform its peers right
now". See src/cross_sectional.py and docs/decisions/ for the full rationale.

Both variants are evaluated with the ATR-stop mechanism OFF and full-day
session/sizing (same as tune_model.py's methodology), to isolate model-signal
differences from risk-parameter effects.

Usage:
    python scripts/tune_model_cross_sectional.py
    python scripts/tune_model_cross_sectional.py --days 30 --holdout-days 10
    python scripts/tune_model_cross_sectional.py RELIANCE.NS TCS.NS ...   # explicit tickers, faster iteration
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cross_sectional import build_cross_sectional_training_set  # noqa: E402
from src.features import FEATURE_COLUMNS  # noqa: E402
from src.market_data import fetch_bars_by_ticker, fetch_benchmark_close  # noqa: E402
from src.model import MomentumClassifier  # noqa: E402
from src.portfolio import PortfolioManager  # noqa: E402
from src.screener import load_universe  # noqa: E402
from src.universe import load_sector_map  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_trade import simulate  # noqa: E402
from tune_model import build_variants as build_absolute_baseline_variant  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_absolute_baseline(ticker_bars: dict, benchmark_close, cutoff: pd.Timestamp) -> tuple[str, float, dict]:
    """The existing baseline_h1_default_gbm variant from tune_model.py,
    trained on this run's own pre-cutoff data (not a prior run's numbers --
    tune_strategy.py already showed results swing hugely just from the
    trailing window shifting between runs, so a fair comparison needs both
    variants trained on the exact same fetch)."""
    from src.features import add_features, add_labels

    variant = build_absolute_baseline_variant()[0]  # baseline_h1_default_gbm
    ticker_full_features = {
        ticker: add_features(bars, benchmark_close=benchmark_close) for ticker, bars in ticker_bars.items()
    }
    train_frames = []
    for ticker, full_features in ticker_full_features.items():
        pre_cutoff = full_features[full_features.index.normalize() < cutoff]
        if pre_cutoff.empty:
            continue
        labeled = add_labels(pre_cutoff, epsilon=variant["epsilon"], horizon=variant["horizon"])
        labeled = labeled.dropna(subset=FEATURE_COLUMNS + ["label"])
        if not labeled.empty:
            train_frames.append(labeled)

    from sklearn.ensemble import GradientBoostingClassifier
    classifier = MomentumClassifier(model=GradientBoostingClassifier(**variant["gbm_params"]))
    report = classifier.train_multi(train_frames)
    return "absolute_baseline", report["accuracy"], classifier


def run_cross_sectional_variant(ticker_bars: dict, benchmark_close, cutoff: pd.Timestamp,
                                 top_quantile: float, bottom_quantile: float) -> tuple[str, float, MomentumClassifier]:
    """Rank labels are computed over the FULL fetched range (each row's label
    only depends on other tickers' returns at that SAME timestamp, never a
    future one, so there's no leakage from including post-cutoff timestamps
    in this call) -- only the *training* rows are then filtered to pre-cutoff."""
    full_labeled = build_cross_sectional_training_set(
        ticker_bars, benchmark_close=benchmark_close, top_quantile=top_quantile, bottom_quantile=bottom_quantile,
    )
    train_set = full_labeled[full_labeled["timestamp"].dt.normalize() < cutoff]
    label = f"cross_sectional_q{int(top_quantile * 100)}"
    if train_set.empty:
        logger.warning("No training data for %s, skipping.", label)
        return label, float("nan"), None

    classifier = MomentumClassifier()
    report = classifier.train_prelabeled(train_set.reset_index(drop=True), test_size=0.2)
    return label, report["accuracy"], classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", help="Tickers to tune on. Defaults to the full universe.")
    parser.add_argument("--days", type=int, default=59, help="Total lookback window in days.")
    parser.add_argument("--holdout-days", type=int, default=14, help="Trailing days held out as the OOS test window.")
    args = parser.parse_args()

    tickers = args.tickers or load_universe()
    logger.info("Fetching %d days of 15m bars for %d tickers (ONCE, reused across every variant)...",
                args.days, len(tickers))
    ticker_bars = fetch_bars_by_ticker(tickers, args.days)
    benchmark_close = fetch_benchmark_close(args.days)

    last_date = max(bars.index.max() for bars in ticker_bars.values()).normalize()
    cutoff = last_date - pd.Timedelta(days=args.holdout_days)
    logger.info("Train window: < %s | Held-out test window: >= %s", cutoff.date(), cutoff.date())

    sector_map = load_sector_map()
    results = []

    variants = [
        ("absolute_baseline", lambda: run_absolute_baseline(ticker_bars, benchmark_close, cutoff)),
        ("cross_sectional_q30", lambda: run_cross_sectional_variant(ticker_bars, benchmark_close, cutoff, 0.3, 0.3)),
        ("cross_sectional_q20", lambda: run_cross_sectional_variant(ticker_bars, benchmark_close, cutoff, 0.2, 0.2)),
    ]

    for name, build_fn in variants:
        logger.info("--- Variant: %s ---", name)
        label, accuracy, classifier = build_fn()
        if classifier is None:
            continue
        logger.info("%s: holdout accuracy=%.3f", label, accuracy)

        portfolio = PortfolioManager(
            sector_map=sector_map, primary_session_end="15:30", continuation_session_end="15:30",
        )
        result = simulate(
            ticker_bars, classifier, start_date=cutoff, benchmark_close=benchmark_close,
            portfolio=portfolio, use_atr_stop=False, log_to_db=False,
        )
        results.append((label, accuracy, result))
        logger.info(
            "%s: trades=%d gross=%.2f (%.1f%% WR) net=%.2f (%.1f%% WR)",
            label, result["total_trades"], result["total_pnl"], result["win_rate"] * 100,
            result["total_net_pnl"], result["net_win_rate"] * 100,
        )

    results.sort(key=lambda r: r[2]["total_net_pnl"], reverse=True)
    logger.info("\n--- Ranked by net PnL (best first) ---")
    for label, accuracy, result in results:
        logger.info(
            "%-25s holdout_acc=%.3f  net=%9.2f (%.1f%% WR, %d trades)  gross=%9.2f (%.1f%% WR)",
            label, accuracy, result["total_net_pnl"], result["net_win_rate"] * 100, result["total_trades"],
            result["total_pnl"], result["win_rate"] * 100,
        )


if __name__ == "__main__":
    main()
