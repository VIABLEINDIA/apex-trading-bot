"""Compare the existing absolute-direction model against a cross-sectional
ranking reformulation, across several holdout windows, fetching data and
computing features only once.

Why this exists: scripts/tune_model.py already showed that neither a cleaner
label horizon nor a more regularized GBM moves holdout accuracy off ~52% --
every variant of "will this stock's price go up, asked in isolation" landed
in the same place. The one thing not yet tried: reframing the *training
objective* to match what scripts/paper_trade.py's simulate() already does at
*execution* time -- score every candidate at a timestamp, then fill slots
strongest-confidence-first across the whole watchlist. That's a
cross-sectional ranking, but the model has only ever been trained to answer
"will this go up in isolation", never "will this outperform its peers right
now". See src/cross_sectional.py and docs/decisions/0005 for the full
rationale and the single-window result that motivated this multi-window run:
one 14-day window showed cross-sectional beating the absolute baseline, but
this project already watched an analogous single-window result (BALU, see
docs/decisions/0003) flip entirely under a proper multi-window sweep. This
script is that sweep, applied to the cross-sectional question.

Every variant is evaluated with the ATR-stop mechanism OFF and full-day
session/sizing (same as tune_model.py's methodology), to isolate model-signal
differences from risk-parameter effects.

Cost note: feature computation (add_features per ticker) and cross-sectional
rank labeling both only depend on the fetched history, never on which window
is being evaluated -- a stock's rank at time T depends only on other stocks'
returns at that same T, not on any train/test cutoff. So both are computed
ONCE regardless of how many --holdout-days values are given; only the GBM
training (which does depend on the cutoff, since it only sees pre-cutoff
rows) is repeated per (window, variant) pair.

Usage:
    python scripts/tune_model_cross_sectional.py                                # sweeps 7/14/21/28-day holdouts
    python scripts/tune_model_cross_sectional.py --days 59 --holdout-days 7 14
    python scripts/tune_model_cross_sectional.py RELIANCE.NS TCS.NS ... --holdout-days 7   # explicit tickers, faster iteration
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cross_sectional import build_cross_sectional_training_set  # noqa: E402
from src.features import FEATURE_COLUMNS, add_features, add_labels  # noqa: E402
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


def train_baseline_for_cutoff(ticker_full_features: dict, epsilon: float, horizon: int,
                               gbm_params: dict, cutoff: pd.Timestamp) -> tuple[float, MomentumClassifier | None]:
    """Re-labels (cheap) and trains (not cheap) on whatever's before `cutoff`
    in the already-featurized (expensive, precomputed once) frames."""
    train_frames = []
    for full_features in ticker_full_features.values():
        pre_cutoff = full_features[full_features.index.normalize() < cutoff]
        if pre_cutoff.empty:
            continue
        labeled = add_labels(pre_cutoff, epsilon=epsilon, horizon=horizon).dropna(subset=FEATURE_COLUMNS + ["label"])
        if not labeled.empty:
            train_frames.append(labeled)

    if not train_frames:
        return float("nan"), None

    classifier = MomentumClassifier(model=GradientBoostingClassifier(**gbm_params))
    report = classifier.train_multi(train_frames)
    return report["accuracy"], classifier


def train_cross_sectional_for_cutoff(full_labeled: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[float, MomentumClassifier | None]:
    """Filters the already-rank-labeled (expensive, precomputed once) frame
    to pre-cutoff rows and trains on just that slice."""
    train_set = full_labeled[full_labeled["timestamp"].dt.normalize() < cutoff]
    if train_set.empty:
        return float("nan"), None

    classifier = MomentumClassifier()
    report = classifier.train_prelabeled(train_set.reset_index(drop=True), test_size=0.2)
    return report["accuracy"], classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", help="Tickers to tune on. Defaults to the full universe.")
    parser.add_argument("--days", type=int, default=59, help="Total lookback window in days.")
    parser.add_argument("--holdout-days", type=int, nargs="+", default=[7, 14, 21, 28],
                        help="One or more trailing-day holdout windows to sweep (same set as the README tables).")
    args = parser.parse_args()

    tickers = args.tickers or load_universe()
    logger.info("Fetching %d days of 15m bars for %d tickers (ONCE, reused across every window and variant)...",
                args.days, len(tickers))
    ticker_bars = fetch_bars_by_ticker(tickers, args.days)
    benchmark_close = fetch_benchmark_close(args.days)
    last_date = max(bars.index.max() for bars in ticker_bars.values()).normalize()

    logger.info("Precomputing features once per ticker (reused across every window and variant)...")
    ticker_full_features = {
        ticker: add_features(bars, benchmark_close=benchmark_close) for ticker, bars in ticker_bars.items()
    }

    logger.info("Precomputing cross-sectional rank labels once per quantile setting "
                "(ranks only depend on same-timestamp returns, never on a cutoff)...")
    cross_sectional_frames = {
        "cross_sectional_q30": build_cross_sectional_training_set(
            ticker_bars, benchmark_close=benchmark_close, top_quantile=0.3, bottom_quantile=0.3,
        ),
        "cross_sectional_q20": build_cross_sectional_training_set(
            ticker_bars, benchmark_close=benchmark_close, top_quantile=0.2, bottom_quantile=0.2,
        ),
    }

    baseline_variant = build_absolute_baseline_variant()[0]  # baseline_h1_default_gbm
    sector_map = load_sector_map()
    all_results = []  # (holdout_days, variant_label, accuracy, result)

    for holdout_days in args.holdout_days:
        cutoff = last_date - pd.Timedelta(days=holdout_days)
        logger.info("=== Holdout window: %d days (train < %s, test >= %s) ===",
                    holdout_days, cutoff.date(), cutoff.date())

        variants = [
            ("absolute_baseline", lambda: train_baseline_for_cutoff(
                ticker_full_features, baseline_variant["epsilon"], baseline_variant["horizon"],
                baseline_variant["gbm_params"], cutoff,
            )),
            ("cross_sectional_q30", lambda: train_cross_sectional_for_cutoff(
                cross_sectional_frames["cross_sectional_q30"], cutoff,
            )),
            ("cross_sectional_q20", lambda: train_cross_sectional_for_cutoff(
                cross_sectional_frames["cross_sectional_q20"], cutoff,
            )),
        ]

        for label, train_fn in variants:
            accuracy, classifier = train_fn()
            if classifier is None:
                logger.warning("No training data for %s at %d-day holdout, skipping.", label, holdout_days)
                continue
            logger.info("[%dd] %s: holdout accuracy=%.3f", holdout_days, label, accuracy)

            portfolio = PortfolioManager(
                sector_map=sector_map, primary_session_end="15:30", continuation_session_end="15:30",
            )
            result = simulate(
                ticker_bars, classifier, start_date=cutoff, benchmark_close=benchmark_close,
                portfolio=portfolio, use_atr_stop=False, log_to_db=False,
            )
            all_results.append((holdout_days, label, accuracy, result))
            logger.info(
                "[%dd] %s: trades=%d gross=%.2f (%.1f%% WR) net=%.2f (%.1f%% WR)",
                holdout_days, label, result["total_trades"], result["total_pnl"], result["win_rate"] * 100,
                result["total_net_pnl"], result["net_win_rate"] * 100,
            )

    logger.info("\n--- Multi-window sweep results (grouped by holdout window) ---")
    for holdout_days in args.holdout_days:
        logger.info("Holdout: %d days", holdout_days)
        window_results = [r for r in all_results if r[0] == holdout_days]
        window_results.sort(key=lambda r: r[3]["total_net_pnl"], reverse=True)
        for _, label, accuracy, result in window_results:
            logger.info(
                "  %-25s holdout_acc=%.3f  net=%9.2f (%.1f%% WR, %d trades)  gross=%9.2f (%.1f%% WR)",
                label, accuracy, result["total_net_pnl"], result["net_win_rate"] * 100, result["total_trades"],
                result["total_pnl"], result["win_rate"] * 100,
            )

    logger.info("\n--- Wins by variant (how often each variant was least-bad across all windows tested) ---")
    win_counts: dict[str, int] = {}
    for holdout_days in args.holdout_days:
        window_results = [r for r in all_results if r[0] == holdout_days]
        if not window_results:
            continue
        best_label = max(window_results, key=lambda r: r[3]["total_net_pnl"])[1]
        win_counts[best_label] = win_counts.get(best_label, 0) + 1
    for label, count in sorted(win_counts.items(), key=lambda kv: kv[1], reverse=True):
        logger.info("  %-25s best in %d/%d windows", label, count, len(args.holdout_days))


if __name__ == "__main__":
    main()
