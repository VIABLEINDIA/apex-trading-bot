"""Compare a handful of model-signal variants (label horizon, hyperparameters)
against the same held-out window, fetching data only once.

Why this exists: the risk-parameter sweep (scripts/tune_strategy.py) showed
that no combination of stop width / sizing / session boundaries fixes a
losing result -- every configuration, including the original baseline, came
back net-negative on that window. That points at the model's signal itself
(~52% holdout accuracy) as the actual bottleneck, not risk management.

Unlike the risk-parameter sweep, each variant here needs its own GBM training
run (10-25 min on the full universe), since horizon/hyperparameters change
what the model learns, not just how positions are sized -- so this only
fetches the raw bars once and reuses precomputed *features* (not labels)
across variants, since add_features() doesn't depend on the label horizon at
all. Only add_labels() does, and re-labeling an already-featurized frame is
cheap (no indicator recomputation).

Every variant is evaluated with the ATR-stop mechanism OFF and default
session/sizing (the "baseline_off" config from the risk-parameter sweep) --
this isolates model-quality differences from risk-parameter effects, which
were already shown to not matter much on their own.

Usage:
    python scripts/tune_model.py
    python scripts/tune_model.py --days 59 --holdout-days 14
    python scripts/tune_model.py RELIANCE.NS TCS.NS ...   # explicit tickers, faster iteration
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import add_features, add_labels, add_labels_cost_aware, FEATURE_COLUMNS  # noqa: E402
from src.market_data import fetch_bars_by_ticker, fetch_benchmark_close  # noqa: E402
from src.model import MomentumClassifier  # noqa: E402
from src.portfolio import PortfolioManager  # noqa: E402
from src.screener import load_universe  # noqa: E402
from src.universe import load_sector_map  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_trade import simulate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_variants() -> list[dict]:
    """Each variant: label horizon/epsilon (what the model is trained to
    predict) plus GBM hyperparameters (how it learns it)."""
    default_gbm = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
    regularized_gbm = dict(n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.7, random_state=42)

    return [
        dict(label="baseline_h1_default_gbm", horizon=1, epsilon=0.0005, gbm_params=default_gbm),
        dict(label="horizon3_default_gbm", horizon=3, epsilon=0.0010, gbm_params=default_gbm),
        dict(label="horizon5_default_gbm", horizon=5, epsilon=0.0015, gbm_params=default_gbm),
        dict(label="h1_regularized_gbm", horizon=1, epsilon=0.0005, gbm_params=regularized_gbm),
        # Dead-zone threshold derived from src/costs.py's real round-trip cost
        # model (time-of-day slippage + statutory charges) instead of the
        # fixed, guessed epsilon above -- see src/features.py:add_labels_cost_aware.
        # `epsilon` is unused here (cost_aware ignores it) but kept in the
        # dict for a uniform log line across variants.
        dict(label="h1_cost_aware_default_gbm", horizon=1, epsilon=None, gbm_params=default_gbm, cost_aware=True),
    ]


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

    # Features don't depend on label horizon -- compute ONCE per ticker over
    # the full history (train + test), reused by every variant. Only
    # add_labels() differs per variant, and re-labeling an already-featurized
    # frame is cheap (no indicator recomputation).
    logger.info("Precomputing features once per ticker...")
    ticker_full_features = {
        ticker: add_features(bars, benchmark_close=benchmark_close)
        for ticker, bars in ticker_bars.items()
    }

    sector_map = load_sector_map()
    results = []

    for variant in build_variants():
        label, horizon, epsilon, gbm_params = variant["label"], variant["horizon"], variant["epsilon"], variant["gbm_params"]
        cost_aware = variant.get("cost_aware", False)
        logger.info("--- Variant: %s (horizon=%d, epsilon=%s, cost_aware=%s, gbm=%s) ---",
                    label, horizon, epsilon, cost_aware, gbm_params)

        train_frames = []
        for ticker, full_features in ticker_full_features.items():
            pre_cutoff = full_features[full_features.index.normalize() < cutoff]
            if pre_cutoff.empty:
                continue
            if cost_aware:
                labeled = add_labels_cost_aware(pre_cutoff, horizon=horizon).dropna(subset=FEATURE_COLUMNS + ["label"])
            else:
                labeled = add_labels(pre_cutoff, epsilon=epsilon, horizon=horizon).dropna(subset=FEATURE_COLUMNS + ["label"])
            if not labeled.empty:
                train_frames.append(labeled)

        if not train_frames:
            logger.warning("No training data for variant %s, skipping.", label)
            continue

        classifier = MomentumClassifier(model=GradientBoostingClassifier(**gbm_params))
        report = classifier.train_multi(train_frames)
        logger.info(
            "%s: trained on %d tickers, in-sample-holdout accuracy=%.3f, class-1 recall=%.3f",
            label, len(train_frames), report["accuracy"], report.get("1", report.get("1.0", {})).get("recall", float("nan")),
        )

        portfolio = PortfolioManager(
            sector_map=sector_map, primary_session_end="15:30", continuation_session_end="15:30",
        )
        result = simulate(
            ticker_bars, classifier, start_date=cutoff, benchmark_close=benchmark_close,
            portfolio=portfolio, use_atr_stop=False, log_to_db=False,
        )
        results.append((label, report["accuracy"], result))
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
