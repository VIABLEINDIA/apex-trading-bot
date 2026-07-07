"""Genuine out-of-sample validation: train on an earlier window, backtest only
on a strictly later, held-out window the model never saw.

This exists because a naive "train on last 59 days, backtest on last 30 days"
workflow (train_model.py + paper_trade.py run separately) has the two windows
overlap almost entirely -- yfinance always serves "the most recent N days", so
the backtest ends up measuring in-sample fit, not real edge.

Here we fetch the full history ONCE, split each ticker's series at a single
cutoff date, train only on bars before the cutoff, and only allow the
simulator to evaluate signals / open trades on bars on/after the cutoff (bars
before it are still fed through for indicator warm-up -- see
scripts/paper_trade.py's `start_date` parameter).

Usage:
    python scripts/validate_walkforward.py                  # screener watchlist, defaults below
    python scripts/validate_walkforward.py --holdout-days 14 --days 59
    python scripts/validate_walkforward.py RELIANCE.NS TCS.NS --holdout-days 10
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import database  # noqa: E402
from src.config import settings  # noqa: E402
from src.market_data import fetch_bars_by_ticker, fetch_benchmark_close  # noqa: E402
from src.model import MomentumClassifier  # noqa: E402
from src.screener import load_universe  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_trade import simulate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", help="Tickers to validate. Defaults to the full universe.")
    parser.add_argument("--days", type=int, default=59, help="Total lookback window in days.")
    parser.add_argument("--holdout-days", type=int, default=14, help="Trailing days held out as the OOS test window.")
    parser.add_argument("--save-model", action="store_true",
                         help="If set, overwrite the production model with the one trained here on pre-cutoff data. "
                              "Without this flag the run is purely diagnostic and models/momentum_classifier.joblib is untouched.")
    args = parser.parse_args()

    tickers = args.tickers or load_universe()
    logger.info("Fetching %d days of 15m bars for %d tickers...", args.days, len(tickers))
    ticker_bars = fetch_bars_by_ticker(tickers, args.days)
    benchmark_close = fetch_benchmark_close(args.days)

    last_date = max(bars.index.max() for bars in ticker_bars.values()).normalize()
    cutoff = last_date - pd.Timedelta(days=args.holdout_days)
    logger.info("Train window: < %s | Held-out test window: >= %s", cutoff.date(), cutoff.date())

    train_bars = {}
    for ticker, bars in ticker_bars.items():
        pre_cutoff = bars[bars.index.normalize() < cutoff]
        if not pre_cutoff.empty:
            train_bars[ticker] = pre_cutoff
    train_benchmark_close = benchmark_close[benchmark_close.index.normalize() < cutoff]

    if not train_bars:
        raise RuntimeError("No data left before the cutoff -- shorten --holdout-days or lengthen --days.")

    classifier = MomentumClassifier()
    report = classifier.train_multi(list(train_bars.values()), benchmark_close=train_benchmark_close)
    logger.info(
        "Trained on pre-cutoff data only (%d tickers). In-sample-of-training holdout accuracy=%.3f",
        len(train_bars), report["accuracy"],
    )

    if args.save_model:
        classifier.save()
        logger.info("Saved as the production model at %s", settings.model_path)
    else:
        logger.info("Not overwriting the production model (pass --save-model to promote this one).")

    database.init_db()
    result = simulate(ticker_bars, classifier, start_date=cutoff, benchmark_close=benchmark_close)

    logger.info(
        "OUT-OF-SAMPLE result (%s to %s, held out from training): %d trades, %d closed. "
        "Gross (frictionless): PnL=%.2f win_rate=%.1f%%. "
        "Net (after slippage/STT/charges): PnL=%.2f win_rate=%.1f%%.",
        cutoff.date(), last_date.date(), result["total_trades"], result["closed_trades"],
        result["total_pnl"], result["win_rate"] * 100,
        result["total_net_pnl"], result["net_win_rate"] * 100,
    )


if __name__ == "__main__":
    main()
