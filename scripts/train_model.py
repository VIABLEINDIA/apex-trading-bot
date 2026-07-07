"""Train (or retrain) the GradientBoostingClassifier on historical 15-min bars.

Usage:
    python scripts/train_model.py RELIANCE.NS TCS.NS INFY.NS
    python scripts/train_model.py --days 14   # walk-forward retrain, see README

yfinance only serves ~60 days of 15-minute history, which fits the blueprint's
bi-weekly walk-forward retraining cadence (section 4).
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.market_data import fetch_bars_by_ticker, fetch_benchmark_close  # noqa: E402
from src.model import MomentumClassifier  # noqa: E402
from src.screener import load_universe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", help="Tickers to train on, e.g. RELIANCE.NS. Defaults to the full universe.")
    parser.add_argument("--days", type=int, default=59, help="Lookback window in days (yfinance max ~60 for 15m bars).")
    args = parser.parse_args()

    tickers = args.tickers or load_universe()
    logger.info("Fetching %d days of 15m bars for %d tickers...", args.days, len(tickers))
    ticker_bars = fetch_bars_by_ticker(tickers, args.days)
    benchmark_close = fetch_benchmark_close(args.days)

    classifier = MomentumClassifier()
    report = classifier.train_multi(list(ticker_bars.values()), benchmark_close=benchmark_close)
    logger.info("Holdout classification report:\n%s", report)
    classifier.save()


if __name__ == "__main__":
    main()
