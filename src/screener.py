"""Phase 1: Nifty 500 Momentum Screener.

Runs pre-market (08:45 IST). Filters the ~500 stock NSE universe down to the
`top_n` most active, highest-momentum tickers using a 30-day Rate of Change
(ROC), so the live execution phase only has to stream/process a manageable
subset.
"""
import logging

import pandas as pd
import yfinance as yf

from src.config import settings
from src.universe import load_sector_map, load_universe  # noqa: F401 -- re-exported for existing callers

logger = logging.getLogger(__name__)


def compute_momentum_scores(tickers: list[str], lookback_days: int) -> pd.DataFrame:
    """Compute 30-day ROC MomentumScore for each ticker from EOD closes.

    MomentumScore = (close_today - close_n_days_ago) / close_n_days_ago
    """
    period_days = lookback_days + 15  # buffer for weekends/holidays
    raw = yf.download(
        tickers,
        period=f"{period_days}d",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    rows = []
    for ticker in tickers:
        try:
            closes = raw[ticker]["Close"].dropna()
        except (KeyError, TypeError):
            continue
        if len(closes) < lookback_days + 1:
            continue
        current = closes.iloc[-1]
        past = closes.iloc[-(lookback_days + 1)]
        if past == 0 or pd.isna(past) or pd.isna(current):
            continue
        momentum_score = (current - past) / past
        rows.append({"ticker": ticker, "close": current, "momentum_score": momentum_score})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("momentum_score", ascending=False).reset_index(drop=True)


def run_screener() -> pd.DataFrame:
    """Entry point: load universe, score it, return the top_n momentum leaders."""
    tickers = load_universe()
    logger.info("Screening %d tickers for momentum...", len(tickers))
    scored = compute_momentum_scores(tickers, settings.screener.roc_lookback_days)
    top = scored.head(settings.screener.top_n)
    logger.info("Screener selected %d tickers.", len(top))
    return top


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_screener()
    print(result.to_string(index=False))
