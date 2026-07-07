"""Shared historical-data fetching: batches yfinance downloads across many
tickers (fast, few HTTP round trips) with retry/backoff, used by both the
training script and the historical paper-trading simulator.
"""
import logging
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CHUNK_SIZE = 50  # tickers per yf.download batch call
CHUNK_PAUSE_SECONDS = 2  # be polite to Yahoo's endpoint between batches
MAX_RETRIES = 3


def _download_chunk(tickers: list[str], days: int, interval: str) -> pd.DataFrame | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(
                tickers, period=f"{days}d", interval=interval, auto_adjust=True,
                progress=False, group_by="ticker", threads=True,
            )
            if not df.empty:
                return df
        except Exception as exc:  # yfinance raises assorted network/HTTP errors
            logger.warning("Batch download attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
        time.sleep(2 * attempt)  # backoff before retrying
    logger.warning("Giving up on batch of %d tickers after %d attempts.", len(tickers), MAX_RETRIES)
    return None


def fetch_bars_by_ticker(tickers: list[str], days: int, interval: str = "15m") -> dict[str, pd.DataFrame]:
    """Batch-download OHLCV bars and return one raw frame per ticker (no
    features/labels attached -- callers featurize per ticker themselves)."""
    result: dict[str, pd.DataFrame] = {}
    chunks = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]

    for i, chunk in enumerate(chunks, start=1):
        logger.info("Downloading chunk %d/%d (%d tickers)...", i, len(chunks), len(chunk))
        batch = _download_chunk(chunk, days, interval)
        if batch is None:
            continue

        # yf.download with group_by="ticker" returns MultiIndex columns
        # (ticker, field) regardless of chunk size -- even for a single
        # ticker -- so always index by ticker rather than special-casing
        # len(chunk) == 1 as "already flat".
        is_multiindex = isinstance(batch.columns, pd.MultiIndex)
        for ticker in chunk:
            try:
                bars = batch[ticker] if is_multiindex else batch
            except KeyError:
                logger.warning("No %s data for %s, skipping.", interval, ticker)
                continue
            bars = bars.dropna(how="all")
            if bars.empty:
                logger.warning("No %s data for %s, skipping.", interval, ticker)
                continue
            result[ticker] = bars[["Open", "High", "Low", "Close", "Volume"]]

        if i < len(chunks):
            time.sleep(CHUNK_PAUSE_SECONDS)

    if not result:
        raise RuntimeError("No data fetched for any ticker.")
    logger.info("Fetched usable data for %d/%d tickers.", len(result), len(tickers))
    return result


BENCHMARK_TICKER = "^NSEI"  # Nifty 50 index


def fetch_benchmark_close(days: int, interval: str = "15m") -> pd.Series:
    """Fetch the Nifty 50 index's Close series, used as the market baseline
    for the `relative_strength` feature (src/features.py)."""
    bars = fetch_bars_by_ticker([BENCHMARK_TICKER], days, interval=interval)
    return bars[BENCHMARK_TICKER]["Close"]
