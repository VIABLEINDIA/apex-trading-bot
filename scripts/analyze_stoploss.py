"""Quantify the no-stop-loss risk in the current strategy and estimate what
several stop-loss thresholds would have changed.

The portfolio manager currently has no intraday exit logic at all -- a
position opened at any point in the day is held until the 15:30 IST
square-off regardless of what happens to it in between (see
src/live_session.py's LiveSession.flush / scripts/paper_trade.py's
flatten_all). This script re-fetches the real 15-min bar path for every trade
already logged in data/trading_journal.db (from the walk-forward validation
runs) and answers two questions:

1. How deep did positions actually go intraday before their eventual close
   (max adverse excursion, "MAE")? The realized PnL in the journal only shows
   the end-of-day outcome, not how much worse it got in between.
2. If a stop-loss at various thresholds (0.5%/1%/1.5%/2%/3%) had force-closed
   a position the first time it breached that loss level, what would total
   PnL and win rate have been instead of the no-stop-loss baseline?

Usage:
    python scripts/analyze_stoploss.py
"""
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.market_data import fetch_bars_by_ticker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STOP_LOSS_THRESHOLDS = [0.005, 0.01, 0.015, 0.02, 0.03]


def load_trades() -> pd.DataFrame:
    import sqlite3
    conn = sqlite3.connect(settings.db_path)
    df = pd.read_sql_query(
        "SELECT ticker, entry_price, exit_price, quantity, pnl, opened_at, closed_at "
        "FROM trades WHERE closed_at IS NOT NULL",
        conn,
        parse_dates=["opened_at", "closed_at"],
    )
    conn.close()
    return df


def analyze(trades: pd.DataFrame, ticker_bars: dict) -> None:
    mae_pct = []  # max adverse excursion, as a fraction of entry price, per trade
    stop_loss_pnls = {th: [] for th in STOP_LOSS_THRESHOLDS}
    stop_loss_hits = {th: 0 for th in STOP_LOSS_THRESHOLDS}
    missing = 0

    for row in trades.itertuples():
        bars = ticker_bars.get(row.ticker)
        if bars is None:
            missing += 1
            continue
        path = bars.loc[row.opened_at:row.closed_at, "Close"]
        if path.empty:
            missing += 1
            continue

        drawdown = (path / row.entry_price - 1).clip(upper=0)  # only care about adverse moves
        worst = -drawdown.min()  # positive number = worst % drop from entry
        mae_pct.append(worst)

        for th in STOP_LOSS_THRESHOLDS:
            breach = path[path <= row.entry_price * (1 - th)]
            if not breach.empty:
                stop_loss_hits[th] += 1
                exit_price = float(breach.iloc[0])
            else:
                exit_price = row.exit_price
            stop_loss_pnls[th].append((exit_price - row.entry_price) * row.quantity)

    if missing:
        logger.warning("Skipped %d/%d trades (no matching bar data).", missing, len(trades))

    mae = pd.Series(mae_pct)
    logger.info("\n--- Intraday drawdown (MAE) on the %d trades we have bar data for ---", len(mae))
    logger.info("mean=%.2f%% median=%.2f%% p90=%.2f%% max=%.2f%%",
                mae.mean() * 100, mae.median() * 100, mae.quantile(0.9) * 100, mae.max() * 100)
    for th in STOP_LOSS_THRESHOLDS:
        frac = (mae >= th).mean()
        logger.info("fraction of trades whose intraday drawdown reached >= %.1f%%: %.1f%%", th * 100, frac * 100)

    baseline_pnl = trades["pnl"].sum()
    baseline_wr = (trades["pnl"] > 0).mean()
    logger.info("\n--- Baseline (no stop-loss, actual journal outcome) ---")
    logger.info("total_pnl=%.2f win_rate=%.1f%% worst_trade=%.2f", baseline_pnl, baseline_wr * 100, trades["pnl"].min())

    logger.info("\n--- What each stop-loss threshold would have changed ---")
    for th in STOP_LOSS_THRESHOLDS:
        pnls = pd.Series(stop_loss_pnls[th])
        logger.info(
            "stop=%.1f%%: hit on %d/%d trades, total_pnl=%.2f (baseline %.2f), win_rate=%.1f%%, worst_trade=%.2f",
            th * 100, stop_loss_hits[th], len(trades), pnls.sum(), baseline_pnl, (pnls > 0).mean() * 100, pnls.min(),
        )


def main() -> None:
    trades = load_trades()
    if trades.empty:
        logger.error("No closed trades found in %s -- run a walk-forward/paper-trade simulation first.", settings.db_path)
        return

    tickers = sorted(trades["ticker"].unique())
    start = trades["opened_at"].min().normalize()
    end = trades["closed_at"].max().normalize()
    days = (pd.Timestamp.now(tz=start.tz).normalize() - start).days + 2
    logger.info("Re-fetching 15m bars for %d tickers (%s to %s, ~%d days back)...", len(tickers), start.date(), end.date(), days)

    ticker_bars = fetch_bars_by_ticker(tickers, days)
    analyze(trades, ticker_bars)


if __name__ == "__main__":
    main()
