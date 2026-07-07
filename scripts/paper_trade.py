"""Historical paper-trading simulation.

Runs the full pipeline (screener -> features -> model -> portfolio) against
real historical 15-min bars from yfinance. Every simulated fill/signal is
written to the same SQLite journal (trades, ai_signals) the live system uses,
so results can be inspected the same way. Does NOT require Kotak Neo
credentials -- it never touches src/execution.py, since paper mode doesn't
need a broker connection at all.

Indicators (RSI/MACD/ATR/etc.) are computed ONCE per ticker over the full
history up front (they're causal -- rolling/EWM only look backward -- so this
gives identical values to recomputing on every growing window, just without
the O(n^2) cost of redoing it per bar). Positions are force-flattened at the
end of each trading day, matching the blueprint's intraday-only (MIS)
constraint -- no overnight holding.

Usage:
    python scripts/paper_trade.py                      # today's screener watchlist
    python scripts/paper_trade.py RELIANCE.NS TCS.NS    # explicit tickers
    python scripts/paper_trade.py --days 20
"""
import argparse
import itertools
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import database  # noqa: E402
from src.config import settings  # noqa: E402
from src.costs import net_pnl as compute_net_pnl  # noqa: E402
from src.features import FEATURE_COLUMNS, add_features  # noqa: E402
from src.market_data import fetch_bars_by_ticker, fetch_benchmark_close  # noqa: E402
from src.model import MomentumClassifier  # noqa: E402
from src.portfolio import PortfolioManager  # noqa: E402
from src.screener import run_screener  # noqa: E402
from src.universe import load_sector_map  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def simulate(ticker_bars: dict, model: MomentumClassifier, start_date=None, benchmark_close=None,
            portfolio: PortfolioManager | None = None, use_atr_stop: bool = True, log_to_db: bool = True) -> dict:
    """Replay `ticker_bars` bar-by-bar through the live decision pipeline.

    `start_date`, if given, restricts *trading* (signal evaluation and order
    entry) to bars on/after that date, while earlier bars are still fed
    through so indicators (RSI/MACD/ATR/...) are warmed up on real history
    instead of restarting cold at the window boundary. This is what makes it
    possible to do a genuine out-of-sample backtest: pass the full history for
    indicator continuity, but set `start_date` to the first day the model
    didn't see during training (see scripts/validate_walkforward.py).

    `benchmark_close` is the Nifty 50 index Close series, used for the
    `relative_strength` feature (see src/features.py).

    `portfolio`, if given, is used as-is (with whatever risk/sizing/session
    parameters it was constructed with) instead of building a default one --
    lets callers like scripts/tune_strategy.py sweep many configurations
    against the same precomputed features without re-fetching/re-training.

    `use_atr_stop=False` disables the ATR stop/trailing mechanism entirely
    (always passes atr=None to evaluate_signal/open_trade, regardless of the
    computed value) -- lets a parameter sweep test "mechanism off" as a
    candidate alongside different ATR ratios, since the pre-BALU-strategy
    baseline (flat sizing, no per-trade stop) was net-positive in most
    windows while the ATR mechanism has so far tested worse (see README
    "Model performance").

    `log_to_db=False` skips writing to the SQLite journal -- a sweep running
    dozens of configurations shouldn't flood data/trading_journal.db.

    Candidates are evaluated in batches per timestamp (every ticker with a bar
    at that instant), ranked by confidence, and slots are filled
    highest-confidence-first -- not first-come-first-served in ticker-name
    order, which is what a naive per-event loop does.
    """
    if portfolio is None:
        portfolio = PortfolioManager(sector_map=load_sector_map())

    # Precompute features once per ticker -- rolling/EWM windows are causal
    # (only look backward), so this is equivalent to recomputing on every
    # growing slice, just O(n) instead of O(n^2).
    ticker_features = {
        ticker: add_features(bars, benchmark_close=benchmark_close).dropna(subset=FEATURE_COLUMNS)
        for ticker, bars in ticker_bars.items()
    }

    events = sorted(
        (ts, ticker) for ticker, feat in ticker_features.items() for ts in feat.index
    )

    open_trades: dict[str, dict] = {}  # ticker -> {entry_price, quantity, order_id, opened_at}
    signal_rows = []
    trade_rows = []
    last_price_seen: dict[str, float] = {}
    last_ts_seen: dict[str, object] = {}
    total_trades = 0
    total_pnl = 0.0
    total_net_pnl = 0.0
    current_date = None

    def close_position(ticker: str, exit_price: float) -> None:
        nonlocal total_pnl, total_net_pnl
        pnl = portfolio.close_trade(ticker, exit_price)
        total_pnl += pnl or 0.0
        trade = open_trades.pop(ticker)
        net = compute_net_pnl(
            trade["entry_price"], exit_price, trade["quantity"],
            entry_ts=trade["opened_at"], exit_ts=last_ts_seen[ticker],
        )
        total_net_pnl += net
        trade_rows.append((
            ticker, trade["entry_price"], exit_price, trade["quantity"], pnl or 0.0, net,
            trade["order_id"], True, trade["opened_at"], last_ts_seen[ticker],
        ))

    def flatten_all(reason: str) -> None:
        for ticker in portfolio.flush_all():
            close_position(ticker, last_price_seen[ticker])
        logger.debug("Flattened all positions (%s)", reason)

    for ts, ts_events in itertools.groupby(events, key=lambda e: e[0]):
        tickers_at_ts = [ticker for _, ticker in ts_events]
        ts_date = ts.normalize()

        if current_date is not None and ts_date != current_date:
            flatten_all(f"end of trading day {current_date.date()}")
            portfolio.reset_day()
        current_date = ts_date

        for ticker in tickers_at_ts:
            price = float(ticker_features[ticker].loc[ts, "Close"])
            last_price_seen[ticker] = price
            last_ts_seen[ticker] = ts

        if portfolio.check_circuit_breaker(last_price_seen):
            flatten_all(f"circuit breaker: {portfolio.circuit_breaker_reason}")

        # ATR stop / trailing profit-lock: checked every bar for every open
        # position, independent of the day-end/circuit-breaker flatten above.
        for ticker, exit_price, exit_reason in portfolio.check_exits(last_price_seen):
            close_position(ticker, exit_price)
            logger.debug("%s closed by %s @ %.2f", ticker, exit_reason, exit_price)

        if start_date is not None and ts_date < start_date:
            continue  # warm-up-only period: indicators update, but no trading yet

        # Cross-sectional ranking: score every candidate at this timestamp
        # first, then fill slots strongest-confidence-first instead of in
        # ticker-name order.
        candidates = []
        for ticker in tickers_at_ts:
            feature_row = ticker_features[ticker].loc[[ts], FEATURE_COLUMNS]
            _, confidence = model.is_buy_signal(feature_row)
            atr = float(feature_row["atr_pct"].iloc[0]) * last_price_seen[ticker] if use_atr_stop else None
            candidates.append((ticker, confidence, last_price_seen[ticker], atr))
        candidates.sort(key=lambda c: c[1], reverse=True)

        for ticker, confidence, price, atr in candidates:
            decision = portfolio.evaluate_signal(
                ticker, confidence, price, timestamp=ts, current_prices=last_price_seen, atr=atr,
            )
            signal_rows.append((ticker, confidence, decision.approved, decision.reason, ts))
            if not decision.approved:
                continue

            order_id = f"PAPER-BT-{total_trades:06d}"
            portfolio.open_trade(ticker, price, decision.quantity, order_id, timestamp=ts, atr=atr)
            open_trades[ticker] = {
                "entry_price": price, "quantity": decision.quantity,
                "order_id": order_id, "opened_at": ts,
            }
            total_trades += 1

    flatten_all("end of simulation window")

    if log_to_db:
        database.log_signals_bulk(signal_rows)
        database.log_trades_bulk(trade_rows)

    wins = sum(1 for r in trade_rows if r[4] > 0)
    net_wins = sum(1 for r in trade_rows if r[5] > 0)
    closed = len(trade_rows)
    return {
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "total_net_pnl": total_net_pnl,
        "closed_trades": closed,
        "win_rate": wins / closed if closed else 0.0,
        "net_win_rate": net_wins / closed if closed else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", help="Tickers to simulate. Defaults to today's screener watchlist.")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days.")
    args = parser.parse_args()

    database.init_db()

    if args.tickers:
        tickers = args.tickers
    else:
        screened = run_screener()
        if screened.empty:
            logger.error("Screener returned no tickers; pass explicit tickers instead.")
            return
        tickers = screened["ticker"].tolist()

    logger.info(
        "Simulating %d tickers over %d days (confidence_threshold=%.2f, max_slots=%d)...",
        len(tickers), args.days, settings.risk.confidence_threshold, settings.risk.max_slots,
    )

    ticker_bars = fetch_bars_by_ticker(tickers, args.days)
    benchmark_close = fetch_benchmark_close(args.days)
    model = MomentumClassifier.load()
    result = simulate(ticker_bars, model, benchmark_close=benchmark_close)

    logger.info(
        "Paper-trading simulation complete: %d trades opened. "
        "Gross (frictionless): PnL=%.2f win_rate=%.1f%%. "
        "Net (after slippage/STT/charges): PnL=%.2f win_rate=%.1f%%.",
        result["total_trades"], result["total_pnl"], result["win_rate"] * 100,
        result["total_net_pnl"], result["net_win_rate"] * 100,
    )
    logger.info("Full signal/trade log written to %s", settings.db_path)


if __name__ == "__main__":
    main()
