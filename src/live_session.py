"""Live execution loop (09:15-15:30 IST): connects to Kotak Neo, streams
ticks, runs the AI signal + portfolio pipeline, and routes orders.

Runs in the LIVE venv (neo_api_client + ML libs, no yfinance -- see
src/premarket.py and requirements-live.txt for why these are split). Reads
the watchlist written by src/premarket.py; refuses to start if that hasn't
run yet rather than trading with an empty/stale watchlist.

Started/stopped by PM2 + cron (scripts/deploy/). On SIGTERM (pm2 stop) or
SIGINT it flushes any open positions before exiting -- MIS positions cannot
be held overnight, matching the blueprint's 15:30 square-off.
"""
import json
import logging
import signal
import sys
import time
from pathlib import Path

import pandas as pd

from src import database
from src.config import settings
from src.costs import net_pnl as compute_net_pnl
from src.execution import KotakNeoClient, TickBarAggregator, reconcile_fill
from src.features import latest_feature_row
from src.model import MomentumClassifier
from src.portfolio import PortfolioManager
from src.universe import load_sector_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("apex_live")

WATCHLIST_PATH = Path(settings.db_path).parent / "watchlist.json"

# Kotak Neo's representation for the Nifty 50 index instrument -- unverified
# against a live session, like the other Kotak Neo field names in this file
# (see README "Known limitations").
NIFTY_50_TOKEN = "NIFTY 50"


class LiveSession:
    def __init__(self):
        self.portfolio = PortfolioManager(sector_map=load_sector_map())
        self.aggregator = TickBarAggregator()
        self.model: MomentumClassifier | None = None
        self.client: KotakNeoClient | None = None
        self.watchlist: list[str] = []
        self.trade_ids: dict[str, int] = {}
        self.last_price: dict[str, float] = {}

    def load_watchlist(self) -> None:
        if not WATCHLIST_PATH.exists():
            raise RuntimeError(
                f"No watchlist at {WATCHLIST_PATH}. Run `python -m src.premarket` "
                "in the MAIN venv first (it needs yfinance, which this venv doesn't have)."
            )
        data = json.loads(WATCHLIST_PATH.read_text())
        self.watchlist = data.get("tickers", [])
        if not self.watchlist:
            raise RuntimeError("Watchlist is empty; refusing to start a session with nothing to trade.")

    def start(self) -> None:
        self.load_watchlist()
        self.model = MomentumClassifier.load()

        self.client = KotakNeoClient()
        self.client.login()
        self.client.subscribe_ticks(self.watchlist, on_tick=self._on_tick)
        self.client.subscribe_ticks([NIFTY_50_TOKEN], on_tick=self._on_tick, is_index=True)
        logger.info(
            "Live execution started for %d tickers + Nifty 50 benchmark (live_trading=%s).",
            len(self.watchlist), settings.live_trading,
        )

    def _on_tick(self, message: dict) -> None:
        """WebSocket callback: buffers the tick, then re-evaluates the AI signal
        for that ticker using the freshest 15-min bar. Index ticks (the Nifty
        50 benchmark) just update its bars for the relative_strength feature --
        there's no position to manage for an index."""
        ticker = message.get("trading_symbol") or message.get("tk")
        price = float(message.get("ltp", 0) or 0)
        volume = float(message.get("v", 0) or 0)
        if not ticker or price <= 0:
            return

        self.aggregator.add_tick(ticker, price, volume, pd.Timestamp.now(tz="Asia/Kolkata"))

        if ticker == NIFTY_50_TOKEN:
            return

        self.last_price[ticker] = price

        if self.portfolio.check_circuit_breaker(self.last_price):
            logger.critical("Circuit breaker tripped: %s", self.portfolio.circuit_breaker_reason)
            self.flush()
            return

        # ATR stop / trailing profit-lock: checked on every tick for every
        # open position, independent of new-signal evaluation below.
        for exit_ticker, exit_price, exit_reason in self.portfolio.check_exits(self.last_price):
            logger.info("%s closed by %s @ %.2f", exit_ticker, exit_reason, exit_price)
            self._close_position(exit_ticker, exit_price, sell_order=True)

        self._evaluate(ticker, price)

    def _evaluate(self, ticker: str, last_price: float) -> None:
        bars = self.aggregator.get_bars(ticker)
        benchmark_bars = self.aggregator.get_bars(NIFTY_50_TOKEN)
        benchmark_close = benchmark_bars["Close"] if not benchmark_bars.empty else None
        feature_row = latest_feature_row(bars, benchmark_close=benchmark_close)
        if feature_row is None:
            return  # still warming up

        now = pd.Timestamp.now(tz="Asia/Kolkata")
        atr = float(feature_row["atr_pct"].iloc[0]) * last_price
        is_buy, confidence = self.model.is_buy_signal(feature_row)
        decision = self.portfolio.evaluate_signal(
            ticker, confidence, last_price, timestamp=now, current_prices=self.last_price, atr=atr,
        )
        database.log_signal(ticker, confidence, decision.approved, decision.reason)

        if not decision.approved:
            if is_buy:
                logger.info("Signal for %s rejected: %s", ticker, decision.reason)
            return

        result = self.client.place_order(ticker, decision.quantity, transaction_type="BUY")
        fill = reconcile_fill(self.client, result.order_id, decision.quantity)
        if fill.filled_quantity == 0:
            logger.error(
                "%s BUY order %s did not fill (status=%s); not opening a position for it.",
                ticker, result.order_id, fill.order_status,
            )
            return

        entry_price = fill.avg_price if fill.avg_price is not None else last_price
        entry_quantity = fill.filled_quantity
        self.portfolio.open_trade(ticker, entry_price, entry_quantity, result.order_id, timestamp=now, atr=atr)
        trade_id = database.log_trade_open(
            ticker, entry_price, entry_quantity, result.order_id, is_paper=result.paper
        )
        self.trade_ids[ticker] = trade_id

    def _close_position(self, ticker: str, exit_price: float, sell_order: bool) -> None:
        """Shared close path for both stop/trail exits (check_exits, mid-day)
        and the end-of-day/shutdown flush below."""
        trade = self.portfolio.active_trades[ticker]
        if sell_order and self.client is not None:
            result = self.client.place_order(ticker, trade.quantity, transaction_type="SELL")
            fill = reconcile_fill(self.client, result.order_id, trade.quantity)
            if fill.avg_price is not None:
                exit_price = fill.avg_price
            if not fill.matches_expected:
                logger.error(
                    "%s SELL order %s only filled %d/%d; position may not be fully flat.",
                    ticker, result.order_id, fill.filled_quantity, trade.quantity,
                )
        pnl = self.portfolio.close_trade(ticker, exit_price)
        # Estimated, not the broker's actual charged amount (we don't parse
        # the charges/contract-note API) -- same cost model as the backtests,
        # for like-for-like comparison against paper-trading results.
        net = compute_net_pnl(
            trade.entry_price, exit_price, trade.quantity,
            entry_ts=trade.opened_at, exit_ts=pd.Timestamp.now(tz="Asia/Kolkata"),
        )
        trade_id = self.trade_ids.pop(ticker, None)
        if trade_id is not None:
            database.log_trade_close(trade_id, exit_price, pnl or 0.0, net)

    def flush(self) -> None:
        """Force-close everything still open -- end of day, or shutdown signal."""
        logger.info("Flushing open positions...")
        for ticker in self.portfolio.flush_all():
            bars = self.aggregator.get_bars(ticker)
            trade = self.portfolio.active_trades[ticker]
            exit_price = float(bars["Close"].iloc[-1]) if not bars.empty else trade.entry_price
            self._close_position(ticker, exit_price, sell_order=True)
        logger.info("Session complete.")


def main() -> None:
    database.init_db()
    session = LiveSession()

    def handle_shutdown(signum, _frame):
        logger.info("Shutdown signal received (%s); flushing before exit.", signum)
        session.flush()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    session.start()

    # Ticks arrive on the WebSocket client's own thread; just keep the process
    # alive until a shutdown signal (pm2 stop / Ctrl+C) arrives. Avoids
    # signal.pause(), which isn't available on Windows.
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
