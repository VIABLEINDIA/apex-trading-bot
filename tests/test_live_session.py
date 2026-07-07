"""Unit tests for src/live_session.py orchestration. Broker, model, database,
and portfolio are all mocked -- these tests cover the wiring (what calls what,
with what data), not real trading logic (covered by test_portfolio.py,
test_features.py, test_execution.py)."""
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.execution import FillReport, OrderResult
from src.live_session import NIFTY_50_TOKEN, LiveSession
from src.portfolio import Decision, Trade


@pytest.fixture(autouse=True)
def no_sector_map(monkeypatch):
    # Keeps tests independent of data/nifty500.csv's actual contents.
    monkeypatch.setattr("src.live_session.load_sector_map", lambda: {})


@pytest.fixture(autouse=True)
def no_real_db(monkeypatch):
    monkeypatch.setattr("src.live_session.database.log_signal", MagicMock())
    monkeypatch.setattr("src.live_session.database.log_trade_open", MagicMock(return_value=1))
    monkeypatch.setattr("src.live_session.database.log_trade_close", MagicMock())


def make_session() -> LiveSession:
    session = LiveSession()
    session.portfolio = MagicMock()
    session.client = MagicMock()
    session.model = MagicMock()
    return session


def tick(ticker="RELIANCE.NS", price=101.0, volume=10):
    return {"trading_symbol": ticker, "ltp": price, "v": volume}


# --- load_watchlist ---------------------------------------------------

def test_load_watchlist_raises_if_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.live_session.WATCHLIST_PATH", tmp_path / "watchlist.json")
    session = make_session()
    with pytest.raises(RuntimeError, match="No watchlist"):
        session.load_watchlist()


def test_load_watchlist_raises_if_tickers_empty(tmp_path, monkeypatch):
    path = tmp_path / "watchlist.json"
    path.write_text(json.dumps({"tickers": []}))
    monkeypatch.setattr("src.live_session.WATCHLIST_PATH", path)
    session = make_session()
    with pytest.raises(RuntimeError, match="empty"):
        session.load_watchlist()


def test_load_watchlist_loads_tickers(tmp_path, monkeypatch):
    path = tmp_path / "watchlist.json"
    path.write_text(json.dumps({"tickers": ["RELIANCE.NS", "TCS.NS"]}))
    monkeypatch.setattr("src.live_session.WATCHLIST_PATH", path)
    session = make_session()
    session.load_watchlist()
    assert session.watchlist == ["RELIANCE.NS", "TCS.NS"]


# --- _on_tick -----------------------------------------------------------

def test_on_tick_ignores_message_without_ticker():
    session = make_session()
    session._on_tick({"ltp": 100, "v": 10})
    session.portfolio.check_circuit_breaker.assert_not_called()


def test_on_tick_ignores_nonpositive_price():
    session = make_session()
    session._on_tick(tick(price=0))
    session.portfolio.check_circuit_breaker.assert_not_called()


def test_on_tick_updates_index_bars_and_returns_early():
    session = make_session()
    session._on_tick(tick(ticker=NIFTY_50_TOKEN, price=25000))
    assert not session.aggregator.get_bars(NIFTY_50_TOKEN).empty
    session.portfolio.check_circuit_breaker.assert_not_called()


def test_on_tick_flushes_and_stops_when_circuit_breaker_trips():
    session = make_session()
    session.portfolio.check_circuit_breaker.return_value = True
    session.portfolio.circuit_breaker_reason = "daily loss limit breached"
    session.flush = MagicMock()
    session._evaluate = MagicMock()

    session._on_tick(tick())

    session.flush.assert_called_once()
    session._evaluate.assert_not_called()


def test_on_tick_closes_positions_returned_by_check_exits():
    session = make_session()
    session.portfolio.check_circuit_breaker.return_value = False
    session.portfolio.check_exits.return_value = [("TCS.NS", 3500.0, "trailing stop")]
    session._close_position = MagicMock()
    session._evaluate = MagicMock()

    session._on_tick(tick(ticker="RELIANCE.NS", price=101.0))

    session._close_position.assert_called_once_with("TCS.NS", 3500.0, sell_order=True)


# --- _evaluate ------------------------------------------------------------

def test_evaluate_returns_early_while_warming_up(monkeypatch):
    monkeypatch.setattr("src.live_session.latest_feature_row", lambda bars, benchmark_close=None: None)
    session = make_session()
    session.aggregator.add_tick("RELIANCE.NS", 100, 10, pd.Timestamp.now(tz="Asia/Kolkata"))

    session._evaluate("RELIANCE.NS", 100)

    session.model.is_buy_signal.assert_not_called()
    session.client.place_order.assert_not_called()


def test_evaluate_logs_rejected_signal_and_places_no_order(monkeypatch):
    fake_row = pd.DataFrame({"atr_pct": [0.01]})
    monkeypatch.setattr("src.live_session.latest_feature_row", lambda bars, benchmark_close=None: fake_row)
    session = make_session()
    session.model.is_buy_signal.return_value = (True, 0.6)
    session.portfolio.evaluate_signal.return_value = Decision(False, "no free slot")

    session._evaluate("RELIANCE.NS", 100.0)

    from src.live_session import database
    database.log_signal.assert_called_once()
    session.client.place_order.assert_not_called()


def test_evaluate_opens_trade_on_full_fill(monkeypatch):
    fake_row = pd.DataFrame({"atr_pct": [0.01]})
    monkeypatch.setattr("src.live_session.latest_feature_row", lambda bars, benchmark_close=None: fake_row)
    session = make_session()
    session.model.is_buy_signal.return_value = (True, 0.7)
    session.portfolio.evaluate_signal.return_value = Decision(True, "approved", quantity=10, capital_allocated=1000)
    session.client.place_order.return_value = OrderResult(order_id="PAPER-abc", status="simulated", paper=True)
    monkeypatch.setattr(
        "src.live_session.reconcile_fill",
        lambda client, order_id, expected_quantity: FillReport(
            order_id=order_id, order_status="simulated", filled_quantity=expected_quantity,
            avg_price=None, matches_expected=True,
        ),
    )

    session._evaluate("RELIANCE.NS", 100.0)

    session.portfolio.open_trade.assert_called_once()
    args, kwargs = session.portfolio.open_trade.call_args
    assert args[0] == "RELIANCE.NS"
    assert args[1] == 100.0  # entry price falls back to last_price (no broker avg_price)
    assert args[2] == 10
    assert session.trade_ids["RELIANCE.NS"] == 1


def test_evaluate_skips_opening_trade_when_order_unfilled(monkeypatch):
    fake_row = pd.DataFrame({"atr_pct": [0.01]})
    monkeypatch.setattr("src.live_session.latest_feature_row", lambda bars, benchmark_close=None: fake_row)
    session = make_session()
    session.model.is_buy_signal.return_value = (True, 0.7)
    session.portfolio.evaluate_signal.return_value = Decision(True, "approved", quantity=10, capital_allocated=1000)
    session.client.place_order.return_value = OrderResult(order_id="ORD1", status="rejected", paper=False)
    monkeypatch.setattr(
        "src.live_session.reconcile_fill",
        lambda client, order_id, expected_quantity: FillReport(
            order_id=order_id, order_status="rejected", filled_quantity=0,
            avg_price=None, matches_expected=False,
        ),
    )

    session._evaluate("RELIANCE.NS", 100.0)

    session.portfolio.open_trade.assert_not_called()
    from src.live_session import database
    database.log_trade_open.assert_not_called()
    assert "RELIANCE.NS" not in session.trade_ids


def test_evaluate_uses_broker_confirmed_avg_price_when_available(monkeypatch):
    fake_row = pd.DataFrame({"atr_pct": [0.01]})
    monkeypatch.setattr("src.live_session.latest_feature_row", lambda bars, benchmark_close=None: fake_row)
    session = make_session()
    session.model.is_buy_signal.return_value = (True, 0.7)
    session.portfolio.evaluate_signal.return_value = Decision(True, "approved", quantity=10, capital_allocated=1000)
    session.client.place_order.return_value = OrderResult(order_id="ORD1", status="complete", paper=False)
    monkeypatch.setattr(
        "src.live_session.reconcile_fill",
        lambda client, order_id, expected_quantity: FillReport(
            order_id=order_id, order_status="complete", filled_quantity=10,
            avg_price=100.75, matches_expected=True,
        ),
    )

    session._evaluate("RELIANCE.NS", 100.0)

    args, _ = session.portfolio.open_trade.call_args
    assert args[1] == 100.75  # broker-confirmed avg price, not the last-seen tick


# --- _close_position -------------------------------------------------------

def test_close_position_uses_broker_confirmed_exit_price(monkeypatch):
    session = make_session()
    trade = Trade(ticker="RELIANCE.NS", entry_price=100.0, quantity=10, opened_at=pd.Timestamp.now(tz="Asia/Kolkata"))
    session.portfolio.active_trades = {"RELIANCE.NS": trade}
    session.client.place_order.return_value = OrderResult(order_id="ORD2", status="complete", paper=False)
    monkeypatch.setattr(
        "src.live_session.reconcile_fill",
        lambda client, order_id, expected_quantity: FillReport(
            order_id=order_id, order_status="complete", filled_quantity=10,
            avg_price=99.25, matches_expected=True,
        ),
    )
    session.portfolio.close_trade.return_value = -7.5

    session._close_position("RELIANCE.NS", exit_price=99.0, sell_order=True)

    session.portfolio.close_trade.assert_called_once_with("RELIANCE.NS", 99.25)


def test_close_position_without_broker_uses_passed_exit_price():
    session = make_session()
    trade = Trade(ticker="RELIANCE.NS", entry_price=100.0, quantity=10, opened_at=pd.Timestamp.now(tz="Asia/Kolkata"))
    session.portfolio.active_trades = {"RELIANCE.NS": trade}
    session.portfolio.close_trade.return_value = 50.0

    session._close_position("RELIANCE.NS", exit_price=105.0, sell_order=False)

    session.portfolio.close_trade.assert_called_once_with("RELIANCE.NS", 105.0)
    session.client.place_order.assert_not_called()


# --- flush ------------------------------------------------------------------

def test_flush_closes_all_open_positions_using_last_bar_close():
    session = make_session()
    trade = Trade(ticker="RELIANCE.NS", entry_price=100.0, quantity=10, opened_at=pd.Timestamp.now(tz="Asia/Kolkata"))
    session.portfolio.active_trades = {"RELIANCE.NS": trade}
    session.portfolio.flush_all.return_value = ["RELIANCE.NS"]
    session.aggregator.add_tick("RELIANCE.NS", 103.5, 10, pd.Timestamp.now(tz="Asia/Kolkata"))
    session._close_position = MagicMock()

    session.flush()

    session._close_position.assert_called_once_with("RELIANCE.NS", 103.5, sell_order=True)


def test_flush_falls_back_to_entry_price_when_no_bars_seen():
    session = make_session()
    trade = Trade(ticker="RELIANCE.NS", entry_price=100.0, quantity=10, opened_at=pd.Timestamp.now(tz="Asia/Kolkata"))
    session.portfolio.active_trades = {"RELIANCE.NS": trade}
    session.portfolio.flush_all.return_value = ["RELIANCE.NS"]
    session._close_position = MagicMock()

    session.flush()

    session._close_position.assert_called_once_with("RELIANCE.NS", 100.0, sell_order=True)
