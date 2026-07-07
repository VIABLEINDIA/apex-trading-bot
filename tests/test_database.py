import sqlite3
from datetime import datetime, timezone

import pytest

from src import database


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    # settings.db_path lives on a frozen dataclass, so patch the helper that
    # reads it instead of trying to mutate the (immutable) field directly.
    db_path = tmp_path / "journal.db"
    monkeypatch.setattr("src.database._db_path", lambda: str(db_path))
    database.init_db()
    return db_path


def test_init_db_creates_both_tables(temp_db):
    with sqlite3.connect(temp_db) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"trades", "ai_signals"} <= tables


def test_log_signal_inserts_a_row(temp_db):
    database.log_signal("RELIANCE.NS", 0.62, True, "approved")

    with sqlite3.connect(temp_db) as conn:
        rows = conn.execute("SELECT ticker, confidence, approved, reason FROM ai_signals").fetchall()
    assert rows == [("RELIANCE.NS", 0.62, 1, "approved")]


def test_log_trade_open_returns_new_row_id_and_persists_fields(temp_db):
    trade_id = database.log_trade_open("RELIANCE.NS", 101.5, 10, "PAPER-abc", is_paper=True)

    assert trade_id is not None
    with sqlite3.connect(temp_db) as conn:
        row = conn.execute(
            "SELECT ticker, entry_price, quantity, order_id, is_paper FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()
    assert row == ("RELIANCE.NS", 101.5, 10, "PAPER-abc", 1)


def test_log_trade_close_updates_the_matching_row(temp_db):
    trade_id = database.log_trade_open("RELIANCE.NS", 100.0, 10, "PAPER-abc", is_paper=True)

    database.log_trade_close(trade_id, exit_price=105.0, pnl=50.0, net_pnl=42.0)

    with sqlite3.connect(temp_db) as conn:
        row = conn.execute(
            "SELECT exit_price, pnl, net_pnl FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()
    assert row == (105.0, 50.0, 42.0)


def test_log_signals_bulk_inserts_all_rows(temp_db):
    rows = [
        ("RELIANCE.NS", 0.6, True, "approved", datetime(2026, 1, 5, 9, 30, tzinfo=timezone.utc)),
        ("TCS.NS", 0.4, False, "below threshold", datetime(2026, 1, 5, 9, 31, tzinfo=timezone.utc)),
    ]

    database.log_signals_bulk(rows)

    with sqlite3.connect(temp_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM ai_signals").fetchone()[0]
    assert count == 2


def test_log_trades_bulk_inserts_already_closed_trades(temp_db):
    opened = datetime(2026, 1, 5, 9, 30, tzinfo=timezone.utc)
    closed = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    rows = [
        ("RELIANCE.NS", 100.0, 105.0, 10, 50.0, 42.0, "PAPER-1", True, opened, closed),
    ]

    database.log_trades_bulk(rows)

    with sqlite3.connect(temp_db) as conn:
        row = conn.execute("SELECT ticker, pnl, net_pnl, is_paper FROM trades").fetchone()
    assert row == ("RELIANCE.NS", 50.0, 42.0, 1)


def test_get_connection_rolls_back_nothing_but_commits_on_clean_exit(temp_db):
    with database.get_connection() as conn:
        conn.execute(
            "INSERT INTO ai_signals (ticker, confidence, approved, reason, created_at) VALUES (?, ?, ?, ?, ?)",
            ("RELIANCE.NS", 0.5, 0, "test", "2026-01-05T09:30:00+00:00"),
        )

    with sqlite3.connect(temp_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM ai_signals").fetchone()[0]
    assert count == 1
