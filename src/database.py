"""Phase 5: SQLite journal.

Creates and writes to `trades` (executed trades, for PnL tracking) and
`ai_signals` (every signal the AI produced, including rejections, for
debugging model drift) so the journal can be pulled for offline
walk-forward analysis.
"""
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity INTEGER NOT NULL,
    pnl REAL,
    net_pnl REAL,
    order_id TEXT,
    is_paper INTEGER NOT NULL DEFAULT 1,
    opened_at TEXT NOT NULL,
    closed_at TEXT
);

CREATE TABLE IF NOT EXISTS ai_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    confidence REAL NOT NULL,
    approved INTEGER NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _db_path() -> str:
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    return settings.db_path


@contextmanager
def get_connection():
    conn = sqlite3.connect(_db_path())
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
    logger.info("Database initialized at %s", _db_path())


def log_signal(ticker: str, confidence: float, approved: bool, reason: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO ai_signals (ticker, confidence, approved, reason, created_at) VALUES (?, ?, ?, ?, ?)",
            (ticker, confidence, int(approved), reason, datetime.now(timezone.utc).isoformat()),
        )


def log_trade_open(ticker: str, entry_price: float, quantity: int, order_id: str, is_paper: bool) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO trades (ticker, entry_price, quantity, order_id, is_paper, opened_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ticker, entry_price, quantity, order_id, int(is_paper), datetime.now(timezone.utc).isoformat()),
        )
        return cursor.lastrowid


def log_trade_close(trade_id: int, exit_price: float, pnl: float, net_pnl: float | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE trades SET exit_price = ?, pnl = ?, net_pnl = ?, closed_at = ? WHERE id = ?",
            (exit_price, pnl, net_pnl, datetime.now(timezone.utc).isoformat(), trade_id),
        )


def log_signals_bulk(rows: list[tuple[str, float, bool, str, object]]) -> None:
    """Batch insert for offline simulations (e.g. scripts/paper_trade.py) that
    would otherwise open/commit/close a connection per event. Each row is
    (ticker, confidence, approved, reason, timestamp)."""
    with get_connection() as conn:
        conn.executemany(
            "INSERT INTO ai_signals (ticker, confidence, approved, reason, created_at) VALUES (?, ?, ?, ?, ?)",
            [(t, c, int(a), r, ts.isoformat() if hasattr(ts, "isoformat") else ts) for t, c, a, r, ts in rows],
        )


def log_trades_bulk(rows: list[tuple[str, float, float, int, float, float, str, bool, object, object]]) -> None:
    """Batch insert already-closed trades for offline simulations. Each row is
    (ticker, entry_price, exit_price, quantity, pnl, net_pnl, order_id, is_paper, opened_at, closed_at)."""
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO trades (ticker, entry_price, exit_price, quantity, pnl, net_pnl, order_id, is_paper, opened_at, closed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    t, entry, exit_, qty, pnl, net_pnl, oid, int(paper),
                    opened.isoformat() if hasattr(opened, "isoformat") else opened,
                    closed.isoformat() if hasattr(closed, "isoformat") else closed,
                )
                for t, entry, exit_, qty, pnl, net_pnl, oid, paper, opened, closed in rows
            ],
        )
