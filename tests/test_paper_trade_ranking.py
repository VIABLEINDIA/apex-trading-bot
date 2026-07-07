"""Tests for the cross-sectional ranking behavior in scripts/paper_trade.py's
simulate(): when several tickers cross the confidence threshold at the same
timestamp, the strongest signal should fill the available slot(s) first,
regardless of ticker name / arrival order.

Confidence is driven entirely through `volume_surge` (Volume / 10-bar rolling
average) rather than a real trained model, so trade timing is fully
deterministic: every bar but the last has a flat Volume history (surge
exactly 1.0, confidence exactly 0.5, never approved), and only the final bar's
Volume is set per-ticker to produce a distinct, controlled confidence.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from paper_trade import simulate  # noqa: E402

from src import database
from src.portfolio import PortfolioManager

# Real default confidence_threshold is 0.55 (see .env.example / src/config.py).
# Tests are built around that value rather than patching it, since Settings
# and RiskConfig are frozen dataclasses -- their fields can't be monkeypatched
# in place, only the module-level `settings` name itself can be rebound.


def _primary_session_index(n: int) -> pd.DatetimeIndex:
    """n timestamps confined to the 09:15-10:15 primary session (4 bars/day),
    spanning as many days as needed. `simulate()` now applies session-based
    entry gating (see PortfolioManager.session_state), so tests that check
    ranking/sizing behavior at the final (signal-bearing) bar need it to fall
    within the primary window regardless of how many bars `n` requires for
    feature warmup -- a plain multi-hour date_range would run the last bar
    into the continuation session or past it entirely.
    """
    bars_per_day = 4  # 09:15, 09:30, 09:45, 10:00
    timestamps = []
    day_offset = 0
    while len(timestamps) < n:
        day_start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day_offset, hours=9, minutes=15)
        timestamps.extend(pd.date_range(day_start, periods=bars_per_day, freq="15min"))
        day_offset += 1
    return pd.DatetimeIndex(timestamps[:n])


def make_bars_with_final_volume(final_volume: float, n: int = 40, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = _primary_session_index(n)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0, 0.3, n)
    low = close - rng.uniform(0, 0.3, n)
    open_ = close + rng.normal(0, 0.1, n)
    volume = np.full(n, 1000.0)
    volume[-1] = final_volume  # only the last bar deviates -> confidence spikes only there
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


class VolumeSurgeConfidenceModel:
    """Confidence = volume_surge / 2, clipped to [0, 1] -- lets tests control
    confidence deterministically via each ticker's final-bar Volume."""

    def is_buy_signal(self, feature_row: pd.DataFrame):
        raw = float(feature_row["volume_surge"].iloc[0])
        confidence = min(max(raw / 2.0, 0.0), 1.0)
        return confidence > 0.5, confidence


def test_cross_sectional_ranking_prefers_strongest_signal_for_the_only_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "settings", types.SimpleNamespace(db_path=str(tmp_path / "test.db")))
    database.init_db()

    # Final-bar volume_surge: AAA=1.2 -> confidence 0.6 (candidate, threshold is 0.55)
    #                          BBB=0.8 -> confidence 0.4 (below threshold, never a candidate)
    #                          CCC=1.8 -> confidence 0.9 (candidate, strongest)
    ticker_bars = {
        "AAA.NS": make_bars_with_final_volume(1200.0, seed=1),
        "BBB.NS": make_bars_with_final_volume(800.0, seed=2),
        "CCC.NS": make_bars_with_final_volume(1800.0, seed=3),
    }

    only_one_slot = PortfolioManager(max_slots=1)
    monkeypatch.setattr("paper_trade.PortfolioManager", lambda **kwargs: only_one_slot)

    result = simulate(ticker_bars, VolumeSurgeConfidenceModel())

    assert result["total_trades"] == 1
    opened = _opened_tickers(tmp_path)
    assert opened == {"CCC.NS"}, (
        "with only 1 slot and both AAA (0.6) and CCC (0.9) crossing threshold, "
        "the higher-confidence CCC should win the slot -- not AAA, which would "
        "win under naive alphabetical/arrival-order processing"
    )


def test_cross_sectional_ranking_fills_multiple_slots_by_confidence(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "settings", types.SimpleNamespace(db_path=str(tmp_path / "test.db")))
    database.init_db()

    # 4 candidates above the 0.55 threshold, only 2 slots -- the top 2 by
    # confidence (CCC=0.9, DDD=0.8) should win, not the alphabetically-first
    # two (AAA, BBB).
    ticker_bars = {
        "AAA.NS": make_bars_with_final_volume(1120.0, seed=1),  # confidence 0.56
        "BBB.NS": make_bars_with_final_volume(1150.0, seed=2),  # confidence 0.575
        "CCC.NS": make_bars_with_final_volume(1800.0, seed=3),  # confidence 0.9
        "DDD.NS": make_bars_with_final_volume(1600.0, seed=4),  # confidence 0.8
    }

    two_slots = PortfolioManager(max_slots=2)
    monkeypatch.setattr("paper_trade.PortfolioManager", lambda **kwargs: two_slots)

    result = simulate(ticker_bars, VolumeSurgeConfidenceModel())

    assert result["total_trades"] == 2
    assert _opened_tickers(tmp_path) == {"CCC.NS", "DDD.NS"}


def _opened_tickers(tmp_path):
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    rows = {r[0] for r in conn.execute("SELECT ticker FROM trades").fetchall()}
    conn.close()
    return rows
