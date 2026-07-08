"""Tests that scripts/paper_trade.py's simulate() actually drives
PortfolioManager.on_timestamp() -- the wiring that lets a stateful regime
gate (src/regime.py:RealizedVolatilityGate) answer "as of" the simulated
instant instead of needing a wall clock. See src/portfolio.py:on_timestamp
for why a regime_gate's check() has to stay a zero-arg call.
"""
import sys
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from paper_trade import simulate  # noqa: E402

from src import database
from src.portfolio import PortfolioManager
from tests.test_paper_trade_ranking import VolumeSurgeConfidenceModel, make_bars_with_final_volume


def _init_test_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "settings", types.SimpleNamespace(db_path=str(tmp_path / "test.db")))
    database.init_db()


def test_on_timestamp_called_once_per_event_timestamp_in_order(tmp_path, monkeypatch):
    _init_test_db(tmp_path, monkeypatch)
    ticker_bars = {"AAA.NS": make_bars_with_final_volume(1800.0, seed=1)}  # confidence 0.9 on the final bar

    calls = []

    class SpyGate:
        def on_timestamp(self, timestamp):
            calls.append(timestamp)

        def check(self):
            return True

    gate = SpyGate()
    pm = PortfolioManager(max_slots=1, regime_gate=gate)
    monkeypatch.setattr("paper_trade.PortfolioManager", lambda **kwargs: pm)

    simulate(ticker_bars, VolumeSurgeConfidenceModel())

    assert len(calls) > 0
    assert calls == sorted(calls), "on_timestamp must be called in non-decreasing timestamp order"


def test_regime_gate_blocks_all_entries_when_check_returns_false(tmp_path, monkeypatch):
    _init_test_db(tmp_path, monkeypatch)
    # Would normally open a trade (final-bar confidence 0.9, well above the
    # 0.55 threshold) if not for the regime gate blocking every entry.
    ticker_bars = {"AAA.NS": make_bars_with_final_volume(1800.0, seed=1)}

    class AlwaysBlockGate:
        def on_timestamp(self, timestamp):
            pass

        def check(self):
            return False

    gate = AlwaysBlockGate()
    pm = PortfolioManager(max_slots=1, regime_gate=gate)
    monkeypatch.setattr("paper_trade.PortfolioManager", lambda **kwargs: pm)

    result = simulate(ticker_bars, VolumeSurgeConfidenceModel())

    assert result["total_trades"] == 0
