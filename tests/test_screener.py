from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import src.screener as screener


def multiindex_daily_closes(ticker_closes: dict[str, list[float]]) -> pd.DataFrame:
    """Build a fake yf.download(group_by="ticker") response: MultiIndex
    columns (ticker, field), one row per close value given. Tickers with
    fewer rows than the longest series are padded with NaN (not a repeated
    value), so a shorter series genuinely reads as "less history" after
    dropna() -- matching what a real partial-data response looks like."""
    n = max(len(v) for v in ticker_closes.values())
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    frames = {}
    for ticker, closes in ticker_closes.items():
        padded = closes + [float("nan")] * (n - len(closes))
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            frames[(ticker, field)] = padded if field == "Close" else [0] * n
    return pd.DataFrame(frames, index=idx, columns=pd.MultiIndex.from_tuples(frames.keys()))


def test_compute_momentum_scores_ranks_by_roc(monkeypatch):
    # RELIANCE: 100 -> 110 (+10%), TCS: 100 -> 95 (-5%)
    closes = {
        "RELIANCE.NS": [100.0] * 30 + [110.0],
        "TCS.NS": [100.0] * 30 + [95.0],
    }
    monkeypatch.setattr(screener.yf, "download", MagicMock(return_value=multiindex_daily_closes(closes)))

    result = screener.compute_momentum_scores(["RELIANCE.NS", "TCS.NS"], lookback_days=30)

    assert list(result["ticker"]) == ["RELIANCE.NS", "TCS.NS"]
    assert result.iloc[0]["momentum_score"] == pytest.approx(0.10)
    assert result.iloc[1]["momentum_score"] == pytest.approx(-0.05)


def test_compute_momentum_scores_skips_ticker_with_insufficient_history(monkeypatch):
    closes = {
        "RELIANCE.NS": [100.0] * 30 + [110.0],
        "TCS.NS": [100.0] * 5,  # fewer than lookback_days + 1 rows
    }
    monkeypatch.setattr(screener.yf, "download", MagicMock(return_value=multiindex_daily_closes(closes)))

    result = screener.compute_momentum_scores(["RELIANCE.NS", "TCS.NS"], lookback_days=30)

    assert list(result["ticker"]) == ["RELIANCE.NS"]


def test_compute_momentum_scores_skips_ticker_missing_from_response(monkeypatch):
    closes = {"RELIANCE.NS": [100.0] * 30 + [110.0]}
    monkeypatch.setattr(screener.yf, "download", MagicMock(return_value=multiindex_daily_closes(closes)))

    result = screener.compute_momentum_scores(["RELIANCE.NS", "TCS.NS"], lookback_days=30)

    assert list(result["ticker"]) == ["RELIANCE.NS"]


def test_compute_momentum_scores_returns_empty_frame_when_nothing_usable(monkeypatch):
    monkeypatch.setattr(screener.yf, "download", MagicMock(return_value=pd.DataFrame()))

    result = screener.compute_momentum_scores(["RELIANCE.NS"], lookback_days=30)

    assert result.empty


def test_run_screener_selects_top_n_from_scored_universe(monkeypatch):
    monkeypatch.setattr(screener, "load_universe", lambda: ["A.NS", "B.NS", "C.NS"])
    scored = pd.DataFrame({
        "ticker": ["A.NS", "B.NS", "C.NS"],
        "close": [100, 200, 300],
        "momentum_score": [0.3, 0.2, 0.1],
    })
    monkeypatch.setattr(screener, "compute_momentum_scores", lambda tickers, lookback_days: scored)
    # settings.screener is a frozen dataclass -- swap the module-level name
    # screener.py reads instead of mutating the real (frozen) instance.
    monkeypatch.setattr(screener, "settings", SimpleNamespace(screener=SimpleNamespace(top_n=2, roc_lookback_days=30)))

    result = screener.run_screener()

    assert list(result["ticker"]) == ["A.NS", "B.NS"]
