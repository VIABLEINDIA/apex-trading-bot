from unittest.mock import MagicMock

import pandas as pd
import pytest

import src.market_data as market_data


def multiindex_ohlcv(tickers: list[str], rows: int = 3) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
    data = {(t, f): range(rows) for t in tickers for f in ["Open", "High", "Low", "Close", "Volume"]}
    return pd.DataFrame(data, columns=columns)


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    monkeypatch.setattr(market_data.time, "sleep", lambda _seconds: None)


def test_fetch_bars_by_ticker_happy_path(monkeypatch):
    tickers = ["RELIANCE.NS", "TCS.NS"]
    monkeypatch.setattr(market_data.yf, "download", MagicMock(return_value=multiindex_ohlcv(tickers)))

    result = market_data.fetch_bars_by_ticker(tickers, days=5, interval="15m")

    assert set(result.keys()) == set(tickers)
    assert list(result["RELIANCE.NS"].columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_fetch_bars_by_ticker_skips_ticker_missing_from_response(monkeypatch):
    # yf.download only returned data for one of the two requested tickers.
    monkeypatch.setattr(market_data.yf, "download", MagicMock(return_value=multiindex_ohlcv(["RELIANCE.NS"])))

    result = market_data.fetch_bars_by_ticker(["RELIANCE.NS", "TCS.NS"], days=5)

    assert set(result.keys()) == {"RELIANCE.NS"}


def test_fetch_bars_by_ticker_retries_after_a_failed_attempt(monkeypatch):
    calls = {"n": 0}

    def flaky_download(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("simulated network failure")
        return multiindex_ohlcv(["RELIANCE.NS"])

    monkeypatch.setattr(market_data.yf, "download", flaky_download)

    result = market_data.fetch_bars_by_ticker(["RELIANCE.NS"], days=5)

    assert calls["n"] == 2
    assert "RELIANCE.NS" in result


def test_fetch_bars_by_ticker_raises_when_all_tickers_fail(monkeypatch):
    monkeypatch.setattr(market_data.yf, "download", MagicMock(return_value=pd.DataFrame()))

    with pytest.raises(RuntimeError, match="No data fetched"):
        market_data.fetch_bars_by_ticker(["RELIANCE.NS"], days=5)


def test_fetch_benchmark_close_returns_close_series(monkeypatch):
    monkeypatch.setattr(
        market_data.yf, "download",
        MagicMock(return_value=multiindex_ohlcv([market_data.BENCHMARK_TICKER])),
    )

    close = market_data.fetch_benchmark_close(days=5)

    assert isinstance(close, pd.Series)
    assert len(close) == 3
