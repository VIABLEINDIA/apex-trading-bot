"""Unit tests for src/execution.py. neo_api_client is never imported here --
KotakNeoClient's real __init__ is bypassed (via __new__) and a mock stands in
for self.client, since the whole point of the lazy import in __init__ is that
this module (and its tests) don't need the package or real credentials."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.execution import (
    FillReport,
    KotakNeoClient,
    OrderResult,
    TickBarAggregator,
    reconcile_fill,
    to_kotak_trading_symbol,
)


def make_client(mock_sdk_client=None) -> KotakNeoClient:
    """Build a KotakNeoClient without running __init__ (which lazily imports
    neo_api_client and would need real credentials to construct NeoAPI)."""
    client = KotakNeoClient.__new__(KotakNeoClient)
    client.client = mock_sdk_client if mock_sdk_client is not None else MagicMock()
    client._logged_in = False
    return client


def test_to_kotak_trading_symbol_strips_ns_suffix_and_appends_eq():
    assert to_kotak_trading_symbol("RELIANCE.NS") == "RELIANCE-EQ"


def test_to_kotak_trading_symbol_is_noop_without_ns_suffix():
    assert to_kotak_trading_symbol("RELIANCE") == "RELIANCE-EQ"


def test_login_unwraps_data_wrapped_response(monkeypatch):
    # settings.kotak is a frozen dataclass (can't monkeypatch its fields
    # directly), so stub pyotp.TOTP instead -- login() only needs *a* TOTP
    # code, not a genuine one, regardless of whatever secret is in .env.
    monkeypatch.setattr("src.execution.pyotp.TOTP", lambda secret: MagicMock(now=lambda: "123456"))

    sdk = MagicMock()
    sdk.totp_login.return_value = {"data": {"sid": "session123"}}
    sdk.totp_validate.return_value = {"data": {"token": "tok456"}}
    client = make_client(sdk)

    client.login()

    assert client._logged_in is True
    sdk.totp_login.assert_called_once()
    sdk.totp_validate.assert_called_once()


def test_login_handles_flat_response_without_data_wrapper(monkeypatch):
    monkeypatch.setattr("src.execution.pyotp.TOTP", lambda secret: MagicMock(now=lambda: "123456"))

    sdk = MagicMock()
    sdk.totp_login.return_value = {"sid": "session123"}
    sdk.totp_validate.return_value = {"token": "tok456"}
    client = make_client(sdk)

    client.login()

    assert client._logged_in is True


def test_subscribe_ticks_requires_login_first():
    client = make_client()
    with pytest.raises(RuntimeError):
        client.subscribe_ticks(["RELIANCE.NS"], on_tick=lambda msg: None)


def test_subscribe_ticks_calls_sdk_subscribe_after_login():
    sdk = MagicMock()
    client = make_client(sdk)
    client._logged_in = True

    client.subscribe_ticks(["RELIANCE.NS", "TCS.NS"], on_tick=lambda msg: None, is_index=False)

    sdk.subscribe.assert_called_once_with(
        instrument_tokens=["RELIANCE.NS", "TCS.NS"], isIndex=False, isDepth=False,
    )


def test_place_order_is_paper_by_default(monkeypatch):
    # Settings is a frozen dataclass singleton -- swap the module-level name
    # execution.py reads instead of mutating the real (frozen) instance.
    monkeypatch.setattr("src.execution.settings", SimpleNamespace(live_trading=False))

    client = make_client()
    result = client.place_order("RELIANCE.NS", 10, transaction_type="BUY")

    assert result.paper is True
    assert result.order_id.startswith("PAPER-")
    assert result.status == "simulated"
    client.client.place_order.assert_not_called()


def test_place_order_routes_to_broker_when_live_trading_enabled(monkeypatch):
    monkeypatch.setattr("src.execution.settings", SimpleNamespace(live_trading=True))

    sdk = MagicMock()
    sdk.place_order.return_value = {"nOrdNo": "ORD789", "stat": "Ok"}
    client = make_client(sdk)

    result = client.place_order("RELIANCE.NS", 10, transaction_type="SELL")

    assert result.paper is False
    assert result.order_id == "ORD789"
    kwargs = sdk.place_order.call_args.kwargs
    assert kwargs["transaction_type"] == "S"
    assert kwargs["trading_symbol"] == "RELIANCE-EQ"
    assert kwargs["quantity"] == "10"


def test_get_order_status_returns_last_entry_of_history_list():
    sdk = MagicMock()
    sdk.order_history.return_value = {
        "data": [
            {"ordSt": "open", "fldQty": "0"},
            {"ordSt": "complete", "fldQty": "10", "avgPrc": "101.5"},
        ]
    }
    client = make_client(sdk)

    status = client.get_order_status("ORD789")

    assert status == {"ordSt": "complete", "fldQty": "10", "avgPrc": "101.5"}


def test_get_order_status_handles_flat_dict_response():
    sdk = MagicMock()
    sdk.order_history.return_value = {"ordSt": "complete", "fldQty": "10"}
    client = make_client(sdk)

    status = client.get_order_status("ORD789")

    assert status["ordSt"] == "complete"


def test_get_order_status_handles_empty_history():
    sdk = MagicMock()
    sdk.order_history.return_value = {"data": []}
    client = make_client(sdk)

    assert client.get_order_status("ORD789") == {}


def test_reconcile_fill_is_always_a_match_for_paper_orders():
    client = make_client()
    fill = reconcile_fill(client, "PAPER-abc123", expected_quantity=10)

    assert fill.matches_expected is True
    assert fill.filled_quantity == 10
    assert fill.avg_price is None
    client.client.order_history.assert_not_called()


def test_reconcile_fill_matches_when_broker_confirms_full_fill():
    sdk = MagicMock()
    sdk.order_history.return_value = {"data": [{"ordSt": "complete", "fldQty": "10", "avgPrc": "101.5"}]}
    client = make_client(sdk)

    fill = reconcile_fill(client, "ORD789", expected_quantity=10, max_attempts=1)

    assert fill.matches_expected is True
    assert fill.filled_quantity == 10
    assert fill.avg_price == 101.5
    assert fill.order_status == "complete"


def test_reconcile_fill_flags_partial_fill_as_mismatch():
    sdk = MagicMock()
    sdk.order_history.return_value = {"data": [{"ordSt": "complete", "fldQty": "4", "avgPrc": "101.5"}]}
    client = make_client(sdk)

    fill = reconcile_fill(client, "ORD789", expected_quantity=10, max_attempts=1)

    assert fill.matches_expected is False
    assert fill.filled_quantity == 4


def test_reconcile_fill_flags_rejected_order_as_zero_fill():
    sdk = MagicMock()
    sdk.order_history.return_value = {"data": [{"ordSt": "rejected", "fldQty": "0"}]}
    client = make_client(sdk)

    fill = reconcile_fill(client, "ORD789", expected_quantity=10, max_attempts=1)

    assert fill.matches_expected is False
    assert fill.filled_quantity == 0
    assert fill.avg_price is None


def test_reconcile_fill_retries_until_a_fill_quantity_appears(monkeypatch):
    sdk = MagicMock()
    sdk.order_history.side_effect = [
        {"data": [{"ordSt": "open", "fldQty": "0"}]},
        {"data": [{"ordSt": "open", "fldQty": "0"}]},
        {"data": [{"ordSt": "complete", "fldQty": "10", "avgPrc": "100"}]},
    ]
    client = make_client(sdk)
    monkeypatch.setattr("src.execution.time.sleep", lambda _seconds: None)

    fill = reconcile_fill(client, "ORD789", expected_quantity=10, max_attempts=3, poll_interval_seconds=0)

    assert fill.matches_expected is True
    assert sdk.order_history.call_count == 3


def test_tick_bar_aggregator_resamples_ticks_into_ohlcv_bars():
    import pandas as pd

    agg = TickBarAggregator(bar_interval="15min")
    base = pd.Timestamp("2026-01-05 09:15:00", tz="Asia/Kolkata")
    agg.add_tick("RELIANCE.NS", price=100, volume=10, timestamp=base)
    agg.add_tick("RELIANCE.NS", price=102, volume=5, timestamp=base + pd.Timedelta(minutes=5))
    agg.add_tick("RELIANCE.NS", price=99, volume=8, timestamp=base + pd.Timedelta(minutes=10))

    bars = agg.get_bars("RELIANCE.NS")

    assert len(bars) == 1
    row = bars.iloc[0]
    assert row["Open"] == 100
    assert row["High"] == 102
    assert row["Low"] == 99
    assert row["Close"] == 99
    assert row["Volume"] == 23


def test_tick_bar_aggregator_returns_empty_frame_for_unknown_ticker():
    agg = TickBarAggregator()
    bars = agg.get_bars("NOPE.NS")
    assert bars.empty
    assert list(bars.columns) == ["Open", "High", "Low", "Close", "Volume"]
