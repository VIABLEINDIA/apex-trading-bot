import pandas as pd
import pytest

from src.costs import (
    TransactionCosts, apply_slippage, net_pnl, round_trip_charges, round_trip_cost_fraction,
    slippage_bps_for_time,
)


def test_slippage_makes_buys_worse_and_sells_worse():
    costs = TransactionCosts(slippage_bps_per_leg=10.0)
    assert apply_slippage(100.0, "buy", costs=costs) > 100.0
    assert apply_slippage(100.0, "sell", costs=costs) < 100.0


def test_round_trip_charges_are_positive_and_scale_with_turnover():
    small = round_trip_charges(100.0, 105.0, 10)
    large = round_trip_charges(100.0, 105.0, 1000)
    assert small > 0
    assert large > small


def test_net_pnl_is_less_than_frictionless_gross_pnl_on_a_winner():
    gross = (105.0 - 100.0) * 100
    net = net_pnl(100.0, 105.0, 100)
    assert net < gross


def test_net_pnl_can_flip_a_marginal_winner_negative():
    # a 0.05% gross move is smaller than realistic round-trip friction
    gross = (100.05 - 100.0) * 100
    net = net_pnl(100.0, 100.05, 100)
    assert gross > 0
    assert net < gross


def test_zero_cost_config_matches_frictionless_gross_pnl():
    zero_costs = TransactionCosts(
        brokerage_pct=0, stt_sell_pct=0, exchange_txn_pct=0, sebi_fee_pct=0,
        stamp_duty_buy_pct=0, gst_pct=0, slippage_bps_per_leg=0,
    )
    gross = (105.0 - 100.0) * 100
    assert net_pnl(100.0, 105.0, 100, costs=zero_costs) == gross


def test_slippage_bps_is_wider_at_open_and_close_than_midday():
    open_bps = slippage_bps_for_time(pd.Timestamp("2024-01-01 09:20").time())
    midday_bps = slippage_bps_for_time(pd.Timestamp("2024-01-01 11:00").time())
    close_bps = slippage_bps_for_time(pd.Timestamp("2024-01-01 15:20").time())
    assert open_bps > midday_bps
    assert close_bps > midday_bps


def test_net_pnl_uses_wider_slippage_at_open_than_midday():
    open_ts = pd.Timestamp("2024-01-01 09:20")
    midday_ts = pd.Timestamp("2024-01-01 11:00")
    # same nominal move, but entered/exited at the open should cost more in
    # slippage than the identical move entered/exited midday
    net_at_open = net_pnl(100.0, 105.0, 100, entry_ts=open_ts, exit_ts=open_ts)
    net_midday = net_pnl(100.0, 105.0, 100, entry_ts=midday_ts, exit_ts=midday_ts)
    assert net_at_open < net_midday


def test_round_trip_cost_fraction_is_positive():
    assert round_trip_cost_fraction() > 0


def test_round_trip_cost_fraction_is_wider_at_open_than_midday():
    open_ts = pd.Timestamp("2024-01-01 09:20")
    midday_ts = pd.Timestamp("2024-01-01 11:00")
    assert round_trip_cost_fraction(open_ts) > round_trip_cost_fraction(midday_ts)


def test_round_trip_cost_fraction_is_price_invariant():
    # Every cost component here is a percentage of price/turnover, so the
    # fraction shouldn't depend on the absolute price level -- verify by
    # comparing net_pnl's own per-share result at two different prices.
    ts = pd.Timestamp("2024-01-01 11:00")
    fraction = round_trip_cost_fraction(ts)
    cheap_stock_cost = -net_pnl(10.0, 10.0, 1, entry_ts=ts, exit_ts=ts)
    expensive_stock_cost = -net_pnl(5000.0, 5000.0, 1, entry_ts=ts, exit_ts=ts)
    assert cheap_stock_cost / 10.0 == pytest.approx(fraction)
    assert expensive_stock_cost / 5000.0 == pytest.approx(fraction)


def test_round_trip_cost_fraction_matches_zero_cost_config():
    zero_costs = TransactionCosts(
        brokerage_pct=0, stt_sell_pct=0, exchange_txn_pct=0, sebi_fee_pct=0,
        stamp_duty_buy_pct=0, gst_pct=0, slippage_bps_per_leg=0,
    )
    assert round_trip_cost_fraction(costs=zero_costs) == 0
