"""Realistic transaction cost modeling for NSE intraday (MIS) equity trades.

The backtests so far assumed frictionless fills at the exact simulated bar
close, with no charges at all. Kotak Neo's MIS product is zero-brokerage
(per the blueprint), but statutory charges and execution slippage still eat
into a strategy whose average edge is only ~0.15-0.2% of position size per
trade (see README "Model performance") -- worth modeling explicitly rather
than assuming they wash out.

Rates below are the standard NSE intraday-equity structure as of this
writing; STT applies only to the sell leg, stamp duty only to the buy leg.

Slippage is time-of-day dependent rather than a flat assumption: NSE cash
equity spreads and market impact are visibly wider in the first/last 15-30
minutes (opening/closing auction volatility) than during the liquid midday
session. The specific bps-by-window figures are a reference table (no live
order-book data of our own to calibrate against), not a measured constant.
"""
from dataclasses import dataclass
from datetime import time


@dataclass(frozen=True)
class TransactionCosts:
    brokerage_pct: float = 0.0            # Kotak Neo Zero-Brokerage MIS
    stt_sell_pct: float = 0.00025          # 0.025% on sell turnover only (intraday equity)
    exchange_txn_pct: float = 0.0000325    # ~0.00325% NSE transaction charge, each leg
    sebi_fee_pct: float = 0.000001         # ~Rs 10/crore, each leg
    stamp_duty_buy_pct: float = 0.00003    # 0.003% on buy turnover only
    gst_pct: float = 0.18                  # on (brokerage + exchange charges), each leg
    slippage_bps_per_leg: float = 5.0      # flat fallback when no timestamp is given


DEFAULT_COSTS = TransactionCosts()

# (window start, window end, slippage in bps) -- NSE cash equity, approximate.
# Widest around the open/close auctions, tightest in the liquid midday session.
TIME_OF_DAY_SLIPPAGE_BPS = [
    (time(9, 15), time(9, 30), 15.0),
    (time(9, 30), time(10, 0), 8.0),
    (time(10, 0), time(13, 0), 4.0),
    (time(13, 0), time(14, 30), 5.0),
    (time(14, 30), time(15, 15), 8.0),
    (time(15, 15), time(15, 30), 12.0),
]


def slippage_bps_for_time(t: time, costs: TransactionCosts = DEFAULT_COSTS) -> float:
    """Look up the time-of-day slippage bucket for `t`. Falls back to the flat
    `costs.slippage_bps_per_leg` for timestamps outside the NSE trading window
    (shouldn't happen for real market data, but keeps this total rather than
    raising on unexpected input)."""
    for start, end, bps in TIME_OF_DAY_SLIPPAGE_BPS:
        if start <= t < end:
            return bps
    return costs.slippage_bps_per_leg


def apply_slippage(price: float, side: str, ts=None, costs: TransactionCosts = DEFAULT_COSTS) -> float:
    """A buy fills a touch higher than the quoted/bar-close price; a sell
    fills a touch lower -- slippage always works against the trader.

    `ts` (a datetime/pandas Timestamp, or None) selects the time-of-day
    slippage bucket; omit it to use the flat `slippage_bps_per_leg` fallback.
    """
    bps = slippage_bps_for_time(ts.time(), costs) if ts is not None else costs.slippage_bps_per_leg
    slip = price * (bps / 10_000)
    return price + slip if side == "buy" else price - slip


def round_trip_charges(entry_price: float, exit_price: float, quantity: int,
                        costs: TransactionCosts = DEFAULT_COSTS) -> float:
    """Statutory charges for one buy + one sell leg (slippage is applied to
    the fill price directly, not included here)."""
    buy_turnover = entry_price * quantity
    sell_turnover = exit_price * quantity

    exchange_txn = (buy_turnover + sell_turnover) * costs.exchange_txn_pct
    sebi_fee = (buy_turnover + sell_turnover) * costs.sebi_fee_pct
    stt = sell_turnover * costs.stt_sell_pct
    stamp_duty = buy_turnover * costs.stamp_duty_buy_pct
    gst = (costs.brokerage_pct * (buy_turnover + sell_turnover) + exchange_txn) * costs.gst_pct

    return exchange_txn + sebi_fee + stt + stamp_duty + gst


def net_pnl(entry_price: float, exit_price: float, quantity: int,
            entry_ts=None, exit_ts=None, costs: TransactionCosts = DEFAULT_COSTS) -> float:
    """Realistic PnL: slippage-adjusted fills, minus statutory charges. This
    is what a trade would have actually cleared, vs. the frictionless
    `(exit_price - entry_price) * quantity` used elsewhere for gross PnL.

    `entry_ts`/`exit_ts` (optional) select time-of-day slippage buckets for
    each leg independently -- a trade opened at 09:20 and closed at 15:25
    pays the wider open/close spreads on both ends, not the midday rate.
    """
    filled_entry = apply_slippage(entry_price, "buy", entry_ts, costs)
    filled_exit = apply_slippage(exit_price, "sell", exit_ts, costs)
    gross = (filled_exit - filled_entry) * quantity
    charges = round_trip_charges(filled_entry, filled_exit, quantity, costs)
    return gross - charges
