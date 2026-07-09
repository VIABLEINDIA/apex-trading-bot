import pandas as pd

from src.portfolio import PortfolioManager


def make_pm(**overrides):
    defaults = dict(total_capital=100_000, max_slots=10, capital_per_slot_pct=0.10, daily_loss_limit_pct=0.02)
    defaults.update(overrides)
    return PortfolioManager(**defaults)


def test_slot_capital_is_ten_percent_of_total():
    pm = make_pm()
    assert pm.slot_capital == 10_000


def test_signal_below_confidence_threshold_is_rejected():
    pm = make_pm()
    decision = pm.evaluate_signal("RELIANCE.NS", confidence=0.5, price=100)
    assert not decision.approved


def test_signal_above_threshold_with_open_slot_is_approved():
    pm = make_pm()
    decision = pm.evaluate_signal("RELIANCE.NS", confidence=0.70, price=100)
    assert decision.approved
    assert decision.quantity == 100  # 10_000 slot capital / 100 price


def test_duplicate_ticker_is_rejected_even_with_high_confidence():
    pm = make_pm()
    pm.open_trade("RELIANCE.NS", price=100, quantity=100, order_id="X1")
    decision = pm.evaluate_signal("RELIANCE.NS", confidence=0.90, price=105)
    assert not decision.approved
    assert "dedup" in decision.reason


def test_hard_slot_limit_enforced():
    pm = make_pm(max_slots=2)
    pm.open_trade("A.NS", price=100, quantity=100, order_id="X1")
    pm.open_trade("B.NS", price=100, quantity=100, order_id="X2")
    decision = pm.evaluate_signal("C.NS", confidence=0.90, price=100)
    assert not decision.approved
    assert "no free slot" in decision.reason


def test_close_trade_computes_pnl():
    pm = make_pm()
    pm.open_trade("RELIANCE.NS", price=100, quantity=100, order_id="X1")
    pnl = pm.close_trade("RELIANCE.NS", exit_price=110)
    assert pnl == 1000
    assert not pm.is_already_active("RELIANCE.NS")


def test_circuit_breaker_trips_on_realized_loss():
    pm = make_pm()  # 2% of 100_000 = 2_000 daily loss limit
    pm.open_trade("A.NS", price=100, quantity=100, order_id="X1")
    pm.close_trade("A.NS", exit_price=79)  # -2100 realized, breaches -2000
    assert pm.check_circuit_breaker() is True
    assert pm.circuit_breaker_tripped


def test_circuit_breaker_trips_on_unrealized_mark_to_market_loss():
    pm = make_pm()
    pm.open_trade("A.NS", price=100, quantity=100, order_id="X1")
    tripped = pm.check_circuit_breaker(current_prices={"A.NS": 79})  # -2100 unrealized
    assert tripped is True
    assert pm.circuit_breaker_tripped


def test_circuit_breaker_only_reports_newly_tripped_once():
    pm = make_pm()
    pm.open_trade("A.NS", price=100, quantity=100, order_id="X1")
    pm.close_trade("A.NS", exit_price=70)
    assert pm.check_circuit_breaker() is True
    assert pm.check_circuit_breaker() is False  # already tripped, no re-trigger


def test_circuit_breaker_blocks_new_entries_once_tripped():
    pm = make_pm()
    pm.open_trade("A.NS", price=100, quantity=100, order_id="X1")
    pm.close_trade("A.NS", exit_price=70)
    pm.check_circuit_breaker()
    decision = pm.evaluate_signal("B.NS", confidence=0.90, price=100)
    assert not decision.approved
    assert "circuit breaker" in decision.reason


def test_reset_day_clears_circuit_breaker_state():
    pm = make_pm()
    pm.open_trade("A.NS", price=100, quantity=100, order_id="X1")
    pm.close_trade("A.NS", exit_price=70)
    pm.check_circuit_breaker()
    assert pm.circuit_breaker_tripped

    pm.reset_day()
    assert not pm.circuit_breaker_tripped
    assert pm.realized_pnl_today == 0.0
    decision = pm.evaluate_signal("B.NS", confidence=0.90, price=100)
    assert decision.approved


def test_small_losses_do_not_trip_breaker():
    pm = make_pm()
    pm.open_trade("A.NS", price=100, quantity=100, order_id="X1")
    pm.close_trade("A.NS", exit_price=99)  # -100, well within -2000 limit
    assert pm.check_circuit_breaker() is False
    assert not pm.circuit_breaker_tripped


SECTOR_MAP = {"A.NS": "Banks", "B.NS": "Banks", "C.NS": "Banks", "D.NS": "IT"}


def test_sector_limit_rejects_once_sector_cap_reached():
    pm = make_pm(max_positions_per_sector=2, sector_map=SECTOR_MAP)
    pm.open_trade("A.NS", price=100, quantity=10, order_id="X1")
    pm.open_trade("B.NS", price=100, quantity=10, order_id="X2")
    decision = pm.evaluate_signal("C.NS", confidence=0.90, price=100)  # 3rd Banks name
    assert not decision.approved
    assert "sector limit" in decision.reason


def test_sector_limit_does_not_block_other_sectors():
    pm = make_pm(max_positions_per_sector=2, sector_map=SECTOR_MAP)
    pm.open_trade("A.NS", price=100, quantity=10, order_id="X1")
    pm.open_trade("B.NS", price=100, quantity=10, order_id="X2")
    decision = pm.evaluate_signal("D.NS", confidence=0.90, price=100)  # IT, not Banks
    assert decision.approved


def test_sector_limit_is_a_noop_without_a_sector_map():
    pm = make_pm(max_positions_per_sector=1)  # no sector_map given
    pm.open_trade("A.NS", price=100, quantity=10, order_id="X1")
    decision = pm.evaluate_signal("B.NS", confidence=0.90, price=100)
    assert decision.approved  # unknown sector for both -> limit never applies


def test_sector_position_count_only_counts_matching_sector():
    pm = make_pm(sector_map=SECTOR_MAP)
    pm.open_trade("A.NS", price=100, quantity=10, order_id="X1")
    pm.open_trade("D.NS", price=100, quantity=10, order_id="X2")
    assert pm.sector_position_count("Banks") == 1
    assert pm.sector_position_count("IT") == 1
    assert pm.sector_position_count("Pharma") == 0


def test_hourly_trade_budget_blocks_once_reached():
    pm = make_pm(max_trades_per_hour=2)
    t1 = pd.Timestamp("2024-01-01 09:20")
    t2 = pd.Timestamp("2024-01-01 09:45")
    t3 = pd.Timestamp("2024-01-01 09:50")  # same hour bucket as t1/t2
    pm.open_trade("A.NS", price=100, quantity=10, order_id="X1", timestamp=t1)
    pm.open_trade("B.NS", price=100, quantity=10, order_id="X2", timestamp=t2)
    decision = pm.evaluate_signal("C.NS", confidence=0.90, price=100, timestamp=t3)
    assert not decision.approved
    assert "hourly trade budget" in decision.reason


def test_hourly_trade_budget_resets_in_a_new_hour():
    pm = make_pm(max_trades_per_hour=1)
    t1 = pd.Timestamp("2024-01-01 09:20")
    t2 = pd.Timestamp("2024-01-01 10:05")  # next hour bucket
    pm.open_trade("A.NS", price=100, quantity=10, order_id="X1", timestamp=t1)
    decision = pm.evaluate_signal("B.NS", confidence=0.90, price=100, timestamp=t2)
    assert decision.approved


def test_hourly_trade_budget_is_a_noop_without_a_timestamp():
    pm = make_pm(max_trades_per_hour=1)
    pm.open_trade("A.NS", price=100, quantity=10, order_id="X1")  # no timestamp
    decision = pm.evaluate_signal("B.NS", confidence=0.90, price=100)  # no timestamp
    assert decision.approved  # hourly check skipped entirely


def test_reset_day_clears_hourly_trade_counts():
    pm = make_pm(max_trades_per_hour=1)
    t1 = pd.Timestamp("2024-01-01 09:20")
    pm.open_trade("A.NS", price=100, quantity=10, order_id="X1", timestamp=t1)
    pm.reset_day()
    # same hour-of-day, next calendar day -- should be a fresh bucket even
    # without reset_day(), but confirms reset_day() doesn't leave stale state
    t2 = pd.Timestamp("2024-01-02 09:20")
    decision = pm.evaluate_signal("B.NS", confidence=0.90, price=100, timestamp=t2)
    assert decision.approved


def test_capital_shield_full_size_when_flat_or_positive():
    pm = make_pm()
    assert pm.capital_shield_multiplier() == 1.0


def test_capital_shield_ramps_down_with_realized_drawdown():
    pm = make_pm(total_capital=100_000)
    pm.open_trade("A.NS", price=100, quantity=100, order_id="X1")
    pm.close_trade("A.NS", exit_price=99.4)  # -60, i.e. -0.06% dd -> still tier 1 (< 0.5%)
    assert pm.capital_shield_multiplier() == 1.00

    pm.open_trade("B.NS", price=100, quantity=100, order_id="X2")
    pm.close_trade("B.NS", exit_price=91.5)  # cumulative dd ~ -0.85% -> tier 2 (< 1.0%)
    assert pm.capital_shield_multiplier() == 0.75


def test_capital_shield_reduces_approved_quantity():
    pm = make_pm(total_capital=100_000, capital_per_slot_pct=0.10)  # slot_capital=10_000
    pm.open_trade("A.NS", price=100, quantity=100, order_id="X1")
    pm.close_trade("A.NS", exit_price=91.5)  # -850, ~0.85% dd -> 0.75x tier
    decision = pm.evaluate_signal("B.NS", confidence=0.90, price=100)
    assert decision.approved
    assert decision.quantity == 75  # 10_000 * 0.75 / 100, vs. 100 at full size


def test_capital_shield_uses_unrealized_pnl_when_current_prices_given():
    pm = make_pm(total_capital=100_000)
    pm.open_trade("A.NS", price=100, quantity=100, order_id="X1")
    # no realized loss yet, but marking to market shows an unrealized one
    multiplier = pm.capital_shield_multiplier(current_prices={"A.NS": 91.5})
    assert multiplier == 0.75


# --- ATR stop / trailing profit-lock ---

def test_open_trade_without_atr_has_no_stop():
    pm = make_pm()
    trade = pm.open_trade("A.NS", price=100, quantity=10, order_id="X1")  # no atr
    assert trade.stop_price is None
    assert pm.check_exits({"A.NS": 50}) == []  # no stop set -> never triggers, even at a huge loss


def test_atr_stop_set_at_entry():
    pm = make_pm(atr_stop_mult=0.75)
    trade = pm.open_trade("A.NS", price=100, quantity=10, order_id="X1", atr=4.0)
    assert trade.stop_price == 100 - 0.75 * 4.0  # 97.0


def test_atr_stop_triggers_before_trail_activates():
    pm = make_pm(atr_stop_mult=0.75, atr_trail_activation_mult=0.25, atr_trail_distance_mult=0.12)
    pm.open_trade("A.NS", price=100, quantity=10, order_id="X1", atr=4.0)  # stop=97.0
    triggered = pm.check_exits({"A.NS": 96.5})  # below stop, trail never activated
    assert triggered == [("A.NS", 96.5, "ATR stop")]


def test_trailing_stop_activates_and_ratchets_up():
    pm = make_pm(atr_stop_mult=0.75, atr_trail_activation_mult=0.25, atr_trail_distance_mult=0.12)
    pm.open_trade("A.NS", price=100, quantity=10, order_id="X1", atr=4.0)  # activation dist = 1.0

    # price rises enough to activate the trail (100 + 1.0 = 101)
    assert pm.check_exits({"A.NS": 101.0}) == []
    trade = pm.active_trades["A.NS"]
    assert trade.trail_activated
    assert trade.stop_price == 101.0 - 0.12 * 4.0  # 100.52

    # price rises further -- trail should ratchet up
    assert pm.check_exits({"A.NS": 105.0}) == []
    assert pm.active_trades["A.NS"].stop_price == 105.0 - 0.12 * 4.0  # 104.52

    # price pulls back but not below the ratcheted stop -- stays open
    assert pm.check_exits({"A.NS": 104.6}) == []
    assert pm.active_trades["A.NS"].stop_price == 104.52  # unchanged, not ratcheted down

    # price drops through the ratcheted stop -- triggers
    triggered = pm.check_exits({"A.NS": 104.0})
    assert triggered == [("A.NS", 104.0, "trailing stop")]


def test_check_exits_skips_tickers_without_a_current_price():
    pm = make_pm()
    pm.open_trade("A.NS", price=100, quantity=10, order_id="X1", atr=4.0)
    assert pm.check_exits({}) == []  # no price for A.NS -- can't evaluate, not a false trigger


# --- Trading-window session gating ---

def test_primary_session_uses_normal_threshold_and_full_size():
    pm = make_pm(total_capital=100_000, capital_per_slot_pct=0.10,
                 primary_session_end="10:15", continuation_session_end="13:15",
                 continuation_confidence_bonus=0.05, continuation_size_mult=0.4)
    ts = pd.Timestamp("2024-01-01 09:30")
    decision = pm.evaluate_signal("A.NS", confidence=0.60, price=100, timestamp=ts)
    assert decision.approved
    assert decision.quantity == 100  # full slot size, no session reduction


def test_continuation_session_requires_higher_confidence():
    pm = make_pm(primary_session_end="10:15", continuation_session_end="13:15",
                 continuation_confidence_bonus=0.05)
    ts = pd.Timestamp("2024-01-01 11:00")
    # 0.58 clears the plain 0.55 threshold but not 0.55+0.05=0.60 in continuation
    decision = pm.evaluate_signal("A.NS", confidence=0.58, price=100, timestamp=ts)
    assert not decision.approved
    assert "threshold" in decision.reason


def test_continuation_session_reduces_size():
    pm = make_pm(total_capital=100_000, capital_per_slot_pct=0.10,
                 primary_session_end="10:15", continuation_session_end="13:15",
                 continuation_confidence_bonus=0.05, continuation_size_mult=0.4)
    ts = pd.Timestamp("2024-01-01 11:00")
    decision = pm.evaluate_signal("A.NS", confidence=0.90, price=100, timestamp=ts)
    assert decision.approved
    assert decision.quantity == 40  # 10_000 * 0.4 / 100


def test_no_new_entries_after_continuation_session():
    pm = make_pm(primary_session_end="10:15", continuation_session_end="13:15")
    ts = pd.Timestamp("2024-01-01 14:00")
    decision = pm.evaluate_signal("A.NS", confidence=0.95, price=100, timestamp=ts)
    assert not decision.approved
    assert "outside trading window" in decision.reason


def test_session_gating_is_a_noop_without_a_timestamp():
    pm = make_pm(continuation_session_end="13:15")
    decision = pm.evaluate_signal("A.NS", confidence=0.90, price=100)  # no timestamp
    assert decision.approved  # would be "outside window" at any real 14:00+ timestamp


# --- Risk-based position sizing ---

def test_risk_based_sizing_can_be_smaller_than_flat_slot_size():
    pm = make_pm(total_capital=100_000, capital_per_slot_pct=0.10, risk_per_trade_pct=0.013,
                 atr_stop_mult=0.75)
    # stop distance = 0.75 * 4.0 = 3.0; risk budget = 100_000 * 0.013 = 1300
    # risk_quantity = floor(1300 / 3.0) = 433; flat_quantity = floor(10_000/100) = 100
    # flat is the tighter constraint here (low ATR relative to price)
    decision = pm.evaluate_signal("A.NS", confidence=0.90, price=100, atr=4.0)
    assert decision.approved
    assert decision.quantity == 100  # capped by flat slot size, not risk size


def test_risk_based_sizing_caps_below_flat_size_for_a_volatile_stock():
    pm = make_pm(total_capital=100_000, capital_per_slot_pct=0.10, risk_per_trade_pct=0.013,
                 atr_stop_mult=0.75)
    # a much more volatile stock: ATR=50 -> stop distance = 37.5
    # risk_quantity = floor(1300 / 37.5) = 34; flat_quantity = floor(10_000/100) = 100
    decision = pm.evaluate_signal("A.NS", confidence=0.90, price=100, atr=50.0)
    assert decision.approved
    assert decision.quantity == 34


def test_sizing_without_atr_falls_back_to_flat_slot_sizing():
    pm = make_pm(total_capital=100_000, capital_per_slot_pct=0.10)
    decision = pm.evaluate_signal("A.NS", confidence=0.90, price=100)  # no atr
    assert decision.approved
    assert decision.quantity == 100


# --- Consecutive-loss halt ---

def test_consecutive_losses_halt_after_limit_reached():
    pm = make_pm(max_consecutive_losses=3)
    for i in range(3):
        pm.open_trade(f"T{i}.NS", price=100, quantity=10, order_id=f"X{i}")
        pm.close_trade(f"T{i}.NS", exit_price=90)  # loss
    decision = pm.evaluate_signal("NEW.NS", confidence=0.90, price=100)
    assert not decision.approved
    assert "consecutive loss limit" in decision.reason


def test_a_win_resets_the_consecutive_loss_counter():
    pm = make_pm(max_consecutive_losses=3)
    pm.open_trade("A.NS", price=100, quantity=10, order_id="X1")
    pm.close_trade("A.NS", exit_price=90)  # loss 1
    pm.open_trade("B.NS", price=100, quantity=10, order_id="X2")
    pm.close_trade("B.NS", exit_price=110)  # win -- resets counter
    assert pm.consecutive_losses == 0
    decision = pm.evaluate_signal("C.NS", confidence=0.90, price=100)
    assert decision.approved


def test_reset_day_clears_consecutive_losses():
    pm = make_pm(max_consecutive_losses=1)
    pm.open_trade("A.NS", price=100, quantity=10, order_id="X1")
    pm.close_trade("A.NS", exit_price=90)  # loss, hits the limit of 1
    assert not pm.evaluate_signal("B.NS", confidence=0.90, price=100).approved

    pm.reset_day()
    assert pm.consecutive_losses == 0
    assert pm.evaluate_signal("B.NS", confidence=0.90, price=100).approved


# --- Kill switch file ---

def test_kill_switch_blocks_new_entries_when_file_exists(tmp_path):
    kill_switch = tmp_path / "KILL_SWITCH"
    kill_switch.write_text("stop")
    pm = make_pm(kill_switch_path=str(kill_switch))
    decision = pm.evaluate_signal("A.NS", confidence=0.90, price=100)
    assert not decision.approved
    assert "kill switch" in decision.reason


def test_kill_switch_allows_trading_when_file_absent(tmp_path):
    kill_switch = tmp_path / "KILL_SWITCH"  # never created
    pm = make_pm(kill_switch_path=str(kill_switch))
    decision = pm.evaluate_signal("A.NS", confidence=0.90, price=100)
    assert decision.approved


def test_kill_switch_resumes_trading_once_file_is_removed(tmp_path):
    kill_switch = tmp_path / "KILL_SWITCH"
    kill_switch.write_text("stop")
    pm = make_pm(kill_switch_path=str(kill_switch))
    assert not pm.evaluate_signal("A.NS", confidence=0.90, price=100).approved

    kill_switch.unlink()
    assert pm.evaluate_signal("A.NS", confidence=0.90, price=100).approved


def test_no_regime_gate_is_a_full_noop_by_default():
    pm = make_pm()
    assert pm.evaluate_signal("A.NS", confidence=0.90, price=100).approved


def test_regime_gate_blocks_new_entries_when_it_returns_false():
    pm = make_pm(regime_gate=lambda: False)
    decision = pm.evaluate_signal("A.NS", confidence=0.90, price=100)
    assert not decision.approved
    assert "regime" in decision.reason


def test_regime_gate_allows_trading_when_it_returns_true():
    pm = make_pm(regime_gate=lambda: True)
    assert pm.evaluate_signal("A.NS", confidence=0.90, price=100).approved


def test_regime_gate_is_reevaluated_on_every_call():
    state = {"ok": False}
    pm = make_pm(regime_gate=lambda: state["ok"])
    assert not pm.evaluate_signal("A.NS", confidence=0.90, price=100).approved

    state["ok"] = True
    assert pm.evaluate_signal("A.NS", confidence=0.90, price=100).approved


def test_regime_gate_object_with_check_method_is_used_over_bare_call():
    class Gate:
        def check(self):
            return False

    pm = make_pm(regime_gate=Gate())
    assert not pm.evaluate_signal("A.NS", confidence=0.90, price=100).approved


def test_on_timestamp_is_a_noop_without_a_regime_gate():
    pm = make_pm()
    pm.on_timestamp(pd.Timestamp("2024-01-01 09:15"))  # must not raise


def test_on_timestamp_forwards_to_regime_gate():
    calls = []

    class FakeGate:
        def on_timestamp(self, timestamp):
            calls.append(timestamp)

    pm = make_pm(regime_gate=FakeGate())
    ts = pd.Timestamp("2024-01-01 09:15")
    pm.on_timestamp(ts)
    assert calls == [ts]
