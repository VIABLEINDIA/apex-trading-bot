import numpy as np
import pandas as pd

from src.regime import RealizedVolatilityGate, realized_volatility_regime_ok


def make_benchmark(n=60, daily_return_std=0.001, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="15min")
    returns = rng.normal(0, daily_return_std, n)
    close = 20000 * (1 + returns).cumprod()
    return pd.Series(close, index=idx)


def test_returns_true_with_insufficient_history():
    short_benchmark = make_benchmark(n=5)
    assert realized_volatility_regime_ok(short_benchmark, lookback=20) is True


def test_fails_open_when_lookback_window_is_too_small_for_a_defined_std():
    # lookback=1 means the trailing window is a single return; sample std
    # (ddof=1, pandas' default) is undefined (NaN) for n=1. That must still
    # fail open ("not enough history to judge volatility"), not fail closed.
    benchmark = make_benchmark(n=10, daily_return_std=0.0002)
    assert realized_volatility_regime_ok(benchmark, lookback=1, max_annualized_vol=0.30) is True


def test_low_volatility_regime_is_ok():
    calm = make_benchmark(n=60, daily_return_std=0.0002)
    assert realized_volatility_regime_ok(calm, lookback=20, max_annualized_vol=0.30) is True


def test_high_volatility_regime_is_blocked():
    volatile = make_benchmark(n=60, daily_return_std=0.02)
    assert realized_volatility_regime_ok(volatile, lookback=20, max_annualized_vol=0.30) is False


def test_only_looks_at_the_trailing_lookback_window():
    # A volatile start followed by a long calm stretch -- once enough calm
    # bars accumulate to fill the lookback window, the earlier volatility
    # should no longer affect the decision.
    idx = pd.date_range("2024-01-01 09:15", periods=100, freq="15min")
    rng = np.random.default_rng(2)
    volatile_returns = rng.normal(0, 0.05, 30)
    calm_returns = rng.normal(0, 0.0002, 70)
    returns = np.concatenate([volatile_returns, calm_returns])
    close = pd.Series(20000 * (1 + returns).cumprod(), index=idx)

    assert realized_volatility_regime_ok(close, lookback=20, max_annualized_vol=0.30) is True


def test_threshold_is_configurable():
    moderate = make_benchmark(n=60, daily_return_std=0.005)
    assert realized_volatility_regime_ok(moderate, lookback=20, max_annualized_vol=0.01) is False
    assert realized_volatility_regime_ok(moderate, lookback=20, max_annualized_vol=5.0) is True


def test_routine_overnight_gap_does_not_spuriously_trigger_high_volatility():
    # A calm session (Monday), a routine (non-extreme) +2% overnight gap into
    # the next day's open, then another calm session (Tuesday). The lookback
    # window at Tuesday's open is entirely within the current day (one bar),
    # so it should fail open (insufficient same-day history) rather than be
    # dominated by the overnight gap return -- returns must be computed
    # within each calendar day only, never across the day boundary.
    mon = pd.date_range("2026-01-05 09:15", periods=25, freq="15min")
    tue = pd.date_range("2026-01-06 09:15", periods=25, freq="15min")
    idx = mon.append(tue)

    rng = np.random.default_rng(0)
    mon_close = 22000 * (1 + rng.normal(0, 0.0005, 25)).cumprod()
    overnight_gap = 1.02
    tue_close = mon_close[-1] * overnight_gap * (1 + rng.normal(0, 0.0005, 25)).cumprod()
    close = pd.Series(np.concatenate([mon_close, tue_close]), index=idx)

    for i in range(5):
        as_of = close.loc[:tue[i]]
        assert realized_volatility_regime_ok(as_of, lookback=20, max_annualized_vol=0.30) is True, (
            f"overnight gap spuriously blocked trading at {tue[i]}"
        )


def test_gate_fails_open_before_on_timestamp_is_ever_called():
    volatile = make_benchmark(n=60, daily_return_std=0.02)
    gate = RealizedVolatilityGate(volatile, lookback=20, max_annualized_vol=0.30)
    assert gate.check() is True


def test_gate_reflects_regime_as_of_the_last_on_timestamp_call():
    calm = make_benchmark(n=60, daily_return_std=0.0002)
    volatile = make_benchmark(n=60, daily_return_std=0.02, seed=2)

    gate = RealizedVolatilityGate(calm, lookback=20, max_annualized_vol=0.30)
    gate.on_timestamp(calm.index[-1])
    assert gate.check() is True

    gate = RealizedVolatilityGate(volatile, lookback=20, max_annualized_vol=0.30)
    gate.on_timestamp(volatile.index[-1])
    assert gate.check() is False


def test_gate_has_no_look_ahead_into_future_volatility():
    # Calm for the first 40 bars, then a volatile tail -- on_timestamp to a
    # point inside the calm stretch must not be affected by the volatile
    # bars that come later in the series.
    idx = pd.date_range("2024-01-01 09:15", periods=80, freq="15min")
    rng = np.random.default_rng(3)
    calm_returns = rng.normal(0, 0.0002, 40)
    volatile_returns = rng.normal(0, 0.05, 40)
    returns = np.concatenate([calm_returns, volatile_returns])
    close = pd.Series(20000 * (1 + returns).cumprod(), index=idx)

    gate = RealizedVolatilityGate(close, lookback=20, max_annualized_vol=0.30)
    gate.on_timestamp(idx[39])
    assert gate.check() is True

    gate.on_timestamp(idx[-1])
    assert gate.check() is False
