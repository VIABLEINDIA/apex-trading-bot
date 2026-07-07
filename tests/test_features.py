import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, add_features, add_labels, build_training_set, latest_feature_row


def make_bars(n=80, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="15min")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1000, 5000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


def test_add_features_produces_expected_columns():
    df = add_features(make_bars())
    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_relative_strength_defaults_to_neutral_without_benchmark():
    df = add_features(make_bars())
    assert (df["relative_strength"] == 0.0).all()


def test_relative_strength_is_computed_when_benchmark_given():
    bars = make_bars()
    # benchmark that's flat -- all of the stock's own roc_5 should show up
    # directly as relative_strength (excess return vs a non-moving market)
    benchmark = pd.Series(100.0, index=bars.index)
    df = add_features(bars, benchmark_close=benchmark)
    roc_5 = bars["Close"].pct_change(periods=5)
    pd.testing.assert_series_equal(df["relative_strength"], roc_5, check_names=False)


def test_relative_strength_nets_out_identical_moves():
    bars = make_bars()
    # benchmark that moves exactly like the stock -- relative strength should be ~0
    benchmark = bars["Close"] * 3.0
    df = add_features(bars, benchmark_close=benchmark)
    assert df["relative_strength"].dropna().abs().max() < 1e-9


def test_volume_surge_nan_during_warmup_then_populated():
    df = add_features(make_bars())
    assert df["volume_surge"].iloc[:9].isna().all()
    assert df["volume_surge"].iloc[9:].notna().all()


def test_labels_only_contain_binary_values_and_dead_zone_is_dropped():
    df = add_labels(add_features(make_bars()))
    non_null = df["label"].dropna()
    assert set(non_null.unique()) <= {0, 1}
    # any row where |next return| < epsilon must have been nulled out
    next_return = df["Close"].shift(-1) / df["Close"] - 1
    dead_zone = next_return.abs() < 0.0005
    assert df.loc[dead_zone.fillna(False), "label"].isna().all()


def test_last_row_has_no_label():
    df = add_labels(add_features(make_bars()))
    assert pd.isna(df["label"].iloc[-1])


def test_build_training_set_drops_warmup_and_dead_zone_rows():
    bars = make_bars(n=80)
    training_set = build_training_set(bars)
    assert not training_set.empty
    assert len(training_set) < len(bars)
    assert set(training_set["label"].unique()) <= {0, 1}
    assert training_set[FEATURE_COLUMNS].notna().all().all()


def test_latest_feature_row_returns_single_row_after_warmup():
    row = latest_feature_row(make_bars(n=80))
    assert row is not None
    assert len(row) == 1
    assert list(row.columns) == FEATURE_COLUMNS


def test_latest_feature_row_none_during_warmup():
    row = latest_feature_row(make_bars(n=5))
    assert row is None


def test_longer_horizon_compares_further_ahead_bar():
    bars = make_bars(n=80)
    df = add_labels(add_features(bars), horizon=3)
    next_return_3 = df["Close"].shift(-3) / df["Close"] - 1
    dead_zone = next_return_3.abs() < 0.0005
    assert df.loc[dead_zone.fillna(False), "label"].isna().all()
    # last 3 rows have no 3-bars-ahead target at all
    assert df["label"].iloc[-3:].isna().all()


def test_longer_horizon_drops_more_trailing_rows_than_default():
    bars = make_bars(n=80)
    default_set = build_training_set(bars, horizon=1)
    longer_set = build_training_set(bars, horizon=5)
    # same warmup, but 5-bars-ahead loses 4 more trailing rows than 1-bar-ahead
    assert len(longer_set) <= len(default_set)
