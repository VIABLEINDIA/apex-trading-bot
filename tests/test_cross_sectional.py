import numpy as np
import pandas as pd

from src.cross_sectional import build_cross_sectional_training_set
from src.features import FEATURE_COLUMNS


def make_bars(n=80, seed=1, drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="15min")
    close = 100 + np.cumsum(rng.normal(drift, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1000, 5000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


def make_universe(n_tickers=8, n=80):
    """A universe where ticker "WINNER" drifts up every bar and "LOSER"
    drifts down every bar, relative to a flat/noisy middle pack -- so a
    correct cross-sectional ranking should consistently rank WINNER at the
    top and LOSER at the bottom regardless of the absolute market regime."""
    bars = {"WINNER": make_bars(n=n, seed=100, drift=0.5), "LOSER": make_bars(n=n, seed=101, drift=-0.5)}
    for i in range(n_tickers - 2):
        bars[f"MID_{i}"] = make_bars(n=n, seed=i, drift=0.0)
    return bars


def test_returns_expected_columns():
    result = build_cross_sectional_training_set(make_universe(), min_cross_section_size=3)
    for col in FEATURE_COLUMNS + ["label", "ticker", "timestamp"]:
        assert col in result.columns


def test_result_is_sorted_by_timestamp():
    result = build_cross_sectional_training_set(make_universe(), min_cross_section_size=3)
    assert result["timestamp"].is_monotonic_increasing


def test_consistent_outperformer_is_labeled_top_more_often_than_bottom():
    result = build_cross_sectional_training_set(make_universe(), top_quantile=0.3, bottom_quantile=0.3, min_cross_section_size=3)
    winner_labels = result[result["ticker"] == "WINNER"]["label"]
    assert (winner_labels == 1).sum() > (winner_labels == 0).sum()


def test_consistent_underperformer_is_labeled_bottom_more_often_than_top():
    result = build_cross_sectional_training_set(make_universe(), top_quantile=0.3, bottom_quantile=0.3, min_cross_section_size=3)
    loser_labels = result[result["ticker"] == "LOSER"]["label"]
    assert (loser_labels == 0).sum() > (loser_labels == 1).sum()


def test_dead_zone_drops_middle_of_distribution():
    # Narrow top/bottom quantiles leave a wide dead zone -- most of the
    # "MID" tickers' bars (no consistent drift) should fall in it and get
    # dropped rather than forced into a near-random label.
    universe = make_universe(n_tickers=10)
    result = build_cross_sectional_training_set(universe, top_quantile=0.1, bottom_quantile=0.1, min_cross_section_size=3)
    mid_rows = result[result["ticker"].str.startswith("MID")]
    total_mid_bars = sum(len(make_universe(n_tickers=10)[f"MID_{i}"]) for i in range(8))
    assert len(mid_rows) < total_mid_bars


def test_timestamps_below_min_cross_section_size_are_excluded():
    # Only 2 tickers total -- below any reasonable min_cross_section_size.
    universe = {"A": make_bars(seed=1), "B": make_bars(seed=2)}
    result = build_cross_sectional_training_set(universe, min_cross_section_size=5)
    assert result.empty


def test_min_cross_section_size_is_configurable_for_small_universes():
    universe = {"A": make_bars(seed=1), "B": make_bars(seed=2), "C": make_bars(seed=3)}
    result = build_cross_sectional_training_set(universe, min_cross_section_size=3)
    assert not result.empty


def test_empty_universe_returns_empty_frame_with_expected_columns():
    result = build_cross_sectional_training_set({})
    assert result.empty
    for col in FEATURE_COLUMNS + ["label", "ticker", "timestamp"]:
        assert col in result.columns


def test_ticker_missing_from_all_valid_timestamps_is_excluded_without_error():
    # A ticker whose bars don't overlap in time with anyone else's should
    # simply contribute no rows, not raise.
    universe = make_universe(n_tickers=5)
    offset_bars = make_bars(seed=999)
    offset_bars.index = offset_bars.index + pd.Timedelta(days=365)
    universe["ISOLATED"] = offset_bars

    result = build_cross_sectional_training_set(universe, min_cross_section_size=3)

    assert "ISOLATED" not in set(result["ticker"])
