"""Feature engineering: converts raw 15-minute OHLCV bars into the alpha
features consumed by the AI Classification Engine.

Expects a DataFrame indexed by time with columns: Open, High, Low, Close, Volume.

The original 3-feature set (EMA spread, HL spread, volume surge) turned out to
carry almost no predictive signal for 15-min-ahead direction: a model trained
on the real Nifty 500 universe never produced confidence above ~0.52 (see
README "Known limitations"). This module adds momentum/oscillator features
(RSI, ROC, MACD histogram, Bollinger %B, ATR) that are more standard for this
kind of short-horizon directional classification, and switches the label from
raw sign-of-next-bar (mostly noise) to a dead-zone threshold so near-flat bars
(which carry no tradeable edge net of costs) don't pollute training with
effectively random labels.
"""
import pandas as pd

from src.costs import round_trip_cost_fraction

FEATURE_COLUMNS = [
    "ema_spread",
    "hl_spread",
    "volume_surge",
    "rsi_14",
    "roc_3",
    "roc_5",
    "macd_hist",
    "bb_pct_b",
    "atr_pct",
    "relative_strength",
]

# Next-bar moves smaller than this (as a fraction of price) are treated as
# "no signal" and dropped from training rather than forced into up/down --
# a bar that barely moves is not something the strategy could profitably
# trade anyway once slippage/brokerage are considered.
LABEL_EPSILON = 0.0005


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)  # avg_loss == 0 means pure upward moves -> RSI 100


def _macd_hist(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return (macd_line - signal) / close


def _bollinger_pct_b(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    sma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = (upper - lower).replace(0, pd.NA)
    return (close - lower) / band_width


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr / close


def add_features(bars: pd.DataFrame, benchmark_close: pd.Series | None = None) -> pd.DataFrame:
    """Return a copy of `bars` with alpha feature columns appended.

    `benchmark_close` is the Nifty 50 index's Close series (any reasonably
    aligned timestamps -- it's forward-filled onto `bars.index`), used to
    compute `relative_strength`: is this stock's 5-bar return outperforming
    the broader market's, or is it just moving with the index? None of the
    other features capture this cross-sectional signal. If omitted (e.g. a
    benchmark feed isn't available), `relative_strength` is filled with 0.0
    (neutral -- "no information") rather than NaN, so callers that forget to
    supply it degrade gracefully instead of losing every row to dropna.
    """
    df = bars.copy()

    ema_fast = df["Close"].ewm(span=8, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema_spread"] = (ema_fast - ema_slow) / ema_slow

    df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]

    volume_avg_10 = df["Volume"].rolling(window=10, min_periods=10).mean()
    df["volume_surge"] = df["Volume"] / volume_avg_10

    df["rsi_14"] = _rsi(df["Close"], period=14)
    df["roc_3"] = df["Close"].pct_change(periods=3)
    df["roc_5"] = df["Close"].pct_change(periods=5)
    df["macd_hist"] = _macd_hist(df["Close"])
    df["bb_pct_b"] = _bollinger_pct_b(df["Close"])
    df["atr_pct"] = _atr_pct(df["High"], df["Low"], df["Close"])

    if benchmark_close is not None and not benchmark_close.empty:
        aligned_benchmark = benchmark_close.reindex(df.index, method="ffill")
        benchmark_roc_5 = aligned_benchmark.pct_change(periods=5)
        df["relative_strength"] = df["roc_5"] - benchmark_roc_5
    else:
        df["relative_strength"] = 0.0

    return df


def _dead_zone_label(next_return: pd.Series, epsilon) -> pd.Series:
    """Shared labeling core for add_labels/add_labels_cost_aware: 1 if the
    forward return clears +epsilon, 0 if it clears -epsilon, NaN (dropped by
    the caller's dropna) otherwise. `epsilon` may be a scalar (fixed
    threshold) or a per-row Series (e.g. a cost-derived, time-of-day-dependent
    threshold) -- pandas broadcasts the comparison either way."""
    label = pd.Series(pd.NA, index=next_return.index, dtype="Int64")
    label[next_return > epsilon] = 1
    label[next_return < -epsilon] = 0
    return label


def add_labels(bars_with_features: pd.DataFrame, epsilon: float = LABEL_EPSILON, horizon: int = 1) -> pd.DataFrame:
    """Add the binary training label: 1 if the bar `horizon` steps ahead closes
    at least `epsilon` higher than now, 0 if at least `epsilon` lower. Bars
    whose forward move is smaller than `epsilon` (dead zone) or unknown (the
    last `horizon` rows) get a NaN label so they're dropped rather than forced
    into a near-random class.

    `horizon=1` (the default, next 15-min bar) is what's used live -- the
    strategy evaluates a fresh signal every bar and doesn't hold for a fixed
    number of bars. Longer horizons are a training-time experiment only (see
    scripts/tune_model.py): a noisier 1-bar target may just not have enough
    signal-to-noise for the model to learn from, and a longer forward window
    changes nothing about live inference (features are identical either way,
    only the training label does).
    """
    df = bars_with_features.copy()
    next_return = df["Close"].shift(-horizon) / df["Close"] - 1
    df["label"] = _dead_zone_label(next_return, epsilon)
    return df


def build_training_set(bars: pd.DataFrame, epsilon: float = LABEL_EPSILON,
                        benchmark_close: pd.Series | None = None, horizon: int = 1) -> pd.DataFrame:
    """Full pipeline: features + labels, with warm-up/dead-zone/NaN rows dropped."""
    df = add_labels(add_features(bars, benchmark_close=benchmark_close), epsilon=epsilon, horizon=horizon)
    return df.dropna(subset=FEATURE_COLUMNS + ["label"])


def add_labels_cost_aware(bars_with_features: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Same dead-zone label logic as add_labels, but the dead-zone threshold
    is derived per-bar from src/costs.py's actual round-trip cost model
    (time-of-day slippage + statutory charges) instead of the fixed,
    guessed LABEL_EPSILON constant. A bar's forward move must clear what a
    real round trip would actually cost -- wider near the open/close, where
    slippage is worse -- before counting as a genuine "up"/"down" label,
    rather than an arbitrary fixed threshold that's the same size regardless
    of when the bar happens to fall.

    Purely additive: add_labels/build_training_set/LABEL_EPSILON are
    unchanged, so no existing training path's numbers shift under it.
    """
    df = bars_with_features.copy()
    next_return = df["Close"].shift(-horizon) / df["Close"] - 1

    # round_trip_cost_fraction only varies by time-of-day bucket (a handful of
    # distinct windows across a trading day -- see src/costs.py's
    # TIME_OF_DAY_SLIPPAGE_BPS), so memoize per distinct time() instead of
    # recomputing the same handful of values once per row across the whole
    # (potentially multi-ticker, multi-day) frame.
    cost_by_time: dict = {}

    def _epsilon_for(ts) -> float:
        t = ts.time()
        if t not in cost_by_time:
            cost_by_time[t] = round_trip_cost_fraction(ts)
        return cost_by_time[t]

    epsilon = pd.Series([_epsilon_for(ts) for ts in df.index], index=df.index)
    df["label"] = _dead_zone_label(next_return, epsilon)
    return df


def build_training_set_cost_aware(bars: pd.DataFrame, benchmark_close: pd.Series | None = None,
                                   horizon: int = 1) -> pd.DataFrame:
    """Full pipeline using the cost-derived dead zone (add_labels_cost_aware)
    instead of build_training_set's fixed LABEL_EPSILON."""
    df = add_labels_cost_aware(add_features(bars, benchmark_close=benchmark_close), horizon=horizon)
    return df.dropna(subset=FEATURE_COLUMNS + ["label"])


def latest_feature_row(bars: pd.DataFrame, benchmark_close: pd.Series | None = None) -> pd.DataFrame | None:
    """Return the most recent fully-formed feature row for live inference, or
    None if there isn't enough history yet (warm-up period)."""
    df = add_features(bars, benchmark_close=benchmark_close).dropna(subset=FEATURE_COLUMNS)
    if df.empty:
        return None
    return df.iloc[[-1]][FEATURE_COLUMNS]
