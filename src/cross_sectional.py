"""Cross-sectional label construction for the AI Classification Engine.

The model has always predicted each stock's own future direction
independently (src/features.py: "will THIS stock's next bar close higher
than an epsilon threshold, in isolation"). But scripts/paper_trade.py's
simulate() already *executes* cross-sectionally: every candidate at a given
timestamp is scored, then slots are filled strongest-confidence-first across
the whole watchlist, not ticker-by-ticker. The training objective was never
changed to match that -- this module does, by asking "will THIS stock
outperform its peers over the next bar" instead of "will THIS stock's price
go up in isolation".

See README "Model-signal sweep" for why this was worth trying (the ~52%
holdout-accuracy ceiling held across every horizon/hyperparameter variant
already tested) and docs/decisions/ for the risk-management context this
sits alongside.
"""
import pandas as pd

from src.features import FEATURE_COLUMNS, add_features

# Need at least this many tickers with data at a given timestamp for a
# percentile rank within that cross-section to mean anything.
MIN_CROSS_SECTION_SIZE = 5


def build_cross_sectional_training_set(
    ticker_bars: dict[str, pd.DataFrame],
    benchmark_close: pd.Series | None = None,
    top_quantile: float = 0.3,
    bottom_quantile: float = 0.3,
    horizon: int = 1,
    min_cross_section_size: int = MIN_CROSS_SECTION_SIZE,
) -> pd.DataFrame:
    """Featurize each ticker independently (identical features to the
    absolute model), then label each bar by its forward-return RANK within
    the cross-section of tickers that have data at that same timestamp,
    instead of an absolute return threshold:

      - label=1 if the ticker's forward return is in the top `top_quantile`
        of the cross-section at that timestamp
      - label=0 if in the bottom `bottom_quantile`
      - dropped (dead zone) otherwise -- mirroring add_labels' dead-zone
        concept (src/features.py), but relative instead of absolute

    Timestamps where fewer than `min_cross_section_size` tickers have data
    are dropped entirely -- a percentile rank across 2 stocks isn't a
    meaningful cross-section.

    Returns a single frame with FEATURE_COLUMNS + "label" + "ticker" +
    "timestamp", sorted by timestamp -- feed it directly to
    MomentumClassifier.train_prelabeled() for a genuine time-ordered holdout.
    """
    featured: dict[str, pd.DataFrame] = {}
    next_returns: dict[str, pd.Series] = {}
    for ticker, bars in ticker_bars.items():
        df = add_features(bars, benchmark_close=benchmark_close)
        next_returns[ticker] = df["Close"].shift(-horizon) / df["Close"] - 1
        featured[ticker] = df

    # Wide frame: index=timestamp, columns=ticker, values=forward return.
    # Tickers without a bar at a given timestamp are NaN there -- rank()
    # below ignores them, so the cross-section is only the tickers that
    # actually have data at that instant.
    returns_wide = pd.DataFrame(next_returns)
    cross_section_size = returns_wide.notna().sum(axis=1)
    valid_timestamps = cross_section_size[cross_section_size >= min_cross_section_size].index
    returns_wide = returns_wide.loc[valid_timestamps]

    percentile_ranks = returns_wide.rank(axis=1, pct=True, na_option="keep")

    rows = []
    for ticker, df in featured.items():
        if ticker not in percentile_ranks.columns:
            continue
        rank = percentile_ranks[ticker].reindex(df.index)

        label = pd.Series(pd.NA, index=df.index, dtype="Int64")
        label[rank >= 1 - top_quantile] = 1
        label[rank <= bottom_quantile] = 0

        sub = df.copy()
        sub["label"] = label
        sub["ticker"] = ticker
        sub["timestamp"] = sub.index
        rows.append(sub.dropna(subset=FEATURE_COLUMNS + ["label"]))

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["label", "ticker", "timestamp"])

    combined = pd.concat(rows, ignore_index=True)
    return combined.sort_values("timestamp", kind="stable").reset_index(drop=True)
