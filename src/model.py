"""Phase 2: The AI Classification Engine.

A GradientBoostingClassifier predicts the probability that a stock's price
moves upward over the next 15-minute bar. Classification (not regression) is
used deliberately to filter out intraday noise rather than chase exact price
targets.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

from src.config import settings
from src.features import FEATURE_COLUMNS, build_training_set

logger = logging.getLogger(__name__)


class MomentumClassifier:
    def __init__(self, model: GradientBoostingClassifier | None = None):
        # `model or GradientBoostingClassifier(...)` looks equivalent but isn't:
        # Python falls back to __len__ for truthiness when __bool__ isn't
        # defined, and an *unfitted* GBM has no `estimators_` yet, so
        # `len(model)` raises AttributeError instead of just being falsy.
        # Only ever surfaced once something passed in an unfitted custom
        # model (scripts/tune_model.py) -- previously this was always either
        # None or an already-fitted model from load(), which happens to have
        # `estimators_` set and a nonzero (truthy) length.
        self.model = model if model is not None else GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    def train(self, bars: pd.DataFrame, benchmark_close: pd.Series | None = None, test_size: float = 0.2) -> dict:
        """Train on a single contiguous OHLCV series (one ticker), time-ordered
        holdout (train on the earlier period, test on the later one)."""
        return self.train_multi([bars], benchmark_close=benchmark_close, test_size=test_size)

    def train_prelabeled(self, training_set: pd.DataFrame, test_size: float = 0.2) -> dict:
        """Train on a single already featurized+labeled frame with a naive
        time-ordered split. Prefer `train_multi` when you have several tickers,
        since splitting *after* concatenation would put entire tickers into the
        test set instead of a genuine held-out time window."""
        return self._fit_and_evaluate(training_set, training_set, test_size=test_size)

    def train_multi(self, ticker_bars: list[pd.DataFrame], benchmark_close: pd.Series | None = None,
                     test_size: float = 0.2) -> dict:
        """Train across many tickers with a proper walk-forward-style holdout:
        each ticker's own labeled series is split into an earlier train segment
        and a later test segment *before* concatenating across tickers, so the
        holdout genuinely measures generalization to unseen future bars rather
        than to alphabetically-later tickers."""
        train_frames, test_frames = [], []
        for bars in ticker_bars:
            labeled = build_training_set(bars, benchmark_close=benchmark_close) if "label" not in bars.columns else bars
            if labeled.empty:
                continue
            split_at = int(len(labeled) * (1 - test_size))
            if split_at == 0 or split_at == len(labeled):
                train_frames.append(labeled)
                continue
            train_frames.append(labeled.iloc[:split_at])
            test_frames.append(labeled.iloc[split_at:])

        train_set = pd.concat(train_frames, ignore_index=True)
        test_set = pd.concat(test_frames, ignore_index=True) if test_frames else train_set
        return self._fit_and_evaluate(train_set, test_set, test_size=None)

    def _fit_and_evaluate(self, train_set: pd.DataFrame, test_set: pd.DataFrame, test_size: float | None) -> dict:
        X_train, y_train = train_set[FEATURE_COLUMNS], train_set["label"].astype(int)
        X_test, y_test = test_set[FEATURE_COLUMNS], test_set["label"].astype(int)

        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        report = classification_report(y_test, self.model.predict(X_test), output_dict=True)
        logger.info("Model trained on %d rows, holdout accuracy=%.3f", len(X_train), report["accuracy"])
        return report

    def predict_confidence(self, feature_row: pd.DataFrame) -> float:
        """Return P(price moves up) for a single feature row."""
        proba = self.model.predict_proba(feature_row[FEATURE_COLUMNS])
        # predict_proba columns follow self.model.classes_; locate the "1" (up) class.
        up_index = list(self.model.classes_).index(1)
        return float(proba[0][up_index])

    def is_buy_signal(self, feature_row: pd.DataFrame) -> tuple[bool, float]:
        confidence = self.predict_confidence(feature_row)
        return confidence > settings.risk.confidence_threshold, confidence

    def save(self, path: str | None = None) -> None:
        path = path or settings.model_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | None = None) -> "MomentumClassifier":
        path = path or settings.model_path
        model = joblib.load(path)
        return cls(model=model)
