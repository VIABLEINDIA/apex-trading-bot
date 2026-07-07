import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.features import FEATURE_COLUMNS, build_training_set
from src.model import MomentumClassifier


def make_bars(n=200, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="15min")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1000, 5000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


def test_default_constructor_builds_the_documented_gbm_hyperparameters():
    clf = MomentumClassifier()
    assert isinstance(clf.model, GradientBoostingClassifier)
    assert clf.model.n_estimators == 300
    assert clf.model.max_depth == 4
    assert clf.model.learning_rate == 0.05


def test_constructor_accepts_an_unfitted_custom_model_without_the_len_bug():
    """Regression test for the bug documented in README: `model or
    GradientBoostingClassifier(...)` looked like a safe None-check but wasn't
    -- Python falls back to __len__ for truthiness when __bool__ isn't
    defined, and an unfitted GBM has no `estimators_` yet, so `len(model)`
    raised AttributeError instead of evaluating as falsy. Fixed with an
    explicit `is not None` check; this test pins that fix."""
    custom = GradientBoostingClassifier(n_estimators=10, max_depth=2)
    clf = MomentumClassifier(model=custom)
    assert clf.model is custom
    assert clf.model.n_estimators == 10


def test_constructor_accepts_an_already_fitted_model():
    fitted = GradientBoostingClassifier(n_estimators=5).fit([[0, 0], [1, 1]], [0, 1])
    clf = MomentumClassifier(model=fitted)
    assert clf.model is fitted


def test_train_multi_returns_holdout_classification_report():
    clf = MomentumClassifier()
    ticker_bars = [make_bars(seed=1), make_bars(seed=2), make_bars(seed=3)]

    report = clf.train_multi(ticker_bars)

    assert "accuracy" in report
    assert 0.0 <= report["accuracy"] <= 1.0


def test_train_delegates_to_train_multi_with_a_single_ticker():
    clf = MomentumClassifier()
    report = clf.train(make_bars())
    assert "accuracy" in report


def test_train_prelabeled_uses_precomputed_training_set():
    clf = MomentumClassifier()
    training_set = build_training_set(make_bars())

    report = clf.train_prelabeled(training_set)

    assert "accuracy" in report


def test_predict_confidence_is_a_probability():
    clf = MomentumClassifier()
    clf.train(make_bars())
    training_set = build_training_set(make_bars(seed=99))

    confidence = clf.predict_confidence(training_set.iloc[[0]])

    assert 0.0 <= confidence <= 1.0


def test_is_buy_signal_matches_confidence_threshold(monkeypatch):
    from src.config import settings
    clf = MomentumClassifier()
    clf.train(make_bars())
    training_set = build_training_set(make_bars(seed=99))
    row = training_set.iloc[[0]]

    is_buy, confidence = clf.is_buy_signal(row)

    assert is_buy == (confidence > settings.risk.confidence_threshold)


def test_save_and_load_round_trip_preserves_predictions(tmp_path):
    clf = MomentumClassifier()
    clf.train(make_bars())
    training_set = build_training_set(make_bars(seed=99))
    row = training_set.iloc[[0]]
    original_confidence = clf.predict_confidence(row)

    path = tmp_path / "nested" / "model.joblib"
    clf.save(str(path))
    assert path.exists()

    reloaded = MomentumClassifier.load(str(path))
    assert reloaded.predict_confidence(row) == original_confidence
