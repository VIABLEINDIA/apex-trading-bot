import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from src.features import FEATURE_COLUMNS, build_training_set
from src.model import EnsembleMomentumClassifier, MomentumClassifier


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


def test_train_prelabeled_holds_out_the_last_test_size_fraction_by_row_order():
    """Regression test: train_prelabeled used to pass the same `training_set`
    object as both train_set and test_set to _fit_and_evaluate, silently
    ignoring test_size and evaluating in-sample -- despite its own docstring
    claiming a time-ordered split. Fixed to actually split by row order."""
    clf = MomentumClassifier()
    training_set = build_training_set(make_bars(n=300, seed=7))
    seen_train_len = {}

    original_fit = clf.model.fit

    def spy_fit(X, y, **kwargs):
        seen_train_len["n"] = len(X)
        return original_fit(X, y, **kwargs)

    clf.model.fit = spy_fit

    clf.train_prelabeled(training_set, test_size=0.25)

    assert seen_train_len["n"] < len(training_set)
    assert seen_train_len["n"] == int(len(training_set) * 0.75)


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


class _StubMember:
    """A minimal stand-in for MomentumClassifier that returns a fixed
    confidence, so ensemble averaging math can be tested independently of
    real model training."""

    def __init__(self, confidence: float):
        self.confidence = confidence

    def predict_confidence(self, feature_row: pd.DataFrame) -> float:
        return self.confidence


def test_ensemble_rejects_empty_member_list():
    with pytest.raises(ValueError):
        EnsembleMomentumClassifier(members=[])


def test_ensemble_with_one_member_matches_that_members_confidence():
    member = _StubMember(0.73)
    ensemble = EnsembleMomentumClassifier(members=[member])
    assert ensemble.predict_confidence(pd.DataFrame()) == pytest.approx(0.73)


def test_ensemble_averages_confidence_across_members():
    ensemble = EnsembleMomentumClassifier(members=[_StubMember(0.4), _StubMember(0.6), _StubMember(0.8)])
    assert ensemble.predict_confidence(pd.DataFrame()) == pytest.approx(0.6)


def test_ensemble_is_buy_signal_uses_averaged_confidence(monkeypatch):
    from types import SimpleNamespace
    # settings.risk is a frozen dataclass -- swap the module-level name
    # model.py reads instead of mutating the real (frozen) instance.
    monkeypatch.setattr("src.model.settings", SimpleNamespace(risk=SimpleNamespace(confidence_threshold=0.5)))
    ensemble = EnsembleMomentumClassifier(members=[_StubMember(0.3), _StubMember(0.9)])  # average 0.6

    is_buy, confidence = ensemble.is_buy_signal(pd.DataFrame())

    assert confidence == pytest.approx(0.6)
    assert is_buy is True


def test_ensemble_of_real_trained_classifiers_averages_their_predictions():
    members = [MomentumClassifier() for _ in range(3)]
    for i, clf in enumerate(members):
        clf.train(make_bars(seed=i))
    row = build_training_set(make_bars(seed=99)).iloc[[0]]

    ensemble = EnsembleMomentumClassifier(members=members)
    expected = sum(m.predict_confidence(row) for m in members) / len(members)

    assert ensemble.predict_confidence(row) == pytest.approx(expected)


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
