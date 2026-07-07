import importlib

import src.config as config
from src.config import _bool


def test_bool_helper_recognizes_truthy_strings(monkeypatch):
    for value in ("1", "true", "True", "TRUE", "yes", "on", " true "):
        monkeypatch.setenv("SOME_FLAG_UNDER_TEST", value)
        assert _bool("SOME_FLAG_UNDER_TEST", default=False) is True


def test_bool_helper_recognizes_falsy_strings(monkeypatch):
    for value in ("0", "false", "no", "off", "garbage"):
        monkeypatch.setenv("SOME_FLAG_UNDER_TEST", value)
        assert _bool("SOME_FLAG_UNDER_TEST", default=True) is False


def test_bool_helper_falls_back_to_default_when_unset():
    assert _bool("DEFINITELY_UNSET_VAR_XYZ", default=True) is True
    assert _bool("DEFINITELY_UNSET_VAR_XYZ", default=False) is False


def test_settings_defaults_match_env_example():
    # A real .env in dev may override these, so assert invariants that must
    # hold regardless of environment-specific tuning rather than exact values.
    risk = config.settings.risk
    assert risk.max_slots > 0
    assert 0 < risk.capital_per_slot_pct <= 1
    assert 0 < risk.confidence_threshold < 1
    assert risk.daily_loss_limit_pct > 0
    assert risk.atr_stop_mult > 0
    assert config.settings.db_path.endswith(".db")
    assert config.settings.model_path.endswith(".joblib")


def test_env_override_requires_module_reload_to_take_effect(monkeypatch):
    """Documents a real, non-obvious gotcha: KotakNeoConfig/RiskConfig/etc.
    fields default to `os.getenv(...)` expressions evaluated once at class
    *definition* time (module import), not per-instance. Setting an env var
    and constructing a fresh dataclass instance WITHOUT reimporting the
    module silently keeps the old value -- only reloading src.config picks
    up the change. Anyone debugging "I set the env var but nothing changed"
    should look here first.
    """
    original_max_slots = config.settings.risk.max_slots
    try:
        with monkeypatch.context() as m:
            m.setenv("MAX_SLOTS", "25")

            # Constructing a fresh instance directly does NOT see the new
            # env var -- the default expression already ran at import time.
            assert config.RiskConfig().max_slots == original_max_slots

            # Only a full module reload re-evaluates the default expressions.
            importlib.reload(config)
            assert config.settings.risk.max_slots == 25
    finally:
        # Restore real settings for every other test module that imports
        # `from src.config import settings` for the first time after this.
        importlib.reload(config)

    assert config.settings.risk.max_slots == original_max_slots
